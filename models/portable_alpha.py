"""
Optimiseur d'alpha portable pour les caisses de retraite.

L'alpha portable est une strategie qui separe la generation de beta
(exposition au marche) de la generation d'alpha (rendement excedentaire).

Architecture :
1. Portefeuille beta : replique un benchmark (ex: 60/40) de facon passive
2. Overlay alpha : positions long/short visant a generer de l'alpha
3. Portefeuille combine = beta + alpha overlay

L'optimiseur maximise le ratio d'information (alpha / tracking error)
sous les contraintes reglementaires de levier.
"""

import numpy as np
import cvxpy as cp
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from models.base import BaseOptimizer, OptimizationResult
from constraints.manager import ConstraintManager, LeveragedConstraintSet
from constraints.regulatory import PortableAlphaRegulations


@dataclass
class PortableAlphaResult:
    """Resultat complet d'une optimisation d'alpha portable."""
    # Portefeuille combine
    combined_weights: np.ndarray
    asset_names: List[str]
    # Decomposition
    beta_weights: np.ndarray
    alpha_overlay: np.ndarray  # = combined - beta
    # Performance du combine
    expected_return: float
    volatility: float
    sharpe_ratio: float
    # Metriques alpha
    alpha: float                  # Rendement excedentaire vs benchmark
    tracking_error: float         # Volatilite de l'alpha
    information_ratio: float      # alpha / tracking_error
    # Metriques de levier
    gross_leverage: float
    net_exposure: float
    long_exposure: float
    short_exposure: float
    n_long_positions: int
    n_short_positions: int
    # Couts
    financing_cost: float
    net_alpha: float              # Alpha apres couts de financement
    # Contributions au risque
    risk_contributions: np.ndarray
    # Metadata
    status: str = "optimal"
    solver_time: float = 0.0
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convertit en dictionnaire pour l'affichage."""
        return {
            "combined_weights": dict(zip(self.asset_names, self.combined_weights.tolist())),
            "beta_weights": dict(zip(self.asset_names, self.beta_weights.tolist())),
            "alpha_overlay": dict(zip(self.asset_names, self.alpha_overlay.tolist())),
            "expected_return": self.expected_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "alpha": self.alpha,
            "tracking_error": self.tracking_error,
            "information_ratio": self.information_ratio,
            "gross_leverage": self.gross_leverage,
            "net_exposure": self.net_exposure,
            "financing_cost": self.financing_cost,
            "net_alpha": self.net_alpha,
            "status": self.status,
        }


class PortableAlphaOptimizer:
    """Optimiseur d'alpha portable multi-classes d'actifs.

    Strategies disponibles :
    1. max_info_ratio : maximise le ratio d'information (alpha / TE)
    2. max_alpha      : maximise l'alpha brut sous contrainte de TE
    3. min_te         : minimise le tracking error pour un alpha cible
    4. risk_budgeted  : budgete le risque entre beta et alpha
    """

    def __init__(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        benchmark_weights: np.ndarray,
        risk_free_rate: float = 0.025,
        asset_names: Optional[List[str]] = None,
    ):
        self.mu = expected_returns
        self.sigma = cov_matrix
        self.benchmark = benchmark_weights
        self.rf = risk_free_rate
        self.n_assets = len(expected_returns)
        self.asset_names = asset_names or [f"Actif_{i}" for i in range(self.n_assets)]

        # Validation des entrees
        self._validate_inputs()

        # Calcul du rendement et risque du benchmark
        self.benchmark_return = self.benchmark @ self.mu
        self.benchmark_vol = np.sqrt(self.benchmark @ self.sigma @ self.benchmark)

    def _validate_inputs(self):
        """Verifie les dimensions et la PSD."""
        assert len(self.mu) == self.n_assets
        assert self.sigma.shape == (self.n_assets, self.n_assets)
        assert len(self.benchmark) == self.n_assets
        # PSD
        if not np.allclose(self.sigma, self.sigma.T, atol=1e-8):
            self.sigma = (self.sigma + self.sigma.T) / 2
        eigvals = np.linalg.eigvalsh(self.sigma)
        if np.min(eigvals) < -1e-8:
            min_eig = np.min(eigvals)
            self.sigma += (-min_eig + 1e-6) * np.eye(self.n_assets)

    def optimize(
        self,
        strategy: str = "max_info_ratio",
        max_gross_leverage: float = 1.5,
        target_net_exposure: float = 1.0,
        max_short_per_asset: float = 0.15,
        short_eligible: Optional[List[int]] = None,
        tracking_error_budget: Optional[float] = None,
        alpha_target: Optional[float] = None,
        financing_spread: float = 0.005,
        max_weights: Optional[np.ndarray] = None,
        min_weights: Optional[np.ndarray] = None,
        esg_min_score: Optional[float] = None,
        esg_scores: Optional[np.ndarray] = None,
        group_constraints: Optional[list] = None,
        regularization: float = 0.001,
    ) -> PortableAlphaResult:
        """Execute l'optimisation d'alpha portable.

        Parameters
        ----------
        strategy : str
            'max_info_ratio', 'max_alpha', 'min_te', 'risk_budgeted'
        max_gross_leverage : float
            Levier brut maximal (ex: 1.5 = 150%)
        target_net_exposure : float
            Exposition nette cible (typiquement 1.0)
        max_short_per_asset : float
            Position courte maximale par actif
        short_eligible : list
            Indices des actifs eligibles au short
        tracking_error_budget : float
            Budget de tracking error (pour max_alpha)
        alpha_target : float
            Alpha cible (pour min_te)
        financing_spread : float
            Cout de financement annualise
        regularization : float
            Terme de regularisation L2 pour la stabilite

        Returns
        -------
        PortableAlphaResult
        """
        start_time = time.time()

        if short_eligible is None:
            short_eligible = PortableAlphaRegulations.SHORT_ELIGIBLE_INDICES

        # Default min/max weights
        if min_weights is None:
            min_w = np.full(self.n_assets, -max_short_per_asset)
            # Non-shortable assets -> min 0
            for i in range(self.n_assets):
                if i not in short_eligible:
                    min_w[i] = 0.0
        else:
            min_w = min_weights.copy()

        if max_weights is None:
            max_w = np.full(self.n_assets, 0.50)
        else:
            max_w = max_weights.copy()

        # Variable d'optimisation
        w = cp.Variable(self.n_assets)

        # Contraintes de base
        constraints = []

        # Exposition nette
        net_tol = 0.05
        constraints.append(cp.sum(w) >= target_net_exposure - net_tol)
        constraints.append(cp.sum(w) <= target_net_exposure + net_tol)

        # Levier brut
        constraints.append(cp.norm(w, 1) <= max_gross_leverage)

        # Bornes
        constraints.append(w >= min_w)
        constraints.append(w <= max_w)

        # Positions courtes limitees aux actifs eligibles
        for i in range(self.n_assets):
            if i not in short_eligible:
                constraints.append(w[i] >= 0)
            else:
                constraints.append(w[i] >= -max_short_per_asset)

        # Contraintes de groupe
        if group_constraints:
            for gc in group_constraints:
                group_sum = cp.sum(w[gc.asset_indices])
                constraints.append(group_sum >= gc.min_allocation)
                constraints.append(group_sum <= gc.max_allocation)

        # ESG
        if esg_min_score is not None and esg_scores is not None:
            constraints.append(esg_scores @ w >= esg_min_score)

        # Sigma PSD (regularise pour CVXPY)
        sigma_reg = self.sigma + regularization * np.eye(self.n_assets)

        # Objectif selon la strategie
        if strategy == "max_info_ratio":
            # Maximiser alpha - lambda * TE^2 (approximation du ratio d'info)
            alpha_expr = self.mu @ w - self.mu @ self.benchmark
            te_squared = cp.quad_form(w - self.benchmark, sigma_reg)
            # Penalite de financement
            financing_penalty = financing_spread * cp.norm(w, 1)
            # Lambda pour le trade-off alpha / TE
            lam = 2.0
            objective = cp.Maximize(
                alpha_expr - lam * te_squared - financing_penalty
            )

        elif strategy == "max_alpha":
            # Maximiser alpha sous contrainte de TE
            alpha_expr = self.mu @ w - self.mu @ self.benchmark
            financing_penalty = financing_spread * cp.norm(w, 1)
            objective = cp.Maximize(alpha_expr - financing_penalty)
            if tracking_error_budget is not None:
                te_squared = cp.quad_form(w - self.benchmark, sigma_reg)
                constraints.append(te_squared <= tracking_error_budget ** 2)

        elif strategy == "min_te":
            # Minimiser TE sous contrainte d'alpha minimum
            te_squared = cp.quad_form(w - self.benchmark, sigma_reg)
            objective = cp.Minimize(te_squared)
            if alpha_target is not None:
                alpha_expr = self.mu @ w - self.mu @ self.benchmark
                constraints.append(alpha_expr >= alpha_target)

        elif strategy == "risk_budgeted":
            # Maximise le rendement ajuste au risque total du portefeuille
            port_return = self.mu @ w
            port_risk = cp.quad_form(w, sigma_reg)
            financing_penalty = financing_spread * cp.norm(w, 1)
            lam = 3.0
            objective = cp.Maximize(
                port_return - lam * port_risk - financing_penalty
            )

        else:
            raise ValueError(f"Strategie inconnue: {strategy}")

        # Resolution
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.SCS, max_iters=10000, eps=1e-8, verbose=False)
        except cp.SolverError:
            try:
                prob.solve(solver=cp.ECOS, max_iters=500, verbose=False)
            except cp.SolverError:
                prob.solve(solver=cp.OSQP, max_iter=25000, verbose=False)

        if prob.status not in ("optimal", "optimal_inaccurate"):
            # Retourner le benchmark en cas d'echec
            return self._build_fallback_result(start_time, prob.status)

        optimal_weights = w.value.copy()

        # Nettoyage des poids tres petits
        optimal_weights[np.abs(optimal_weights) < 1e-6] = 0.0

        return self._build_result(optimal_weights, start_time, financing_spread)

    def _build_result(
        self,
        weights: np.ndarray,
        start_time: float,
        financing_spread: float,
    ) -> PortableAlphaResult:
        """Construit le resultat a partir des poids optimaux."""
        # Performance du portefeuille combine
        port_return = weights @ self.mu
        port_vol = np.sqrt(weights @ self.sigma @ weights)
        sharpe = (port_return - self.rf) / port_vol if port_vol > 1e-10 else 0.0

        # Alpha et tracking error
        alpha = port_return - self.benchmark_return
        diff = weights - self.benchmark
        te = np.sqrt(diff @ self.sigma @ diff) if np.any(np.abs(diff) > 1e-8) else 0.0
        ir = alpha / te if te > 1e-10 else 0.0

        # Metriques de levier
        long_mask = weights > 0
        short_mask = weights < 0
        gross = np.sum(np.abs(weights))
        net = np.sum(weights)
        long_exp = np.sum(weights[long_mask])
        short_exp = np.sum(np.abs(weights[short_mask]))
        n_long = int(np.sum(weights > 1e-4))
        n_short = int(np.sum(weights < -1e-4))

        # Cout de financement
        fin_cost = PortableAlphaRegulations.compute_financing_cost(
            weights, financing_spread, self.rf
        )
        net_alpha = alpha - fin_cost["cout_total_annuel"]

        # Contributions au risque
        if port_vol > 1e-10:
            marginal = self.sigma @ weights
            risk_contrib = weights * marginal / port_vol
        else:
            risk_contrib = np.zeros(self.n_assets)

        # Overlay alpha
        alpha_overlay = weights - self.benchmark

        return PortableAlphaResult(
            combined_weights=weights,
            asset_names=self.asset_names,
            beta_weights=self.benchmark.copy(),
            alpha_overlay=alpha_overlay,
            expected_return=port_return,
            volatility=port_vol,
            sharpe_ratio=sharpe,
            alpha=alpha,
            tracking_error=te,
            information_ratio=ir,
            gross_leverage=gross,
            net_exposure=net,
            long_exposure=long_exp,
            short_exposure=short_exp,
            n_long_positions=n_long,
            n_short_positions=n_short,
            financing_cost=fin_cost["cout_total_annuel"],
            net_alpha=net_alpha,
            risk_contributions=risk_contrib,
            status="optimal",
            solver_time=time.time() - start_time,
            metadata={
                "benchmark_return": self.benchmark_return,
                "benchmark_vol": self.benchmark_vol,
                "financing_detail": fin_cost,
            },
        )

    def _build_fallback_result(
        self, start_time: float, status: str,
    ) -> PortableAlphaResult:
        """Retourne le benchmark en cas d'echec de l'optimisation."""
        port_return = self.benchmark @ self.mu
        port_vol = np.sqrt(self.benchmark @ self.sigma @ self.benchmark)
        sharpe = (port_return - self.rf) / port_vol if port_vol > 1e-10 else 0.0

        return PortableAlphaResult(
            combined_weights=self.benchmark.copy(),
            asset_names=self.asset_names,
            beta_weights=self.benchmark.copy(),
            alpha_overlay=np.zeros(self.n_assets),
            expected_return=port_return,
            volatility=port_vol,
            sharpe_ratio=sharpe,
            alpha=0.0,
            tracking_error=0.0,
            information_ratio=0.0,
            gross_leverage=1.0,
            net_exposure=1.0,
            long_exposure=np.sum(self.benchmark),
            short_exposure=0.0,
            n_long_positions=int(np.sum(self.benchmark > 1e-4)),
            n_short_positions=0,
            financing_cost=0.0,
            net_alpha=0.0,
            risk_contributions=np.zeros(self.n_assets),
            status=f"echec ({status}) - benchmark retourne",
            solver_time=time.time() - start_time,
        )

    def compute_efficient_alpha_frontier(
        self,
        n_points: int = 20,
        max_gross_leverage: float = 1.5,
        short_eligible: Optional[List[int]] = None,
        max_short_per_asset: float = 0.15,
        financing_spread: float = 0.005,
    ) -> List[PortableAlphaResult]:
        """Calcule la frontiere efficiente alpha/tracking error.

        Genere des portefeuilles optimaux a differents niveaux de TE budget.
        """
        te_budgets = np.linspace(0.005, 0.10, n_points)
        results = []

        for te_budget in te_budgets:
            result = self.optimize(
                strategy="max_alpha",
                max_gross_leverage=max_gross_leverage,
                tracking_error_budget=te_budget,
                short_eligible=short_eligible,
                max_short_per_asset=max_short_per_asset,
                financing_spread=financing_spread,
            )
            if result.status == "optimal":
                results.append(result)

        return results

    def decompose_risk(
        self, weights: np.ndarray,
    ) -> Dict:
        """Decompose le risque entre beta et alpha.

        Risque total^2 = Risque beta^2 + Risque alpha^2 + 2*Cov(beta, alpha)
        """
        overlay = weights - self.benchmark

        # Risque beta
        beta_risk = np.sqrt(self.benchmark @ self.sigma @ self.benchmark)

        # Risque overlay (active risk)
        alpha_risk = np.sqrt(overlay @ self.sigma @ overlay) if np.any(np.abs(overlay) > 1e-8) else 0.0

        # Risque total
        total_risk = np.sqrt(weights @ self.sigma @ weights)

        # Covariance beta-alpha
        cov_beta_alpha = self.benchmark @ self.sigma @ overlay

        # Correlation
        if beta_risk > 1e-10 and alpha_risk > 1e-10:
            corr_beta_alpha = cov_beta_alpha / (beta_risk * alpha_risk)
        else:
            corr_beta_alpha = 0.0

        # Contribution relative
        total_var = total_risk ** 2
        if total_var > 1e-10:
            beta_contrib_pct = (beta_risk ** 2 + cov_beta_alpha) / total_var
            alpha_contrib_pct = (alpha_risk ** 2 + cov_beta_alpha) / total_var
        else:
            beta_contrib_pct = 1.0
            alpha_contrib_pct = 0.0

        return {
            "risque_total": total_risk,
            "risque_beta": beta_risk,
            "risque_alpha": alpha_risk,
            "covariance_beta_alpha": cov_beta_alpha,
            "correlation_beta_alpha": corr_beta_alpha,
            "contribution_beta_pct": beta_contrib_pct,
            "contribution_alpha_pct": alpha_contrib_pct,
        }

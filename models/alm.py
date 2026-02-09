"""
Gestion Actif-Passif (ALM) et Liability-Driven Investing (LDI)
pour les fonds de pension.
"""

import numpy as np
import cvxpy as cp
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from models.base import OptimizationResult


@dataclass
class LiabilityProfile:
    """Profil du passif du fonds de pension."""
    present_value: float = 950_000_000.0
    duration: float = 15.0
    convexity: float = 250.0
    discount_rate: float = 0.05
    inflation_sensitivity: float = 0.30
    growth_rate: float = 0.03


class ALMOptimizer:
    """Optimiseur Actif-Passif pour les fonds de pension."""

    def __init__(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        asset_durations: np.ndarray,
        liability_profile: LiabilityProfile,
        risk_free_rate: float = 0.025,
        asset_names: Optional[List[str]] = None,
        min_weights: Optional[np.ndarray] = None,
        max_weights: Optional[np.ndarray] = None,
    ):
        self.mu = expected_returns
        self.sigma = cov_matrix
        self.n_assets = len(expected_returns)
        self.durations = asset_durations
        self.liability = liability_profile
        self.rf = risk_free_rate
        self.asset_names = asset_names or [f"Actif_{i}" for i in range(self.n_assets)]
        self.min_weights = min_weights if min_weights is not None else np.zeros(self.n_assets)
        self.max_weights = max_weights if max_weights is not None else np.ones(self.n_assets)

    def compute_funded_ratio(self, asset_value: float) -> float:
        """FR = Actifs / VP(Passifs)"""
        return asset_value / self.liability.present_value

    def compute_surplus(self, asset_value: float) -> float:
        """S = Actifs - VP(Passifs)"""
        return asset_value - self.liability.present_value

    def compute_duration_gap(
        self, weights: np.ndarray, asset_value: float
    ) -> float:
        """
        Ecart de duration = D_actifs - (L/A) * D_passifs.
        D_actifs = sum(w_i * D_i) pour les actifs a revenu fixe.
        """
        asset_duration = weights @ self.durations
        leverage = self.liability.present_value / asset_value
        return asset_duration - leverage * self.liability.duration

    def compute_interest_rate_sensitivity(
        self, weights: np.ndarray, asset_value: float, rate_change_bps: float = 100,
    ) -> Dict:
        """Sensibilite du surplus aux variations de taux d'interet."""
        rate_change = rate_change_bps / 10000

        # Impact sur les actifs (via duration)
        asset_duration = weights @ self.durations
        asset_change = -asset_duration * rate_change * asset_value

        # Impact sur le passif (via duration du passif)
        liability_change = -self.liability.duration * rate_change * self.liability.present_value

        surplus_change = asset_change - liability_change

        return {
            "variation_taux_bps": rate_change_bps,
            "impact_actif": float(asset_change),
            "impact_passif": float(liability_change),
            "impact_surplus": float(surplus_change),
            "impact_ratio_capit": float(surplus_change / self.liability.present_value),
        }

    def optimize_surplus(
        self,
        asset_value: float,
        target_surplus_return: Optional[float] = None,
        constraint_set=None,
    ) -> OptimizationResult:
        """
        Optimisation du surplus.

        Maximise le rendement du surplus sous contrainte de volatilite du surplus.
        R_S = R_A - (L/A)*R_L
        """
        start_time = time.time()
        leverage = self.liability.present_value / asset_value

        w = cp.Variable(self.n_assets)

        # Variance du surplus (approximation: passif correle aux obligations)
        # sigma_S^2 ≈ w'Sigma_A w (simplification car passif est deterministe ici)
        surplus_variance = cp.quad_form(w, self.sigma)
        surplus_return = self.mu @ w - leverage * self.liability.growth_rate

        if constraint_set is not None:
            from constraints.manager import ConstraintManager
            cm = ConstraintManager(self.n_assets)
            constraints = cm.to_cvxpy_constraints(w, constraint_set, self.sigma)
        else:
            constraints = [
                cp.sum(w) == 1,
                w >= self.min_weights,
                w <= self.max_weights,
            ]

        # Objectif: maximiser rendement surplus pour variance donnee
        # ou minimiser variance surplus
        lambda_risk = 5.0  # Aversion au risque
        objective = cp.Maximize(surplus_return - lambda_risk * surplus_variance)

        try:
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.CLARABEL, verbose=False)

            if prob.status in ["optimal", "optimal_inaccurate"]:
                w_optimal = w.value
                w_optimal = np.maximum(w_optimal, 0)
                w_optimal /= w_optimal.sum()

                port_return = w_optimal @ self.mu
                port_vol = np.sqrt(w_optimal @ self.sigma @ w_optimal)
                sharpe = (port_return - self.rf) / port_vol if port_vol > 1e-10 else 0.0

                # Contributions au risque
                marginal = self.sigma @ w_optimal
                risk_contrib = w_optimal * marginal / port_vol if port_vol > 1e-10 else np.zeros(self.n_assets)

                return OptimizationResult(
                    weights=w_optimal,
                    asset_names=self.asset_names,
                    expected_return=port_return,
                    volatility=port_vol,
                    sharpe_ratio=sharpe,
                    risk_contributions=risk_contrib,
                    metadata={
                        "model": "ALM_Surplus",
                        "surplus_return": float(port_return - leverage * self.liability.growth_rate),
                        "duration_gap": float(self.compute_duration_gap(w_optimal, asset_value)),
                        "funded_ratio": float(self.compute_funded_ratio(asset_value)),
                        "solver_status": prob.status,
                    },
                    status="optimal",
                    solver_time=time.time() - start_time,
                )
            else:
                return OptimizationResult(
                    weights=np.ones(self.n_assets) / self.n_assets,
                    asset_names=self.asset_names,
                    expected_return=0.0,
                    volatility=0.0,
                    sharpe_ratio=0.0,
                    risk_contributions=np.zeros(self.n_assets),
                    metadata={"solver_status": prob.status},
                    status="infeasible",
                    solver_time=time.time() - start_time,
                )
        except cp.SolverError as e:
            return OptimizationResult(
                weights=np.ones(self.n_assets) / self.n_assets,
                asset_names=self.asset_names,
                expected_return=0.0,
                volatility=0.0,
                sharpe_ratio=0.0,
                risk_contributions=np.zeros(self.n_assets),
                metadata={"error": str(e)},
                status="solver_error",
                solver_time=time.time() - start_time,
            )

    def optimize_liability_hedge(
        self, asset_value: float, hedge_ratio_target: float = 0.80,
    ) -> Dict:
        """
        Trouve l'allocation qui atteint le ratio de couverture cible.
        Concentre sur l'appariement des durations cles.
        """
        # Actifs de couverture: obligations gouvernementales, indexees inflation
        hedging_weight_target = (
            hedge_ratio_target * self.liability.duration
            / np.max(self.durations[self.durations > 0])
        )
        hedging_weight_target = min(hedging_weight_target, 0.80)

        return {
            "ratio_couverture_cible": hedge_ratio_target,
            "poids_actifs_couverture_recommande": float(hedging_weight_target),
            "duration_passif": self.liability.duration,
            "recommandation": (
                f"Allouer environ {hedging_weight_target:.0%} aux obligations "
                f"pour atteindre un ratio de couverture de {hedge_ratio_target:.0%}"
            ),
        }

    def optimize_glide_path(
        self,
        current_funded_ratio: float,
        horizon_years: int = 10,
        target_funded_ratio: float = 1.10,
        asset_value: float = 1_000_000_000.0,
    ) -> List[Dict]:
        """
        Trajectoire de desensibilisation dynamique.
        A mesure que le ratio de capitalisation s'ameliore,
        on passe des actifs de croissance aux actifs de couverture.
        """
        glide_path = []

        for year in range(horizon_years + 1):
            # Interpolation lineaire du ratio de capitalisation projete
            progress = year / horizon_years
            projected_fr = current_funded_ratio + progress * (target_funded_ratio - current_funded_ratio)

            # Allocation croissance vs couverture
            if projected_fr < 0.85:
                growth_pct = 0.65
                hedge_pct = 0.30
                cash_pct = 0.05
            elif projected_fr < 1.0:
                growth_pct = 0.55 - 0.30 * (projected_fr - 0.85) / 0.15
                hedge_pct = 0.40 + 0.20 * (projected_fr - 0.85) / 0.15
                cash_pct = 0.05
            elif projected_fr < 1.10:
                growth_pct = 0.35 - 0.15 * (projected_fr - 1.0) / 0.10
                hedge_pct = 0.60 + 0.10 * (projected_fr - 1.0) / 0.10
                cash_pct = 0.05
            else:
                growth_pct = 0.20
                hedge_pct = 0.75
                cash_pct = 0.05

            glide_path.append({
                "Annee": year,
                "Ratio capitalisation projete": projected_fr,
                "Actifs de croissance (%)": growth_pct * 100,
                "Actifs de couverture (%)": hedge_pct * 100,
                "Encaisse (%)": cash_pct * 100,
            })

        return glide_path

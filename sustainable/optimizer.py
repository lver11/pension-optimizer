# sustainable/optimizer.py
"""
Optimiseur de portefeuille durable bi-critère (Sharpe + durabilité).

Objectif: Maximiser μ'w - (γ/2)·w'Σw + λ·S'w
"""

import time
import numpy as np
import cvxpy as cp
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.base import BaseOptimizer, OptimizationResult


@dataclass
class DurableResult(OptimizationResult):
    """Résultat d'une optimisation durable."""
    sustainability_score: float = 0.0
    lambda_used: float = 0.0
    gamma_used: float = 2.5
    sustainability_breakdown: Dict[str, float] = field(default_factory=dict)
    variant_used: Dict[str, str] = field(default_factory=dict)  # asset_name → "standard" | "durable"


class DurableOptimizer(BaseOptimizer):
    """
    Optimiseur bi-critère: rendement-risque + durabilité.

    Maximise: μ'w - (γ/2)·w'Σw + λ·S'w
    """

    def optimize(self, lam: float = 0.0, **kwargs) -> DurableResult:
        """Délègue à optimize_durable. lam=0 correspond à l'optimisation purement financière."""
        return self.optimize_durable(lam=lam, **kwargs)

    def optimize_durable(
        self,
        lam: float,
        gamma: float,
        sustainability_scores: np.ndarray,
        constraint_set=None,
        use_durable_map: Optional[Dict[str, bool]] = None,
    ) -> DurableResult:
        """
        Optimise le portefeuille pour un λ donné.

        lam: poids de la durabilité (0 = purement financier)
        gamma: aversion au risque
        sustainability_scores: vecteur (n,) de scores composites par actif
        use_durable_map: {asset_name: True} pour variante durable (propagé dans DurableResult)
        """
        start_time = time.time()
        w = cp.Variable(self.n_assets)

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

        portfolio_return = self.mu @ w
        portfolio_variance = cp.quad_form(w, self.sigma)
        sustainability_term = sustainability_scores @ w

        objective = cp.Maximize(
            portfolio_return - (gamma / 2) * portfolio_variance + lam * sustainability_term
        )

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)
        except (cp.SolverError, cp.DCPError, ValueError, Exception):
            return self._make_fallback(lam, gamma, sustainability_scores, start_time, use_durable_map)

        if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
            w_opt = np.clip(w.value, self.min_weights, self.max_weights)
            w_opt /= w_opt.sum()
            status = "optimal" if prob.status == "optimal" else "optimal_inaccurate"
            return self._build_durable_result(
                w_opt, status, start_time, lam, gamma, sustainability_scores, use_durable_map
            )
        else:
            return self._make_fallback(lam, gamma, sustainability_scores, start_time, use_durable_map)

    def _make_fallback(self, lam, gamma, sustainability_scores, start_time, use_durable_map=None):
        w_eq = np.ones(self.n_assets) / self.n_assets
        w_eq = np.clip(w_eq, self.min_weights, self.max_weights)
        if w_eq.sum() > 1e-10:
            w_eq /= w_eq.sum()
        return self._build_durable_result(
            w_eq, "infeasible", start_time, lam, gamma, sustainability_scores, use_durable_map
        )

    def _build_durable_result(
        self, weights, status, start_time, lam, gamma, sustainability_scores,
        use_durable_map=None,
    ) -> DurableResult:
        port_return, port_vol, sharpe = self._compute_portfolio_stats(weights)
        risk_contrib = self._compute_risk_contributions(weights)
        sustain_score = float(sustainability_scores @ weights)
        sustainability_breakdown = {
            self.asset_names[i]: float(weights[i] * sustainability_scores[i])
            for i in range(self.n_assets)
        }
        variant_used = {}
        if use_durable_map:
            for i, name in enumerate(self.asset_names):
                variant_used[name] = "durable" if use_durable_map.get(name, False) else "standard"
        return DurableResult(
            weights=weights,
            asset_names=self.asset_names,
            expected_return=port_return,
            volatility=port_vol,
            sharpe_ratio=sharpe,
            risk_contributions=risk_contrib,
            sustainability_score=sustain_score,
            sustainability_breakdown=sustainability_breakdown,
            lambda_used=lam,
            gamma_used=gamma,
            variant_used=variant_used,
            status=status,
            solver_time=time.time() - start_time,
        )

    def pareto_frontier(
        self,
        n_points: int = 50,
        gamma: float = 2.5,
        sustainability_scores: Optional[np.ndarray] = None,
        lambda_max: float = 5.0,
        constraint_set=None,
        use_durable_map: Optional[Dict[str, bool]] = None,
    ) -> List[DurableResult]:
        """
        Calcule n_points sur la frontière Pareto en faisant varier λ de 0 à lambda_max.
        Retourne les points faisables triés par score de durabilité croissant.
        """
        if sustainability_scores is None:
            sustainability_scores = np.ones(self.n_assets)

        lambdas = np.linspace(0, lambda_max, n_points)
        results = []
        for lam in lambdas:
            r = self.optimize_durable(lam, gamma, sustainability_scores, constraint_set, use_durable_map)
            if r.status in ("optimal", "optimal_inaccurate"):
                results.append(r)

        results.sort(key=lambda r: r.sustainability_score)
        return results

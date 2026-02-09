"""
Optimisation CVaR (Conditional Value-at-Risk) via la formulation
lineaire de Rockafellar & Uryasev (2000) avec cvxpy.
"""

import numpy as np
import cvxpy as cp
import time
from typing import Dict, List, Optional
from models.base import BaseOptimizer, OptimizationResult


class CVaROptimizer(BaseOptimizer):
    """Optimiseur CVaR utilisant la programmation lineaire."""

    def __init__(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        scenarios: np.ndarray,
        confidence_level: float = 0.95,
        risk_free_rate: float = 0.025,
        asset_names: Optional[List[str]] = None,
        min_weights: Optional[np.ndarray] = None,
        max_weights: Optional[np.ndarray] = None,
    ):
        super().__init__(
            expected_returns, cov_matrix, risk_free_rate,
            asset_names, min_weights, max_weights,
        )
        self.scenarios = scenarios  # (S, n) matrice de rendements par scenario
        self.beta = confidence_level
        self.S = scenarios.shape[0]

    def optimize(
        self,
        objective: str = "min_cvar",
        target_return: Optional[float] = None,
        target_cvar: Optional[float] = None,
        constraint_set=None,
    ) -> OptimizationResult:
        """
        Optimise le portefeuille en minimisant la CVaR.

        Formulation Rockafellar-Uryasev:
        min alpha + 1/((1-beta)*S) * sum(u_s)
        s.t. u_s >= -r_s^T * w - alpha, u_s >= 0

        Objectifs:
        - min_cvar: Minimise la CVaR
        - target_return: Minimise CVaR pour un rendement cible
        - max_return_target_cvar: Maximise le rendement pour une CVaR cible
        """
        start_time = time.time()

        w = cp.Variable(self.n_assets)
        alpha = cp.Variable()  # Seuil VaR
        u = cp.Variable(self.S)  # Variables auxiliaires

        # Pertes par scenario
        losses = -self.scenarios @ w  # (S,)

        # Contraintes CVaR
        constraints = [
            u >= losses - alpha,
            u >= 0,
        ]

        # Contraintes de portefeuille
        if constraint_set is not None:
            from constraints.manager import ConstraintManager
            cm = ConstraintManager(self.n_assets)
            constraints += cm.to_cvxpy_constraints(w, constraint_set, self.sigma)
        else:
            constraints += [
                cp.sum(w) == 1,
                w >= self.min_weights,
                w <= self.max_weights,
            ]

        # CVaR
        cvar = alpha + cp.sum(u) / ((1 - self.beta) * self.S)
        portfolio_return = self.mu @ w

        try:
            if objective == "min_cvar":
                prob = cp.Problem(cp.Minimize(cvar), constraints)

            elif objective == "target_return":
                if target_return is None:
                    target_return = np.mean(self.mu)
                constraints.append(portfolio_return >= target_return)
                prob = cp.Problem(cp.Minimize(cvar), constraints)

            elif objective == "max_return_target_cvar":
                if target_cvar is None:
                    target_cvar = 0.15
                constraints.append(cvar <= target_cvar)
                prob = cp.Problem(cp.Maximize(portfolio_return), constraints)

            else:
                raise ValueError(f"Objectif inconnu: {objective}")

            prob.solve(solver=cp.CLARABEL, verbose=False)

            if prob.status in ["optimal", "optimal_inaccurate"]:
                w_optimal = w.value
                w_optimal = np.maximum(w_optimal, 0)
                w_optimal /= w_optimal.sum()

                # Calculer VaR et CVaR du portefeuille optimal
                port_scenario_returns = self.scenarios @ w_optimal
                var_value = -np.percentile(port_scenario_returns, (1 - self.beta) * 100)
                tail_losses = port_scenario_returns[port_scenario_returns <= -var_value]
                cvar_value = -np.mean(tail_losses) if len(tail_losses) > 0 else var_value

                return self._build_result(
                    w_optimal, "optimal", start_time,
                    {
                        "objective": objective,
                        "var": float(var_value),
                        "cvar": float(cvar_value),
                        "confidence_level": self.beta,
                        "n_scenarios": self.S,
                        "solver_status": prob.status,
                    }
                )
            else:
                return self._build_result(
                    np.ones(self.n_assets) / self.n_assets,
                    "infeasible", start_time,
                    {"objective": objective, "solver_status": prob.status}
                )

        except cp.SolverError as e:
            return self._build_result(
                np.ones(self.n_assets) / self.n_assets,
                "solver_error", start_time,
                {"objective": objective, "error": str(e)}
            )

    def efficient_frontier_cvar(
        self, n_points: int = 30, constraint_set=None,
    ) -> List[OptimizationResult]:
        """Trace la frontiere efficiente moyenne-CVaR."""
        # Trouver le rendement min (min CVaR sans contrainte de rendement)
        min_cvar_result = self.optimize("min_cvar", constraint_set=constraint_set)
        min_return = min_cvar_result.expected_return

        # Rendement max
        max_return = np.max(self.mu)
        if constraint_set is not None:
            max_return = min(max_return, np.sum(self.mu * constraint_set.max_weights))

        target_returns = np.linspace(min_return, max_return * 0.90, n_points)

        frontier_points = []
        for target in target_returns:
            result = self.optimize(
                "target_return",
                target_return=target,
                constraint_set=constraint_set,
            )
            if result.status == "optimal":
                frontier_points.append(result)

        return frontier_points

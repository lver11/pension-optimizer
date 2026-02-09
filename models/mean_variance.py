"""
Optimisation Moyenne-Variance (Markowitz) avec cvxpy.
"""

import numpy as np
import cvxpy as cp
import time
from typing import Dict, List, Optional
from models.base import BaseOptimizer, OptimizationResult


class MeanVarianceOptimizer(BaseOptimizer):
    """Optimiseur Moyenne-Variance classique (Markowitz) via cvxpy."""

    def optimize(
        self,
        objective: str = "max_sharpe",
        target_return: Optional[float] = None,
        target_risk: Optional[float] = None,
        constraint_set=None,
    ) -> OptimizationResult:
        """
        Optimise le portefeuille selon l'objectif choisi.

        Objectifs:
        - max_sharpe: Maximise le ratio de Sharpe
        - min_variance: Minimise la variance du portefeuille
        - target_return: Minimise la variance pour un rendement cible
        - target_risk: Maximise le rendement pour un risque cible
        """
        start_time = time.time()

        w = cp.Variable(self.n_assets)

        # Contraintes de base
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

        portfolio_variance = cp.quad_form(w, self.sigma)
        portfolio_return = self.mu @ w

        try:
            if objective == "min_variance":
                prob = cp.Problem(cp.Minimize(portfolio_variance), constraints)
                prob.solve(solver=cp.CLARABEL, verbose=False)

            elif objective == "target_return":
                if target_return is None:
                    target_return = np.mean(self.mu)
                constraints.append(portfolio_return >= target_return)
                prob = cp.Problem(cp.Minimize(portfolio_variance), constraints)
                prob.solve(solver=cp.CLARABEL, verbose=False)

            elif objective == "target_risk":
                if target_risk is None:
                    target_risk = 0.10
                constraints.append(portfolio_variance <= target_risk ** 2)
                prob = cp.Problem(cp.Maximize(portfolio_return), constraints)
                prob.solve(solver=cp.CLARABEL, verbose=False)

            elif objective == "max_sharpe":
                # Transformation de Cornuejols-Tutuncu pour convexifier
                y = cp.Variable(self.n_assets)
                kappa = cp.Variable()

                sharpe_constraints = [
                    cp.sum(y) == kappa,
                    (self.mu - self.rf) @ y == 1,
                    y >= self.min_weights * kappa,
                    y <= self.max_weights * kappa,
                    kappa >= 0,
                ]

                # Ajouter les contraintes de groupe si presentes
                if constraint_set is not None:
                    for gc in constraint_set.group_constraints:
                        group_sum = cp.sum(y[gc.asset_indices])
                        sharpe_constraints.append(group_sum >= gc.min_allocation * kappa)
                        sharpe_constraints.append(group_sum <= gc.max_allocation * kappa)

                    if constraint_set.esg_min_score is not None and constraint_set.esg_scores is not None:
                        sharpe_constraints.append(
                            constraint_set.esg_scores @ y >= constraint_set.esg_min_score * kappa
                        )

                prob = cp.Problem(
                    cp.Minimize(cp.quad_form(y, self.sigma)),
                    sharpe_constraints,
                )
                prob.solve(solver=cp.CLARABEL, verbose=False)

                if prob.status in ["optimal", "optimal_inaccurate"]:
                    kappa_val = kappa.value
                    if kappa_val is not None and kappa_val > 1e-8:
                        w_optimal = y.value / kappa_val
                        return self._build_result(
                            w_optimal, "optimal", start_time,
                            {"objective": objective, "solver_status": prob.status}
                        )
                return self._build_result(
                    np.ones(self.n_assets) / self.n_assets,
                    "infeasible", start_time,
                    {"objective": objective, "solver_status": prob.status}
                )
            else:
                raise ValueError(f"Objectif inconnu: {objective}")

            if prob.status in ["optimal", "optimal_inaccurate"]:
                w_optimal = w.value
                w_optimal = np.maximum(w_optimal, 0)
                w_optimal /= w_optimal.sum()
                return self._build_result(
                    w_optimal, "optimal", start_time,
                    {"objective": objective, "solver_status": prob.status}
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

    def efficient_frontier_points(
        self, n_points: int = 50, constraint_set=None,
    ) -> List[OptimizationResult]:
        """
        Calcule n_points le long de la frontiere efficiente
        en variant le rendement cible de min_variance a max_return.
        """
        min_var_result = self.optimize("min_variance", constraint_set=constraint_set)
        min_return = min_var_result.expected_return

        max_return = np.max(self.mu)
        if constraint_set is not None:
            max_return = min(max_return, np.sum(self.mu * constraint_set.max_weights))

        target_returns = np.linspace(min_return, max_return * 0.95, n_points)

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

"""
Optimisation Parite de Risque / Equal Risk Contribution avec cvxpy.
"""

import numpy as np
import cvxpy as cp
import time
from typing import Dict, List, Optional
from scipy.optimize import minimize
from models.base import BaseOptimizer, OptimizationResult


class RiskParityOptimizer(BaseOptimizer):
    """Optimiseur Parite de Risque (Equal Risk Contribution) et Risk Budgeting."""

    def __init__(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_budgets: Optional[np.ndarray] = None,
        risk_free_rate: float = 0.025,
        asset_names: Optional[List[str]] = None,
        min_weights: Optional[np.ndarray] = None,
        max_weights: Optional[np.ndarray] = None,
    ):
        super().__init__(
            expected_returns, cov_matrix, risk_free_rate,
            asset_names, min_weights, max_weights,
        )
        if risk_budgets is not None:
            self.budgets = risk_budgets / risk_budgets.sum()
        else:
            self.budgets = np.ones(self.n_assets) / self.n_assets

    def optimize(
        self,
        method: str = "log_barrier",
        constraint_set=None,
        **kwargs,
    ) -> OptimizationResult:
        """
        Optimise le portefeuille en parite de risque.

        Methodes:
        - log_barrier: min w'Sigma*w - sum(b_i * log(w_i)) via cvxpy
        - scipy_rc: minimise la difference des contributions au risque via scipy
        """
        start_time = time.time()

        if method == "log_barrier":
            return self._optimize_log_barrier(start_time, constraint_set)
        else:
            return self._optimize_scipy_rc(start_time, constraint_set)

    def _optimize_log_barrier(
        self, start_time: float, constraint_set=None
    ) -> OptimizationResult:
        """
        Formulation log-barrier:
        min w'Sigma*w - sum(b_i * log(w_i))
        Puis normalisation: w_final = w / sum(w)
        """
        w = cp.Variable(self.n_assets, pos=True)

        portfolio_risk = cp.quad_form(w, self.sigma)
        log_barrier = self.budgets @ cp.log(w)

        objective = cp.Minimize(portfolio_risk - log_barrier)

        constraints = [w >= 1e-6]

        try:
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.CLARABEL, verbose=False)

            if prob.status in ["optimal", "optimal_inaccurate"]:
                w_raw = w.value
                w_normalized = w_raw / w_raw.sum()

                # Appliquer les bornes si necessaire
                if constraint_set is not None:
                    w_normalized = np.clip(
                        w_normalized,
                        constraint_set.min_weights,
                        constraint_set.max_weights,
                    )
                    w_normalized /= w_normalized.sum()

                risk_contrib = self.compute_risk_contributions(w_normalized)

                return self._build_result(
                    w_normalized, "optimal", start_time,
                    {
                        "method": "log_barrier",
                        "risk_budgets": self.budgets.tolist(),
                        "risk_contributions_pct": (risk_contrib / risk_contrib.sum()).tolist()
                        if risk_contrib.sum() > 0 else self.budgets.tolist(),
                        "solver_status": prob.status,
                    }
                )
            else:
                return self._optimize_scipy_rc(start_time, constraint_set)

        except cp.SolverError:
            return self._optimize_scipy_rc(start_time, constraint_set)

    def _optimize_scipy_rc(
        self, start_time: float, constraint_set=None
    ) -> OptimizationResult:
        """
        Methode alternative scipy: minimise sum_ij (RC_i/b_i - RC_j/b_j)^2
        """
        def risk_budget_objective(w):
            sigma_w = self.sigma @ w
            port_vol = np.sqrt(w @ sigma_w)
            if port_vol < 1e-10:
                return 1e10
            rc = w * sigma_w / port_vol
            # On veut RC_i / b_i = RC_j / b_j pour tout i,j
            rc_normalized = rc / self.budgets
            return np.sum((rc_normalized - rc_normalized.mean()) ** 2)

        # Point initial: proportionnel a 1/vol
        inv_vol = 1 / np.sqrt(np.diag(self.sigma))
        w0 = inv_vol / inv_vol.sum()

        if constraint_set is not None:
            bounds = list(zip(
                constraint_set.min_weights.tolist(),
                constraint_set.max_weights.tolist()
            ))
        else:
            bounds = [(max(0.001, self.min_weights[i]), self.max_weights[i])
                      for i in range(self.n_assets)]

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        result = minimize(
            risk_budget_objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        if result.success:
            w_optimal = result.x
            w_optimal = np.maximum(w_optimal, 0)
            w_optimal /= w_optimal.sum()
            risk_contrib = self.compute_risk_contributions(w_optimal)

            return self._build_result(
                w_optimal, "optimal", start_time,
                {
                    "method": "scipy_rc",
                    "risk_budgets": self.budgets.tolist(),
                    "risk_contributions_pct": (risk_contrib / risk_contrib.sum()).tolist()
                    if risk_contrib.sum() > 0 else self.budgets.tolist(),
                    "scipy_message": result.message,
                }
            )
        else:
            w_equal = np.ones(self.n_assets) / self.n_assets
            return self._build_result(
                w_equal, "suboptimal", start_time,
                {"method": "scipy_rc", "scipy_message": result.message}
            )

    def compute_risk_contributions(self, weights: np.ndarray) -> np.ndarray:
        """RC_i = w_i * (Sigma @ w)_i / sqrt(w^T @ Sigma @ w)"""
        sigma_w = self.sigma @ weights
        port_vol = np.sqrt(weights @ self.sigma @ weights)
        if port_vol < 1e-10:
            return np.zeros(self.n_assets)
        return weights * sigma_w / port_vol

"""
Modele Black-Litterman combinant l'equilibre de marche avec les vues de l'investisseur.
"""

import numpy as np
import time
from typing import Dict, List, Optional, Tuple
from models.base import BaseOptimizer, OptimizationResult


class BlackLittermanOptimizer(BaseOptimizer):
    """Modele Black-Litterman pour l'optimisation de portefeuille."""

    def __init__(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        market_weights: np.ndarray,
        risk_free_rate: float = 0.025,
        tau: float = 0.05,
        asset_names: Optional[List[str]] = None,
        min_weights: Optional[np.ndarray] = None,
        max_weights: Optional[np.ndarray] = None,
    ):
        super().__init__(
            expected_returns, cov_matrix, risk_free_rate,
            asset_names, min_weights, max_weights,
        )
        self.w_mkt = market_weights
        self.tau = tau
        self.P = None
        self.Q = None
        self.omega = None

    def set_views(
        self,
        P: np.ndarray,
        Q: np.ndarray,
        omega: Optional[np.ndarray] = None,
        confidence: Optional[np.ndarray] = None,
    ):
        """
        Definit les vues de l'investisseur.

        P: matrice de selection (k x n), k vues, n actifs
        Q: vecteur de rendements attendus des vues (k,)
        omega: matrice d'incertitude des vues (k x k)
        confidence: niveaux de confiance par vue (0-1)
        """
        self.P = np.atleast_2d(P)
        self.Q = np.atleast_1d(Q)

        if omega is not None:
            self.omega = omega
        elif confidence is not None:
            confidence = np.atleast_1d(confidence)
            base_omega = np.diag(np.diag(self.P @ (self.tau * self.sigma) @ self.P.T))
            scaling = np.array([(1 - c) / c if c > 0.01 else 100.0 for c in confidence])
            self.omega = base_omega * np.diag(scaling)
        else:
            self.omega = np.diag(np.diag(self.P @ (self.tau * self.sigma) @ self.P.T))

    def compute_implied_returns(self) -> np.ndarray:
        """
        Optimisation inverse: extrait les rendements d'equilibre implicites.
        Pi = delta * Sigma * w_mkt
        """
        mkt_return = self.mu @ self.w_mkt
        mkt_var = self.w_mkt @ self.sigma @ self.w_mkt
        delta = (mkt_return - self.rf) / mkt_var
        self._delta = delta
        return delta * self.sigma @ self.w_mkt

    def compute_posterior(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcule les rendements posterieurs et la covariance posterieure.
        """
        Pi = self.compute_implied_returns()

        if self.P is None or self.Q is None:
            return Pi, self.sigma

        tau_sigma = self.tau * self.sigma
        tau_sigma_inv = np.linalg.inv(tau_sigma)
        omega_inv = np.linalg.inv(self.omega)

        M = tau_sigma_inv + self.P.T @ omega_inv @ self.P
        M_inv = np.linalg.inv(M)

        mu_BL = M_inv @ (tau_sigma_inv @ Pi + self.P.T @ omega_inv @ self.Q)
        sigma_BL = M_inv + self.sigma

        return mu_BL, sigma_BL

    def optimize(
        self,
        objective: str = "max_sharpe",
        constraint_set=None,
        **kwargs,
    ) -> OptimizationResult:
        """
        Calcule le posterieur puis delegue a l'optimiseur Moyenne-Variance.
        """
        start_time = time.time()

        mu_BL, sigma_BL = self.compute_posterior()

        from models.mean_variance import MeanVarianceOptimizer
        mv_optimizer = MeanVarianceOptimizer(
            expected_returns=mu_BL,
            cov_matrix=sigma_BL,
            risk_free_rate=self.rf,
            asset_names=self.asset_names,
            min_weights=self.min_weights,
            max_weights=self.max_weights,
        )

        result = mv_optimizer.optimize(
            objective=objective,
            constraint_set=constraint_set,
            **kwargs,
        )

        result.metadata.update({
            "model": "Black-Litterman",
            "tau": self.tau,
            "implied_returns": self.compute_implied_returns().tolist(),
            "posterior_returns": mu_BL.tolist(),
            "n_views": 0 if self.P is None else self.P.shape[0],
        })
        result.solver_time = time.time() - start_time

        return result

    @staticmethod
    def build_view_absolute(
        asset_index: int, n_assets: int, expected_return: float
    ) -> Tuple[np.ndarray, float]:
        """Construit une vue absolue: L'actif i rendra X%."""
        P_row = np.zeros(n_assets)
        P_row[asset_index] = 1.0
        return P_row, expected_return

    @staticmethod
    def build_view_relative(
        asset_long: int, asset_short: int, n_assets: int, spread: float
    ) -> Tuple[np.ndarray, float]:
        """Construit une vue relative: L'actif i surperformera l'actif j de X%."""
        P_row = np.zeros(n_assets)
        P_row[asset_long] = 1.0
        P_row[asset_short] = -1.0
        return P_row, spread

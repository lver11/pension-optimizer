"""
Classe abstraite de base pour tous les optimiseurs de portefeuille.
"""

import numpy as np
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class OptimizationResult:
    """Resultat d'une optimisation de portefeuille."""
    weights: np.ndarray
    asset_names: List[str]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    risk_contributions: np.ndarray
    metadata: Dict = field(default_factory=dict)
    status: str = "optimal"
    solver_time: float = 0.0

    def to_dict(self) -> Dict:
        """Convertit le resultat en dictionnaire."""
        return {
            "weights": dict(zip(self.asset_names, self.weights.tolist())),
            "expected_return": self.expected_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "risk_contributions": dict(
                zip(self.asset_names, self.risk_contributions.tolist())
            ),
            "status": self.status,
            "solver_time": self.solver_time,
        }


class BaseOptimizer(ABC):
    """Classe abstraite de base pour tous les modeles d'optimisation."""

    def __init__(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_free_rate: float = 0.025,
        asset_names: Optional[List[str]] = None,
        min_weights: Optional[np.ndarray] = None,
        max_weights: Optional[np.ndarray] = None,
    ):
        self.mu = expected_returns
        self.sigma = cov_matrix
        self.rf = risk_free_rate
        self.n_assets = len(expected_returns)
        self.asset_names = asset_names or [f"Actif_{i}" for i in range(self.n_assets)]
        self.min_weights = min_weights if min_weights is not None else np.zeros(self.n_assets)
        self.max_weights = max_weights if max_weights is not None else np.ones(self.n_assets)
        self._validate_inputs()

    def _validate_inputs(self):
        """Verifie les dimensions et la PSD de la matrice de covariance."""
        assert len(self.mu) == self.n_assets, "Dimension mu incorrecte"
        assert self.sigma.shape == (self.n_assets, self.n_assets), "Dimension sigma incorrecte"
        # Verifier symetrie
        if not np.allclose(self.sigma, self.sigma.T, atol=1e-8):
            self.sigma = (self.sigma + self.sigma.T) / 2
        # Verifier PSD
        eigvals = np.linalg.eigvalsh(self.sigma)
        if np.min(eigvals) < -1e-8:
            from risk.covariance import CovarianceEstimator
            self.sigma = CovarianceEstimator.nearest_psd(self.sigma)

    @abstractmethod
    def optimize(self, **kwargs) -> OptimizationResult:
        """Execute l'optimisation. A implementer par les sous-classes."""
        pass

    def _compute_portfolio_stats(
        self, weights: np.ndarray
    ) -> Tuple[float, float, float]:
        """Calcule rendement, volatilite et ratio de Sharpe du portefeuille."""
        port_return = weights @ self.mu
        port_vol = np.sqrt(weights @ self.sigma @ weights)
        sharpe = (port_return - self.rf) / port_vol if port_vol > 1e-10 else 0.0
        return port_return, port_vol, sharpe

    def _compute_risk_contributions(self, weights: np.ndarray) -> np.ndarray:
        """Calcule la contribution au risque de chaque actif."""
        port_vol = np.sqrt(weights @ self.sigma @ weights)
        if port_vol < 1e-10:
            return np.zeros(self.n_assets)
        marginal_risk = self.sigma @ weights
        risk_contrib = weights * marginal_risk / port_vol
        return risk_contrib

    def _build_result(
        self, weights: np.ndarray, status: str, start_time: float, metadata: Dict = None
    ) -> OptimizationResult:
        """Construit un OptimizationResult a partir des poids optimaux."""
        port_return, port_vol, sharpe = self._compute_portfolio_stats(weights)
        risk_contrib = self._compute_risk_contributions(weights)
        return OptimizationResult(
            weights=weights,
            asset_names=self.asset_names,
            expected_return=port_return,
            volatility=port_vol,
            sharpe_ratio=sharpe,
            risk_contributions=risk_contrib,
            metadata=metadata or {},
            status=status,
            solver_time=time.time() - start_time,
        )

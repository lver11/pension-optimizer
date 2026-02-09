"""
Calcul de la frontiere efficiente pour differents modeles d'optimisation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from models.base import OptimizationResult


class EfficientFrontierComputer:
    """Calcule les frontieres efficientes pour differents modeles."""

    def __init__(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        scenarios: Optional[np.ndarray] = None,
        risk_free_rate: float = 0.025,
        asset_names: Optional[List[str]] = None,
        min_weights: Optional[np.ndarray] = None,
        max_weights: Optional[np.ndarray] = None,
    ):
        self.mu = expected_returns
        self.sigma = cov_matrix
        self.scenarios = scenarios
        self.rf = risk_free_rate
        self.n_assets = len(expected_returns)
        self.asset_names = asset_names or [f"Actif_{i}" for i in range(self.n_assets)]
        self.min_weights = min_weights if min_weights is not None else np.zeros(self.n_assets)
        self.max_weights = max_weights if max_weights is not None else np.ones(self.n_assets)

    def compute_mv_frontier(
        self, n_points: int = 50, constraint_set=None,
    ) -> pd.DataFrame:
        """
        Frontiere efficiente Moyenne-Variance.
        Retourne un DataFrame avec colonnes: [return, volatility, sharpe, weights...]
        """
        from models.mean_variance import MeanVarianceOptimizer

        optimizer = MeanVarianceOptimizer(
            self.mu, self.sigma, self.rf,
            self.asset_names, self.min_weights, self.max_weights,
        )

        results = optimizer.efficient_frontier_points(n_points, constraint_set)
        return self._results_to_dataframe(results)

    def compute_cvar_frontier(
        self, n_points: int = 30, constraint_set=None,
    ) -> pd.DataFrame:
        """
        Frontiere efficiente Moyenne-CVaR.
        Necessite des scenarios de rendement.
        """
        if self.scenarios is None:
            raise ValueError("Les scenarios sont requis pour la frontiere CVaR")

        from models.cvar_optimizer import CVaROptimizer

        optimizer = CVaROptimizer(
            self.mu, self.sigma, self.scenarios, 0.95,
            self.rf, self.asset_names, self.min_weights, self.max_weights,
        )

        results = optimizer.efficient_frontier_cvar(n_points, constraint_set)
        return self._results_to_dataframe(results, risk_col="cvar")

    def compute_unconstrained_frontier(self, n_points: int = 50) -> pd.DataFrame:
        """Frontiere sans contraintes (pour comparaison)."""
        from models.mean_variance import MeanVarianceOptimizer

        optimizer = MeanVarianceOptimizer(
            self.mu, self.sigma, self.rf,
            self.asset_names,
            np.zeros(self.n_assets),
            np.ones(self.n_assets),
        )

        results = optimizer.efficient_frontier_points(n_points)
        return self._results_to_dataframe(results)

    def find_tangency_portfolio(self, constraint_set=None) -> OptimizationResult:
        """Portefeuille tangent (ratio de Sharpe maximum)."""
        from models.mean_variance import MeanVarianceOptimizer

        optimizer = MeanVarianceOptimizer(
            self.mu, self.sigma, self.rf,
            self.asset_names, self.min_weights, self.max_weights,
        )
        return optimizer.optimize("max_sharpe", constraint_set=constraint_set)

    def find_min_variance_portfolio(self, constraint_set=None) -> OptimizationResult:
        """Portefeuille a variance minimale globale."""
        from models.mean_variance import MeanVarianceOptimizer

        optimizer = MeanVarianceOptimizer(
            self.mu, self.sigma, self.rf,
            self.asset_names, self.min_weights, self.max_weights,
        )
        return optimizer.optimize("min_variance", constraint_set=constraint_set)

    def _results_to_dataframe(
        self, results: List[OptimizationResult], risk_col: str = "volatility"
    ) -> pd.DataFrame:
        """Convertit une liste de resultats en DataFrame."""
        records = []
        for r in results:
            record = {
                "return": r.expected_return,
                "volatility": r.volatility,
                "sharpe": r.sharpe_ratio,
            }
            if risk_col == "cvar" and "cvar" in r.metadata:
                record["cvar"] = r.metadata["cvar"]
            for i, name in enumerate(r.asset_names):
                record[f"w_{name}"] = r.weights[i]
            records.append(record)

        return pd.DataFrame(records)

    def compute_capital_market_line(
        self, tangency: OptimizationResult, n_points: int = 50,
    ) -> pd.DataFrame:
        """
        Calcule la ligne du marche des capitaux (CML).
        CML: E[r] = rf + (E[r_tang] - rf) / vol_tang * vol
        """
        max_vol = tangency.volatility * 2.0
        vols = np.linspace(0, max_vol, n_points)
        slope = (tangency.expected_return - self.rf) / tangency.volatility
        returns = self.rf + slope * vols

        return pd.DataFrame({
            "volatility": vols,
            "return": returns,
        })

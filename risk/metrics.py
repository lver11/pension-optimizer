"""
Metriques de risque completes pour l'analyse de portefeuille.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from scipy import stats


class RiskMetrics:
    """Calcul complet des metriques de risque de portefeuille."""

    @staticmethod
    def value_at_risk(
        returns: np.ndarray,
        confidence: float = 0.95,
        method: str = "historical",
    ) -> float:
        """
        Valeur a Risque (VaR).

        Methodes:
        - historical: VaR = -percentile(returns, (1-confidence)*100)
        - parametric: VaR = -(mu + z_alpha * sigma)
        - cornish_fisher: ajuste z pour asymetrie et kurtosis
        """
        if method == "historical":
            return -np.percentile(returns, (1 - confidence) * 100)
        elif method == "parametric":
            mu = np.mean(returns)
            sigma = np.std(returns)
            z = stats.norm.ppf(1 - confidence)
            return -(mu + z * sigma)
        elif method == "cornish_fisher":
            mu = np.mean(returns)
            sigma = np.std(returns)
            s = stats.skew(returns)
            k = stats.kurtosis(returns)
            z = stats.norm.ppf(1 - confidence)
            z_cf = (z + (z**2 - 1) * s / 6
                    + (z**3 - 3*z) * k / 24
                    - (2*z**3 - 5*z) * s**2 / 36)
            return -(mu + z_cf * sigma)
        return -np.percentile(returns, (1 - confidence) * 100)

    @staticmethod
    def conditional_var(
        returns: np.ndarray, confidence: float = 0.95,
    ) -> float:
        """CVaR = -E[r | r <= -VaR] = moyenne des pertes au-dela de la VaR."""
        var = RiskMetrics.value_at_risk(returns, confidence, "historical")
        tail_returns = returns[returns <= -var]
        if len(tail_returns) == 0:
            return var
        return -np.mean(tail_returns)

    @staticmethod
    def sharpe_ratio(
        returns: np.ndarray, rf: float = 0.025, annualize: bool = True,
        periods_per_year: int = 12,
    ) -> float:
        """Ratio de Sharpe annualise."""
        excess = returns - rf / periods_per_year
        if annualize:
            return np.mean(excess) * np.sqrt(periods_per_year) / np.std(returns) if np.std(returns) > 1e-10 else 0.0
        return np.mean(excess) / np.std(returns) if np.std(returns) > 1e-10 else 0.0

    @staticmethod
    def sortino_ratio(
        returns: np.ndarray,
        rf: float = 0.025,
        target: float = 0.0,
        periods_per_year: int = 12,
    ) -> float:
        """
        Ratio de Sortino = (rendement moyen - rf) / deviation a la baisse.
        """
        excess_mean = np.mean(returns) - rf / periods_per_year
        downside = returns[returns < target] - target
        downside_std = np.sqrt(np.mean(downside**2)) if len(downside) > 0 else 1e-10
        return excess_mean * np.sqrt(periods_per_year) / downside_std if downside_std > 1e-10 else 0.0

    @staticmethod
    def maximum_drawdown(returns: np.ndarray) -> Tuple[float, int, int]:
        """
        Perte maximale (Maximum Drawdown).
        Retourne (mdd, index_pic, index_creux).
        """
        cum_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = (cum_returns - running_max) / running_max

        mdd = np.min(drawdowns)
        trough_idx = np.argmin(drawdowns)
        peak_idx = np.argmax(cum_returns[:trough_idx + 1])

        return abs(mdd), peak_idx, trough_idx

    @staticmethod
    def tracking_error(
        portfolio_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        periods_per_year: int = 12,
    ) -> float:
        """Erreur de suivi annualisee = std(r_p - r_b) * sqrt(periods)."""
        diff = portfolio_returns - benchmark_returns
        return np.std(diff) * np.sqrt(periods_per_year)

    @staticmethod
    def information_ratio(
        portfolio_returns: np.ndarray,
        benchmark_returns: np.ndarray,
        periods_per_year: int = 12,
    ) -> float:
        """Ratio d'information = mean(r_p - r_b) / TE."""
        diff = portfolio_returns - benchmark_returns
        te = np.std(diff)
        if te < 1e-10:
            return 0.0
        return np.mean(diff) * periods_per_year / (te * np.sqrt(periods_per_year))

    @staticmethod
    def calmar_ratio(
        returns: np.ndarray, periods_per_year: int = 12,
    ) -> float:
        """Ratio de Calmar = rendement annualise / |MDD|."""
        ann_return = np.mean(returns) * periods_per_year
        mdd, _, _ = RiskMetrics.maximum_drawdown(returns)
        return ann_return / mdd if mdd > 1e-10 else 0.0

    @staticmethod
    def omega_ratio(
        returns: np.ndarray, threshold: float = 0.0,
    ) -> float:
        """Ratio Omega = sum(max(r - seuil, 0)) / sum(max(seuil - r, 0))."""
        gains = np.sum(np.maximum(returns - threshold, 0))
        losses = np.sum(np.maximum(threshold - returns, 0))
        return gains / losses if losses > 1e-10 else float("inf")

    @staticmethod
    def tail_ratio(returns: np.ndarray, confidence: float = 0.95) -> float:
        """Ratio de queue = |percentile(95)| / |percentile(5)|."""
        upper = np.percentile(returns, confidence * 100)
        lower = np.percentile(returns, (1 - confidence) * 100)
        return abs(upper / lower) if abs(lower) > 1e-10 else float("inf")

    @staticmethod
    def annualized_return(
        returns: np.ndarray, periods_per_year: int = 12,
    ) -> float:
        """Rendement annualise geometrique."""
        cum = np.prod(1 + returns)
        n_years = len(returns) / periods_per_year
        if n_years <= 0:
            return 0.0
        return cum ** (1 / n_years) - 1

    @staticmethod
    def annualized_volatility(
        returns: np.ndarray, periods_per_year: int = 12,
    ) -> float:
        """Volatilite annualisee."""
        return np.std(returns) * np.sqrt(periods_per_year)

    @staticmethod
    def compute_all(
        returns: np.ndarray,
        rf: float = 0.025,
        benchmark: Optional[np.ndarray] = None,
        confidence: float = 0.95,
        periods_per_year: int = 12,
    ) -> Dict[str, float]:
        """Calcule toutes les metriques en une fois."""
        mdd, _, _ = RiskMetrics.maximum_drawdown(returns)

        metrics = {
            "Rendement annualise": RiskMetrics.annualized_return(returns, periods_per_year),
            "Volatilite annualisee": RiskMetrics.annualized_volatility(returns, periods_per_year),
            "Ratio de Sharpe": RiskMetrics.sharpe_ratio(returns, rf, True, periods_per_year),
            "Ratio de Sortino": RiskMetrics.sortino_ratio(returns, rf, 0.0, periods_per_year),
            "VaR (historique)": RiskMetrics.value_at_risk(returns, confidence, "historical"),
            "VaR (parametrique)": RiskMetrics.value_at_risk(returns, confidence, "parametric"),
            "VaR (Cornish-Fisher)": RiskMetrics.value_at_risk(returns, confidence, "cornish_fisher"),
            "CVaR": RiskMetrics.conditional_var(returns, confidence),
            "Perte maximale": mdd,
            "Ratio de Calmar": RiskMetrics.calmar_ratio(returns, periods_per_year),
            "Ratio Omega": RiskMetrics.omega_ratio(returns),
            "Ratio de queue": RiskMetrics.tail_ratio(returns, confidence),
            "Asymetrie": float(stats.skew(returns)),
            "Kurtosis": float(stats.kurtosis(returns)),
        }

        if benchmark is not None:
            metrics["Erreur de suivi"] = RiskMetrics.tracking_error(
                returns, benchmark, periods_per_year
            )
            metrics["Ratio d'information"] = RiskMetrics.information_ratio(
                returns, benchmark, periods_per_year
            )

        return metrics

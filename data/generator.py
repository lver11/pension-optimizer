"""
Generateur de donnees de marche simulees realistes.
Utilise des distributions multivariees avec copule Student-t et regime switching.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import AssetClass, ASSET_CLASSES_ORDER, ASSET_DEFAULTS, DEFAULT_CORRELATION_MATRIX


class MarketDataGenerator:
    """Genere des donnees de marche simulees mais realistes."""

    def __init__(
        self,
        expected_returns: Optional[np.ndarray] = None,
        volatilities: Optional[np.ndarray] = None,
        correlation_matrix: Optional[np.ndarray] = None,
        seed: int = 42,
    ):
        self.n_assets = len(ASSET_CLASSES_ORDER)
        self.mu = expected_returns if expected_returns is not None else np.array(
            [ASSET_DEFAULTS[ac].expected_return for ac in ASSET_CLASSES_ORDER]
        )
        self.vols = volatilities if volatilities is not None else np.array(
            [ASSET_DEFAULTS[ac].volatility for ac in ASSET_CLASSES_ORDER]
        )
        self.corr = correlation_matrix if correlation_matrix is not None else DEFAULT_CORRELATION_MATRIX
        self.rng = np.random.default_rng(seed)

    def _build_covariance(self, frequency: str = "monthly") -> np.ndarray:
        """Construit la matrice de covariance ajustee pour la frequence."""
        if frequency == "monthly":
            scale = 1 / 12
        elif frequency == "quarterly":
            scale = 1 / 4
        elif frequency == "daily":
            scale = 1 / 252
        else:
            scale = 1.0
        D = np.diag(self.vols * np.sqrt(scale))
        return D @ self.corr @ D

    def _get_scaled_returns(self, frequency: str = "monthly") -> np.ndarray:
        """Rendements attendus ajustes pour la frequence."""
        if frequency == "monthly":
            return self.mu / 12
        elif frequency == "quarterly":
            return self.mu / 4
        elif frequency == "daily":
            return self.mu / 252
        return self.mu

    def generate_returns(
        self,
        n_years: int = 20,
        frequency: str = "monthly",
        regime_switching: bool = True,
        fat_tails: bool = True,
        df_student: int = 5,
    ) -> pd.DataFrame:
        """
        Genere des rendements correles multi-actifs.

        Methode:
        1. Matrice de covariance: Sigma = D @ corr @ D
        2. Echantillonnage MVN ou Student-t
        3. Regime switching optionnel (Normal/Crise)
        """
        if frequency == "monthly":
            n_periods = n_years * 12
            freq_str = "ME"
        elif frequency == "quarterly":
            n_periods = n_years * 4
            freq_str = "QE"
        elif frequency == "daily":
            n_periods = n_years * 252
            freq_str = "B"
        else:
            n_periods = n_years
            freq_str = "YE"

        mu_freq = self._get_scaled_returns(frequency)
        cov_freq = self._build_covariance(frequency)

        if fat_tails:
            # Copule Student-t pour queues epaisses
            chi2_samples = self.rng.chisquare(df_student, size=n_periods)
            sqrt_chi2 = np.sqrt(df_student / chi2_samples)
            Z = self.rng.multivariate_normal(np.zeros(self.n_assets), cov_freq, size=n_periods)
            returns = mu_freq + Z * sqrt_chi2[:, np.newaxis]
        else:
            returns = self.rng.multivariate_normal(mu_freq, cov_freq, size=n_periods)

        if regime_switching:
            returns = self._apply_regime_switching(returns, frequency)

        start_date = datetime(2005, 1, 31)
        dates = pd.date_range(start=start_date, periods=n_periods, freq=freq_str)

        asset_names = [ASSET_DEFAULTS[ac].nom_fr for ac in ASSET_CLASSES_ORDER]
        df = pd.DataFrame(returns, index=dates, columns=asset_names)
        return df

    def _apply_regime_switching(
        self, returns: np.ndarray, frequency: str
    ) -> np.ndarray:
        """Applique un modele de regime switching a 2 etats (Normal/Crise)."""
        n_periods = returns.shape[0]
        if frequency == "monthly":
            p_normal_to_crisis = 0.03
            p_crisis_to_normal = 0.15
        elif frequency == "quarterly":
            p_normal_to_crisis = 0.08
            p_crisis_to_normal = 0.35
        else:
            p_normal_to_crisis = 0.03
            p_crisis_to_normal = 0.15

        regimes = np.zeros(n_periods, dtype=int)
        for t in range(1, n_periods):
            if regimes[t - 1] == 0:
                if self.rng.random() < p_normal_to_crisis:
                    regimes[t] = 1
            else:
                if self.rng.random() < p_crisis_to_normal:
                    regimes[t] = 0
                else:
                    regimes[t] = 1

        crisis_mask = regimes == 1
        if frequency == "monthly":
            vol_scale = 1 / np.sqrt(12)
        elif frequency == "quarterly":
            vol_scale = 1 / np.sqrt(4)
        else:
            vol_scale = 1 / np.sqrt(252)

        returns[crisis_mask] *= 2.0
        returns[crisis_mask] -= 0.5 * self.vols * vol_scale

        return returns

    def generate_yield_curve(self, n_years: int = 20) -> pd.DataFrame:
        """
        Genere une courbe de rendement canadienne simulee
        avec le modele Nelson-Siegel et des parametres stochastiques AR(1).
        """
        n_months = n_years * 12
        maturities = np.array([1, 2, 5, 10, 20, 30])

        beta0 = np.zeros(n_months)
        beta1 = np.zeros(n_months)
        beta2 = np.zeros(n_months)
        lam = 1.5

        beta0[0] = 0.04
        beta1[0] = -0.02
        beta2[0] = 0.01

        for t in range(1, n_months):
            beta0[t] = 0.04 + 0.98 * (beta0[t-1] - 0.04) + self.rng.normal(0, 0.002)
            beta1[t] = -0.02 + 0.95 * (beta1[t-1] + 0.02) + self.rng.normal(0, 0.003)
            beta2[t] = 0.01 + 0.90 * (beta2[t-1] - 0.01) + self.rng.normal(0, 0.004)

        yields_data = np.zeros((n_months, len(maturities)))
        for t in range(n_months):
            for j, tau in enumerate(maturities):
                x = tau / lam
                factor1 = (1 - np.exp(-x)) / x if x > 0 else 1.0
                factor2 = factor1 - np.exp(-x)
                yields_data[t, j] = beta0[t] + beta1[t] * factor1 + beta2[t] * factor2

        dates = pd.date_range(start="2005-01-31", periods=n_months, freq="ME")
        columns = [f"{m}Y" for m in maturities]
        return pd.DataFrame(yields_data, index=dates, columns=columns)

    def generate_liability_cashflows(
        self,
        n_years: int = 30,
        initial_liability: float = 950e6,
        growth_rate: float = 0.03,
    ) -> pd.DataFrame:
        """Genere les flux de tresorerie projetes du passif du regime de retraite."""
        years = np.arange(1, n_years + 1)
        annual_benefit = initial_liability * 0.06
        benefits = annual_benefit * (1 + growth_rate) ** years

        mortality_adjustment = 1 - 0.005 * years
        mortality_adjustment = np.maximum(mortality_adjustment, 0.5)
        benefits = benefits * mortality_adjustment

        contributions = initial_liability * 0.04 * (1 - 0.02 * years)
        contributions = np.maximum(contributions, 0)

        discount_rate = 0.05
        liability_pv = np.zeros(n_years)
        remaining_benefits = np.zeros(n_years)
        for t in range(n_years):
            future_benefits = benefits[t:]
            discount_factors = (1 + discount_rate) ** (-np.arange(1, len(future_benefits) + 1))
            liability_pv[t] = np.sum(future_benefits * discount_factors)
            remaining_benefits[t] = np.sum(future_benefits)

        df = pd.DataFrame({
            "Annee": years,
            "Prestations": benefits,
            "Cotisations": contributions,
            "Flux_net": contributions - benefits,
            "Valeur_passif_PV": liability_pv,
            "Prestations_restantes": remaining_benefits,
        })
        return df

    def generate_inflation_series(
        self, n_years: int = 20, target: float = 0.02, phi: float = 0.7
    ) -> pd.Series:
        """
        Processus AR(1) d'inflation centre sur la cible de la Banque du Canada.
        pi_t = target + phi*(pi_{t-1} - target) + epsilon_t
        """
        n_months = n_years * 12
        inflation = np.zeros(n_months)
        inflation[0] = target

        for t in range(1, n_months):
            epsilon = self.rng.normal(0, 0.002)
            inflation[t] = target + phi * (inflation[t-1] - target) + epsilon

        dates = pd.date_range(start="2005-01-31", periods=n_months, freq="ME")
        return pd.Series(inflation, index=dates, name="Inflation")

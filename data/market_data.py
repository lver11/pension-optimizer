"""
Utilitaires pour les donnees de marche.
"""

import numpy as np
import pandas as pd
from typing import Optional


def compute_rolling_correlation(
    returns: pd.DataFrame, window: int = 60,
) -> dict:
    """Calcule les correlations roulantes entre les classes d'actifs."""
    rolling_corr = returns.rolling(window=window).corr()
    return rolling_corr


def compute_rolling_volatility(
    returns: pd.DataFrame, window: int = 60,
    annualize: bool = True, periods_per_year: int = 12,
) -> pd.DataFrame:
    """Calcule la volatilite roulante annualisee."""
    vol = returns.rolling(window=window).std()
    if annualize:
        vol *= np.sqrt(periods_per_year)
    return vol

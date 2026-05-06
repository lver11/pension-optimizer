"""Tests pour compute_portfolio_metrics (risk/metrics.py)."""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from risk.metrics import compute_portfolio_metrics


def _simple_setup():
    """Retourne mu, cov, rf, weights simples pour les tests."""
    n = 3
    mu = np.array([0.08, 0.04, 0.02])
    cov = np.diag([0.16**2, 0.08**2, 0.01**2])  # Pas de corrélation
    rf = 0.025
    w = np.array([0.5, 0.3, 0.2])
    return mu, cov, rf, w


def test_rendement_correct():
    mu, cov, rf, w = _simple_setup()
    m = compute_portfolio_metrics(w, mu, cov, rf)
    expected = float(w @ mu)
    assert abs(m["rendement"] - expected) < 1e-10


def test_volatilite_correct():
    mu, cov, rf, w = _simple_setup()
    m = compute_portfolio_metrics(w, mu, cov, rf)
    expected = float(np.sqrt(w @ cov @ w))
    assert abs(m["volatilite"] - expected) < 1e-10


def test_sharpe_correct():
    mu, cov, rf, w = _simple_setup()
    m = compute_portfolio_metrics(w, mu, cov, rf)
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ cov @ w))
    expected = (ret - rf) / vol
    assert abs(m["sharpe"] - expected) < 1e-10


def test_returns_data_none_gaussian_fallback():
    """Sans returns_data, VaR et CVaR calculées via approximation gaussienne avec drift."""
    from scipy import stats as _stats
    mu, cov, rf, w = _simple_setup()
    m = compute_portfolio_metrics(w, mu, cov, rf, returns_data=None)
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ cov @ w))
    expected_var = -(ret + vol * _stats.norm.ppf(0.05))
    expected_cvar = -(ret - vol * _stats.norm.pdf(_stats.norm.ppf(0.05)) / 0.05)
    assert abs(m["var_95"] - expected_var) < 1e-8
    assert abs(m["cvar_95"] - expected_cvar) < 1e-8
    assert m["cvar_95"] >= m["var_95"]


def test_returns_data_historical():
    """Avec returns_data, VaR et CVaR sont calculées historiquement."""
    mu, cov, rf, w = _simple_setup()
    np.random.seed(42)
    returns_data = pd.DataFrame(
        np.random.randn(240, 3) * [0.16, 0.08, 0.01] / np.sqrt(12),
        columns=["A", "B", "C"],
    )
    m = compute_portfolio_metrics(w, mu, cov, rf, returns_data=returns_data)
    assert m["var_95"] > 0
    assert m["cvar_95"] >= m["var_95"]


def test_all_keys_present():
    mu, cov, rf, w = _simple_setup()
    m = compute_portfolio_metrics(w, mu, cov, rf)
    for key in ("rendement", "volatilite", "sharpe", "var_95", "cvar_95"):
        assert key in m, f"Clé manquante: {key}"


def test_zero_vol_sharpe():
    """Volatilité nulle → Sharpe = 0.0 (pas de division par zéro)."""
    mu = np.array([0.05, 0.05])
    cov = np.zeros((2, 2))  # Vol nulle
    rf = 0.025
    w = np.array([0.5, 0.5])
    m = compute_portfolio_metrics(w, mu, cov, rf)
    assert m["sharpe"] == 0.0


def test_returns_data_ndarray():
    mu, cov, rf, w = _simple_setup()
    np.random.seed(0)
    returns_data = np.random.randn(120, 3) * [0.16, 0.08, 0.01] / np.sqrt(12)
    m = compute_portfolio_metrics(w, mu, cov, rf, returns_data=returns_data)
    assert m["var_95"] > 0
    assert m["cvar_95"] >= m["var_95"]

# tests/test_acwi_asset_class.py
"""
Tests for the MSCI ACWI 14th asset class addition.
Run: pytest tests/test_acwi_asset_class.py -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from config import (
    get_asset_names_fr, get_expected_returns, get_volatilities,
    get_min_weights, get_max_weights,
    DEFAULT_CORRELATION_MATRIX, DEFAULT_CURRENT_WEIGHTS,
    ASSET_CLASSES_ORDER, ASSET_DEFAULTS, BENCHMARK_PORTFOLIOS,
    CHART_COLORS, ALPHA_ELIGIBLE_SHORT, AssetClass,
)
from constraints.regulatory import QuebecPensionRegulations, PortableAlphaRegulations


N = 14  # expected number of asset classes after adding ACWI


class TestAssetCount:
    def test_asset_names_has_14_entries(self):
        assert len(get_asset_names_fr()) == N

    def test_asset_classes_order_has_14_entries(self):
        assert len(ASSET_CLASSES_ORDER) == N

    def test_acwi_enum_exists(self):
        assert hasattr(AssetClass, "ACTIONS_ACWI")

    def test_acwi_is_last_in_order(self):
        assert ASSET_CLASSES_ORDER[-1] == AssetClass.ACTIONS_ACWI

    def test_acwi_name_fr(self):
        names = get_asset_names_fr()
        assert names[13] == "Actions MSCI ACWI"


class TestACWIParams:
    def test_expected_return(self):
        mu = get_expected_returns()
        assert len(mu) == N
        assert abs(mu[13] - 0.078) < 1e-9

    def test_volatility(self):
        vols = get_volatilities()
        assert len(vols) == N
        assert abs(vols[13] - 0.16) < 1e-9

    def test_min_weight_zero(self):
        assert get_min_weights()[13] == 0.0

    def test_max_weight_40pct(self):
        assert abs(get_max_weights()[13] - 0.40) < 1e-9


class TestCorrelationMatrix:
    def test_shape_is_14x14(self):
        assert DEFAULT_CORRELATION_MATRIX.shape == (N, N)

    def test_symmetric(self):
        assert np.allclose(DEFAULT_CORRELATION_MATRIX, DEFAULT_CORRELATION_MATRIX.T)

    def test_diagonal_ones(self):
        assert np.allclose(np.diag(DEFAULT_CORRELATION_MATRIX), 1.0)

    def test_acwi_us_correlation(self):
        """ACWI is ~65% US, so corr(ACWI, US) should be ~0.95."""
        assert abs(DEFAULT_CORRELATION_MATRIX[13, 1] - 0.95) < 1e-9
        assert abs(DEFAULT_CORRELATION_MATRIX[1, 13] - 0.95) < 1e-9

    def test_acwi_eafe_correlation(self):
        assert abs(DEFAULT_CORRELATION_MATRIX[13, 2] - 0.90) < 1e-9

    def test_acwi_em_correlation(self):
        assert abs(DEFAULT_CORRELATION_MATRIX[13, 3] - 0.82) < 1e-9


class TestWeights:
    def test_current_weights_length(self):
        assert len(DEFAULT_CURRENT_WEIGHTS) == N

    def test_current_weights_sum_to_1(self):
        assert abs(DEFAULT_CURRENT_WEIGHTS.sum() - 1.0) < 1e-9

    def test_acwi_default_weight_zero(self):
        assert DEFAULT_CURRENT_WEIGHTS[13] == 0.0

    @pytest.mark.parametrize("key,expected_len", [
        ("60_40_equilibre", N),
        ("obligations_pures", N),
        ("croissance_70_30", N),
        ("politique_placement", N),
    ])
    def test_benchmark_weight_lengths(self, key, expected_len):
        w = BENCHMARK_PORTFOLIOS[key]["weights"]
        assert len(w) == expected_len, f"{key}: expected {expected_len} weights, got {len(w)}"

    @pytest.mark.parametrize("key", [
        "60_40_equilibre", "obligations_pures", "croissance_70_30", "politique_placement",
    ])
    def test_benchmark_weights_sum_to_1(self, key):
        w = BENCHMARK_PORTFOLIOS[key]["weights"]
        assert abs(w.sum() - 1.0) < 1e-9, f"{key}: weights sum to {w.sum()}"


class TestChartColors:
    def test_chart_colors_has_14_entries(self):
        assert len(CHART_COLORS) == N


class TestAlphaEligible:
    def test_acwi_in_alpha_eligible_short(self):
        assert 13 in ALPHA_ELIGIBLE_SHORT


class TestRegulatoryIndices:
    def test_acwi_in_equity_indices(self):
        assert 13 in QuebecPensionRegulations.EQUITY_INDICES

    def test_acwi_in_foreign_indices(self):
        assert 13 in QuebecPensionRegulations.FOREIGN_INDICES

    def test_acwi_in_short_eligible(self):
        assert 13 in PortableAlphaRegulations.SHORT_ELIGIBLE_INDICES

    def test_cash_index_unchanged(self):
        """Encaisse remains at index 12."""
        assert QuebecPensionRegulations.CASH_INDEX == [12]

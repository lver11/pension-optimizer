# tests/test_emd_asset_class.py
"""
Tests for the Dette pays emergents asset class (index 14).
Run: pytest tests/test_emd_asset_class.py -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from config import (
    get_asset_names_fr, get_expected_returns, get_volatilities,
    get_min_weights, get_max_weights, get_asset_durations,
    DEFAULT_CORRELATION_MATRIX, DEFAULT_CURRENT_WEIGHTS,
    ASSET_CLASSES_ORDER, BENCHMARK_PORTFOLIOS,
    CHART_COLORS, ALPHA_ELIGIBLE_SHORT, AssetClass,
)
from constraints.regulatory import QuebecPensionRegulations, PortableAlphaRegulations


N = 17  # current number of asset classes


class TestAssetCount:
    def test_asset_names_has_N_entries(self):
        assert len(get_asset_names_fr()) == N

    def test_asset_classes_order_has_N_entries(self):
        assert len(ASSET_CLASSES_ORDER) == N

    def test_emd_enum_exists(self):
        assert hasattr(AssetClass, "DETTE_EMERGENTE")

    def test_emd_is_at_index_14(self):
        assert ASSET_CLASSES_ORDER[14] == AssetClass.DETTE_EMERGENTE

    def test_emd_name_fr(self):
        names = get_asset_names_fr()
        assert names[14] == "Dette pays emergents"


class TestEMDParams:
    def test_expected_return(self):
        mu = get_expected_returns()
        assert len(mu) == N
        assert abs(mu[14] - 0.058) < 1e-9

    def test_volatility(self):
        vols = get_volatilities()
        assert len(vols) == N
        assert abs(vols[14] - 0.11) < 1e-9

    def test_duration(self):
        durations = get_asset_durations()
        assert len(durations) == N
        assert abs(durations[14] - 7.0) < 1e-9

    def test_min_weight_zero(self):
        assert get_min_weights()[14] == 0.0

    def test_max_weight_15pct(self):
        assert abs(get_max_weights()[14] - 0.15) < 1e-9


class TestCorrelationMatrix:
    def test_shape_is_NxN(self):
        assert DEFAULT_CORRELATION_MATRIX.shape == (N, N)

    def test_symmetric(self):
        assert np.allclose(DEFAULT_CORRELATION_MATRIX, DEFAULT_CORRELATION_MATRIX.T)

    def test_diagonal_ones(self):
        assert np.allclose(np.diag(DEFAULT_CORRELATION_MATRIX), 1.0)

    def test_positive_semidefinite(self):
        """Matrix must be PSD for portfolio optimization to work correctly."""
        eigvals = np.linalg.eigvalsh(DEFAULT_CORRELATION_MATRIX)
        assert eigvals.min() >= -1e-10, f"Matrix not PSD: min eigenvalue = {eigvals.min():.6f}"

    def test_emd_em_equity_correlation(self):
        """EMD vs EM equities: shared EM factor (moderated for PSD)."""
        assert abs(DEFAULT_CORRELATION_MATRIX[14, 3] - 0.45) < 1e-9
        assert abs(DEFAULT_CORRELATION_MATRIX[3, 14] - 0.45) < 1e-9

    def test_emd_corp_bond_correlation(self):
        assert abs(DEFAULT_CORRELATION_MATRIX[14, 5] - 0.35) < 1e-9

    def test_emd_acwi_correlation(self):
        assert abs(DEFAULT_CORRELATION_MATRIX[14, 13] - 0.30) < 1e-9


class TestWeights:
    def test_current_weights_length(self):
        assert len(DEFAULT_CURRENT_WEIGHTS) == N

    def test_current_weights_sum_to_1(self):
        assert abs(DEFAULT_CURRENT_WEIGHTS.sum() - 1.0) < 1e-9

    def test_emd_default_weight_zero(self):
        assert DEFAULT_CURRENT_WEIGHTS[14] == 0.0

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
    def test_chart_colors_has_N_entries(self):
        assert len(CHART_COLORS) == N


class TestAlphaEligible:
    def test_emd_in_alpha_eligible_short(self):
        assert 14 in ALPHA_ELIGIBLE_SHORT


class TestRegulatoryIndices:
    def test_emd_in_bond_indices(self):
        assert 14 in QuebecPensionRegulations.BOND_INDICES

    def test_emd_in_foreign_indices(self):
        assert 14 in QuebecPensionRegulations.FOREIGN_INDICES

    def test_emd_in_short_eligible(self):
        assert 14 in PortableAlphaRegulations.SHORT_ELIGIBLE_INDICES

    def test_emd_not_in_equity_indices(self):
        """EMD is a bond instrument, not equity."""
        assert 14 not in QuebecPensionRegulations.EQUITY_INDICES

    def test_cash_index_unchanged(self):
        assert QuebecPensionRegulations.CASH_INDEX == [12]

    def test_acwi_still_in_equity_indices(self):
        """Adding EMD must not disturb ACWI's equity classification."""
        assert 13 in QuebecPensionRegulations.EQUITY_INDICES

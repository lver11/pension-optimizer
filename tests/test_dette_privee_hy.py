# tests/test_dette_privee_hy.py
"""
Tests for Dette privee (index 15) and Obligations HY (index 16).
Run: pytest tests/test_dette_privee_hy.py -v
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
    CHART_COLORS, ALPHA_ELIGIBLE_SHORT, AssetClass, ASSET_DEFAULTS,
)
from constraints.regulatory import QuebecPensionRegulations, PortableAlphaRegulations


N = 17  # expected number of asset classes after adding DP + HY


class TestAssetCount:
    def test_asset_names_has_17_entries(self):
        assert len(get_asset_names_fr()) == N

    def test_asset_classes_order_has_17_entries(self):
        assert len(ASSET_CLASSES_ORDER) == N

    def test_dette_privee_enum_exists(self):
        assert hasattr(AssetClass, "DETTE_PRIVEE")

    def test_obligations_hy_enum_exists(self):
        assert hasattr(AssetClass, "OBLIGATIONS_HY")

    def test_dette_privee_is_second_to_last(self):
        assert ASSET_CLASSES_ORDER[-2] == AssetClass.DETTE_PRIVEE

    def test_obligations_hy_is_last(self):
        assert ASSET_CLASSES_ORDER[-1] == AssetClass.OBLIGATIONS_HY

    def test_dette_privee_name_fr(self):
        names = get_asset_names_fr()
        assert names[15] == "Dette privee"

    def test_obligations_hy_name_fr(self):
        names = get_asset_names_fr()
        assert names[16] == "Obligations HY"


class TestDettePriveeParams:
    def test_expected_return(self):
        mu = get_expected_returns()
        assert len(mu) == N
        assert abs(mu[15] - 0.075) < 1e-9

    def test_volatility(self):
        vols = get_volatilities()
        assert len(vols) == N
        assert abs(vols[15] - 0.06) < 1e-9

    def test_duration(self):
        durations = get_asset_durations()
        assert len(durations) == N
        assert abs(durations[15] - 5.0) < 1e-9

    def test_min_weight_zero(self):
        assert get_min_weights()[15] == 0.0

    def test_max_weight_10pct(self):
        assert abs(get_max_weights()[15] - 0.10) < 1e-9

    def test_is_alternative_true(self):
        assert ASSET_DEFAULTS[AssetClass.DETTE_PRIVEE].is_alternative is True

    def test_liquidity_score(self):
        assert abs(ASSET_DEFAULTS[AssetClass.DETTE_PRIVEE].liquidity_score - 0.10) < 1e-9

    def test_esg_score(self):
        assert abs(ASSET_DEFAULTS[AssetClass.DETTE_PRIVEE].esg_score - 50.0) < 1e-9


class TestObligationsHYParams:
    def test_expected_return(self):
        mu = get_expected_returns()
        assert len(mu) == N
        assert abs(mu[16] - 0.055) < 1e-9

    def test_volatility(self):
        vols = get_volatilities()
        assert len(vols) == N
        assert abs(vols[16] - 0.09) < 1e-9

    def test_duration(self):
        durations = get_asset_durations()
        assert len(durations) == N
        assert abs(durations[16] - 4.5) < 1e-9

    def test_min_weight_zero(self):
        assert get_min_weights()[16] == 0.0

    def test_max_weight_15pct(self):
        assert abs(get_max_weights()[16] - 0.15) < 1e-9

    def test_is_alternative_false(self):
        assert ASSET_DEFAULTS[AssetClass.OBLIGATIONS_HY].is_alternative is False

    def test_liquidity_score(self):
        assert abs(ASSET_DEFAULTS[AssetClass.OBLIGATIONS_HY].liquidity_score - 0.80) < 1e-9

    def test_esg_score(self):
        assert abs(ASSET_DEFAULTS[AssetClass.OBLIGATIONS_HY].esg_score - 48.0) < 1e-9


class TestCorrelationMatrix:
    def test_shape_is_17x17(self):
        assert DEFAULT_CORRELATION_MATRIX.shape == (N, N)

    def test_symmetric(self):
        assert np.allclose(DEFAULT_CORRELATION_MATRIX, DEFAULT_CORRELATION_MATRIX.T)

    def test_diagonal_ones(self):
        assert np.allclose(np.diag(DEFAULT_CORRELATION_MATRIX), 1.0)

    def test_positive_semidefinite(self):
        """Matrix must be PSD for portfolio optimization to work correctly."""
        eigvals = np.linalg.eigvalsh(DEFAULT_CORRELATION_MATRIX)
        assert eigvals.min() >= -1e-10, (
            f"Matrix not PSD: min eigenvalue = {eigvals.min():.6f}"
        )

    def test_dette_privee_pe_correlation(self):
        """DP vs Capital investissement (9): same private credit universe."""
        assert abs(DEFAULT_CORRELATION_MATRIX[15, 9] - 0.45) < 1e-9
        assert abs(DEFAULT_CORRELATION_MATRIX[9, 15] - 0.45) < 1e-9

    def test_dette_privee_corp_bond_correlation(self):
        assert abs(DEFAULT_CORRELATION_MATRIX[15, 5] - 0.35) < 1e-9

    def test_dette_privee_emd_correlation(self):
        assert abs(DEFAULT_CORRELATION_MATRIX[15, 14] - 0.25) < 1e-9

    def test_hy_us_correlation(self):
        """HY vs Actions US (1): dominant HY market."""
        assert abs(DEFAULT_CORRELATION_MATRIX[16, 1] - 0.50) < 1e-9
        assert abs(DEFAULT_CORRELATION_MATRIX[1, 16] - 0.50) < 1e-9

    def test_hy_corp_bond_correlation(self):
        assert abs(DEFAULT_CORRELATION_MATRIX[16, 5] - 0.60) < 1e-9

    def test_hy_emd_correlation(self):
        assert abs(DEFAULT_CORRELATION_MATRIX[16, 14] - 0.55) < 1e-9

    def test_hy_dette_privee_correlation(self):
        assert abs(DEFAULT_CORRELATION_MATRIX[16, 15] - 0.45) < 1e-9
        assert abs(DEFAULT_CORRELATION_MATRIX[15, 16] - 0.45) < 1e-9

    def test_existing_correlations_unchanged(self):
        """Adding DP+HY must not disturb the existing 15x15 block."""
        # US-CDN
        assert abs(DEFAULT_CORRELATION_MATRIX[0, 1] - 0.75) < 1e-9
        # ACWI-US (moderated for PSD)
        assert abs(DEFAULT_CORRELATION_MATRIX[13, 1] - 0.88) < 1e-9
        # EMD-EM equity (moderated for PSD)
        assert abs(DEFAULT_CORRELATION_MATRIX[14, 3] - 0.45) < 1e-9


class TestWeights:
    def test_current_weights_length(self):
        assert len(DEFAULT_CURRENT_WEIGHTS) == N

    def test_current_weights_sum_to_1(self):
        assert abs(DEFAULT_CURRENT_WEIGHTS.sum() - 1.0) < 1e-9

    def test_dette_privee_default_weight_zero(self):
        assert DEFAULT_CURRENT_WEIGHTS[15] == 0.0

    def test_obligations_hy_default_weight_zero(self):
        assert DEFAULT_CURRENT_WEIGHTS[16] == 0.0

    @pytest.mark.parametrize("key,expected_len", [
        ("60_40_equilibre", N),
        ("obligations_pures", N),
        ("croissance_70_30", N),
        ("politique_placement", N),
    ])
    def test_benchmark_weight_lengths(self, key, expected_len):
        w = BENCHMARK_PORTFOLIOS[key]["weights"]
        assert len(w) == expected_len, (
            f"{key}: expected {expected_len} weights, got {len(w)}"
        )

    @pytest.mark.parametrize("key", [
        "60_40_equilibre", "obligations_pures", "croissance_70_30", "politique_placement",
    ])
    def test_benchmark_weights_sum_to_1(self, key):
        w = BENCHMARK_PORTFOLIOS[key]["weights"]
        assert abs(w.sum() - 1.0) < 1e-9, f"{key}: weights sum to {w.sum()}"


class TestChartColors:
    def test_chart_colors_has_17_entries(self):
        assert len(CHART_COLORS) == N


class TestAlphaEligible:
    def test_hy_in_alpha_eligible_short(self):
        """Obligations HY (index 16): liquid ETF, eligible for short."""
        assert 16 in ALPHA_ELIGIBLE_SHORT

    def test_dette_privee_not_in_alpha_eligible_short(self):
        """Dette privee (index 15): illiquid private market, not eligible."""
        assert 15 not in ALPHA_ELIGIBLE_SHORT


class TestRegulatoryIndices:
    def test_dette_privee_in_alternative_indices(self):
        assert 15 in QuebecPensionRegulations.ALTERNATIVE_INDICES

    def test_obligations_hy_in_bond_indices(self):
        assert 16 in QuebecPensionRegulations.BOND_INDICES

    def test_dette_privee_in_foreign_indices(self):
        assert 15 in QuebecPensionRegulations.FOREIGN_INDICES

    def test_obligations_hy_in_foreign_indices(self):
        assert 16 in QuebecPensionRegulations.FOREIGN_INDICES

    def test_obligations_hy_in_short_eligible(self):
        assert 16 in PortableAlphaRegulations.SHORT_ELIGIBLE_INDICES

    def test_dette_privee_not_in_short_eligible(self):
        """DP is illiquid - short selling prohibited."""
        assert 15 not in PortableAlphaRegulations.SHORT_ELIGIBLE_INDICES

    def test_dette_privee_not_in_equity_indices(self):
        assert 15 not in QuebecPensionRegulations.EQUITY_INDICES

    def test_obligations_hy_not_in_equity_indices(self):
        assert 16 not in QuebecPensionRegulations.EQUITY_INDICES

    def test_cash_index_unchanged(self):
        assert QuebecPensionRegulations.CASH_INDEX == [12]

    def test_emd_still_in_bond_indices(self):
        """Adding HY must not disturb EMD's bond classification."""
        assert 14 in QuebecPensionRegulations.BOND_INDICES

    def test_acwi_still_in_equity_indices(self):
        """Adding DP+HY must not disturb ACWI's equity classification."""
        assert 13 in QuebecPensionRegulations.EQUITY_INDICES

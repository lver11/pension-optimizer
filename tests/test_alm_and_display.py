"""Tests : diagnostic ALM, ratio de couverture, nettoyage des poids."""

import numpy as np
import pytest

from config import (
    ASSET_CLASSES_ORDER, AssetClass, DEFAULT_CURRENT_WEIGHTS,
    get_expected_returns, get_covariance_matrix, get_asset_durations,
    get_min_weights, get_max_weights, get_asset_names_fr,
)
from models.alm import ALMOptimizer, LiabilityProfile
from models.base import clean_weights
from models.mean_variance import MeanVarianceOptimizer

N = len(ASSET_CLASSES_ORDER)
GOV = ASSET_CLASSES_ORDER.index(AssetClass.OBLIGATIONS_GOV_CDN)


def make_alm(pv=1e9, duration=15.0):
    return ALMOptimizer(
        get_expected_returns(), get_covariance_matrix(), get_asset_durations(),
        LiabilityProfile(present_value=pv, duration=duration), 0.025,
        get_asset_names_fr(), get_min_weights(), get_max_weights(),
        reference_bond_index=GOV,
    )


class TestALMStatus:
    def test_full_funding_with_large_duration_gap_is_not_adequate(self):
        diag = make_alm().assess_status(DEFAULT_CURRENT_WEIGHTS, 1e9)
        assert diag["composante_capitalisation"]["niveau"] == "adequat"
        assert diag["composante_taux"]["niveau"] == "eleve"
        assert diag["statut"] != "adequat"
        assert any("couverture" in a for a in diag["actions"])

    def test_matched_duration_gives_low_rate_risk(self):
        alm = make_alm(duration=7.5)  # passif aussi long que les obligations gouvernementales
        w = np.zeros(N)
        w[GOV] = 1.0
        diag = alm.assess_status(w, 1e9)
        assert diag["composante_taux"]["niveau"] == "faible"
        assert abs(diag["ratio_couverture"] - 1.0) < 1e-9
        assert diag["statut"] == "adequat"

    def test_deficit_is_flagged(self):
        diag = make_alm(pv=1.3e9).assess_status(DEFAULT_CURRENT_WEIGHTS, 1e9)
        assert diag["statut"] == "critique"


class TestHedgeRatio:
    def test_hedge_ratio_formula(self):
        alm = make_alm()
        w = np.zeros(N)
        w[GOV] = 1.0
        # A x D_A / (L x D_L) = 1e9 x 7.5 / (1e9 x 15) = 50 %
        assert abs(alm.compute_hedge_ratio(w, 1e9) - 0.5) < 1e-12

    def test_required_weight_accounts_for_funding_and_duration(self):
        alm = make_alm(pv=1e9, duration=15.0)
        plan = alm.optimize_liability_hedge(2e9, 0.80, hedge_duration=15.0)
        # 0,8 x 1e9 x 15 / (2e9 x 15) = 40 %
        assert abs(plan["poids_actifs_couverture_recommande"] - 0.40) < 1e-12
        assert plan["faisable_sans_levier"]

    def test_infeasible_without_leverage_is_reported(self):
        plan = make_alm().optimize_liability_hedge(1e9, 0.80, hedge_duration=10.0)
        assert plan["poids_actifs_couverture_recommande"] > 1.0
        assert not plan["faisable_sans_levier"]
        assert "levier" in plan["recommandation"]

    def test_surplus_optimization_reduces_surplus_volatility_vs_asset_only(self):
        alm = make_alm()
        res = alm.optimize_surplus(1e9)
        assert res.status == "optimal"
        proxy = alm.liability_proxy_weights(1e9)
        cov = get_covariance_matrix()
        current = DEFAULT_CURRENT_WEIGHTS
        sv_current = np.sqrt((current - proxy) @ cov @ (current - proxy))
        assert res.metadata["surplus_volatility"] <= sv_current + 1e-9


class TestCleanWeights:
    def test_tiny_weights_become_zero_and_sum_is_one(self):
        w = np.array([0.5, 3e-8, 0.4999999700, 0.0])
        out = clean_weights(w)
        assert out[1] == 0.0
        assert abs(out.sum() - 1.0) < 1e-12

    def test_respects_max_bounds(self):
        w = np.array([0.30, 0.69997, 0.00003])
        out = clean_weights(w, max_weights=np.array([0.30, 1.0, 1.0]))
        assert out[0] <= 0.30 + 1e-12 and out[2] == 0.0

    def test_short_positions_untouched(self):
        w = np.array([1.2, -0.2])
        assert np.array_equal(clean_weights(w), w)

    @pytest.mark.parametrize("objective", ["max_sharpe", "min_variance", "target_return"])
    def test_optimizer_output_has_no_residuals(self, objective):
        mn, mx = get_min_weights(), get_max_weights()
        res = MeanVarianceOptimizer(get_expected_returns(), get_covariance_matrix(), 0.025,
                                    get_asset_names_fr(), mn, mx).optimize(objective, target_return=0.06)
        w = res.weights
        assert not np.any((w > 0) & (w < 5e-5))
        assert abs(w.sum() - 1.0) < 1e-9
        assert np.all(w >= mn - 1e-9) and np.all(w <= mx + 1e-9)

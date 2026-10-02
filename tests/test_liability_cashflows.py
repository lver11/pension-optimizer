"""Tests du passif par flux : courbe, valeur actualisee, convexite, durations cles."""

import numpy as np
import pytest

from config import (
    ASSET_CLASSES_ORDER, AssetClass, DEFAULT_CURRENT_WEIGHTS,
    get_expected_returns, get_covariance_matrix, get_asset_durations,
    get_asset_names_fr, get_min_weights, get_max_weights,
)
from fund_profile import FundProfile, PROFILS_TYPES
from models.alm import ALMOptimizer, LiabilityProfile
from models.liability import (
    YieldCurve, LiabilityCashflows, KEY_TENORS, example_cashflows,
    approx_convexity, estimated_liability_convexity, bond_key_rate_profile,
)

GOV = ASSET_CLASSES_ORDER.index(AssetClass.OBLIGATIONS_GOV_CDN)


@pytest.fixture
def flat():
    return YieldCurve.flat(0.05)


class TestCurveAndPricing:
    def test_zero_coupon_price_and_duration(self, flat):
        zc = LiabilityCashflows(np.array([10.0]), np.array([100.0]))
        assert zc.present_value(flat) == pytest.approx(100 / 1.05 ** 10)
        assert zc.effective_duration(flat) == pytest.approx(10 / 1.05, rel=1e-4)
        assert zc.effective_convexity(flat) == pytest.approx(approx_convexity(10, 0.05), rel=1e-4)

    def test_curve_interpolation_and_flat_extrapolation(self):
        c = YieldCurve.from_dict({"2": 0.03, "10": 0.04})
        assert c.rate(6) == pytest.approx(0.035)
        assert c.rate(1) == pytest.approx(0.03) and c.rate(40) == pytest.approx(0.04)

    def test_example_cashflows_calibrated(self, flat):
        cf = LiabilityCashflows.from_records(example_cashflows(1e9, 15, 0.05))
        assert cf.present_value(flat) == pytest.approx(1e9, rel=1e-9)
        assert cf.effective_duration(flat) == pytest.approx(15, abs=1e-3)

    def test_key_rate_durations_sum_to_duration(self, flat):
        cf = LiabilityCashflows.from_records(example_cashflows(1e9, 15, 0.05))
        krd = cf.key_rate_durations(flat)
        assert set(krd) == set(KEY_TENORS)
        assert sum(krd.values()) == pytest.approx(cf.effective_duration(flat), rel=1e-3)

    def test_spread_out_liability_more_convex_than_zero_coupon(self, flat):
        assert estimated_liability_convexity(15, 0.05) > approx_convexity(15, 0.05)

    def test_bond_key_rate_profile_preserves_duration(self):
        for d in (0.25, 3.0, 7.5, 12.0, 35.0):
            assert sum(bond_key_rate_profile(d).values()) == pytest.approx(d)


class TestConvexityInALM:
    def make(self, cashflows=None, curve=None):
        return ALMOptimizer(
            get_expected_returns(), get_covariance_matrix(), get_asset_durations(),
            LiabilityProfile(present_value=1e9, duration=15.0,
                             convexity=estimated_liability_convexity(15, 0.05)),
            0.025, get_asset_names_fr(), get_min_weights(), get_max_weights(),
            reference_bond_index=GOV, liability_cashflows=cashflows, curve=curve,
        )

    def test_exact_revaluation_exceeds_duration_only_when_rates_fall(self, flat):
        cf = LiabilityCashflows.from_records(example_cashflows(1e9, 15, 0.05))
        sens = self.make(cf, flat).compute_interest_rate_sensitivity(DEFAULT_CURRENT_WEIGHTS, 1e9, -100)
        exact = cf.present_value(flat, -0.01) - 1e9
        assert sens["impact_passif"] == pytest.approx(exact, rel=1e-9)
        assert sens["impact_passif"] > sens["impact_passif_duration_seule"] > 0
        assert sens["methode_passif"] == "reevaluation des flux"

    def test_convexity_fallback_close_to_exact(self, flat):
        cf = LiabilityCashflows.from_records(example_cashflows(1e9, 15, 0.05))
        exact = self.make(cf, flat).compute_interest_rate_sensitivity(DEFAULT_CURRENT_WEIGHTS, 1e9, -100)
        approx = self.make().compute_interest_rate_sensitivity(DEFAULT_CURRENT_WEIGHTS, 1e9, -100)
        assert approx["impact_passif"] == pytest.approx(exact["impact_passif"], rel=0.02)

    def test_key_rate_exposures_sum_to_total_dv01(self, flat):
        cf = LiabilityCashflows.from_records(example_cashflows(1e9, 15, 0.05))
        alm = self.make(cf, flat)
        krd = alm.key_rate_exposures(DEFAULT_CURRENT_WEIGHTS, 1e9)
        liab_total = sum(v["passif_dv01"] for v in krd.values())
        asset_total = sum(v["actif_dv01"] for v in krd.values())
        assert liab_total == pytest.approx(1e9 * cf.effective_duration(flat) * 1e-4, rel=1e-3)
        assert asset_total == pytest.approx(1e9 * (DEFAULT_CURRENT_WEIGHTS @ get_asset_durations()) * 1e-4)


class TestProfileWithCashflows:
    def test_profile_syncs_value_and_duration_from_flows(self):
        p = PROFILS_TYPES["pd_generique"]()
        p.flux_passif = example_cashflows(1.2e9, 12, 0.05)
        p.sync_liability_from_cashflows()
        assert p.valeur_passif == pytest.approx(1.2e9, rel=1e-6)
        assert p.duration_passif == pytest.approx(12, abs=1e-3)
        assert p.validate()[0] == []
        assert FundProfile.from_json(p.to_json()).liability_measures()["source"] == "flux"

    def test_curve_changes_value(self):
        p = PROFILS_TYPES["pd_generique"]()
        p.flux_passif = example_cashflows(1e9, 15, 0.05)
        v_flat = p.liability_measures()["valeur"]
        p.courbe_actualisation = {"2": 0.03, "10": 0.035, "30": 0.04}
        assert p.liability_measures()["valeur"] > v_flat

    @pytest.mark.parametrize("bad", [
        [{"annee": 0, "nominal": 10}],
        [{"annee": 1, "nominal": -5}],
        [{"annee": 1, "nominal": 5}, {"annee": 1, "nominal": 5}],
        [{"year": 1, "nominal": 5}],
    ])
    def test_invalid_flows_rejected(self, bad):
        p = PROFILS_TYPES["pd_generique"]()
        p.flux_passif = bad
        assert p.validate()[0]

    def test_without_flows_convexity_is_estimated(self):
        m = PROFILS_TYPES["pd_generique"]().liability_measures()
        assert m["source"] == "valeur_duration" and m["convexite_estimee"]
        assert m["convexite"] == pytest.approx(estimated_liability_convexity(15, 0.05))

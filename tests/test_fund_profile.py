"""Tests du profil de fonds (application generique multi-fonds)."""

import json

import numpy as np
import pytest

from config import (
    ASSET_CLASSES_ORDER, AssetClass, DEFAULT_CURRENT_WEIGHTS,
    get_category_indices, get_expected_returns, get_covariance_matrix,
    get_asset_names_fr, get_min_weights, get_max_weights, get_policy_weights,
    get_benchmark_portfolios,
)
from constraints.manager import ConstraintSet, ConstraintManager
from constraints.regulatory import PolicyLimits, QuebecPensionRegulations
from fund_profile import (
    FundProfile, GroupLimit, PROFILS_TYPES, PROFIL_PAR_DEFAUT, TYPES_FONDS,
)
from models.mean_variance import MeanVarianceOptimizer
from models.monte_carlo import MonteCarloSimulator

N = len(ASSET_CLASSES_ORDER)
ACWI = ASSET_CLASSES_ORDER.index(AssetClass.ACTIONS_ACWI)


@pytest.fixture(params=list(PROFILS_TYPES))
def preset(request):
    return PROFILS_TYPES[request.param]()


class TestPresets:
    def test_preset_is_valid(self, preset):
        errors, warnings = preset.validate()
        assert errors == []
        assert warnings == [], warnings

    def test_preset_weights_sum_to_one(self, preset):
        assert abs(preset.weights_array().sum() - 1.0) < 1e-9

    def test_preset_type_known(self, preset):
        assert preset.type_fonds in TYPES_FONDS

    def test_preset_is_optimizable(self, preset):
        cs = ConstraintSet(min_weights=preset.min_array(), max_weights=preset.max_array(),
                           group_constraints=preset.group_constraints())
        opt = MeanVarianceOptimizer(get_expected_returns(), get_covariance_matrix(),
                                    preset.taux_sans_risque, get_asset_names_fr(),
                                    preset.min_array(), preset.max_array())
        result = opt.optimize("max_sharpe", constraint_set=cs)
        assert result.status == "optimal"
        ok, violations = ConstraintManager(N, get_asset_names_fr()).validate_allocation(result.weights, cs)
        assert ok, violations

    def test_json_round_trip(self, preset):
        assert FundProfile.from_json(preset.to_json()) == preset

    def test_default_profile_is_generic(self):
        p = PROFILS_TYPES[PROFIL_PAR_DEFAUT]()
        assert "fondaction" not in p.nom.lower()
        assert np.allclose(p.weights_array(), DEFAULT_CURRENT_WEIGHTS)


class TestValidation:
    def base(self, **kw):
        d = dict(nom="Test", poids_politique={"actions_acwi": 0.6, "obligations_corporatives": 0.4},
                 bornes_min={c.value: 0.0 for c in ASSET_CLASSES_ORDER})
        d.update(kw)
        return FundProfile(**d)

    def test_valid_minimal_profile(self):
        assert self.base().validate()[0] == []

    def test_weights_must_sum_to_one(self):
        errors, _ = self.base(poids_politique={"actions_acwi": 0.5}).validate()
        assert any("totalisent" in e for e in errors)

    def test_unknown_asset_code(self):
        errors, _ = self.base(poids_politique={"bitcoin": 1.0}).validate()
        assert any("inconnues" in e for e in errors)

    def test_min_greater_than_max(self):
        errors, _ = self.base(bornes_min={"actions_acwi": 0.5}, bornes_max={"actions_acwi": 0.4}).validate()
        assert any("min > max" in e for e in errors)

    def test_infeasible_min_sum(self):
        errors, _ = self.base(bornes_min={c.value: 0.1 for c in ASSET_CLASSES_ORDER}).validate()
        assert any("bornes minimales" in e for e in errors)

    def test_policy_outside_group_limit_is_warning(self):
        p = self.base(limites_groupes=[GroupLimit("Actions", ["actions_acwi"], 0.0, 0.5)])
        errors, warnings = p.validate()
        assert errors == [] and any("Actions" in w for w in warnings)

    def test_unknown_json_field_rejected(self):
        d = self.base().to_dict()
        d["champ_mystere"] = 1
        with pytest.raises(ValueError):
            FundProfile.from_json(json.dumps(d))

    def test_no_liability_profile(self):
        p = self.base(valeur_passif=None)
        assert not p.a_un_passif and p.ratio_capitalisation is None


class TestCategoriesAndLimits:
    def test_acwi_is_equity(self):
        assert ACWI in get_category_indices("actions")

    def test_every_class_has_one_category(self):
        cats = ["actions", "obligations", "alternatifs", "matieres_premieres", "liquidites"]
        all_idx = sorted(i for c in cats for i in get_category_indices(c))
        assert all_idx == list(range(N))

    def test_group_limits_include_acwi(self):
        p = PROFILS_TYPES["pd_generique"]()
        equity = [g for g in p.group_constraints() if g.name_fr.startswith("Actions")][0]
        assert ACWI in equity.asset_indices

    def test_policy_limits_alias(self):
        assert QuebecPensionRegulations is PolicyLimits
        assert ACWI in PolicyLimits.EQUITY_INDICES


class TestGettersWithoutSession:
    def test_fallback_to_defaults(self):
        assert np.allclose(get_policy_weights(), DEFAULT_CURRENT_WEIGHTS)
        assert len(get_min_weights()) == N and len(get_max_weights()) == N
        assert "politique_placement" in get_benchmark_portfolios()


class TestMonteCarloFlows:
    def test_contribution_growth_is_used(self):
        w = DEFAULT_CURRENT_WEIGHTS
        common = dict(weights=w, expected_returns=get_expected_returns(),
                      cov_matrix=get_covariance_matrix(), initial_assets=1e9,
                      annual_contribution=50e6, annual_benefit=0.0, n_simulations=200, seed=1)
        flat = MonteCarloSimulator(contribution_growth_rate=0.0, **common).simulate(10)
        growing = MonteCarloSimulator(contribution_growth_rate=0.05, **common).simulate(10)
        assert np.median(growing.asset_paths[:, -1]) > np.median(flat.asset_paths[:, -1])

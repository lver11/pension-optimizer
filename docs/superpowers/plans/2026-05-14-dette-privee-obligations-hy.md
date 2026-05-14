# Dette privée (index 15) + Obligations HY (index 16) Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the asset universe from 15 to 17 classes by adding Dette privée (illiquid private credit, index 15) and Obligations HY (liquid high-yield, index 16).

**Architecture:** Three files hold hardcoded arrays that must be updated; every other page is already dynamic via `ASSET_CLASSES_ORDER`. Changes flow: `config.py` (enum + params + 17×17 correlation matrix) → `constraints/regulatory.py` (index lists) → `pages/page_rebalancing.py` (transaction costs). Tests are written first and must all fail before any implementation begins.

**Tech Stack:** Python 3.x, NumPy, pytest, Streamlit (no Streamlit changes required).

**Spec:** `docs/superpowers/specs/2026-05-14-dette-privee-obligations-hy-design.md`

---

## Chunk 1: Test file + config.py

### Task 1: Write failing tests

**Files:**
- Create: `tests/test_dette_privee_hy.py`

- [ ] **Step 1: Create the test file**

Create `tests/test_dette_privee_hy.py` with this exact content:

```python
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


class TestDettePrveeParams:
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
        # US–CDN
        assert abs(DEFAULT_CORRELATION_MATRIX[0, 1] - 0.75) < 1e-9
        # ACWI–US (moderated for PSD)
        assert abs(DEFAULT_CORRELATION_MATRIX[13, 1] - 0.88) < 1e-9
        # EMD–EM equity (moderated for PSD)
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
        """DP is illiquid — short selling prohibited."""
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "/Users/macbookprom1max/Library/CloudStorage/OneDrive-FONDACTION(CSN)/Documents/Claude/pension_optimizer"
pytest tests/test_dette_privee_hy.py -v 2>&1 | head -40
```

Expected: Multiple FAILs / ERRORs (AttributeError on `AssetClass.DETTE_PRIVEE`, wrong lengths, etc.). If they all pass already, stop — something is wrong.

- [ ] **Step 3: Commit the test file**

```bash
git add tests/test_dette_privee_hy.py
git commit -m "test: add failing tests for Dette privee (15) + Obligations HY (16)"
```

---

### Task 2: Implement config.py

**Files:**
- Modify: `config.py`

> **Context:** `config.py` is at the repo root. The file currently has 15 asset classes (indices 0–14). You must add 2 enum members, 2 `AssetClassConfig` entries, expand `ASSET_CLASSES_ORDER`, replace the 15×15 correlation matrix with a 17×17 one, extend `DEFAULT_CURRENT_WEIGHTS`, extend 3 hardcoded `BENCHMARK_PORTFOLIOS` arrays, add index 16 to `ALPHA_ELIGIBLE_SHORT`, and add 2 colors to `CHART_COLORS`.

- [ ] **Step 1: Add enum members to `AssetClass`**

In `config.py`, the `AssetClass` enum ends with `DETTE_EMERGENTE = "dette_emergente"`. Add after it:

```python
    DETTE_PRIVEE = "dette_privee"
    OBLIGATIONS_HY = "obligations_hy"
```

The enum block should now end with:
```python
    ACTIONS_ACWI = "actions_acwi"
    DETTE_EMERGENTE = "dette_emergente"
    DETTE_PRIVEE = "dette_privee"
    OBLIGATIONS_HY = "obligations_hy"
```

- [ ] **Step 2: Add `AssetClassConfig` entries to `ASSET_DEFAULTS`**

After the `DETTE_EMERGENTE` entry in `ASSET_DEFAULTS`, add:

```python
    AssetClass.DETTE_PRIVEE: AssetClassConfig(
        AssetClass.DETTE_PRIVEE, "Dette privee",
        0.075, 0.06, 0.10, 50.0, 0.00, 0.10, True, 5.0
    ),
    AssetClass.OBLIGATIONS_HY: AssetClassConfig(
        AssetClass.OBLIGATIONS_HY, "Obligations HY",
        0.055, 0.09, 0.80, 48.0, 0.00, 0.15, False, 4.5
    ),
```

`AssetClassConfig` field order: `(code, nom_fr, expected_return, volatility, liquidity_score, esg_score, min_allocation, max_allocation, is_alternative, duration)`.

- [ ] **Step 3: Append to `ASSET_CLASSES_ORDER`**

Change the list from:
```python
ASSET_CLASSES_ORDER = [
    ...
    AssetClass.DETTE_EMERGENTE,
]
```
to:
```python
ASSET_CLASSES_ORDER = [
    AssetClass.ACTIONS_CDN, AssetClass.ACTIONS_US, AssetClass.ACTIONS_EAFE,
    AssetClass.ACTIONS_EMERGENTES, AssetClass.OBLIGATIONS_GOV_CDN,
    AssetClass.OBLIGATIONS_CORP, AssetClass.OBLIGATIONS_INFLATION,
    AssetClass.IMMOBILIER, AssetClass.INFRASTRUCTURE,
    AssetClass.CAPITAL_INVESTISSEMENT, AssetClass.RENDEMENT_ABSOLU,
    AssetClass.MATIERES_PREMIERES, AssetClass.ENCAISSE,
    AssetClass.ACTIONS_ACWI,
    AssetClass.DETTE_EMERGENTE,
    AssetClass.DETTE_PRIVEE,
    AssetClass.OBLIGATIONS_HY,
]
```

- [ ] **Step 4: Replace `DEFAULT_CORRELATION_MATRIX` with the 17×17 matrix**

Replace the entire `DEFAULT_CORRELATION_MATRIX` block with:

```python
DEFAULT_CORRELATION_MATRIX = np.array([
    #  CDN   US   EAFE  EM   GovB CorpB InflB Immo Infra  PE   RA  Comm  Cash ACWI  EMD   DP    HY
    [ 1.00, 0.75, 0.70, 0.60,-0.15, 0.10,-0.05, 0.35, 0.30, 0.55, 0.30, 0.30, 0.00, 0.68, 0.15, 0.35, 0.40],
    [ 0.75, 1.00, 0.80, 0.65,-0.20, 0.05,-0.10, 0.30, 0.25, 0.60, 0.35, 0.25, 0.00, 0.88, 0.10, 0.35, 0.50],
    [ 0.70, 0.80, 1.00, 0.70,-0.10, 0.08,-0.05, 0.30, 0.28, 0.55, 0.30, 0.28, 0.00, 0.83, 0.15, 0.30, 0.35],
    [ 0.60, 0.65, 0.70, 1.00,-0.05, 0.10, 0.00, 0.25, 0.22, 0.50, 0.25, 0.35, 0.00, 0.75, 0.45, 0.25, 0.40],
    [-0.15,-0.20,-0.10,-0.05, 1.00, 0.60, 0.70,-0.05, 0.05,-0.15, 0.10,-0.10, 0.10,-0.15, 0.25, 0.10, 0.10],
    [ 0.10, 0.05, 0.08, 0.10, 0.60, 1.00, 0.50, 0.15, 0.15, 0.05, 0.15, 0.05, 0.05,-0.05, 0.35, 0.35, 0.60],
    [-0.05,-0.10,-0.05, 0.00, 0.70, 0.50, 1.00, 0.10, 0.12,-0.10, 0.05, 0.35, 0.05,-0.10, 0.20, 0.10, 0.10],
    [ 0.35, 0.30, 0.30, 0.25,-0.05, 0.15, 0.10, 1.00, 0.45, 0.40, 0.25, 0.15, 0.00, 0.40, 0.15, 0.25, 0.25],
    [ 0.30, 0.25, 0.28, 0.22, 0.05, 0.15, 0.12, 0.45, 1.00, 0.35, 0.20, 0.20, 0.00, 0.30, 0.18, 0.30, 0.25],
    [ 0.55, 0.60, 0.55, 0.50,-0.15, 0.05,-0.10, 0.40, 0.35, 1.00, 0.30, 0.20, 0.00, 0.55, 0.20, 0.45, 0.35],
    [ 0.30, 0.35, 0.30, 0.25, 0.10, 0.15, 0.05, 0.25, 0.20, 0.30, 1.00, 0.15, 0.05, 0.35, 0.20, 0.20, 0.35],
    [ 0.30, 0.25, 0.28, 0.35,-0.10, 0.05, 0.35, 0.15, 0.20, 0.20, 0.15, 1.00, 0.00, 0.30, 0.25, 0.15, 0.25],
    [ 0.00, 0.00, 0.00, 0.00, 0.10, 0.05, 0.05, 0.00, 0.00, 0.00, 0.05, 0.00, 1.00,-0.10, 0.05, 0.05, 0.05],
    [ 0.68, 0.88, 0.83, 0.75,-0.15,-0.05,-0.10, 0.40, 0.30, 0.55, 0.35, 0.30,-0.10, 1.00, 0.30, 0.30, 0.45],
    [ 0.15, 0.10, 0.15, 0.45, 0.25, 0.35, 0.20, 0.15, 0.18, 0.20, 0.20, 0.25, 0.05, 0.30, 1.00, 0.25, 0.55],
    [ 0.35, 0.35, 0.30, 0.25, 0.10, 0.35, 0.10, 0.25, 0.30, 0.45, 0.20, 0.15, 0.05, 0.30, 0.25, 1.00, 0.45],
    [ 0.40, 0.50, 0.35, 0.40, 0.10, 0.60, 0.10, 0.25, 0.25, 0.35, 0.35, 0.25, 0.05, 0.45, 0.55, 0.45, 1.00],
])
```

The first 15 rows × 15 columns are IDENTICAL to the current matrix. Only columns 15 and 16 are new, plus rows 15 and 16.

- [ ] **Step 5: Update `DEFAULT_CURRENT_WEIGHTS`**

Change from 15 elements to 17:
```python
DEFAULT_CURRENT_WEIGHTS = np.array([
    0.12, 0.14, 0.08, 0.05, 0.19, 0.10, 0.05, 0.07, 0.07, 0.05, 0.03, 0.03, 0.02, 0.00, 0.00, 0.00, 0.00,
])
```

- [ ] **Step 6: Update 3 hardcoded `BENCHMARK_PORTFOLIOS` arrays**

`"politique_placement"` uses `DEFAULT_CURRENT_WEIGHTS.copy()` — no change needed.

Update the 3 hardcoded arrays to 17 elements (append `0.00, 0.00`):

```python
BENCHMARK_PORTFOLIOS = {
    "60_40_equilibre": {
        "nom_fr": "60/40 Equilibre",
        "weights": np.array([0.10, 0.15, 0.08, 0.07, 0.25, 0.15, 0.05,
                             0.05, 0.05, 0.03, 0.00, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00]),
    },
    "politique_placement": {
        "nom_fr": "Politique de placement actuelle",
        "weights": DEFAULT_CURRENT_WEIGHTS.copy(),
    },
    "obligations_pures": {
        "nom_fr": "Obligations pures (LDI)",
        "weights": np.array([0.00, 0.00, 0.00, 0.00, 0.40, 0.30, 0.25,
                             0.00, 0.00, 0.00, 0.00, 0.00, 0.05, 0.00, 0.00, 0.00, 0.00]),
    },
    "croissance_70_30": {
        "nom_fr": "Croissance (70/30)",
        "weights": np.array([0.15, 0.20, 0.10, 0.10, 0.15, 0.10, 0.05,
                             0.05, 0.05, 0.03, 0.00, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00]),
    },
}
```

- [ ] **Step 7: Update `ALPHA_ELIGIBLE_SHORT`**

Change from:
```python
# Indices: 0-3 equities, 4-6 bonds, 11 commodities, 13 ACWI, 14 EMD
ALPHA_ELIGIBLE_SHORT = [0, 1, 2, 3, 4, 5, 6, 11, 13, 14]
```
to:
```python
# Indices: 0-3 equities, 4-6 bonds, 11 commodities, 13 ACWI, 14 EMD, 16 HY
# Note: 15 (Dette privee) excluded — illiquid private market
ALPHA_ELIGIBLE_SHORT = [0, 1, 2, 3, 4, 5, 6, 11, 13, 14, 16]
```

- [ ] **Step 8: Update `CHART_COLORS`**

Change from 15 entries to 17 by appending 2 colors:

```python
CHART_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#636efa", "#aec7e8", "#ffbb78",
    "#ffa07a",  # Actions MSCI ACWI
    "#20b2aa",  # Dette pays emergents
    "#8b4513",  # Dette privee (saddle brown)
    "#ff6347",  # Obligations HY (tomato)
]
```

- [ ] **Step 9: Run the config-related tests**

```bash
cd "/Users/macbookprom1max/Library/CloudStorage/OneDrive-FONDACTION(CSN)/Documents/Claude/pension_optimizer"
pytest tests/test_dette_privee_hy.py -v -k "not Regulatory" 2>&1
```

Expected: All non-regulatory tests PASS. Regulatory tests still fail (not yet updated).

- [ ] **Step 10: Commit config.py**

```bash
git add config.py
git commit -m "feat: add Dette privee (15) + Obligations HY (16) to config — 17x17 correlation matrix"
```

---

## Chunk 2: regulatory.py + rebalancing.py + push

### Task 3: Update constraints/regulatory.py

**Files:**
- Modify: `constraints/regulatory.py`

> **Context:** `regulatory.py` holds 4 hardcoded index lists inside `QuebecPensionRegulations` and `PortableAlphaRegulations`. Indices 0–14 exist today. You are adding 15 (Dette privée) and 16 (Obligations HY).

- [ ] **Step 1: Update `ALTERNATIVE_INDICES`**

Change from:
```python
ALTERNATIVE_INDICES = [7, 8, 9, 10]    # Immobilier, Infrastructure, PE, Rendement absolu
```
to:
```python
ALTERNATIVE_INDICES = [7, 8, 9, 10, 15]  # + Dette privee
```

- [ ] **Step 2: Update `BOND_INDICES`**

Change from:
```python
BOND_INDICES = [4, 5, 6, 14]           # Oblig Gov, Corp, Inflation, Dette EM
```
to:
```python
BOND_INDICES = [4, 5, 6, 14, 16]       # + Obligations HY
```

- [ ] **Step 3: Update `FOREIGN_INDICES`**

Change from:
```python
FOREIGN_INDICES = [1, 2, 3, 13, 14]    # Actions intl, ACWI, Dette EM
```
to:
```python
FOREIGN_INDICES = [1, 2, 3, 13, 14, 15, 16]  # + Dette privee, Obligations HY
```

- [ ] **Step 4: Update `SHORT_ELIGIBLE_INDICES` in `PortableAlphaRegulations`**

Change from:
```python
SHORT_ELIGIBLE_INDICES = [0, 1, 2, 3, 4, 5, 6, 11, 13, 14]  # Equities + Bonds + Commodities + ACWI + EMD
```
to:
```python
SHORT_ELIGIBLE_INDICES = [0, 1, 2, 3, 4, 5, 6, 11, 13, 14, 16]  # + Obligations HY (liquid ETF)
# Note: 15 (Dette privee) excluded — illiquid, short selling prohibited
```

- [ ] **Step 5: Run regulatory tests**

```bash
cd "/Users/macbookprom1max/Library/CloudStorage/OneDrive-FONDACTION(CSN)/Documents/Claude/pension_optimizer"
pytest tests/test_dette_privee_hy.py -v -k "Regulatory" 2>&1
```

Expected: All `TestRegulatoryIndices` tests PASS.

- [ ] **Step 6: Run full test suite to check no regressions**

```bash
pytest tests/ -v 2>&1
```

Expected: All tests pass (including older test_acwi and test_emd files).

- [ ] **Step 7: Commit regulatory.py**

```bash
git add constraints/regulatory.py
git commit -m "feat: update regulatory indices for Dette privee (15) + Obligations HY (16)"
```

---

### Task 4: Update page_rebalancing.py + final verification + push

**Files:**
- Modify: `pages/page_rebalancing.py`

> **Context:** `TRANSACTION_COSTS_BPS` at the top of `page_rebalancing.py` is a dict keyed by French asset names. The `render()` function looks up each asset name via `TRANSACTION_COSTS_BPS.get(name, 10)` — missing entries silently fall back to 10 bps. Still, we add explicit entries for correctness.

- [ ] **Step 1: Add two entries to `TRANSACTION_COSTS_BPS`**

After `"Dette pays emergents": 25,` add:

```python
    "Dette privee": 300,      # Marche prive tres illiquide
    "Obligations HY": 30,     # Liquide mais spread bid-ask plus large que IG
```

The full dict should now look like:
```python
TRANSACTION_COSTS_BPS = {
    "Actions canadiennes": 15,
    "Actions americaines": 10,
    "Actions EAFE": 20,
    "Actions emergentes": 30,
    "Obligations gouvernementales CDN": 5,
    "Obligations corporatives": 10,
    "Obligations indexees inflation": 8,
    "Immobilier": 150,
    "Infrastructure": 200,
    "Capital investissement": 200,
    "Rendement absolu": 15,
    "Matieres premieres": 20,
    "Encaisse": 1,
    "Actions MSCI ACWI": 10,
    "Dette pays emergents": 25,
    "Dette privee": 300,
    "Obligations HY": 30,
}
```

- [ ] **Step 2: Run the full test suite one final time**

```bash
cd "/Users/macbookprom1max/Library/CloudStorage/OneDrive-FONDACTION(CSN)/Documents/Claude/pension_optimizer"
pytest tests/ -v 2>&1
```

Expected: All tests pass. Confirm `test_dette_privee_hy.py` shows 38+ passing tests, zero failures.

- [ ] **Step 3: Commit page_rebalancing.py**

```bash
git add pages/page_rebalancing.py
git commit -m "feat: add transaction costs for Dette privee (300 bps) + Obligations HY (30 bps)"
```

- [ ] **Step 4: Push to remote (triggers Streamlit Cloud deploy)**

```bash
git push
```

Expected: `git push` completes successfully. Streamlit Cloud will redeploy automatically.

# Actions MSCI ACWI — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add "Actions MSCI ACWI" as a 14th asset class (index 13) to the pension optimizer, alongside the existing US/EAFE/EM classes.

**Architecture:** Three files need manual updates because they contain hardcoded arrays indexed by asset position: `config.py` (enum, defaults, correlation matrix 13×13→14×14, weight arrays), `constraints/regulatory.py` (hardcoded index lists), and `pages/page_rebalancing.py` (transaction cost dict keyed by French name). All other components already iterate over `ASSET_CLASSES_ORDER` dynamically and require no changes.

**Tech Stack:** Python, NumPy, Streamlit — no new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-06-acwi-asset-class-design.md`

---

## Chunk 1: config.py + regulatory.py + rebalancing.py

### Task 1: Write failing tests for config.py changes

**Files:**
- Create: `tests/test_acwi_asset_class.py`

- [ ] **Step 1: Create the test file**

```python
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
```

- [ ] **Step 2: Run to confirm ALL tests fail (expected: 14-entry assertions will fail because we still have 13)**

```bash
cd "/Users/macbookprom1max/Library/CloudStorage/OneDrive-FONDACTION(CSN)/Documents/Claude/pension_optimizer"
pytest tests/test_acwi_asset_class.py -v 2>&1 | head -60
```

Expected: many `AssertionError` / `AttributeError: ACTIONS_ACWI`, no passes except possibly `test_cash_index_unchanged`.

---

### Task 2: Implement config.py — all ACWI changes

**Files:**
- Modify: `config.py`

> **Read `config.py` before editing.** The entire file is at the top of the project root.

- [ ] **Step 3: Add `ACTIONS_ACWI` to the `AssetClass` enum (after `ENCAISSE`)**

In `config.py`, line 25 currently ends the enum with:
```python
    ENCAISSE = "encaisse"
```
Change it to:
```python
    ENCAISSE = "encaisse"
    ACTIONS_ACWI = "actions_acwi"
```

- [ ] **Step 4: Add ACWI config to `ASSET_DEFAULTS`**

After the `ENCAISSE` entry (line 64):
```python
    AssetClass.ENCAISSE: AssetClassConfig(AssetClass.ENCAISSE, "Encaisse", 0.025, 0.01, 1.00, 75.0, 0.02, 0.10, False, 0.25),
```
Add:
```python
    AssetClass.ACTIONS_ACWI: AssetClassConfig(
        AssetClass.ACTIONS_ACWI, "Actions MSCI ACWI",
        0.078, 0.16, 0.95, 58.0, 0.00, 0.40, False
    ),
```

- [ ] **Step 5: Append ACWI to `ASSET_CLASSES_ORDER`**

Current last line of the list (line 48):
```python
    AssetClass.MATIERES_PREMIERES, AssetClass.ENCAISSE,
```
Change to:
```python
    AssetClass.MATIERES_PREMIERES, AssetClass.ENCAISSE,
    AssetClass.ACTIONS_ACWI,
```

- [ ] **Step 6: Expand `DEFAULT_CORRELATION_MATRIX` from 13×13 to 14×14**

Replace the entire `DEFAULT_CORRELATION_MATRIX` block (lines 67-82) with:

```python
DEFAULT_CORRELATION_MATRIX = np.array([
    #  CDN   US   EAFE  EM   GovB CorpB InflB Immo Infra  PE   RA  Comm  Cash ACWI
    [ 1.00, 0.75, 0.70, 0.60,-0.15, 0.10,-0.05, 0.35, 0.30, 0.55, 0.30, 0.30, 0.00, 0.72],
    [ 0.75, 1.00, 0.80, 0.65,-0.20, 0.05,-0.10, 0.30, 0.25, 0.60, 0.35, 0.25, 0.00, 0.95],
    [ 0.70, 0.80, 1.00, 0.70,-0.10, 0.08,-0.05, 0.30, 0.28, 0.55, 0.30, 0.28, 0.00, 0.90],
    [ 0.60, 0.65, 0.70, 1.00,-0.05, 0.10, 0.00, 0.25, 0.22, 0.50, 0.25, 0.35, 0.00, 0.82],
    [-0.15,-0.20,-0.10,-0.05, 1.00, 0.60, 0.70,-0.05, 0.05,-0.15, 0.10,-0.10, 0.10,-0.15],
    [ 0.10, 0.05, 0.08, 0.10, 0.60, 1.00, 0.50, 0.15, 0.15, 0.05, 0.15, 0.05, 0.05,-0.05],
    [-0.05,-0.10,-0.05, 0.00, 0.70, 0.50, 1.00, 0.10, 0.12,-0.10, 0.05, 0.35, 0.05,-0.10],
    [ 0.35, 0.30, 0.30, 0.25,-0.05, 0.15, 0.10, 1.00, 0.45, 0.40, 0.25, 0.15, 0.00, 0.40],
    [ 0.30, 0.25, 0.28, 0.22, 0.05, 0.15, 0.12, 0.45, 1.00, 0.35, 0.20, 0.20, 0.00, 0.30],
    [ 0.55, 0.60, 0.55, 0.50,-0.15, 0.05,-0.10, 0.40, 0.35, 1.00, 0.30, 0.20, 0.00, 0.55],
    [ 0.30, 0.35, 0.30, 0.25, 0.10, 0.15, 0.05, 0.25, 0.20, 0.30, 1.00, 0.15, 0.05, 0.35],
    [ 0.30, 0.25, 0.28, 0.35,-0.10, 0.05, 0.35, 0.15, 0.20, 0.20, 0.15, 1.00, 0.00, 0.30],
    [ 0.00, 0.00, 0.00, 0.00, 0.10, 0.05, 0.05, 0.00, 0.00, 0.00, 0.05, 0.00, 1.00,-0.10],
    [ 0.72, 0.95, 0.90, 0.82,-0.15,-0.05,-0.10, 0.40, 0.30, 0.55, 0.35, 0.30,-0.10, 1.00],
])
```

- [ ] **Step 7: Update `DEFAULT_CURRENT_WEIGHTS` to 14 elements**

Replace lines 84-86:
```python
DEFAULT_CURRENT_WEIGHTS = np.array([
    0.12, 0.14, 0.08, 0.05, 0.19, 0.10, 0.05, 0.07, 0.07, 0.05, 0.03, 0.03, 0.02,
])
```
With:
```python
DEFAULT_CURRENT_WEIGHTS = np.array([
    0.12, 0.14, 0.08, 0.05, 0.19, 0.10, 0.05, 0.07, 0.07, 0.05, 0.03, 0.03, 0.02, 0.00,
])
```

- [ ] **Step 8: Update the 3 hardcoded benchmark weight arrays in `BENCHMARK_PORTFOLIOS`**

Replace the three `np.array(...)` calls with 14-element versions (append `0.00` to each):

```python
BENCHMARK_PORTFOLIOS = {
    "60_40_equilibre": {
        "nom_fr": "60/40 Equilibre",
        "weights": np.array([0.10, 0.15, 0.08, 0.07, 0.25, 0.15, 0.05,
                             0.05, 0.05, 0.03, 0.00, 0.02, 0.00, 0.00]),
    },
    "politique_placement": {
        "nom_fr": "Politique de placement actuelle",
        "weights": DEFAULT_CURRENT_WEIGHTS.copy(),
    },
    "obligations_pures": {
        "nom_fr": "Obligations pures (LDI)",
        "weights": np.array([0.00, 0.00, 0.00, 0.00, 0.40, 0.30, 0.25,
                             0.00, 0.00, 0.00, 0.00, 0.00, 0.05, 0.00]),
    },
    "croissance_70_30": {
        "nom_fr": "Croissance (70/30)",
        "weights": np.array([0.15, 0.20, 0.10, 0.10, 0.15, 0.10, 0.05,
                             0.05, 0.05, 0.03, 0.00, 0.02, 0.00, 0.00]),
    },
}
```

- [ ] **Step 9: Update `ALPHA_ELIGIBLE_SHORT` — add index 13**

Replace line 200:
```python
ALPHA_ELIGIBLE_SHORT = [0, 1, 2, 3, 4, 5, 6, 11]
```
With:
```python
ALPHA_ELIGIBLE_SHORT = [0, 1, 2, 3, 4, 5, 6, 11, 13]
```

- [ ] **Step 10: Add 14th color to `CHART_COLORS`**

Replace lines 121-124:
```python
CHART_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#636efa", "#aec7e8", "#ffbb78",
]
```
With:
```python
CHART_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#636efa", "#aec7e8", "#ffbb78",
    "#ffa07a",  # Actions MSCI ACWI
]
```

---

### Task 3: Update constraints/regulatory.py — index lists

**Files:**
- Modify: `constraints/regulatory.py:25-33,172`

- [ ] **Step 11: Add index 13 to `EQUITY_INDICES`**

Replace line 25:
```python
    EQUITY_INDICES = [0, 1, 2, 3]          # Actions CDN, US, EAFE, Emergentes
```
With:
```python
    EQUITY_INDICES = [0, 1, 2, 3, 13]      # Actions CDN, US, EAFE, Emergentes, ACWI
```

- [ ] **Step 12: Add index 13 to `FOREIGN_INDICES`**

Replace line 33:
```python
    FOREIGN_INDICES = [1, 2, 3]            # Actions internationales
```
With:
```python
    FOREIGN_INDICES = [1, 2, 3, 13]        # Actions internationales (dont ACWI)
```

- [ ] **Step 13: Add index 13 to `SHORT_ELIGIBLE_INDICES`**

Replace line 172:
```python
    SHORT_ELIGIBLE_INDICES = [0, 1, 2, 3, 4, 5, 6, 11]  # Equities + Bonds + Commodities
```
With:
```python
    SHORT_ELIGIBLE_INDICES = [0, 1, 2, 3, 4, 5, 6, 11, 13]  # Equities + Bonds + Commodities + ACWI
```

---

### Task 4: Update page_rebalancing.py — transaction costs

**Files:**
- Modify: `pages/page_rebalancing.py:20-33`

- [ ] **Step 14: Add ACWI to `TRANSACTION_COSTS_BPS`**

Replace lines 20-33:
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
    "Matieres premieres": 20,
    "Encaisse": 1,
}
```
With:
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
}
```

> Note: "Rendement absolu" is also added here — it was a pre-existing gap in the dict (index 10, score liquidité 0.50). Its transaction cost is set conservatively at 15 bps.

---

### Task 5: Run tests and verify all pass

- [ ] **Step 15: Run the full test suite**

```bash
cd "/Users/macbookprom1max/Library/CloudStorage/OneDrive-FONDACTION(CSN)/Documents/Claude/pension_optimizer"
pytest tests/test_acwi_asset_class.py -v
```

Expected output: all tests pass. If any fail, debug and fix before committing.

- [ ] **Step 16: Quick smoke test — import all modules to catch any runtime error**

```bash
cd "/Users/macbookprom1max/Library/CloudStorage/OneDrive-FONDACTION(CSN)/Documents/Claude/pension_optimizer"
python -c "
import config
import constraints.regulatory as reg
import pages.page_rebalancing  # imports at module level — will fail fast if broken
print('n_assets =', len(config.ASSET_CLASSES_ORDER))
print('cov shape =', config.DEFAULT_CORRELATION_MATRIX.shape)
print('weights sum =', config.DEFAULT_CURRENT_WEIGHTS.sum())
print('equity_indices =', reg.QuebecPensionRegulations.EQUITY_INDICES)
print('All OK')
"
```

Expected output:
```
n_assets = 14
cov shape = (14, 14)
weights sum = 1.0
equity_indices = [0, 1, 2, 3, 13]
All OK
```

---

### Task 6: Commit

- [ ] **Step 17: Stage and commit**

```bash
cd "/Users/macbookprom1max/Library/CloudStorage/OneDrive-FONDACTION(CSN)/Documents/Claude/pension_optimizer"
git add config.py constraints/regulatory.py pages/page_rebalancing.py tests/test_acwi_asset_class.py
git commit -m "$(cat <<'EOF'
feat(config): add Actions MSCI ACWI as 14th asset class

Adds ACWI alongside existing US/EAFE/EM classes (not a replacement).
- config.py: enum, AssetClassConfig, 14×14 correlation matrix, weight arrays
- constraints/regulatory.py: EQUITY_INDICES, FOREIGN_INDICES, SHORT_ELIGIBLE_INDICES
- page_rebalancing.py: TRANSACTION_COSTS_BPS (also adds missing Rendement absolu)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 18: Push so Streamlit Cloud deploys**

```bash
git push
```

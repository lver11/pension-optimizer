# Dette pays émergents — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add "Dette pays emergents" (hybrid 70% EMBI / 30% GBI-EM) as a 15th asset class (index 14) to the pension optimizer.

**Architecture:** Three files need manual updates due to hardcoded arrays: `config.py` (enum, defaults, correlation matrix 14×14→15×15, weight arrays), `constraints/regulatory.py` (BOND_INDICES, FOREIGN_INDICES, SHORT_ELIGIBLE_INDICES), `pages/page_rebalancing.py` (transaction cost dict). All other components iterate over `ASSET_CLASSES_ORDER` dynamically and require no changes. Note: correlation values are pre-verified PSD (min eigenvalue +0.033) — see comments in matrix below.

**Tech Stack:** Python, NumPy, Streamlit — no new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-13-emd-asset-class-design.md`

---

## Chunk 1: All changes

### Task 1: Write failing tests

**Files:**
- Create: `tests/test_emd_asset_class.py`

- [ ] **Step 1: Create the test file**

```python
# tests/test_emd_asset_class.py
"""
Tests for the Dette pays emergents 15th asset class addition.
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


N = 15  # expected number of asset classes after adding EMD


class TestAssetCount:
    def test_asset_names_has_15_entries(self):
        assert len(get_asset_names_fr()) == N

    def test_asset_classes_order_has_15_entries(self):
        assert len(ASSET_CLASSES_ORDER) == N

    def test_emd_enum_exists(self):
        assert hasattr(AssetClass, "DETTE_EMERGENTE")

    def test_emd_is_last_in_order(self):
        assert ASSET_CLASSES_ORDER[-1] == AssetClass.DETTE_EMERGENTE

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
    def test_shape_is_15x15(self):
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
    def test_chart_colors_has_15_entries(self):
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
```

- [ ] **Step 2: Run to confirm tests fail**

```bash
cd "/Users/macbookprom1max/Library/CloudStorage/OneDrive-FONDACTION(CSN)/Documents/Claude/pension_optimizer"
python3 -m pytest tests/test_emd_asset_class.py -v 2>&1 | head -40
```

Expected: `AttributeError: DETTE_EMERGENTE` and `AssertionError` (N=15 vs actual 14). No passes except possibly `test_cash_index_unchanged`.

---

### Task 2: Implement config.py — all EMD changes

**Files:**
- Modify: `config.py`

> Read `config.py` before editing. It is at the project root.

- [ ] **Step 3: Add `DETTE_EMERGENTE` to the `AssetClass` enum (after `ACTIONS_ACWI`)**

Change:
```python
    ACTIONS_ACWI = "actions_acwi"
```
To:
```python
    ACTIONS_ACWI = "actions_acwi"
    DETTE_EMERGENTE = "dette_emergente"
```

- [ ] **Step 4: Add EMD config to `ASSET_DEFAULTS` (after the ACTIONS_ACWI entry)**

After:
```python
    AssetClass.ACTIONS_ACWI: AssetClassConfig(
        AssetClass.ACTIONS_ACWI, "Actions MSCI ACWI",
        0.078, 0.16, 0.95, 58.0, 0.00, 0.40, False
    ),
```
Add:
```python
    AssetClass.DETTE_EMERGENTE: AssetClassConfig(
        AssetClass.DETTE_EMERGENTE, "Dette pays emergents",
        0.058, 0.11, 0.70, 45.0, 0.00, 0.15, False, 7.0
    ),
```

- [ ] **Step 5: Append EMD to `ASSET_CLASSES_ORDER`**

Change:
```python
    AssetClass.MATIERES_PREMIERES, AssetClass.ENCAISSE,
    AssetClass.ACTIONS_ACWI,
]
```
To:
```python
    AssetClass.MATIERES_PREMIERES, AssetClass.ENCAISSE,
    AssetClass.ACTIONS_ACWI,
    AssetClass.DETTE_EMERGENTE,
]
```

- [ ] **Step 6: Replace `DEFAULT_CORRELATION_MATRIX` with the 15×15 version**

Replace the entire `DEFAULT_CORRELATION_MATRIX` block with:

```python
DEFAULT_CORRELATION_MATRIX = np.array([
    #  CDN   US   EAFE  EM   GovB CorpB InflB Immo Infra  PE   RA  Comm  Cash ACWI  EMD
    # ACWI correls moderated vs. composition weights to preserve PSD property.
    # EMD correls: EM=0.45 (vs 0.55 naive), Corp=0.35 (vs 0.45), ACWI=0.30 (vs 0.40)
    [ 1.00, 0.75, 0.70, 0.60,-0.15, 0.10,-0.05, 0.35, 0.30, 0.55, 0.30, 0.30, 0.00, 0.68, 0.15],
    [ 0.75, 1.00, 0.80, 0.65,-0.20, 0.05,-0.10, 0.30, 0.25, 0.60, 0.35, 0.25, 0.00, 0.88, 0.10],
    [ 0.70, 0.80, 1.00, 0.70,-0.10, 0.08,-0.05, 0.30, 0.28, 0.55, 0.30, 0.28, 0.00, 0.83, 0.15],
    [ 0.60, 0.65, 0.70, 1.00,-0.05, 0.10, 0.00, 0.25, 0.22, 0.50, 0.25, 0.35, 0.00, 0.75, 0.45],
    [-0.15,-0.20,-0.10,-0.05, 1.00, 0.60, 0.70,-0.05, 0.05,-0.15, 0.10,-0.10, 0.10,-0.15, 0.25],
    [ 0.10, 0.05, 0.08, 0.10, 0.60, 1.00, 0.50, 0.15, 0.15, 0.05, 0.15, 0.05, 0.05,-0.05, 0.35],
    [-0.05,-0.10,-0.05, 0.00, 0.70, 0.50, 1.00, 0.10, 0.12,-0.10, 0.05, 0.35, 0.05,-0.10, 0.20],
    [ 0.35, 0.30, 0.30, 0.25,-0.05, 0.15, 0.10, 1.00, 0.45, 0.40, 0.25, 0.15, 0.00, 0.40, 0.15],
    [ 0.30, 0.25, 0.28, 0.22, 0.05, 0.15, 0.12, 0.45, 1.00, 0.35, 0.20, 0.20, 0.00, 0.30, 0.18],
    [ 0.55, 0.60, 0.55, 0.50,-0.15, 0.05,-0.10, 0.40, 0.35, 1.00, 0.30, 0.20, 0.00, 0.55, 0.20],
    [ 0.30, 0.35, 0.30, 0.25, 0.10, 0.15, 0.05, 0.25, 0.20, 0.30, 1.00, 0.15, 0.05, 0.35, 0.20],
    [ 0.30, 0.25, 0.28, 0.35,-0.10, 0.05, 0.35, 0.15, 0.20, 0.20, 0.15, 1.00, 0.00, 0.30, 0.25],
    [ 0.00, 0.00, 0.00, 0.00, 0.10, 0.05, 0.05, 0.00, 0.00, 0.00, 0.05, 0.00, 1.00,-0.10, 0.05],
    [ 0.68, 0.88, 0.83, 0.75,-0.15,-0.05,-0.10, 0.40, 0.30, 0.55, 0.35, 0.30,-0.10, 1.00, 0.30],
    [ 0.15, 0.10, 0.15, 0.45, 0.25, 0.35, 0.20, 0.15, 0.18, 0.20, 0.20, 0.25, 0.05, 0.30, 1.00],
])
```

- [ ] **Step 7: Update `DEFAULT_CURRENT_WEIGHTS` to 15 elements**

Replace:
```python
DEFAULT_CURRENT_WEIGHTS = np.array([
    0.12, 0.14, 0.08, 0.05, 0.19, 0.10, 0.05, 0.07, 0.07, 0.05, 0.03, 0.03, 0.02, 0.00,
])
```
With:
```python
DEFAULT_CURRENT_WEIGHTS = np.array([
    0.12, 0.14, 0.08, 0.05, 0.19, 0.10, 0.05, 0.07, 0.07, 0.05, 0.03, 0.03, 0.02, 0.00, 0.00,
])
```

- [ ] **Step 8: Update the 3 hardcoded benchmark weight arrays (append `0.00`)**

Replace the entire `BENCHMARK_PORTFOLIOS` block with:

```python
BENCHMARK_PORTFOLIOS = {
    "60_40_equilibre": {
        "nom_fr": "60/40 Equilibre",
        "weights": np.array([0.10, 0.15, 0.08, 0.07, 0.25, 0.15, 0.05,
                             0.05, 0.05, 0.03, 0.00, 0.02, 0.00, 0.00, 0.00]),
    },
    "politique_placement": {
        "nom_fr": "Politique de placement actuelle",
        "weights": DEFAULT_CURRENT_WEIGHTS.copy(),
    },
    "obligations_pures": {
        "nom_fr": "Obligations pures (LDI)",
        "weights": np.array([0.00, 0.00, 0.00, 0.00, 0.40, 0.30, 0.25,
                             0.00, 0.00, 0.00, 0.00, 0.00, 0.05, 0.00, 0.00]),
    },
    "croissance_70_30": {
        "nom_fr": "Croissance (70/30)",
        "weights": np.array([0.15, 0.20, 0.10, 0.10, 0.15, 0.10, 0.05,
                             0.05, 0.05, 0.03, 0.00, 0.02, 0.00, 0.00, 0.00]),
    },
}
```

- [ ] **Step 9: Update `ALPHA_ELIGIBLE_SHORT` — add index 14**

Replace lines (comment + array):
```python
# Indices: 0-3 equities, 4-6 bonds, 11 commodities, 13 ACWI
ALPHA_ELIGIBLE_SHORT = [0, 1, 2, 3, 4, 5, 6, 11, 13]
```
With:
```python
# Indices: 0-3 equities, 4-6 bonds, 11 commodities, 13 ACWI, 14 EMD
ALPHA_ELIGIBLE_SHORT = [0, 1, 2, 3, 4, 5, 6, 11, 13, 14]
```

- [ ] **Step 10: Add 15th color to `CHART_COLORS`**

Replace:
```python
CHART_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#636efa", "#aec7e8", "#ffbb78",
    "#ffa07a",  # Actions MSCI ACWI
]
```
With:
```python
CHART_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#636efa", "#aec7e8", "#ffbb78",
    "#ffa07a",  # Actions MSCI ACWI
    "#20b2aa",  # Dette pays emergents
]
```

---

### Task 3: Update constraints/regulatory.py — index lists

**Files:**
- Modify: `constraints/regulatory.py`

> Read `constraints/regulatory.py` before editing. Current state: `BOND_INDICES = [4, 5, 6]`, `FOREIGN_INDICES = [1, 2, 3, 13]`, `SHORT_ELIGIBLE_INDICES = [0, 1, 2, 3, 4, 5, 6, 11, 13]`.

- [ ] **Step 11: Add index 14 to `BOND_INDICES`**

Replace:
```python
    BOND_INDICES = [4, 5, 6]               # Oblig Gov, Corp, Inflation
```
With:
```python
    BOND_INDICES = [4, 5, 6, 14]           # Oblig Gov, Corp, Inflation, Dette EM
```

- [ ] **Step 12: Add index 14 to `FOREIGN_INDICES`**

Replace:
```python
    FOREIGN_INDICES = [1, 2, 3, 13]        # Actions internationales (dont ACWI)
```
With:
```python
    FOREIGN_INDICES = [1, 2, 3, 13, 14]    # Actions intl, ACWI, Dette EM
```

- [ ] **Step 13: Add index 14 to `SHORT_ELIGIBLE_INDICES`**

Replace:
```python
    SHORT_ELIGIBLE_INDICES = [0, 1, 2, 3, 4, 5, 6, 11, 13]  # Equities + Bonds + Commodities + ACWI
```
With:
```python
    SHORT_ELIGIBLE_INDICES = [0, 1, 2, 3, 4, 5, 6, 11, 13, 14]  # Equities + Bonds + Commodities + ACWI + EMD
```

---

### Task 4: Update page_rebalancing.py — transaction costs

**Files:**
- Modify: `pages/page_rebalancing.py`

- [ ] **Step 14: Add EMD to `TRANSACTION_COSTS_BPS`**

The current dict ends with `"Actions MSCI ACWI": 10,`. Add after it:
```python
    "Dette pays emergents": 25,
```
(Less liquid than developed market bonds — reflects bid-ask spread on EMBI/GBI-EM ETFs)

---

### Task 5: Run tests and verify

- [ ] **Step 15: Run the full test suite**

```bash
cd "/Users/macbookprom1max/Library/CloudStorage/OneDrive-FONDACTION(CSN)/Documents/Claude/pension_optimizer"
python3 -m pytest tests/test_emd_asset_class.py -v
```

Expected: all tests pass, including `test_positive_semidefinite`.

- [ ] **Step 16: Smoke test**

```bash
cd "/Users/macbookprom1max/Library/CloudStorage/OneDrive-FONDACTION(CSN)/Documents/Claude/pension_optimizer"
python3 -c "
import config
import constraints.regulatory as reg
print('n_assets =', len(config.ASSET_CLASSES_ORDER))
print('cov shape =', config.DEFAULT_CORRELATION_MATRIX.shape)
print('weights sum =', config.DEFAULT_CURRENT_WEIGHTS.sum())
print('bond_indices =', reg.QuebecPensionRegulations.BOND_INDICES)
print('equity_indices =', reg.QuebecPensionRegulations.EQUITY_INDICES)
import numpy as np
e = np.linalg.eigvalsh(config.DEFAULT_CORRELATION_MATRIX)
print('min eigenvalue =', round(e.min(), 6))
print('All OK')
"
```

Expected output:
```
n_assets = 15
cov shape = (15, 15)
weights sum = 1.0
bond_indices = [4, 5, 6, 14]
equity_indices = [0, 1, 2, 3, 13]
min eigenvalue = 0.032...
All OK
```

---

### Task 6: Commit and push

- [ ] **Step 17: Stage and commit**

```bash
cd "/Users/macbookprom1max/Library/CloudStorage/OneDrive-FONDACTION(CSN)/Documents/Claude/pension_optimizer"
git add config.py constraints/regulatory.py pages/page_rebalancing.py tests/test_emd_asset_class.py
git commit -m "$(cat <<'EOF'
feat(config): add Dette pays emergents as 15th asset class

Adds hybrid EM debt (70% EMBI / 30% GBI-EM) alongside existing classes.
- config.py: enum, AssetClassConfig (5.8%/11%/dur 7y), 15x15 correlation
  matrix (correls moderated for PSD: EM=0.45, Corp=0.35, ACWI=0.30)
- constraints/regulatory.py: BOND_INDICES, FOREIGN_INDICES, SHORT_ELIGIBLE
- page_rebalancing.py: TRANSACTION_COSTS_BPS 25 bps

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 18: Push to Streamlit Cloud**

```bash
git push
```

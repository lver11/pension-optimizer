# Optimisation Durable — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a 5-page sustainable portfolio optimization section to the existing Streamlit app, integrating multi-dimensional sustainability scores from the Excel cartography file and a bi-criteria Pareto frontier (Sharpe ↔ durability).

**Architecture:** New `sustainable/` module holds all data and optimizer logic; 5 new `pages/page_durable_*.py` files handle UI; `app.py` gains a new "🌱 Optimisation durable" navigation section. The optimizer extends `BaseOptimizer` and adds a λ-weighted sustainability term to the standard mean-variance objective.

**Tech Stack:** Python 3.9+, Streamlit, CVXPY (CLARABEL solver), NumPy, Pandas, Plotly

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `sustainable/__init__.py` | Package marker |
| Create | `sustainable/config.py` | 15 DurableAsset definitions with scores from Excel |
| Create | `sustainable/optimizer.py` | DurableResult dataclass + DurableOptimizer class |
| Create | `pages/page_durable_univers.py` | Asset universe config + dimension weights |
| Create | `pages/page_durable_scores.py` | Editable sustainability scores table |
| Create | `pages/page_durable_frontier.py` | Pareto frontier visualization |
| Create | `pages/page_durable_optimization.py` | Run optimization + compare allocations |
| Create | `pages/page_durable_rapport.py` | Summary report + CSV export |
| Create | `tests/sustainable/__init__.py` | Package marker |
| Create | `tests/sustainable/test_config.py` | Tests for data model and score computation |
| Create | `tests/sustainable/test_optimizer.py` | Tests for DurableOptimizer |
| Modify | `app.py` | Add new navigation section |

---

## Chunk 1: Data Foundation (`sustainable/config.py`)

### Task 1: Create DurableAsset dataclass and DURABLE_ASSETS registry

**Files:**
- Create: `sustainable/__init__.py`
- Create: `sustainable/config.py`
- Create: `tests/sustainable/__init__.py`
- Create: `tests/sustainable/test_config.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/sustainable/test_config.py
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sustainable.config import (
    DurableAsset, DURABLE_ASSETS, DURABLE_ASSET_ORDER,
    get_durable_returns, get_durable_volatilities,
    get_durable_correlation_matrix, get_durable_covariance_matrix,
    compute_composite_score, get_score_matrix,
)


def test_asset_count():
    assert len(DURABLE_ASSETS) == 15
    assert len(DURABLE_ASSET_ORDER) == 15


def test_asset_has_required_fields():
    asset = DURABLE_ASSETS["obligations_univers"]
    assert 0 < asset.rendement < 0.20
    assert 0 < asset.volatilite < 0.40
    assert 1 <= asset.score_durabilite <= 5
    assert asset.has_durable_variant is True
    assert asset.nom_durable != ""


def test_durable_variant_has_higher_durability_score():
    asset = DURABLE_ASSETS["obligations_univers"]
    assert asset.score_durabilite_durable >= asset.score_durabilite


def test_get_durable_returns_shape():
    r = get_durable_returns()
    assert r.shape == (15,)
    assert np.all(r > 0)


def test_get_durable_volatilities_shape():
    v = get_durable_volatilities()
    assert v.shape == (15,)
    assert np.all(v > 0)


def test_get_durable_correlation_matrix_valid():
    corr = get_durable_correlation_matrix()
    assert corr.shape == (15, 15)
    assert np.allclose(np.diag(corr), 1.0)
    assert np.allclose(corr, corr.T, atol=1e-8)
    eigvals = np.linalg.eigvalsh(corr)
    assert np.min(eigvals) >= -1e-8, "Correlation matrix must be PSD"


def test_get_durable_covariance_matrix_consistent():
    cov = get_durable_covariance_matrix()
    corr = get_durable_correlation_matrix()
    vols = get_durable_volatilities()
    # Diagonal should equal vol^2
    assert np.allclose(np.diag(cov), vols**2, atol=1e-10)
    assert cov.shape == (15, 15)


def test_compute_composite_score_equal_weights():
    weights_port = np.ones(15) / 15
    dim_weights = {"durabilite": 0.2, "additionnalite": 0.2,
                   "disponibilite": 0.2, "retombees_qc": 0.2, "liquidite": 0.2}
    score = compute_composite_score(weights_port, dim_weights, use_durable={})
    # Scores range 1–5 (source: Excel cartography, liquidité dimension uses 1–5)
    assert 1.0 <= score <= 5.0


def test_compute_composite_score_durable_variant_higher():
    weights_port = np.zeros(15)
    idx = DURABLE_ASSET_ORDER.index("obligations_univers")
    weights_port[idx] = 1.0
    dim_weights = {"durabilite": 1.0, "additionnalite": 0.0,
                   "disponibilite": 0.0, "retombees_qc": 0.0, "liquidite": 0.0}
    score_std = compute_composite_score(weights_port, dim_weights, use_durable={})
    score_dur = compute_composite_score(weights_port, dim_weights,
                                        use_durable={"obligations_univers": True})
    assert score_dur > score_std


def test_get_score_matrix_shape():
    dim_weights = {"durabilite": 0.4, "additionnalite": 0.3,
                   "disponibilite": 0.1, "retombees_qc": 0.1, "liquidite": 0.1}
    S_std, S_dur = get_score_matrix(dim_weights)
    assert S_std.shape == (15,)
    assert S_dur.shape == (15,)
    # Durable scores should be >= standard for assets with durable variant
    for i, aid in enumerate(DURABLE_ASSET_ORDER):
        if DURABLE_ASSETS[aid].has_durable_variant:
            assert S_dur[i] >= S_std[i]
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd "/Users/macbookprom1max/Library/CloudStorage/OneDrive-FONDACTION(CSN)/Documents/Claude/pension_optimizer"
python -m pytest tests/sustainable/test_config.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'sustainable'`

- [ ] **Step 3: Create `sustainable/__init__.py` and `tests/sustainable/__init__.py`**

```python
# sustainable/__init__.py
# (empty)
```

```python
# tests/sustainable/__init__.py
# (empty)
```

- [ ] **Step 4: Create `sustainable/config.py`**

```python
# sustainable/config.py
"""
Données des classes d'actifs durables.
Source: WG_Catégories_actifs_v4.xlsx (Cartographie_complète_v1 + Anticipations + Corrélations)
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np

# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class DurableAsset:
    id: str
    nom: str
    nom_durable: str                   # nom de la variante durable (vide si aucune)
    rendement: float                   # rendement espéré (moyenne Aon/JPM/BlackRock)
    volatilite: float                  # volatilité anticipée (moyenne)
    rendement_durable: float           # rendement de la variante durable
    volatilite_durable: float          # volatilité de la variante durable
    has_durable_variant: bool          # True si variante durable disponible
    # Scores standard (source: Excel, colonne Potentiel)
    score_durabilite: float            # 1–5
    score_additionnalite: float
    score_disponibilite: float
    score_retombees_qc: float
    score_liquidite: float
    # Scores variante durable
    score_durabilite_durable: float = 0.0
    score_additionnalite_durable: float = 0.0
    score_disponibilite_durable: float = 0.0
    score_retombees_qc_durable: float = 0.0
    score_liquidite_durable: float = 0.0


# ---------------------------------------------------------------------------
# Registre des 15 classes d'actifs
# Source: Cartographie_complète_v1 + Anticipations (moyennes Aon/JPM/BlackRock)
# ---------------------------------------------------------------------------

DURABLE_ASSETS: Dict[str, DurableAsset] = {
    # --- Revenu fixe ---
    "obligations_ct": DurableAsset(
        id="obligations_ct",
        nom="Obligations court terme",
        nom_durable="",
        rendement=0.034, volatilite=0.026,
        rendement_durable=0.034, volatilite_durable=0.026,
        has_durable_variant=False,
        score_durabilite=1, score_additionnalite=1,
        score_disponibilite=1, score_retombees_qc=1, score_liquidite=5,
    ),
    "obligations_univers": DurableAsset(
        id="obligations_univers",
        nom="Obligations univers",
        nom_durable="Obligations vertes",
        rendement=0.0367, volatilite=0.0484,
        rendement_durable=0.035, volatilite_durable=0.050,
        has_durable_variant=True,
        score_durabilite=3, score_additionnalite=3,
        score_disponibilite=3, score_retombees_qc=2, score_liquidite=5,
        score_durabilite_durable=4, score_additionnalite_durable=4,
        score_disponibilite_durable=4, score_retombees_qc_durable=3,
        score_liquidite_durable=4,
    ),
    "hypotheques": DurableAsset(
        id="hypotheques",
        nom="Hypothèques commerciales",
        nom_durable="Hypothèques durables",
        rendement=0.045, volatilite=0.046,
        rendement_durable=0.044, volatilite_durable=0.047,
        has_durable_variant=True,
        score_durabilite=2.5, score_additionnalite=2.5,
        score_disponibilite=2.5, score_retombees_qc=2, score_liquidite=2,
        score_durabilite_durable=3.5, score_additionnalite_durable=3.5,
        score_disponibilite_durable=3.0, score_retombees_qc_durable=3,
        score_liquidite_durable=2,
    ),
    "oblig_rendement_eleve": DurableAsset(
        id="oblig_rendement_eleve",
        nom="Obligations à rendement élevé",
        nom_durable="",
        rendement=0.0567, volatilite=0.0773,
        rendement_durable=0.0567, volatilite_durable=0.0773,
        has_durable_variant=False,
        score_durabilite=1, score_additionnalite=1,
        score_disponibilite=1, score_retombees_qc=1, score_liquidite=4,
    ),
    "dettes_emergentes": DurableAsset(
        id="dettes_emergentes",
        nom="Dettes de pays émergents",
        nom_durable="",
        rendement=0.0587, volatilite=0.0925,
        rendement_durable=0.0587, volatilite_durable=0.0925,
        has_durable_variant=False,
        score_durabilite=1, score_additionnalite=1,
        score_disponibilite=1, score_retombees_qc=1, score_liquidite=3,
    ),
    # --- Actions ---
    "actions_cdn": DurableAsset(
        id="actions_cdn",
        nom="Actions canadiennes",
        nom_durable="",
        rendement=0.0587, volatilite=0.1388,
        rendement_durable=0.0587, volatilite_durable=0.1388,
        has_durable_variant=False,
        score_durabilite=2, score_additionnalite=2,
        score_disponibilite=2, score_retombees_qc=2, score_liquidite=5,
    ),
    "actions_globales": DurableAsset(
        id="actions_globales",
        nom="Actions globales",
        nom_durable="Actions ACWI Sustainable",
        rendement=0.063, volatilite=0.1385,
        rendement_durable=0.062, volatilite_durable=0.138,
        has_durable_variant=True,
        score_durabilite=2.5, score_additionnalite=2,
        score_disponibilite=3, score_retombees_qc=1, score_liquidite=5,
        score_durabilite_durable=4, score_additionnalite_durable=3,
        score_disponibilite_durable=4, score_retombees_qc_durable=1,
        score_liquidite_durable=5,
    ),
    "actions_petite_cap": DurableAsset(
        id="actions_petite_cap",
        nom="Actions mondiales petite cap.",
        nom_durable="",
        rendement=0.0633, volatilite=0.1708,
        rendement_durable=0.0633, volatilite_durable=0.1708,
        has_durable_variant=False,
        score_durabilite=2.5, score_additionnalite=2,
        score_disponibilite=3, score_retombees_qc=1, score_liquidite=4,
    ),
    "eqp": DurableAsset(
        id="eqp",
        nom="EQP (micro-cap Qc)",
        nom_durable="",
        rendement=0.0587, volatilite=0.1442,
        rendement_durable=0.0587, volatilite_durable=0.1442,
        has_durable_variant=False,
        score_durabilite=2.5, score_additionnalite=2,
        score_disponibilite=3, score_retombees_qc=3, score_liquidite=4,
    ),
    # --- Autres actifs liquides ---
    "fonds_couverture": DurableAsset(
        id="fonds_couverture",
        nom="Fonds de couverture",
        nom_durable="",
        rendement=0.0563, volatilite=0.0597,
        rendement_durable=0.0563, volatilite_durable=0.0597,
        has_durable_variant=False,
        score_durabilite=1.5, score_additionnalite=1,
        score_disponibilite=2, score_retombees_qc=1, score_liquidite=3,
    ),
    # --- Actifs privés ---
    "dette_privee": DurableAsset(
        id="dette_privee",
        nom="Dette privée",
        nom_durable="Dette privée (admissible)",
        rendement=0.0695, volatilite=0.081,
        rendement_durable=0.075, volatilite_durable=0.08,
        has_durable_variant=True,
        score_durabilite=3.25, score_additionnalite=3,
        score_disponibilite=3.5, score_retombees_qc=1, score_liquidite=2,
        score_durabilite_durable=4, score_additionnalite_durable=4,
        score_disponibilite_durable=4, score_retombees_qc_durable=5,
        score_liquidite_durable=1,
    ),
    "immo_prive": DurableAsset(
        id="immo_prive",
        nom="Immobilier privé",
        nom_durable="Immobilier (admissible)",
        rendement=0.0527, volatilite=0.117,
        rendement_durable=0.075, volatilite_durable=0.11,
        has_durable_variant=True,
        score_durabilite=3.75, score_additionnalite=4,
        score_disponibilite=3.5, score_retombees_qc=1, score_liquidite=1,
        score_durabilite_durable=4.5, score_additionnalite_durable=5,
        score_disponibilite_durable=4, score_retombees_qc_durable=5,
        score_liquidite_durable=1,
    ),
    "infra_privee": DurableAsset(
        id="infra_privee",
        nom="Infrastructures privées",
        nom_durable="Infrastructures (admissible)",
        rendement=0.0793, volatilite=0.142,
        rendement_durable=0.065, volatilite_durable=0.14,
        has_durable_variant=True,
        score_durabilite=4.25, score_additionnalite=4,
        score_disponibilite=4.5, score_retombees_qc=1, score_liquidite=2,
        score_durabilite_durable=3.75, score_additionnalite_durable=4,
        score_disponibilite_durable=3.5, score_retombees_qc_durable=5,
        score_liquidite_durable=1,
    ),
    "buyout": DurableAsset(
        id="buyout",
        nom="Buyout",
        nom_durable="Buyout (admissible)",
        rendement=0.089, volatilite=0.1951,
        rendement_durable=0.089, volatilite_durable=0.195,
        has_durable_variant=True,
        score_durabilite=4, score_additionnalite=4,
        score_disponibilite=4, score_retombees_qc=2, score_liquidite=2,
        score_durabilite_durable=3.5, score_additionnalite_durable=5,
        score_disponibilite_durable=2, score_retombees_qc_durable=5,
        score_liquidite_durable=1,
    ),
    "capital_risque": DurableAsset(
        id="capital_risque",
        nom="Capital de risque",
        nom_durable="Capital de risque (admissible)",
        rendement=0.077, volatilite=0.1804,
        rendement_durable=0.085, volatilite_durable=0.18,
        has_durable_variant=True,
        score_durabilite=4.5, score_additionnalite=4,
        score_disponibilite=5, score_retombees_qc=2, score_liquidite=1,
        score_durabilite_durable=5, score_additionnalite_durable=5,
        score_disponibilite_durable=5, score_retombees_qc_durable=5,
        score_liquidite_durable=1,
    ),
}

# Ordre canonique des actifs (détermine l'ordre des vecteurs/matrices)
DURABLE_ASSET_ORDER = [
    "obligations_ct", "obligations_univers", "hypotheques",
    "oblig_rendement_eleve", "dettes_emergentes",
    "actions_cdn", "actions_globales", "actions_petite_cap", "eqp",
    "fonds_couverture",
    "dette_privee", "immo_prive", "infra_privee", "buyout", "capital_risque",
]

N_DURABLE_ASSETS = len(DURABLE_ASSET_ORDER)

# Noms FR pour l'affichage
DURABLE_ASSET_NAMES_FR = [DURABLE_ASSETS[a].nom for a in DURABLE_ASSET_ORDER]

# Poids par défaut (équipondéré sur actifs liquides, fraction réduite pour privés)
DURABLE_DEFAULT_WEIGHTS = np.array([
    0.05, 0.12, 0.05, 0.03, 0.03,   # revenu fixe
    0.10, 0.15, 0.05, 0.05, 0.05,   # actions + couverture
    0.10, 0.08, 0.08, 0.04, 0.02,   # privés
])
DURABLE_DEFAULT_WEIGHTS /= DURABLE_DEFAULT_WEIGHTS.sum()

# Contraintes par défaut
DURABLE_MIN_WEIGHTS = np.array([
    0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0,
])
DURABLE_MAX_WEIGHTS = np.array([
    0.10, 0.40, 0.15, 0.10, 0.10,
    0.30, 0.40, 0.10, 0.15, 0.15,
    0.20, 0.15, 0.20, 0.15, 0.15,
])

# Matrice de corrélation 15×15
# Source: Corrélations (Aon 31 déc 2024), mapping manuel vers 15 actifs
# Ordre: [oblig_ct, oblig_univ, hypo, oblig_re, dettes_em,
#          act_cdn, act_glob, act_pc, eqp, fonds_couv,
#          dette_priv, immo_priv, infra_priv, buyout, cap_risque]
_CORR = np.array([
    # 0obCT  1obUN  2hypo  3obRE  4dEm   5acCDN 6acGL  7acPC  8EQP   9couv  10dPriv 11immoP 12infra 13buy  14capR
    [ 1.00,  0.90,  0.60,  0.20,  0.20,  0.00,  0.00,  0.00,  0.00,  0.10,  0.60,  0.10,  0.40,  0.00,  0.00],  # obCT
    [ 0.90,  1.00,  0.55,  0.10,  0.20,  0.00,  0.00,  0.00,  0.00,  0.10,  0.70,  0.00,  0.40,  0.00,  0.00],  # obUN
    [ 0.60,  0.55,  1.00,  0.20,  0.30,  0.10,  0.10,  0.10,  0.10,  0.10,  0.40,  0.30,  0.30,  0.10,  0.10],  # hypo
    [ 0.20,  0.10,  0.20,  1.00,  0.90,  0.30,  0.50,  0.40,  0.40,  0.70,  0.30,  0.20,  0.10,  0.40,  0.30],  # obRE
    [ 0.20,  0.20,  0.30,  0.90,  1.00,  0.30,  0.40,  0.30,  0.30,  0.60,  0.40,  0.20,  0.10,  0.30,  0.20],  # dEm
    [ 0.00,  0.00,  0.10,  0.30,  0.30,  1.00,  0.80,  0.70,  0.70,  0.00,  0.10,  0.40,  0.10,  0.70,  0.70],  # acCDN
    [ 0.00,  0.00,  0.10,  0.50,  0.40,  0.80,  1.00,  0.80,  0.70,  0.00,  0.10,  0.40,  0.10,  0.70,  0.70],  # acGL
    [ 0.00,  0.00,  0.10,  0.40,  0.30,  0.70,  0.80,  1.00,  0.50,  0.00,  0.10,  0.30,  0.10,  0.60,  0.60],  # acPC
    [ 0.00,  0.00,  0.10,  0.40,  0.30,  0.70,  0.70,  0.50,  1.00,  0.10,  0.10,  0.30,  0.10,  0.85,  0.90],  # EQP
    [ 0.10,  0.10,  0.10,  0.70,  0.60,  0.00,  0.00,  0.00,  0.10,  1.00,  0.20,  0.30,  0.10,  0.50,  0.40],  # couv
    [ 0.60,  0.70,  0.40,  0.30,  0.40,  0.10,  0.10,  0.10,  0.10,  0.20,  1.00,  0.30,  0.30,  0.10,  0.10],  # dPriv
    [ 0.10,  0.00,  0.30,  0.20,  0.20,  0.40,  0.40,  0.30,  0.30,  0.30,  0.30,  1.00,  0.30,  0.30,  0.30],  # immoP
    [ 0.40,  0.40,  0.30,  0.10,  0.10,  0.10,  0.10,  0.10,  0.10,  0.10,  0.30,  0.30,  1.00,  0.30,  0.20],  # infra
    [ 0.00,  0.00,  0.10,  0.40,  0.30,  0.70,  0.70,  0.60,  0.85,  0.50,  0.10,  0.30,  0.30,  1.00,  0.90],  # buy
    [ 0.00,  0.00,  0.10,  0.30,  0.20,  0.70,  0.70,  0.60,  0.90,  0.40,  0.10,  0.30,  0.20,  0.90,  1.00],  # capR
])
# Enforce symmetry and diagonal = 1
_CORR = (_CORR + _CORR.T) / 2
np.fill_diagonal(_CORR, 1.0)
# Clip to [-1, 1]
_CORR = np.clip(_CORR, -1.0, 1.0)


# ---------------------------------------------------------------------------
# Accessor functions
# ---------------------------------------------------------------------------

def get_durable_returns(use_durable: Optional[Dict[str, bool]] = None) -> np.ndarray:
    """Rendements espérés. use_durable: {asset_id: True} pour variante durable."""
    use_durable = use_durable or {}
    return np.array([
        DURABLE_ASSETS[a].rendement_durable
        if use_durable.get(a) and DURABLE_ASSETS[a].has_durable_variant
        else DURABLE_ASSETS[a].rendement
        for a in DURABLE_ASSET_ORDER
    ])


def get_durable_volatilities(use_durable: Optional[Dict[str, bool]] = None) -> np.ndarray:
    """Volatilités anticipées."""
    use_durable = use_durable or {}
    return np.array([
        DURABLE_ASSETS[a].volatilite_durable
        if use_durable.get(a) and DURABLE_ASSETS[a].has_durable_variant
        else DURABLE_ASSETS[a].volatilite
        for a in DURABLE_ASSET_ORDER
    ])


def get_durable_correlation_matrix() -> np.ndarray:
    """Matrice de corrélation 15×15."""
    return _CORR.copy()


def get_durable_covariance_matrix(
    use_durable: Optional[Dict[str, bool]] = None
) -> np.ndarray:
    """Matrice de covariance = diag(vol) @ corr @ diag(vol)."""
    vols = get_durable_volatilities(use_durable)
    corr = get_durable_correlation_matrix()
    return np.diag(vols) @ corr @ np.diag(vols)


def compute_composite_score(
    weights_port: np.ndarray,
    dim_weights: Dict[str, float],
    use_durable: Optional[Dict[str, bool]] = None,
    custom_scores: Optional[Dict[str, Dict[str, float]]] = None,
) -> float:
    """
    Score composite de durabilité du portefeuille.

    weights_port: poids du portefeuille (15,)
    dim_weights: pondérations des dimensions (somme = 1)
    use_durable: {asset_id: True} pour variante durable
    custom_scores: {asset_id: {dim: score}} pour scores manuels
    """
    S = get_score_matrix(dim_weights, use_durable, custom_scores)
    return float(weights_port @ S[0])  # S[0] = scores selon variante choisie


def get_score_matrix(
    dim_weights: Dict[str, float],
    use_durable: Optional[Dict[str, bool]] = None,
    custom_scores: Optional[Dict[str, Dict[str, float]]] = None,
) -> tuple:
    """
    Retourne (S_current, S_durable) vecteurs 15 composites.

    S_current: score selon la sélection use_durable courante
    S_durable: score si toutes les variantes durables sont activées
    """
    use_durable = use_durable or {}
    custom_scores = custom_scores or {}
    dims = ["durabilite", "additionnalite", "disponibilite", "retombees_qc", "liquidite"]

    S_current = np.zeros(N_DURABLE_ASSETS)
    S_durable = np.zeros(N_DURABLE_ASSETS)

    for i, aid in enumerate(DURABLE_ASSET_ORDER):
        asset = DURABLE_ASSETS[aid]
        w_dur = use_durable.get(aid, False) and asset.has_durable_variant
        cust = custom_scores.get(aid, {})

        for dim in dims:
            dw = dim_weights.get(dim, 0.0)
            if dw == 0.0:
                continue
            # Score actuel
            if dim in cust:
                s_cur = cust[dim]
            elif w_dur:
                s_cur = getattr(asset, f"score_{dim}_durable")
            else:
                s_cur = getattr(asset, f"score_{dim}")
            S_current[i] += dw * s_cur

            # Score si toutes variantes durables
            if dim in cust:
                s_dur = cust[dim]  # custom override takes precedence
            elif asset.has_durable_variant:
                s_dur = getattr(asset, f"score_{dim}_durable")
            else:
                s_dur = getattr(asset, f"score_{dim}")
            S_durable[i] += dw * s_dur

    return S_current, S_durable


# Pondérations de dimensions par défaut
DEFAULT_DIM_WEIGHTS = {
    "durabilite": 0.35,
    "additionnalite": 0.25,
    "disponibilite": 0.15,
    "retombees_qc": 0.15,
    "liquidite": 0.10,
}
```

- [ ] **Step 5: Run tests**

```bash
cd "/Users/macbookprom1max/Library/CloudStorage/OneDrive-FONDACTION(CSN)/Documents/Claude/pension_optimizer"
python -m pytest tests/sustainable/test_config.py -v
```

Expected: All 9 tests pass.

- [ ] **Step 6: Commit**

```bash
git add sustainable/__init__.py sustainable/config.py \
        tests/sustainable/__init__.py tests/sustainable/test_config.py
git commit -m "feat(sustainable): add DurableAsset dataclass and 15-asset config

Source: WG_Catégories_actifs_v4.xlsx (Cartographie_complète_v1 + Anticipations)
Includes score computation and covariance matrix helpers."
git push
```

---

## Chunk 2: Optimizer (`sustainable/optimizer.py`)

### Task 2: DurableResult dataclass + DurableOptimizer

**Files:**
- Create: `sustainable/optimizer.py`
- Create: `tests/sustainable/test_optimizer.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/sustainable/test_optimizer.py
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sustainable.config import (
    DURABLE_ASSET_ORDER, DURABLE_ASSET_NAMES_FR, DURABLE_DEFAULT_WEIGHTS,
    DURABLE_MIN_WEIGHTS, DURABLE_MAX_WEIGHTS,
    get_durable_returns, get_durable_covariance_matrix,
    get_score_matrix, DEFAULT_DIM_WEIGHTS,
)
from sustainable.optimizer import DurableOptimizer, DurableResult


@pytest.fixture
def optimizer():
    mu = get_durable_returns()
    cov = get_durable_covariance_matrix()
    return DurableOptimizer(
        expected_returns=mu,
        cov_matrix=cov,
        risk_free_rate=0.03,
        asset_names=DURABLE_ASSET_NAMES_FR,
        min_weights=DURABLE_MIN_WEIGHTS,
        max_weights=DURABLE_MAX_WEIGHTS,
    )


@pytest.fixture
def sustainability_scores():
    S_cur, _ = get_score_matrix(DEFAULT_DIM_WEIGHTS)
    return S_cur


def test_optimize_durable_lambda_zero_feasible(optimizer, sustainability_scores):
    """λ=0 doit donner un résultat faisable."""
    result = optimizer.optimize_durable(lam=0.0, gamma=2.5,
                                        sustainability_scores=sustainability_scores)
    assert result.status == "optimal"
    assert abs(result.weights.sum() - 1.0) < 1e-4
    assert np.all(result.weights >= -1e-6)


def test_optimize_durable_weights_sum_to_one(optimizer, sustainability_scores):
    result = optimizer.optimize_durable(lam=0.5, gamma=2.5,
                                        sustainability_scores=sustainability_scores)
    assert abs(result.weights.sum() - 1.0) < 1e-4


def test_optimize_durable_higher_lambda_higher_sustainability(optimizer, sustainability_scores):
    """Un λ plus élevé doit donner un score de durabilité plus élevé."""
    r0 = optimizer.optimize_durable(lam=0.0, gamma=2.5,
                                    sustainability_scores=sustainability_scores)
    r1 = optimizer.optimize_durable(lam=2.0, gamma=2.5,
                                    sustainability_scores=sustainability_scores)
    assert r1.sustainability_score >= r0.sustainability_score - 1e-4


def test_durable_result_has_breakdown(optimizer, sustainability_scores):
    result = optimizer.optimize_durable(lam=1.0, gamma=2.5,
                                        sustainability_scores=sustainability_scores)
    assert isinstance(result, DurableResult)
    assert isinstance(result.sustainability_score, float)
    assert 1.0 <= result.sustainability_score <= 5.0
    assert result.lambda_used == 1.0


def test_pareto_frontier_length(optimizer, sustainability_scores):
    frontier = optimizer.pareto_frontier(
        n_points=10, gamma=2.5,
        sustainability_scores=sustainability_scores,
    )
    assert len(frontier) >= 5  # au moins 5 points faisables sur 10


def test_pareto_frontier_monotone_sustainability(optimizer, sustainability_scores):
    """Le score durabilité doit être non-décroissant avec λ."""
    frontier = optimizer.pareto_frontier(
        n_points=15, gamma=2.5,
        sustainability_scores=sustainability_scores,
    )
    scores = [r.sustainability_score for r in frontier]
    for i in range(1, len(scores)):
        assert scores[i] >= scores[i-1] - 0.01, \
            f"Frontier not monotone at index {i}: {scores[i-1]:.3f} -> {scores[i]:.3f}"


def test_optimize_durable_respects_max_weights(optimizer, sustainability_scores):
    result = optimizer.optimize_durable(lam=1.0, gamma=2.5,
                                        sustainability_scores=sustainability_scores)
    assert np.all(result.weights <= DURABLE_MAX_WEIGHTS + 1e-4)
    assert np.all(result.weights >= DURABLE_MIN_WEIGHTS - 1e-4)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python -m pytest tests/sustainable/test_optimizer.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'sustainable.optimizer'`

- [ ] **Step 3: Create `sustainable/optimizer.py`**

```python
# sustainable/optimizer.py
"""
Optimiseur de portefeuille durable bi-critère (Sharpe + durabilité).

Objectif: Maximiser μ'w - (γ/2)·w'Σw + λ·S'w
          sous contraintes de poids et de somme.
"""

import time
import numpy as np
import cvxpy as cp
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.base import BaseOptimizer, OptimizationResult


@dataclass
class DurableResult(OptimizationResult):
    """Résultat d'une optimisation durable."""
    sustainability_score: float = 0.0
    lambda_used: float = 0.0
    gamma_used: float = 2.5
    sustainability_breakdown: Dict[str, float] = field(default_factory=dict)
    variant_used: Dict[str, str] = field(default_factory=dict)  # asset_id → "standard" | "durable"


class DurableOptimizer(BaseOptimizer):
    """
    Optimiseur bi-critère: rendement-risque + durabilité.

    Maximise: μ'w - (γ/2)·w'Σw + λ·S'w
    """

    def optimize(self, **kwargs) -> DurableResult:
        """Délègue à optimize_durable avec des defaults."""
        return self.optimize_durable(**kwargs)

    def optimize_durable(
        self,
        lam: float,
        gamma: float,
        sustainability_scores: np.ndarray,
        constraint_set=None,
        use_durable_map: Optional[Dict[str, bool]] = None,
    ) -> DurableResult:
        """
        Optimise le portefeuille pour un λ donné.

        lam: poids de la durabilité (0 = purement financier)
        gamma: aversion au risque
        sustainability_scores: vecteur (n,) de scores composites par actif
        use_durable_map: {asset_id: True} pour variante durable (propagé dans DurableResult)
        """
        start_time = time.time()
        w = cp.Variable(self.n_assets)

        # Contraintes
        if constraint_set is not None:
            from constraints.manager import ConstraintManager
            cm = ConstraintManager(self.n_assets)
            constraints = cm.to_cvxpy_constraints(w, constraint_set, self.sigma)
        else:
            constraints = [
                cp.sum(w) == 1,
                w >= self.min_weights,
                w <= self.max_weights,
            ]

        # Objectif: maximiser utilité financière + bonus durabilité
        portfolio_return = self.mu @ w
        portfolio_variance = cp.quad_form(w, self.sigma)
        sustainability_term = sustainability_scores @ w

        objective = cp.Maximize(
            portfolio_return - (gamma / 2) * portfolio_variance + lam * sustainability_term
        )

        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.CLARABEL, verbose=False)
        except cp.SolverError:
            return self._make_fallback(lam, gamma, sustainability_scores, start_time)

        if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
            w_opt = np.maximum(w.value, 0)
            w_opt /= w_opt.sum()
            return self._build_durable_result(
                w_opt, "optimal", start_time, lam, gamma, sustainability_scores, use_durable_map
            )
        else:
            return self._make_fallback(lam, gamma, sustainability_scores, start_time, use_durable_map)

    def _make_fallback(self, lam, gamma, sustainability_scores, start_time, use_durable_map=None):
        w_eq = np.ones(self.n_assets) / self.n_assets
        return self._build_durable_result(
            w_eq, "infeasible", start_time, lam, gamma, sustainability_scores, use_durable_map
        )

    def _build_durable_result(
        self, weights, status, start_time, lam, gamma, sustainability_scores,
        use_durable_map=None,
    ) -> DurableResult:
        port_return, port_vol, sharpe = self._compute_portfolio_stats(weights)
        risk_contrib = self._compute_risk_contributions(weights)
        sustain_score = float(sustainability_scores @ weights)
        # Build variant_used dict (asset_name → "standard" | "durable")
        variant_used = {}
        if use_durable_map:
            for i, name in enumerate(self.asset_names):
                variant_used[name] = "durable" if use_durable_map.get(name, False) else "standard"
        return DurableResult(
            weights=weights,
            asset_names=self.asset_names,
            expected_return=port_return,
            volatility=port_vol,
            sharpe_ratio=sharpe,
            risk_contributions=risk_contrib,
            sustainability_score=sustain_score,
            lambda_used=lam,
            gamma_used=gamma,
            variant_used=variant_used,
            status=status,
            solver_time=time.time() - start_time,
        )

    def pareto_frontier(
        self,
        n_points: int = 50,
        gamma: float = 2.5,
        sustainability_scores: Optional[np.ndarray] = None,
        lambda_max: float = 5.0,
        constraint_set=None,
    ) -> List[DurableResult]:
        """
        Calcule n_points sur la frontière Pareto en faisant varier λ.

        Retourne les points faisables triés par score de durabilité croissant.
        """
        if sustainability_scores is None:
            sustainability_scores = np.ones(self.n_assets)

        lambdas = np.linspace(0, lambda_max, n_points)
        results = []
        for lam in lambdas:
            r = self.optimize_durable(lam, gamma, sustainability_scores, constraint_set)
            if r.status == "optimal":
                results.append(r)

        # Trier par score de durabilité
        results.sort(key=lambda r: r.sustainability_score)
        return results
```

- [ ] **Step 4: Run tests**

```bash
python -m pytest tests/sustainable/test_optimizer.py -v
```

Expected: All 7 tests pass.

- [ ] **Step 5: Commit**

```bash
git add sustainable/optimizer.py tests/sustainable/test_optimizer.py
git commit -m "feat(sustainable): add DurableOptimizer with Pareto frontier

Bi-criteria objective: maximize μ'w - (γ/2)·w'Σw + λ·S'w
Pareto frontier via λ sweep, sorted by sustainability score."
git push
```

---

## Chunk 3: Pages Univers + Scores

### Task 3: `page_durable_univers.py`

**Files:**
- Create: `pages/page_durable_univers.py`

- [ ] **Step 1: Create the page**

```python
# pages/page_durable_univers.py
"""
Configuration de l'univers d'actifs durable.
- Activation/désactivation des variantes durables par classe
- Pondération des dimensions de durabilité
- Paramètre d'aversion au risque γ
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sustainable.config import (
    DURABLE_ASSETS, DURABLE_ASSET_ORDER, DURABLE_ASSET_NAMES_FR,
    DURABLE_DEFAULT_WEIGHTS, DURABLE_MIN_WEIGHTS, DURABLE_MAX_WEIGHTS,
    DEFAULT_DIM_WEIGHTS, get_score_matrix, N_DURABLE_ASSETS,
)


def _init_session():
    if "durable_universe" not in st.session_state:
        st.session_state.durable_universe = {
            aid: {"active": True, "use_durable": False}
            for aid in DURABLE_ASSET_ORDER
        }
    if "durable_dim_weights" not in st.session_state:
        st.session_state.durable_dim_weights = DEFAULT_DIM_WEIGHTS.copy()
    if "durable_gamma" not in st.session_state:
        st.session_state.durable_gamma = 2.5
    if "durable_min_weights" not in st.session_state:
        st.session_state.durable_min_weights = DURABLE_MIN_WEIGHTS.copy()
    if "durable_max_weights" not in st.session_state:
        st.session_state.durable_max_weights = DURABLE_MAX_WEIGHTS.copy()


def render():
    st.title("🌍 Univers durable")
    st.caption("Configurez les classes d'actifs, les variantes durables et les priorités de durabilité.")
    _init_session()

    # --- Section 1: Univers d'actifs ---
    st.markdown("### Classes d'actifs")

    categories = {
        "Revenu fixe": ["obligations_ct", "obligations_univers", "hypotheques",
                        "oblig_rendement_eleve", "dettes_emergentes"],
        "Actions & liquidités": ["actions_cdn", "actions_globales", "actions_petite_cap",
                                 "eqp", "fonds_couverture"],
        "Actifs privés": ["dette_privee", "immo_prive", "infra_privee", "buyout", "capital_risque"],
    }

    universe = st.session_state.durable_universe

    for cat_name, asset_ids in categories.items():
        st.markdown(f"**{cat_name}**")
        for aid in asset_ids:
            asset = DURABLE_ASSETS[aid]
            col_check, col_name, col_toggle, col_label = st.columns([0.5, 3, 1.5, 3])

            with col_check:
                active = st.checkbox("", value=universe[aid]["active"],
                                     key=f"active_{aid}", label_visibility="collapsed")
            with col_name:
                st.write(asset.nom)
            with col_toggle:
                if asset.has_durable_variant and active:
                    use_dur = st.toggle("Durable", value=universe[aid]["use_durable"],
                                        key=f"dur_{aid}")
                else:
                    use_dur = False
                    if not asset.has_durable_variant:
                        st.caption("—")
            with col_label:
                if asset.has_durable_variant and active and use_dur:
                    st.caption(f"↪ {asset.nom_durable}")
                elif not active:
                    st.caption("*(exclu)*")

            universe[aid] = {"active": active, "use_durable": use_dur and asset.has_durable_variant}

    n_active = sum(1 for v in universe.values() if v["active"])
    st.info(f"{n_active} classe(s) d'actifs active(s) | "
            f"{sum(1 for v in universe.values() if v['use_durable'])} variante(s) durable(s) activée(s)")

    st.divider()

    # --- Section 2: Pondérations des dimensions ---
    st.markdown("### Priorités de durabilité")
    st.caption("Définissez l'importance relative de chaque dimension. Total = 100%.")

    col1, col2 = st.columns(2)
    dims = {
        "durabilite": "Durabilité",
        "additionnalite": "Additionnalité",
        "disponibilite": "Disponibilité",
        "retombees_qc": "Retombées Québec",
        "liquidite": "Liquidité",
    }
    dw = st.session_state.durable_dim_weights
    new_dw = {}

    with col1:
        for k, label in list(dims.items())[:3]:
            new_dw[k] = st.slider(label, 0, 100, int(dw[k]*100), 5, key=f"dw_{k}") / 100
    with col2:
        for k, label in list(dims.items())[3:]:
            new_dw[k] = st.slider(label, 0, 100, int(dw[k]*100), 5, key=f"dw_{k}") / 100

    total_dw = sum(new_dw.values())
    if abs(total_dw - 1.0) > 0.01:
        st.warning(f"Total: {total_dw*100:.0f}% — doit être 100%. Normalisation automatique.")
        if total_dw > 0:
            new_dw = {k: v/total_dw for k, v in new_dw.items()}

    # Graphique radar
    categories_radar = list(dims.values())
    values_radar = [new_dw[k] * 100 for k in dims]
    fig_radar = go.Figure(go.Scatterpolar(
        r=values_radar + [values_radar[0]],
        theta=categories_radar + [categories_radar[0]],
        fill="toself", name="Priorités",
        line_color="#2ca02c",
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 60])),
        height=300, margin=dict(t=20, b=20),
        showlegend=False,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.divider()

    # --- Section 3: Aversion au risque ---
    st.markdown("### Paramètre financier")
    gamma = st.slider(
        "Aversion au risque (γ) — plus élevé = plus conservateur",
        1.0, 6.0, st.session_state.durable_gamma, 0.1,
    )

    st.divider()

    if st.button("✅ Appliquer la configuration", type="primary", use_container_width=True):
        st.session_state.durable_universe = universe
        st.session_state.durable_dim_weights = new_dw
        st.session_state.durable_gamma = gamma
        # Invalider les résultats existants
        st.session_state.pop("durable_frontier", None)
        st.session_state.pop("durable_result", None)
        st.success("Configuration enregistrée. La frontière sera recalculée.")

    # Aperçu des scores courants
    with st.expander("Aperçu des scores composites actuels"):
        use_durable_map = {
            aid: v["use_durable"] for aid, v in universe.items() if v["active"]
        }
        S_cur, S_dur = get_score_matrix(new_dw, use_durable_map)
        active_ids = [aid for aid in DURABLE_ASSET_ORDER if universe[aid]["active"]]
        active_names = [DURABLE_ASSETS[aid].nom_durable
                        if universe[aid]["use_durable"] and DURABLE_ASSETS[aid].has_durable_variant
                        else DURABLE_ASSETS[aid].nom
                        for aid in active_ids]
        active_scores = [S_cur[DURABLE_ASSET_ORDER.index(aid)] for aid in active_ids]

        preview_df = pd.DataFrame({
            "Classe d'actifs": active_names,
            "Score composite": active_scores,
        }).sort_values("Score composite", ascending=True)
        st.dataframe(preview_df.style.format({"Score composite": "{:.2f}"}),
                     use_container_width=True, hide_index=True)


render()
```

- [ ] **Step 2: Verify page loads without error**

```bash
cd "/Users/macbookprom1max/Library/CloudStorage/OneDrive-FONDACTION(CSN)/Documents/Claude/pension_optimizer"
python -c "import ast; ast.parse(open('pages/page_durable_univers.py').read()); print('Syntax OK')"
```

- [ ] **Step 3: Commit**

```bash
git add pages/page_durable_univers.py
git commit -m "feat(sustainable): add Univers durable page

Asset universe configuration with durable variant toggles,
dimension weight sliders, radar chart, and risk aversion parameter."
git push
```

---

### Task 4: `page_durable_scores.py`

**Files:**
- Create: `pages/page_durable_scores.py`

- [ ] **Step 1: Create the page**

```python
# pages/page_durable_scores.py
"""
Tableau éditable des scores de durabilité par classe d'actifs.
Valeurs par défaut issues du fichier Excel WG_Catégories_actifs_v4.xlsx.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sustainable.config import (
    DURABLE_ASSETS, DURABLE_ASSET_ORDER, DEFAULT_DIM_WEIGHTS, get_score_matrix,
)

DIMS = ["durabilite", "additionnalite", "disponibilite", "retombees_qc", "liquidite"]
DIM_LABELS = {
    "durabilite": "Durabilité",
    "additionnalite": "Additionnalité",
    "disponibilite": "Disponibilité",
    "retombees_qc": "Retombées Qc",
    "liquidite": "Liquidité",
}


def _default_scores_df(use_durable_map: dict) -> pd.DataFrame:
    rows = []
    for aid in DURABLE_ASSET_ORDER:
        asset = DURABLE_ASSETS[aid]
        use_dur = use_durable_map.get(aid, False) and asset.has_durable_variant
        row = {
            "id": aid,
            "Classe d'actifs": asset.nom_durable if use_dur else asset.nom,
        }
        for dim in DIMS:
            key = f"score_{dim}_durable" if use_dur else f"score_{dim}"
            row[DIM_LABELS[dim]] = float(getattr(asset, key))
        rows.append(row)
    return pd.DataFrame(rows)


def render():
    st.title("📊 Scores de durabilité")
    st.caption("Modifiez les scores par classe d'actifs (1 = faible, 5 = élevé). "
               "Les valeurs par défaut proviennent du fichier de cartographie Excel.")

    universe = st.session_state.get("durable_universe", {})
    use_durable_map = {aid: v.get("use_durable", False)
                       for aid, v in universe.items()}
    dim_weights = st.session_state.get("durable_dim_weights", DEFAULT_DIM_WEIGHTS)

    # Charger ou initialiser les scores
    if "durable_scores" not in st.session_state:
        st.session_state.durable_scores = _default_scores_df(use_durable_map)

    scores_df = st.session_state.durable_scores.copy()

    # Éditeur
    st.markdown("### Scores par classe d'actifs *(éditables)*")
    dim_cols = list(DIM_LABELS.values())
    edited = st.data_editor(
        scores_df[["Classe d'actifs"] + dim_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            col: st.column_config.NumberColumn(col, min_value=0.0, max_value=5.0,
                                               format="%.1f", step=0.25)
            for col in dim_cols
        },
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Réinitialiser aux valeurs Excel", use_container_width=True):
            st.session_state.durable_scores = _default_scores_df(use_durable_map)
            st.rerun()
    with col2:
        if st.button("✅ Appliquer les scores", type="primary", use_container_width=True):
            # Valider les scores
            valid = True
            for col in dim_cols:
                if edited[col].isnull().any() or (edited[col] < 0).any() or (edited[col] > 5).any():
                    st.error(f"Scores invalides dans '{col}': doit être entre 0 et 5.")
                    valid = False
            if valid:
                updated = scores_df.copy()
                updated[dim_cols] = edited[dim_cols]
                st.session_state.durable_scores = updated
                # Invalider résultats
                st.session_state.pop("durable_frontier", None)
                st.session_state.pop("durable_result", None)
                st.success("Scores enregistrés.")

    st.divider()

    # Calcul et affichage du score composite
    st.markdown("### Score composite par classe d'actifs")
    st.caption(f"Pondérations actives : "
               + " | ".join([f"{DIM_LABELS[k]}: {v*100:.0f}%" for k, v in dim_weights.items()]))

    current_df = st.session_state.durable_scores
    custom_scores = {}
    for _, row in current_df.iterrows():
        aid = row["id"]
        custom_scores[aid] = {dim: row[DIM_LABELS[dim]] for dim in DIMS}

    S_cur, _ = get_score_matrix(dim_weights, use_durable_map, custom_scores)

    composite_df = pd.DataFrame({
        "Classe d'actifs": current_df["Classe d'actifs"].values,
        "Score composite": S_cur,
    }).sort_values("Score composite")

    fig = go.Figure(go.Bar(
        x=composite_df["Score composite"],
        y=composite_df["Classe d'actifs"],
        orientation="h",
        marker_color=composite_df["Score composite"].map(
            lambda s: "#d62728" if s < 2 else "#ff7f0e" if s < 3 else "#2ca02c"
        ),
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 5], title="Score composite (1–5)"),
        height=450, margin=dict(t=20, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


render()
```

- [ ] **Step 2: Verify syntax**

```bash
python -c "import ast; ast.parse(open('pages/page_durable_scores.py').read()); print('Syntax OK')"
```

- [ ] **Step 3: Commit**

```bash
git add pages/page_durable_scores.py
git commit -m "feat(sustainable): add Scores de durabilité page

Editable data_editor with Excel defaults, composite score bar chart,
reset to defaults and apply buttons."
git push
```

---

## Chunk 4: Pages Frontière + Optimisation + Rapport

### Task 5: `page_durable_frontier.py`

**Files:**
- Create: `pages/page_durable_frontier.py`

- [ ] **Step 1: Create the page**

```python
# pages/page_durable_frontier.py
"""
Frontière Pareto interactive : Sharpe ↔ Durabilité.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sustainable.config import (
    DURABLE_ASSETS, DURABLE_ASSET_ORDER, DURABLE_ASSET_NAMES_FR,
    DURABLE_MIN_WEIGHTS, DURABLE_MAX_WEIGHTS, DEFAULT_DIM_WEIGHTS,
    get_durable_returns, get_durable_covariance_matrix,
    get_score_matrix, DURABLE_DEFAULT_WEIGHTS,
)
from sustainable.optimizer import DurableOptimizer


def _get_active_config():
    universe = st.session_state.get("durable_universe", {})
    active_ids = [aid for aid in DURABLE_ASSET_ORDER
                  if universe.get(aid, {}).get("active", True)]
    use_durable_map = {aid: universe.get(aid, {}).get("use_durable", False)
                       for aid in active_ids}
    return active_ids, use_durable_map


def _build_optimizer(active_ids, use_durable_map):
    mu = get_durable_returns(use_durable_map)
    cov = get_durable_covariance_matrix(use_durable_map)
    n = len(DURABLE_ASSET_ORDER)
    # Filter to active indices
    active_idx = [DURABLE_ASSET_ORDER.index(aid) for aid in active_ids]
    mu_active = mu[active_idx]
    cov_active = cov[np.ix_(active_idx, active_idx)]
    min_w = np.array([DURABLE_MIN_WEIGHTS[i] for i in active_idx])
    max_w = np.array([DURABLE_MAX_WEIGHTS[i] for i in active_idx])
    names = [DURABLE_ASSETS[aid].nom_durable
             if use_durable_map.get(aid) and DURABLE_ASSETS[aid].has_durable_variant
             else DURABLE_ASSETS[aid].nom
             for aid in active_ids]

    # Normaliser max_w pour qu'ils puissent sommer à 1
    if min_w.sum() > 1.0:
        min_w = min_w / min_w.sum()
    return DurableOptimizer(
        expected_returns=mu_active, cov_matrix=cov_active,
        risk_free_rate=st.session_state.get("pension_config",
                                            type("C", (), {"taux_sans_risque": 0.03})()).taux_sans_risque
                       if hasattr(st.session_state.get("pension_config", None), "taux_sans_risque") else 0.03,
        asset_names=names, min_weights=min_w, max_weights=max_w,
    ), active_idx


def render():
    st.title("📈 Frontière durable")
    st.caption("Frontière Pareto entre performance financière et durabilité. "
               "Chaque point représente un portefeuille optimal pour un niveau de pondération λ.")

    active_ids, use_durable_map = _get_active_config()
    if len(active_ids) < 2:
        st.error("Activez au moins 2 classes d'actifs dans 'Univers durable'.")
        return

    dim_weights = st.session_state.get("durable_dim_weights", DEFAULT_DIM_WEIGHTS)
    gamma = st.session_state.get("durable_gamma", 2.5)

    # Scores de durabilité personnalisés
    custom_scores = {}
    if "durable_scores" in st.session_state:
        for _, row in st.session_state.durable_scores.iterrows():
            aid = row["id"]
            custom_scores[aid] = {
                dim: row[label] for dim, label in {
                    "durabilite": "Durabilité", "additionnalite": "Additionnalité",
                    "disponibilite": "Disponibilité", "retombees_qc": "Retombées Qc",
                    "liquidite": "Liquidité",
                }.items()
            }

    S_cur, _ = get_score_matrix(dim_weights, use_durable_map, custom_scores)
    active_idx = [DURABLE_ASSET_ORDER.index(aid) for aid in active_ids]
    S_active = S_cur[active_idx]

    col1, col2, col3 = st.columns(3)
    with col1:
        n_points = st.slider("Nombre de points", 20, 100, 50, 10)
    with col2:
        lambda_max = st.slider("λ maximum", 1.0, 10.0, 5.0, 0.5)
    with col3:
        st.metric("γ actuel", f"{gamma:.1f}")

    if st.button("🚀 Calculer la frontière Pareto", type="primary", use_container_width=True):
        with st.spinner("Calcul de la frontière en cours..."):
            optimizer, _ = _build_optimizer(active_ids, use_durable_map)
            frontier = optimizer.pareto_frontier(
                n_points=n_points, gamma=gamma,
                sustainability_scores=S_active,
                lambda_max=lambda_max,
            )
            if len(frontier) < 3:
                st.error("Moins de 3 points faisables. Vérifiez les contraintes.")
                return
            st.session_state.durable_frontier = frontier
            st.session_state.durable_frontier_ids = active_ids
            st.success(f"{len(frontier)} points calculés.")

    if "durable_frontier" not in st.session_state:
        st.info("Cliquez sur 'Calculer la frontière Pareto' pour afficher les résultats.")
        return

    frontier = st.session_state.durable_frontier
    scores = [r.sustainability_score for r in frontier]
    sharpes = [r.sharpe_ratio for r in frontier]
    returns = [r.expected_return for r in frontier]
    vols = [r.volatility for r in frontier]
    lambdas = [r.lambda_used for r in frontier]

    # Scatter plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=scores, y=sharpes,
        mode="markers+lines",
        marker=dict(size=8, color=lambdas, colorscale="Viridis",
                    showscale=True, colorbar=dict(title="λ")),
        line=dict(color="rgba(100,100,200,0.3)", width=1),
        text=[f"λ={l:.2f}<br>Rend: {r:.2%}<br>Vol: {v:.2%}<br>Score: {s:.2f}"
              for l, r, v, s in zip(lambdas, returns, vols, scores)],
        hoverinfo="text", name="Frontière Pareto",
    ))

    # Portefeuille actuel comme référence
    current_weights = st.session_state.get("current_weights")
    if current_weights is not None:
        active_w = np.array([current_weights[i] if i < len(current_weights) else 0
                             for i in active_idx])
        if active_w.sum() > 1e-6:
            active_w /= active_w.sum()
        cur_sharpe = float((active_w @ get_durable_returns(use_durable_map)[active_idx]
                           - 0.03)
                           / max(np.sqrt(active_w @ get_durable_covariance_matrix(use_durable_map)[np.ix_(active_idx, active_idx)] @ active_w), 1e-10))
        cur_score = float(S_active @ active_w)
        fig.add_trace(go.Scatter(
            x=[cur_score], y=[cur_sharpe],
            mode="markers", marker=dict(symbol="star", size=16, color="red"),
            name="Portefeuille actuel",
        ))

    fig.update_layout(
        title="Frontière Pareto : Durabilité ↔ Ratio de Sharpe",
        xaxis_title="Score de durabilité du portefeuille",
        yaxis_title="Ratio de Sharpe",
        height=480, margin=dict(t=60, b=60),
        hovermode="closest",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Curseur pour sélectionner un point
    st.markdown("### Sélectionner un point sur la frontière")
    idx_sel = st.slider("Index du point (par score croissant)", 0, len(frontier)-1,
                        len(frontier)//2)
    selected = frontier[idx_sel]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rendement", f"{selected.expected_return:.2%}")
    col2.metric("Volatilité", f"{selected.volatility:.2%}")
    col3.metric("Sharpe", f"{selected.sharpe_ratio:.3f}")
    col4.metric("Score durabilité", f"{selected.sustainability_score:.2f}")

    if st.button("➡ Utiliser ce portefeuille dans Optimisation durable"):
        st.session_state.durable_result = selected
        st.session_state.durable_lambda = selected.lambda_used
        st.success(f"Portefeuille sélectionné (λ={selected.lambda_used:.2f}). "
                   "Ouvrez la page 'Optimisation durable'.")


render()
```

- [ ] **Step 2: Verify syntax**

```bash
python -c "import ast; ast.parse(open('pages/page_durable_frontier.py').read()); print('Syntax OK')"
```

- [ ] **Step 3: Commit**

```bash
git add pages/page_durable_frontier.py
git commit -m "feat(sustainable): add Frontière durable page

Pareto frontier scatter plot (sustainability vs Sharpe), λ sweep,
current portfolio reference point, point selector."
git push
```

---

### Task 6: `page_durable_optimization.py`

**Files:**
- Create: `pages/page_durable_optimization.py`

- [ ] **Step 1: Create the page**

```python
# pages/page_durable_optimization.py
"""
Optimisation durable : lancer une optimisation, comparer les portefeuilles.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sustainable.config import (
    DURABLE_ASSETS, DURABLE_ASSET_ORDER, DURABLE_MIN_WEIGHTS, DURABLE_MAX_WEIGHTS,
    DEFAULT_DIM_WEIGHTS, get_durable_returns, get_durable_covariance_matrix,
    get_score_matrix, DURABLE_DEFAULT_WEIGHTS, N_DURABLE_ASSETS,
)
from sustainable.optimizer import DurableOptimizer, DurableResult

DIMS = ["durabilite", "additionnalite", "disponibilite", "retombees_qc", "liquidite"]
DIM_LABELS = ["Durabilité", "Additionnalité", "Disponibilité", "Retombées Qc", "Liquidité"]


def _get_scores_and_optimizer():
    universe = st.session_state.get("durable_universe", {})
    active_ids = [aid for aid in DURABLE_ASSET_ORDER
                  if universe.get(aid, {}).get("active", True)]
    use_durable_map = {aid: universe.get(aid, {}).get("use_durable", False)
                       for aid in active_ids}
    dim_weights = st.session_state.get("durable_dim_weights", DEFAULT_DIM_WEIGHTS)
    gamma = st.session_state.get("durable_gamma", 2.5)

    custom_scores = {}
    if "durable_scores" in st.session_state:
        for _, row in st.session_state.durable_scores.iterrows():
            aid = row["id"]
            custom_scores[aid] = {dim: row[lbl] for dim, lbl in zip(DIMS, DIM_LABELS)}

    S_cur, _ = get_score_matrix(dim_weights, use_durable_map, custom_scores)
    active_idx = [DURABLE_ASSET_ORDER.index(aid) for aid in active_ids]
    S_active = S_cur[active_idx]

    mu = get_durable_returns(use_durable_map)[active_idx]
    cov = get_durable_covariance_matrix(use_durable_map)[np.ix_(active_idx, active_idx)]
    min_w = np.array([DURABLE_MIN_WEIGHTS[i] for i in active_idx])
    max_w = np.array([DURABLE_MAX_WEIGHTS[i] for i in active_idx])
    names = [DURABLE_ASSETS[aid].nom_durable
             if use_durable_map.get(aid) and DURABLE_ASSETS[aid].has_durable_variant
             else DURABLE_ASSETS[aid].nom
             for aid in active_ids]

    opt = DurableOptimizer(
        expected_returns=mu, cov_matrix=cov, risk_free_rate=0.03,
        asset_names=names, min_weights=min_w, max_weights=max_w,
    )
    return opt, S_active, active_ids, active_idx, names, gamma


def render():
    st.title("⚙️ Optimisation durable")
    st.caption("Lancez l'optimisation et comparez avec le portefeuille actuel.")

    lam_default = float(st.session_state.get("durable_lambda", 1.0))
    lam = st.slider("Poids durabilité (λ)", 0.0, 10.0, lam_default, 0.1)
    st.session_state.durable_lambda = lam

    if st.button("🚀 Lancer l'optimisation durable", type="primary", use_container_width=True):
        with st.spinner("Optimisation en cours..."):
            try:
                opt, S_active, active_ids, active_idx, names, gamma = _get_scores_and_optimizer()

                # Résultat λ choisi
                result = opt.optimize_durable(lam=lam, gamma=gamma, sustainability_scores=S_active)
                # Résultat purement financier (λ=0)
                result_fin = opt.optimize_durable(lam=0.0, gamma=gamma, sustainability_scores=S_active)

                st.session_state.durable_result = result
                st.session_state.durable_result_fin = result_fin
                st.session_state.durable_active_ids = active_ids
                st.session_state.durable_active_idx = active_idx

                if result.status == "optimal":
                    st.success(
                        f"Optimisation réussie ({result.solver_time:.2f}s) | "
                        f"Rendement: {result.expected_return:.2%} | "
                        f"Volatilité: {result.volatility:.2%} | "
                        f"Sharpe: {result.sharpe_ratio:.3f} | "
                        f"Score durabilité: {result.sustainability_score:.2f}"
                    )
                else:
                    st.warning(f"Statut: {result.status}. Portefeuille équipondéré utilisé.")
            except Exception as e:
                st.error(f"Erreur: {e}")
                return

    if "durable_result" not in st.session_state:
        st.info("Configurez l'univers et cliquez sur 'Lancer l'optimisation'.")
        return

    result = st.session_state.durable_result
    result_fin = st.session_state.get("durable_result_fin", result)
    active_ids = st.session_state.get("durable_active_ids", DURABLE_ASSET_ORDER)
    names = [DURABLE_ASSETS[aid].nom for aid in active_ids]

    st.divider()
    st.markdown("## Résultats comparatifs")

    # Tableau 3 colonnes
    metrics_labels = ["Rendement attendu", "Volatilité", "Ratio de Sharpe", "Score durabilité"]
    col_h1, col_h2, col_h3 = st.columns(3)
    col_h1.markdown("**Métrique**")
    col_h2.markdown("**Optimal financier (λ=0)**")
    col_h3.markdown(f"**Optimal durable (λ={lam:.1f})**")
    for label, v_fin, v_dur in zip(
        metrics_labels,
        [result_fin.expected_return, result_fin.volatility,
         result_fin.sharpe_ratio, result_fin.sustainability_score],
        [result.expected_return, result.volatility,
         result.sharpe_ratio, result.sustainability_score],
    ):
        c1, c2, c3 = st.columns(3)
        c1.write(label)
        fmt = "{:.2%}" if "%" in label.lower() or label in ("Rendement attendu", "Volatilité") else "{:.3f}"
        c2.write(f"{v_fin:.2%}" if label in ("Rendement attendu", "Volatilité") else f"{v_fin:.3f}")
        c3.write(f"{v_dur:.2%}" if label in ("Rendement attendu", "Volatilité") else f"{v_dur:.3f}")

    st.divider()

    tab1, tab2, tab3 = st.tabs(["Allocations", "Score par dimension", "Tableau détaillé"])

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Optimal financier", x=names, y=result_fin.weights * 100,
                             marker_color="#1f77b4"))
        fig.add_trace(go.Bar(name="Optimal durable", x=names, y=result.weights * 100,
                             marker_color="#2ca02c"))
        fig.update_layout(barmode="group", xaxis_tickangle=-45, yaxis_title="Poids (%)",
                          height=400, margin=dict(t=20, b=100))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        dim_weights = st.session_state.get("durable_dim_weights", DEFAULT_DIM_WEIGHTS)
        # Score par dimension pour le portefeuille durable
        universe = st.session_state.get("durable_universe", {})
        use_durable_map = {aid: universe.get(aid, {}).get("use_durable", False)
                           for aid in active_ids}
        custom_scores = {}
        if "durable_scores" in st.session_state:
            for _, row in st.session_state.durable_scores.iterrows():
                custom_scores[row["id"]] = {
                    dim: row[lbl] for dim, lbl in zip(DIMS, DIM_LABELS)
                }

        dim_scores = {}
        for dim, lbl in zip(DIMS, DIM_LABELS):
            dim_w = {d: (1.0 if d == dim else 0.0) for d in DIMS}
            S_dim, _ = get_score_matrix(dim_w, use_durable_map, custom_scores)
            active_idx = [DURABLE_ASSET_ORDER.index(aid) for aid in active_ids]
            score_val = float(result.weights @ S_dim[active_idx])
            dim_scores[lbl] = score_val

        fig_dim = go.Figure(go.Bar(
            x=list(dim_scores.values()),
            y=list(dim_scores.keys()),
            orientation="h",
            marker_color="#2ca02c",
        ))
        fig_dim.update_layout(xaxis=dict(range=[0, 5], title="Score (1–5)"),
                               height=300, margin=dict(t=20, b=40))
        st.plotly_chart(fig_dim, use_container_width=True)

    with tab3:
        detail_df = pd.DataFrame({
            "Classe d'actifs": names,
            "Optimal financier (%)": result_fin.weights * 100,
            "Optimal durable (%)": result.weights * 100,
            "Écart (pp)": (result.weights - result_fin.weights) * 100,
        })
        st.dataframe(
            detail_df.style.format({
                "Optimal financier (%)": "{:.1f}",
                "Optimal durable (%)": "{:.1f}",
                "Écart (pp)": "{:+.1f}",
            }),
            use_container_width=True, hide_index=True,
        )

    st.divider()
    if st.button("📥 Adopter comme portefeuille actuel (outil principal)"):
        # Remap weights to full 15-asset vector
        full_weights = np.zeros(N_DURABLE_ASSETS)
        for i, aid in enumerate(active_ids):
            full_weights[DURABLE_ASSET_ORDER.index(aid)] = result.weights[i]
        st.session_state.durable_adopted_weights = full_weights
        st.success("Poids enregistrés dans 'durable_adopted_weights'.")


render()
```

- [ ] **Step 2: Verify syntax**

```bash
python -c "import ast; ast.parse(open('pages/page_durable_optimization.py').read()); print('Syntax OK')"
```

- [ ] **Step 3: Commit**

```bash
git add pages/page_durable_optimization.py
git commit -m "feat(sustainable): add Optimisation durable page

3-column comparison (financial vs durable), allocation bar chart,
dimension score breakdown, adopt portfolio button."
git push
```

---

### Task 7: `page_durable_rapport.py`

**Files:**
- Create: `pages/page_durable_rapport.py`

- [ ] **Step 1: Create the page**

```python
# pages/page_durable_rapport.py
"""
Rapport de synthèse de l'optimisation durable.
Export CSV. Prêt pour présentation au conseil.
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sustainable.config import (
    DURABLE_ASSETS, DURABLE_ASSET_ORDER, DEFAULT_DIM_WEIGHTS,
    get_score_matrix,
)

DIMS = ["durabilite", "additionnalite", "disponibilite", "retombees_qc", "liquidite"]
DIM_LABELS = ["Durabilité", "Additionnalité", "Disponibilité", "Retombées Qc", "Liquidité"]


def render():
    st.title("📄 Rapport durable")
    st.caption("Résumé de l'optimisation durable pour présentation au conseil d'administration.")

    result = st.session_state.get("durable_result")
    if result is None:
        st.info("Lancez une optimisation dans la page 'Optimisation durable' d'abord.")
        return

    active_ids = st.session_state.get("durable_active_ids", DURABLE_ASSET_ORDER)
    dim_weights = st.session_state.get("durable_dim_weights", DEFAULT_DIM_WEIGHTS)
    gamma = st.session_state.get("durable_gamma", 2.5)
    lam = st.session_state.get("durable_lambda", 1.0)
    universe = st.session_state.get("durable_universe", {})
    use_durable_map = {aid: universe.get(aid, {}).get("use_durable", False)
                       for aid in active_ids}
    names = [DURABLE_ASSETS[aid].nom_durable
             if use_durable_map.get(aid) and DURABLE_ASSETS[aid].has_durable_variant
             else DURABLE_ASSETS[aid].nom
             for aid in active_ids]

    # --- Section 1: Métriques financières ---
    st.markdown("## 1. Métriques financières")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rendement attendu", f"{result.expected_return:.2%}")
    col2.metric("Volatilité", f"{result.volatility:.2%}")
    col3.metric("Ratio de Sharpe", f"{result.sharpe_ratio:.3f}")
    col4.metric("Score durabilité", f"{result.sustainability_score:.2f} / 5")

    # --- Section 2: Score de durabilité ---
    st.markdown("## 2. Score de durabilité")
    custom_scores = {}
    if "durable_scores" in st.session_state:
        for _, row in st.session_state.durable_scores.iterrows():
            custom_scores[row["id"]] = {dim: row[lbl] for dim, lbl in zip(DIMS, DIM_LABELS)}

    dim_score_rows = []
    for dim, lbl in zip(DIMS, DIM_LABELS):
        dim_w = {d: (1.0 if d == dim else 0.0) for d in DIMS}
        S_dim, _ = get_score_matrix(dim_w, use_durable_map, custom_scores)
        active_idx = [DURABLE_ASSET_ORDER.index(aid) for aid in active_ids]
        score_val = float(result.weights @ S_dim[active_idx])
        dim_score_rows.append({
            "Dimension": lbl,
            "Pondération (%)": dim_weights.get(dim, 0) * 100,
            "Score du portefeuille": score_val,
        })

    dim_df = pd.DataFrame(dim_score_rows)
    st.dataframe(
        dim_df.style.format({
            "Pondération (%)": "{:.0f}",
            "Score du portefeuille": "{:.2f}",
        }),
        use_container_width=True, hide_index=True,
    )

    # --- Section 3: Allocation ---
    st.markdown("## 3. Allocation du portefeuille")
    alloc_df = pd.DataFrame({
        "Classe d'actifs": names,
        "Poids (%)": result.weights * 100,
        "Variante durable": [
            "✅" if use_durable_map.get(aid) and DURABLE_ASSETS[aid].has_durable_variant else "—"
            for aid in active_ids
        ],
    }).sort_values("Poids (%)", ascending=False)
    st.dataframe(
        alloc_df.style.format({"Poids (%)": "{:.1f}"}),
        use_container_width=True, hide_index=True,
    )

    # --- Section 4: Hypothèses ---
    st.markdown("## 4. Hypothèses utilisées")
    col1, col2, col3 = st.columns(3)
    col1.metric("Aversion au risque (γ)", f"{gamma:.1f}")
    col2.metric("Poids durabilité (λ)", f"{lam:.1f}")
    col3.metric("Classes d'actifs actives", len(active_ids))

    with st.expander("Pondérations des dimensions"):
        for dim, lbl in zip(DIMS, DIM_LABELS):
            st.write(f"- **{lbl}**: {dim_weights.get(dim, 0)*100:.0f}%")

    with st.expander("Variantes durables activées"):
        for aid in active_ids:
            if use_durable_map.get(aid) and DURABLE_ASSETS[aid].has_durable_variant:
                st.write(f"- {DURABLE_ASSETS[aid].nom} → **{DURABLE_ASSETS[aid].nom_durable}**")

    # --- Export CSV ---
    st.divider()
    st.markdown("## 5. Export")
    export_df = pd.DataFrame({
        "Classe d'actifs": names,
        "Poids (%)": (result.weights * 100).round(2),
        "Variante durable": [
            "Oui" if use_durable_map.get(aid) and DURABLE_ASSETS[aid].has_durable_variant else "Non"
            for aid in active_ids
        ],
        "Rendement attendu (%)": [
            DURABLE_ASSETS[aid].rendement_durable * 100
            if use_durable_map.get(aid) and DURABLE_ASSETS[aid].has_durable_variant
            else DURABLE_ASSETS[aid].rendement * 100
            for aid in active_ids
        ],
    })
    csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Télécharger l'allocation (CSV)",
        data=csv,
        file_name="portefeuille_durable.csv",
        mime="text/csv",
        use_container_width=True,
    )


render()
```

- [ ] **Step 2: Verify syntax**

```bash
python -c "import ast; ast.parse(open('pages/page_durable_rapport.py').read()); print('Syntax OK')"
```

- [ ] **Step 3: Commit**

```bash
git add pages/page_durable_rapport.py
git commit -m "feat(sustainable): add Rapport durable page

Financial metrics, dimension score table, allocation with durable
variant indicators, assumptions, CSV export."
git push
```

---

## Chunk 5: Integration (`app.py`)

### Task 8: Add new navigation section to `app.py`

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Locate insertion point in `app.py`**

In `app.py` at line ~108, the `pages` dict ends with `"Gestion"` followed by `"Aide"`:

```python
        "Gestion": [
            st.Page(os.path.join(pages_dir, "page_rebalancing.py"), title="Reequilibrage", ...),
            st.Page(os.path.join(pages_dir, "page_reports.py"), title="Rapports", ...),
        ],
        "Aide": [   # ← insert the new section BEFORE this line
            ...
        ],
```

- [ ] **Step 2: Add the 5 new durable pages**

Insert the new section between the closing `],` of "Gestion" and the `"Aide":` line (around line 111 of `app.py`):

```python
        "🌱 Optimisation durable": [
            st.Page(os.path.join(pages_dir, "page_durable_univers.py"),
                    title="Univers durable", icon=":material/eco:"),
            st.Page(os.path.join(pages_dir, "page_durable_scores.py"),
                    title="Scores de durabilité", icon=":material/star:"),
            st.Page(os.path.join(pages_dir, "page_durable_frontier.py"),
                    title="Frontière durable", icon=":material/scatter_plot:"),
            st.Page(os.path.join(pages_dir, "page_durable_optimization.py"),
                    title="Optimisation durable", icon=":material/tune:"),
            st.Page(os.path.join(pages_dir, "page_durable_rapport.py"),
                    title="Rapport durable", icon=":material/description:"),
        ],
```

The full modified block (lines 91–116) should look like:

```python
    pages = {
        "Vue d'ensemble": [...],
        "Optimisation": [...],
        "Analyse de risque": [...],
        "Strategies": [...],
        "Gestion": [...],
        "🌱 Optimisation durable": [   # ← NEW
            st.Page(os.path.join(pages_dir, "page_durable_univers.py"), title="Univers durable", icon=":material/eco:"),
            st.Page(os.path.join(pages_dir, "page_durable_scores.py"), title="Scores de durabilité", icon=":material/star:"),
            st.Page(os.path.join(pages_dir, "page_durable_frontier.py"), title="Frontière durable", icon=":material/scatter_plot:"),
            st.Page(os.path.join(pages_dir, "page_durable_optimization.py"), title="Optimisation durable", icon=":material/tune:"),
            st.Page(os.path.join(pages_dir, "page_durable_rapport.py"), title="Rapport durable", icon=":material/description:"),
        ],
        "Aide": [...],
    }
```

- [ ] **Step 3: Verify app.py syntax**

```bash
python -c "import ast; ast.parse(open('app.py').read()); print('Syntax OK')"
```

- [ ] **Step 4: Run full test suite**

```bash
python -m pytest tests/ -v --tb=short 2>&1 | tail -30
```

Expected: All tests pass (including the 16 new sustainable tests).

- [ ] **Step 5: Commit and push**

```bash
git add app.py
git commit -m "feat(sustainable): integrate 5 durable pages into app navigation

New section '🌱 Optimisation durable' with 5 pages:
Univers, Scores, Frontière, Optimisation, Rapport."
git push
```

---

## Quick smoke test after deployment

1. Open the app → verify "🌱 Optimisation durable" appears in sidebar
2. Go to **Univers durable** → toggle "Obligations vertes" durable variant → Appliquer
3. Go to **Scores de durabilité** → modify one score → Appliquer
4. Go to **Frontière durable** → Calculer (50 points) → verify scatter plot appears
5. Select a point → "Utiliser ce portefeuille"
6. Go to **Optimisation durable** → Lancer → verify 3-column comparison
7. Go to **Rapport durable** → verify metrics → download CSV

# sustainable/config.py
"""
Données des classes d'actifs durables.
Source: WG_Catégories_actifs_v4.xlsx (Cartographie_complète_v1 + Anticipations + Corrélations)
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np


@dataclass
class DurableAsset:
    id: str
    nom: str
    nom_durable: str
    rendement: float
    volatilite: float
    rendement_durable: float
    volatilite_durable: float
    has_durable_variant: bool
    score_durabilite: float
    score_additionnalite: float
    score_disponibilite: float
    score_retombees_qc: float
    score_liquidite: float
    score_durabilite_durable: float = 0.0
    score_additionnalite_durable: float = 0.0
    score_disponibilite_durable: float = 0.0
    score_retombees_qc_durable: float = 0.0
    score_liquidite_durable: float = 0.0


DURABLE_ASSETS: Dict[str, DurableAsset] = {
    "obligations_ct": DurableAsset(
        id="obligations_ct", nom="Obligations court terme", nom_durable="",
        rendement=0.034, volatilite=0.026, rendement_durable=0.034, volatilite_durable=0.026,
        has_durable_variant=False,
        score_durabilite=1, score_additionnalite=1, score_disponibilite=1, score_retombees_qc=1, score_liquidite=5,
    ),
    "obligations_univers": DurableAsset(
        id="obligations_univers", nom="Obligations univers", nom_durable="Obligations vertes",
        rendement=0.0367, volatilite=0.0484, rendement_durable=0.035, volatilite_durable=0.050,
        has_durable_variant=True,
        score_durabilite=3, score_additionnalite=3, score_disponibilite=3, score_retombees_qc=2, score_liquidite=5,
        score_durabilite_durable=4, score_additionnalite_durable=4, score_disponibilite_durable=4,
        score_retombees_qc_durable=3, score_liquidite_durable=4,
    ),
    "hypotheques": DurableAsset(
        id="hypotheques", nom="Hypothèques commerciales", nom_durable="Hypothèques durables",
        rendement=0.045, volatilite=0.046, rendement_durable=0.044, volatilite_durable=0.047,
        has_durable_variant=True,
        score_durabilite=2.5, score_additionnalite=2.5, score_disponibilite=2.5, score_retombees_qc=2, score_liquidite=2,
        score_durabilite_durable=3.5, score_additionnalite_durable=3.5, score_disponibilite_durable=3.0,
        score_retombees_qc_durable=3, score_liquidite_durable=2,
    ),
    "oblig_rendement_eleve": DurableAsset(
        id="oblig_rendement_eleve", nom="Obligations à rendement élevé", nom_durable="",
        rendement=0.0567, volatilite=0.0773, rendement_durable=0.0567, volatilite_durable=0.0773,
        has_durable_variant=False,
        score_durabilite=1, score_additionnalite=1, score_disponibilite=1, score_retombees_qc=1, score_liquidite=4,
    ),
    "dettes_emergentes": DurableAsset(
        id="dettes_emergentes", nom="Dettes de pays émergents", nom_durable="",
        rendement=0.0587, volatilite=0.0925, rendement_durable=0.0587, volatilite_durable=0.0925,
        has_durable_variant=False,
        score_durabilite=1, score_additionnalite=1, score_disponibilite=1, score_retombees_qc=1, score_liquidite=3,
    ),
    "actions_cdn": DurableAsset(
        id="actions_cdn", nom="Actions canadiennes", nom_durable="",
        rendement=0.0587, volatilite=0.1388, rendement_durable=0.0587, volatilite_durable=0.1388,
        has_durable_variant=False,
        score_durabilite=2, score_additionnalite=2, score_disponibilite=2, score_retombees_qc=2, score_liquidite=5,
    ),
    "actions_globales": DurableAsset(
        id="actions_globales", nom="Actions globales", nom_durable="Actions ACWI Sustainable",
        rendement=0.063, volatilite=0.1385, rendement_durable=0.062, volatilite_durable=0.138,
        has_durable_variant=True,
        score_durabilite=2.5, score_additionnalite=2, score_disponibilite=3, score_retombees_qc=1, score_liquidite=5,
        score_durabilite_durable=4, score_additionnalite_durable=3, score_disponibilite_durable=4,
        score_retombees_qc_durable=1, score_liquidite_durable=5,
    ),
    "actions_petite_cap": DurableAsset(
        id="actions_petite_cap", nom="Actions mondiales petite cap.", nom_durable="",
        rendement=0.0633, volatilite=0.1708, rendement_durable=0.0633, volatilite_durable=0.1708,
        has_durable_variant=False,
        score_durabilite=2.5, score_additionnalite=2, score_disponibilite=3, score_retombees_qc=1, score_liquidite=4,
    ),
    "eqp": DurableAsset(
        id="eqp", nom="EQP (micro-cap Qc)", nom_durable="",
        rendement=0.0587, volatilite=0.1442, rendement_durable=0.0587, volatilite_durable=0.1442,
        has_durable_variant=False,
        score_durabilite=2.5, score_additionnalite=2, score_disponibilite=3, score_retombees_qc=3, score_liquidite=4,
    ),
    "fonds_couverture": DurableAsset(
        id="fonds_couverture", nom="Fonds de couverture", nom_durable="",
        rendement=0.0563, volatilite=0.0597, rendement_durable=0.0563, volatilite_durable=0.0597,
        has_durable_variant=False,
        score_durabilite=1.5, score_additionnalite=1, score_disponibilite=2, score_retombees_qc=1, score_liquidite=3,
    ),
    "dette_privee": DurableAsset(
        id="dette_privee", nom="Dette privée", nom_durable="Dette privée (admissible)",
        rendement=0.0695, volatilite=0.081, rendement_durable=0.075, volatilite_durable=0.08,
        has_durable_variant=True,
        score_durabilite=3.25, score_additionnalite=3, score_disponibilite=3.5, score_retombees_qc=1, score_liquidite=2,
        score_durabilite_durable=4, score_additionnalite_durable=4, score_disponibilite_durable=4,
        score_retombees_qc_durable=5, score_liquidite_durable=1,
    ),
    "immo_prive": DurableAsset(
        id="immo_prive", nom="Immobilier privé", nom_durable="Immobilier (admissible)",
        rendement=0.0527, volatilite=0.117, rendement_durable=0.075, volatilite_durable=0.11,
        has_durable_variant=True,
        score_durabilite=3.75, score_additionnalite=4, score_disponibilite=3.5, score_retombees_qc=1, score_liquidite=1,
        score_durabilite_durable=4.5, score_additionnalite_durable=5, score_disponibilite_durable=4,
        score_retombees_qc_durable=5, score_liquidite_durable=1,
    ),
    "infra_privee": DurableAsset(
        id="infra_privee", nom="Infrastructures privées", nom_durable="Infrastructures (admissible)",
        rendement=0.0793, volatilite=0.142, rendement_durable=0.065, volatilite_durable=0.14,
        has_durable_variant=True,
        score_durabilite=4.25, score_additionnalite=4, score_disponibilite=4.5, score_retombees_qc=1, score_liquidite=2,
        score_durabilite_durable=4.0, score_additionnalite_durable=4, score_disponibilite_durable=3.5,
        score_retombees_qc_durable=5, score_liquidite_durable=1,
    ),
    "buyout": DurableAsset(
        id="buyout", nom="Buyout", nom_durable="Buyout (admissible)",
        rendement=0.089, volatilite=0.1951, rendement_durable=0.089, volatilite_durable=0.195,
        has_durable_variant=True,
        score_durabilite=4, score_additionnalite=4, score_disponibilite=4, score_retombees_qc=2, score_liquidite=2,
        score_durabilite_durable=3.5, score_additionnalite_durable=5, score_disponibilite_durable=2,
        score_retombees_qc_durable=5, score_liquidite_durable=1,
    ),
    "capital_risque": DurableAsset(
        id="capital_risque", nom="Capital de risque", nom_durable="Capital de risque (admissible)",
        rendement=0.077, volatilite=0.1804, rendement_durable=0.085, volatilite_durable=0.18,
        has_durable_variant=True,
        score_durabilite=4.5, score_additionnalite=4, score_disponibilite=5, score_retombees_qc=2, score_liquidite=1,
        score_durabilite_durable=5, score_additionnalite_durable=5, score_disponibilite_durable=5,
        score_retombees_qc_durable=5, score_liquidite_durable=1,
    ),
}

DURABLE_ASSET_ORDER = [
    "obligations_ct", "obligations_univers", "hypotheques",
    "oblig_rendement_eleve", "dettes_emergentes",
    "actions_cdn", "actions_globales", "actions_petite_cap", "eqp",
    "fonds_couverture",
    "dette_privee", "immo_prive", "infra_privee", "buyout", "capital_risque",
]

N_DURABLE_ASSETS = len(DURABLE_ASSET_ORDER)
DURABLE_ASSET_NAMES_FR = [DURABLE_ASSETS[a].nom for a in DURABLE_ASSET_ORDER]

DURABLE_DEFAULT_WEIGHTS = np.array([
    0.05, 0.12, 0.05, 0.03, 0.03,
    0.10, 0.15, 0.05, 0.05, 0.05,
    0.10, 0.08, 0.08, 0.04, 0.02,
], dtype=float)
DURABLE_DEFAULT_WEIGHTS /= DURABLE_DEFAULT_WEIGHTS.sum()

DURABLE_MIN_WEIGHTS = np.zeros(N_DURABLE_ASSETS)
DURABLE_MAX_WEIGHTS = np.array([
    0.10, 0.40, 0.15, 0.10, 0.10,
    0.30, 0.40, 0.10, 0.15, 0.15,
    0.20, 0.15, 0.20, 0.15, 0.15,
])

# Correlation matrix 15x15 — source: Corrélations sheet (Aon 31 déc 2024), manually mapped
# Order: [obCT, obUN, hypo, obRE, dEm, acCDN, acGL, acPC, EQP, couv, dPriv, immoP, infra, buy, capR]
_CORR = np.array([
    [ 1.00,  0.90,  0.60,  0.20,  0.20,  0.00,  0.00,  0.00,  0.00,  0.10,  0.60,  0.10,  0.40,  0.00,  0.00],
    [ 0.90,  1.00,  0.55,  0.10,  0.20,  0.00,  0.00,  0.00,  0.00,  0.10,  0.70,  0.00,  0.40,  0.00,  0.00],
    [ 0.60,  0.55,  1.00,  0.20,  0.30,  0.10,  0.10,  0.10,  0.10,  0.10,  0.40,  0.30,  0.30,  0.10,  0.10],
    [ 0.20,  0.10,  0.20,  1.00,  0.90,  0.30,  0.50,  0.40,  0.40,  0.70,  0.30,  0.20,  0.10,  0.40,  0.30],
    [ 0.20,  0.20,  0.30,  0.90,  1.00,  0.30,  0.40,  0.30,  0.30,  0.60,  0.40,  0.20,  0.10,  0.30,  0.20],
    [ 0.00,  0.00,  0.10,  0.30,  0.30,  1.00,  0.80,  0.70,  0.70,  0.00,  0.10,  0.40,  0.10,  0.70,  0.70],
    [ 0.00,  0.00,  0.10,  0.50,  0.40,  0.80,  1.00,  0.80,  0.70,  0.00,  0.10,  0.40,  0.10,  0.70,  0.70],
    [ 0.00,  0.00,  0.10,  0.40,  0.30,  0.70,  0.80,  1.00,  0.50,  0.00,  0.10,  0.30,  0.10,  0.60,  0.60],
    [ 0.00,  0.00,  0.10,  0.40,  0.30,  0.70,  0.70,  0.50,  1.00,  0.10,  0.10,  0.30,  0.10,  0.85,  0.90],
    [ 0.10,  0.10,  0.10,  0.70,  0.60,  0.00,  0.00,  0.00,  0.10,  1.00,  0.20,  0.30,  0.10,  0.50,  0.40],
    [ 0.60,  0.70,  0.40,  0.30,  0.40,  0.10,  0.10,  0.10,  0.10,  0.20,  1.00,  0.30,  0.30,  0.10,  0.10],
    [ 0.10,  0.00,  0.30,  0.20,  0.20,  0.40,  0.40,  0.30,  0.30,  0.30,  0.30,  1.00,  0.30,  0.30,  0.30],
    [ 0.40,  0.40,  0.30,  0.10,  0.10,  0.10,  0.10,  0.10,  0.10,  0.10,  0.30,  0.30,  1.00,  0.30,  0.20],
    [ 0.00,  0.00,  0.10,  0.40,  0.30,  0.70,  0.70,  0.60,  0.85,  0.50,  0.10,  0.30,  0.30,  1.00,  0.90],
    [ 0.00,  0.00,  0.10,  0.30,  0.20,  0.70,  0.70,  0.60,  0.90,  0.40,  0.10,  0.30,  0.20,  0.90,  1.00],
])
_CORR = (_CORR + _CORR.T) / 2
np.fill_diagonal(_CORR, 1.0)
_CORR = np.clip(_CORR, -1.0, 1.0)

# Project to nearest positive semi-definite matrix (Higham 2002)
# Required because the source Excel matrix is not PSD due to estimation noise
def _nearest_psd(C: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.maximum(eigvals, 0.0)
    C_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    # Re-normalise to correlation matrix
    d = np.sqrt(np.diag(C_psd))
    C_psd = C_psd / np.outer(d, d)
    np.fill_diagonal(C_psd, 1.0)
    return C_psd

_CORR = _nearest_psd(_CORR)


def get_durable_returns(use_durable: Optional[Dict[str, bool]] = None) -> np.ndarray:
    use_durable = use_durable or {}
    return np.array([
        DURABLE_ASSETS[a].rendement_durable
        if use_durable.get(a) and DURABLE_ASSETS[a].has_durable_variant
        else DURABLE_ASSETS[a].rendement
        for a in DURABLE_ASSET_ORDER
    ])


def get_durable_volatilities(use_durable: Optional[Dict[str, bool]] = None) -> np.ndarray:
    use_durable = use_durable or {}
    return np.array([
        DURABLE_ASSETS[a].volatilite_durable
        if use_durable.get(a) and DURABLE_ASSETS[a].has_durable_variant
        else DURABLE_ASSETS[a].volatilite
        for a in DURABLE_ASSET_ORDER
    ])


def get_durable_correlation_matrix() -> np.ndarray:
    return _CORR.copy()


def get_durable_covariance_matrix(use_durable: Optional[Dict[str, bool]] = None) -> np.ndarray:
    vols = get_durable_volatilities(use_durable)
    corr = get_durable_correlation_matrix()
    return np.diag(vols) @ corr @ np.diag(vols)


def compute_composite_score(
    weights_port: np.ndarray,
    dim_weights: Dict[str, float],
    use_durable: Optional[Dict[str, bool]] = None,
    custom_scores: Optional[Dict[str, Dict[str, float]]] = None,
) -> float:
    S, _ = get_score_matrix(dim_weights, use_durable, custom_scores)
    return float(weights_port @ S)


def get_score_matrix(
    dim_weights: Dict[str, float],
    use_durable: Optional[Dict[str, bool]] = None,
    custom_scores: Optional[Dict[str, Dict[str, float]]] = None,
) -> tuple:
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
            if dim in cust:
                s_cur = cust[dim]
            elif w_dur:
                s_cur = getattr(asset, f"score_{dim}_durable")
            else:
                s_cur = getattr(asset, f"score_{dim}")
            S_current[i] += dw * s_cur

            if dim in cust:
                s_dur = cust[dim]
            elif asset.has_durable_variant:
                s_dur = getattr(asset, f"score_{dim}_durable")
            else:
                s_dur = getattr(asset, f"score_{dim}")
            S_durable[i] += dw * s_dur

    return S_current, S_durable


DEFAULT_DIM_WEIGHTS = {
    "durabilite": 0.35,
    "additionnalite": 0.25,
    "disponibilite": 0.15,
    "retombees_qc": 0.15,
    "liquidite": 0.10,
}

"""
Configuration centrale de l optimiseur de portefeuille.
Definit les classes d actifs, parametres par defaut, matrice de correlation et labels FR.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class AssetClass(Enum):
    ACTIONS_CDN = "actions_canadiennes"
    ACTIONS_US = "actions_americaines"
    ACTIONS_EAFE = "actions_eafe"
    ACTIONS_EMERGENTES = "actions_emergentes"
    OBLIGATIONS_GOV_CDN = "obligations_gouvernementales_cdn"
    OBLIGATIONS_CORP = "obligations_corporatives"
    OBLIGATIONS_INFLATION = "obligations_indexees_inflation"
    IMMOBILIER = "immobilier"
    INFRASTRUCTURE = "infrastructure"
    CAPITAL_INVESTISSEMENT = "capital_investissement"
    RENDEMENT_ABSOLU = "rendement_absolu"
    MATIERES_PREMIERES = "matieres_premieres"
    ENCAISSE = "encaisse"
    ACTIONS_ACWI = "actions_acwi"
    DETTE_EMERGENTE = "dette_emergente"
    DETTE_PRIVEE = "dette_privee"
    OBLIGATIONS_HY = "obligations_hy"


@dataclass
class AssetClassConfig:
    code: AssetClass
    nom_fr: str
    expected_return: float
    volatility: float
    liquidity_score: float
    esg_score: float
    min_allocation: float
    max_allocation: float
    is_alternative: bool
    duration: Optional[float] = None


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

ASSET_DEFAULTS: Dict[AssetClass, AssetClassConfig] = {
    AssetClass.ACTIONS_CDN: AssetClassConfig(AssetClass.ACTIONS_CDN, "Actions canadiennes", 0.075, 0.16, 0.95, 65.0, 0.05, 0.30, False),
    AssetClass.ACTIONS_US: AssetClassConfig(AssetClass.ACTIONS_US, "Actions americaines", 0.080, 0.17, 0.98, 60.0, 0.05, 0.30, False),
    AssetClass.ACTIONS_EAFE: AssetClassConfig(AssetClass.ACTIONS_EAFE, "Actions EAFE", 0.070, 0.18, 0.90, 70.0, 0.00, 0.20, False),
    AssetClass.ACTIONS_EMERGENTES: AssetClassConfig(AssetClass.ACTIONS_EMERGENTES, "Actions emergentes", 0.090, 0.22, 0.75, 45.0, 0.00, 0.15, False),
    AssetClass.OBLIGATIONS_GOV_CDN: AssetClassConfig(AssetClass.OBLIGATIONS_GOV_CDN, "Obligations gouvernementales CDN", 0.035, 0.06, 1.00, 80.0, 0.10, 0.40, False, 7.5),
    AssetClass.OBLIGATIONS_CORP: AssetClassConfig(AssetClass.OBLIGATIONS_CORP, "Obligations corporatives", 0.045, 0.08, 0.85, 55.0, 0.00, 0.25, False, 5.5),
    AssetClass.OBLIGATIONS_INFLATION: AssetClassConfig(AssetClass.OBLIGATIONS_INFLATION, "Obligations indexees inflation", 0.030, 0.07, 0.90, 80.0, 0.00, 0.15, False, 10.0),
    AssetClass.IMMOBILIER: AssetClassConfig(AssetClass.IMMOBILIER, "Immobilier", 0.070, 0.12, 0.30, 60.0, 0.00, 0.15, True),
    AssetClass.INFRASTRUCTURE: AssetClassConfig(AssetClass.INFRASTRUCTURE, "Infrastructure", 0.075, 0.10, 0.25, 70.0, 0.00, 0.15, True),
    AssetClass.CAPITAL_INVESTISSEMENT: AssetClassConfig(AssetClass.CAPITAL_INVESTISSEMENT, "Capital investissement", 0.100, 0.20, 0.10, 50.0, 0.00, 0.15, True),
    AssetClass.RENDEMENT_ABSOLU: AssetClassConfig(AssetClass.RENDEMENT_ABSOLU, "Rendement absolu", 0.055, 0.08, 0.50, 50.0, 0.00, 0.15, True),
    AssetClass.MATIERES_PREMIERES: AssetClassConfig(AssetClass.MATIERES_PREMIERES, "Matieres premieres", 0.040, 0.18, 0.80, 35.0, 0.00, 0.10, False),
    AssetClass.ENCAISSE: AssetClassConfig(AssetClass.ENCAISSE, "Encaisse", 0.025, 0.01, 1.00, 75.0, 0.02, 0.10, False, 0.25),
    AssetClass.ACTIONS_ACWI: AssetClassConfig(
        AssetClass.ACTIONS_ACWI, "Actions MSCI ACWI",
        0.078, 0.16, 0.95, 58.0, 0.00, 0.40, False
    ),
    AssetClass.DETTE_EMERGENTE: AssetClassConfig(
        AssetClass.DETTE_EMERGENTE, "Dette pays emergents",
        0.058, 0.11, 0.70, 45.0, 0.00, 0.15, False, 7.0
    ),
    AssetClass.DETTE_PRIVEE: AssetClassConfig(
        AssetClass.DETTE_PRIVEE, "Dette privee",
        0.075, 0.06, 0.10, 50.0, 0.00, 0.10, True, 5.0
    ),
    AssetClass.OBLIGATIONS_HY: AssetClassConfig(
        AssetClass.OBLIGATIONS_HY, "Obligations HY",
        0.055, 0.09, 0.80, 48.0, 0.00, 0.15, False, 4.5
    ),
}

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

DEFAULT_CURRENT_WEIGHTS = np.array([
    0.12, 0.14, 0.08, 0.05, 0.19, 0.10, 0.05, 0.07, 0.07, 0.05, 0.03, 0.03, 0.02, 0.00, 0.00, 0.00, 0.00,
])


@dataclass
class PensionFundConfig:
    nom: str = "Caisse de retraite"
    horizon_annees: int = 20
    taux_actualisation: float = 0.05
    valeur_actif: float = 1_000_000_000.0
    taux_inflation_cible: float = 0.02
    niveau_confiance_var: float = 0.95
    niveau_confiance_cvar: float = 0.95
    frequence_reequilibrage: str = "trimestriel"
    taux_sans_risque: float = 0.025


LABELS_FR = {
    "app_title": "Optimiseur de Portefeuille - Caisse de Retraite",
    "dashboard": "Tableau de bord",
    "optimization": "Moteur d optimisation",
    "constraints": "Gestionnaire de contraintes",
    "risk_analytics": "Analytique de risque",
    "efficient_frontier": "Frontiere efficiente",
    "monte_carlo": "Simulation Monte Carlo",
    "rebalancing": "Reequilibrage",
    "reports": "Rapports",
    "portable_alpha": "Alpha portable",
    "beta_portfolio": "Portefeuille beta",
    "alpha_overlay": "Overlay alpha",
    "gross_leverage": "Levier brut",
    "net_exposure": "Exposition nette",
    "financing_cost": "Cout de financement",
    "information_ratio": "Ratio d information",
}

CHART_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#636efa", "#aec7e8", "#ffbb78",
    "#ffa07a",  # Actions MSCI ACWI
    "#20b2aa",  # Dette pays emergents
    "#8b4513",  # Dette privee (saddle brown)
    "#ff6347",  # Obligations HY (tomato)
]


def get_asset_names_fr() -> List[str]:
    return [ASSET_DEFAULTS[ac].nom_fr for ac in ASSET_CLASSES_ORDER]

def get_expected_returns() -> np.ndarray:
    try:
        import streamlit as st
        if "custom_expected_returns" in st.session_state:
            return st.session_state.custom_expected_returns.copy()
    except Exception:
        pass
    return np.array([ASSET_DEFAULTS[ac].expected_return for ac in ASSET_CLASSES_ORDER])

def get_volatilities() -> np.ndarray:
    try:
        import streamlit as st
        if "custom_volatilities" in st.session_state:
            return st.session_state.custom_volatilities.copy()
    except Exception:
        pass
    return np.array([ASSET_DEFAULTS[ac].volatility for ac in ASSET_CLASSES_ORDER])

def get_covariance_matrix() -> np.ndarray:
    vols = get_volatilities()
    D = np.diag(vols)
    return D @ DEFAULT_CORRELATION_MATRIX @ D

def get_min_weights() -> np.ndarray:
    return np.array([ASSET_DEFAULTS[ac].min_allocation for ac in ASSET_CLASSES_ORDER])

def get_max_weights() -> np.ndarray:
    return np.array([ASSET_DEFAULTS[ac].max_allocation for ac in ASSET_CLASSES_ORDER])

def get_esg_scores() -> np.ndarray:
    return np.array([ASSET_DEFAULTS[ac].esg_score for ac in ASSET_CLASSES_ORDER])

def get_liquidity_scores() -> np.ndarray:
    return np.array([ASSET_DEFAULTS[ac].liquidity_score for ac in ASSET_CLASSES_ORDER])

def get_asset_durations() -> np.ndarray:
    return np.array([
        ASSET_DEFAULTS[ac].duration if ASSET_DEFAULTS[ac].duration is not None else 0.0
        for ac in ASSET_CLASSES_ORDER
    ])


# ============================================================
# Alpha Portable - Benchmarks et configuration
# ============================================================

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

# Actifs eligibles a la vente a decouvert (liquides uniquement)
# Indices: 0-3 equities, 4-6 bonds, 11 commodities, 13 ACWI, 14 EMD, 16 HY
# Note: 15 (Dette privee) excluded — illiquid private market
ALPHA_ELIGIBLE_SHORT = [0, 1, 2, 3, 4, 5, 6, 11, 13, 14, 16]

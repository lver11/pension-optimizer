"""
Profil de fonds : tout ce qui distingue un fonds d'un autre.

Un profil regroupe les parametres propres a une organisation (actif, passif,
flux, portefeuille de politique, bornes par classe d'actifs, limites de groupe,
hypotheses de marche facultatives). Le reste de l'application lit le profil actif
au lieu de valeurs codees en dur, ce qui permet de l'utiliser pour n'importe quel
fonds : caisse a prestations determinees, regime a cotisations determinees,
fondation, fonds de travailleurs, etc.

Les profils se sauvegardent et se chargent en JSON.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
    ASSET_CLASSES_ORDER, ASSET_DEFAULTS, ASSET_CATEGORIES, CATEGORY_LABELS_FR,
    DEFAULT_CURRENT_WEIGHTS, PensionFundConfig, get_asset_codes,
)

PROFILE_FORMAT_VERSION = 1

TYPES_FONDS = {
    "prestations_determinees": "Regime a prestations determinees (PD)",
    "cotisations_determinees": "Regime a cotisations determinees (CD)",
    "fondation": "Fondation / fonds de dotation",
    "fonds_travailleurs": "Fonds de travailleurs",
    "autre": "Autre",
}

TOL = 1e-6


@dataclass
class GroupLimit:
    """Limite de groupe de la politique de placement (ex. actions totales <= 70 %)."""
    nom: str
    classes: List[str]
    min: float = 0.0
    max: float = 1.0


@dataclass
class FundProfile:
    nom: str
    type_fonds: str = "prestations_determinees"
    description: str = ""
    devise: str = "CAD"

    # Bilan
    valeur_actif: float = 1_000_000_000.0
    valeur_passif: Optional[float] = None       # None = pas de passif actuariel
    duration_passif: float = 15.0
    taux_actualisation: float = 0.05
    croissance_passif: float = 0.03

    # Marche et horizon
    taux_sans_risque: float = 0.025
    horizon_annees: int = 20

    # Flux annuels (entrees et sorties de fonds)
    cotisations_annuelles: float = 0.0
    croissance_cotisations: float = 0.02
    prestations_annuelles: float = 0.0
    croissance_prestations: float = 0.03

    # Politique de placement (cles = codes de classes d'actifs, ex. "actions_acwi")
    poids_politique: Dict[str, float] = field(default_factory=dict)
    bornes_min: Dict[str, float] = field(default_factory=dict)
    bornes_max: Dict[str, float] = field(default_factory=dict)
    limites_groupes: List[GroupLimit] = field(default_factory=list)

    # Hypotheses de marche propres au fonds (facultatif ; sinon valeurs par defaut)
    rendements_attendus: Optional[Dict[str, float]] = None
    volatilites: Optional[Dict[str, float]] = None

    # ------------------------------------------------------------------
    # Conversion en vecteurs alignes sur ASSET_CLASSES_ORDER
    # ------------------------------------------------------------------
    def _array(self, values: Dict[str, float], default_attr: Optional[str], default: float = 0.0) -> np.ndarray:
        out = []
        for ac in ASSET_CLASSES_ORDER:
            if ac.value in values:
                out.append(float(values[ac.value]))
            elif default_attr is not None:
                out.append(float(getattr(ASSET_DEFAULTS[ac], default_attr)))
            else:
                out.append(default)
        return np.array(out)

    def weights_array(self) -> np.ndarray:
        return self._array(self.poids_politique, None, 0.0)

    def min_array(self) -> np.ndarray:
        return self._array(self.bornes_min, "min_allocation")

    def max_array(self) -> np.ndarray:
        return self._array(self.bornes_max, "max_allocation")

    def expected_returns_array(self) -> Optional[np.ndarray]:
        if not self.rendements_attendus:
            return None
        return self._array(self.rendements_attendus, "expected_return")

    def volatilities_array(self) -> Optional[np.ndarray]:
        if not self.volatilites:
            return None
        return self._array(self.volatilites, "volatility")

    @property
    def a_un_passif(self) -> bool:
        return self.valeur_passif is not None and self.valeur_passif > 0

    @property
    def ratio_capitalisation(self) -> Optional[float]:
        if not self.a_un_passif:
            return None
        return self.valeur_actif / self.valeur_passif

    def group_constraints(self):
        """Limites de groupe converties en GroupConstraint (indices)."""
        from constraints.manager import GroupConstraint
        codes = get_asset_codes()
        out = []
        for g in self.limites_groupes:
            idx = [codes.index(c) for c in g.classes if c in codes]
            if idx:
                out.append(GroupConstraint(g.nom, idx, g.min, g.max))
        return out

    def to_pension_config(self) -> PensionFundConfig:
        return PensionFundConfig(
            nom=self.nom,
            horizon_annees=int(self.horizon_annees),
            taux_actualisation=self.taux_actualisation,
            valeur_actif=self.valeur_actif,
            valeur_passif=self.valeur_passif or 0.0,
            taux_sans_risque=self.taux_sans_risque,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def validate(self) -> Tuple[List[str], List[str]]:
        """Retourne (erreurs, avertissements). Les erreurs empechent l'application."""
        errors, warnings = [], []
        codes = set(get_asset_codes())

        for label, d in [("poids_politique", self.poids_politique),
                         ("bornes_min", self.bornes_min),
                         ("bornes_max", self.bornes_max),
                         ("rendements_attendus", self.rendements_attendus or {}),
                         ("volatilites", self.volatilites or {})]:
            unknown = set(d) - codes
            if unknown:
                errors.append(f"{label}: classes inconnues {sorted(unknown)}")
        for g in self.limites_groupes:
            unknown = set(g.classes) - codes
            if unknown:
                errors.append(f"Limite '{g.nom}': classes inconnues {sorted(unknown)}")
            if g.min > g.max + TOL:
                errors.append(f"Limite '{g.nom}': min ({g.min:.0%}) > max ({g.max:.0%})")

        if self.type_fonds not in TYPES_FONDS:
            errors.append(f"Type de fonds inconnu: {self.type_fonds}")
        if self.valeur_actif <= 0:
            errors.append("La valeur de l'actif doit etre positive.")
        if errors:
            return errors, warnings

        w, lo, hi = self.weights_array(), self.min_array(), self.max_array()
        names = [ASSET_DEFAULTS[ac].nom_fr for ac in ASSET_CLASSES_ORDER]

        if np.any(w < -TOL):
            errors.append("Les poids de politique doivent etre positifs.")
        if abs(w.sum() - 1.0) > 1e-4:
            errors.append(f"Les poids de politique totalisent {w.sum():.2%} (doit etre 100 %).")
        bad = [names[i] for i in range(len(w)) if lo[i] > hi[i] + TOL]
        if bad:
            errors.append(f"Borne min > max pour: {', '.join(bad)}")
        if lo.sum() > 1.0 + TOL:
            errors.append(f"La somme des bornes minimales ({lo.sum():.0%}) depasse 100 %: aucune allocation possible.")
        if hi.sum() < 1.0 - TOL:
            errors.append(f"La somme des bornes maximales ({hi.sum():.0%}) est inferieure a 100 %: aucune allocation possible.")

        out_of_bounds = [names[i] for i in range(len(w)) if w[i] < lo[i] - TOL or w[i] > hi[i] + TOL]
        if out_of_bounds:
            warnings.append("Le portefeuille de politique sort des bornes pour: " + ", ".join(out_of_bounds))
        for gc in self.group_constraints():
            s = w[gc.asset_indices].sum()
            if s < gc.min_allocation - TOL or s > gc.max_allocation + TOL:
                warnings.append(
                    f"Le portefeuille de politique viole la limite '{gc.name_fr}' "
                    f"({s:.1%} hors de {gc.min_allocation:.0%}-{gc.max_allocation:.0%})."
                )
        if self.type_fonds == "prestations_determinees" and not self.a_un_passif:
            warnings.append("Regime PD sans valeur de passif : l'analyse ALM sera desactivee.")
        return errors, warnings

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict:
        d = asdict(self)
        d["format_version"] = PROFILE_FORMAT_VERSION
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    @classmethod
    def from_dict(cls, d: Dict) -> "FundProfile":
        d = dict(d)
        d.pop("format_version", None)
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Champs inconnus dans le profil: {sorted(unknown)}")
        if "nom" not in d:
            raise ValueError("Le profil doit avoir un champ 'nom'.")
        d["limites_groupes"] = [g if isinstance(g, GroupLimit) else GroupLimit(**g)
                                for g in d.get("limites_groupes", [])]
        return cls(**d)

    @classmethod
    def from_json(cls, text: str) -> "FundProfile":
        return cls.from_dict(json.loads(text))

    def copy(self) -> "FundProfile":
        return copy.deepcopy(self)


# ======================================================================
# Profils types
# ======================================================================

def _weights(**kw) -> Dict[str, float]:
    """Construit un dictionnaire de poids et verifie les codes."""
    codes = set(get_asset_codes())
    unknown = set(kw) - codes
    assert not unknown, unknown
    return {k: v for k, v in kw.items() if v}


def _category_codes(*categories: str) -> List[str]:
    return [ac.value for ac in ASSET_CLASSES_ORDER if ASSET_CATEGORIES[ac] in categories]


def _limits(max_actions=None, min_oblig=None, max_oblig=None, max_alt=None, max_pe=None, min_liq=None):
    out = []
    if max_actions is not None:
        out.append(GroupLimit(f"{CATEGORY_LABELS_FR['actions']} totales", _category_codes("actions"), 0.0, max_actions))
    if min_oblig is not None or max_oblig is not None:
        out.append(GroupLimit(f"{CATEGORY_LABELS_FR['obligations']}", _category_codes("obligations"),
                              min_oblig or 0.0, 1.0 if max_oblig is None else max_oblig))
    if max_alt is not None:
        out.append(GroupLimit(f"{CATEGORY_LABELS_FR['alternatifs']}", _category_codes("alternatifs"), 0.0, max_alt))
    if max_pe is not None:
        out.append(GroupLimit("Capital investissement", ["capital_investissement"], 0.0, max_pe))
    if min_liq is not None:
        out.append(GroupLimit("Liquidites minimales", _category_codes("liquidites"), min_liq, 1.0))
    return out


def _default_weights_dict() -> Dict[str, float]:
    return {ac.value: float(w) for ac, w in zip(ASSET_CLASSES_ORDER, DEFAULT_CURRENT_WEIGHTS) if w}


def profil_pd_generique() -> FundProfile:
    return FundProfile(
        nom="Caisse de retraite PD - generique",
        type_fonds="prestations_determinees",
        description="Regime a prestations determinees de taille moyenne, en croisiere. "
                    "Valeurs illustratives a remplacer par celles de votre fonds.",
        valeur_actif=1_000_000_000.0,
        valeur_passif=1_000_000_000.0,
        duration_passif=15.0,
        cotisations_annuelles=40_000_000.0,
        prestations_annuelles=57_000_000.0,
        poids_politique=_default_weights_dict(),
        limites_groupes=_limits(max_actions=0.70, min_oblig=0.10, max_oblig=0.70,
                                max_alt=0.40, max_pe=0.20, min_liq=0.02),
    )


def profil_pd_mature() -> FundProfile:
    return FundProfile(
        nom="Caisse de retraite PD mature - orientation LDI",
        type_fonds="prestations_determinees",
        description="Regime ferme ou mature : prestations superieures aux cotisations, "
                    "appariement du passif par les obligations. Valeurs illustratives.",
        valeur_actif=2_000_000_000.0,
        valeur_passif=2_100_000_000.0,
        duration_passif=13.0,
        cotisations_annuelles=30_000_000.0,
        prestations_annuelles=140_000_000.0,
        poids_politique=_weights(
            actions_canadiennes=0.05, actions_americaines=0.06, actions_eafe=0.04,
            actions_emergentes=0.02, actions_acwi=0.05,
            obligations_gouvernementales_cdn=0.30, obligations_corporatives=0.15,
            obligations_indexees_inflation=0.10,
            immobilier=0.06, infrastructure=0.08, capital_investissement=0.03,
            dette_privee=0.04, encaisse=0.02,
        ),
        bornes_max={"obligations_gouvernementales_cdn": 0.50},
        limites_groupes=_limits(max_actions=0.40, min_oblig=0.45, max_oblig=0.80,
                                max_alt=0.30, max_pe=0.10, min_liq=0.02),
    )


def profil_cd_equilibre() -> FundProfile:
    return FundProfile(
        nom="Regime CD - fonds equilibre",
        type_fonds="cotisations_determinees",
        description="Option equilibree par defaut d'un regime a cotisations determinees : "
                    "liquidite quotidienne, pas de placements prives. Valeurs illustratives.",
        valeur_actif=500_000_000.0,
        valeur_passif=None,
        cotisations_annuelles=50_000_000.0,
        prestations_annuelles=20_000_000.0,
        croissance_prestations=0.04,
        poids_politique=_weights(
            actions_canadiennes=0.10, actions_americaines=0.05, actions_eafe=0.05,
            actions_acwi=0.30,
            obligations_gouvernementales_cdn=0.15, obligations_corporatives=0.15,
            dette_emergente=0.03, obligations_hy=0.05,
            immobilier=0.05, infrastructure=0.05, encaisse=0.02,
        ),
        bornes_max={"capital_investissement": 0.0, "dette_privee": 0.0,
                    "rendement_absolu": 0.05, "immobilier": 0.10, "infrastructure": 0.10},
        limites_groupes=_limits(max_actions=0.65, min_oblig=0.25, max_alt=0.15, min_liq=0.01),
    )


def profil_fondation() -> FundProfile:
    return FundProfile(
        nom="Fondation - fonds de dotation",
        type_fonds="fondation",
        description="Horizon perpetuel, decaissement annuel d'environ 4 % de l'actif, "
                    "forte place aux placements prives. Valeurs illustratives.",
        valeur_actif=250_000_000.0,
        valeur_passif=None,
        horizon_annees=30,
        cotisations_annuelles=5_000_000.0,
        prestations_annuelles=10_000_000.0,
        croissance_prestations=0.02,
        poids_politique=_weights(
            actions_canadiennes=0.05, actions_americaines=0.05, actions_emergentes=0.05,
            actions_acwi=0.35,
            obligations_gouvernementales_cdn=0.10, obligations_corporatives=0.05,
            immobilier=0.08, infrastructure=0.07, capital_investissement=0.10,
            rendement_absolu=0.05, dette_privee=0.03, encaisse=0.02,
        ),
        bornes_max={"actions_acwi": 0.50, "capital_investissement": 0.20, "dette_privee": 0.15},
        limites_groupes=_limits(max_actions=0.70, max_alt=0.45, max_pe=0.20, min_liq=0.01),
    )


def profil_exemple_fondaction() -> FundProfile:
    gov = 0.10 + 0.25 * 2 / 3
    corp = 0.25 / 3 + 0.05
    return FundProfile(
        nom="Exemple - Fondaction (portefeuille de reference)",
        type_fonds="fonds_travailleurs",
        description="Portefeuille de reference de Fondaction (marches publics) mappe sur les classes "
                    "de l'application : obligations canadiennes 25 % reparties 2/3 gouvernemental, "
                    "1/3 corporatif ; obligations vertes 5 % classees en corporatif. "
                    "Flux annuels a completer.",
        valeur_actif=1_900_000_000.0,
        valeur_passif=None,
        poids_politique=_weights(
            actions_acwi=0.46,
            obligations_gouvernementales_cdn=gov, obligations_corporatives=corp,
            dette_emergente=0.06, rendement_absolu=0.08,
        ),
        bornes_min={"actions_canadiennes": 0.0, "actions_americaines": 0.0, "encaisse": 0.0},
        bornes_max={"actions_acwi": 0.60},
        limites_groupes=_limits(max_actions=0.70, max_alt=0.40),
    )


PROFILS_TYPES = {
    "pd_generique": profil_pd_generique,
    "pd_mature": profil_pd_mature,
    "cd_equilibre": profil_cd_equilibre,
    "fondation": profil_fondation,
    "exemple_fondaction": profil_exemple_fondaction,
}

PROFIL_PAR_DEFAUT = "pd_generique"


# ======================================================================
# Session Streamlit
# ======================================================================

def apply_profile(profile: FundProfile, reset_weights: bool = True) -> None:
    """Active un profil dans la session Streamlit et synchronise l'etat derive."""
    import streamlit as st
    st.session_state.fund_profile = profile
    st.session_state.profile_version = st.session_state.get("profile_version", 0) + 1
    st.session_state.pension_config = profile.to_pension_config()
    if reset_weights or "current_weights" not in st.session_state:
        st.session_state.current_weights = profile.weights_array()
    # Les contraintes sauvegardees et resultats dependent du profil precedent
    for key in ("constraint_set", "optimization_result"):
        st.session_state.pop(key, None)
    mu = profile.expected_returns_array()
    vol = profile.volatilities_array()
    if mu is not None:
        st.session_state.custom_expected_returns = mu
    else:
        st.session_state.pop("custom_expected_returns", None)
    if vol is not None:
        st.session_state.custom_volatilities = vol
    else:
        st.session_state.pop("custom_volatilities", None)


def get_active_profile() -> FundProfile:
    """Profil actif ; active le profil par defaut s'il n'y en a pas."""
    import streamlit as st
    if st.session_state.get("fund_profile") is None:
        apply_profile(PROFILS_TYPES[PROFIL_PAR_DEFAUT]())
    return st.session_state.fund_profile


def ensure_session_state() -> None:
    """Initialise les donnees de session manquantes sans ecraser celles qui existent."""
    import streamlit as st
    profile = get_active_profile()
    if st.session_state.get("pension_config") is None:
        st.session_state.pension_config = profile.to_pension_config()
    if st.session_state.get("current_weights") is None:
        st.session_state.current_weights = profile.weights_array()
    if st.session_state.get("returns_data") is None:
        from data.generator import MarketDataGenerator
        generator = MarketDataGenerator(seed=42)
        st.session_state.returns_data = generator.generate_returns(n_years=20, frequency="monthly")

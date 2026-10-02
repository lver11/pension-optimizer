"""
Rappels affiches pres des resultats : caractere illustratif des hypotheses
et explication du score de durabilite.
"""

from typing import Dict, Optional

import streamlit as st

DIM_LABELS_FR = {
    "durabilite": "Durabilite",
    "additionnalite": "Additionnalite",
    "disponibilite": "Disponibilite",
    "retombees_qc": "Retombees regionales",
    "liquidite": "Liquidite",
}


def show_assumption_notes(simulated_returns_used: bool = False) -> None:
    """
    Bandeau court indiquant ce qui, dans les resultats de la page, repose sur des
    valeurs illustratives plutot que sur les donnees du fonds.
    """
    notes = []
    profile = st.session_state.get("fund_profile")
    if profile is not None and getattr(profile, "profil_type", None):
        notes.append(f"profil type **{profile.nom}** (valeurs illustratives)")
    if "custom_expected_returns" not in st.session_state and "custom_volatilities" not in st.session_state:
        notes.append("hypotheses de marche par defaut de l'application (rendements, volatilites, correlations)")
    if simulated_returns_used:
        src = st.session_state.get("returns_source") or {"type": "simulees"}
        if src.get("type") != "importees":
            notes.append("rendements simules, pas l'historique reel")
    if notes:
        st.caption(
            "⚠️ Resultats illustratifs - bases sur : " + " ; ".join(notes) + ". "
            "Remplacez-les par vos donnees (pages Profil du fonds et Source de donnees) "
            "avant toute decision."
        )


def show_durable_example_note() -> None:
    st.caption(
        "⚠️ Univers durable d'exemple : hypotheses de rendement et scores tires du contexte "
        "d'un fonds de travailleurs quebecois (fichier WG_Categories_actifs_v4). "
        "A adapter a votre fonds avant toute decision."
    )


def explain_sustainability_score(
    dim_weights: Dict[str, float],
    score_fin: Optional[float] = None,
    score_dur: Optional[float] = None,
    sharpe_fin: Optional[float] = None,
    sharpe_dur: Optional[float] = None,
) -> None:
    """Expander expliquant l'echelle, la source et la composition du score."""
    total = sum(dim_weights.values()) or 1.0
    rows = "\n".join(
        f"| {DIM_LABELS_FR.get(k, k)} | {v / total:.0%} |" for k, v in dim_weights.items()
    )
    tradeoff = ""
    if None not in (score_fin, score_dur, sharpe_fin, sharpe_dur):
        d_score, d_sharpe = score_dur - score_fin, sharpe_dur - sharpe_fin
        tradeoff = (
            f"\n\n**Arbitrage ici** : le score passe de {score_fin:.2f} a {score_dur:.2f} "
            f"({d_score:+.2f} point sur une echelle de 1 a 5) et le ratio de Sharpe de "
            f"{sharpe_fin:.3f} a {sharpe_dur:.3f} ({d_sharpe:+.3f})."
        )
        if d_score > 1e-9:
            tradeoff += f" Soit {abs(d_sharpe) / d_score:.3f} de Sharpe par point de score."
    with st.expander("Comment le score de durabilite est calcule"):
        st.markdown(f"""
**Echelle** : de 1 (faible) a 5 (eleve). Chaque classe d'actifs recoit une note de 1 a 5
sur cinq dimensions ; le score d'une classe est la moyenne ponderee de ses cinq notes, et le
score du portefeuille est la moyenne de ces scores ponderee par les poids du portefeuille.

**Ponderations des dimensions** (modifiables dans Univers durable) :

| Dimension | Poids |
|---|---|
{rows}

**Source des notes** : grille d'exemple tiree du fichier WG_Categories_actifs_v4
(Cartographie complete) ; elles sont modifiables dans la page Scores de durabilite.
Les notes sont des jugements qualitatifs, pas des mesures : un ecart de quelques
dixiemes de point est a interpreter avec prudence.{tradeoff}
""")

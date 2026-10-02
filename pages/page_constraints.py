"""
Gestionnaire de contraintes.
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    get_asset_names_fr, get_min_weights, get_max_weights,
    get_esg_scores, get_liquidity_scores, get_policy_weights, PensionFundConfig,
    ASSET_DEFAULTS, ASSET_CLASSES_ORDER,
)
from ui_notes import show_assumption_notes
from fund_profile import ensure_session_state
from config import get_policy_weights
from constraints.manager import ConstraintManager, ConstraintSet, GroupConstraint
from constraints.esg import ESGConstraintEngine
from fund_profile import get_active_profile


def render():
    st.title("Gestionnaire de contraintes")
    ensure_session_state()
    show_assumption_notes(simulated_returns_used=False)

    asset_names = get_asset_names_fr()
    n_assets = len(asset_names)
    current_weights = st.session_state.get("current_weights", get_policy_weights()).copy()

    # ---------- Portefeuille actuel ----------
    st.markdown("### Portefeuille actuel")
    st.caption("Modifiez les poids du portefeuille de depart. Le total doit etre egal a 100%.")

    edited_weights = np.zeros(n_assets)

    # En-tete
    hdr1, hdr2 = st.columns([3, 2])
    hdr1.markdown("**Classe d'actifs**")
    hdr2.markdown("**Poids (%)**")

    for i in range(n_assets):
        col_name, col_weight = st.columns([3, 2])
        col_name.markdown(f"{asset_names[i]}")
        edited_weights[i] = col_weight.number_input(
            f"Poids {asset_names[i]}", 0.0, 100.0,
            float(current_weights[i] * 100), 0.5,
            key=f"pw_{i}", label_visibility="collapsed",
        ) / 100.0

    total = edited_weights.sum()
    col_total, col_actions = st.columns([3, 2])
    if abs(total - 1.0) < 1e-6:
        col_total.success(f"Total: {total:.1%}")
    else:
        col_total.warning(f"Total: {total:.1%} (doit etre 100%)")

    btn1, btn2, btn3 = col_actions.columns(3)
    if btn1.button("Normaliser", help="Ajuster proportionnellement pour atteindre 100%"):
        if total > 0:
            edited_weights = edited_weights / total
            st.session_state.current_weights = edited_weights
            st.rerun()
    if btn2.button("Appliquer", type="primary", help="Sauvegarder les poids tels quels"):
        st.session_state.current_weights = edited_weights
        st.rerun()
    if btn3.button("Reset", help="Revenir au portefeuille de politique du profil"):
        st.session_state.current_weights = get_policy_weights()
        st.rerun()

    current_weights = edited_weights

    st.divider()

    # ---------- Bornes par classe d'actifs ----------
    st.markdown("### Bornes par classe d'actifs")
    st.caption("Definissez l'allocation minimale et maximale pour chaque classe d'actifs.")

    min_w = get_min_weights().copy()
    max_w = get_max_weights().copy()

    for i in range(n_assets):
        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
        col1.markdown(f"**{asset_names[i]}**")
        min_w[i] = col2.number_input(
            f"Min", 0.0, 1.0, float(min_w[i]), 0.01,
            key=f"min_{i}", label_visibility="collapsed",
        )
        max_w[i] = col3.number_input(
            f"Max", 0.0, 1.0, float(max_w[i]), 0.01,
            key=f"max_{i}", label_visibility="collapsed",
        )
        col4.markdown(f"Actuel: {current_weights[i]:.1%}")

    st.divider()

    # ---------- Contraintes de groupe ----------
    st.markdown("### Contraintes de groupe")
    st.caption(
        "Limites de la politique de placement du profil de fonds actif. "
        "Modifiez-les ici pour cette session, ou de facon permanente dans la page Profil du fonds."
    )

    profile = get_active_profile()
    group_constraints = []
    for j, gc in enumerate(profile.group_constraints()):
        members = ", ".join(asset_names[i] for i in gc.asset_indices)
        col1, col2, col3 = st.columns([3, 2, 2])
        col1.markdown(f"**{gc.name_fr}**  \n<small>{members}</small>", unsafe_allow_html=True)
        g_min = col2.number_input(
            "Min (%)", 0.0, 100.0, float(gc.min_allocation * 100), 1.0, key=f"gmin_{j}",
        ) / 100
        g_max = col3.number_input(
            "Max (%)", 0.0, 100.0, float(gc.max_allocation * 100), 1.0, key=f"gmax_{j}",
        ) / 100
        group_constraints.append(GroupConstraint(gc.name_fr, gc.asset_indices, g_min, g_max))
    if not group_constraints:
        st.info("Aucune limite de groupe dans le profil actif.")

    st.divider()

    # ---------- Contraintes ESG ----------
    st.markdown("### Contraintes ESG")
    apply_esg = st.checkbox("Appliquer les contraintes ESG", True)
    min_esg_score = 0.0
    if apply_esg:
        min_esg_score = st.slider("Score ESG minimum du portefeuille", 0, 100, 60)

        esg_engine = ESGConstraintEngine(asset_names)
        current_esg = esg_engine.compute_portfolio_esg_score(current_weights)
        current_carbon = esg_engine.compute_carbon_intensity(current_weights)

        col1, col2 = st.columns(2)
        col1.metric("Score ESG actuel", f"{current_esg:.1f}/100")
        col2.metric("Intensite carbone", f"{current_carbon:.0f} tCO2e/M$")

        esg_scores = get_esg_scores()
        esg_df = pd.DataFrame({
            "Classe d'actifs": asset_names,
            "Score ESG": esg_scores,
            "Poids actuel": current_weights * 100,
            "Contribution ESG": (current_weights * esg_scores),
        })
        st.dataframe(esg_df.style.format({
            "Score ESG": "{:.0f}",
            "Poids actuel": "{:.1f}%",
            "Contribution ESG": "{:.1f}",
        }), use_container_width=True, hide_index=True)

    st.divider()

    # ---------- Contrainte de liquidite ----------
    st.markdown("### Contrainte de liquidite")
    min_liquid = st.slider("Minimum en actifs liquides (%)", 0, 100, 0) / 100
    liquid_indices = [i for i, s in enumerate(get_liquidity_scores()) if s >= 0.75]
    st.caption(
        "Actifs liquides (score de liquidite >= 0,75) : "
        + ", ".join(asset_names[i] for i in liquid_indices)
    )
    if min_liquid > 0:
        group_constraints.append(
            GroupConstraint("Actifs liquides", liquid_indices, min_liquid, 1.0)
        )

    # ---------- Contrainte de rotation ----------
    st.markdown("### Contrainte de rotation")
    max_turnover = st.slider("Rotation maximale (%)", 0, 100, 20) / 100

    st.divider()

    # ---------- Sauvegarde et validation ----------
    if st.button("Sauvegarder et valider les contraintes", type="primary", use_container_width=True):
        constraint_set = ConstraintSet(
            min_weights=min_w,
            max_weights=max_w,
            group_constraints=group_constraints,
            turnover_limit=max_turnover,
            current_weights=current_weights,
            esg_min_score=min_esg_score if apply_esg else None,
            esg_scores=get_esg_scores() if apply_esg else None,
        )

        cm = ConstraintManager(n_assets, asset_names)
        is_valid, violations = cm.validate_allocation(current_weights, constraint_set)

        st.session_state.constraint_set = constraint_set

        if is_valid:
            st.success("Toutes les contraintes sont satisfaites par l'allocation actuelle!")
        else:
            st.warning("L'allocation actuelle viole certaines contraintes:")
            for v in violations:
                st.error(f"- {v}")

    # Afficher les contraintes sauvegardees
    if "constraint_set" in st.session_state:
        st.markdown("### Resume des contraintes actives")
        cs = st.session_state.constraint_set
        summary = pd.DataFrame({
            "Classe d'actifs": asset_names,
            "Min (%)": cs.min_weights * 100,
            "Max (%)": cs.max_weights * 100,
            "Actuel (%)": current_weights * 100,
        })
        st.dataframe(summary.style.format({
            "Min (%)": "{:.1f}", "Max (%)": "{:.1f}", "Actuel (%)": "{:.1f}",
        }), use_container_width=True, hide_index=True)


render()

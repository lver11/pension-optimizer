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
    get_esg_scores, DEFAULT_CURRENT_WEIGHTS, PensionFundConfig,
    ASSET_DEFAULTS, ASSET_CLASSES_ORDER,
)
from constraints.manager import ConstraintManager, ConstraintSet, GroupConstraint
from constraints.regulatory import QuebecPensionRegulations
from constraints.esg import ESGConstraintEngine


def render():
    st.title("Gestionnaire de contraintes")

    asset_names = get_asset_names_fr()
    n_assets = len(asset_names)
    current_weights = st.session_state.get("current_weights", DEFAULT_CURRENT_WEIGHTS)

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

    col1, col2 = st.columns(2)
    with col1:
        max_equity = st.slider("Actions totales (max %)", 0, 100, 70) / 100
        max_alternatives = st.slider("Actifs alternatifs (max %)", 0, 100, 40) / 100
    with col2:
        max_pe = st.slider("Capital investissement (max %)", 0, 100, 20) / 100
        min_bonds = st.slider("Obligations totales (min %)", 0, 100, 10) / 100

    group_constraints = [
        GroupConstraint("Actions totales", [0, 1, 2, 3], 0.0, max_equity),
        GroupConstraint("Actifs alternatifs", [7, 8, 9], 0.0, max_alternatives),
        GroupConstraint("Capital investissement", [9], 0.0, max_pe),
        GroupConstraint("Obligations totales", [4, 5, 6], min_bonds, 0.70),
    ]

    st.divider()

    # ---------- Contraintes reglementaires ----------
    st.markdown("### Contraintes reglementaires")
    apply_regulatory = st.checkbox("Appliquer les contraintes reglementaires du Quebec", True)
    if apply_regulatory:
        st.info("""
        **Contraintes reglementaires actives:**
        - Actions totales <= 70%
        - Capital investissement <= 20%
        - Actifs alternatifs (PE + Infra + Immobilier) <= 40%
        - Liquidite minimum >= 2%
        - Obligations totales >= 10%
        """)

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
    min_liquid = st.slider("Minimum en actifs liquides (%)", 0, 30, 5) / 100

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

        if apply_regulatory:
            reg_valid, reg_violations = QuebecPensionRegulations.validate_compliance(
                current_weights, asset_names
            )
            if not reg_valid:
                violations.extend(reg_violations)
                is_valid = False

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

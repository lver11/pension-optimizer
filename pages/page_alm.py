"""
Gestion Actif-Passif (ALM) - pour les fonds qui ont un passif actuariel (regimes PD).
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    get_asset_names_fr, get_expected_returns, get_covariance_matrix,
    get_min_weights, get_max_weights, DEFAULT_CURRENT_WEIGHTS,
    PensionFundConfig, ASSET_DEFAULTS, ASSET_CLASSES_ORDER,
)
from fund_profile import ensure_session_state, get_active_profile
from config import get_policy_weights
from data.generator import MarketDataGenerator
from models.alm import ALMOptimizer, LiabilityProfile
from constraints.manager import ConstraintSet
from constraints.regulatory import PolicyLimits
from visualization.charts import ChartBuilder


def render():
    st.title("Gestion actif-passif (ALM)")

    ensure_session_state()

    config = st.session_state.get("pension_config", PensionFundConfig())
    weights = st.session_state.get("current_weights", get_policy_weights())
    asset_names = get_asset_names_fr()
    profile = get_active_profile()

    if not profile.a_un_passif:
        st.info(
            f"Le profil actif (**{profile.nom}**) n'a pas de passif actuariel. "
            "L'analyse actif-passif s'applique aux regimes a prestations determinees : "
            "saisissez une valeur de passif dans la page Profil du fonds pour l'activer."
        )
        return

    # ---------- Configuration du passif (valeurs du profil) ----------
    st.sidebar.markdown("### Configuration du passif")
    pv_liabilities = st.sidebar.number_input(
        "Valeur actuelle du passif (M$)", 1.0, 500000.0,
        float(profile.valeur_passif / 1e6), 10.0,
    ) * 1e6
    liability_duration = st.sidebar.slider(
        "Duration du passif (annees)", 1.0, 30.0, float(profile.duration_passif), 0.5)
    discount_rate = st.sidebar.slider(
        "Taux d'actualisation (%)", 0.0, 10.0, float(profile.taux_actualisation * 100), 0.1) / 100
    liability_growth = st.sidebar.slider(
        "Croissance du passif (%)", 0.0, 10.0, float(profile.croissance_passif * 100), 0.5) / 100

    liability_profile = LiabilityProfile(
        present_value=pv_liabilities,
        duration=liability_duration,
        discount_rate=discount_rate,
        growth_rate=liability_growth,
    )

    # Durations des actifs
    asset_durations = np.array([
        ASSET_DEFAULTS[ac].duration if ASSET_DEFAULTS[ac].duration is not None else 0.0
        for ac in ASSET_CLASSES_ORDER
    ])

    asset_value = config.valeur_actif
    mu = get_expected_returns()
    cov = get_covariance_matrix()

    alm = ALMOptimizer(
        mu, cov, asset_durations, liability_profile,
        config.taux_sans_risque, asset_names,
        get_min_weights(), get_max_weights(),
    )

    # ---------- Tableau de bord ALM ----------
    st.markdown("### Indicateurs actif-passif")
    funded_ratio = alm.compute_funded_ratio(asset_value)
    surplus = alm.compute_surplus(asset_value)
    duration_gap = alm.compute_duration_gap(weights, asset_value)

    funding_check = PolicyLimits.funding_policy_check(funded_ratio)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ratio de capitalisation", f"{funded_ratio:.1%}")
    col2.metric("Surplus", f"{surplus/1e6:,.0f} M$")
    col3.metric("Ecart de duration", f"{duration_gap:.1f} ans")
    col4.metric("Statut", funding_check["statut"].upper())

    # Alerte selon le statut
    if funding_check["couleur"] == "red":
        st.error(f"**{funding_check['action_recommandee']}**")
    elif funding_check["couleur"] == "orange":
        st.warning(f"**{funding_check['action_recommandee']}**")
    elif funding_check["couleur"] == "yellow":
        st.info(f"**{funding_check['action_recommandee']}**")
    else:
        st.success(f"**{funding_check['action_recommandee']}**")

    st.divider()

    # ---------- Sensibilite aux taux ----------
    st.markdown("### Sensibilite aux taux d'interet")

    rate_scenarios = [-200, -100, -50, 50, 100, 200]
    sensitivity_results = []
    for shock in rate_scenarios:
        sens = alm.compute_interest_rate_sensitivity(weights, asset_value, shock)
        sensitivity_results.append({
            "Choc taux (bps)": shock,
            "Impact actif (M$)": sens["impact_actif"] / 1e6,
            "Impact passif (M$)": sens["impact_passif"] / 1e6,
            "Impact surplus (M$)": sens["impact_surplus"] / 1e6,
            "Impact ratio capit. (pp)": sens["impact_ratio_capit"] * 100,
        })

    sens_df = pd.DataFrame(sensitivity_results)
    st.dataframe(sens_df.style.format({
        "Impact actif (M$)": "{:+,.0f}",
        "Impact passif (M$)": "{:+,.0f}",
        "Impact surplus (M$)": "{:+,.0f}",
        "Impact ratio capit. (pp)": "{:+.1f}",
    }), use_container_width=True, hide_index=True)

    st.divider()

    # ---------- Optimisation du surplus ----------
    st.markdown("### Optimisation du surplus")

    if st.button("Optimiser le surplus", type="primary"):
        constraint_set = ConstraintSet(
            min_weights=get_min_weights(),
            max_weights=get_max_weights(),
            group_constraints=profile.group_constraints(),
        )

        with st.spinner("Optimisation du surplus en cours..."):
            result = alm.optimize_surplus(asset_value, constraint_set=constraint_set)

        if result.status == "optimal":
            st.success("Optimisation terminee!")
            st.session_state.alm_result = result

            col1, col2, col3 = st.columns(3)
            col1.metric("Rendement", f"{result.expected_return:.2%}")
            col2.metric("Volatilite", f"{result.volatility:.2%}")
            col3.metric("Ecart duration",
                        f"{result.metadata.get('duration_gap', 0):.1f} ans")

            fig = ChartBuilder.allocation_comparison_bar(
                weights, result.weights, asset_names,
                "Actuel vs Optimise (surplus)",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Optimisation echouee: {result.status}")

    st.divider()

    # ---------- Couverture du passif ----------
    st.markdown("### Ratio de couverture")
    hedge_target = st.slider("Ratio de couverture cible", 0.50, 1.00, 0.80, 0.05)
    hedge_result = alm.optimize_liability_hedge(asset_value, hedge_target)
    st.info(hedge_result["recommandation"])

    st.divider()

    # ---------- Glide path ----------
    st.markdown("### Trajectoire de desensibilisation (Glide Path)")
    col1, col2 = st.columns(2)
    with col1:
        gp_horizon = st.slider("Horizon glide path (annees)", 5, 20, 10)
    with col2:
        target_fr = st.slider("Ratio cible", 1.00, 1.30, 1.10, 0.05)

    glide_path = alm.optimize_glide_path(
        funded_ratio, gp_horizon, target_fr, asset_value,
    )

    fig_gp = ChartBuilder.glide_path_area(glide_path)
    st.plotly_chart(fig_gp, use_container_width=True)

    gp_df = pd.DataFrame(glide_path)
    st.dataframe(gp_df.style.format({
        "Ratio capitalisation projete": "{:.1%}",
        "Actifs de croissance (%)": "{:.1f}",
        "Actifs de couverture (%)": "{:.1f}",
        "Encaisse (%)": "{:.1f}",
    }), use_container_width=True, hide_index=True)

    # ---------- Flux de tresorerie ----------
    st.markdown("### Flux de tresorerie projetes du passif")
    generator = MarketDataGenerator(seed=42)
    cashflows = generator.generate_liability_cashflows(
        n_years=30, initial_liability=pv_liabilities,
    )

    import plotly.graph_objects as go
    fig_cf = go.Figure()
    fig_cf.add_trace(go.Bar(
        x=cashflows["Annee"], y=cashflows["Cotisations"] / 1e6,
        name="Cotisations", marker_color="green",
    ))
    fig_cf.add_trace(go.Bar(
        x=cashflows["Annee"], y=-cashflows["Prestations"] / 1e6,
        name="Prestations", marker_color="red",
    ))
    fig_cf.add_trace(go.Scatter(
        x=cashflows["Annee"], y=cashflows["Flux_net"] / 1e6,
        mode="lines+markers", name="Flux net",
        line=dict(color="blue", width=2),
    ))
    fig_cf.update_layout(
        title="Flux de tresorerie annuels projetes",
        xaxis_title="Annee",
        yaxis_title="Montant (M$)",
        barmode="relative",
        height=400,
    )
    st.plotly_chart(fig_cf, use_container_width=True)


render()

"""
Simulation Monte Carlo pour les projections du portefeuille.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    get_asset_names_fr, get_expected_returns, get_covariance_matrix,
    DEFAULT_CURRENT_WEIGHTS, PensionFundConfig,
)
from fund_profile import ensure_session_state, get_active_profile
from config import get_policy_weights
from data.generator import MarketDataGenerator
from models.monte_carlo import MonteCarloSimulator
from visualization.charts import ChartBuilder


def render():
    st.title("Simulation Monte Carlo")

    ensure_session_state()

    config = st.session_state.get("pension_config", PensionFundConfig())
    weights = st.session_state.get("current_weights", get_policy_weights())

    profile = get_active_profile()

    # Parametres (valeurs initiales tirees du profil de fonds actif)
    st.markdown("### Parametres de simulation")
    st.caption(f"Valeurs par defaut du profil : **{profile.nom}** (modifiables pour cette simulation).")
    col1, col2 = st.columns(2)
    with col1:
        horizon = st.slider("Horizon (annees)", 5, 40, int(profile.horizon_annees))
        n_sims = st.select_slider(
            "Nombre de simulations",
            [1000, 2500, 5000, 10000, 25000], 5000,
        )
    with col2:
        initial_assets = st.number_input(
            "Valeur initiale du portefeuille (M$)",
            1.0, 500000.0, float(config.valeur_actif / 1e6), 10.0,
        ) * 1e6

    col1, col2 = st.columns(2)
    with col1:
        annual_contribution = st.number_input(
            "Entrees annuelles - cotisations, dons, souscriptions (M$)",
            0.0, 50000.0, float(profile.cotisations_annuelles / 1e6), 1.0,
        ) * 1e6
        contribution_growth = st.slider(
            "Croissance des entrees (%)", -10.0, 10.0,
            float(profile.croissance_cotisations * 100), 0.5,
        ) / 100
    with col2:
        annual_benefit = st.number_input(
            "Sorties annuelles - prestations, decaissements, rachats (M$)",
            0.0, 50000.0, float(profile.prestations_annuelles / 1e6), 1.0,
        ) * 1e6
        benefit_growth = st.slider(
            "Croissance des sorties (%)", -10.0, 10.0,
            float(profile.croissance_prestations * 100), 0.5,
        ) / 100

    # Lancer la simulation
    if st.button("Lancer la simulation", type="primary", use_container_width=True):
        mu = get_expected_returns()
        cov = get_covariance_matrix()

        simulator = MonteCarloSimulator(
            weights=weights,
            expected_returns=mu,
            cov_matrix=cov,
            initial_assets=initial_assets,
            annual_contribution=annual_contribution,
            annual_benefit=annual_benefit,
            benefit_growth_rate=benefit_growth,
            contribution_growth_rate=contribution_growth,
            n_simulations=n_sims,
            seed=42,
        )

        with st.spinner(f"Simulation de {n_sims:,} trajectoires sur {horizon} ans..."):
            mc_result = simulator.simulate(horizon)
            st.session_state.mc_result = mc_result

        st.success(f"Simulation terminee! ({n_sims:,} trajectoires)")

    # Affichage des resultats
    if "mc_result" in st.session_state:
        mc = st.session_state.mc_result
        stats = mc.compute_statistics()

        # KPIs
        st.markdown("### Statistiques sommaires")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Valeur mediane (fin)", f"{stats['median_assets']/1e9:.2f} G$")
        col2.metric("Rendement annuel median", f"{stats['median_annual_return']:.1%}")
        col3.metric("Prob. de perte", f"{stats['prob_loss']:.1%}")
        col4.metric("Prob. de ruine", f"{stats['prob_ruin']:.1%}")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Valeur (5e perc.)", f"{stats['p5_assets']/1e9:.2f} G$")
        col2.metric("Valeur (95e perc.)", f"{stats['p95_assets']/1e9:.2f} G$")
        col3.metric("Valeur moyenne (fin)", f"{stats['mean_assets']/1e9:.2f} G$")
        col4.metric("Simulations", f"{mc.n_simulations:,}")

        st.divider()

        # Graphique actifs
        st.markdown("### Projection de la valeur du portefeuille")
        asset_fan = mc.get_fan_data(mc.asset_paths)
        fig_assets = ChartBuilder.monte_carlo_fan_chart(
            asset_fan, mc.years,
            title="Projection de la valeur du portefeuille",
            y_label="Valeur (M$)",
            scale=1e6,
        )
        st.plotly_chart(fig_assets, use_container_width=True)

        # Distribution terminale
        st.markdown("### Distribution de la valeur terminale")
        terminal_assets = mc.asset_paths[:, -1] / 1e9
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=terminal_assets,
            nbinsx=50,
            marker_color="rgba(31, 119, 180, 0.7)",
            name="Distribution",
        ))
        fig_dist.add_vline(
            x=initial_assets / 1e9, line_dash="dash", line_color="red",
            annotation_text="Valeur initiale",
        )
        fig_dist.update_layout(
            title="Distribution de la valeur du portefeuille a l'horizon",
            xaxis_title="Valeur (G$)",
            yaxis_title="Frequence",
            height=400,
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        # Comparaison avec l'allocation optimisee
        if "optimization_result" in st.session_state and st.session_state.optimization_result is not None:
            st.markdown("### Comparaison: Actuel vs Optimise")
            opt_result = st.session_state.optimization_result

            if st.button("Simuler l'allocation optimisee"):
                mu = get_expected_returns()
                cov = get_covariance_matrix()
                sim_opt = MonteCarloSimulator(
                    opt_result.weights, mu, cov,
                    initial_assets=initial_assets,
                    annual_contribution=annual_contribution,
                    annual_benefit=annual_benefit,
                    benefit_growth_rate=benefit_growth,
                    contribution_growth_rate=contribution_growth,
                    n_simulations=n_sims, seed=123,
                )
                mc_opt = sim_opt.simulate(mc.horizon_years)
                stats_opt = mc_opt.compute_statistics()

                comp_df = pd.DataFrame({
                    "Metrique": [
                        "Valeur mediane (G$)",
                        "Rendement annuel median",
                        "Prob. de perte",
                        "Valeur 5e perc. (G$)",
                    ],
                    "Allocation actuelle": [
                        f"{stats['median_assets']/1e9:.2f}",
                        f"{stats['median_annual_return']:.1%}",
                        f"{stats['prob_loss']:.1%}",
                        f"{stats['p5_assets']/1e9:.2f}",
                    ],
                    "Allocation optimisee": [
                        f"{stats_opt['median_assets']/1e9:.2f}",
                        f"{stats_opt['median_annual_return']:.1%}",
                        f"{stats_opt['prob_loss']:.1%}",
                        f"{stats_opt['p5_assets']/1e9:.2f}",
                    ],
                })
                st.dataframe(comp_df, use_container_width=True, hide_index=True)


render()

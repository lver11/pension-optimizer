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
from data.generator import MarketDataGenerator
from models.monte_carlo import MonteCarloSimulator
from visualization.charts import ChartBuilder


def render():
    st.title("Simulation Monte Carlo")

    if "returns_data" not in st.session_state or st.session_state.returns_data is None:
        generator = MarketDataGenerator(seed=42)
        st.session_state.returns_data = generator.generate_returns(n_years=20, frequency="monthly")
        st.session_state.current_weights = DEFAULT_CURRENT_WEIGHTS.copy()
        st.session_state.pension_config = PensionFundConfig()

    config = st.session_state.get("pension_config", PensionFundConfig())
    weights = st.session_state.get("current_weights", DEFAULT_CURRENT_WEIGHTS)

    # Parametres
    st.markdown("### Parametres de simulation")
    col1, col2 = st.columns(2)
    with col1:
        horizon = st.slider("Horizon (annees)", 5, 40, 20)
        n_sims = st.select_slider(
            "Nombre de simulations",
            [1000, 2500, 5000, 10000, 25000], 5000,
        )
    with col2:
        initial_assets = st.number_input(
            "Valeur initiale du portefeuille (M$)",
            100.0, 10000.0, config.valeur_actif / 1e6, 50.0,
        ) * 1e6
        annual_contribution = st.number_input(
            "Cotisations annuelles (M$)", 0.0, 500.0, 40.0, 5.0,
        ) * 1e6

    col1, col2 = st.columns(2)
    with col1:
        annual_benefit = st.number_input(
            "Retraits annuels (M$)", 0.0, 500.0, 57.0, 5.0,
        ) * 1e6
    with col2:
        benefit_growth = st.slider(
            "Croissance des retraits (%)", 0.0, 8.0, 3.0, 0.5,
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

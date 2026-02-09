"""
Simulation Monte Carlo pour les projections du fonds de pension.
"""

import streamlit as st
import numpy as np
import pandas as pd
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
            "Valeur initiale de l'actif (M$)",
            100.0, 10000.0, config.valeur_actif / 1e6, 50.0,
        ) * 1e6
        initial_liabilities = st.number_input(
            "Valeur initiale du passif (M$)",
            100.0, 10000.0, config.valeur_passif / 1e6, 50.0,
        ) * 1e6

    col1, col2 = st.columns(2)
    with col1:
        annual_contribution = st.number_input(
            "Cotisations annuelles (M$)", 0.0, 500.0, 40.0, 5.0,
        ) * 1e6
        benefit_growth = st.slider(
            "Croissance des prestations (%)", 0.0, 8.0, 3.0, 0.5,
        ) / 100
    with col2:
        annual_benefit = st.number_input(
            "Prestations annuelles (M$)", 0.0, 500.0, 57.0, 5.0,
        ) * 1e6
        liability_growth = st.slider(
            "Croissance du passif (%)", 0.0, 10.0, 5.0, 0.5,
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
            initial_liabilities=initial_liabilities,
            annual_contribution=annual_contribution,
            annual_benefit=annual_benefit,
            liability_growth_rate=liability_growth,
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
        col1.metric("Ratio capit. median (fin)", f"{stats['median_fr']:.1%}")
        col2.metric("Prob. sous-capitalisation", f"{stats['prob_underfunded']:.1%}")
        col3.metric("Valeur mediane actif (fin)", f"{stats['median_assets']/1e9:.2f} G$")
        col4.metric("Prob. ratio < 80%", f"{stats['prob_severely_underfunded']:.1%}")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Ratio capit. (5e perc.)", f"{stats['p5_fr']:.1%}")
        col2.metric("Ratio capit. (95e perc.)", f"{stats['p95_fr']:.1%}")
        col3.metric("Actif (5e perc.)", f"{stats['p5_assets']/1e9:.2f} G$")
        col4.metric("Surplus VaR 5%", f"{stats['surplus_var_5']/1e6:,.0f} M$")

        st.divider()

        # Graphique actifs
        st.markdown("### Projection de la valeur de l'actif")
        asset_fan = mc.get_fan_data(mc.asset_paths)
        fig_assets = ChartBuilder.monte_carlo_fan_chart(
            asset_fan, mc.years,
            title="Projection de la valeur de l'actif",
            y_label="Valeur (M$)",
            scale=1e6,
        )
        # Ajouter la ligne du passif
        liability_median = np.median(mc.liability_paths, axis=0)
        import plotly.graph_objects as go
        fig_assets.add_trace(go.Scatter(
            x=mc.years, y=liability_median / 1e6,
            mode="lines", name="Passif (mediane)",
            line=dict(color="red", width=2, dash="dash"),
        ))
        st.plotly_chart(fig_assets, use_container_width=True)

        # Graphique ratio de capitalisation
        st.markdown("### Projection du ratio de capitalisation")
        fr_fan = mc.get_fan_data(mc.funded_ratio_paths)
        fig_fr = ChartBuilder.funded_ratio_projection(fr_fan, mc.years)
        st.plotly_chart(fig_fr, use_container_width=True)

        # Distribution terminale du ratio de capitalisation
        st.markdown("### Distribution du ratio de capitalisation terminal")
        terminal_fr = mc.funded_ratio_paths[:, -1]
        import plotly.graph_objects as go
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=terminal_fr * 100,
            nbinsx=50,
            marker_color="rgba(31, 119, 180, 0.7)",
            name="Distribution",
        ))
        fig_dist.add_vline(x=100, line_dash="dash", line_color="red",
                           annotation_text="100% capitalisation")
        fig_dist.add_vline(x=80, line_dash="dash", line_color="orange",
                           annotation_text="80% seuil critique")
        fig_dist.update_layout(
            title="Distribution du ratio de capitalisation a l'horizon",
            xaxis_title="Ratio de capitalisation (%)",
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
                    initial_assets, initial_liabilities,
                    annual_contribution, annual_benefit,
                    liability_growth, benefit_growth,
                    n_sims, seed=123,
                )
                mc_opt = sim_opt.simulate(mc.horizon_years)
                stats_opt = mc_opt.compute_statistics()

                comp_df = pd.DataFrame({
                    "Metrique": [
                        "Ratio capit. median",
                        "Prob. sous-capitalisation",
                        "Valeur mediane actif (G$)",
                        "Ratio capit. 5e perc.",
                    ],
                    "Allocation actuelle": [
                        f"{stats['median_fr']:.1%}",
                        f"{stats['prob_underfunded']:.1%}",
                        f"{stats['median_assets']/1e9:.2f}",
                        f"{stats['p5_fr']:.1%}",
                    ],
                    "Allocation optimisee": [
                        f"{stats_opt['median_fr']:.1%}",
                        f"{stats_opt['prob_underfunded']:.1%}",
                        f"{stats_opt['median_assets']/1e9:.2f}",
                        f"{stats_opt['p5_fr']:.1%}",
                    ],
                })
                st.dataframe(comp_df, use_container_width=True, hide_index=True)


render()

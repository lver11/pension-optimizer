"""
Page de la frontiere efficiente interactive.
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    get_asset_names_fr, get_expected_returns, get_covariance_matrix,
    get_min_weights, get_max_weights, DEFAULT_CURRENT_WEIGHTS, PensionFundConfig,
)
from data.generator import MarketDataGenerator
from models.efficient_frontier import EfficientFrontierComputer
from constraints.manager import ConstraintSet
from constraints.regulatory import QuebecPensionRegulations
from risk.covariance import CovarianceEstimator
from visualization.charts import ChartBuilder


def render():
    st.title("Frontiere efficiente")

    if "returns_data" not in st.session_state or st.session_state.returns_data is None:
        generator = MarketDataGenerator(seed=42)
        st.session_state.returns_data = generator.generate_returns(n_years=20, frequency="monthly")
        st.session_state.current_weights = DEFAULT_CURRENT_WEIGHTS.copy()
        st.session_state.pension_config = PensionFundConfig()

    asset_names = get_asset_names_fr()
    mu = get_expected_returns()
    rf = st.session_state.get("pension_config", PensionFundConfig()).taux_sans_risque
    returns_data = st.session_state.returns_data

    # Estimation covariance
    cov_method = st.sidebar.selectbox("Methode covariance", [
        "Ledoit-Wolf", "Echantillon", "EWMA"
    ])
    method_map = {"Ledoit-Wolf": "ledoit_wolf", "Echantillon": "sample", "EWMA": "ewma"}
    cov_matrix = CovarianceEstimator.estimate(returns_data, method_map[cov_method])

    # Options
    col1, col2, col3 = st.columns(3)
    with col1:
        frontier_type = st.selectbox("Type de frontiere", [
            "Moyenne-Variance", "Moyenne-CVaR"
        ])
    with col2:
        n_points = st.slider("Nombre de points", 20, 100, 50)
    with col3:
        apply_constraints = st.checkbox("Contraintes reglementaires", True)

    show_current = st.checkbox("Afficher le portefeuille actuel", True)
    show_tangency = st.checkbox("Afficher le portefeuille tangent", True)
    show_unconstrained = st.checkbox("Afficher la frontiere non contrainte", False)
    show_cml = st.checkbox("Afficher la ligne du marche des capitaux", True)

    # Contraintes
    constraint_set = None
    if apply_constraints:
        constraint_set = ConstraintSet(
            min_weights=get_min_weights(),
            max_weights=get_max_weights(),
            group_constraints=QuebecPensionRegulations.get_group_constraints(),
        )

    # Calcul
    if st.button("Calculer la frontiere", type="primary", use_container_width=True):
        with st.spinner("Calcul de la frontiere efficiente..."):
            scenarios = None
            if frontier_type == "Moyenne-CVaR":
                gen = MarketDataGenerator(seed=42)
                scenarios_df = gen.generate_returns(n_years=10, frequency="monthly")
                scenarios = scenarios_df.values

            ef = EfficientFrontierComputer(
                mu, cov_matrix, scenarios, rf, asset_names,
                get_min_weights(), get_max_weights(),
            )

            try:
                if frontier_type == "Moyenne-Variance":
                    frontier_df = ef.compute_mv_frontier(n_points, constraint_set)
                else:
                    frontier_df = ef.compute_cvar_frontier(n_points, constraint_set)

                st.session_state.frontier_data = frontier_df

                # Portefeuille tangent
                tangency = None
                if show_tangency:
                    tangency_result = ef.find_tangency_portfolio(constraint_set)
                    tangency = (tangency_result.expected_return, tangency_result.volatility)
                    st.session_state.tangency_result = tangency_result

                # CML
                cml_data = None
                if show_cml and tangency is not None:
                    cml_data = ef.compute_capital_market_line(st.session_state.tangency_result)

                # Frontiere non contrainte
                if show_unconstrained:
                    unconstrained_df = ef.compute_unconstrained_frontier(n_points)
                    st.session_state.unconstrained_frontier = unconstrained_df

                st.success("Frontiere calculee!")

            except Exception as e:
                st.error(f"Erreur: {str(e)}")
                return

    # Affichage
    if "frontier_data" in st.session_state:
        frontier_df = st.session_state.frontier_data

        current_weights = st.session_state.get("current_weights", DEFAULT_CURRENT_WEIGHTS)
        curr_ret = current_weights @ mu
        curr_vol = np.sqrt(current_weights @ cov_matrix @ current_weights)
        current_portfolio = (curr_ret, curr_vol) if show_current else None

        tangency = None
        if show_tangency and "tangency_result" in st.session_state:
            tr = st.session_state.tangency_result
            tangency = (tr.expected_return, tr.volatility)

        cml_data = None
        if show_cml and "tangency_result" in st.session_state:
            ef_temp = EfficientFrontierComputer(mu, cov_matrix, None, rf, asset_names)
            cml_data = ef_temp.compute_capital_market_line(st.session_state.tangency_result)

        optimized = None
        if "optimization_result" in st.session_state and st.session_state.optimization_result is not None:
            opt = st.session_state.optimization_result
            optimized = (opt.expected_return, opt.volatility)

        risk_measure = "cvar" if frontier_type == "Moyenne-CVaR" else "volatilite"
        fig = ChartBuilder.efficient_frontier(
            frontier_df,
            current_portfolio=current_portfolio,
            optimized_portfolio=optimized,
            tangency_portfolio=tangency,
            cml_data=cml_data,
            risk_measure=risk_measure,
        )

        # Superposer la frontiere non contrainte
        if show_unconstrained and "unconstrained_frontier" in st.session_state:
            unc = st.session_state.unconstrained_frontier
            fig.add_trace(go.Scatter(
                x=unc["volatility"] * 100,
                y=unc["return"] * 100,
                mode="lines",
                name="Frontiere non contrainte",
                line=dict(color="lightgray", width=2, dash="dot"),
            ))

        st.plotly_chart(fig, use_container_width=True)

        # Portefeuille selectionne
        st.markdown("### Explorer un point de la frontiere")
        if len(frontier_df) > 0:
            min_ret = frontier_df["return"].min() * 100
            max_ret = frontier_df["return"].max() * 100
            selected_return = st.slider(
                "Rendement cible (%)", float(min_ret), float(max_ret),
                float((min_ret + max_ret) / 2), 0.1,
            )

            closest_idx = (frontier_df["return"] * 100 - selected_return).abs().idxmin()
            selected_point = frontier_df.iloc[closest_idx]

            col1, col2, col3 = st.columns(3)
            col1.metric("Rendement", f"{selected_point['return']:.2%}")
            col2.metric("Volatilite", f"{selected_point['volatility']:.2%}")
            col3.metric("Sharpe", f"{selected_point.get('sharpe', 0):.3f}")

            # Poids du point selectionne
            weight_cols = [c for c in frontier_df.columns if c.startswith("w_")]
            if weight_cols:
                w_vals = selected_point[weight_cols].values
                w_names = [c.replace("w_", "") for c in weight_cols]
                fig_bar = ChartBuilder.allocation_pie(w_vals, w_names, "Allocation du point selectionne")
                st.plotly_chart(fig_bar, use_container_width=True)

        # Tableau de la frontiere
        with st.expander("Voir les donnees de la frontiere"):
            display_df = frontier_df[["return", "volatility", "sharpe"]].copy()
            display_df.columns = ["Rendement (%)", "Volatilite (%)", "Sharpe"]
            display_df["Rendement (%)"] *= 100
            display_df["Volatilite (%)"] *= 100
            st.dataframe(display_df.style.format({
                "Rendement (%)": "{:.2f}",
                "Volatilite (%)": "{:.2f}",
                "Sharpe": "{:.3f}",
            }), use_container_width=True)


try:
    import plotly.graph_objects as go
except ImportError:
    pass

render()

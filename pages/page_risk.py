"""
Analytique de risque du portefeuille.
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
from risk.metrics import RiskMetrics
from risk.stress_testing import StressTester, HISTORICAL_SCENARIOS
from visualization.charts import ChartBuilder


def render():
    st.title("Analytique de risque")

    if "returns_data" not in st.session_state or st.session_state.returns_data is None:
        generator = MarketDataGenerator(seed=42)
        st.session_state.returns_data = generator.generate_returns(n_years=20, frequency="monthly")
        st.session_state.current_weights = DEFAULT_CURRENT_WEIGHTS.copy()
        st.session_state.pension_config = PensionFundConfig()

    asset_names = get_asset_names_fr()
    weights = st.session_state.get("current_weights", DEFAULT_CURRENT_WEIGHTS)
    returns_data = st.session_state.returns_data
    config = st.session_state.get("pension_config", PensionFundConfig())

    # Rendements du portefeuille
    portfolio_returns = returns_data.values @ weights
    metrics = RiskMetrics.compute_all(portfolio_returns, config.taux_sans_risque)

    # ---------- KPIs ----------
    st.markdown("### Metriques de risque principales")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("VaR (95%)", f"{metrics.get('VaR (historique)', 0):.2%}")
    col2.metric("CVaR (95%)", f"{metrics.get('CVaR', 0):.2%}")
    col3.metric("Ratio de Sharpe", f"{metrics.get('Ratio de Sharpe', 0):.3f}")
    col4.metric("Perte maximale", f"{metrics.get('Perte maximale', 0):.2%}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ratio de Sortino", f"{metrics.get('Ratio de Sortino', 0):.3f}")
    col2.metric("Ratio de Calmar", f"{metrics.get('Ratio de Calmar', 0):.3f}")
    col3.metric("Ratio Omega", f"{metrics.get('Ratio Omega', 0):.3f}")
    col4.metric("Asymetrie", f"{metrics.get('Asymetrie', 0):.3f}")

    st.divider()

    # ---------- Onglets ----------
    tab1, tab2, tab3, tab4 = st.tabs([
        "Distribution des rendements",
        "Tests de tension historiques",
        "Tests de tension parametriques",
        "Analyse de perte maximale",
    ])

    with tab1:
        st.markdown("### Distribution des rendements du portefeuille")
        fig = ChartBuilder.return_distribution(
            portfolio_returns,
            var_level=metrics.get("VaR (historique)"),
            cvar_level=metrics.get("CVaR"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Tableau des metriques detaillees
        st.markdown("### Toutes les metriques")
        metrics_df = pd.DataFrame([
            {"Metrique": k, "Valeur": f"{v:.4f}" if abs(v) < 10 else f"{v:.2f}"}
            for k, v in metrics.items()
        ])
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    with tab2:
        st.markdown("### Tests de tension historiques")

        stress_tester = StressTester(asset_names)

        # Tous les scenarios
        all_results = stress_tester.run_all_historical(weights, config.valeur_actif)
        st.dataframe(all_results.style.format({
            "Impact (%)": "{:+.2f}",
            "Perte (M$)": "{:+,.0f}",
        }), use_container_width=True, hide_index=True)

        fig_stress = ChartBuilder.stress_test_waterfall(all_results)
        st.plotly_chart(fig_stress, use_container_width=True)

        # Detail d'un scenario
        st.markdown("### Detail d'un scenario")
        selected_scenario = st.selectbox(
            "Choisir un scenario",
            list(HISTORICAL_SCENARIOS.keys()),
            format_func=lambda k: HISTORICAL_SCENARIOS[k]["nom_fr"],
        )
        detail = stress_tester.run_historical_stress(weights, selected_scenario, config.valeur_actif)

        col1, col2 = st.columns(2)
        col1.metric("Impact portefeuille", f"{detail['impact_portefeuille']:.2%}")
        col2.metric("Perte absolue", f"{detail['perte_absolue']/1e6:,.0f} M$")

        st.markdown(f"*{detail['description']}*")

        contrib_df = pd.DataFrame({
            "Classe d'actifs": list(detail["contributions_par_actif"].keys()),
            "Choc (%)": [v * 100 for v in detail["chocs_par_actif"].values()],
            "Contribution (%)": [v * 100 for v in detail["contributions_par_actif"].values()],
        })
        st.dataframe(contrib_df.style.format({
            "Choc (%)": "{:+.1f}",
            "Contribution (%)": "{:+.2f}",
        }), use_container_width=True, hide_index=True)

        # Impact sur le ratio de capitalisation
        st.markdown("### Impact sur le ratio de capitalisation")
        fr_results = stress_tester.stress_funded_ratio(
            weights, config.valeur_actif, config.valeur_passif,
        )
        st.dataframe(fr_results.style.format({
            "Impact actif (%)": "{:+.2f}",
            "Nouvel actif (M$)": "{:,.0f}",
            "Nouveau passif (M$)": "{:,.0f}",
            "Ratio capit. actuel": "{:.2%}",
            "Ratio capit. stresse": "{:.2%}",
            "Variation ratio": "{:+.2f} pp",
        }), use_container_width=True, hide_index=True)

    with tab3:
        st.markdown("### Test de tension parametrique")
        st.caption("Definissez vos propres chocs de marche.")

        col1, col2 = st.columns(2)
        with col1:
            equity_shock = st.slider("Choc actions (%)", -50.0, 10.0, -20.0, 1.0) / 100
            rate_shock = st.slider("Choc taux (bps)", -200, 300, 100, 10)
        with col2:
            spread_shock = st.slider("Choc spreads (bps)", -50, 300, 50, 10)
            inflation_shock = st.slider("Choc inflation (%)", -2.0, 5.0, 2.0, 0.5) / 100

        if st.button("Executer le test parametrique", type="primary"):
            param_result = stress_tester.run_parametric_stress(
                weights, equity_shock, rate_shock, spread_shock,
                inflation_shock, config.valeur_actif,
            )

            col1, col2 = st.columns(2)
            col1.metric("Impact portefeuille", f"{param_result['impact_portefeuille']:.2%}")
            col2.metric("Perte absolue", f"{param_result['perte_absolue']/1e6:,.0f} M$")

            contrib_df = pd.DataFrame({
                "Classe d'actifs": list(param_result["contributions_par_actif"].keys()),
                "Choc (%)": [v * 100 for v in param_result["chocs_par_actif"].values()],
                "Contribution (%)": [v * 100 for v in param_result["contributions_par_actif"].values()],
            })
            st.dataframe(contrib_df.style.format({
                "Choc (%)": "{:+.1f}",
                "Contribution (%)": "{:+.2f}",
            }), use_container_width=True, hide_index=True)

        # Test de tension inverse
        st.markdown("### Test de tension inverse")
        loss_threshold = st.slider("Seuil de perte (%)", -50, -5, -20, 1) / 100
        if st.button("Trouver le scenario minimal"):
            cov = get_covariance_matrix()
            reverse_result = stress_tester.reverse_stress_test(weights, loss_threshold, cov)

            if "erreur" not in reverse_result:
                st.metric("Norme du choc minimal", f"{reverse_result['norme_choc']:.3f}")
                shock_df = pd.DataFrame({
                    "Classe d'actifs": list(reverse_result["chocs_minimaux"].keys()),
                    "Choc minimal (%)": [v * 100 for v in reverse_result["chocs_minimaux"].values()],
                })
                st.dataframe(shock_df.style.format({
                    "Choc minimal (%)": "{:+.1f}",
                }), use_container_width=True, hide_index=True)
            else:
                st.error("Le test de tension inverse n'a pas converge.")

    with tab4:
        st.markdown("### Analyse de perte maximale (Drawdown)")
        fig_dd = ChartBuilder.drawdown_chart(portfolio_returns, returns_data.index)
        st.plotly_chart(fig_dd, use_container_width=True)

        mdd, peak, trough = RiskMetrics.maximum_drawdown(portfolio_returns)
        col1, col2, col3 = st.columns(3)
        col1.metric("Perte maximale", f"{mdd:.2%}")
        col2.metric("Debut du pic", str(returns_data.index[peak].date()))
        col3.metric("Creux", str(returns_data.index[min(trough, len(returns_data.index)-1)].date()))


render()

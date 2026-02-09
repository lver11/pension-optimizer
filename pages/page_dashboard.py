"""
Tableau de bord principal du portefeuille.
KPIs, allocation, contribution au risque, correlation, performance.
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    get_asset_names_fr, get_expected_returns, get_covariance_matrix,
    get_min_weights, get_max_weights, DEFAULT_CURRENT_WEIGHTS,
    DEFAULT_CORRELATION_MATRIX, PensionFundConfig, ASSET_DEFAULTS, ASSET_CLASSES_ORDER,
)
from data.generator import MarketDataGenerator
from risk.metrics import RiskMetrics
from visualization.charts import ChartBuilder


def render():
    st.title("Tableau de bord")

    # --- Initialisation des donnees ---
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

    # --- KPIs principaux ---
    st.markdown("### Indicateurs cles de performance")
    col1, col2, col3, col4, col5 = st.columns(5)

    # Ratio de capitalisation
    fr = config.valeur_actif / config.valeur_passif
    col1.metric(
        "Ratio de capitalisation",
        f"{fr:.1%}",
        delta=f"{'Excedentaire' if fr >= 1.0 else 'Deficitaire'}",
        delta_color="normal" if fr >= 1.0 else "inverse",
    )
    col2.metric(
        "Rendement attendu",
        f"{metrics.get('Rendement annualise', 0):.2%}",
    )
    col3.metric(
        "Volatilite",
        f"{metrics.get('Volatilite annualisee', 0):.2%}",
    )
    col4.metric(
        "Ratio de Sharpe",
        f"{metrics.get('Ratio de Sharpe', 0):.3f}",
    )
    col5.metric(
        "VaR (95%)",
        f"{metrics.get('VaR (historique)', 0):.2%}",
    )

    # Deuxieme ligne de KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Valeur de l'actif", f"{config.valeur_actif/1e6:,.0f} M$")
    col2.metric("Valeur du passif", f"{config.valeur_passif/1e6:,.0f} M$")
    surplus = config.valeur_actif - config.valeur_passif
    col3.metric("Surplus (deficit)", f"{surplus/1e6:,.0f} M$")
    col4.metric("CVaR (95%)", f"{metrics.get('CVaR', 0):.2%}")
    col5.metric("Perte maximale", f"{metrics.get('Perte maximale', 0):.2%}")

    st.divider()

    # --- Section graphiques ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "Allocation et risque",
        "Performance",
        "Correlations",
        "Metriques detaillees",
    ])

    with tab1:
        col_left, col_right = st.columns(2)

        with col_left:
            # Diagramme en anneau de l'allocation
            fig_pie = ChartBuilder.allocation_pie(weights, asset_names)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_right:
            # Contribution au risque
            cov_matrix = get_covariance_matrix()
            port_vol = np.sqrt(weights @ cov_matrix @ weights)
            if port_vol > 1e-10:
                marginal_risk = cov_matrix @ weights
                risk_contributions = weights * marginal_risk / port_vol
            else:
                risk_contributions = np.zeros(len(weights))

            fig_risk = ChartBuilder.risk_contribution_bar(risk_contributions, asset_names)
            st.plotly_chart(fig_risk, use_container_width=True)

        # Tableau d'allocation detaille
        st.markdown("### Detail de l'allocation")
        mu = get_expected_returns()
        vols = np.array([ASSET_DEFAULTS[ac].volatility for ac in ASSET_CLASSES_ORDER])
        liquidity = np.array([ASSET_DEFAULTS[ac].liquidity_score for ac in ASSET_CLASSES_ORDER])

        alloc_df = pd.DataFrame({
            "Classe d'actifs": asset_names,
            "Poids (%)": weights * 100,
            "Rendement attendu (%)": mu * 100,
            "Volatilite (%)": vols * 100,
            "Score de liquidite": liquidity,
            "Contrib. risque (%)": (risk_contributions / np.sum(np.abs(risk_contributions)) * 100
                                    if np.sum(np.abs(risk_contributions)) > 0
                                    else np.zeros(len(weights))),
        })
        st.dataframe(
            alloc_df.style.format({
                "Poids (%)": "{:.1f}",
                "Rendement attendu (%)": "{:.1f}",
                "Volatilite (%)": "{:.1f}",
                "Score de liquidite": "{:.2f}",
                "Contrib. risque (%)": "{:.1f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    with tab2:
        # Performance cumulee
        st.markdown("### Performance cumulee des classes d'actifs")
        fig_perf = ChartBuilder.cumulative_returns_chart(returns_data)
        st.plotly_chart(fig_perf, use_container_width=True)

        # Performance du portefeuille
        st.markdown("### Performance cumulee du portefeuille")
        port_cum = np.cumprod(1 + portfolio_returns)
        import plotly.graph_objects as go
        fig_port = go.Figure()
        fig_port.add_trace(go.Scatter(
            x=returns_data.index,
            y=port_cum,
            mode="lines",
            name="Portefeuille",
            line=dict(color="#1f77b4", width=3),
        ))
        fig_port.update_layout(
            title=dict(text="Performance cumulee du portefeuille", font=dict(size=16)),
            xaxis_title="Date",
            yaxis_title="Valeur (base 1.0)",
            height=400,
            margin=dict(t=60, b=60),
            hovermode="x unified",
        )
        st.plotly_chart(fig_port, use_container_width=True)

        # Distribution des rendements
        fig_dist = ChartBuilder.return_distribution(
            portfolio_returns,
            var_level=metrics.get("VaR (historique)"),
            cvar_level=metrics.get("CVaR"),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    with tab3:
        st.markdown("### Matrice de correlation des classes d'actifs")
        # Correlation empirique
        corr_empirique = returns_data.corr().values
        fig_corr = ChartBuilder.correlation_heatmap(corr_empirique, asset_names, "Correlation empirique")
        st.plotly_chart(fig_corr, use_container_width=True)

        # Correlation theorique
        with st.expander("Voir la matrice de correlation theorique"):
            fig_corr_theo = ChartBuilder.correlation_heatmap(
                DEFAULT_CORRELATION_MATRIX, asset_names, "Correlation theorique (defaut)"
            )
            st.plotly_chart(fig_corr_theo, use_container_width=True)

    with tab4:
        st.markdown("### Toutes les metriques de risque")
        metrics_df = pd.DataFrame([
            {"Metrique": k, "Valeur": f"{v:.4f}" if abs(v) < 10 else f"{v:.2f}"}
            for k, v in metrics.items()
        ])
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

        # Rendements mensuels recents
        st.markdown("### Rendements mensuels recents du portefeuille")
        n_recent = 24
        recent_returns = portfolio_returns[-n_recent:]
        recent_dates = returns_data.index[-n_recent:]
        recent_df = pd.DataFrame({
            "Date": [d.strftime("%Y-%m") for d in recent_dates],
            "Rendement (%)": recent_returns * 100,
        })
        recent_df["Couleur"] = recent_df["Rendement (%)"].apply(lambda x: "Positif" if x >= 0 else "Negatif")
        st.dataframe(
            recent_df.style.format({"Rendement (%)": "{:+.2f}"}),
            use_container_width=True,
            hide_index=True,
        )

        # Statistiques par classe d'actifs
        st.markdown("### Statistiques par classe d'actifs")
        stats_rows = []
        for i, name in enumerate(asset_names):
            r = returns_data.iloc[:, i].values
            ann_ret = RiskMetrics.annualized_return(r)
            ann_vol = RiskMetrics.annualized_volatility(r)
            sharpe = RiskMetrics.sharpe_ratio(r, config.taux_sans_risque)
            mdd, _, _ = RiskMetrics.maximum_drawdown(r)
            stats_rows.append({
                "Classe d'actifs": name,
                "Rendement ann. (%)": ann_ret * 100,
                "Volatilite ann. (%)": ann_vol * 100,
                "Sharpe": sharpe,
                "Perte max. (%)": mdd * 100,
            })
        stats_df = pd.DataFrame(stats_rows)
        st.dataframe(
            stats_df.style.format({
                "Rendement ann. (%)": "{:.2f}",
                "Volatilite ann. (%)": "{:.2f}",
                "Sharpe": "{:.3f}",
                "Perte max. (%)": "{:.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )


render()

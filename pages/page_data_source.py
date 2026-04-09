"""
Source de donnees: hypotheses et donnees simulees pour l'optimisation.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    get_asset_names_fr, get_expected_returns, get_volatilities,
    get_covariance_matrix, DEFAULT_CORRELATION_MATRIX,
    DEFAULT_CURRENT_WEIGHTS, CHART_COLORS,
)


def render():
    st.title("Source de donnees")

    asset_names = get_asset_names_fr()
    n_assets = len(asset_names)
    pension_config = st.session_state.get("pension_config", None)
    horizon = pension_config.horizon_annees if pension_config else 20
    rfr = pension_config.taux_sans_risque if pension_config else 0.025

    # ===================== Parametres =====================
    st.markdown("### Parametres generaux")
    col1, col2, col3 = st.columns(3)
    col1.metric("Horizon de placement", f"{horizon} ans")
    col2.metric("Taux sans risque", f"{rfr:.1%}")
    col3.metric("Classes d'actifs", f"{n_assets}")

    st.divider()

    # ===================== Rendements et risques =====================
    st.markdown("### Rendements et risques attendus")

    exp_ret = get_expected_returns()
    vols = get_volatilities()
    sharpe = (exp_ret - rfr) / vols

    hyp_df = pd.DataFrame({
        "Classe d'actifs": asset_names,
        "Rendement (%)": exp_ret * 100,
        "Volatilite (%)": vols * 100,
        "Ratio de Sharpe": sharpe,
    })

    st.dataframe(
        hyp_df.style.format({
            "Rendement (%)": "{:.1f}",
            "Volatilite (%)": "{:.1f}",
            "Ratio de Sharpe": "{:.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # Graphique rendement vs risque
    fig_rr = go.Figure()
    fig_rr.add_trace(go.Scatter(
        x=vols * 100, y=exp_ret * 100,
        mode="markers+text",
        text=[n.split()[0] for n in asset_names],
        textposition="top center",
        marker=dict(size=12, color=CHART_COLORS[:n_assets]),
        textfont=dict(size=10),
    ))
    fig_rr.update_layout(
        title="Rendement vs Risque",
        xaxis_title="Volatilite (%)",
        yaxis_title="Rendement attendu (%)",
        height=450,
    )
    st.plotly_chart(fig_rr, use_container_width=True)

    st.divider()

    # ===================== Matrice de correlation =====================
    st.markdown("### Matrice de correlation")

    short_names = [n.replace("Obligations ", "Oblig. ")
                    .replace("gouvernementales CDN", "Gov")
                    .replace("corporatives", "Corp")
                    .replace("indexees inflation", "Infl")
                    .replace("canadiennes", "CDN")
                    .replace("americaines", "US")
                    .replace("emergentes", "EM")
                    .replace("Capital investissement", "PE")
                    .replace("Matieres premieres", "Comm")
                    .replace("Rendement absolu", "RA")
                   for n in asset_names]

    corr = DEFAULT_CORRELATION_MATRIX

    fig_corr = go.Figure(data=go.Heatmap(
        z=corr,
        x=short_names,
        y=short_names,
        colorscale="RdBu_r",
        zmid=0,
        zmin=-1, zmax=1,
        text=np.round(corr, 2),
        texttemplate="%{text}",
        textfont=dict(size=9),
    ))
    fig_corr.update_layout(
        title="Correlations entre classes d'actifs",
        height=550,
        xaxis=dict(tickangle=45),
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.divider()

    # ===================== Donnees simulees =====================
    st.markdown("### Donnees simulees")
    st.caption("Generez un historique de rendements correles avec queues epaisses et changement de regime.")

    col1, col2 = st.columns(2)
    with col1:
        seed = st.number_input("Graine aleatoire", 1, 99999, 42)
    with col2:
        n_years = st.slider("Historique (annees)", 5, 30, 20)

    if st.button("Regenerer les donnees", type="primary", use_container_width=True):
        from data.generator import MarketDataGenerator
        generator = MarketDataGenerator(seed=seed)
        st.session_state.returns_data = generator.generate_returns(
            n_years=n_years, frequency="monthly",
        )
        st.session_state.current_weights = DEFAULT_CURRENT_WEIGHTS.copy()
        st.success("Donnees regenerees!")

    if "returns_data" in st.session_state:
        df = st.session_state.returns_data
        st.markdown(f"**Donnees en memoire:** {len(df)} observations mensuelles "
                    f"({len(df) // 12} ans)")

        st.markdown("#### Apercu des rendements mensuels")
        preview = df.copy()
        preview.columns = asset_names[:df.shape[1]]
        st.dataframe(
            preview.tail(12).style.format("{:.2%}"),
            use_container_width=True,
        )
    else:
        st.info("Aucune donnee generee. Cliquez sur 'Regenerer les donnees'.")


render()

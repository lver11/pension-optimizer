"""
Recommandations de reequilibrage et analyse des couts de transaction.
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    get_asset_names_fr, get_expected_returns, get_covariance_matrix,
    DEFAULT_CURRENT_WEIGHTS, PensionFundConfig, ASSET_DEFAULTS, ASSET_CLASSES_ORDER,
)
from data.generator import MarketDataGenerator
from visualization.charts import ChartBuilder


# Couts de transaction par classe d'actifs (en bps)
TRANSACTION_COSTS_BPS = {
    "Actions canadiennes": 15,
    "Actions americaines": 10,
    "Actions EAFE": 20,
    "Actions emergentes": 30,
    "Obligations gouvernementales CDN": 5,
    "Obligations corporatives": 10,
    "Obligations indexees inflation": 8,
    "Immobilier": 150,
    "Infrastructure": 200,
    "Capital investissement": 200,
    "Matieres premieres": 20,
    "Encaisse": 1,
}


def render():
    st.title("Recommandations de reequilibrage")

    if "returns_data" not in st.session_state or st.session_state.returns_data is None:
        generator = MarketDataGenerator(seed=42)
        st.session_state.returns_data = generator.generate_returns(n_years=20, frequency="monthly")
        st.session_state.current_weights = DEFAULT_CURRENT_WEIGHTS.copy()
        st.session_state.pension_config = PensionFundConfig()

    config = st.session_state.get("pension_config", PensionFundConfig())
    current_weights = st.session_state.get("current_weights", DEFAULT_CURRENT_WEIGHTS)
    asset_names = get_asset_names_fr()
    portfolio_value = config.valeur_actif

    # Allocation cible
    if "optimization_result" in st.session_state and st.session_state.optimization_result is not None:
        target_weights = st.session_state.optimization_result.weights
        source = "Allocation optimisee"
    else:
        target_weights = DEFAULT_CURRENT_WEIGHTS.copy()
        source = "Allocation politique par defaut"

    st.info(f"Allocation cible: **{source}**")

    # ---------- Ecarts d'allocation ----------
    st.markdown("### Ecarts d'allocation")

    deviations = target_weights - current_weights
    trades = deviations * portfolio_value

    rebal_df = pd.DataFrame({
        "Classe d'actifs": asset_names,
        "Actuel (%)": current_weights * 100,
        "Cible (%)": target_weights * 100,
        "Ecart (pp)": deviations * 100,
        "Transaction (M$)": trades / 1e6,
        "Direction": ["Acheter" if d > 0.001 else "Vendre" if d < -0.001 else "Maintenir"
                      for d in deviations],
    })

    st.dataframe(rebal_df.style.format({
        "Actuel (%)": "{:.1f}",
        "Cible (%)": "{:.1f}",
        "Ecart (pp)": "{:+.2f}",
        "Transaction (M$)": "{:+,.1f}",
    }).applymap(
        lambda v: "color: green" if "Acheter" in str(v)
        else ("color: red" if "Vendre" in str(v) else ""),
        subset=["Direction"],
    ), use_container_width=True, hide_index=True)

    # Graphique des ecarts
    import plotly.graph_objects as go
    fig_dev = go.Figure(go.Bar(
        x=asset_names,
        y=deviations * 100,
        marker_color=["green" if d > 0 else "red" for d in deviations],
        text=[f"{d:+.2f} pp" for d in deviations * 100],
        textposition="outside",
    ))
    fig_dev.update_layout(
        title="Ecarts par rapport a l'allocation cible",
        yaxis_title="Ecart (points de pourcentage)",
        xaxis_tickangle=-45,
        height=400,
    )
    fig_dev.add_hline(y=0, line_color="black", line_width=1)
    st.plotly_chart(fig_dev, use_container_width=True)

    st.divider()

    # ---------- Couts de transaction ----------
    st.markdown("### Analyse des couts de transaction")

    costs_bps = np.array([TRANSACTION_COSTS_BPS.get(name, 10) for name in asset_names])
    abs_trades = np.abs(trades)
    costs = abs_trades * costs_bps / 10000

    cost_df = pd.DataFrame({
        "Classe d'actifs": asset_names,
        "Transaction absolue (M$)": abs_trades / 1e6,
        "Cout unitaire (bps)": costs_bps,
        "Cout estime ($)": costs,
        "Cout estime (M$)": costs / 1e6,
    })

    st.dataframe(cost_df.style.format({
        "Transaction absolue (M$)": "{:.1f}",
        "Cout unitaire (bps)": "{:.0f}",
        "Cout estime ($)": "{:,.0f}",
        "Cout estime (M$)": "{:.3f}",
    }), use_container_width=True, hide_index=True)

    total_cost = costs.sum()
    total_turnover = np.sum(np.abs(deviations)) / 2
    col1, col2, col3 = st.columns(3)
    col1.metric("Cout total estime", f"{total_cost:,.0f} $")
    col2.metric("Cout en bps du portefeuille", f"{total_cost / portfolio_value * 10000:.1f} bps")
    col3.metric("Rotation (one-way)", f"{total_turnover:.1%}")

    st.divider()

    # ---------- Strategie de reequilibrage ----------
    st.markdown("### Strategie de reequilibrage")

    strategy = st.radio(
        "Type de strategie",
        ["Calendaire", "A seuil", "Hybride"],
        horizontal=True,
    )

    if strategy == "Calendaire":
        freq = st.selectbox("Frequence", ["Mensuel", "Trimestriel", "Semestriel", "Annuel"])
        st.info(f"""
        **Reequilibrage {freq.lower()}:** Le portefeuille est ramene a l'allocation cible
        a chaque {freq.lower().replace('iel', 'ois').replace('triel', 'tre')},
        independamment de l'ampleur des ecarts.
        """)

    elif strategy == "A seuil":
        threshold = st.slider("Seuil de deviation (%)", 1.0, 10.0, 5.0, 0.5)
        exceeded = np.any(np.abs(deviations) * 100 > threshold)
        if exceeded:
            violating = [asset_names[i] for i in range(len(deviations))
                         if abs(deviations[i]) * 100 > threshold]
            st.warning(f"Seuil depasse pour: {', '.join(violating)}")
            st.info("**Recommandation:** Reequilibrer maintenant.")
        else:
            st.success(f"Aucun ecart ne depasse le seuil de {threshold}%. Pas de reequilibrage requis.")

    else:  # Hybride
        threshold = st.slider("Seuil de deviation (%)", 1.0, 10.0, 5.0, 0.5)
        freq = st.selectbox("Frequence de verification", ["Mensuel", "Trimestriel"])
        st.info(f"""
        **Strategie hybride:** Verification {freq.lower()} + reequilibrage
        si un ecart depasse {threshold}%.
        """)

    st.divider()

    # ---------- Simulation des strategies ----------
    st.markdown("### Analyse comparative des strategies")

    if st.button("Analyser les strategies de reequilibrage"):
        returns_data = st.session_state.returns_data
        mu = get_expected_returns()
        cov = get_covariance_matrix()

        strategies_results = _simulate_rebalancing_strategies(
            returns_data.values, current_weights, target_weights,
            costs_bps, portfolio_value,
        )

        strat_df = pd.DataFrame(strategies_results)
        st.dataframe(strat_df.style.format({
            "Rendement annualise (%)": "{:.2f}",
            "Volatilite (%)": "{:.2f}",
            "Sharpe": "{:.3f}",
            "Cout cumule (bps)": "{:.1f}",
            "Rotation annuelle (%)": "{:.1f}",
        }), use_container_width=True, hide_index=True)


def _simulate_rebalancing_strategies(
    returns: np.ndarray,
    current_weights: np.ndarray,
    target_weights: np.ndarray,
    costs_bps: np.ndarray,
    portfolio_value: float,
) -> list:
    """Simule differentes strategies de reequilibrage."""
    T = returns.shape[0]
    results = []

    for freq_name, freq_months in [
        ("Mensuel", 1), ("Trimestriel", 3),
        ("Semestriel", 6), ("Annuel", 12),
        ("Jamais", T+1),
    ]:
        weights = target_weights.copy()
        port_returns = []
        total_turnover = 0

        for t in range(T):
            # Rendement du portefeuille
            r = returns[t] @ weights
            port_returns.append(r)

            # Mise a jour des poids apres rendement
            new_values = weights * (1 + returns[t])
            weights = new_values / new_values.sum()

            # Reequilibrage
            if (t + 1) % freq_months == 0 and t < T - 1:
                turnover = np.sum(np.abs(weights - target_weights)) / 2
                total_turnover += turnover
                weights = target_weights.copy()

        port_returns = np.array(port_returns)
        ann_return = np.mean(port_returns) * 12
        ann_vol = np.std(port_returns) * np.sqrt(12)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0

        n_rebalances = T // freq_months if freq_months <= T else 0
        cost_per_rebal = np.sum(np.abs(target_weights - current_weights) * costs_bps) / 10000
        total_cost_bps = total_turnover * np.mean(costs_bps)

        results.append({
            "Strategie": freq_name,
            "Rendement annualise (%)": ann_return * 100,
            "Volatilite (%)": ann_vol * 100,
            "Sharpe": sharpe,
            "Nombre reequilibrages": n_rebalances,
            "Cout cumule (bps)": total_cost_bps,
            "Rotation annuelle (%)": total_turnover / max(T/12, 1) * 100,
        })

    return results


render()

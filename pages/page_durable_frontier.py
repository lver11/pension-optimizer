# pages/page_durable_frontier.py
"""
Frontière Pareto interactive : Ratio de Sharpe ↔ Score de durabilité.
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sustainable.config import (
    DURABLE_ASSETS, DURABLE_ASSET_ORDER, DURABLE_MIN_WEIGHTS, DURABLE_MAX_WEIGHTS,
    DEFAULT_DIM_WEIGHTS, get_durable_returns, get_durable_covariance_matrix,
    get_score_matrix,
)
from sustainable.optimizer import DurableOptimizer

DIMS = ["durabilite", "additionnalite", "disponibilite", "retombees_qc", "liquidite"]
DIM_LABELS = ["Durabilité", "Additionnalité", "Disponibilité", "Retombées Qc", "Liquidité"]


def _get_active_config():
    universe = st.session_state.get("durable_universe", {
        aid: {"active": True, "use_durable": False} for aid in DURABLE_ASSET_ORDER
    })
    active_ids = [aid for aid in DURABLE_ASSET_ORDER
                  if universe.get(aid, {}).get("active", True)]
    use_durable_map = {aid: universe.get(aid, {}).get("use_durable", False)
                       for aid in active_ids}
    return active_ids, use_durable_map


def _build_optimizer(active_ids, use_durable_map, rf):
    active_idx = [DURABLE_ASSET_ORDER.index(aid) for aid in active_ids]
    mu_full = get_durable_returns(use_durable_map)
    cov_full = get_durable_covariance_matrix(use_durable_map)
    mu_active = mu_full[active_idx]
    cov_active = cov_full[np.ix_(active_idx, active_idx)]
    min_w = np.array([DURABLE_MIN_WEIGHTS[i] for i in active_idx])
    max_w = np.array([DURABLE_MAX_WEIGHTS[i] for i in active_idx])
    # Ensure min_w can sum to ≤ 1
    if min_w.sum() > 1.0:
        min_w = min_w / min_w.sum() * 0.99
    names = [
        DURABLE_ASSETS[aid].nom_durable
        if use_durable_map.get(aid) and DURABLE_ASSETS[aid].has_durable_variant
        else DURABLE_ASSETS[aid].nom
        for aid in active_ids
    ]
    opt = DurableOptimizer(
        expected_returns=mu_active, cov_matrix=cov_active,
        risk_free_rate=rf, asset_names=names,
        min_weights=min_w, max_weights=max_w,
    )
    return opt, active_idx


def _get_custom_scores(use_durable_map):
    custom_scores = {}
    if "durable_scores" in st.session_state:
        for _, row in st.session_state.durable_scores.iterrows():
            custom_scores[row["id"]] = {dim: row[lbl] for dim, lbl in zip(DIMS, DIM_LABELS)}
    return custom_scores


def render():
    st.title("📈 Frontière durable")
    st.caption("Frontière Pareto entre performance financière (Sharpe) et durabilité. "
               "Chaque point est un portefeuille optimal pour un λ différent.")

    active_ids, use_durable_map = _get_active_config()
    if len(active_ids) < 2:
        st.error("Activez au moins 2 classes d'actifs dans 'Univers durable'.")
        return

    dim_weights = st.session_state.get("durable_dim_weights", DEFAULT_DIM_WEIGHTS)
    gamma = st.session_state.get("durable_gamma", 2.5)
    rf = getattr(st.session_state.get("pension_config"), "taux_sans_risque", 0.03)

    col1, col2, col3 = st.columns(3)
    with col1:
        n_points = st.slider("Nombre de points", 20, 100, 50, 10)
    with col2:
        lambda_max = st.slider("λ maximum", 1.0, 10.0, 5.0, 0.5)
    with col3:
        st.metric("γ (aversion au risque)", f"{gamma:.1f}")

    if st.button("🚀 Calculer la frontière Pareto", type="primary", use_container_width=True):
        with st.spinner("Calcul en cours..."):
            custom_scores = _get_custom_scores(use_durable_map)
            S_cur, _ = get_score_matrix(dim_weights, use_durable_map, custom_scores)
            active_idx = [DURABLE_ASSET_ORDER.index(aid) for aid in active_ids]
            S_active = S_cur[active_idx]

            opt, _ = _build_optimizer(active_ids, use_durable_map, rf)
            frontier = opt.pareto_frontier(
                n_points=n_points, gamma=gamma,
                sustainability_scores=S_active, lambda_max=lambda_max,
            )
            if len(frontier) < 3:
                st.error("Moins de 3 points faisables. Vérifiez les contraintes dans 'Univers durable'.")
                return
            st.session_state.durable_frontier = frontier
            st.session_state.durable_frontier_active_ids = active_ids
            st.success(f"{len(frontier)} points calculés sur la frontière.")

    if "durable_frontier" not in st.session_state:
        st.info("Cliquez sur 'Calculer la frontière Pareto' pour afficher les résultats.")
        return

    frontier = st.session_state.durable_frontier
    scores = [r.sustainability_score for r in frontier]
    sharpes = [r.sharpe_ratio for r in frontier]
    returns = [r.expected_return for r in frontier]
    vols = [r.volatility for r in frontier]
    lambdas = [r.lambda_used for r in frontier]

    # Main scatter plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=scores, y=sharpes,
        mode="markers+lines",
        marker=dict(size=8, color=lambdas, colorscale="Viridis",
                    showscale=True, colorbar=dict(title="λ")),
        line=dict(color="rgba(100,100,200,0.3)", width=1),
        text=[f"λ={l:.2f}<br>Rend: {r:.2%}<br>Vol: {v:.2%}<br>Score: {s:.2f}"
              for l, r, v, s in zip(lambdas, returns, vols, scores)],
        hoverinfo="text", name="Frontière Pareto",
    ))

    fig.update_layout(
        title="Frontière Pareto : Score durabilité ↔ Ratio de Sharpe",
        xaxis_title="Score de durabilité du portefeuille",
        yaxis_title="Ratio de Sharpe",
        height=480, margin=dict(t=60, b=60),
        hovermode="closest",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Point selector
    st.markdown("### Sélectionner un point")
    idx_sel = st.slider("Index du point (score croissant)", 0, len(frontier)-1,
                        len(frontier)//2, key="frontier_idx")
    selected = frontier[idx_sel]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rendement", f"{selected.expected_return:.2%}")
    col2.metric("Volatilité", f"{selected.volatility:.2%}")
    col3.metric("Sharpe", f"{selected.sharpe_ratio:.3f}")
    col4.metric("Score durabilité", f"{selected.sustainability_score:.2f}")

    if st.button("➡ Utiliser ce portefeuille dans Optimisation durable"):
        st.session_state.durable_result = selected
        st.session_state.durable_lambda = selected.lambda_used
        # Also write active_ids so rapport page can align weights correctly
        st.session_state.durable_active_ids = st.session_state.get(
            "durable_frontier_active_ids", active_ids
        )
        # Clear fin result so optimization page knows it needs to re-run
        st.session_state.pop("durable_result_fin", None)
        st.success(f"Portefeuille sélectionné (λ={selected.lambda_used:.2f}). Ouvrez 'Optimisation durable'.")


render()

# pages/page_durable_optimization.py
"""
Optimisation durable : lancer une optimisation, comparer les allocations.
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
    get_score_matrix, N_DURABLE_ASSETS,
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


def render():
    st.title("⚙️ Optimisation durable")
    st.caption("Comparez l'allocation financièrement optimale (λ=0) avec l'allocation durable optimale.")

    active_ids, use_durable_map = _get_active_config()
    if len(active_ids) < 2:
        st.error("Activez au moins 2 classes d'actifs dans 'Univers durable'.")
        return

    dim_weights = st.session_state.get("durable_dim_weights", DEFAULT_DIM_WEIGHTS)
    gamma = st.session_state.get("durable_gamma", 2.5)
    rf = getattr(st.session_state.get("pension_config"), "taux_sans_risque", 0.03)

    lam_default = float(st.session_state.get("durable_lambda", 1.0))
    lam = st.slider("Poids durabilité (λ)", 0.0, 10.0, lam_default, 0.1,
                    key="opt_lambda_slider")
    st.session_state.durable_lambda = lam

    if st.button("🚀 Lancer l'optimisation durable", type="primary", use_container_width=True):
        with st.spinner("Optimisation en cours..."):
            try:
                custom_scores = {}
                if "durable_scores" in st.session_state:
                    for _, row in st.session_state.durable_scores.iterrows():
                        custom_scores[row["id"]] = {dim: row[lbl]
                                                    for dim, lbl in zip(DIMS, DIM_LABELS)}

                S_cur, _ = get_score_matrix(dim_weights, use_durable_map, custom_scores)
                active_idx = [DURABLE_ASSET_ORDER.index(aid) for aid in active_ids]
                S_active = S_cur[active_idx]

                opt, _ = _build_optimizer(active_ids, use_durable_map, rf)

                result = opt.optimize_durable(lam=lam, gamma=gamma,
                                             sustainability_scores=S_active)
                result_fin = opt.optimize_durable(lam=0.0, gamma=gamma,
                                                  sustainability_scores=S_active)

                st.session_state.durable_result = result
                st.session_state.durable_result_fin = result_fin
                st.session_state.durable_active_ids = active_ids
                st.session_state.durable_active_idx = active_idx

                if result.status in ("optimal", "optimal_inaccurate"):
                    st.success(
                        f"Optimisation réussie ({result.solver_time:.2f}s) | "
                        f"Rendement: {result.expected_return:.2%} | "
                        f"Volatilité: {result.volatility:.2%} | "
                        f"Sharpe: {result.sharpe_ratio:.3f} | "
                        f"Score durabilité: {result.sustainability_score:.2f}"
                    )
                else:
                    st.warning(f"Statut: {result.status}.")
            except Exception as e:
                st.error(f"Erreur: {e}")
                return

    if "durable_result" not in st.session_state:
        st.info("Configurez les paramètres et cliquez sur 'Lancer l'optimisation'.")
        return

    if "durable_result_fin" not in st.session_state:
        st.info("Résultat provenant de la frontière. Cliquez sur 'Lancer l'optimisation' pour obtenir la comparaison complète (financier vs durable).")
        # Still show the durable result metrics if available
        result = st.session_state.durable_result
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rendement", f"{result.expected_return:.2%}")
        col2.metric("Volatilité", f"{result.volatility:.2%}")
        col3.metric("Sharpe", f"{result.sharpe_ratio:.3f}")
        col4.metric("Score durabilité", f"{result.sustainability_score:.2f}")
        return

    result = st.session_state.durable_result
    result_fin = st.session_state.durable_result_fin
    stored_ids = st.session_state.get("durable_active_ids", active_ids)
    names = [
        DURABLE_ASSETS[aid].nom_durable
        if use_durable_map.get(aid) and DURABLE_ASSETS[aid].has_durable_variant
        else DURABLE_ASSETS[aid].nom
        for aid in stored_ids
    ]

    st.divider()
    st.markdown("## Résultats comparatifs")

    # 3-column comparison table
    col_h1, col_h2, col_h3 = st.columns(3)
    col_h1.markdown("**Métrique**")
    col_h2.markdown("**Optimal financier (λ=0)**")
    col_h3.markdown(f"**Optimal durable (λ={lam:.1f})**")

    rows = [
        ("Rendement attendu", f"{result_fin.expected_return:.2%}", f"{result.expected_return:.2%}"),
        ("Volatilité", f"{result_fin.volatility:.2%}", f"{result.volatility:.2%}"),
        ("Ratio de Sharpe", f"{result_fin.sharpe_ratio:.3f}", f"{result.sharpe_ratio:.3f}"),
        ("Score durabilité", f"{result_fin.sustainability_score:.2f}", f"{result.sustainability_score:.2f}"),
    ]
    for label, v_fin, v_dur in rows:
        c1, c2, c3 = st.columns(3)
        c1.write(label)
        c2.write(v_fin)
        c3.write(v_dur)

    st.divider()

    tab1, tab2, tab3 = st.tabs(["Allocations", "Score par dimension", "Tableau détaillé"])

    with tab1:
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Optimal financier", x=names,
                             y=result_fin.weights * 100, marker_color="#1f77b4"))
        fig.add_trace(go.Bar(name="Optimal durable", x=names,
                             y=result.weights * 100, marker_color="#2ca02c"))
        fig.update_layout(barmode="group", xaxis_tickangle=-45,
                          yaxis_title="Poids (%)", height=400, margin=dict(t=20, b=100))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        dim_scores_dur = {}
        dim_scores_fin = {}
        custom_scores = {}
        if "durable_scores" in st.session_state:
            for _, row in st.session_state.durable_scores.iterrows():
                custom_scores[row["id"]] = {dim: row[lbl]
                                            for dim, lbl in zip(DIMS, DIM_LABELS)}
        active_idx_stored = st.session_state.get("durable_active_idx",
                                                  [DURABLE_ASSET_ORDER.index(aid) for aid in stored_ids])
        for dim, lbl in zip(DIMS, DIM_LABELS):
            dim_w = {d: (1.0 if d == dim else 0.0) for d in DIMS}
            S_dim, _ = get_score_matrix(dim_w, use_durable_map, custom_scores)
            S_dim_active = S_dim[active_idx_stored]
            dim_scores_dur[lbl] = float(result.weights @ S_dim_active)
            dim_scores_fin[lbl] = float(result_fin.weights @ S_dim_active)

        fig_dim = go.Figure()
        fig_dim.add_trace(go.Bar(name="Financier", x=list(dim_scores_fin.values()),
                                 y=list(dim_scores_fin.keys()), orientation="h",
                                 marker_color="#1f77b4"))
        fig_dim.add_trace(go.Bar(name="Durable", x=list(dim_scores_dur.values()),
                                 y=list(dim_scores_dur.keys()), orientation="h",
                                 marker_color="#2ca02c"))
        fig_dim.update_layout(barmode="group", xaxis=dict(range=[0, 5], title="Score (1–5)"),
                               height=350, margin=dict(t=20, b=40))
        st.plotly_chart(fig_dim, use_container_width=True)

    with tab3:
        detail_df = pd.DataFrame({
            "Classe d'actifs": names,
            "Optimal financier (%)": result_fin.weights * 100,
            "Optimal durable (%)": result.weights * 100,
            "Écart (pp)": (result.weights - result_fin.weights) * 100,
        })
        st.dataframe(
            detail_df.style.format({
                "Optimal financier (%)": "{:.1f}",
                "Optimal durable (%)": "{:.1f}",
                "Écart (pp)": "{:+.1f}",
            }),
            use_container_width=True, hide_index=True,
        )

    st.divider()
    if st.button("📥 Enregistrer le portefeuille durable"):
        full_weights = np.zeros(N_DURABLE_ASSETS)
        for i, aid in enumerate(stored_ids):
            full_weights[DURABLE_ASSET_ORDER.index(aid)] = result.weights[i]
        st.session_state.durable_adopted_weights = full_weights
        st.success("Poids enregistrés dans 'durable_adopted_weights'.")


render()

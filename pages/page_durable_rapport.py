# pages/page_durable_rapport.py
"""
Rapport de synthèse de l'optimisation durable. Export CSV.
"""
import streamlit as st
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sustainable.config import (
    DURABLE_ASSETS, DURABLE_ASSET_ORDER, DEFAULT_DIM_WEIGHTS,
    get_score_matrix,
)

DIMS = ["durabilite", "additionnalite", "disponibilite", "retombees_qc", "liquidite"]
DIM_LABELS = ["Durabilité", "Additionnalité", "Disponibilité", "Retombées Qc", "Liquidité"]


def render():
    st.title("📄 Rapport durable")
    st.caption("Résumé de l'optimisation durable pour présentation au conseil.")

    result = st.session_state.get("durable_result")
    if result is None:
        st.info("Lancez une optimisation dans 'Optimisation durable' d'abord.")
        return

    active_ids = st.session_state.get("durable_active_ids")
    if active_ids is None:
        # Infer active_ids from result.weights size to avoid misalignment
        n = len(result.weights)
        active_ids = DURABLE_ASSET_ORDER[:n]  # best-effort fallback
        st.warning("Configuration d'univers non trouvée — les labels peuvent être approximatifs.")
    dim_weights = st.session_state.get("durable_dim_weights", DEFAULT_DIM_WEIGHTS)
    gamma = st.session_state.get("durable_gamma", 2.5)
    lam = st.session_state.get("durable_lambda", 1.0)
    universe = st.session_state.get("durable_universe", {})
    use_durable_map = {aid: universe.get(aid, {}).get("use_durable", False)
                       for aid in active_ids}
    names = [
        DURABLE_ASSETS[aid].nom_durable
        if use_durable_map.get(aid) and DURABLE_ASSETS[aid].has_durable_variant
        else DURABLE_ASSETS[aid].nom
        for aid in active_ids
    ]

    # Section 1: Financial KPIs
    st.markdown("## 1. Métriques financières")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rendement attendu", f"{result.expected_return:.2%}")
    col2.metric("Volatilité", f"{result.volatility:.2%}")
    col3.metric("Ratio de Sharpe", f"{result.sharpe_ratio:.3f}")
    col4.metric("Score durabilité", f"{result.sustainability_score:.2f} / 5")

    # Section 2: Sustainability breakdown
    st.markdown("## 2. Score de durabilité par dimension")
    custom_scores = {}
    if "durable_scores" in st.session_state:
        for _, row in st.session_state.durable_scores.iterrows():
            custom_scores[row["id"]] = {dim: row[lbl] for dim, lbl in zip(DIMS, DIM_LABELS)}

    active_idx = [DURABLE_ASSET_ORDER.index(aid) for aid in active_ids]
    dim_score_rows = []
    for dim, lbl in zip(DIMS, DIM_LABELS):
        dim_w = {d: (1.0 if d == dim else 0.0) for d in DIMS}
        S_dim, _ = get_score_matrix(dim_w, use_durable_map, custom_scores)
        score_val = float(result.weights @ S_dim[active_idx])
        dim_score_rows.append({
            "Dimension": lbl,
            "Pondération (%)": dim_weights.get(dim, 0) * 100,
            "Score du portefeuille": score_val,
        })
    dim_df = pd.DataFrame(dim_score_rows)
    st.dataframe(
        dim_df.style.format({"Pondération (%)": "{:.0f}", "Score du portefeuille": "{:.2f}"}),
        use_container_width=True, hide_index=True,
    )

    # Section 3: Allocation
    st.markdown("## 3. Allocation du portefeuille")
    alloc_df = pd.DataFrame({
        "Classe d'actifs": names,
        "Poids (%)": result.weights * 100,
        "Variante durable": [
            "✅" if use_durable_map.get(aid) and DURABLE_ASSETS[aid].has_durable_variant else "—"
            for aid in active_ids
        ],
        "Rendement attendu (%)": [
            DURABLE_ASSETS[aid].rendement_durable * 100
            if use_durable_map.get(aid) and DURABLE_ASSETS[aid].has_durable_variant
            else DURABLE_ASSETS[aid].rendement * 100
            for aid in active_ids
        ],
    }).sort_values("Poids (%)", ascending=False)
    st.dataframe(
        alloc_df.style.format({"Poids (%)": "{:.1f}", "Rendement attendu (%)": "{:.2f}"}),
        use_container_width=True, hide_index=True,
    )

    # Section 4: Assumptions
    st.markdown("## 4. Hypothèses utilisées")
    col1, col2, col3 = st.columns(3)
    col1.metric("Aversion au risque (γ)", f"{gamma:.1f}")
    col2.metric("Poids durabilité (λ)", f"{lam:.1f}")
    col3.metric("Classes d'actifs actives", len(active_ids))

    with st.expander("Pondérations des dimensions"):
        for dim, lbl in zip(DIMS, DIM_LABELS):
            st.write(f"- **{lbl}**: {dim_weights.get(dim, 0)*100:.0f}%")

    durable_used = [(aid, DURABLE_ASSETS[aid].nom_durable) for aid in active_ids
                    if use_durable_map.get(aid) and DURABLE_ASSETS[aid].has_durable_variant]
    if durable_used:
        with st.expander("Variantes durables activées"):
            for aid, durable_name in durable_used:
                st.write(f"- {DURABLE_ASSETS[aid].nom} → **{durable_name}**")

    # Section 5: Export
    st.divider()
    st.markdown("## 5. Export")
    export_df = pd.DataFrame({
        "Classe d'actifs": names,
        "Poids (%)": (result.weights * 100).round(2),
        "Variante durable": [
            "Oui" if use_durable_map.get(aid) and DURABLE_ASSETS[aid].has_durable_variant else "Non"
            for aid in active_ids
        ],
        "Rendement attendu (%)": [
            round(DURABLE_ASSETS[aid].rendement_durable * 100, 2)
            if use_durable_map.get(aid) and DURABLE_ASSETS[aid].has_durable_variant
            else round(DURABLE_ASSETS[aid].rendement * 100, 2)
            for aid in active_ids
        ],
    })
    export_df = export_df.sort_values("Poids (%)", ascending=False)
    csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Télécharger l'allocation (CSV)",
        data=csv,
        file_name="portefeuille_durable.csv",
        mime="text/csv",
        use_container_width=True,
    )


render()

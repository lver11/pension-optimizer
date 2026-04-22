# pages/page_durable_scores.py
"""
Tableau éditable des scores de durabilité par classe d'actifs.
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sustainable.config import (
    DURABLE_ASSETS, DURABLE_ASSET_ORDER, DEFAULT_DIM_WEIGHTS, get_score_matrix,
)

DIMS = ["durabilite", "additionnalite", "disponibilite", "retombees_qc", "liquidite"]
DIM_LABELS = {
    "durabilite": "Durabilité",
    "additionnalite": "Additionnalité",
    "disponibilite": "Disponibilité",
    "retombees_qc": "Retombées Qc",
    "liquidite": "Liquidité",
}


def _default_scores_df(use_durable_map: dict) -> pd.DataFrame:
    rows = []
    for aid in DURABLE_ASSET_ORDER:
        asset = DURABLE_ASSETS[aid]
        use_dur = use_durable_map.get(aid, False) and asset.has_durable_variant
        row = {
            "id": aid,
            "Classe d'actifs": asset.nom_durable if use_dur else asset.nom,
        }
        for dim in DIMS:
            key = f"score_{dim}_durable" if use_dur else f"score_{dim}"
            row[DIM_LABELS[dim]] = float(getattr(asset, key))
        rows.append(row)
    return pd.DataFrame(rows)


def render():
    st.title("📊 Scores de durabilité")
    st.caption("Modifiez les scores par classe d'actifs (1 = faible, 5 = élevé). "
               "Valeurs par défaut issues du fichier Excel de cartographie.")

    universe = st.session_state.get("durable_universe", {})
    use_durable_map = {aid: v.get("use_durable", False) for aid, v in universe.items()}
    dim_weights = st.session_state.get("durable_dim_weights", DEFAULT_DIM_WEIGHTS)

    if "durable_scores" not in st.session_state:
        st.session_state.durable_scores = _default_scores_df(use_durable_map)

    # Re-initialize if durable variant selection has changed
    universe_hash = str({aid: v.get("use_durable", False) for aid, v in universe.items()})
    if st.session_state.get("_durable_scores_universe_hash") != universe_hash:
        st.session_state.durable_scores = _default_scores_df(use_durable_map)
        st.session_state["_durable_scores_universe_hash"] = universe_hash

    scores_df = st.session_state.durable_scores.copy()
    dim_cols = list(DIM_LABELS.values())

    st.markdown("### Scores par classe d'actifs *(éditables)*")
    edited = st.data_editor(
        scores_df[["id", "Classe d'actifs"] + dim_cols],
        use_container_width=True,
        hide_index=True,
        disabled=["id", "Classe d'actifs"],
        column_config={
            "id": st.column_config.TextColumn("ID", width="small"),
            "Classe d'actifs": st.column_config.TextColumn("Classe d'actifs"),
            **{
                col: st.column_config.NumberColumn(col, min_value=1.0, max_value=5.0,
                                                   format="%.1f", step=0.25)
                for col in dim_cols
            },
        },
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Réinitialiser aux valeurs Excel", use_container_width=True):
            st.session_state.durable_scores = _default_scores_df(use_durable_map)
            st.session_state.pop("durable_frontier", None)
            st.session_state.pop("durable_result", None)
            st.rerun()
    with col2:
        if st.button("✅ Appliquer les scores", type="primary", use_container_width=True):
            valid = True
            for col in dim_cols:
                if edited[col].isnull().any() or (edited[col] < 1).any() or (edited[col] > 5).any():
                    st.error(f"Scores invalides dans '{col}': doit être entre 1 et 5.")
                    valid = False
            if valid:
                id_to_scores = edited.set_index("id")[dim_cols]
                updated = scores_df.copy()
                updated = updated.set_index("id")
                updated[dim_cols] = id_to_scores
                updated = updated.reset_index()
                st.session_state.durable_scores = updated
                st.session_state.pop("durable_frontier", None)
                st.session_state.pop("durable_result", None)
                st.success("Scores enregistrés.")

    st.divider()

    st.markdown("### Score composite par classe d'actifs")
    st.caption("Pondérations : " + " | ".join([f"{DIM_LABELS[k]}: {v*100:.0f}%"
                                                for k, v in dim_weights.items()]))

    current_df = st.session_state.durable_scores
    custom_scores = {}
    for _, row in current_df.iterrows():
        aid = row["id"]
        custom_scores[aid] = {dim: row[DIM_LABELS[dim]] for dim in DIMS}

    S_cur, _ = get_score_matrix(dim_weights, use_durable_map, custom_scores)

    composite_df = pd.DataFrame({
        "Classe d'actifs": current_df["Classe d'actifs"].values,
        "Score composite": S_cur,
    }).sort_values("Score composite")

    colors = composite_df["Score composite"].apply(
        lambda s: "#d62728" if s < 2 else "#ff7f0e" if s < 3 else "#2ca02c"
    ).tolist()

    fig = go.Figure(go.Bar(
        x=composite_df["Score composite"],
        y=composite_df["Classe d'actifs"],
        orientation="h",
        marker_color=colors,
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 5], title="Score composite (1–5)"),
        height=450, margin=dict(t=20, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


render()

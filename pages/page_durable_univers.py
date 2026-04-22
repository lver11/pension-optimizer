# pages/page_durable_univers.py
"""
Configuration de l'univers d'actifs durable.
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sustainable.config import (
    DURABLE_ASSETS, DURABLE_ASSET_ORDER, DURABLE_ASSET_NAMES_FR,
    DURABLE_DEFAULT_WEIGHTS, DURABLE_MIN_WEIGHTS, DURABLE_MAX_WEIGHTS,
    DEFAULT_DIM_WEIGHTS, get_score_matrix, N_DURABLE_ASSETS,
)


def _init_session():
    if "durable_universe" not in st.session_state:
        st.session_state.durable_universe = {
            aid: {"active": True, "use_durable": False}
            for aid in DURABLE_ASSET_ORDER
        }
    if "durable_dim_weights" not in st.session_state:
        st.session_state.durable_dim_weights = DEFAULT_DIM_WEIGHTS.copy()
    if "durable_gamma" not in st.session_state:
        st.session_state.durable_gamma = 2.5
    if "durable_min_weights" not in st.session_state:
        st.session_state.durable_min_weights = DURABLE_MIN_WEIGHTS.copy()
    if "durable_max_weights" not in st.session_state:
        st.session_state.durable_max_weights = DURABLE_MAX_WEIGHTS.copy()


def render():
    st.title("🌍 Univers durable")
    st.caption("Configurez les classes d'actifs, les variantes durables et les priorités de durabilité.")
    _init_session()

    categories = {
        "Revenu fixe": ["obligations_ct", "obligations_univers", "hypotheques",
                        "oblig_rendement_eleve", "dettes_emergentes"],
        "Actions & liquidités": ["actions_cdn", "actions_globales", "actions_petite_cap",
                                 "eqp", "fonds_couverture"],
        "Actifs privés": ["dette_privee", "immo_prive", "infra_privee", "buyout", "capital_risque"],
    }

    universe = st.session_state.durable_universe.copy()

    for cat_name, asset_ids in categories.items():
        st.markdown(f"**{cat_name}**")
        for aid in asset_ids:
            asset = DURABLE_ASSETS[aid]
            col_check, col_name, col_toggle, col_label = st.columns([0.5, 3, 1.5, 3])
            with col_check:
                active = st.checkbox("", value=universe[aid]["active"],
                                     key=f"active_{aid}", label_visibility="collapsed")
            with col_name:
                st.write(asset.nom)
            with col_toggle:
                if asset.has_durable_variant and active:
                    use_dur = st.toggle("Durable", value=universe[aid]["use_durable"],
                                        key=f"dur_{aid}")
                else:
                    use_dur = False
                    if not asset.has_durable_variant:
                        st.caption("—")
            with col_label:
                if asset.has_durable_variant and active and use_dur:
                    st.caption(f"↪ {asset.nom_durable}")
                elif not active:
                    st.caption("*(exclu)*")
            universe[aid] = {"active": active, "use_durable": use_dur and asset.has_durable_variant}

    n_active = sum(1 for v in universe.values() if v["active"])
    n_durable = sum(1 for v in universe.values() if v["use_durable"])
    st.info(f"{n_active} classe(s) active(s) | {n_durable} variante(s) durable(s)")

    st.divider()

    st.markdown("### Priorités de durabilité")
    dims = {
        "durabilite": "Durabilité",
        "additionnalite": "Additionnalité",
        "disponibilite": "Disponibilité",
        "retombees_qc": "Retombées Québec",
        "liquidite": "Liquidité",
    }
    dw = st.session_state.durable_dim_weights
    new_dw = {}

    col1, col2 = st.columns(2)
    with col1:
        for k, label in list(dims.items())[:3]:
            new_dw[k] = st.slider(label, 0, 100, int(dw[k]*100), 5, key=f"dw_{k}") / 100
    with col2:
        for k, label in list(dims.items())[3:]:
            new_dw[k] = st.slider(label, 0, 100, int(dw[k]*100), 5, key=f"dw_{k}") / 100

    total_dw = sum(new_dw.values())
    if abs(total_dw - 1.0) > 0.01:
        st.warning(f"Total: {total_dw*100:.0f}% (doit être 100%). Normalisation automatique à l'application.")
    else:
        st.success(f"Total: {total_dw*100:.0f}%")

    categories_radar = list(dims.values())
    values_radar = [new_dw[k] * 100 for k in dims]
    fig_radar = go.Figure(go.Scatterpolar(
        r=values_radar + [values_radar[0]],
        theta=categories_radar + [categories_radar[0]],
        fill="toself", name="Priorités",
        line_color="#2ca02c",
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 60])),
        height=300, margin=dict(t=20, b=20),
        showlegend=False,
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.divider()
    st.markdown("### Paramètre financier")
    gamma = st.slider(
        "Aversion au risque (γ) — plus élevé = plus conservateur",
        1.0, 6.0, st.session_state.durable_gamma, 0.1,
    )

    st.divider()

    if st.button("✅ Appliquer la configuration", type="primary", use_container_width=True):
        if total_dw > 0:
            new_dw = {k: v/total_dw for k, v in new_dw.items()}
        st.session_state.durable_universe = universe
        st.session_state.durable_dim_weights = new_dw
        st.session_state.durable_gamma = gamma
        st.session_state.pop("durable_frontier", None)
        st.session_state.pop("durable_result", None)
        st.success("Configuration enregistrée.")

    with st.expander("Aperçu des scores composites"):
        use_durable_map = {aid: v["use_durable"] for aid, v in universe.items() if v["active"]}
        norm_dw = {k: v/total_dw for k, v in new_dw.items()} if total_dw > 0 else new_dw
        S_cur, _ = get_score_matrix(norm_dw, use_durable_map)
        active_ids = [aid for aid in DURABLE_ASSET_ORDER if universe[aid]["active"]]
        active_names = [
            DURABLE_ASSETS[aid].nom_durable
            if universe[aid]["use_durable"] and DURABLE_ASSETS[aid].has_durable_variant
            else DURABLE_ASSETS[aid].nom
            for aid in active_ids
        ]
        active_scores = [S_cur[DURABLE_ASSET_ORDER.index(aid)] for aid in active_ids]
        preview_df = pd.DataFrame({
            "Classe d'actifs": active_names,
            "Score composite": active_scores,
        }).sort_values("Score composite", ascending=True)
        st.dataframe(preview_df.style.format({"Score composite": "{:.2f}"}),
                     use_container_width=True, hide_index=True)


render()

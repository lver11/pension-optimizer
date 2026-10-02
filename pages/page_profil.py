"""
Profil du fonds : parametres propres a l'organisation utilisatrice.

Tout le reste de l'application (optimisation, contraintes, Monte Carlo, ALM,
reequilibrage, alpha portable, rapports) lit le profil actif.
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    ASSET_CLASSES_ORDER, ASSET_DEFAULTS, ASSET_CATEGORIES, CATEGORY_LABELS_FR,
    get_asset_codes,
)
from fund_profile import (
    FundProfile, GroupLimit, PROFILS_TYPES, TYPES_FONDS,
    apply_profile, get_active_profile, ensure_session_state,
)


def _group_options(profile: FundProfile):
    """Options proposees pour les limites de groupe : categories, classes, groupes existants."""
    options = {}
    for cat, label in CATEGORY_LABELS_FR.items():
        options[f"Categorie : {label}"] = [ac.value for ac in ASSET_CLASSES_ORDER
                                           if ASSET_CATEGORIES[ac] == cat]
    for ac in ASSET_CLASSES_ORDER:
        options[f"Classe : {ASSET_DEFAULTS[ac].nom_fr}"] = [ac.value]
    for g in profile.limites_groupes:
        if not any(sorted(v) == sorted(g.classes) for v in options.values()):
            options[f"Personnalise : {g.nom}"] = list(g.classes)
    return options


def _label_for(classes, options):
    for label, codes in options.items():
        if sorted(codes) == sorted(classes):
            return label
    return None


def render():
    st.title("Profil du fonds")
    ensure_session_state()
    profile = get_active_profile()
    version = st.session_state.get("profile_version", 0)

    st.caption(
        "Le profil regroupe tout ce qui est propre a votre organisation : taille, passif, flux, "
        "portefeuille de politique, bornes et limites de placement. Toutes les pages de "
        "l'application l'utilisent. Les profils types fournis sont illustratifs."
    )

    st.info(f"Profil actif : **{profile.nom}** - {TYPES_FONDS.get(profile.type_fonds, profile.type_fonds)}")

    # ------------------------------------------------------------------
    # Charger / importer / exporter
    # ------------------------------------------------------------------
    st.markdown("### Charger un profil")
    col1, col2 = st.columns(2)
    with col1:
        preset_names = {k: f().nom for k, f in PROFILS_TYPES.items()}
        preset_key = st.selectbox(
            "Profil type", list(preset_names), format_func=lambda k: preset_names[k],
        )
        if st.button("Charger ce profil type"):
            apply_profile(PROFILS_TYPES[preset_key]())
            st.rerun()
    with col2:
        uploaded = st.file_uploader("Importer un profil (JSON)", type=["json"])
        if uploaded is not None and st.button("Importer ce profil"):
            try:
                imported = FundProfile.from_json(uploaded.getvalue().decode("utf-8"))
                errors, warnings = imported.validate()
            except Exception as exc:  # JSON invalide ou champs inconnus
                st.error(f"Profil invalide : {exc}")
            else:
                if errors:
                    for e in errors:
                        st.error(e)
                else:
                    apply_profile(imported)
                    st.rerun()

    safe_name = "".join(c if c.isalnum() else "_" for c in profile.nom).strip("_").lower() or "profil"
    st.download_button(
        "Exporter le profil actif (JSON)",
        data=profile.to_json(),
        file_name=f"profil_{safe_name}.json",
        mime="application/json",
    )

    st.divider()

    # ------------------------------------------------------------------
    # Edition
    # ------------------------------------------------------------------
    st.markdown("### Modifier le profil")
    with st.form(f"profile_form_{version}"):
        st.markdown("#### Identification")
        c1, c2 = st.columns(2)
        nom = c1.text_input("Nom du fonds", profile.nom)
        type_keys = list(TYPES_FONDS)
        type_fonds = c2.selectbox(
            "Type de fonds", type_keys,
            index=type_keys.index(profile.type_fonds) if profile.type_fonds in type_keys else 0,
            format_func=lambda k: TYPES_FONDS[k],
        )
        description = st.text_area("Description / notes", profile.description, height=70)

        st.markdown("#### Bilan")
        c1, c2, c3 = st.columns(3)
        valeur_actif = c1.number_input(
            "Valeur de l'actif (M$)", 0.1, 1_000_000.0, float(profile.valeur_actif / 1e6), 10.0,
        ) * 1e6
        a_passif = c2.checkbox("Le fonds a un passif actuariel", profile.a_un_passif)
        valeur_passif = c3.number_input(
            "Valeur du passif (M$)", 0.0, 1_000_000.0,
            float((profile.valeur_passif or profile.valeur_actif) / 1e6), 10.0,
        ) * 1e6
        c1, c2, c3 = st.columns(3)
        duration_passif = c1.number_input("Duration du passif (ans)", 0.0, 40.0, float(profile.duration_passif), 0.5)
        taux_actualisation = c2.number_input(
            "Taux d'actualisation (%)", 0.0, 15.0, float(profile.taux_actualisation * 100), 0.1) / 100
        croissance_passif = c3.number_input(
            "Croissance du passif (%)", 0.0, 15.0, float(profile.croissance_passif * 100), 0.1) / 100

        st.markdown("#### Marche et horizon")
        c1, c2 = st.columns(2)
        taux_sans_risque = c1.number_input(
            "Taux sans risque (%)", 0.0, 15.0, float(profile.taux_sans_risque * 100), 0.1) / 100
        horizon = c2.number_input("Horizon de placement (ans)", 1, 100, int(profile.horizon_annees), 1)

        st.markdown("#### Flux annuels")
        st.caption("Entrees : cotisations, dons, souscriptions. Sorties : prestations, decaissements, rachats.")
        c1, c2, c3, c4 = st.columns(4)
        cotisations = c1.number_input(
            "Entrees (M$/an)", 0.0, 100_000.0, float(profile.cotisations_annuelles / 1e6), 1.0) * 1e6
        croiss_cotis = c2.number_input(
            "Croissance entrees (%)", -20.0, 20.0, float(profile.croissance_cotisations * 100), 0.5) / 100
        prestations = c3.number_input(
            "Sorties (M$/an)", 0.0, 100_000.0, float(profile.prestations_annuelles / 1e6), 1.0) * 1e6
        croiss_prest = c4.number_input(
            "Croissance sorties (%)", -20.0, 20.0, float(profile.croissance_prestations * 100), 0.5) / 100

        st.markdown("#### Portefeuille de politique et bornes")
        w, lo, hi = profile.weights_array(), profile.min_array(), profile.max_array()
        alloc_df = pd.DataFrame({
            "Classe d'actifs": [ASSET_DEFAULTS[ac].nom_fr for ac in ASSET_CLASSES_ORDER],
            "Categorie": [CATEGORY_LABELS_FR[ASSET_CATEGORIES[ac]] for ac in ASSET_CLASSES_ORDER],
            "Politique (%)": np.round(w * 100, 4),
            "Min (%)": np.round(lo * 100, 4),
            "Max (%)": np.round(hi * 100, 4),
        })
        pct = dict(min_value=0.0, max_value=100.0, step=0.5, format="%.2f")
        alloc_edit = st.data_editor(
            alloc_df, hide_index=True, use_container_width=True, key=f"alloc_{version}",
            disabled=["Classe d'actifs", "Categorie"],
            column_config={
                "Politique (%)": st.column_config.NumberColumn(**pct),
                "Min (%)": st.column_config.NumberColumn(**pct),
                "Max (%)": st.column_config.NumberColumn(**pct),
            },
        )
        st.caption("Une classe que le fonds n'utilise pas : mettre Max a 0.")

        st.markdown("#### Limites de groupe de la politique de placement")
        options = _group_options(profile)
        groups_df = pd.DataFrame({
            "Nom": [g.nom for g in profile.limites_groupes],
            "Classes visees": [_label_for(g.classes, options) for g in profile.limites_groupes],
            "Min (%)": [g.min * 100 for g in profile.limites_groupes],
            "Max (%)": [g.max * 100 for g in profile.limites_groupes],
        }, columns=["Nom", "Classes visees", "Min (%)", "Max (%)"])
        groups_edit = st.data_editor(
            groups_df, hide_index=True, use_container_width=True, num_rows="dynamic",
            key=f"groups_{version}",
            column_config={
                "Classes visees": st.column_config.SelectboxColumn(options=list(options), required=True),
                "Min (%)": st.column_config.NumberColumn(**pct, default=0.0),
                "Max (%)": st.column_config.NumberColumn(**pct, default=100.0),
            },
        )

        st.markdown("#### Hypotheses de marche propres au fonds (facultatif)")
        use_cma = st.checkbox(
            "Utiliser des hypotheses de rendement et de volatilite propres a ce profil",
            profile.rendements_attendus is not None or profile.volatilites is not None,
        )
        mu = profile.expected_returns_array()
        vol = profile.volatilities_array()
        cma_df = pd.DataFrame({
            "Classe d'actifs": [ASSET_DEFAULTS[ac].nom_fr for ac in ASSET_CLASSES_ORDER],
            "Rendement attendu (%)": np.round(
                (mu if mu is not None else [ASSET_DEFAULTS[ac].expected_return for ac in ASSET_CLASSES_ORDER])
                * np.ones(len(ASSET_CLASSES_ORDER)) * 100, 4),
            "Volatilite (%)": np.round(
                (vol if vol is not None else [ASSET_DEFAULTS[ac].volatility for ac in ASSET_CLASSES_ORDER])
                * np.ones(len(ASSET_CLASSES_ORDER)) * 100, 4),
        })
        cma_edit = st.data_editor(
            cma_df, hide_index=True, use_container_width=True, key=f"cma_{version}",
            disabled=["Classe d'actifs"],
        )

        submitted = st.form_submit_button("Valider et appliquer le profil", type="primary",
                                          use_container_width=True)

    if submitted:
        codes = get_asset_codes()
        new_groups = []
        for _, row in groups_edit.iterrows():
            if pd.isna(row["Classes visees"]) or row["Classes visees"] not in options:
                continue
            new_groups.append(GroupLimit(
                nom=str(row["Nom"]) if not pd.isna(row["Nom"]) and str(row["Nom"]).strip()
                else str(row["Classes visees"]).split(" : ", 1)[-1],
                classes=list(options[row["Classes visees"]]),
                min=float(row["Min (%)"] if not pd.isna(row["Min (%)"]) else 0.0) / 100,
                max=float(row["Max (%)"] if not pd.isna(row["Max (%)"]) else 100.0) / 100,
            ))

        def col(name):
            return alloc_edit[name].fillna(0.0).astype(float).to_numpy() / 100

        new_profile = FundProfile(
            nom=nom.strip() or "Fonds sans nom",
            type_fonds=type_fonds,
            description=description,
            devise=profile.devise,
            valeur_actif=valeur_actif,
            valeur_passif=valeur_passif if a_passif and valeur_passif > 0 else None,
            duration_passif=duration_passif,
            taux_actualisation=taux_actualisation,
            croissance_passif=croissance_passif,
            taux_sans_risque=taux_sans_risque,
            horizon_annees=int(horizon),
            cotisations_annuelles=cotisations,
            croissance_cotisations=croiss_cotis,
            prestations_annuelles=prestations,
            croissance_prestations=croiss_prest,
            poids_politique={c: v for c, v in zip(codes, col("Politique (%)")) if v > 0},
            bornes_min=dict(zip(codes, col("Min (%)"))),
            bornes_max=dict(zip(codes, col("Max (%)"))),
            limites_groupes=new_groups,
            rendements_attendus=dict(zip(codes, cma_edit["Rendement attendu (%)"].astype(float) / 100))
            if use_cma else None,
            volatilites=dict(zip(codes, cma_edit["Volatilite (%)"].astype(float) / 100))
            if use_cma else None,
        )
        errors, warnings = new_profile.validate()
        if errors:
            st.error("Le profil n'a pas ete applique :")
            for e in errors:
                st.error(f"- {e}")
        else:
            apply_profile(new_profile)
            st.session_state.profile_messages = warnings
            st.rerun()

    for msg in st.session_state.pop("profile_messages", []) or []:
        st.warning(msg)

    # ------------------------------------------------------------------
    # Resume
    # ------------------------------------------------------------------
    st.divider()
    st.markdown("### Resume du profil actif")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Actif", f"{profile.valeur_actif / 1e6:,.0f} M$")
    ratio = profile.ratio_capitalisation
    c2.metric("Ratio de capitalisation", f"{ratio:.1%}" if ratio is not None else "s.o.")
    c3.metric("Flux net annuel", f"{(profile.cotisations_annuelles - profile.prestations_annuelles) / 1e6:+,.0f} M$")
    c4.metric("Limites de groupe", f"{len(profile.limites_groupes)}")
    errors, warnings = profile.validate()
    if not errors and not warnings:
        st.success("Le portefeuille de politique respecte toutes les bornes et limites du profil.")
    for e in errors:
        st.error(e)
    for wmsg in warnings:
        st.warning(wmsg)


render()

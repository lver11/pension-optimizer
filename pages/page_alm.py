"""
Gestion Actif-Passif (ALM) - pour les fonds qui ont un passif actuariel (regimes PD).
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    get_asset_names_fr, get_expected_returns, get_covariance_matrix,
    get_min_weights, get_max_weights, DEFAULT_CURRENT_WEIGHTS,
    PensionFundConfig, ASSET_DEFAULTS, ASSET_CLASSES_ORDER, AssetClass,
)
from ui_notes import show_assumption_notes
from fund_profile import ensure_session_state, get_active_profile
from config import get_policy_weights
from data.generator import MarketDataGenerator
from models.alm import ALMOptimizer, LiabilityProfile
from constraints.manager import ConstraintSet
from constraints.regulatory import PolicyLimits
from visualization.charts import ChartBuilder


def render():
    st.title("Gestion actif-passif (ALM)")

    ensure_session_state()
    show_assumption_notes(simulated_returns_used=False)

    config = st.session_state.get("pension_config", PensionFundConfig())
    weights = st.session_state.get("current_weights", get_policy_weights())
    asset_names = get_asset_names_fr()
    profile = get_active_profile()

    if not profile.a_un_passif:
        st.info(
            f"Le profil actif (**{profile.nom}**) n'a pas de passif actuariel. "
            "L'analyse actif-passif s'applique aux regimes a prestations determinees : "
            "saisissez une valeur de passif dans la page Profil du fonds pour l'activer."
        )
        return

    # ---------- Configuration du passif (valeurs du profil) ----------
    measures = profile.liability_measures()
    cashflows = profile.liability_cashflows()
    curve = profile.yield_curve()
    st.sidebar.markdown("### Configuration du passif")
    if cashflows is not None:
        st.sidebar.caption(
            f"Passif calcule a partir de {measures['nb_annees']} annees de flux, "
            "actualises sur la courbe du profil."
        )
        pv_liabilities = measures["valeur"]
        liability_duration = measures["duration"]
        liability_convexity = measures["convexite"]
        discount_rate = float(curve.rate(liability_duration))
        st.sidebar.metric("Valeur du passif", f"{pv_liabilities/1e6:,.0f} M$")
        st.sidebar.metric("Duration effective", f"{liability_duration:.1f} ans")
        st.sidebar.metric("Convexite effective", f"{liability_convexity:.0f}")
    else:
        pv_liabilities = st.sidebar.number_input(
            "Valeur actuelle du passif (M$)", 1.0, 500000.0,
            float(profile.valeur_passif / 1e6), 10.0,
        ) * 1e6
        liability_duration = st.sidebar.slider(
            "Duration du passif (annees)", 1.0, 30.0, float(profile.duration_passif), 0.5)
        discount_rate = st.sidebar.slider(
            "Taux d'actualisation (%)", 0.0, 10.0, float(profile.taux_actualisation * 100), 0.1) / 100
        from models.liability import estimated_liability_convexity
        default_conv = (profile.convexite_passif if profile.convexite_passif is not None
                        else estimated_liability_convexity(liability_duration, discount_rate))
        liability_convexity = st.sidebar.number_input(
            "Convexite du passif", 0.0, 2000.0, float(round(default_conv)), 10.0,
            help="Estimee pour un passif de retraite typique de cette duration. "
                 "Importez les flux de prestations (page Profil du fonds) pour une valeur exacte.",
        )
    liability_growth = st.sidebar.slider(
        "Croissance du passif (%)", 0.0, 10.0, float(profile.croissance_passif * 100), 0.5) / 100

    liability_profile = LiabilityProfile(
        present_value=pv_liabilities,
        duration=liability_duration,
        convexity=liability_convexity,
        discount_rate=discount_rate,
        growth_rate=liability_growth,
    )

    # Durations des actifs
    asset_durations = np.array([
        ASSET_DEFAULTS[ac].duration if ASSET_DEFAULTS[ac].duration is not None else 0.0
        for ac in ASSET_CLASSES_ORDER
    ])

    asset_value = config.valeur_actif
    mu = get_expected_returns()
    cov = get_covariance_matrix()

    gov_index = ASSET_CLASSES_ORDER.index(AssetClass.OBLIGATIONS_GOV_CDN)
    alm = ALMOptimizer(
        mu, cov, asset_durations, liability_profile,
        config.taux_sans_risque, asset_names,
        get_min_weights(), get_max_weights(),
        reference_bond_index=gov_index,
        liability_cashflows=cashflows,
        curve=curve,
    )

    # ---------- Tableau de bord ALM ----------
    st.markdown("### Indicateurs actif-passif")
    funded_ratio = alm.compute_funded_ratio(asset_value)
    surplus = alm.compute_surplus(asset_value)
    duration_gap = alm.compute_duration_gap(weights, asset_value)

    diag = alm.assess_status(weights, asset_value)
    hedge_ratio_now = diag["ratio_couverture"]

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Ratio de capitalisation", f"{funded_ratio:.1%}")
    col2.metric("Surplus", f"{surplus/1e6:,.0f} M$")
    col3.metric("Ecart de duration", f"{duration_gap:.1f} ans",
                help="D_actif - (Passif/Actif) x D_passif. Negatif : le passif est plus sensible "
                     "aux taux que l'actif.")
    col4.metric("Couverture du risque de taux", f"{hedge_ratio_now:.0%}",
                help="Duration-dollar de l'actif / duration-dollar du passif.")
    col5.metric("Statut", diag["statut"].upper())

    message = (
        f"**{diag['composante_capitalisation']['texte']}** "
        f"**{diag['composante_taux']['texte']}**\n\n"
        + "\n".join(f"- {a}" for a in diag["actions"])
    )
    {"red": st.error, "orange": st.warning, "yellow": st.warning}.get(diag["couleur"], st.success)(message)

    with st.expander("Comment le statut est calcule"):
        st.markdown(f"""
Le statut combine deux composantes et retient **la plus severe** des deux.

**1. Niveau de capitalisation** (actif / passif)

| Ratio | Niveau |
|---|---|
| < 80 % | critique |
| 80 a 90 % | insuffisant |
| 90 a 100 % | a surveiller |
| 100 a 110 % | adequat |
| >= 110 % | excedentaire |

**2. Risque de taux** : perte de capitalisation si les taux baissent de 100 pb,
calculee avec les durations de l'actif et du passif (approximation de premier ordre, sans convexite).

| Perte de capitalisation | Niveau |
|---|---|
| <= 3 points | faible |
| 3 a 8 points | modere (a surveiller) |
| > 8 points | eleve |

Ici : capitalisation **{diag['composante_capitalisation']['niveau']}**, risque de taux
**{diag['composante_taux']['niveau']}** ({diag['delta_ratio_100pb_pts']:+.1f} points pour -100 pb).

Ces seuils sont des reperes illustratifs, pas des exigences reglementaires : chaque fonds
les fixe dans sa politique de financement et de placement.
""")

    st.divider()

    # ---------- Sensibilite aux taux ----------
    st.markdown("### Sensibilite aux taux d'interet")

    if cashflows is not None:
        st.caption("Passif : reevaluation exacte des flux sur la courbe choquee (convexite incluse). "
                   "Actif : duration + convexite approchee des classes a revenu fixe.")
    else:
        st.caption("Passif : duration + convexite (estimee ou saisie). Importez les flux de prestations "
                   "dans le profil pour une reevaluation exacte. Actif : duration + convexite approchee.")

    rate_scenarios = [-200, -100, -50, 50, 100, 200]
    sensitivity_results = []
    for shock in rate_scenarios:
        sens = alm.compute_interest_rate_sensitivity(weights, asset_value, shock)
        sensitivity_results.append({
            "Choc taux (pb)": shock,
            "Impact actif (M$)": sens["impact_actif"] / 1e6,
            "Impact passif (M$)": sens["impact_passif"] / 1e6,
            "dont convexite du passif (M$)": sens["effet_convexite_passif"] / 1e6,
            "Impact surplus (M$)": sens["impact_surplus"] / 1e6,
            "Impact ratio capit. (pts)": sens["impact_ratio_capit"] * 100,
        })

    sens_df = pd.DataFrame(sensitivity_results)
    st.dataframe(sens_df.style.format({
        "Impact actif (M$)": "{:+,.0f}",
        "Impact passif (M$)": "{:+,.0f}",
        "dont convexite du passif (M$)": "{:+,.0f}",
        "Impact surplus (M$)": "{:+,.0f}",
        "Impact ratio capit. (pts)": "{:+.1f}",
    }), use_container_width=True, hide_index=True)
    st.caption("« dont convexite » : ecart entre la variation reelle du passif et l'estimation par la "
               "duration seule. Il augmente le passif quand les taux baissent et attenue sa baisse quand ils montent.")

    # ---------- Durations cles ----------
    st.markdown("#### Exposition par segment de courbe")
    krd = alm.key_rate_exposures(weights, asset_value)
    krd_df = pd.DataFrame([
        {"Echeance": f"{int(t)} ans", "Actif (k$/pb)": v["actif_dv01"] / 1e3,
         "Passif (k$/pb)": v["passif_dv01"] / 1e3,
         "Ecart (k$/pb)": (v["actif_dv01"] - v["passif_dv01"]) / 1e3,
         "Couverture (%)": 100 * v["actif_dv01"] / v["passif_dv01"] if v["passif_dv01"] > 0 else np.nan}
        for t, v in krd.items()
    ])
    import plotly.graph_objects as go
    fig_krd = go.Figure()
    fig_krd.add_trace(go.Bar(name="Actif", x=krd_df["Echeance"], y=krd_df["Actif (k$/pb)"],
                             hovertemplate="%{x}<br>%{y:,.0f} k$/pb<extra>Actif</extra>"))
    fig_krd.add_trace(go.Bar(name="Passif", x=krd_df["Echeance"], y=krd_df["Passif (k$/pb)"],
                             hovertemplate="%{x}<br>%{y:,.0f} k$/pb<extra>Passif</extra>"))
    fig_krd.update_layout(barmode="group", yaxis_title="Valeur d'un point de base (k$)",
                          height=320, margin=dict(t=20, b=40))
    st.plotly_chart(fig_krd, use_container_width=True)
    st.dataframe(krd_df.style.format({
        "Actif (k$/pb)": "{:,.0f}", "Passif (k$/pb)": "{:,.0f}",
        "Ecart (k$/pb)": "{:+,.0f}", "Couverture (%)": "{:.0f}",
    }), use_container_width=True, hide_index=True)
    st.caption(
        "Gain ou perte en milliers de dollars pour une baisse de 1 pb du taux a chaque echeance. "
        + ("Passif : durations cles exactes calculees sur les flux. " if cashflows is not None
           else "Passif : sans flux, toute la duration est placee a l'echeance egale a la duration (approximation). ")
        + "Actif : chaque classe obligataire est repartie sur les deux echeances qui encadrent sa duration."
    )

    st.divider()

    # ---------- Optimisation du surplus ----------
    st.markdown("### Optimisation du surplus")

    if st.button("Optimiser le surplus", type="primary"):
        constraint_set = ConstraintSet(
            min_weights=get_min_weights(),
            max_weights=get_max_weights(),
            group_constraints=profile.group_constraints(),
        )

        with st.spinner("Optimisation du surplus en cours..."):
            result = alm.optimize_surplus(asset_value, constraint_set=constraint_set)

        if result.status == "optimal":
            st.success("Optimisation terminee!")
            st.session_state.alm_result = result

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Rendement de l'actif", f"{result.expected_return:.2%}")
            col2.metric("Volatilite du surplus", f"{result.metadata.get('surplus_volatility', 0):.2%}")
            col3.metric("Ecart duration",
                        f"{result.metadata.get('duration_gap', 0):.1f} ans")
            col4.metric("Couverture du risque de taux",
                        f"{result.metadata.get('hedge_ratio', 0):.0%}")
            st.caption(
                "Volatilite du surplus : le passif est modelise comme une position courte en "
                "obligations gouvernementales, mise a l'echelle par le rapport des durations "
                "(passif / obligations). Approximation : seul le risque de taux du passif est pris en compte."
            )

            fig = ChartBuilder.allocation_comparison_bar(
                weights, result.weights, asset_names,
                "Actuel vs Optimise (surplus)",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Optimisation echouee: {result.status}")

    st.divider()

    # ---------- Couverture du passif ----------
    st.markdown("### Ratio de couverture du risque de taux")
    st.caption(
        "Ratio de couverture = duration-dollar de l'actif / duration-dollar du passif. "
        "A 100 %, une variation des taux change l'actif et le passif du meme montant en dollars. "
        "Ce n'est pas la meme chose que le pourcentage de l'actif en obligations : "
        "il depend aussi de la duration des obligations et du ratio de capitalisation."
    )
    col1, col2 = st.columns(2)
    hedge_target = col1.slider("Ratio de couverture cible", 0.0, 1.0, 0.80, 0.05)
    hedge_duration = col2.number_input(
        "Duration des obligations de couverture (ans)", 1.0, 30.0,
        float(max(asset_durations)), 0.5,
        help="Par defaut, la plus longue duration disponible dans l'univers. "
             "Des obligations long terme ont souvent une duration de 14 a 18 ans.",
    )
    hedge_result = alm.optimize_liability_hedge(
        asset_value, hedge_target, hedge_duration=hedge_duration, weights=weights,
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Couverture actuelle", f"{hedge_ratio_now:.0%}")
    c2.metric("Couverture cible", f"{hedge_target:.0%}")
    c3.metric("Poids obligataire requis", f"{hedge_result['poids_actifs_couverture_recommande']:.0%}")
    (st.info if hedge_result["faisable_sans_levier"] else st.warning)(hedge_result["recommandation"])

    st.divider()

    # ---------- Glide path ----------
    st.markdown("### Trajectoire de desensibilisation (Glide Path)")
    col1, col2 = st.columns(2)
    with col1:
        gp_horizon = st.slider("Horizon glide path (annees)", 5, 20, 10)
    with col2:
        target_fr = st.slider("Ratio cible", 1.00, 1.30, 1.10, 0.05)

    glide_path = alm.optimize_glide_path(
        funded_ratio, gp_horizon, target_fr, asset_value,
    )

    fig_gp = ChartBuilder.glide_path_area(glide_path)
    st.plotly_chart(fig_gp, use_container_width=True)

    gp_df = pd.DataFrame(glide_path)
    st.dataframe(gp_df.style.format({
        "Ratio capitalisation projete": "{:.1%}",
        "Actifs de croissance (%)": "{:.1f}",
        "Actifs de couverture (%)": "{:.1f}",
        "Encaisse (%)": "{:.1f}",
    }), use_container_width=True, hide_index=True)

    # ---------- Flux de tresorerie ----------
    st.markdown("### Flux de tresorerie projetes (profil du fonds)")
    st.caption("Entrees et sorties annuelles du profil, avec leurs taux de croissance.")
    years = np.arange(1, 31)
    contrib = profile.cotisations_annuelles * (1 + profile.croissance_cotisations) ** (years - 1)
    benef = profile.prestations_annuelles * (1 + profile.croissance_prestations) ** (years - 1)
    cashflows = pd.DataFrame({
        "Annee": years, "Cotisations": contrib, "Prestations": benef, "Flux_net": contrib - benef,
    })

    import plotly.graph_objects as go
    fig_cf = go.Figure()
    fig_cf.add_trace(go.Bar(
        x=cashflows["Annee"], y=cashflows["Cotisations"] / 1e6,
        name="Cotisations", marker_color="green",
    ))
    fig_cf.add_trace(go.Bar(
        x=cashflows["Annee"], y=-cashflows["Prestations"] / 1e6,
        name="Prestations", marker_color="red",
    ))
    fig_cf.add_trace(go.Scatter(
        x=cashflows["Annee"], y=cashflows["Flux_net"] / 1e6,
        mode="lines+markers", name="Flux net",
        line=dict(color="blue", width=2),
    ))
    fig_cf.update_layout(
        title="Flux de tresorerie annuels projetes",
        xaxis_title="Annee",
        yaxis_title="Montant (M$)",
        barmode="relative",
        height=400,
    )
    st.plotly_chart(fig_cf, use_container_width=True)


render()

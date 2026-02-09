"""
Moteur d'optimisation de portefeuille.
Selection du modele, parametres, resultats comparatifs.
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    get_asset_names_fr, get_expected_returns, get_covariance_matrix,
    get_min_weights, get_max_weights, DEFAULT_CURRENT_WEIGHTS,
    PensionFundConfig, ASSET_DEFAULTS, ASSET_CLASSES_ORDER,
)
from data.generator import MarketDataGenerator
from models.mean_variance import MeanVarianceOptimizer
from models.black_litterman import BlackLittermanOptimizer
from models.risk_parity import RiskParityOptimizer
from models.cvar_optimizer import CVaROptimizer
from risk.covariance import CovarianceEstimator
from constraints.manager import ConstraintSet
from constraints.regulatory import QuebecPensionRegulations
from visualization.charts import ChartBuilder


def render():
    st.title("Moteur d'optimisation")

    # --- Initialisation des donnees ---
    if "returns_data" not in st.session_state or st.session_state.returns_data is None:
        generator = MarketDataGenerator(seed=42)
        st.session_state.returns_data = generator.generate_returns(n_years=20, frequency="monthly")
        st.session_state.current_weights = DEFAULT_CURRENT_WEIGHTS.copy()
        st.session_state.pension_config = PensionFundConfig()

    asset_names = get_asset_names_fr()
    returns_data = st.session_state.returns_data
    config = st.session_state.get("pension_config", PensionFundConfig())
    current_weights = st.session_state.get("current_weights", DEFAULT_CURRENT_WEIGHTS)

    # --- Selection du modele ---
    st.markdown("### Configuration de l'optimisation")

    col1, col2, col3 = st.columns(3)

    with col1:
        model_choice = st.selectbox("Modele d'optimisation", [
            "Moyenne-Variance (Markowitz)",
            "Black-Litterman",
            "Parite de risque",
            "CVaR (Rockafellar-Uryasev)",
        ])

    with col2:
        cov_method = st.selectbox("Estimation de covariance", [
            "Ledoit-Wolf", "Echantillon", "EWMA",
        ])

    with col3:
        apply_constraints = st.checkbox("Contraintes reglementaires Quebec", True)

    method_map = {"Ledoit-Wolf": "ledoit_wolf", "Echantillon": "sample", "EWMA": "ewma"}
    cov_matrix = CovarianceEstimator.estimate(returns_data, method_map[cov_method])

    mu = get_expected_returns()
    rf = config.taux_sans_risque

    # Contraintes
    constraint_set = None
    if apply_constraints:
        constraint_set = ConstraintSet(
            min_weights=get_min_weights(),
            max_weights=get_max_weights(),
            group_constraints=QuebecPensionRegulations.get_group_constraints(),
        )

    st.divider()

    # --- Parametres specifiques au modele ---
    if model_choice == "Moyenne-Variance (Markowitz)":
        st.markdown("### Parametres Moyenne-Variance")
        objective = st.selectbox("Objectif", [
            "max_sharpe", "min_variance", "target_return", "target_risk",
        ], format_func=lambda x: {
            "max_sharpe": "Maximiser le ratio de Sharpe",
            "min_variance": "Minimiser la variance",
            "target_return": "Rendement cible",
            "target_risk": "Risque cible",
        }[x])

        target_return = None
        target_risk = None
        if objective == "target_return":
            target_return = st.slider(
                "Rendement cible annuel (%)", 2.0, 12.0, 6.0, 0.1,
            ) / 100
        elif objective == "target_risk":
            target_risk = st.slider(
                "Volatilite cible annuelle (%)", 2.0, 20.0, 10.0, 0.5,
            ) / 100

    elif model_choice == "Black-Litterman":
        st.markdown("### Parametres Black-Litterman")
        st.caption("Definissez vos vues sur les rendements des classes d'actifs.")

        # Market cap weights par defaut
        market_weights = current_weights.copy()
        delta = st.slider("Aversion au risque (delta)", 1.0, 5.0, 2.5, 0.1)
        tau = st.slider("Parametre d'incertitude (tau)", 0.01, 0.1, 0.05, 0.01)

        # Interface pour ajouter des vues
        st.markdown("#### Vues de l'investisseur")
        n_views = st.number_input("Nombre de vues", 0, 5, 1)

        views_P = []
        views_Q = []
        views_confidence = []

        for v_idx in range(int(n_views)):
            st.markdown(f"**Vue {v_idx + 1}**")
            col_v1, col_v2, col_v3 = st.columns(3)
            with col_v1:
                view_type = st.selectbox(
                    f"Type",
                    ["Absolue", "Relative"],
                    key=f"view_type_{v_idx}",
                )
            with col_v2:
                asset_idx1 = st.selectbox(
                    f"Actif",
                    range(len(asset_names)),
                    format_func=lambda i: asset_names[i],
                    key=f"view_asset1_{v_idx}",
                )
            with col_v3:
                view_return = st.slider(
                    f"Rendement annuel (%)",
                    -10.0, 20.0, 8.0, 0.5,
                    key=f"view_return_{v_idx}",
                ) / 100

            if view_type == "Absolue":
                P_row = np.zeros(len(asset_names))
                P_row[asset_idx1] = 1.0
                views_P.append(P_row)
                views_Q.append(view_return)
            else:
                col_r1, col_r2 = st.columns(2)
                with col_r1:
                    asset_idx2 = st.selectbox(
                        f"vs Actif",
                        range(len(asset_names)),
                        format_func=lambda i: asset_names[i],
                        index=min(1, len(asset_names) - 1),
                        key=f"view_asset2_{v_idx}",
                    )
                P_row = np.zeros(len(asset_names))
                P_row[asset_idx1] = 1.0
                P_row[asset_idx2] = -1.0
                views_P.append(P_row)
                views_Q.append(view_return)

            confidence = st.slider(
                f"Confiance (%)",
                10, 100, 75, 5,
                key=f"view_conf_{v_idx}",
            ) / 100
            views_confidence.append(confidence)

    elif model_choice == "Parite de risque":
        st.markdown("### Parametres Parite de risque")
        st.caption("Chaque classe d'actifs contribue egalement (ou selon un budget) au risque total.")

        use_custom_budget = st.checkbox("Budget de risque personnalise", False)
        risk_budget = None
        if use_custom_budget:
            budget_values = []
            cols = st.columns(4)
            for i, name in enumerate(asset_names):
                with cols[i % 4]:
                    b = st.number_input(
                        f"{name[:20]}", 0.01, 1.0,
                        round(1.0 / len(asset_names), 3), 0.01,
                        key=f"budget_{i}",
                    )
                    budget_values.append(b)
            risk_budget = np.array(budget_values)
            risk_budget /= risk_budget.sum()

    elif model_choice == "CVaR (Rockafellar-Uryasev)":
        st.markdown("### Parametres CVaR")
        cvar_objective = st.selectbox("Objectif CVaR", [
            "min_cvar", "target_return", "max_return_target_cvar",
        ], format_func=lambda x: {
            "min_cvar": "Minimiser la CVaR",
            "target_return": "Rendement cible (min CVaR)",
            "max_return_target_cvar": "Maximiser le rendement (CVaR cible)",
        }[x])

        confidence_cvar = st.slider(
            "Niveau de confiance (%)", 90.0, 99.0, 95.0, 0.5,
        ) / 100

        cvar_target_return = None
        cvar_target_cvar = None
        if cvar_objective == "target_return":
            cvar_target_return = st.slider(
                "Rendement cible annuel (%)", 2.0, 12.0, 6.0, 0.1,
            ) / 100
        elif cvar_objective == "max_return_target_cvar":
            cvar_target_cvar = st.slider(
                "CVaR cible (%)", 1.0, 15.0, 5.0, 0.5,
            ) / 100

    st.divider()

    # --- Bouton d'optimisation ---
    if st.button("Lancer l'optimisation", type="primary", use_container_width=True):
        with st.spinner("Optimisation en cours..."):
            try:
                if model_choice == "Moyenne-Variance (Markowitz)":
                    optimizer = MeanVarianceOptimizer(
                        expected_returns=mu,
                        cov_matrix=cov_matrix,
                        risk_free_rate=rf,
                        asset_names=asset_names,
                        min_weights=get_min_weights(),
                        max_weights=get_max_weights(),
                    )
                    result = optimizer.optimize(
                        objective=objective,
                        target_return=target_return,
                        target_risk=target_risk,
                        constraint_set=constraint_set,
                    )

                elif model_choice == "Black-Litterman":
                    bl_optimizer = BlackLittermanOptimizer(
                        expected_returns=mu,
                        cov_matrix=cov_matrix,
                        market_weights=market_weights,
                        risk_free_rate=rf,
                        tau=tau,
                        asset_names=asset_names,
                        min_weights=get_min_weights(),
                        max_weights=get_max_weights(),
                    )
                    implied = bl_optimizer.compute_implied_returns()

                    if len(views_P) > 0:
                        P = np.array(views_P)
                        Q = np.array(views_Q)
                        bl_optimizer.set_views(
                            P, Q,
                            confidence=np.array(views_confidence),
                        )
                    else:
                        # Pas de vues : utiliser les rendements implicites
                        bl_optimizer.mu = implied

                    result = bl_optimizer.optimize(constraint_set=constraint_set)

                elif model_choice == "Parite de risque":
                    rp_optimizer = RiskParityOptimizer(
                        expected_returns=mu,
                        cov_matrix=cov_matrix,
                        risk_budgets=risk_budget,
                        risk_free_rate=rf,
                        asset_names=asset_names,
                        min_weights=get_min_weights(),
                        max_weights=get_max_weights(),
                    )
                    result = rp_optimizer.optimize(constraint_set=constraint_set)

                elif model_choice == "CVaR (Rockafellar-Uryasev)":
                    cvar_optimizer = CVaROptimizer(
                        expected_returns=mu,
                        cov_matrix=cov_matrix,
                        scenarios=returns_data.values,
                        confidence_level=confidence_cvar,
                        risk_free_rate=rf,
                        asset_names=asset_names,
                        min_weights=get_min_weights(),
                        max_weights=get_max_weights(),
                    )
                    result = cvar_optimizer.optimize(
                        objective=cvar_objective,
                        target_return=cvar_target_return,
                        target_cvar=cvar_target_cvar,
                        constraint_set=constraint_set,
                    )

                st.session_state.optimization_result = result

                if result.status == "optimal":
                    st.success(
                        f"Optimisation reussie ({result.solver_time:.3f}s) | "
                        f"Rendement: {result.expected_return:.2%} | "
                        f"Volatilite: {result.volatility:.2%} | "
                        f"Sharpe: {result.sharpe_ratio:.3f}"
                    )
                else:
                    st.warning(f"Statut: {result.status}. Portefeuille equipondere utilise.")

            except Exception as e:
                st.error(f"Erreur d'optimisation: {str(e)}")
                return

    st.divider()

    # --- Affichage des resultats ---
    if "optimization_result" in st.session_state and st.session_state.optimization_result is not None:
        result = st.session_state.optimization_result

        st.markdown("## Resultats de l'optimisation")

        # KPIs comparatifs
        port_ret_current = current_weights @ mu
        port_vol_current = np.sqrt(current_weights @ cov_matrix @ current_weights)
        sharpe_current = (port_ret_current - rf) / port_vol_current if port_vol_current > 1e-10 else 0.0

        col1, col2, col3 = st.columns(3)

        col1.markdown("**Metrique**")
        col2.markdown("**Portefeuille actuel**")
        col3.markdown("**Portefeuille optimise**")

        metrics_comparison = [
            ("Rendement", f"{port_ret_current:.2%}", f"{result.expected_return:.2%}"),
            ("Volatilite", f"{port_vol_current:.2%}", f"{result.volatility:.2%}"),
            ("Sharpe", f"{sharpe_current:.3f}", f"{result.sharpe_ratio:.3f}"),
        ]

        for name, curr, opt in metrics_comparison:
            col1, col2, col3 = st.columns(3)
            col1.write(name)
            col2.write(curr)
            col3.write(opt)

        st.divider()

        # Graphiques
        tab1, tab2, tab3 = st.tabs([
            "Comparaison des allocations",
            "Allocation optimisee",
            "Contribution au risque",
        ])

        with tab1:
            fig_comp = ChartBuilder.allocation_comparison_bar(
                current_weights, result.weights, asset_names,
            )
            st.plotly_chart(fig_comp, use_container_width=True)

            # Tableau comparatif detaille
            ecarts = result.weights - current_weights
            comp_df = pd.DataFrame({
                "Classe d'actifs": asset_names,
                "Actuel (%)": current_weights * 100,
                "Optimise (%)": result.weights * 100,
                "Ecart (pp)": ecarts * 100,
            })
            st.dataframe(
                comp_df.style.format({
                    "Actuel (%)": "{:.1f}",
                    "Optimise (%)": "{:.1f}",
                    "Ecart (pp)": "{:+.1f}",
                }),
                use_container_width=True,
                hide_index=True,
            )

        with tab2:
            fig_pie = ChartBuilder.allocation_pie(
                result.weights, asset_names, "Allocation optimisee",
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with tab3:
            fig_risk = ChartBuilder.risk_contribution_bar(
                result.risk_contributions, asset_names,
                "Contribution au risque - Portefeuille optimise",
            )
            st.plotly_chart(fig_risk, use_container_width=True)

        # Bouton pour adopter le portefeuille optimise
        st.divider()
        if st.button("Adopter le portefeuille optimise comme portefeuille actuel"):
            st.session_state.current_weights = result.weights.copy()
            st.success("Portefeuille actuel mis a jour avec l'allocation optimisee.")
            st.rerun()

    else:
        st.info(
            "Configurez les parametres ci-dessus et cliquez sur "
            "**Lancer l'optimisation** pour obtenir des resultats."
        )


render()

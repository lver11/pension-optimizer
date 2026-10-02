"""
Optimiseur de portefeuille institutionnel
Application Streamlit multi-pages pour l'optimisation de portefeuille
multi-classes d'actifs de tout fonds institutionnel (regime PD ou CD,
fondation, fonds de travailleurs...). Les parametres propres au fonds
sont regroupes dans un profil (page Profil du fonds).
"""

import streamlit as st
import sys
import os

# Ajouter le repertoire racine au path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def _reset_session_if_universe_changed():
    """Reinitialise le session_state si l'univers d'actifs a change de taille.

    Necessaire lors des redéploiements qui ajoutent ou retirent des classes
    d'actifs : current_weights (ancien n) serait desynchronise avec
    ASSET_CLASSES_ORDER (nouveau n), provoquant des erreurs silencieuses ou
    des crashs dans les graphiques et les optimisations.
    """
    from config import ASSET_CLASSES_ORDER
    n_expected = len(ASSET_CLASSES_ORDER)
    cached_weights = st.session_state.get("current_weights")
    if cached_weights is not None and len(cached_weights) != n_expected:
        keys_to_clear = [
            "returns_data", "current_weights", "optimization_result",
            "custom_expected_returns", "custom_volatilities",
        ]
        for key in keys_to_clear:
            st.session_state.pop(key, None)


def main():
    st.set_page_config(
        page_title="Optimiseur de portefeuille institutionnel",
        page_icon="\U0001F4CA",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Reinitialiser le session_state si l'univers d'actifs a change de taille
    _reset_session_if_universe_changed()

    # Style CSS personnalise (compatible dark/light mode)
    st.markdown("""
    <style>
        .stMetric {
            background-color: rgba(255, 255, 255, 0.05);
            padding: 10px;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .stMetric label {
            font-size: 0.85rem !important;
        }
        div[data-testid="stSidebar"] {
            background-color: #1a1a2e;
        }
        div[data-testid="stSidebar"] .stMarkdown {
            color: white;
        }
        .block-container {
            padding-top: 1rem;
        }
        /* Light mode override */
        @media (prefers-color-scheme: light) {
            .stMetric {
                background-color: #f8f9fa;
                border: 1px solid #e9ecef;
            }
        }
    </style>
    """, unsafe_allow_html=True)

    # Profil de fonds actif (profil type par defaut au premier chargement)
    from fund_profile import ensure_session_state, get_active_profile, TYPES_FONDS
    ensure_session_state()
    profile = get_active_profile()

    # Sidebar - Configuration globale
    with st.sidebar:
        st.markdown(f"## \U0001F3E6 {profile.nom}")
        st.caption(TYPES_FONDS.get(profile.type_fonds, profile.type_fonds)
                   + " - modifier dans Profil du fonds")
        st.markdown("---")

        # Parametres globaux (lies au profil actif)
        st.markdown("### Parametres globaux")

        profile.valeur_actif = st.number_input(
            "Valeur de l'actif (M$)",
            0.1, 1_000_000.0, float(profile.valeur_actif / 1e6), 10.0,
        ) * 1e6

        profile.taux_sans_risque = st.slider(
            "Taux sans risque (%)", 0.0, 15.0,
            float(profile.taux_sans_risque * 100), 0.1,
        ) / 100

        profile.horizon_annees = st.slider(
            "Horizon (annees)", 1, 60, int(profile.horizon_annees),
        )

        # Conserver les autres champs eventuellement modifies par les pages
        config = st.session_state.get("pension_config") or profile.to_pension_config()
        config.nom = profile.nom
        config.valeur_actif = profile.valeur_actif
        config.taux_sans_risque = profile.taux_sans_risque
        config.horizon_annees = int(profile.horizon_annees)
        config.valeur_passif = profile.valeur_passif or 0.0
        st.session_state.pension_config = config

        st.markdown("---")

    # Navigation multi-pages (chemins absolus pour eviter les problemes de CWD)
    pages_dir = os.path.join(ROOT_DIR, "pages")

    pages = {
        "Vue d'ensemble": [
            st.Page(os.path.join(pages_dir, "page_dashboard.py"), title="Tableau de bord", icon=":material/dashboard:", default=True),
            st.Page(os.path.join(pages_dir, "page_profil.py"), title="Profil du fonds", icon=":material/account_balance:"),
            st.Page(os.path.join(pages_dir, "page_data_source.py"), title="Source de donnees", icon=":material/database:"),
        ],
        "Optimisation": [
            st.Page(os.path.join(pages_dir, "page_optimization.py"), title="Moteur d'optimisation", icon=":material/tune:"),
            st.Page(os.path.join(pages_dir, "page_constraints.py"), title="Gestionnaire de contraintes", icon=":material/lock:"),
            st.Page(os.path.join(pages_dir, "page_frontier.py"), title="Frontiere efficiente", icon=":material/show_chart:"),
        ],
        "Analyse de risque": [
            st.Page(os.path.join(pages_dir, "page_risk.py"), title="Analytique de risque", icon=":material/warning:"),
            st.Page(os.path.join(pages_dir, "page_monte_carlo.py"), title="Simulation Monte Carlo", icon=":material/casino:"),
            st.Page(os.path.join(pages_dir, "page_alm.py"), title="Gestion actif-passif", icon=":material/balance:"),
        ],
        "Strategies": [
            st.Page(os.path.join(pages_dir, "page_portable_alpha.py"), title="Alpha portable", icon=":material/trending_up:"),
        ],
        "Gestion": [
            st.Page(os.path.join(pages_dir, "page_rebalancing.py"), title="Reequilibrage", icon=":material/sync:"),
            st.Page(os.path.join(pages_dir, "page_reports.py"), title="Rapports", icon=":material/description:"),
        ],
        "🌱 Optimisation durable": [
            st.Page(os.path.join(pages_dir, "page_durable_univers.py"),
                    title="Univers durable", icon=":material/eco:"),
            st.Page(os.path.join(pages_dir, "page_durable_scores.py"),
                    title="Scores de durabilité", icon=":material/star:"),
            st.Page(os.path.join(pages_dir, "page_durable_frontier.py"),
                    title="Frontière durable", icon=":material/scatter_plot:"),
            st.Page(os.path.join(pages_dir, "page_durable_optimization.py"),
                    title="Optimisation durable", icon=":material/tune:"),
            st.Page(os.path.join(pages_dir, "page_durable_rapport.py"),
                    title="Rapport durable", icon=":material/description:"),
        ],
        "Aide": [
            st.Page(os.path.join(pages_dir, "page_documentation.py"), title="Documentation", icon=":material/menu_book:"),
        ],
    }

    pg = st.navigation(pages)
    pg.run()


if __name__ == "__main__":
    main()

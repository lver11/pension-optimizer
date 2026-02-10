"""
Optimiseur de Portefeuille Institutionnel - Caisse de Retraite
Application Streamlit multi-pages pour l'optimisation de portefeuille
multi-classes d'actifs d'un fonds de pension.
"""

import streamlit as st
import sys
import os

# Ajouter le repertoire racine au path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def main():
    st.set_page_config(
        page_title="Optimiseur de Portefeuille - Caisse de Retraite",
        page_icon="\U0001F4CA",
        layout="wide",
        initial_sidebar_state="expanded",
    )

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

    # Sidebar - Configuration globale
    with st.sidebar:
        st.markdown("## \U0001F3E6 Caisse de Retraite")
        st.markdown("---")

        # Parametres globaux
        st.markdown("### Parametres globaux")

        if "pension_config" not in st.session_state:
            from config import PensionFundConfig
            st.session_state.pension_config = PensionFundConfig()

        config = st.session_state.pension_config

        config.valeur_actif = st.number_input(
            "Valeur de l'actif (M$)",
            100.0, 50000.0, config.valeur_actif / 1e6, 50.0,
        ) * 1e6

        config.valeur_passif = st.number_input(
            "Valeur du passif (M$)",
            100.0, 50000.0, config.valeur_passif / 1e6, 50.0,
        ) * 1e6

        config.taux_sans_risque = st.slider(
            "Taux sans risque (%)", 0.0, 8.0,
            config.taux_sans_risque * 100, 0.1,
        ) / 100

        config.horizon_annees = st.slider(
            "Horizon (annees)", 5, 40, config.horizon_annees,
        )

        st.session_state.pension_config = config

        # Indicateur ratio de capitalisation
        fr = config.valeur_actif / config.valeur_passif
        st.markdown("---")
        if fr >= 1.0:
            st.success(f"Ratio de capitalisation: **{fr:.1%}**")
        elif fr >= 0.85:
            st.warning(f"Ratio de capitalisation: **{fr:.1%}**")
        else:
            st.error(f"Ratio de capitalisation: **{fr:.1%}**")

        st.markdown("---")

        # Source de donnees
        st.markdown("### Source de donnees")
        data_source = st.radio(
            "Donnees",
            ["Simulees", "Importees"],
            label_visibility="collapsed",
        )

        if data_source == "Simulees":
            seed = st.number_input("Graine aleatoire", 1, 99999, 42)
            n_years = st.slider("Historique (annees)", 5, 30, 20)

            if st.button("Regenerer les donnees"):
                from data.generator import MarketDataGenerator
                from config import DEFAULT_CURRENT_WEIGHTS
                generator = MarketDataGenerator(seed=seed)
                st.session_state.returns_data = generator.generate_returns(
                    n_years=n_years, frequency="monthly",
                )
                st.session_state.current_weights = DEFAULT_CURRENT_WEIGHTS.copy()
                st.success("Donnees regenerees!")

    # Navigation multi-pages (chemins absolus pour eviter les problemes de CWD)
    pages_dir = os.path.join(ROOT_DIR, "pages")

    pages = {
        "Vue d'ensemble": [
            st.Page(os.path.join(pages_dir, "page_dashboard.py"), title="Tableau de bord", icon=":material/dashboard:"),
        ],
        "Optimisation": [
            st.Page(os.path.join(pages_dir, "page_optimization.py"), title="Moteur d'optimisation", icon=":material/tune:"),
            st.Page(os.path.join(pages_dir, "page_constraints.py"), title="Gestionnaire de contraintes", icon=":material/lock:"),
            st.Page(os.path.join(pages_dir, "page_frontier.py"), title="Frontiere efficiente", icon=":material/show_chart:"),
        ],
        "Analyse de risque": [
            st.Page(os.path.join(pages_dir, "page_risk.py"), title="Analytique de risque", icon=":material/warning:"),
            st.Page(os.path.join(pages_dir, "page_monte_carlo.py"), title="Simulation Monte Carlo", icon=":material/casino:"),
        ],
        "Strategies": [
            st.Page(os.path.join(pages_dir, "page_portable_alpha.py"), title="Alpha portable", icon=":material/trending_up:"),
        ],
        "Gestion": [
            st.Page(os.path.join(pages_dir, "page_rebalancing.py"), title="Reequilibrage", icon=":material/sync:"),
            st.Page(os.path.join(pages_dir, "page_alm.py"), title="Gestion actif-passif", icon=":material/balance:"),
            st.Page(os.path.join(pages_dir, "page_reports.py"), title="Rapports", icon=":material/description:"),
        ],
        "Aide": [
            st.Page(os.path.join(pages_dir, "page_documentation.py"), title="Documentation", icon=":material/menu_book:"),
        ],
    }

    pg = st.navigation(pages)
    pg.run()


if __name__ == "__main__":
    main()

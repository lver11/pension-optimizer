"""
Page Streamlit - Documentation
Guide d'utilisation et lexique des concepts financiers integres dans l'application.
"""

import streamlit as st
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def load_markdown(filename: str) -> str:
    """Charge un fichier markdown depuis la racine du projet."""
    filepath = os.path.join(ROOT_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    return f"Fichier {filename} introuvable."


st.header("Documentation")

tab_guide, tab_lexique, tab_recherche = st.tabs([
    "Guide d'utilisation",
    "Lexique des concepts",
    "Recherche",
])

# ============================================================
# Onglet 1 : Guide d'utilisation
# ============================================================
with tab_guide:
    guide_md = load_markdown("GUIDE_UTILISATION.md")

    # Sidebar - table des matieres rapide
    with st.sidebar:
        st.markdown("### Navigation rapide")
        st.markdown("""
- [Demarrage](#1-demarrage)
- [Configuration globale](#2-configuration-globale-sidebar)
- [Tableau de bord](#3-1-vue-d-ensemble-tableau-de-bord)
- [Optimisation](#3-2-optimisation-moteur-d-optimisation)
- [Contraintes](#3-3-optimisation-gestionnaire-de-contraintes)
- [Frontiere efficiente](#3-4-optimisation-frontiere-efficiente)
- [Analytique de risque](#3-5-analyse-de-risque-analytique-de-risque)
- [Monte Carlo](#3-6-analyse-de-risque-simulation-monte-carlo)
- [Alpha portable](#3-7-strategies-alpha-portable)
- [Reequilibrage](#3-8-gestion-reequilibrage)
- [Gestion actif-passif](#3-9-gestion-gestion-actif-passif-alm)
- [Rapports](#3-10-gestion-rapports)
- [Flux de travail](#4-flux-de-travail-recommande)
- [Classes d'actifs](#5-classes-d-actifs-disponibles-12)
- [Contraintes reglementaires](#6-contraintes-reglementaires-du-quebec)
        """)

    st.markdown(guide_md)

# ============================================================
# Onglet 2 : Lexique
# ============================================================
with tab_lexique:
    lexique_md = load_markdown("LEXIQUE.md")

    # Sections du lexique pour navigation
    with st.sidebar:
        st.markdown("### Sections du lexique")
        st.markdown("""
- [A. Mesures de rendement](#a-mesures-de-rendement)
- [B. Mesures de risque](#b-mesures-de-risque)
- [C. Ratios de performance](#c-ratios-de-performance)
- [D. Modeles d'optimisation](#d-modeles-d-optimisation)
- [E. Frontiere efficiente](#e-frontiere-efficiente)
- [F. Gestion actif-passif](#f-gestion-actif-passif-alm)
- [G. Simulation Monte Carlo](#g-simulation-monte-carlo)
- [H. Alpha portable](#h-alpha-portable)
- [I. Methodes de covariance](#i-methodes-d-estimation-de-la-covariance)
- [J. Tests de tension](#j-tests-de-tension-stress-testing)
- [K. Reequilibrage](#k-reequilibrage)
- [L. ESG et contraintes](#l-esg-et-contraintes)
- [M. Termes techniques](#m-termes-techniques-de-l-optimisation)
        """)

    st.markdown(lexique_md)

# ============================================================
# Onglet 3 : Recherche dans la documentation
# ============================================================
with tab_recherche:
    st.subheader("Recherche dans la documentation")
    st.markdown("Tapez un terme pour le trouver dans le guide et le lexique.")

    query = st.text_input(
        "Rechercher un terme",
        placeholder="Ex: CVaR, Sharpe, Black-Litterman, levier...",
    )

    if query and len(query) >= 2:
        query_lower = query.lower()

        guide_content = load_markdown("GUIDE_UTILISATION.md")
        lexique_content = load_markdown("LEXIQUE.md")

        # Recherche dans le guide
        guide_results = []
        for i, line in enumerate(guide_content.split("\n")):
            if query_lower in line.lower():
                guide_results.append((i + 1, line.strip()))

        # Recherche dans le lexique
        lexique_results = []
        for i, line in enumerate(lexique_content.split("\n")):
            if query_lower in line.lower():
                lexique_results.append((i + 1, line.strip()))

        total = len(guide_results) + len(lexique_results)

        if total == 0:
            st.warning(f"Aucun resultat pour **{query}**")
        else:
            st.success(f"**{total}** resultats trouves pour **{query}**")

            if guide_results:
                st.markdown(f"### Guide d'utilisation ({len(guide_results)} resultats)")
                for line_num, line_text in guide_results[:20]:
                    # Surligner le terme dans le texte
                    highlighted = line_text
                    idx = highlighted.lower().find(query_lower)
                    if idx >= 0:
                        original_match = highlighted[idx:idx + len(query)]
                        highlighted = highlighted[:idx] + f"**{original_match}**" + highlighted[idx + len(query):]
                    st.markdown(f"- L.{line_num}: {highlighted}")

                if len(guide_results) > 20:
                    st.caption(f"... et {len(guide_results) - 20} autres resultats")

            if lexique_results:
                st.markdown(f"### Lexique ({len(lexique_results)} resultats)")
                for line_num, line_text in lexique_results[:20]:
                    highlighted = line_text
                    idx = highlighted.lower().find(query_lower)
                    if idx >= 0:
                        original_match = highlighted[idx:idx + len(query)]
                        highlighted = highlighted[:idx] + f"**{original_match}**" + highlighted[idx + len(query):]
                    st.markdown(f"- L.{line_num}: {highlighted}")

                if len(lexique_results) > 20:
                    st.caption(f"... et {len(lexique_results) - 20} autres resultats")

    # Termes les plus recherches
    st.markdown("---")
    st.markdown("### Termes frequents")
    cols = st.columns(4)
    frequent_terms = [
        "VaR", "CVaR", "Sharpe", "Volatilite",
        "Black-Litterman", "Alpha portable", "Duration", "Monte Carlo",
        "Frontiere efficiente", "Parite de risque", "Drawdown", "Levier",
    ]
    for i, term in enumerate(frequent_terms):
        with cols[i % 4]:
            if st.button(term, key=f"freq_{term}"):
                st.session_state["doc_search_query"] = term
                st.rerun()

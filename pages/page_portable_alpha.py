"""
Page Streamlit - Alpha Portable
Strategie de separation beta/alpha avec levier pour les caisses de retraite.
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import (
    ASSET_CLASSES_ORDER, ASSET_DEFAULTS, BENCHMARK_PORTFOLIOS,
    ALPHA_ELIGIBLE_SHORT, LABELS_FR,
    get_asset_names_fr, get_expected_returns, get_covariance_matrix,
    get_esg_scores, PensionFundConfig,
)
from models.portable_alpha import PortableAlphaOptimizer, PortableAlphaResult
from constraints.regulatory import PortableAlphaRegulations, QuebecPensionRegulations
from visualization.charts import ChartBuilder


st.header("Alpha Portable")
st.markdown("""
Strategie de **separation beta/alpha** : le portefeuille beta replique passivement
un benchmark, tandis que l'overlay alpha genere du rendement excedentaire via des
positions long/short. Le levier est controle par les contraintes reglementaires.
""")

# ============================================================
# Parametres dans la sidebar
# ============================================================
with st.sidebar:
    st.markdown("### Parametres Alpha Portable")

    # Choix du benchmark beta
    benchmark_key = st.selectbox(
        "Benchmark beta",
        list(BENCHMARK_PORTFOLIOS.keys()),
        format_func=lambda k: BENCHMARK_PORTFOLIOS[k]["nom_fr"],
        index=0,
    )

    # Strategie d'optimisation
    strategy = st.selectbox(
        "Strategie d'optimisation",
        ["max_info_ratio", "max_alpha", "min_te", "risk_budgeted"],
        format_func=lambda s: {
            "max_info_ratio": "Maximiser ratio d'information",
            "max_alpha": "Maximiser alpha (budget TE)",
            "min_te": "Minimiser tracking error (alpha cible)",
            "risk_budgeted": "Budget de risque beta/alpha",
        }[s],
        index=0,
    )

    st.markdown("---")
    st.markdown("#### Contraintes de levier")

    max_gross_leverage = st.slider(
        "Levier brut maximal",
        1.0, PortableAlphaRegulations.MAX_GROSS_LEVERAGE,
        1.5, 0.05,
        help="sum(|w|) - Levier brut maximal autorise",
    )

    max_short_per_asset = st.slider(
        "Position courte max par actif (%)",
        1.0, PortableAlphaRegulations.MAX_SHORT_PER_ASSET * 100,
        10.0, 1.0,
    ) / 100

    financing_spread = st.slider(
        "Spread de financement (bps)",
        0, 100, 50, 5,
        help="Cout de financement des positions courtes en points de base",
    ) / 10000

    # Parametres conditionnels
    tracking_error_budget = None
    alpha_target = None

    if strategy == "max_alpha":
        tracking_error_budget = st.slider(
            "Budget de tracking error (%)",
            0.5, 10.0, 3.0, 0.5,
        ) / 100

    if strategy == "min_te":
        alpha_target = st.slider(
            "Alpha cible (%)",
            0.0, 5.0, 1.0, 0.25,
        ) / 100

    st.markdown("---")
    st.markdown("#### Contraintes ESG")
    use_esg = st.checkbox("Appliquer contrainte ESG", value=False)
    esg_min = st.slider("Score ESG minimum", 0, 100, 50, 5) if use_esg else None

# ============================================================
# Donnees et optimisation
# ============================================================
asset_names = get_asset_names_fr()
mu = get_expected_returns()
sigma = get_covariance_matrix()
benchmark_weights = BENCHMARK_PORTFOLIOS[benchmark_key]["weights"]
benchmark_name = BENCHMARK_PORTFOLIOS[benchmark_key]["nom_fr"]
esg_scores = get_esg_scores() if use_esg else None

# Config du fonds
config = st.session_state.get("pension_config", PensionFundConfig())
rf = config.taux_sans_risque

# Optimiseur
optimizer = PortableAlphaOptimizer(
    expected_returns=mu,
    cov_matrix=sigma,
    benchmark_weights=benchmark_weights,
    risk_free_rate=rf,
    asset_names=asset_names,
)

# Lancer l'optimisation
result = optimizer.optimize(
    strategy=strategy,
    max_gross_leverage=max_gross_leverage,
    max_short_per_asset=max_short_per_asset,
    short_eligible=ALPHA_ELIGIBLE_SHORT,
    tracking_error_budget=tracking_error_budget,
    alpha_target=alpha_target,
    financing_spread=financing_spread,
    esg_min_score=esg_min / 100 if esg_min else None,
    esg_scores=esg_scores,
    group_constraints=PortableAlphaRegulations.get_leverage_group_constraints(),
)

# ============================================================
# Metriques cles
# ============================================================
st.markdown("---")
st.subheader("Metriques cles")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Alpha brut", f"{result.alpha:.2%}",
              delta=f"{result.alpha - 0:.2%}" if result.alpha != 0 else None)
    st.metric("Tracking Error", f"{result.tracking_error:.2%}")

with col2:
    st.metric("Alpha net", f"{result.net_alpha:.2%}",
              delta=f"Cout: -{result.financing_cost:.2%}")
    st.metric(LABELS_FR["information_ratio"], f"{result.information_ratio:.3f}")

with col3:
    st.metric("Rendement combine", f"{result.expected_return:.2%}",
              delta=f"vs bench: {result.alpha:+.2%}")
    st.metric("Volatilite", f"{result.volatility:.2%}")

with col4:
    st.metric(LABELS_FR["gross_leverage"], f"{result.gross_leverage:.2f}x")
    st.metric(LABELS_FR["net_exposure"], f"{result.net_exposure:.2%}")

# Statut
if result.status == "optimal":
    st.success(f"Optimisation reussie en {result.solver_time:.3f}s | "
               f"Sharpe: {result.sharpe_ratio:.3f}")
else:
    st.warning(f"Statut: {result.status}")

# ============================================================
# Conformite reglementaire
# ============================================================
st.markdown("---")
st.subheader("Conformite reglementaire")

is_compliant, violations = PortableAlphaRegulations.validate_leverage_compliance(
    result.combined_weights, asset_names,
)

if is_compliant:
    st.success("Le portefeuille respecte toutes les contraintes reglementaires de levier.")
else:
    st.error("Violations reglementaires detectees :")
    for v in violations:
        st.markdown(f"- {v}")

# Detail des metriques de levier
with st.expander("Detail des metriques de levier", expanded=False):
    lev_cols = st.columns(3)
    with lev_cols[0]:
        st.markdown("**Expositions**")
        st.markdown(f"- Longue : {result.long_exposure:.2%}")
        st.markdown(f"- Courte : {result.short_exposure:.2%}")
        st.markdown(f"- Nette : {result.net_exposure:.2%}")
    with lev_cols[1]:
        st.markdown("**Positions**")
        st.markdown(f"- Longues : {result.n_long_positions}")
        st.markdown(f"- Courtes : {result.n_short_positions}")
        ratio_ls = result.long_exposure / result.short_exposure if result.short_exposure > 0.01 else float("inf")
        st.markdown(f"- Ratio L/S : {ratio_ls:.1f}x")
    with lev_cols[2]:
        st.markdown("**Couts de financement**")
        fin_detail = result.metadata.get("financing_detail", {})
        st.markdown(f"- Cout total : {result.financing_cost:.2%} ({result.financing_cost * 10000:.0f} bps)")
        st.markdown(f"- Cout short : {fin_detail.get('cout_short', 0):.4f}")
        st.markdown(f"- Cout marge : {fin_detail.get('cout_marge', 0):.4f}")

# ============================================================
# Graphique 1 : Decomposition Beta / Alpha
# ============================================================
st.markdown("---")
st.subheader("Decomposition du portefeuille")

fig_decomp = ChartBuilder.alpha_decomposition_bar(
    combined_weights=result.combined_weights,
    beta_weights=result.beta_weights,
    alpha_overlay=result.alpha_overlay,
    names=asset_names,
    title=f"Decomposition Beta ({benchmark_name}) / Alpha overlay",
)
st.plotly_chart(fig_decomp, use_container_width=True)

# ============================================================
# Graphique 2 : Overlay Heatmap
# ============================================================
fig_heatmap = ChartBuilder.overlay_heatmap(
    alpha_overlay=result.alpha_overlay,
    asset_names=asset_names,
    benchmark_name=benchmark_name,
)
st.plotly_chart(fig_heatmap, use_container_width=True)

# ============================================================
# Graphique 3 : Cascade levier et couts
# ============================================================
st.markdown("---")
st.subheader("Analyse du levier et des couts")

fig_waterfall = ChartBuilder.leverage_waterfall(
    long_exposure=result.long_exposure,
    short_exposure=result.short_exposure,
    financing_cost_bps=result.financing_cost * 10000,
    alpha_bps=result.alpha * 10000,
    net_alpha_bps=result.net_alpha * 10000,
)
st.plotly_chart(fig_waterfall, use_container_width=True)

# ============================================================
# Graphique 4 : Decomposition du risque
# ============================================================
st.markdown("---")
st.subheader("Decomposition du risque")

risk_decomp = optimizer.decompose_risk(result.combined_weights)

col_risk1, col_risk2 = st.columns([1, 1])

with col_risk1:
    # Metriques de risque
    st.markdown(f"**Risque total** : {risk_decomp['risque_total']:.2%}")
    st.markdown(f"**Risque beta** : {risk_decomp['risque_beta']:.2%}")
    st.markdown(f"**Risque alpha (TE)** : {risk_decomp['risque_alpha']:.2%}")
    st.markdown(f"**Correlation beta-alpha** : {risk_decomp['correlation_beta_alpha']:.3f}")

    # Pie chart
    interaction_pct = max(0, 1.0 - risk_decomp["contribution_beta_pct"] - risk_decomp["contribution_alpha_pct"])
    fig_risk_pie = ChartBuilder.risk_decomposition_pie(
        beta_risk_pct=risk_decomp["contribution_beta_pct"],
        alpha_risk_pct=risk_decomp["contribution_alpha_pct"],
        interaction_pct=interaction_pct,
    )
    st.plotly_chart(fig_risk_pie, use_container_width=True)

with col_risk2:
    # Contributions au risque par actif
    fig_risk_contrib = ChartBuilder.risk_contribution_bar(
        contributions=result.risk_contributions,
        names=asset_names,
        title="Contribution au risque par actif (portefeuille combine)",
    )
    st.plotly_chart(fig_risk_contrib, use_container_width=True)

# ============================================================
# Graphique 5 : Frontiere efficiente Alpha / TE
# ============================================================
st.markdown("---")
st.subheader("Frontiere efficiente Alpha / Tracking Error")

with st.spinner("Calcul de la frontiere alpha..."):
    frontier_results = optimizer.compute_efficient_alpha_frontier(
        n_points=15,
        max_gross_leverage=max_gross_leverage,
        short_eligible=ALPHA_ELIGIBLE_SHORT,
        max_short_per_asset=max_short_per_asset,
        financing_spread=financing_spread,
    )

if frontier_results:
    fig_frontier = ChartBuilder.alpha_frontier_chart(
        frontier_results=frontier_results,
        title="Frontiere efficiente : Alpha vs Tracking Error",
    )
    st.plotly_chart(fig_frontier, use_container_width=True)

    # Tableau recapitulatif de la frontiere
    with st.expander("Tableau de la frontiere alpha", expanded=False):
        frontier_df = pd.DataFrame([{
            "TE (%)": f"{r.tracking_error:.2%}",
            "Alpha brut (%)": f"{r.alpha:.2%}",
            "Alpha net (%)": f"{r.net_alpha:.2%}",
            "Ratio info": f"{r.information_ratio:.3f}",
            "Levier brut": f"{r.gross_leverage:.2f}x",
            "Nb short": r.n_short_positions,
            "Sharpe": f"{r.sharpe_ratio:.3f}",
        } for r in frontier_results])
        st.dataframe(frontier_df, use_container_width=True, hide_index=True)
else:
    st.warning("Aucun point de frontiere alpha n'a pu etre calcule.")

# ============================================================
# Tableau detaille des allocations
# ============================================================
st.markdown("---")
st.subheader("Detail des allocations")

allocation_df = pd.DataFrame({
    "Classe d'actifs": asset_names,
    "Benchmark": [f"{w:.2%}" for w in result.beta_weights],
    "Combine": [f"{w:.2%}" for w in result.combined_weights],
    "Overlay alpha": [f"{w:+.2%}" for w in result.alpha_overlay],
    "Contribution risque": [f"{r:.2%}" for r in result.risk_contributions],
})
st.dataframe(allocation_df, use_container_width=True, hide_index=True)

# ============================================================
# Comparaison avec le benchmark
# ============================================================
st.markdown("---")
st.subheader("Comparaison avec le benchmark")

bench_ret = result.metadata.get("benchmark_return", benchmark_weights @ mu)
bench_vol = result.metadata.get("benchmark_vol",
                                 np.sqrt(benchmark_weights @ sigma @ benchmark_weights))
bench_sharpe = (bench_ret - rf) / bench_vol if bench_vol > 1e-10 else 0.0

comp_data = {
    "Metrique": [
        "Rendement attendu", "Volatilite", "Ratio de Sharpe",
        "Alpha", "Tracking Error", "Ratio d'information",
        "Levier brut", "Cout de financement",
    ],
    "Benchmark": [
        f"{bench_ret:.2%}", f"{bench_vol:.2%}", f"{bench_sharpe:.3f}",
        "-", "-", "-", "1.00x", "-",
    ],
    "Portefeuille Alpha Portable": [
        f"{result.expected_return:.2%}", f"{result.volatility:.2%}",
        f"{result.sharpe_ratio:.3f}", f"{result.alpha:.2%}",
        f"{result.tracking_error:.2%}", f"{result.information_ratio:.3f}",
        f"{result.gross_leverage:.2f}x", f"{result.financing_cost:.2%}",
    ],
}
comp_df = pd.DataFrame(comp_data)
st.dataframe(comp_df, use_container_width=True, hide_index=True)

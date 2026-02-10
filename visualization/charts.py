"""
Bibliotheque de graphiques Plotly pour la visualisation du portefeuille.
Tous les labels en francais.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
from config import CHART_COLORS


class ChartBuilder:
    """Constructeurs de graphiques Plotly institutionnels."""

    @staticmethod
    def allocation_pie(
        weights: np.ndarray,
        names: List[str],
        title: str = "Allocation du portefeuille",
    ) -> go.Figure:
        """Diagramme en anneau de l'allocation."""
        # Filtrer les poids nuls
        mask = weights > 0.005
        fig = go.Figure(data=[go.Pie(
            labels=[names[i] for i in range(len(names)) if mask[i]],
            values=[weights[i] for i in range(len(weights)) if mask[i]],
            hole=0.45,
            marker_colors=[CHART_COLORS[i % len(CHART_COLORS)] for i in range(len(names)) if mask[i]],
            textinfo="label+percent",
            textposition="outside",
            hovertemplate="<b>%{label}</b><br>Poids: %{value:.1%}<br>Pourcentage: %{percent}<extra></extra>",
        )])
        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
            height=450,
            margin=dict(t=60, b=80, l=20, r=20),
        )
        return fig

    @staticmethod
    def allocation_comparison_bar(
        current: np.ndarray,
        optimized: np.ndarray,
        names: List[str],
        title: str = "Comparaison des allocations",
    ) -> go.Figure:
        """Diagramme a barres groupees: actuel vs optimise."""
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Allocation actuelle",
            x=names, y=current * 100,
            marker_color=CHART_COLORS[0],
            text=[f"{v:.1f}%" for v in current * 100],
            textposition="auto",
        ))
        fig.add_trace(go.Bar(
            name="Allocation optimisee",
            x=names, y=optimized * 100,
            marker_color=CHART_COLORS[1],
            text=[f"{v:.1f}%" for v in optimized * 100],
            textposition="auto",
        ))
        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            barmode="group",
            yaxis_title="Allocation (%)",
            xaxis_tickangle=-45,
            height=500,
            margin=dict(t=60, b=120),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        return fig

    @staticmethod
    def efficient_frontier(
        frontier_data: pd.DataFrame,
        current_portfolio: Optional[Tuple[float, float]] = None,
        optimized_portfolio: Optional[Tuple[float, float]] = None,
        tangency_portfolio: Optional[Tuple[float, float]] = None,
        cml_data: Optional[pd.DataFrame] = None,
        risk_measure: str = "volatilite",
    ) -> go.Figure:
        """Frontiere efficiente interactive."""
        fig = go.Figure()

        risk_col = "cvar" if risk_measure == "cvar" else "volatility"
        x_title = "CVaR (%)" if risk_measure == "cvar" else "Volatilite (%)"

        # Frontiere
        fig.add_trace(go.Scatter(
            x=frontier_data[risk_col] * 100,
            y=frontier_data["return"] * 100,
            mode="lines+markers",
            name="Frontiere efficiente",
            line=dict(color=CHART_COLORS[0], width=3),
            marker=dict(size=4),
            hovertemplate=(
                "Rendement: %{y:.2f}%<br>"
                f"{x_title[:-4]}: %{{x:.2f}}%<br>"
                "Sharpe: %{customdata:.3f}<extra></extra>"
            ),
            customdata=frontier_data["sharpe"] if "sharpe" in frontier_data else None,
        ))

        # CML
        if cml_data is not None:
            fig.add_trace(go.Scatter(
                x=cml_data["volatility"] * 100,
                y=cml_data["return"] * 100,
                mode="lines",
                name="Ligne du marche des capitaux",
                line=dict(color="gray", width=1, dash="dash"),
            ))

        # Portefeuille actuel
        if current_portfolio is not None:
            fig.add_trace(go.Scatter(
                x=[current_portfolio[1] * 100],
                y=[current_portfolio[0] * 100],
                mode="markers+text",
                name="Portefeuille actuel",
                marker=dict(size=15, color="red", symbol="diamond"),
                text=["Actuel"],
                textposition="top right",
            ))

        # Portefeuille optimise
        if optimized_portfolio is not None:
            fig.add_trace(go.Scatter(
                x=[optimized_portfolio[1] * 100],
                y=[optimized_portfolio[0] * 100],
                mode="markers+text",
                name="Portefeuille optimise",
                marker=dict(size=15, color="green", symbol="star"),
                text=["Optimise"],
                textposition="top right",
            ))

        # Portefeuille tangent
        if tangency_portfolio is not None:
            fig.add_trace(go.Scatter(
                x=[tangency_portfolio[1] * 100],
                y=[tangency_portfolio[0] * 100],
                mode="markers+text",
                name="Portefeuille tangent",
                marker=dict(size=15, color="gold", symbol="star-triangle-up"),
                text=["Tangent"],
                textposition="top right",
            ))

        fig.update_layout(
            title=dict(text="Frontiere efficiente", font=dict(size=16)),
            xaxis_title=x_title,
            yaxis_title="Rendement attendu (%)",
            height=550,
            margin=dict(t=60, b=60),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
            hovermode="closest",
        )
        return fig

    @staticmethod
    def monte_carlo_fan_chart(
        fan_data: Dict[str, np.ndarray],
        years: np.ndarray,
        title: str = "Projection Monte Carlo",
        y_label: str = "Valeur (M$)",
        scale: float = 1e6,
    ) -> go.Figure:
        """Graphique en eventail Monte Carlo avec bandes de percentiles."""
        fig = go.Figure()

        # Bande 5-95
        fig.add_trace(go.Scatter(
            x=np.concatenate([years, years[::-1]]),
            y=np.concatenate([fan_data["p95"] / scale, fan_data["p5"][::-1] / scale]),
            fill="toself",
            fillcolor="rgba(31, 119, 180, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Intervalle 5e-95e percentile",
            showlegend=True,
        ))

        # Bande 25-75
        fig.add_trace(go.Scatter(
            x=np.concatenate([years, years[::-1]]),
            y=np.concatenate([fan_data["p75"] / scale, fan_data["p25"][::-1] / scale]),
            fill="toself",
            fillcolor="rgba(31, 119, 180, 0.30)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Intervalle 25e-75e percentile",
            showlegend=True,
        ))

        # Mediane
        fig.add_trace(go.Scatter(
            x=years,
            y=fan_data["p50"] / scale,
            mode="lines",
            name="Mediane (50e percentile)",
            line=dict(color=CHART_COLORS[0], width=3),
        ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            xaxis_title="Annees",
            yaxis_title=y_label,
            height=500,
            margin=dict(t=60, b=60),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15),
            hovermode="x unified",
        )
        return fig

    @staticmethod
    def funded_ratio_projection(
        fan_data: Dict[str, np.ndarray],
        years: np.ndarray,
    ) -> go.Figure:
        """Projection du ratio de capitalisation avec zones colorees."""
        fig = go.Figure()

        # Zone rouge (sous 80%)
        fig.add_hrect(y0=0, y1=0.80, fillcolor="rgba(255,0,0,0.08)",
                      line_width=0, annotation_text="Zone critique",
                      annotation_position="bottom left")

        # Zone jaune (80-100%)
        fig.add_hrect(y0=0.80, y1=1.00, fillcolor="rgba(255,165,0,0.08)",
                      line_width=0)

        # Zone verte (>100%)
        fig.add_hrect(y0=1.00, y1=2.0, fillcolor="rgba(0,128,0,0.05)",
                      line_width=0)

        # Ligne de reference a 100%
        fig.add_hline(y=1.0, line_dash="dash", line_color="black",
                      annotation_text="Capitalisation integrale (100%)")

        # Bande 5-95
        fig.add_trace(go.Scatter(
            x=np.concatenate([years, years[::-1]]),
            y=np.concatenate([fan_data["p95"], fan_data["p5"][::-1]]),
            fill="toself",
            fillcolor="rgba(31, 119, 180, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="5e-95e percentile",
        ))

        # Bande 25-75
        fig.add_trace(go.Scatter(
            x=np.concatenate([years, years[::-1]]),
            y=np.concatenate([fan_data["p75"], fan_data["p25"][::-1]]),
            fill="toself",
            fillcolor="rgba(31, 119, 180, 0.30)",
            line=dict(color="rgba(255,255,255,0)"),
            name="25e-75e percentile",
        ))

        # Mediane
        fig.add_trace(go.Scatter(
            x=years, y=fan_data["p50"],
            mode="lines", name="Mediane",
            line=dict(color=CHART_COLORS[0], width=3),
        ))

        fig.update_layout(
            title=dict(text="Projection du ratio de capitalisation", font=dict(size=16)),
            xaxis_title="Annees",
            yaxis_title="Ratio de capitalisation",
            yaxis_tickformat=".0%",
            height=500,
            margin=dict(t=60, b=60),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15),
        )
        return fig

    @staticmethod
    def risk_contribution_bar(
        contributions: np.ndarray,
        names: List[str],
        title: str = "Contribution au risque par classe d'actifs",
    ) -> go.Figure:
        """Diagramme a barres horizontales des contributions au risque."""
        # Normaliser
        if np.sum(np.abs(contributions)) > 0:
            pct = contributions / np.sum(np.abs(contributions)) * 100
        else:
            pct = np.zeros_like(contributions)

        sorted_idx = np.argsort(pct)
        fig = go.Figure(go.Bar(
            x=pct[sorted_idx],
            y=[names[i] for i in sorted_idx],
            orientation="h",
            marker_color=[CHART_COLORS[i % len(CHART_COLORS)] for i in sorted_idx],
            text=[f"{v:.1f}%" for v in pct[sorted_idx]],
            textposition="auto",
        ))
        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            xaxis_title="Contribution au risque (%)",
            height=450,
            margin=dict(t=60, b=40, l=200),
        )
        return fig

    @staticmethod
    def correlation_heatmap(
        corr_matrix: np.ndarray,
        names: List[str],
        title: str = "Matrice de correlation",
    ) -> go.Figure:
        """Carte de chaleur interactive de la matrice de correlation."""
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=names,
            y=names,
            colorscale="RdBu_r",
            zmin=-1, zmax=1,
            text=np.round(corr_matrix, 2),
            texttemplate="%{text}",
            textfont={"size": 9},
            hovertemplate="%{y} vs %{x}<br>Correlation: %{z:.3f}<extra></extra>",
        ))
        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            height=550,
            margin=dict(t=60, b=120, l=200),
            xaxis_tickangle=-45,
        )
        return fig

    @staticmethod
    def drawdown_chart(
        returns: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
        title: str = "Courbe de perte maximale (Drawdown)",
    ) -> go.Figure:
        """Graphique de drawdown sous l'eau."""
        cum_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdowns = (cum_returns - running_max) / running_max * 100

        x = dates if dates is not None else np.arange(len(drawdowns))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=drawdowns,
            fill="tozeroy",
            fillcolor="rgba(214, 39, 40, 0.3)",
            line=dict(color="rgb(214, 39, 40)", width=1),
            name="Drawdown",
            hovertemplate="Drawdown: %{y:.2f}%<extra></extra>",
        ))
        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            xaxis_title="Periode",
            yaxis_title="Drawdown (%)",
            height=400,
            margin=dict(t=60, b=60),
        )
        return fig

    @staticmethod
    def stress_test_waterfall(
        scenario_results: pd.DataFrame,
        title: str = "Impact des tests de tension",
    ) -> go.Figure:
        """Diagramme en cascade de l'impact des tests de tension."""
        fig = go.Figure(go.Bar(
            x=scenario_results["Scenario"],
            y=scenario_results["Impact (%)"],
            marker_color=[
                "rgb(214, 39, 40)" if v < 0 else "rgb(44, 160, 44)"
                for v in scenario_results["Impact (%)"]
            ],
            text=[f"{v:+.1f}%" for v in scenario_results["Impact (%)"]],
            textposition="outside",
        ))
        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            yaxis_title="Impact sur le portefeuille (%)",
            xaxis_tickangle=-30,
            height=500,
            margin=dict(t=60, b=120),
        )
        fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
        return fig

    @staticmethod
    def return_distribution(
        returns: np.ndarray,
        var_level: Optional[float] = None,
        cvar_level: Optional[float] = None,
        title: str = "Distribution des rendements",
    ) -> go.Figure:
        """Histogramme avec lignes VaR et CVaR."""
        fig = go.Figure()

        fig.add_trace(go.Histogram(
            x=returns * 100,
            nbinsx=50,
            name="Rendements",
            marker_color=CHART_COLORS[0],
            opacity=0.7,
            histnorm="probability density",
        ))

        if var_level is not None:
            fig.add_vline(
                x=-var_level * 100, line_dash="dash", line_color="orange",
                annotation_text=f"VaR: {var_level:.1%}",
                annotation_position="top left",
            )

        if cvar_level is not None:
            fig.add_vline(
                x=-cvar_level * 100, line_dash="dash", line_color="red",
                annotation_text=f"CVaR: {cvar_level:.1%}",
                annotation_position="top left",
            )

        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            xaxis_title="Rendement (%)",
            yaxis_title="Densite",
            height=400,
            margin=dict(t=60, b=60),
        )
        return fig

    @staticmethod
    def glide_path_area(
        glide_path_data: List[Dict],
        title: str = "Trajectoire de desensibilisation",
    ) -> go.Figure:
        """Graphique en aires empilees pour le glide path."""
        df = pd.DataFrame(glide_path_data)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["Annee"], y=df["Actifs de croissance (%)"],
            stackgroup="one", name="Actifs de croissance",
            fillcolor="rgba(31, 119, 180, 0.7)",
            line=dict(color=CHART_COLORS[0]),
        ))
        fig.add_trace(go.Scatter(
            x=df["Annee"], y=df["Actifs de couverture (%)"],
            stackgroup="one", name="Actifs de couverture",
            fillcolor="rgba(44, 160, 44, 0.7)",
            line=dict(color=CHART_COLORS[2]),
        ))
        fig.add_trace(go.Scatter(
            x=df["Annee"], y=df["Encaisse (%)"],
            stackgroup="one", name="Encaisse",
            fillcolor="rgba(127, 127, 127, 0.7)",
            line=dict(color=CHART_COLORS[7]),
        ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            xaxis_title="Annees",
            yaxis_title="Allocation (%)",
            height=450,
            margin=dict(t=60, b=60),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15),
        )
        return fig

    @staticmethod
    def cumulative_returns_chart(
        returns: pd.DataFrame,
        title: str = "Performance cumulee",
    ) -> go.Figure:
        """Graphique de rendements cumules pour plusieurs classes d'actifs."""
        cum_returns = (1 + returns).cumprod()

        fig = go.Figure()
        for i, col in enumerate(cum_returns.columns[:6]):  # Max 6 pour lisibilite
            fig.add_trace(go.Scatter(
                x=cum_returns.index,
                y=cum_returns[col],
                mode="lines",
                name=col,
                line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2),
            ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            xaxis_title="Date",
            yaxis_title="Valeur (base 1.0)",
            height=450,
            margin=dict(t=60, b=60),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
            hovermode="x unified",
        )
        return fig

    # =============================================================
    # Graphiques Alpha Portable
    # =============================================================

    @staticmethod
    def alpha_decomposition_bar(
        combined_weights: np.ndarray,
        beta_weights: np.ndarray,
        alpha_overlay: np.ndarray,
        names: List[str],
        title: str = "Decomposition Beta / Alpha overlay",
    ) -> go.Figure:
        """Barres empilees montrant beta + alpha overlay = combine."""
        fig = go.Figure()

        # Beta (base)
        fig.add_trace(go.Bar(
            name="Portefeuille beta",
            x=names,
            y=beta_weights * 100,
            marker_color=CHART_COLORS[0],
            opacity=0.8,
            hovertemplate="%{x}<br>Beta: %{y:.2f}%<extra></extra>",
        ))

        # Alpha overlay (peut etre negatif)
        colors_alpha = [
            CHART_COLORS[2] if v >= 0 else CHART_COLORS[3]
            for v in alpha_overlay
        ]
        fig.add_trace(go.Bar(
            name="Overlay alpha",
            x=names,
            y=alpha_overlay * 100,
            marker_color=colors_alpha,
            opacity=0.7,
            hovertemplate="%{x}<br>Overlay: %{y:+.2f}%<extra></extra>",
        ))

        # Ligne du combine
        fig.add_trace(go.Scatter(
            name="Portefeuille combine",
            x=names,
            y=combined_weights * 100,
            mode="markers+lines",
            marker=dict(size=8, color="black", symbol="diamond"),
            line=dict(color="black", width=1.5, dash="dot"),
            hovertemplate="%{x}<br>Combine: %{y:.2f}%<extra></extra>",
        ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            barmode="relative",
            yaxis_title="Allocation (%)",
            xaxis_tickangle=-45,
            height=550,
            margin=dict(t=60, b=140),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        fig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=0.5)
        return fig

    @staticmethod
    def alpha_frontier_chart(
        frontier_results: List,
        title: str = "Frontiere efficiente alpha / tracking error",
    ) -> go.Figure:
        """Nuage alpha vs tracking error avec frontiere."""
        if not frontier_results:
            fig = go.Figure()
            fig.update_layout(title="Aucune donnee de frontiere alpha")
            return fig

        alphas = [r.alpha * 100 for r in frontier_results]
        tes = [r.tracking_error * 100 for r in frontier_results]
        irs = [r.information_ratio for r in frontier_results]
        leverages = [r.gross_leverage for r in frontier_results]
        net_alphas = [r.net_alpha * 100 for r in frontier_results]

        fig = go.Figure()

        # Frontiere alpha brut
        fig.add_trace(go.Scatter(
            x=tes, y=alphas,
            mode="lines+markers",
            name="Alpha brut",
            line=dict(color=CHART_COLORS[0], width=3),
            marker=dict(size=8, color=leverages,
                        colorscale="Viridis", showscale=True,
                        colorbar=dict(title="Levier brut", x=1.02)),
            hovertemplate=(
                "TE: %{x:.2f}%<br>"
                "Alpha brut: %{y:.2f}%<br>"
                "Levier: %{marker.color:.2f}<extra></extra>"
            ),
        ))

        # Frontiere alpha net
        fig.add_trace(go.Scatter(
            x=tes, y=net_alphas,
            mode="lines+markers",
            name="Alpha net (apres couts)",
            line=dict(color=CHART_COLORS[3], width=2, dash="dash"),
            marker=dict(size=5),
            hovertemplate=(
                "TE: %{x:.2f}%<br>"
                "Alpha net: %{y:.2f}%<extra></extra>"
            ),
        ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            xaxis_title="Tracking Error (%)",
            yaxis_title="Alpha (%)",
            height=550,
            margin=dict(t=60, b=60),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15),
            hovermode="closest",
        )
        fig.add_hline(y=0, line_dash="solid", line_color="gray", line_width=0.5)
        return fig

    @staticmethod
    def leverage_waterfall(
        long_exposure: float,
        short_exposure: float,
        financing_cost_bps: float,
        alpha_bps: float,
        net_alpha_bps: float,
        title: str = "Decomposition du levier et couts",
    ) -> go.Figure:
        """Diagramme en cascade : exposition -> couts -> alpha net."""
        categories = [
            "Exposition longue",
            "Exposition courte",
            "Levier brut",
            "Alpha brut",
            "Cout financement",
            "Alpha net",
        ]
        values = [
            long_exposure * 100,
            short_exposure * 100,
            0,  # Total
            alpha_bps,
            -financing_cost_bps,
            0,  # Total
        ]
        measures = ["relative", "relative", "total", "relative", "relative", "total"]

        fig = go.Figure(go.Waterfall(
            name="Decomposition",
            orientation="v",
            measure=measures,
            x=categories,
            y=values,
            textposition="outside",
            text=[
                f"{long_exposure:.0%}",
                f"{short_exposure:.0%}",
                f"{(long_exposure + short_exposure):.0%}",
                f"+{alpha_bps:.0f} bps",
                f"-{financing_cost_bps:.0f} bps",
                f"{net_alpha_bps:.0f} bps",
            ],
            connector=dict(line=dict(color="rgb(63, 63, 63)")),
            increasing=dict(marker=dict(color=CHART_COLORS[2])),
            decreasing=dict(marker=dict(color=CHART_COLORS[3])),
            totals=dict(marker=dict(color=CHART_COLORS[0])),
        ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            yaxis_title="Valeur",
            height=500,
            margin=dict(t=60, b=80),
            showlegend=False,
        )
        return fig

    @staticmethod
    def risk_decomposition_pie(
        beta_risk_pct: float,
        alpha_risk_pct: float,
        interaction_pct: float,
        title: str = "Decomposition du risque : Beta vs Alpha",
    ) -> go.Figure:
        """Diagramme en anneau : contribution beta vs alpha au risque total."""
        labels = ["Risque beta", "Risque alpha", "Interaction"]
        values = [
            max(0, beta_risk_pct * 100),
            max(0, alpha_risk_pct * 100),
            max(0, interaction_pct * 100),
        ]
        colors = [CHART_COLORS[0], CHART_COLORS[1], CHART_COLORS[7]]

        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.5,
            marker_colors=colors,
            textinfo="label+percent",
            textposition="outside",
            hovertemplate="<b>%{label}</b><br>%{value:.1f}%<extra></extra>",
        )])

        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            height=450,
            margin=dict(t=60, b=80, l=20, r=20),
            annotations=[dict(text="Risque", x=0.5, y=0.5, font_size=16, showarrow=False)],
        )
        return fig

    @staticmethod
    def overlay_heatmap(
        alpha_overlay: np.ndarray,
        asset_names: List[str],
        benchmark_name: str = "Benchmark",
        title: str = "Carte de chaleur de l'overlay alpha",
    ) -> go.Figure:
        """Heatmap montrant les surponderations/sous-ponderations vs benchmark."""
        # Reshape pour heatmap
        overlay_pct = alpha_overlay * 100
        z = overlay_pct.reshape(1, -1)

        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=asset_names,
            y=[f"vs {benchmark_name}"],
            colorscale="RdBu",
            zmid=0,
            text=[[f"{v:+.2f}%" for v in overlay_pct]],
            texttemplate="%{text}",
            textfont={"size": 11},
            hovertemplate="%{x}<br>Overlay: %{text}<extra></extra>",
            colorbar=dict(title="Overlay (%)", ticksuffix="%"),
        ))

        fig.update_layout(
            title=dict(text=title, font=dict(size=16)),
            height=200,
            margin=dict(t=60, b=40, l=20, r=20),
            xaxis_tickangle=-45,
        )
        return fig

    @staticmethod
    def esg_radar(
        scores: Dict[str, float],
        title: str = "Profil ESG du portefeuille",
    ) -> go.Figure:
        """Graphique radar pour le profil ESG."""
        categories = list(scores.keys())
        values = list(scores.values())
        values.append(values[0])  # Fermer le polygone
        categories.append(categories[0])

        fig = go.Figure(data=go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            fillcolor="rgba(44, 160, 44, 0.3)",
            line=dict(color=CHART_COLORS[2], width=2),
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title=dict(text=title, font=dict(size=16)),
            height=450,
        )
        return fig

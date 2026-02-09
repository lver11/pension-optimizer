"""
Generation de rapports Excel et CSV pour l'optimisation de portefeuille.
"""

import numpy as np
import pandas as pd
import io
from datetime import datetime
from typing import Dict, List, Optional
from config import PensionFundConfig


class ReportGenerator:
    """Generateur de rapports institutionnels."""

    def __init__(
        self,
        asset_names: List[str],
        config: PensionFundConfig = None,
    ):
        self.asset_names = asset_names
        self.config = config or PensionFundConfig()
        self.n_assets = len(asset_names)

    def generate_excel_report(
        self,
        current_weights: np.ndarray,
        risk_metrics: Dict[str, float],
        optimization_result=None,
        stress_results: Optional[pd.DataFrame] = None,
        esg_analysis: Optional[Dict] = None,
        compliance: Optional[Dict] = None,
        mc_stats: Optional[Dict] = None,
        report_type: str = "Rapport complet d'optimisation",
    ) -> bytes:
        """Genere un rapport Excel multi-feuilles."""
        buffer = io.BytesIO()

        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            workbook = writer.book

            # Formats
            header_fmt = workbook.add_format({
                "bold": True, "bg_color": "#1f4e79", "font_color": "white",
                "border": 1, "font_size": 12,
            })
            pct_fmt = workbook.add_format({"num_format": "0.00%", "border": 1})
            num_fmt = workbook.add_format({"num_format": "#,##0.00", "border": 1})
            money_fmt = workbook.add_format({"num_format": "#,##0", "border": 1})

            # --- Feuille Resume ---
            self._write_summary_sheet(
                writer, current_weights, risk_metrics,
                optimization_result, mc_stats,
            )

            # --- Feuille Allocation ---
            self._write_allocation_sheet(
                writer, current_weights, optimization_result,
            )

            # --- Feuille Metriques de risque ---
            self._write_risk_sheet(writer, risk_metrics)

            # --- Feuille Stress Tests ---
            if stress_results is not None:
                stress_results.to_excel(
                    writer, sheet_name="Tests de tension", index=False,
                )

            # --- Feuille ESG ---
            if esg_analysis is not None:
                self._write_esg_sheet(writer, esg_analysis)

            # --- Feuille Conformite ---
            if compliance is not None:
                self._write_compliance_sheet(writer, compliance)

            # --- Feuille Monte Carlo ---
            if mc_stats is not None:
                self._write_mc_sheet(writer, mc_stats)

        buffer.seek(0)
        return buffer.getvalue()

    def _write_summary_sheet(
        self, writer, weights, metrics, opt_result, mc_stats,
    ):
        """Ecrit la feuille de resume executif."""
        summary_data = {
            "Metrique": [
                "Date du rapport",
                "Nom du fonds",
                "Valeur de l'actif",
                "Valeur du passif",
                "Ratio de capitalisation",
                "Rendement attendu (annuel)",
                "Volatilite (annuelle)",
                "Ratio de Sharpe",
                "VaR (95%)",
                "CVaR (95%)",
                "Perte maximale",
            ],
            "Valeur": [
                datetime.now().strftime("%Y-%m-%d"),
                self.config.nom,
                f"{self.config.valeur_actif:,.0f} $",
                f"{self.config.valeur_passif:,.0f} $",
                f"{self.config.valeur_actif / self.config.valeur_passif:.1%}",
                f"{metrics.get('Rendement annualise', 0):.2%}",
                f"{metrics.get('Volatilite annualisee', 0):.2%}",
                f"{metrics.get('Ratio de Sharpe', 0):.3f}",
                f"{metrics.get('VaR (historique)', 0):.2%}",
                f"{metrics.get('CVaR', 0):.2%}",
                f"{metrics.get('Perte maximale', 0):.2%}",
            ],
        }

        if opt_result is not None:
            summary_data["Metrique"].extend([
                "", "--- OPTIMISATION ---",
                "Rendement optimise",
                "Volatilite optimisee",
                "Sharpe optimise",
                "Statut",
            ])
            summary_data["Valeur"].extend([
                "", "",
                f"{opt_result.expected_return:.2%}",
                f"{opt_result.volatility:.2%}",
                f"{opt_result.sharpe_ratio:.3f}",
                opt_result.status,
            ])

        if mc_stats is not None:
            summary_data["Metrique"].extend([
                "", "--- MONTE CARLO ---",
                "Ratio capit. median (terminal)",
                "Prob. sous-capitalisation",
                "Valeur mediane actif (terminal)",
            ])
            summary_data["Valeur"].extend([
                "", "",
                f"{mc_stats.get('median_fr', 0):.1%}",
                f"{mc_stats.get('prob_underfunded', 0):.1%}",
                f"{mc_stats.get('median_assets', 0):,.0f} $",
            ])

        df = pd.DataFrame(summary_data)
        df.to_excel(writer, sheet_name="Resume", index=False)

    def _write_allocation_sheet(self, writer, weights, opt_result):
        """Ecrit la feuille d'allocation."""
        data = {
            "Classe d'actifs": self.asset_names,
            "Allocation actuelle (%)": weights * 100,
        }

        if opt_result is not None:
            data["Allocation optimisee (%)"] = opt_result.weights * 100
            data["Ecart (pp)"] = (opt_result.weights - weights) * 100
            data["Contribution risque (%)"] = (
                opt_result.risk_contributions / opt_result.risk_contributions.sum() * 100
                if opt_result.risk_contributions.sum() > 0
                else np.zeros(self.n_assets)
            )

        df = pd.DataFrame(data)
        df.to_excel(writer, sheet_name="Allocation", index=False)

    def _write_risk_sheet(self, writer, metrics):
        """Ecrit la feuille des metriques de risque."""
        df = pd.DataFrame({
            "Metrique": list(metrics.keys()),
            "Valeur": [f"{v:.6f}" for v in metrics.values()],
        })
        df.to_excel(writer, sheet_name="Metriques de risque", index=False)

    def _write_esg_sheet(self, writer, esg_analysis):
        """Ecrit la feuille ESG."""
        data = {
            "Metrique": [
                "Score ESG du portefeuille",
                "Rating ESG",
                "Intensite carbone (tCO2e/M$)",
            ],
            "Valeur": [
                f"{esg_analysis.get('score_portfolio', 0):.1f}",
                esg_analysis.get("rating", "N/A"),
                f"{esg_analysis.get('intensite_carbone', 0):.0f}",
            ],
        }
        df = pd.DataFrame(data)
        df.to_excel(writer, sheet_name="ESG", index=False, startrow=0)

        # Detail par actif
        if "scores_par_actif" in esg_analysis:
            detail_df = pd.DataFrame({
                "Classe d'actifs": list(esg_analysis["scores_par_actif"].keys()),
                "Score ESG": list(esg_analysis["scores_par_actif"].values()),
            })
            detail_df.to_excel(writer, sheet_name="ESG", index=False, startrow=len(data["Metrique"]) + 3)

    def _write_compliance_sheet(self, writer, compliance):
        """Ecrit la feuille de conformite reglementaire."""
        data = {
            "Statut": ["Conforme" if compliance["conforme"] else "Non conforme"],
        }
        df = pd.DataFrame(data)
        df.to_excel(writer, sheet_name="Conformite", index=False)

        if compliance["violations"]:
            violations_df = pd.DataFrame({
                "Violation": compliance["violations"],
            })
            violations_df.to_excel(writer, sheet_name="Conformite", index=False, startrow=3)

    def _write_mc_sheet(self, writer, mc_stats):
        """Ecrit la feuille Monte Carlo."""
        data = {
            "Metrique": list(mc_stats.keys()),
            "Valeur": [
                f"{v:.4f}" if isinstance(v, float) else str(v)
                for v in mc_stats.values()
            ],
        }
        df = pd.DataFrame(data)
        df.to_excel(writer, sheet_name="Monte Carlo", index=False)

    def generate_csv_report(
        self,
        current_weights: np.ndarray,
        risk_metrics: Dict[str, float],
        optimization_result=None,
    ) -> bytes:
        """Genere un rapport CSV simplifie."""
        rows = []

        # Section allocation
        rows.append(["=== ALLOCATION ===", ""])
        for i, name in enumerate(self.asset_names):
            row = [name, f"{current_weights[i]:.4f}"]
            if optimization_result is not None:
                row.append(f"{optimization_result.weights[i]:.4f}")
            rows.append(row)

        rows.append(["", ""])
        rows.append(["=== METRIQUES DE RISQUE ===", ""])
        for key, val in risk_metrics.items():
            rows.append([key, f"{val:.6f}"])

        header = ["Metrique", "Actuel"]
        if optimization_result is not None:
            header.append("Optimise")

        df = pd.DataFrame(rows, columns=header + [""] * (len(rows[0]) - len(header))
                          if len(rows[0]) > len(header) else header[:len(rows[0])])

        return df.to_csv(index=False).encode("utf-8")

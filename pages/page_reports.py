"""
Generation et telechargement de rapports.
"""

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    get_asset_names_fr, get_expected_returns, get_covariance_matrix,
    DEFAULT_CURRENT_WEIGHTS, PensionFundConfig,
)
from ui_notes import show_assumption_notes
from fund_profile import ensure_session_state, get_active_profile
from config import get_min_weights, get_max_weights
from config import get_policy_weights
from data.generator import MarketDataGenerator
from risk.metrics import RiskMetrics
from risk.stress_testing import StressTester
from constraints.esg import ESGConstraintEngine
from reports.generator import ReportGenerator


def render():
    st.title("Generation de rapports")

    ensure_session_state()
    show_assumption_notes(simulated_returns_used=True)

    config = st.session_state.get("pension_config", PensionFundConfig())
    weights = st.session_state.get("current_weights", get_policy_weights())
    asset_names = get_asset_names_fr()
    returns_data = st.session_state.returns_data

    # Selection du type de rapport
    st.markdown("### Type de rapport")
    report_type = st.selectbox("Choisir le type de rapport", [
        "Rapport complet d'optimisation",
        "Resume executif",
        "Rapport de risque",
        "Rapport ESG",
    ])

    st.markdown("### Format de sortie")
    report_format = st.radio("Format", ["Excel (.xlsx)", "CSV"], horizontal=True)

    st.markdown("### Options du rapport")
    include_stress = st.checkbox("Inclure les tests de tension", True)
    include_esg = st.checkbox("Inclure l'analyse ESG", True)
    include_optimization = st.checkbox(
        "Inclure les resultats d'optimisation",
        "optimization_result" in st.session_state,
    )
    include_mc = st.checkbox(
        "Inclure les resultats Monte Carlo",
        "mc_result" in st.session_state,
    )

    st.divider()

    # Generation
    if st.button("Generer le rapport", type="primary", use_container_width=True):
        with st.spinner("Generation du rapport en cours..."):
            try:
                report_gen = ReportGenerator(asset_names, config)

                # Calculer les metriques
                portfolio_returns = returns_data.values @ weights
                risk_metrics = RiskMetrics.compute_all(
                    portfolio_returns, config.taux_sans_risque,
                )

                # Resultats d'optimisation
                opt_result = None
                if include_optimization and "optimization_result" in st.session_state:
                    opt_result = st.session_state.optimization_result

                # Stress tests
                stress_results = None
                if include_stress:
                    stress_tester = StressTester(asset_names)
                    stress_results = stress_tester.run_all_historical(
                        weights, config.valeur_actif,
                    )

                # ESG
                esg_analysis = None
                if include_esg:
                    esg_engine = ESGConstraintEngine(asset_names)
                    esg_analysis = esg_engine.esg_analysis(weights)

                # Conformite : contraintes du gestionnaire si sauvegardees,
                # sinon bornes et limites de groupe du profil de fonds actif
                from constraints.manager import ConstraintManager, ConstraintSet
                cs = st.session_state.get("constraint_set") or ConstraintSet(
                    min_weights=get_min_weights(),
                    max_weights=get_max_weights(),
                    group_constraints=get_active_profile().group_constraints(),
                )
                is_ok, violations = ConstraintManager(len(asset_names), asset_names).validate_allocation(
                    np.asarray(weights), cs,
                )
                compliance = {
                    "conforme": is_ok,
                    "violations": violations,
                }

                # Monte Carlo
                mc_stats = None
                if include_mc and "mc_result" in st.session_state:
                    mc_stats = st.session_state.mc_result.compute_statistics()

                # Generer le rapport
                if "Excel" in report_format:
                    report_bytes = report_gen.generate_excel_report(
                        current_weights=weights,
                        risk_metrics=risk_metrics,
                        optimization_result=opt_result,
                        stress_results=stress_results,
                        esg_analysis=esg_analysis,
                        compliance=compliance,
                        mc_stats=mc_stats,
                        report_type=report_type,
                    )
                    ext = "xlsx"
                    mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                else:
                    report_bytes = report_gen.generate_csv_report(
                        current_weights=weights,
                        risk_metrics=risk_metrics,
                        optimization_result=opt_result,
                    )
                    ext = "csv"
                    mime = "text/csv"

                # Telechargement
                filename = f"rapport_{report_type.replace(' ', '_').replace("'", '')}_{datetime.now():%Y%m%d_%H%M}.{ext}"

                st.download_button(
                    label=f"Telecharger le rapport ({ext.upper()})",
                    data=report_bytes,
                    file_name=filename,
                    mime=mime,
                    type="primary",
                    use_container_width=True,
                )

                st.success("Rapport genere avec succes!")

            except Exception as e:
                st.error(f"Erreur lors de la generation du rapport: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    st.divider()

    # ---------- Import de donnees ----------
    st.markdown("### Import de donnees")
    st.caption("Importez vos propres donnees de rendement ou parametres.")

    uploaded_file = st.file_uploader(
        "Fichier CSV ou Excel de rendements",
        type=["csv", "xlsx"],
    )

    if uploaded_file is not None:
        try:
            from data.importer import DataImporter
            importer = DataImporter(asset_names)

            if uploaded_file.name.endswith(".csv"):
                imported_data = importer.import_csv(uploaded_file)
            else:
                imported_data = importer.import_excel(uploaded_file)

            if imported_data is not None:
                st.success(f"Donnees importees: {imported_data.shape[0]} periodes, {imported_data.shape[1]} colonnes")
                st.dataframe(imported_data.head(10), use_container_width=True)

                if st.button("Utiliser ces donnees"):
                    st.session_state.returns_data = imported_data
                    st.session_state.returns_source = {"type": "importees", "fichier": uploaded_file.name}
                    st.success("Donnees de rendement mises a jour!")

        except Exception as e:
            st.error(f"Erreur d'import: {str(e)}")

    # Telecharger un template
    st.markdown("### Telecharger un template")
    if st.button("Generer un template de donnees"):
        from data.importer import DataImporter
        importer = DataImporter(asset_names)
        template_bytes = importer.generate_template()
        st.download_button(
            "Telecharger le template Excel",
            data=template_bytes,
            file_name="template_rendements.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )


render()

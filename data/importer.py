"""
Import et export de donnees (CSV, Excel).
"""

import numpy as np
import pandas as pd
import io
from typing import List, Optional


class DataImporter:
    """Gestion de l'import et de l'export des donnees."""

    def __init__(self, asset_names: List[str]):
        self.asset_names = asset_names
        self.n_assets = len(asset_names)

    def import_csv(self, file) -> Optional[pd.DataFrame]:
        """Importe des rendements depuis un fichier CSV."""
        df = pd.read_csv(file, index_col=0, parse_dates=True)
        return self._validate_and_clean(df)

    def import_excel(self, file) -> Optional[pd.DataFrame]:
        """Importe des rendements depuis un fichier Excel."""
        df = pd.read_excel(file, index_col=0, parse_dates=True)
        return self._validate_and_clean(df)

    def _validate_and_clean(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Valide et nettoie les donnees importees."""
        # Verifier que les donnees sont numeriques
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            raise ValueError("Aucune colonne numerique trouvee dans le fichier.")

        df = df[numeric_cols]

        # Gerer les NaN
        nan_pct = df.isnull().mean()
        high_nan_cols = nan_pct[nan_pct > 0.05].index.tolist()
        if high_nan_cols:
            print(f"Attention: colonnes avec >5% NaN: {high_nan_cols}")

        # Forward fill puis backward fill pour les NaN restants
        df = df.fillna(method="ffill").fillna(method="bfill").fillna(0)

        # Verifier que les rendements sont dans une plage raisonnable
        if (df.abs() > 1.0).any().any():
            # Possiblement en pourcentage, convertir
            if (df.abs() > 100).any().any():
                raise ValueError(
                    "Valeurs aberrantes detectees (>100). "
                    "Assurez-vous que les rendements sont en format decimal (0.05 pour 5%) "
                    "ou en pourcentage (5 pour 5%)."
                )
            # Convertir de pourcentage a decimal
            df = df / 100

        return df

    def generate_template(self) -> bytes:
        """Genere un fichier Excel template pour l'import de donnees."""
        dates = pd.date_range(start="2020-01-31", periods=60, freq="ME")
        template_data = pd.DataFrame(
            np.random.randn(60, self.n_assets) * 0.02 + 0.005,
            index=dates,
            columns=self.asset_names,
        )
        template_data.index.name = "Date"

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            template_data.to_excel(writer, sheet_name="Rendements")

            # Feuille d'instructions
            instructions = pd.DataFrame({
                "Instructions": [
                    "Remplissez la feuille 'Rendements' avec vos donnees de rendement.",
                    "L'index doit etre une colonne de dates (format YYYY-MM-DD).",
                    "Les rendements doivent etre en format decimal (0.05 pour 5%).",
                    "Les noms de colonnes doivent correspondre aux classes d'actifs.",
                    f"Classes d'actifs attendues: {', '.join(self.asset_names)}",
                    "Les valeurs manquantes seront interpolees automatiquement.",
                ]
            })
            instructions.to_excel(writer, sheet_name="Instructions", index=False)

        buffer.seek(0)
        return buffer.getvalue()

    @staticmethod
    def export_to_excel(data: dict, filename: str = "export.xlsx") -> bytes:
        """Exporte un dictionnaire de DataFrames en fichier Excel multi-feuilles."""
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            for sheet_name, df in data.items():
                if isinstance(df, pd.DataFrame):
                    df.to_excel(writer, sheet_name=sheet_name[:31])  # Max 31 chars
                elif isinstance(df, dict):
                    pd.DataFrame.from_dict(df, orient="index", columns=["Valeur"]).to_excel(
                        writer, sheet_name=sheet_name[:31],
                    )
        buffer.seek(0)
        return buffer.getvalue()

    @staticmethod
    def export_to_csv(df: pd.DataFrame) -> bytes:
        """Exporte un DataFrame en CSV."""
        return df.to_csv().encode("utf-8")

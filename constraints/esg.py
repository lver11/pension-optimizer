"""
Integration des contraintes ESG (Environnement, Social, Gouvernance).
"""

import numpy as np
from typing import Dict, List, Optional
from config import AssetClass, ASSET_CLASSES_ORDER, ASSET_DEFAULTS


class ESGConstraintEngine:
    """Moteur de scoring et contraintes ESG."""

    DEFAULT_ESG_SCORES = {
        "Actions canadiennes": 65.0,
        "Actions americaines": 60.0,
        "Actions EAFE": 70.0,
        "Actions emergentes": 45.0,
        "Obligations gouvernementales CDN": 80.0,
        "Obligations corporatives": 55.0,
        "Obligations indexees inflation": 80.0,
        "Immobilier": 60.0,
        "Infrastructure": 70.0,
        "Capital investissement": 50.0,
        "Matieres premieres": 35.0,
        "Encaisse": 75.0,
    }

    # Intensite carbone par classe d'actifs (tCO2e/M$ revenu, approximatif)
    DEFAULT_CARBON_INTENSITY = {
        "Actions canadiennes": 180.0,
        "Actions americaines": 150.0,
        "Actions EAFE": 120.0,
        "Actions emergentes": 250.0,
        "Obligations gouvernementales CDN": 50.0,
        "Obligations corporatives": 160.0,
        "Obligations indexees inflation": 50.0,
        "Immobilier": 100.0,
        "Infrastructure": 80.0,
        "Capital investissement": 200.0,
        "Matieres premieres": 350.0,
        "Encaisse": 10.0,
    }

    def __init__(self, asset_names: List[str]):
        self.asset_names = asset_names
        self.n_assets = len(asset_names)

    def get_esg_scores(
        self, custom_scores: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Retourne le vecteur des scores ESG."""
        scores = custom_scores or self.DEFAULT_ESG_SCORES
        return np.array([scores.get(name, 50.0) for name in self.asset_names])

    def compute_portfolio_esg_score(
        self,
        weights: np.ndarray,
        custom_scores: Optional[Dict[str, float]] = None,
    ) -> float:
        """Score ESG moyen pondere du portefeuille."""
        scores = self.get_esg_scores(custom_scores)
        return float(weights @ scores)

    def compute_carbon_intensity(
        self,
        weights: np.ndarray,
        custom_intensity: Optional[Dict[str, float]] = None,
    ) -> float:
        """Intensite carbone moyenne ponderee (tCO2e/M$ revenu)."""
        intensities = custom_intensity or self.DEFAULT_CARBON_INTENSITY
        intensity_array = np.array([
            intensities.get(name, 100.0) for name in self.asset_names
        ])
        return float(weights @ intensity_array)

    def build_esg_constraint_vector(
        self, custom_scores: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Retourne le vecteur de scores pour la contrainte ESG."""
        return self.get_esg_scores(custom_scores)

    def esg_analysis(
        self,
        weights: np.ndarray,
        custom_scores: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """Analyse ESG complete du portefeuille."""
        scores = self.get_esg_scores(custom_scores)
        portfolio_score = float(weights @ scores)
        carbon = self.compute_carbon_intensity(weights)

        # Contributions par classe d'actifs
        contributions = weights * scores
        carbon_contributions = weights * np.array([
            self.DEFAULT_CARBON_INTENSITY.get(name, 100.0)
            for name in self.asset_names
        ])

        # Classification
        if portfolio_score >= 70:
            rating = "A - Excellent"
            couleur = "green"
        elif portfolio_score >= 60:
            rating = "B - Bon"
            couleur = "blue"
        elif portfolio_score >= 50:
            rating = "C - Moyen"
            couleur = "yellow"
        else:
            rating = "D - A ameliorer"
            couleur = "red"

        return {
            "score_portfolio": portfolio_score,
            "rating": rating,
            "couleur": couleur,
            "intensite_carbone": carbon,
            "scores_par_actif": dict(zip(self.asset_names, scores.tolist())),
            "contributions_esg": dict(zip(self.asset_names, contributions.tolist())),
            "contributions_carbone": dict(zip(self.asset_names, carbon_contributions.tolist())),
        }

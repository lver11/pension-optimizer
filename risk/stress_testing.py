"""
Tests de tension historiques et parametriques pour le portefeuille.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from config import ASSET_CLASSES_ORDER, ASSET_DEFAULTS


# Scenarios historiques pre-definis (rendements par classe d'actifs)
HISTORICAL_SCENARIOS = {
    "crise_financiere_2008": {
        "nom_fr": "Crise financiere 2008",
        "description": "Crise des subprimes et effondrement de Lehman Brothers",
        "shocks": {
            "Actions canadiennes": -0.33,
            "Actions americaines": -0.38,
            "Actions EAFE": -0.43,
            "Actions emergentes": -0.53,
            "Obligations gouvernementales CDN": 0.08,
            "Obligations corporatives": -0.05,
            "Obligations indexees inflation": 0.02,
            "Immobilier": -0.25,
            "Infrastructure": -0.15,
            "Capital investissement": -0.40,
            "Matieres premieres": -0.46,
            "Encaisse": 0.02,
        },
    },
    "covid_2020": {
        "nom_fr": "Pandemie COVID-19 (mars 2020)",
        "description": "Choc pandemique et confinement mondial",
        "shocks": {
            "Actions canadiennes": -0.22,
            "Actions americaines": -0.20,
            "Actions EAFE": -0.23,
            "Actions emergentes": -0.24,
            "Obligations gouvernementales CDN": 0.05,
            "Obligations corporatives": -0.08,
            "Obligations indexees inflation": 0.01,
            "Immobilier": -0.15,
            "Infrastructure": -0.10,
            "Capital investissement": -0.20,
            "Matieres premieres": -0.30,
            "Encaisse": 0.01,
        },
    },
    "hausse_taux_2022": {
        "nom_fr": "Hausse des taux 2022",
        "description": "Resserrement monetaire agressif par les banques centrales",
        "shocks": {
            "Actions canadiennes": -0.06,
            "Actions americaines": -0.19,
            "Actions EAFE": -0.14,
            "Actions emergentes": -0.20,
            "Obligations gouvernementales CDN": -0.12,
            "Obligations corporatives": -0.15,
            "Obligations indexees inflation": -0.08,
            "Immobilier": -0.20,
            "Infrastructure": -0.05,
            "Capital investissement": -0.15,
            "Matieres premieres": 0.16,
            "Encaisse": 0.01,
        },
    },
    "crise_tech_2000": {
        "nom_fr": "Eclatement bulle techno 2000-2002",
        "description": "Eclatement de la bulle Internet",
        "shocks": {
            "Actions canadiennes": -0.14,
            "Actions americaines": -0.38,
            "Actions EAFE": -0.30,
            "Actions emergentes": -0.25,
            "Obligations gouvernementales CDN": 0.10,
            "Obligations corporatives": 0.03,
            "Obligations indexees inflation": 0.08,
            "Immobilier": 0.05,
            "Infrastructure": 0.02,
            "Capital investissement": -0.35,
            "Matieres premieres": -0.15,
            "Encaisse": 0.03,
        },
    },
    "crise_dette_euro_2011": {
        "nom_fr": "Crise dette europeenne 2011",
        "description": "Crise de la dette souveraine en Europe",
        "shocks": {
            "Actions canadiennes": -0.11,
            "Actions americaines": 0.02,
            "Actions EAFE": -0.15,
            "Actions emergentes": -0.20,
            "Obligations gouvernementales CDN": 0.10,
            "Obligations corporatives": 0.02,
            "Obligations indexees inflation": 0.12,
            "Immobilier": -0.05,
            "Infrastructure": -0.03,
            "Capital investissement": -0.10,
            "Matieres premieres": -0.10,
            "Encaisse": 0.01,
        },
    },
    "stagflation": {
        "nom_fr": "Scenario de stagflation",
        "description": "Inflation elevee combinee avec stagnation economique",
        "shocks": {
            "Actions canadiennes": -0.15,
            "Actions americaines": -0.20,
            "Actions EAFE": -0.18,
            "Actions emergentes": -0.25,
            "Obligations gouvernementales CDN": -0.08,
            "Obligations corporatives": -0.12,
            "Obligations indexees inflation": 0.05,
            "Immobilier": -0.05,
            "Infrastructure": 0.02,
            "Capital investissement": -0.20,
            "Matieres premieres": 0.20,
            "Encaisse": 0.03,
        },
    },
}


class StressTester:
    """Moteur de tests de tension historiques et parametriques."""

    def __init__(self, asset_names: List[str]):
        self.asset_names = asset_names
        self.n_assets = len(asset_names)

    def run_historical_stress(
        self,
        weights: np.ndarray,
        scenario_key: str,
        portfolio_value: float = 1_000_000_000.0,
    ) -> Dict:
        """Applique un scenario historique au portefeuille."""
        scenario = HISTORICAL_SCENARIOS.get(scenario_key)
        if scenario is None:
            raise ValueError(f"Scenario inconnu: {scenario_key}")

        shocks = np.array([
            scenario["shocks"].get(name, 0.0) for name in self.asset_names
        ])

        # Impact du portefeuille
        portfolio_impact = weights @ shocks
        asset_contributions = weights * shocks

        return {
            "scenario": scenario["nom_fr"],
            "description": scenario["description"],
            "impact_portefeuille": float(portfolio_impact),
            "perte_absolue": float(portfolio_impact * portfolio_value),
            "contributions_par_actif": dict(zip(self.asset_names, asset_contributions.tolist())),
            "chocs_par_actif": dict(zip(self.asset_names, shocks.tolist())),
        }

    def run_all_historical(
        self,
        weights: np.ndarray,
        portfolio_value: float = 1_000_000_000.0,
    ) -> pd.DataFrame:
        """Execute tous les scenarios historiques."""
        results = []
        for key in HISTORICAL_SCENARIOS:
            result = self.run_historical_stress(weights, key, portfolio_value)
            results.append({
                "Scenario": result["scenario"],
                "Impact (%)": result["impact_portefeuille"] * 100,
                "Perte (M$)": result["perte_absolue"] / 1e6,
            })
        return pd.DataFrame(results).sort_values("Impact (%)")

    def run_parametric_stress(
        self,
        weights: np.ndarray,
        equity_shock: float = -0.20,
        rate_shock_bps: float = 100,
        spread_shock_bps: float = 50,
        inflation_shock: float = 0.02,
        portfolio_value: float = 1_000_000_000.0,
    ) -> Dict:
        """
        Test de tension parametrique defini par l'utilisateur.
        Utilise les sensibilites des classes d'actifs.
        """
        # Sensibilites approximatives
        equity_beta = np.array([1.0, 1.0, 1.0, 1.2, 0.0, 0.2, 0.0, 0.4, 0.3, 0.8, 0.3, 0.0])
        duration = np.array([0, 0, 0, 0, 7.5, 5.5, 10.0, 0, 0, 0, 0, 0.25])
        spread_duration = np.array([0, 0, 0, 0, 0, 5.0, 0, 0, 0, 0, 0, 0])
        inflation_sensitivity = np.array([0, 0, 0, 0, -0.3, -0.2, 0.8, 0.3, 0.2, 0, 0.5, 0])

        # Calculer les chocs par classe d'actifs
        shocks = (
            equity_beta * equity_shock
            - duration * rate_shock_bps / 10000
            - spread_duration * spread_shock_bps / 10000
            + inflation_sensitivity * inflation_shock
        )

        portfolio_impact = weights @ shocks
        asset_contributions = weights * shocks

        return {
            "scenario": "Parametrique",
            "parametres": {
                "choc_actions": equity_shock,
                "choc_taux_bps": rate_shock_bps,
                "choc_spreads_bps": spread_shock_bps,
                "choc_inflation": inflation_shock,
            },
            "impact_portefeuille": float(portfolio_impact),
            "perte_absolue": float(portfolio_impact * portfolio_value),
            "contributions_par_actif": dict(zip(self.asset_names, asset_contributions.tolist())),
            "chocs_par_actif": dict(zip(self.asset_names, shocks.tolist())),
        }

    def reverse_stress_test(
        self,
        weights: np.ndarray,
        loss_threshold: float = -0.20,
        cov_matrix: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Test de tension inverse: trouve les scenarios qui causeraient
        une perte >= seuil. Minimise la norme du choc.
        """
        from scipy.optimize import minimize

        if cov_matrix is None:
            cov_matrix = np.eye(self.n_assets) * 0.04

        def objective(shocks):
            return np.sum(shocks ** 2)

        def loss_constraint(shocks):
            return -(weights @ shocks) - abs(loss_threshold)

        # Bornes realistes des chocs
        bounds = [(-0.60, 0.20)] * self.n_assets

        constraints = [{"type": "ineq", "fun": loss_constraint}]

        x0 = np.ones(self.n_assets) * loss_threshold / self.n_assets

        result = minimize(
            objective, x0, method="SLSQP",
            bounds=bounds, constraints=constraints,
        )

        if result.success:
            shocks = result.x
            portfolio_impact = weights @ shocks
            return {
                "seuil_perte": loss_threshold,
                "chocs_minimaux": dict(zip(self.asset_names, shocks.tolist())),
                "impact_portefeuille": float(portfolio_impact),
                "norme_choc": float(np.sqrt(np.sum(shocks ** 2))),
            }
        return {
            "seuil_perte": loss_threshold,
            "erreur": "Optimisation echouee",
        }

    def stress_funded_ratio(
        self,
        weights: np.ndarray,
        current_assets: float,
        current_liabilities: float,
        scenarios: Optional[Dict] = None,
    ) -> pd.DataFrame:
        """Test de tension sur le ratio de capitalisation."""
        if scenarios is None:
            scenarios = HISTORICAL_SCENARIOS

        results = []
        current_fr = current_assets / current_liabilities

        for key, scenario in scenarios.items():
            shocks = np.array([
                scenario["shocks"].get(name, 0.0) for name in self.asset_names
            ])
            portfolio_impact = weights @ shocks
            new_assets = current_assets * (1 + portfolio_impact)
            # Simplification: passif change avec les taux
            rate_impact = shocks[4] * 0.5  # Proxy basee sur obligations gov
            new_liabilities = current_liabilities * (1 - rate_impact * 0.7)
            new_fr = new_assets / new_liabilities

            results.append({
                "Scenario": scenario["nom_fr"],
                "Impact actif (%)": portfolio_impact * 100,
                "Nouvel actif (M$)": new_assets / 1e6,
                "Nouveau passif (M$)": new_liabilities / 1e6,
                "Ratio capit. actuel": current_fr,
                "Ratio capit. stresse": new_fr,
                "Variation ratio": (new_fr - current_fr) * 100,
            })

        return pd.DataFrame(results).sort_values("Ratio capit. stresse")

"""
Simulation Monte Carlo pour les projections de fonds de pension a long horizon.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class MonteCarloResult:
    """Resultat d'une simulation Monte Carlo."""
    asset_paths: np.ndarray         # (n_sim, horizon)
    liability_paths: np.ndarray     # (n_sim, horizon)
    funded_ratio_paths: np.ndarray  # (n_sim, horizon)
    return_paths: np.ndarray        # (n_sim, horizon)
    years: np.ndarray               # (horizon,)
    n_simulations: int = 0
    horizon_years: int = 0

    def percentile(self, data: np.ndarray, p: float) -> np.ndarray:
        """Calcule le percentile p le long de l'axe des simulations."""
        return np.percentile(data, p, axis=0)

    def get_fan_data(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Retourne les percentiles pour le graphique en eventail."""
        return {
            "p5": self.percentile(data, 5),
            "p25": self.percentile(data, 25),
            "p50": self.percentile(data, 50),
            "p75": self.percentile(data, 75),
            "p95": self.percentile(data, 95),
        }

    def compute_statistics(self) -> Dict[str, float]:
        """Calcule les statistiques sommaires."""
        terminal_fr = self.funded_ratio_paths[:, -1]
        terminal_assets = self.asset_paths[:, -1]

        return {
            "median_fr": float(np.median(terminal_fr)),
            "mean_fr": float(np.mean(terminal_fr)),
            "p5_fr": float(np.percentile(terminal_fr, 5)),
            "p95_fr": float(np.percentile(terminal_fr, 95)),
            "prob_underfunded": float(np.mean(terminal_fr < 1.0)),
            "prob_severely_underfunded": float(np.mean(terminal_fr < 0.80)),
            "median_assets": float(np.median(terminal_assets)),
            "p5_assets": float(np.percentile(terminal_assets, 5)),
            "p95_assets": float(np.percentile(terminal_assets, 95)),
            "surplus_var_5": float(np.percentile(
                self.asset_paths[:, -1] - self.liability_paths[:, -1], 5
            )),
            "prob_ruin": float(np.mean(terminal_assets <= 0)),
        }


class MonteCarloSimulator:
    """Simulateur Monte Carlo pour les projections pension a long terme."""

    def __init__(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        initial_assets: float = 1_000_000_000.0,
        initial_liabilities: float = 950_000_000.0,
        annual_contribution: float = 40_000_000.0,
        annual_benefit: float = 57_000_000.0,
        liability_growth_rate: float = 0.05,
        benefit_growth_rate: float = 0.03,
        n_simulations: int = 10_000,
        seed: int = 42,
    ):
        self.weights = weights
        self.mu = expected_returns
        self.sigma = cov_matrix
        self.initial_assets = initial_assets
        self.initial_liabilities = initial_liabilities
        self.annual_contribution = annual_contribution
        self.annual_benefit = annual_benefit
        self.liability_growth_rate = liability_growth_rate
        self.benefit_growth_rate = benefit_growth_rate
        self.n_sims = n_simulations
        self.rng = np.random.default_rng(seed)

        # Rendement et volatilite du portefeuille
        self.port_mu = weights @ expected_returns
        self.port_vol = np.sqrt(weights @ cov_matrix @ weights)

    def simulate(self, horizon_years: int = 20) -> MonteCarloResult:
        """
        Execute la simulation Monte Carlo.

        Pour chaque simulation s = 1,...,n_sims:
          Pour chaque annee t = 1,...,horizon:
            1. r_t ~ N(mu_p - sigma_p^2/2, sigma_p^2) (log-normal)
            2. A_t = A_{t-1} * exp(r_t) + C_t - B_t
            3. L_t = L_{t-1} * (1 + g_L) - B_t
            4. FR_t = A_t / L_t
        """
        asset_paths = np.zeros((self.n_sims, horizon_years))
        liability_paths = np.zeros((self.n_sims, horizon_years))
        funded_ratio_paths = np.zeros((self.n_sims, horizon_years))
        return_paths = np.zeros((self.n_sims, horizon_years))

        # Log-rendements annuels
        log_mu = self.port_mu - 0.5 * self.port_vol ** 2
        annual_returns = self.rng.normal(
            log_mu, self.port_vol, size=(self.n_sims, horizon_years)
        )

        for t in range(horizon_years):
            if t == 0:
                prev_assets = self.initial_assets
                prev_liabilities = self.initial_liabilities
            else:
                prev_assets = asset_paths[:, t - 1]
                prev_liabilities = liability_paths[:, t - 1]

            # Rendement du portefeuille
            port_returns = np.exp(annual_returns[:, t]) - 1
            return_paths[:, t] = port_returns

            # Contributions et prestations (croissantes)
            contrib_t = self.annual_contribution * (1 + 0.02) ** t
            benefit_t = self.annual_benefit * (1 + self.benefit_growth_rate) ** t

            # Evolution de l'actif
            asset_paths[:, t] = prev_assets * (1 + port_returns) + contrib_t - benefit_t
            asset_paths[:, t] = np.maximum(asset_paths[:, t], 0)  # Plancher a zero

            # Evolution du passif
            liability_paths[:, t] = prev_liabilities * (1 + self.liability_growth_rate) - benefit_t
            liability_paths[:, t] = np.maximum(liability_paths[:, t], 1e6)

            # Ratio de capitalisation
            funded_ratio_paths[:, t] = asset_paths[:, t] / liability_paths[:, t]

        years = np.arange(1, horizon_years + 1)

        return MonteCarloResult(
            asset_paths=asset_paths,
            liability_paths=liability_paths,
            funded_ratio_paths=funded_ratio_paths,
            return_paths=return_paths,
            years=years,
            n_simulations=self.n_sims,
            horizon_years=horizon_years,
        )

    def simulate_multiple_allocations(
        self,
        allocations: Dict[str, np.ndarray],
        horizon_years: int = 20,
    ) -> Dict[str, MonteCarloResult]:
        """Simule plusieurs allocations pour comparaison."""
        results = {}
        for name, weights in allocations.items():
            sim = MonteCarloSimulator(
                weights=weights,
                expected_returns=self.mu,
                cov_matrix=self.sigma,
                initial_assets=self.initial_assets,
                initial_liabilities=self.initial_liabilities,
                annual_contribution=self.annual_contribution,
                annual_benefit=self.annual_benefit,
                liability_growth_rate=self.liability_growth_rate,
                benefit_growth_rate=self.benefit_growth_rate,
                n_simulations=self.n_sims,
                seed=self.rng.integers(0, 100000),
            )
            results[name] = sim.simulate(horizon_years)
        return results

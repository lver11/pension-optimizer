"""
Gestion Actif-Passif (ALM) et Liability-Driven Investing (LDI)
pour les fonds de pension.
"""

import numpy as np
import cvxpy as cp
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from models.base import OptimizationResult
from models.liability import (
    LiabilityCashflows, YieldCurve, KEY_TENORS, BP,
    approx_convexity, bond_key_rate_profile,
)


@dataclass
class LiabilityProfile:
    """Profil du passif du fonds de pension."""
    present_value: float = 950_000_000.0
    duration: float = 15.0
    convexity: float = 250.0
    discount_rate: float = 0.05
    inflation_sensitivity: float = 0.30
    growth_rate: float = 0.03


class ALMOptimizer:
    """Optimiseur Actif-Passif pour les fonds de pension."""

    def __init__(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        asset_durations: np.ndarray,
        liability_profile: LiabilityProfile,
        risk_free_rate: float = 0.025,
        asset_names: Optional[List[str]] = None,
        min_weights: Optional[np.ndarray] = None,
        max_weights: Optional[np.ndarray] = None,
        reference_bond_index: Optional[int] = None,
        liability_cashflows: Optional[LiabilityCashflows] = None,
        curve: Optional[YieldCurve] = None,
    ):
        self.mu = expected_returns
        self.sigma = cov_matrix
        self.n_assets = len(expected_returns)
        self.durations = asset_durations
        self.liability = liability_profile
        self.rf = risk_free_rate
        self.asset_names = asset_names or [f"Actif_{i}" for i in range(self.n_assets)]
        self.min_weights = min_weights if min_weights is not None else np.zeros(self.n_assets)
        self.max_weights = max_weights if max_weights is not None else np.ones(self.n_assets)
        self.reference_bond_index = reference_bond_index
        self.cashflows = liability_cashflows
        self.curve = curve or YieldCurve.flat(liability_profile.discount_rate)
        # Convexite approchee des classes d'actifs a revenu fixe (zero-coupon de meme duration)
        self.asset_convexities = np.array([
            approx_convexity(d, self.curve.rate(max(d, 1.0))) if d > 0 else 0.0
            for d in self.durations
        ])

    def compute_funded_ratio(self, asset_value: float) -> float:
        """FR = Actifs / VP(Passifs)"""
        return asset_value / self.liability.present_value

    def compute_surplus(self, asset_value: float) -> float:
        """S = Actifs - VP(Passifs)"""
        return asset_value - self.liability.present_value

    def compute_duration_gap(
        self, weights: np.ndarray, asset_value: float
    ) -> float:
        """
        Ecart de duration = D_actifs - (L/A) * D_passifs.
        D_actifs = sum(w_i * D_i) pour les actifs a revenu fixe.
        """
        asset_duration = weights @ self.durations
        leverage = self.liability.present_value / asset_value
        return asset_duration - leverage * self.liability.duration

    def compute_interest_rate_sensitivity(
        self, weights: np.ndarray, asset_value: float, rate_change_bps: float = 100,
    ) -> Dict:
        """
        Sensibilite du surplus a un choc parallele des taux.

        Passif : reevaluation exacte sur la courbe choquee si les flux sont fournis,
        sinon duration + convexite. Actif : duration + convexite approchee des
        classes a revenu fixe.
        """
        dy = rate_change_bps / 10000
        L = self.liability.present_value

        asset_duration = weights @ self.durations
        asset_convexity = weights @ self.asset_convexities
        asset_change = asset_value * (-asset_duration * dy + 0.5 * asset_convexity * dy * dy)

        liability_linear = -self.liability.duration * dy * L
        if self.cashflows is not None:
            pv0 = self.cashflows.present_value(self.curve)
            liability_change = (self.cashflows.present_value(self.curve, dy) - pv0) * (L / pv0)
            method = "reevaluation des flux"
        else:
            liability_change = liability_linear + 0.5 * self.liability.convexity * dy * dy * L
            method = "duration + convexite"

        surplus_change = asset_change - liability_change
        new_ratio = (asset_value + asset_change) / (L + liability_change)

        return {
            "variation_taux_bps": rate_change_bps,
            "impact_actif": float(asset_change),
            "impact_passif": float(liability_change),
            "impact_passif_duration_seule": float(liability_linear),
            "effet_convexite_passif": float(liability_change - liability_linear),
            "impact_surplus": float(surplus_change),
            "impact_ratio_capit": float(new_ratio - asset_value / L),
            "methode_passif": method,
        }

    def key_rate_exposures(self, weights: np.ndarray, asset_value: float) -> Dict[float, Dict[str, float]]:
        """
        Exposition par segment de courbe, en $ par point de base (DV01 par point cle).
        Passif : durations cles exactes si les flux sont fournis, sinon duration
        repartie comme un zero-coupon. Actif : chaque classe obligataire est repartie
        sur les deux points cles qui encadrent sa duration.
        """
        L = self.liability.present_value
        if self.cashflows is not None:
            liab_krd = self.cashflows.key_rate_durations(self.curve)
        else:
            liab_krd = bond_key_rate_profile(self.liability.duration)
        asset_krd = {t: 0.0 for t in KEY_TENORS}
        for w_i, d_i in zip(weights, self.durations):
            if w_i > 0 and d_i > 0:
                for t, v in bond_key_rate_profile(d_i).items():
                    asset_krd[t] += w_i * v
        return {
            t: {
                "actif_dv01": asset_value * asset_krd[t] * BP,
                "passif_dv01": L * liab_krd[t] * BP,
            }
            for t in KEY_TENORS
        }

    def compute_hedge_ratio(self, weights: np.ndarray, asset_value: float) -> float:
        """
        Ratio de couverture du risque de taux (en duration-dollar) :
        HR = (A x somme(w_i x D_i)) / (L x D_L).
        100 % = une variation de taux change l'actif et le passif du meme montant en $.
        """
        liability_dd = self.liability.present_value * self.liability.duration
        if liability_dd <= 0:
            return 0.0
        return float(asset_value * (weights @ self.durations) / liability_dd)

    def assess_status(self, weights: np.ndarray, asset_value: float) -> Dict:
        """
        Diagnostic actif-passif combinant le niveau de capitalisation et le risque de taux.

        Statut final = le plus severe des deux composantes (voir ALM_STATUS_RULES).
        """
        fr = self.compute_funded_ratio(asset_value)
        gap = self.compute_duration_gap(weights, asset_value)
        hr = self.compute_hedge_ratio(weights, asset_value)
        sens = self.compute_interest_rate_sensitivity(weights, asset_value, -100)
        # Variation du ratio de capitalisation (en points) pour une baisse de 100 pb
        delta_fr_pts = (
            (asset_value + sens["impact_actif"]) / (self.liability.present_value + sens["impact_passif"])
            - fr
        ) * 100

        # Composante 1 : niveau de capitalisation
        if fr < 0.80:
            fund = ("critique", 4, f"Ratio de capitalisation de {fr:.0%} : deficit important.")
        elif fr < 0.90:
            fund = ("insuffisant", 3, f"Ratio de capitalisation de {fr:.0%} : deficit a resorber.")
        elif fr < 1.00:
            fund = ("a surveiller", 2, f"Ratio de capitalisation de {fr:.0%}, sous 100 %.")
        elif fr < 1.10:
            fund = ("adequat", 1, f"Ratio de capitalisation de {fr:.0%}, sans coussin important.")
        else:
            fund = ("excedentaire", 0, f"Ratio de capitalisation de {fr:.0%} : surplus disponible.")

        # Composante 2 : risque de taux (perte de capitalisation pour -100 pb)
        loss = -delta_fr_pts
        if loss > 8:
            rate = ("eleve", 3, f"Une baisse des taux de 100 pb ferait perdre environ "
                                f"{loss:.1f} points de capitalisation.")
        elif loss > 3:
            rate = ("modere", 2, f"Une baisse des taux de 100 pb ferait perdre environ "
                                 f"{loss:.1f} points de capitalisation.")
        else:
            rate = ("faible", 0, f"Une baisse des taux de 100 pb changerait la capitalisation "
                                 f"de {delta_fr_pts:+.1f} points.")

        severity = max(fund[1], rate[1])
        statut = {4: "critique", 3: "insuffisant" if fund[1] >= rate[1] else "risque de taux eleve",
                  2: "a surveiller", 1: "adequat", 0: fund[0]}[severity]
        couleur = {4: "red", 3: "orange", 2: "yellow", 1: "green", 0: "blue" if fund[1] == 0 else "green"}[severity]

        actions = []
        if fund[1] >= 3:
            actions.append("Etablir un plan pour resorber le deficit (cotisations, rendement, "
                           "politique de placement).")
        elif fund[1] == 2:
            actions.append("Suivre l'evolution du ratio de capitalisation de pres.")
        if rate[1] >= 2:
            actions.append(
                f"Le ratio de couverture du risque de taux est de {hr:.0%} "
                f"(ecart de duration {gap:.1f} ans). Envisager d'allonger la duration "
                "des obligations ou d'ajouter une couverture (obligations long terme, swaps) ; "
                "voir la section Ratio de couverture."
            )
        if not actions:
            actions.append("Capitalisation et exposition aux taux dans les seuils : maintenir la "
                           "strategie et reevaluer a chaque revue de la politique.")

        return {
            "statut": statut,
            "couleur": couleur,
            "ratio_capitalisation": fr,
            "ecart_duration": gap,
            "ratio_couverture": hr,
            "delta_ratio_100pb_pts": delta_fr_pts,
            "composante_capitalisation": {"niveau": fund[0], "texte": fund[2]},
            "composante_taux": {"niveau": rate[0], "texte": rate[2]},
            "actions": actions,
        }

    def liability_proxy_weights(self, asset_value: float) -> np.ndarray:
        """
        Exposition equivalente du passif, en fraction de l'actif, sur la classe
        obligataire de reference (celle dont la duration est la plus proche du
        passif parmi les obligations nominales a duration positive).
        """
        proxy = np.zeros(self.n_assets)
        candidates = np.where(self.durations > 1.0)[0]
        if len(candidates) == 0 or asset_value <= 0:
            return proxy
        ref = candidates[np.argmax(self.durations[candidates])] if self.reference_bond_index is None \
            else self.reference_bond_index
        proxy[ref] = (self.liability.present_value / asset_value) * (
            self.liability.duration / self.durations[ref]
        )
        return proxy

    def optimize_surplus(
        self,
        asset_value: float,
        target_surplus_return: Optional[float] = None,
        constraint_set=None,
    ) -> OptimizationResult:
        """
        Optimisation du surplus.

        Maximise le rendement du surplus sous contrainte de volatilite du surplus.
        R_S = R_A - (L/A)*R_L
        """
        start_time = time.time()
        leverage = self.liability.present_value / asset_value

        w = cp.Variable(self.n_assets)

        # Variance du surplus : le passif est represente par une position
        # "courte" dans la classe obligataire de reference, mise a l'echelle par
        # le rapport des durations (L/A x D_L / D_ref). Le surplus varie donc
        # avec les taux comme le passif, au lieu d'etre traite comme fixe.
        liability_proxy = self.liability_proxy_weights(asset_value)
        surplus_variance = cp.quad_form(w - liability_proxy, self.sigma)
        surplus_return = self.mu @ w - leverage * self.liability.growth_rate

        if constraint_set is not None:
            from constraints.manager import ConstraintManager
            cm = ConstraintManager(self.n_assets)
            constraints = cm.to_cvxpy_constraints(w, constraint_set, self.sigma)
        else:
            constraints = [
                cp.sum(w) == 1,
                w >= self.min_weights,
                w <= self.max_weights,
            ]

        # Objectif: maximiser rendement surplus pour variance donnee
        # ou minimiser variance surplus
        lambda_risk = 5.0  # Aversion au risque
        objective = cp.Maximize(surplus_return - lambda_risk * surplus_variance)

        try:
            prob = cp.Problem(objective, constraints)
            prob.solve(solver=cp.CLARABEL, verbose=False)

            if prob.status in ["optimal", "optimal_inaccurate"]:
                from models.base import clean_weights
                w_optimal = np.maximum(w.value, 0)
                w_optimal = clean_weights(w_optimal / w_optimal.sum(), self.max_weights)

                port_return = w_optimal @ self.mu
                port_vol = np.sqrt(w_optimal @ self.sigma @ w_optimal)
                sharpe = (port_return - self.rf) / port_vol if port_vol > 1e-10 else 0.0

                # Contributions au risque
                marginal = self.sigma @ w_optimal
                risk_contrib = w_optimal * marginal / port_vol if port_vol > 1e-10 else np.zeros(self.n_assets)

                return OptimizationResult(
                    weights=w_optimal,
                    asset_names=self.asset_names,
                    expected_return=port_return,
                    volatility=port_vol,
                    sharpe_ratio=sharpe,
                    risk_contributions=risk_contrib,
                    metadata={
                        "model": "ALM_Surplus",
                        "surplus_return": float(port_return - leverage * self.liability.growth_rate),
                        "duration_gap": float(self.compute_duration_gap(w_optimal, asset_value)),
                        "funded_ratio": float(self.compute_funded_ratio(asset_value)),
                        "hedge_ratio": float(self.compute_hedge_ratio(w_optimal, asset_value)),
                        "surplus_volatility": float(np.sqrt(
                            (w_optimal - liability_proxy) @ self.sigma @ (w_optimal - liability_proxy)
                        )),
                        "solver_status": prob.status,
                    },
                    status="optimal",
                    solver_time=time.time() - start_time,
                )
            else:
                return OptimizationResult(
                    weights=np.ones(self.n_assets) / self.n_assets,
                    asset_names=self.asset_names,
                    expected_return=0.0,
                    volatility=0.0,
                    sharpe_ratio=0.0,
                    risk_contributions=np.zeros(self.n_assets),
                    metadata={"solver_status": prob.status},
                    status="infeasible",
                    solver_time=time.time() - start_time,
                )
        except cp.SolverError as e:
            return OptimizationResult(
                weights=np.ones(self.n_assets) / self.n_assets,
                asset_names=self.asset_names,
                expected_return=0.0,
                volatility=0.0,
                sharpe_ratio=0.0,
                risk_contributions=np.zeros(self.n_assets),
                metadata={"error": str(e)},
                status="solver_error",
                solver_time=time.time() - start_time,
            )

    def optimize_liability_hedge(
        self,
        asset_value: float,
        hedge_ratio_target: float = 0.80,
        hedge_duration: Optional[float] = None,
        weights: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Allocation obligataire necessaire pour atteindre un ratio de couverture cible.

        Ratio de couverture (duration-dollar) = (A x w_c x D_c) / (L x D_L)
        donc w_c = HR_cible x (L x D_L) / (A x D_c)
        ou w_c est le poids des obligations de couverture et D_c leur duration.
        """
        L, D_L = self.liability.present_value, self.liability.duration
        if hedge_duration is None:
            hedge_duration = float(np.max(self.durations)) if np.any(self.durations > 0) else 0.0
        required_dd = hedge_ratio_target * L * D_L  # duration-dollar a detenir
        required_weight = required_dd / (asset_value * hedge_duration) if hedge_duration > 0 else np.inf
        current_hr = self.compute_hedge_ratio(weights, asset_value) if weights is not None else None

        if required_weight <= 1.0:
            texte = (
                f"Pour couvrir {hedge_ratio_target:.0%} de la sensibilite aux taux du passif "
                f"({L/1e6:,.0f} M$, duration {D_L:.1f} ans), il faut une duration-dollar de "
                f"{required_dd/1e6:,.0f} M$-an, soit environ {required_weight:.0%} de l'actif "
                f"en obligations de duration {hedge_duration:.1f} ans."
            )
            faisable = True
        else:
            max_hr = asset_value * hedge_duration / (L * D_L) if L * D_L > 0 else 0.0
            texte = (
                f"Avec des obligations de duration {hedge_duration:.1f} ans, couvrir "
                f"{hedge_ratio_target:.0%} exigerait {required_weight:.0%} de l'actif : "
                f"impossible sans levier. Meme 100 % de l'actif dans ces obligations ne couvre que "
                f"{max_hr:.0%}. Il faut des obligations plus longues ou une couverture par derives "
                f"(swaps, contrats a terme obligataires)."
            )
            faisable = False

        return {
            "ratio_couverture_cible": hedge_ratio_target,
            "ratio_couverture_actuel": current_hr,
            "duration_couverture": hedge_duration,
            "duration_dollar_requise": required_dd,
            "poids_actifs_couverture_recommande": float(required_weight),
            "faisable_sans_levier": faisable,
            "duration_passif": D_L,
            "recommandation": texte,
        }

    def optimize_glide_path(
        self,
        current_funded_ratio: float,
        horizon_years: int = 10,
        target_funded_ratio: float = 1.10,
        asset_value: float = 1_000_000_000.0,
    ) -> List[Dict]:
        """
        Trajectoire de desensibilisation dynamique.
        A mesure que le ratio de capitalisation s'ameliore,
        on passe des actifs de croissance aux actifs de couverture.
        """
        glide_path = []

        for year in range(horizon_years + 1):
            # Interpolation lineaire du ratio de capitalisation projete
            progress = year / horizon_years
            projected_fr = current_funded_ratio + progress * (target_funded_ratio - current_funded_ratio)

            # Allocation croissance vs couverture
            if projected_fr < 0.85:
                growth_pct = 0.65
                hedge_pct = 0.30
                cash_pct = 0.05
            elif projected_fr < 1.0:
                growth_pct = 0.55 - 0.30 * (projected_fr - 0.85) / 0.15
                hedge_pct = 0.40 + 0.20 * (projected_fr - 0.85) / 0.15
                cash_pct = 0.05
            elif projected_fr < 1.10:
                growth_pct = 0.35 - 0.15 * (projected_fr - 1.0) / 0.10
                hedge_pct = 0.60 + 0.10 * (projected_fr - 1.0) / 0.10
                cash_pct = 0.05
            else:
                growth_pct = 0.20
                hedge_pct = 0.75
                cash_pct = 0.05

            glide_path.append({
                "Annee": year,
                "Ratio capitalisation projete": projected_fr,
                "Actifs de croissance (%)": growth_pct * 100,
                "Actifs de couverture (%)": hedge_pct * 100,
                "Encaisse (%)": cash_pct * 100,
            })

        return glide_path

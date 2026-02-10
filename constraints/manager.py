"""
Gestionnaire de contraintes pour l'optimisation de portefeuille.
Supporte les contraintes classiques (long-only) et les contraintes
avec levier pour l'alpha portable (positions courtes autorisees).
"""

import numpy as np
import cvxpy as cp
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


@dataclass
class GroupConstraint:
    """Contrainte sur un groupe de classes d'actifs."""
    name_fr: str
    asset_indices: List[int]
    min_allocation: float
    max_allocation: float


@dataclass
class ConstraintSet:
    """Conteneur pour toutes les contraintes d'optimisation."""
    min_weights: np.ndarray
    max_weights: np.ndarray
    group_constraints: List[GroupConstraint] = field(default_factory=list)
    turnover_limit: Optional[float] = None
    current_weights: Optional[np.ndarray] = None
    tracking_error_limit: Optional[float] = None
    benchmark_weights: Optional[np.ndarray] = None
    liquidity_floor: Optional[float] = None
    liquidity_scores: Optional[np.ndarray] = None
    esg_min_score: Optional[float] = None
    esg_scores: Optional[np.ndarray] = None
    max_single_position: Optional[float] = None


@dataclass
class LeveragedConstraintSet:
    """Contraintes pour l'alpha portable avec levier (positions longues/courtes).

    Dans une strategie d'alpha portable :
    - Le portefeuille beta est long-only et sert de collateral
    - L'overlay alpha peut contenir des positions longues et courtes
    - Le levier brut = sum(|w|) peut depasser 1.0
    - L'exposition nette reste geree
    """
    n_assets: int
    # Bornes individuelles (min peut etre negatif pour short)
    min_weights: np.ndarray
    max_weights: np.ndarray
    # Levier brut maximal (sum |w|). Ex: 1.5 = 150%
    max_gross_leverage: float = 1.5
    # Exposition nette cible (sum w). Typiquement 1.0
    target_net_exposure: float = 1.0
    net_exposure_tolerance: float = 0.05
    # Actifs eligibles au short (indices)
    short_eligible: Optional[List[int]] = None
    # Limite de vente a decouvert par actif
    max_short_per_asset: float = 0.15
    # Contraintes de groupe
    group_constraints: List[GroupConstraint] = field(default_factory=list)
    # Tracking error vs benchmark beta
    tracking_error_limit: Optional[float] = None
    benchmark_weights: Optional[np.ndarray] = None
    # Cout de financement annualise (pour penalite dans l'objectif)
    financing_spread: float = 0.005  # 50 bps
    # Limite sur le nombre de positions courtes
    max_short_positions: Optional[int] = None
    # Contraintes ESG (meme avec levier)
    esg_min_score: Optional[float] = None
    esg_scores: Optional[np.ndarray] = None


class ConstraintManager:
    """Construit et valide les ensembles de contraintes."""

    def __init__(self, n_assets: int, asset_names: Optional[List[str]] = None):
        self.n_assets = n_assets
        self.asset_names = asset_names or [f"Actif_{i}" for i in range(n_assets)]

    def build_default_constraints(
        self,
        min_weights: np.ndarray,
        max_weights: np.ndarray,
    ) -> ConstraintSet:
        """Construit les contraintes par defaut."""
        return ConstraintSet(
            min_weights=min_weights.copy(),
            max_weights=max_weights.copy(),
        )

    def add_group_constraint(
        self,
        constraint_set: ConstraintSet,
        name_fr: str,
        asset_indices: List[int],
        min_alloc: float = 0.0,
        max_alloc: float = 1.0,
    ) -> ConstraintSet:
        """Ajoute une contrainte de groupe."""
        constraint_set.group_constraints.append(
            GroupConstraint(
                name_fr=name_fr,
                asset_indices=asset_indices,
                min_allocation=min_alloc,
                max_allocation=max_alloc,
            )
        )
        return constraint_set

    def to_cvxpy_constraints(
        self, w: cp.Variable, constraint_set: ConstraintSet,
        cov_matrix: Optional[np.ndarray] = None,
    ) -> List:
        """Convertit le ConstraintSet en contraintes cvxpy."""
        constraints = [
            cp.sum(w) == 1,
            w >= constraint_set.min_weights,
            w <= constraint_set.max_weights,
        ]

        # Contraintes de groupe
        for gc in constraint_set.group_constraints:
            group_sum = cp.sum(w[gc.asset_indices])
            constraints.append(group_sum >= gc.min_allocation)
            constraints.append(group_sum <= gc.max_allocation)

        # Contrainte de rotation (turnover)
        if constraint_set.turnover_limit is not None and constraint_set.current_weights is not None:
            diff = w - constraint_set.current_weights
            constraints.append(cp.norm(diff, 1) <= constraint_set.turnover_limit * 2)

        # Contrainte d'erreur de suivi
        if (constraint_set.tracking_error_limit is not None
                and constraint_set.benchmark_weights is not None
                and cov_matrix is not None):
            diff = w - constraint_set.benchmark_weights
            te_squared = cp.quad_form(diff, cov_matrix)
            constraints.append(te_squared <= constraint_set.tracking_error_limit ** 2)

        # Contrainte ESG
        if constraint_set.esg_min_score is not None and constraint_set.esg_scores is not None:
            constraints.append(
                constraint_set.esg_scores @ w >= constraint_set.esg_min_score
            )

        # Contrainte de position maximale
        if constraint_set.max_single_position is not None:
            constraints.append(w <= constraint_set.max_single_position)

        return constraints

    # =============================================================
    # Methodes pour l'alpha portable (levier)
    # =============================================================

    def build_leveraged_constraints(
        self,
        min_weights: np.ndarray,
        max_weights: np.ndarray,
        max_gross_leverage: float = 1.5,
        target_net_exposure: float = 1.0,
        short_eligible: Optional[List[int]] = None,
        max_short_per_asset: float = 0.15,
        financing_spread: float = 0.005,
    ) -> LeveragedConstraintSet:
        """Construit les contraintes pour un portefeuille avec levier."""
        return LeveragedConstraintSet(
            n_assets=self.n_assets,
            min_weights=min_weights.copy(),
            max_weights=max_weights.copy(),
            max_gross_leverage=max_gross_leverage,
            target_net_exposure=target_net_exposure,
            short_eligible=short_eligible,
            max_short_per_asset=max_short_per_asset,
            financing_spread=financing_spread,
        )

    def to_cvxpy_leveraged_constraints(
        self,
        w: cp.Variable,
        lcs: LeveragedConstraintSet,
        cov_matrix: Optional[np.ndarray] = None,
    ) -> List:
        """Convertit un LeveragedConstraintSet en contraintes cvxpy.

        A la difference du cas classique :
        - sum(w) ≈ target_net_exposure (avec tolerance)
        - sum(|w|) <= max_gross_leverage
        - Les positions courtes ne sont autorisees que sur les actifs eligibles
        """
        constraints = []

        # Exposition nette (sum w ≈ target)
        net_tgt = lcs.target_net_exposure
        net_tol = lcs.net_exposure_tolerance
        constraints.append(cp.sum(w) >= net_tgt - net_tol)
        constraints.append(cp.sum(w) <= net_tgt + net_tol)

        # Levier brut (sum |w| <= max_gross_leverage)
        constraints.append(cp.norm(w, 1) <= lcs.max_gross_leverage)

        # Bornes individuelles
        constraints.append(w >= lcs.min_weights)
        constraints.append(w <= lcs.max_weights)

        # Actifs non eligibles au short : w >= 0
        if lcs.short_eligible is not None:
            non_shortable = [
                i for i in range(lcs.n_assets)
                if i not in lcs.short_eligible
            ]
            for i in non_shortable:
                constraints.append(w[i] >= 0)

        # Limite de short par actif
        if lcs.short_eligible is not None:
            for i in lcs.short_eligible:
                constraints.append(w[i] >= -lcs.max_short_per_asset)

        # Contraintes de groupe
        for gc in lcs.group_constraints:
            group_sum = cp.sum(w[gc.asset_indices])
            constraints.append(group_sum >= gc.min_allocation)
            constraints.append(group_sum <= gc.max_allocation)

        # Tracking error vs benchmark beta
        if (lcs.tracking_error_limit is not None
                and lcs.benchmark_weights is not None
                and cov_matrix is not None):
            diff = w - lcs.benchmark_weights
            te_squared = cp.quad_form(diff, cov_matrix)
            constraints.append(te_squared <= lcs.tracking_error_limit ** 2)

        # Contrainte ESG (meme leveraged)
        if lcs.esg_min_score is not None and lcs.esg_scores is not None:
            constraints.append(
                lcs.esg_scores @ w >= lcs.esg_min_score
            )

        return constraints

    def validate_leveraged_allocation(
        self, weights: np.ndarray, lcs: LeveragedConstraintSet
    ) -> Tuple[bool, List[str]]:
        """Verifie si une allocation avec levier satisfait les contraintes."""
        violations = []
        tol = 1e-6

        # Exposition nette
        net_exp = np.sum(weights)
        if abs(net_exp - lcs.target_net_exposure) > lcs.net_exposure_tolerance + tol:
            violations.append(
                f"Exposition nette ({net_exp:.4f}) hors tolerance "
                f"[{lcs.target_net_exposure - lcs.net_exposure_tolerance:.2f}, "
                f"{lcs.target_net_exposure + lcs.net_exposure_tolerance:.2f}]"
            )

        # Levier brut
        gross = np.sum(np.abs(weights))
        if gross > lcs.max_gross_leverage + tol:
            violations.append(
                f"Levier brut ({gross:.4f}) excede la limite ({lcs.max_gross_leverage:.2f})"
            )

        # Bornes individuelles
        for i in range(self.n_assets):
            if weights[i] < lcs.min_weights[i] - tol:
                violations.append(
                    f"{self.asset_names[i]}: {weights[i]:.2%} < min {lcs.min_weights[i]:.2%}"
                )
            if weights[i] > lcs.max_weights[i] + tol:
                violations.append(
                    f"{self.asset_names[i]}: {weights[i]:.2%} > max {lcs.max_weights[i]:.2%}"
                )

        # Positions courtes non autorisees
        if lcs.short_eligible is not None:
            for i in range(self.n_assets):
                if i not in lcs.short_eligible and weights[i] < -tol:
                    violations.append(
                        f"{self.asset_names[i]}: position courte ({weights[i]:.2%}) non autorisee"
                    )

        # Contraintes de groupe
        for gc in lcs.group_constraints:
            group_sum = np.sum(weights[gc.asset_indices])
            if group_sum < gc.min_allocation - tol:
                violations.append(
                    f"{gc.name_fr}: {group_sum:.2%} < min {gc.min_allocation:.2%}"
                )
            if group_sum > gc.max_allocation + tol:
                violations.append(
                    f"{gc.name_fr}: {group_sum:.2%} > max {gc.max_allocation:.2%}"
                )

        # ESG
        if lcs.esg_min_score is not None and lcs.esg_scores is not None:
            portfolio_esg = lcs.esg_scores @ weights
            if portfolio_esg < lcs.esg_min_score - tol:
                violations.append(
                    f"Score ESG ({portfolio_esg:.1f}) < minimum ({lcs.esg_min_score:.1f})"
                )

        return len(violations) == 0, violations

    def compute_leverage_metrics(self, weights: np.ndarray) -> Dict:
        """Calcule les metriques de levier d'un portefeuille."""
        long_weights = weights[weights > 0]
        short_weights = weights[weights < 0]

        gross_exposure = np.sum(np.abs(weights))
        net_exposure = np.sum(weights)
        long_exposure = np.sum(long_weights)
        short_exposure = np.sum(np.abs(short_weights))
        n_long = int(np.sum(weights > 1e-4))
        n_short = int(np.sum(weights < -1e-4))

        return {
            "levier_brut": gross_exposure,
            "exposition_nette": net_exposure,
            "exposition_longue": long_exposure,
            "exposition_courte": short_exposure,
            "nb_positions_longues": n_long,
            "nb_positions_courtes": n_short,
            "ratio_long_short": long_exposure / short_exposure if short_exposure > 1e-10 else float("inf"),
        }

    def to_scipy_bounds(self, constraint_set: ConstraintSet) -> List[Tuple[float, float]]:
        """Convertit en bornes pour scipy.optimize."""
        return list(zip(
            constraint_set.min_weights.tolist(),
            constraint_set.max_weights.tolist()
        ))

    def to_scipy_constraints(
        self, constraint_set: ConstraintSet,
        cov_matrix: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """Convertit en contraintes scipy.optimize."""
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        ]

        for gc in constraint_set.group_constraints:
            idx = gc.asset_indices
            min_a = gc.min_allocation
            max_a = gc.max_allocation
            constraints.append(
                {"type": "ineq", "fun": lambda w, i=idx, m=min_a: np.sum(w[i]) - m}
            )
            constraints.append(
                {"type": "ineq", "fun": lambda w, i=idx, m=max_a: m - np.sum(w[i])}
            )

        if constraint_set.esg_min_score is not None and constraint_set.esg_scores is not None:
            scores = constraint_set.esg_scores
            min_score = constraint_set.esg_min_score
            constraints.append(
                {"type": "ineq", "fun": lambda w: scores @ w - min_score}
            )

        return constraints

    def validate_allocation(
        self, weights: np.ndarray, constraint_set: ConstraintSet
    ) -> Tuple[bool, List[str]]:
        """Verifie si une allocation satisfait toutes les contraintes."""
        violations = []
        tol = 1e-6

        # Somme = 1
        if abs(np.sum(weights) - 1.0) > tol:
            violations.append(
                f"La somme des poids ({np.sum(weights):.4f}) ne vaut pas 1.0"
            )

        # Bornes individuelles
        for i in range(self.n_assets):
            if weights[i] < constraint_set.min_weights[i] - tol:
                violations.append(
                    f"{self.asset_names[i]}: {weights[i]:.2%} < min {constraint_set.min_weights[i]:.2%}"
                )
            if weights[i] > constraint_set.max_weights[i] + tol:
                violations.append(
                    f"{self.asset_names[i]}: {weights[i]:.2%} > max {constraint_set.max_weights[i]:.2%}"
                )

        # Contraintes de groupe
        for gc in constraint_set.group_constraints:
            group_sum = np.sum(weights[gc.asset_indices])
            if group_sum < gc.min_allocation - tol:
                violations.append(
                    f"{gc.name_fr}: {group_sum:.2%} < min {gc.min_allocation:.2%}"
                )
            if group_sum > gc.max_allocation + tol:
                violations.append(
                    f"{gc.name_fr}: {group_sum:.2%} > max {gc.max_allocation:.2%}"
                )

        # ESG
        if constraint_set.esg_min_score is not None and constraint_set.esg_scores is not None:
            portfolio_esg = constraint_set.esg_scores @ weights
            if portfolio_esg < constraint_set.esg_min_score - tol:
                violations.append(
                    f"Score ESG du portefeuille ({portfolio_esg:.1f}) < minimum ({constraint_set.esg_min_score:.1f})"
                )

        is_valid = len(violations) == 0
        return is_valid, violations

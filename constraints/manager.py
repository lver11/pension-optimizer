"""
Gestionnaire de contraintes pour l'optimisation de portefeuille.
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

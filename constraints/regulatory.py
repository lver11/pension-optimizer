"""
Contraintes reglementaires specifiques au Quebec pour les fonds de pension.
"""

import numpy as np
from typing import Dict, List, Tuple
from constraints.manager import GroupConstraint


class QuebecPensionRegulations:
    """Contraintes reglementaires des fonds de pension du Quebec."""

    # Limites reglementaires (avec defauts sensibles)
    MAX_EQUITY_TOTAL = 0.70
    MAX_PRIVATE_EQUITY = 0.20
    MAX_ALTERNATIVES = 0.40  # PE + Infra + Immobilier
    MIN_LIQUIDITY = 0.02
    MAX_SINGLE_ISSUER = 0.10
    MIN_DOMESTIC_CONTENT = 0.20
    MAX_FOREIGN_CONTENT = 0.80
    MIN_FUNDED_RATIO_ALERT = 0.80

    # Indices des classes d'actifs (selon ASSET_CLASSES_ORDER dans config)
    EQUITY_INDICES = [0, 1, 2, 3]          # Actions CDN, US, EAFE, Emergentes
    BOND_INDICES = [4, 5, 6]               # Oblig Gov, Corp, Inflation
    ALTERNATIVE_INDICES = [7, 8, 9]        # Immobilier, Infrastructure, PE
    PE_INDEX = [9]                          # Capital investissement
    COMMODITY_INDEX = [10]                  # Matieres premieres
    CASH_INDEX = [11]                       # Encaisse
    DOMESTIC_INDICES = [0, 4, 5, 6]        # Actions CDN + Obligations CDN
    FOREIGN_INDICES = [1, 2, 3]            # Actions internationales

    @classmethod
    def get_group_constraints(cls) -> List[GroupConstraint]:
        """Retourne les contraintes reglementaires standard du Quebec."""
        return [
            GroupConstraint(
                name_fr="Actions totales <= 70%",
                asset_indices=cls.EQUITY_INDICES,
                min_allocation=0.0,
                max_allocation=cls.MAX_EQUITY_TOTAL,
            ),
            GroupConstraint(
                name_fr="Capital investissement <= 20%",
                asset_indices=cls.PE_INDEX,
                min_allocation=0.0,
                max_allocation=cls.MAX_PRIVATE_EQUITY,
            ),
            GroupConstraint(
                name_fr="Actifs alternatifs <= 40%",
                asset_indices=cls.ALTERNATIVE_INDICES,
                min_allocation=0.0,
                max_allocation=cls.MAX_ALTERNATIVES,
            ),
            GroupConstraint(
                name_fr="Liquidite minimum >= 2%",
                asset_indices=cls.CASH_INDEX,
                min_allocation=cls.MIN_LIQUIDITY,
                max_allocation=1.0,
            ),
            GroupConstraint(
                name_fr="Obligations totales",
                asset_indices=cls.BOND_INDICES,
                min_allocation=0.10,
                max_allocation=0.70,
            ),
        ]

    @classmethod
    def validate_compliance(
        cls,
        weights: np.ndarray,
        asset_names: List[str],
    ) -> Tuple[bool, List[str]]:
        """Valide la conformite reglementaire de l'allocation."""
        violations = []
        tol = 1e-6

        # Actions totales
        equity_total = np.sum(weights[cls.EQUITY_INDICES])
        if equity_total > cls.MAX_EQUITY_TOTAL + tol:
            violations.append(
                f"Actions totales ({equity_total:.1%}) excedent la limite de {cls.MAX_EQUITY_TOTAL:.0%}"
            )

        # Capital investissement
        pe_total = np.sum(weights[cls.PE_INDEX])
        if pe_total > cls.MAX_PRIVATE_EQUITY + tol:
            violations.append(
                f"Capital investissement ({pe_total:.1%}) excede la limite de {cls.MAX_PRIVATE_EQUITY:.0%}"
            )

        # Alternatives
        alt_total = np.sum(weights[cls.ALTERNATIVE_INDICES])
        if alt_total > cls.MAX_ALTERNATIVES + tol:
            violations.append(
                f"Actifs alternatifs ({alt_total:.1%}) excedent la limite de {cls.MAX_ALTERNATIVES:.0%}"
            )

        # Liquidite
        cash_total = np.sum(weights[cls.CASH_INDEX])
        if cash_total < cls.MIN_LIQUIDITY - tol:
            violations.append(
                f"Liquidite ({cash_total:.1%}) inferieure au minimum de {cls.MIN_LIQUIDITY:.0%}"
            )

        # Position maximale
        if np.max(weights) > cls.MAX_SINGLE_ISSUER + 0.20 + tol:
            max_idx = np.argmax(weights)
            violations.append(
                f"Position concentree: {asset_names[max_idx]} ({weights[max_idx]:.1%})"
            )

        is_compliant = len(violations) == 0
        return is_compliant, violations

    @classmethod
    def funding_policy_check(
        cls, funded_ratio: float, target_ratio: float = 1.0,
    ) -> Dict:
        """Analyse la politique de capitalisation."""
        if funded_ratio < 0.80:
            status = "critique"
            action = "Action corrective immediate requise. Plan de redressement obligatoire."
            color = "red"
        elif funded_ratio < 0.90:
            status = "insuffisant"
            action = "Plan de redressement a etablir dans les 12 prochains mois."
            color = "orange"
        elif funded_ratio < 1.00:
            status = "surveillance"
            action = "Ratio inferieur a 100%. Surveillance accrue recommandee."
            color = "yellow"
        elif funded_ratio < 1.10:
            status = "adequat"
            action = "Ratio adequat. Maintenir la strategie actuelle."
            color = "green"
        else:
            status = "excedentaire"
            action = "Surplus disponible. Considerer la politique de distribution du surplus."
            color = "blue"

        return {
            "ratio_capitalisation": funded_ratio,
            "statut": status,
            "action_recommandee": action,
            "couleur": color,
            "ecart_cible": funded_ratio - target_ratio,
        }

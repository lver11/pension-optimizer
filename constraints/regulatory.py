"""
Contraintes reglementaires specifiques au Quebec pour les fonds de pension.
Inclut les regles pour les strategies avec levier (alpha portable).
"""

import numpy as np
from typing import Dict, List, Tuple
from constraints.manager import GroupConstraint, LeveragedConstraintSet


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


class PortableAlphaRegulations:
    """Contraintes reglementaires pour les strategies d'alpha portable.

    L'alpha portable utilise le levier via des positions courtes et des
    instruments derives. Les caisses de retraite du Quebec sont soumises
    a des limites specifiques sur l'utilisation du levier.
    """

    # Limites reglementaires pour les strategies avec levier
    MAX_GROSS_LEVERAGE = 2.0       # Levier brut max 200%
    MAX_SHORT_EXPOSURE = 0.50      # Exposition courte max 50% de l'actif net
    MAX_SHORT_PER_ASSET = 0.15     # Position courte max par actif 15%
    MAX_TOTAL_DERIVATIVES = 0.30   # Notionnel derives max 30%
    MIN_COLLATERAL_RATIO = 1.05    # Ratio de collatéral minimum 105%
    MAX_COUNTERPARTY = 0.10        # Exposition max par contrepartie 10%

    # Actifs eligibles a la vente a decouvert
    # Seuls les actifs liquides (score >= 0.75) sont eligibles
    SHORT_ELIGIBLE_INDICES = [0, 1, 2, 3, 4, 5, 6, 10]  # Equities + Bonds + Commodities

    @classmethod
    def get_leverage_group_constraints(cls) -> List[GroupConstraint]:
        """Contraintes de groupe pour un portefeuille avec levier."""
        return [
            GroupConstraint(
                name_fr="Actions totales (nettes) <= 70%",
                asset_indices=QuebecPensionRegulations.EQUITY_INDICES,
                min_allocation=-0.20,  # Short equity autorise
                max_allocation=0.70,
            ),
            GroupConstraint(
                name_fr="Obligations totales (nettes)",
                asset_indices=QuebecPensionRegulations.BOND_INDICES,
                min_allocation=-0.15,
                max_allocation=0.70,
            ),
            GroupConstraint(
                name_fr="Actifs alternatifs <= 40% (long-only)",
                asset_indices=QuebecPensionRegulations.ALTERNATIVE_INDICES,
                min_allocation=0.0,
                max_allocation=QuebecPensionRegulations.MAX_ALTERNATIVES,
            ),
        ]

    @classmethod
    def validate_leverage_compliance(
        cls,
        weights: np.ndarray,
        asset_names: List[str],
    ) -> Tuple[bool, List[str]]:
        """Valide la conformite reglementaire d'une allocation avec levier."""
        violations = []
        tol = 1e-6

        # Levier brut
        gross = np.sum(np.abs(weights))
        if gross > cls.MAX_GROSS_LEVERAGE + tol:
            violations.append(
                f"Levier brut ({gross:.1%}) excede la limite reglementaire de {cls.MAX_GROSS_LEVERAGE:.0%}"
            )

        # Exposition courte totale
        short_exposure = np.sum(np.abs(weights[weights < 0]))
        if short_exposure > cls.MAX_SHORT_EXPOSURE + tol:
            violations.append(
                f"Exposition courte ({short_exposure:.1%}) excede la limite de {cls.MAX_SHORT_EXPOSURE:.0%}"
            )

        # Position courte par actif
        for i in range(len(weights)):
            if weights[i] < -tol:
                if abs(weights[i]) > cls.MAX_SHORT_PER_ASSET + tol:
                    violations.append(
                        f"{asset_names[i]}: position courte ({weights[i]:.1%}) excede {cls.MAX_SHORT_PER_ASSET:.0%}"
                    )
                if i not in cls.SHORT_ELIGIBLE_INDICES:
                    violations.append(
                        f"{asset_names[i]}: vente a decouvert non autorisee (actif illiquide)"
                    )

        # Actions nettes
        equity_net = np.sum(weights[QuebecPensionRegulations.EQUITY_INDICES])
        if equity_net > QuebecPensionRegulations.MAX_EQUITY_TOTAL + tol:
            violations.append(
                f"Actions nettes ({equity_net:.1%}) excedent la limite de {QuebecPensionRegulations.MAX_EQUITY_TOTAL:.0%}"
            )

        # Alternatives (long-only)
        alt_long = np.sum(np.maximum(weights[QuebecPensionRegulations.ALTERNATIVE_INDICES], 0))
        if alt_long > QuebecPensionRegulations.MAX_ALTERNATIVES + tol:
            violations.append(
                f"Actifs alternatifs long ({alt_long:.1%}) excedent {QuebecPensionRegulations.MAX_ALTERNATIVES:.0%}"
            )

        return len(violations) == 0, violations

    @classmethod
    def compute_financing_cost(
        cls,
        weights: np.ndarray,
        financing_spread: float = 0.005,
        risk_free_rate: float = 0.025,
    ) -> Dict:
        """Calcule le cout de financement d'une strategie avec levier.

        Le cout de financement provient :
        - Des emprunts pour les positions courtes (rebate - financing spread)
        - Du cout de marge sur le levier excedentaire
        """
        short_weights = weights[weights < 0]
        short_exposure = np.sum(np.abs(short_weights))
        gross_leverage = np.sum(np.abs(weights))
        excess_leverage = max(0, gross_leverage - 1.0)

        # Cout de financement annualise
        short_financing = short_exposure * financing_spread
        margin_cost = excess_leverage * financing_spread * 0.5  # Cout reduit sur marge
        total_cost = short_financing + margin_cost

        return {
            "exposition_courte": short_exposure,
            "levier_excedentaire": excess_leverage,
            "cout_short": short_financing,
            "cout_marge": margin_cost,
            "cout_total_annuel": total_cost,
            "cout_total_bps": total_cost * 10000,
            "spread_financement": financing_spread,
        }

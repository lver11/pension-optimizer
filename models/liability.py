"""
Passif actuariel decrit par ses flux de prestations projetes.

Avec les flux annee par annee (fournis par l'actuaire), le passif est reevalue
exactement sur une courbe de taux choquee : la convexite est captee sans
approximation et on obtient les durations par segment de courbe (durations cles).

Conventions : flux en fin d'annee, actualisation composee annuellement,
taux zero-coupon interpoles lineairement entre les points de la courbe et
prolonges a plat au-dela.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

KEY_TENORS = (2.0, 5.0, 10.0, 20.0, 30.0)
BP = 1e-4


@dataclass
class YieldCurve:
    """Courbe de taux zero-coupon (taux annuels composes)."""
    tenors: np.ndarray
    rates: np.ndarray

    @classmethod
    def flat(cls, rate: float) -> "YieldCurve":
        return cls(np.array([1.0, 30.0]), np.array([rate, rate]))

    @classmethod
    def from_dict(cls, points: Dict, fallback_rate: float = 0.05) -> "YieldCurve":
        if not points:
            return cls.flat(fallback_rate)
        items = sorted((float(k), float(v)) for k, v in points.items())
        t, r = zip(*items)
        return cls(np.array(t), np.array(r))

    def rate(self, t) -> np.ndarray:
        return np.interp(np.asarray(t, dtype=float), self.tenors, self.rates)

    def discount(self, t, shift: float = 0.0, key_bump: Optional[np.ndarray] = None) -> np.ndarray:
        t = np.asarray(t, dtype=float)
        r = self.rate(t) + shift
        if key_bump is not None:
            r = r + key_bump
        return (1.0 + r) ** (-t)


def key_rate_bump(t: np.ndarray, tenor: float, tenors: Sequence[float] = KEY_TENORS,
                  size: float = BP) -> np.ndarray:
    """
    Choc triangulaire de `size` centre sur `tenor`, nul aux points voisins.
    La somme des chocs de tous les points cles donne un choc parallele.
    """
    tenors = list(tenors)
    i = tenors.index(tenor)
    left = tenors[i - 1] if i > 0 else None
    right = tenors[i + 1] if i < len(tenors) - 1 else None
    w = np.zeros_like(t, dtype=float)
    for k, x in enumerate(t):
        if x == tenor:
            w[k] = 1.0
        elif x < tenor:
            w[k] = 1.0 if left is None else max(0.0, (x - left) / (tenor - left))
        else:
            w[k] = 1.0 if right is None else max(0.0, (right - x) / (right - tenor))
    return size * w


@dataclass
class LiabilityCashflows:
    """Flux de prestations projetes (fin d'annee)."""
    years: np.ndarray
    flows: np.ndarray

    @classmethod
    def from_records(cls, records: List[Dict]) -> "LiabilityCashflows":
        years = np.array([float(r["annee"]) for r in records])
        flows = np.array([float(r.get("nominal", 0.0)) + float(r.get("indexe", 0.0)) for r in records])
        order = np.argsort(years)
        return cls(years[order], flows[order])

    def present_value(self, curve: YieldCurve, shift: float = 0.0,
                      key_bump: Optional[np.ndarray] = None) -> float:
        return float(np.sum(self.flows * curve.discount(self.years, shift, key_bump)))

    def effective_duration(self, curve: YieldCurve, h: float = 10 * BP) -> float:
        pv0 = self.present_value(curve)
        up, down = self.present_value(curve, h), self.present_value(curve, -h)
        return (down - up) / (2 * h * pv0)

    def effective_convexity(self, curve: YieldCurve, h: float = 10 * BP) -> float:
        pv0 = self.present_value(curve)
        up, down = self.present_value(curve, h), self.present_value(curve, -h)
        return (down + up - 2 * pv0) / (h * h * pv0)

    def key_rate_durations(self, curve: YieldCurve,
                           tenors: Sequence[float] = KEY_TENORS) -> Dict[float, float]:
        """Durations cles : sensibilite (en annees) a un choc de 1 pb localise sur chaque point."""
        pv0 = self.present_value(curve)
        out = {}
        for tenor in tenors:
            bump = key_rate_bump(self.years, tenor, tenors)
            up = self.present_value(curve, key_bump=bump)
            down = self.present_value(curve, key_bump=-bump)
            out[tenor] = (down - up) / (2 * BP * pv0)
        return out

    def weighted_average_life(self) -> float:
        return float(np.sum(self.years * self.flows) / np.sum(self.flows))


def approx_convexity(duration: float, rate: float) -> float:
    """
    Convexite approchee d'un titre (ou passif) de duration D : D(D+1)/(1+y)^2.
    Exacte pour un zero-coupon ; sert de defaut quand les flux ne sont pas fournis.
    """
    return duration * (duration + 1.0) / (1.0 + rate) ** 2


def estimated_liability_convexity(duration: float, rate: float) -> float:
    """
    Convexite d'un passif de retraite typique (flux en cloche) de duration donnee.
    Plus realiste que l'approximation zero-coupon, qui sous-estime la convexite
    d'un passif dont les flux sont etales sur plusieurs decennies.
    """
    if duration <= 0:
        return 0.0
    curve = YieldCurve.flat(rate)
    flows = LiabilityCashflows.from_records(example_cashflows(1.0, duration, rate))
    return flows.effective_convexity(curve)


def bond_key_rate_profile(duration: float, tenors: Sequence[float] = KEY_TENORS) -> Dict[float, float]:
    """
    Repartit la duration d'une classe obligataire sur les deux points cles qui
    l'encadrent (approximation : la classe se comporte comme un zero-coupon
    d'echeance egale a sa duration).
    """
    tenors = list(tenors)
    out = {t: 0.0 for t in tenors}
    if duration <= 0:
        return out
    if duration <= tenors[0]:
        out[tenors[0]] = duration
        return out
    if duration >= tenors[-1]:
        out[tenors[-1]] = duration
        return out
    for a, b in zip(tenors[:-1], tenors[1:]):
        if a <= duration <= b:
            wb = (duration - a) / (b - a)
            out[a] = duration * (1 - wb)
            out[b] = duration * wb
            break
    return out


def example_cashflows(present_value: float, target_duration: float, rate: float,
                      n_years: int = 60) -> List[Dict]:
    """
    Flux illustratifs (forme en cloche : croissance puis extinction) calibres pour
    avoir la valeur actualisee et la duration demandees sur une courbe plate.
    Sert de gabarit d'import ; a remplacer par les flux de l'evaluation actuarielle.
    """
    years = np.arange(1, n_years + 1, dtype=float)
    curve = YieldCurve.flat(rate)

    def shape(peak: float) -> np.ndarray:
        return years * np.exp(-years / peak)

    lo, hi = 0.5, 60.0
    for _ in range(100):
        mid = (lo + hi) / 2
        d = LiabilityCashflows(years, shape(mid)).effective_duration(curve)
        if d < target_duration:
            lo = mid
        else:
            hi = mid
    flows = shape((lo + hi) / 2)
    flows *= present_value / LiabilityCashflows(years, flows).present_value(curve)
    return [{"annee": int(y), "nominal": round(float(f), 2), "indexe": 0.0} for y, f in zip(years, flows)]

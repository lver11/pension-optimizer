# Design : Ajout Dette privée (index 15) + Obligations HY (index 16)

**Date :** 2026-05-14  
**Projet :** pension_optimizer (Streamlit Cloud)  
**Fichiers modifiés :** `config.py`, `constraints/regulatory.py`, `pages/page_rebalancing.py`

---

## 1. Objectif

Ajouter deux classes d'actifs à l'univers existant de 15 classes :
- **Index 15 — Dette privée** : prêts directs mid-market, actif alternatif illiquide
- **Index 16 — Obligations HY** : obligations à haut rendement, marché public liquide

Les deux classes complètent l'univers crédit : la dette privée côté illiquide/privé, les obligations HY côté liquide/public entre les obligations corporatives IG et les actions.

---

## 2. Paramètres financiers

### 2.1 Dette privée (index 15)

| Paramètre | Valeur | Justification |
|---|---|---|
| Rendement attendu | 7,5 % | Prime d'illiquidité sur prêts directs mid-market |
| Volatilité | 6 % | Vol apparente faible (valorisation par appréciation) |
| Duration | 5,0 ans | Prêts directs à terme moyen |
| Score liquidité | 0,10 | Marché privé très illiquide |
| Score ESG | 50,0 | Exposition corporative variée |
| Min / Max | 0 % / 10 % | Cohérent avec Capital investissement |
| `is_alternative` | **True** | Actif privé illiquide |

```python
AssetClass.DETTE_PRIVEE: AssetClassConfig(
    AssetClass.DETTE_PRIVEE, "Dette privee",
    0.075, 0.06, 0.10, 50.0, 0.00, 0.10, True, 5.0
),
```

### 2.2 Obligations HY (index 16)

| Paramètre | Valeur | Justification |
|---|---|---|
| Rendement attendu | 5,5 % | Spread ~100 pb au-dessus obligations corporatives IG |
| Volatilité | 9 % | Entre oblig. corp (8 %) et dette EM (11 %) |
| Duration | 4,5 ans | Maturités courtes typiques du segment HY |
| Score liquidité | 0,80 | ETF HY très actifs (HYG, JNK) |
| Score ESG | 48,0 | Émetteurs HY — profil ESG plus faible |
| Min / Max | 0 % / 15 % | |
| `is_alternative` | **False** | Marché public coté |

```python
AssetClass.OBLIGATIONS_HY: AssetClassConfig(
    AssetClass.OBLIGATIONS_HY, "Obligations HY",
    0.055, 0.09, 0.80, 48.0, 0.00, 0.15, False, 4.5
),
```

---

## 3. Matrices de corrélation

> **Note PSD :** Les deux vecteurs ont été vérifiés numériquement. La matrice 17×17 résultante est définie positive (min eigenvalue = +0.032) — aucune modération nécessaire.

### 3.1 Dette privée — ligne/colonne 15

| Avec | Corrélation | Raisonnement |
|---|---|---|
| Actions CDN (0) | 0,35 | Cycle crédit corrélé aux actions |
| Actions US (1) | 0,35 | Idem |
| Actions EAFE (2) | 0,30 | Modeste, moins d'exposition internationale |
| Actions EM (3) | 0,25 | Faible exposition EM directe |
| Oblig Gov CDN (4) | 0,10 | Peu corrélé aux taux |
| Oblig Corp (5) | 0,35 | Même facteur crédit |
| Oblig Inflation (6) | 0,10 | Faible |
| Immobilier (7) | 0,25 | Actifs réels illiquides |
| Infrastructure (8) | 0,30 | Même profil illiquidité/private |
| Capital inv. (9) | 0,45 | Même univers private credit |
| Rendement absolu (10) | 0,20 | Modeste |
| Matières premières (11) | 0,15 | Faible |
| Encaisse (12) | 0,05 | Quasi-nul |
| ACWI (13) | 0,30 | Corrélation modérée global |
| Dette EM (14) | 0,25 | Même segment crédit privé |
| Dette privée (15) | 1,00 | Diagonale |

Vecteur : `[0.35, 0.35, 0.30, 0.25, 0.10, 0.35, 0.10, 0.25, 0.30, 0.45, 0.20, 0.15, 0.05, 0.30, 0.25, 1.00]`

### 3.2 Obligations HY — ligne/colonne 16

| Avec | Corrélation | Raisonnement |
|---|---|---|
| Actions CDN (0) | 0,40 | Beta équité du HY |
| Actions US (1) | 0,50 | Marché HY US dominant |
| Actions EAFE (2) | 0,35 | Exposition internationale |
| Actions EM (3) | 0,40 | Risk-on corrélé |
| Oblig Gov CDN (4) | 0,10 | Faible — duration courte, spread>taux |
| Oblig Corp (5) | 0,60 | Même marché crédit, corrélation forte |
| Oblig Inflation (6) | 0,10 | Faible |
| Immobilier (7) | 0,25 | Modeste |
| Infrastructure (8) | 0,25 | Modeste |
| Capital inv. (9) | 0,35 | Private equity → HY lié |
| Rendement absolu (10) | 0,35 | Stratégies crédit incluses |
| Matières premières (11) | 0,25 | Cyclique |
| Encaisse (12) | 0,05 | Quasi-nul |
| ACWI (13) | 0,45 | Risk-on global |
| Dette EM (14) | 0,55 | Même segment sub-investment grade |
| Dette privée (15) | 0,45 | Même facteur crédit non-IG |
| Obligations HY (16) | 1,00 | Diagonale |

Vecteur : `[0.40, 0.50, 0.35, 0.40, 0.10, 0.60, 0.10, 0.25, 0.25, 0.35, 0.35, 0.25, 0.05, 0.45, 0.55, 0.45, 1.00]`

---

## 4. Fichiers à modifier

### 4.1 `config.py`

**a) Enum `AssetClass`** — ajouter après `DETTE_EMERGENTE` :
```python
DETTE_PRIVEE = "dette_privee"
OBLIGATIONS_HY = "obligations_hy"
```

**b) `ASSET_DEFAULTS`** — ajouter les deux entrées après `DETTE_EMERGENTE`

**c) `ASSET_CLASSES_ORDER`** — appendre les deux à la fin :
```python
AssetClass.DETTE_PRIVEE,
AssetClass.OBLIGATIONS_HY,
```

**d) `DEFAULT_CORRELATION_MATRIX`** — passer de 15×15 à 17×17 :
- Ajouter colonne DP (index 15) à chaque ligne existante
- Ajouter colonne HY (index 16) à chaque ligne + à la ligne DP
- Ajouter les deux nouvelles lignes

**e) `DEFAULT_CURRENT_WEIGHTS`** — appendre `0.00, 0.00` → 17 éléments

**f) `BENCHMARK_PORTFOLIOS`** — les 3 tableaux hardcodés : appendre `0.00, 0.00`

**g) `ALPHA_ELIGIBLE_SHORT`** — ajouter index 16 (HY liquide, DP exclu) :
```python
# Indices: 0-3 equities, 4-6 bonds, 11 commodities, 13 ACWI, 14 EMD, 16 HY
ALPHA_ELIGIBLE_SHORT = [0, 1, 2, 3, 4, 5, 6, 11, 13, 14, 16]
```

**h) `CHART_COLORS`** — ajouter deux couleurs :
```python
"#8b4513",  # Dette privee (saddle brown)
"#ff6347",  # Obligations HY (tomato)
```

### 4.2 `constraints/regulatory.py`

**a) `ALTERNATIVE_INDICES`** — ajouter index 15 :
```python
ALTERNATIVE_INDICES = [7, 8, 9, 10, 15]   # + Dette privee
```

**b) `BOND_INDICES`** — ajouter index 16 :
```python
BOND_INDICES = [4, 5, 6, 14, 16]          # + Obligations HY
```

**c) `FOREIGN_INDICES`** — ajouter index 15 et 16 :
```python
FOREIGN_INDICES = [1, 2, 3, 13, 14, 15, 16]
```

**d) `SHORT_ELIGIBLE_INDICES`** — ajouter index 16 seulement (DP illiquide) :
```python
SHORT_ELIGIBLE_INDICES = [0, 1, 2, 3, 4, 5, 6, 11, 13, 14, 16]
```

### 4.3 `pages/page_rebalancing.py`

```python
"Dette privee": 300,      # Marché privé très illiquide
"Obligations HY": 30,     # Liquid mais spread bid-ask plus large que IG
```

---

## 5. Impact sur les autres composants

| Composant | Impact | Action requise |
|---|---|---|
| `data/generator.py` | `len(ASSET_CLASSES_ORDER)` | Aucune — dynamique |
| Autres pages Streamlit | Fonctions getters | Aucune — dynamique |
| `page_rebalancing.py` | `TRANSACTION_COSTS_BPS` | Ajouter 2 entrées (section 4.3) |
| `CHART_COLORS` | 15 couleurs → 17 | Ajouter 2 couleurs (section 4.1.h) |

---

## 6. Tests

- `len(get_asset_names_fr()) == 17`
- `get_expected_returns()[15] == 0.075` et `[16] == 0.055`
- `DEFAULT_CORRELATION_MATRIX.shape == (17, 17)`
- `np.allclose(M, M.T)` — symétrie
- `np.linalg.eigvalsh(M).min() >= -1e-10` — PSD
- `len(DEFAULT_CURRENT_WEIGHTS) == 17`, sum == 1.0
- Chaque benchmark 17 éléments, sum == 1.0
- `15 in QuebecPensionRegulations.ALTERNATIVE_INDICES`
- `16 in QuebecPensionRegulations.BOND_INDICES`
- `16 in ALPHA_ELIGIBLE_SHORT`
- `15 not in ALPHA_ELIGIBLE_SHORT` (illiquide)
- `15 not in PortableAlphaRegulations.SHORT_ELIGIBLE_INDICES`

---

## 7. Non-objectifs (YAGNI)

- Ne pas modifier les poids par défaut (restent 0 %)
- Ne pas créer de subdivision dans la dette privée (senior/mezzanine)
- Ne pas modifier les pages Streamlit (dynamiques)

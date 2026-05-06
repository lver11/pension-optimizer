# Design : Ajout de la classe d'actifs Actions MSCI ACWI

**Date :** 2026-05-06  
**Projet :** pension_optimizer (Streamlit Cloud)  
**Fichiers modifiés :** `config.py`, `constraints/regulatory.py`

---

## 1. Objectif

Ajouter une 14e classe d'actifs **"Actions MSCI ACWI"** à l'univers existant de 13 classes d'actifs. L'ACWI regroupe les actions mondiales (US, EAFE, EM) en un seul indice diversifié. Cette classe vient **en complément** des classes régionales existantes — elle n'en remplace aucune.

---

## 2. Paramètres financiers

| Paramètre | Valeur | Justification |
|---|---|---|
| Rendement attendu | 7,8 % | Entre US (8 %) et EAFE (7 %) — pondéré par capitalisation |
| Volatilité | 16 % | Légèrement inférieure aux US (17 %) grâce à la diversification |
| Score liquidité | 0,95 | ETF très liquide (ACWI, VT) |
| Score ESG | 58,0 | Moyenne pondérée des marchés développés et émergents |
| Min allocation | 0,0 % | Pas de minimum imposé |
| Max allocation | 40,0 % | Plafond large; contrainte réglementaire sur actions totales s'applique |
| `is_alternative` | False | Actif coté sur marchés publics |
| Index dans `ASSET_CLASSES_ORDER` | **13** (position 0-indexée) |

---

## 3. Matrice de corrélation — nouvelle ligne/colonne (index 13)

Corrélations dérivées de la composition de l'ACWI (≈ 65 % US, 25 % EAFE, 10 % EM) :

| Avec | Corrélation | Raisonnement |
|---|---|---|
| Actions CDN (0) | 0,72 | Forte intégration marchés Nord-Américains |
| Actions US (1) | 0,95 | ACWI est composé à 65 % d'US |
| Actions EAFE (2) | 0,90 | 25 % de l'ACWI |
| Actions Émergentes (3) | 0,82 | 10 % de l'ACWI |
| Oblig Gov CDN (4) | -0,15 | Corrélation négative classique actions/obligations |
| Oblig Corp (5) | -0,05 | Légèrement négative |
| Oblig Inflation (6) | -0,10 | Légèrement négative |
| Immobilier (7) | 0,40 | Correlation modérée avec les actions |
| Infrastructure (8) | 0,30 | Plus défensif, moins corrélé |
| Capital inv. (9) | 0,55 | Corrélation forte avec équités publiques |
| Rendement absolu (10) | 0,35 | Partiellement exposé aux actions |
| Matières premières (11) | 0,30 | Exposé via composante EM/énergie |
| Encaisse (12) | -0,10 | Quasi-nul mais légèrement négatif par substitution |
| ACWI (13) | 1,00 | Diagonale |

Vecteur complet (ligne/colonne 13) :  
`[0.72, 0.95, 0.90, 0.82, -0.15, -0.05, -0.10, 0.40, 0.30, 0.55, 0.35, 0.30, -0.10, 1.00]`

---

## 4. Fichiers à modifier

### 4.1 `config.py`

**a) Enum `AssetClass`** — ajouter :
```python
ACTIONS_ACWI = "actions_acwi"
```

**b) `ASSET_DEFAULTS`** — ajouter l'entrée :
```python
AssetClass.ACTIONS_ACWI: AssetClassConfig(
    AssetClass.ACTIONS_ACWI, "Actions MSCI ACWI",
    0.078, 0.16, 0.95, 58.0, 0.00, 0.40, False
),
```

**c) `ASSET_CLASSES_ORDER`** — appendre à la fin :
```python
AssetClass.ACTIONS_ACWI,
```

**d) `DEFAULT_CORRELATION_MATRIX`** — passer de 13×13 à 14×14 :
- Ajouter la colonne ACWI à chaque ligne existante
- Ajouter la ligne ACWI complète (14 éléments)
- Valeurs de corrélation : voir section 3

**e) `DEFAULT_CURRENT_WEIGHTS`** — appendre `0.0` :
```python
DEFAULT_CURRENT_WEIGHTS = np.array([
    0.12, 0.14, 0.08, 0.05, 0.19, 0.10, 0.05, 0.07, 0.07, 0.05, 0.03, 0.03, 0.02, 0.00,
])
```

**f) `BENCHMARK_PORTFOLIOS`** — les 3 tableaux hardcodés reçoivent un `0.00` supplémentaire :
- `"60_40_equilibre"` : `np.array([0.10, 0.15, 0.08, 0.07, 0.25, 0.15, 0.05, 0.05, 0.05, 0.03, 0.00, 0.02, 0.00, 0.00])`
- `"obligations_pures"` : `np.array([0.00, 0.00, 0.00, 0.00, 0.40, 0.30, 0.25, 0.00, 0.00, 0.00, 0.00, 0.00, 0.05, 0.00])`
- `"croissance_70_30"` : `np.array([0.15, 0.20, 0.10, 0.10, 0.15, 0.10, 0.05, 0.05, 0.05, 0.03, 0.00, 0.02, 0.00, 0.00])`
- `"politique_placement"` : référence `DEFAULT_CURRENT_WEIGHTS.copy()` — se met à jour automatiquement

**g) `ALPHA_ELIGIBLE_SHORT`** — ajouter l'index 13 :
```python
ALPHA_ELIGIBLE_SHORT = [0, 1, 2, 3, 4, 5, 6, 11, 13]
```
(L'ACWI est un ETF très liquide, éligible aux ventes à découvert)

**h) `CHART_COLORS`** — ajouter une 14e couleur (évite que l'ACWI hérite de la même couleur que les Actions CDN dans les graphiques circulaires) :
```python
CHART_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#636efa", "#aec7e8", "#ffbb78",
    "#ffa07a",  # Actions MSCI ACWI
]
```

### 4.2 `constraints/regulatory.py`

**a) `QuebecPensionRegulations.EQUITY_INDICES`** — ajouter index 13 :
```python
EQUITY_INDICES = [0, 1, 2, 3, 13]   # Actions CDN, US, EAFE, Emergentes, ACWI
```

**b) `QuebecPensionRegulations.FOREIGN_INDICES`** — ajouter index 13 :
```python
FOREIGN_INDICES = [1, 2, 3, 13]      # Actions internationales (dont ACWI)
```

**c) `PortableAlphaRegulations.SHORT_ELIGIBLE_INDICES`** — ajouter index 13 :
```python
SHORT_ELIGIBLE_INDICES = [0, 1, 2, 3, 4, 5, 6, 11, 13]
```

### 4.3 `pages/page_rebalancing.py`

Le dict `TRANSACTION_COSTS_BPS` en haut du fichier est indexé par nom d'actif (string). La nouvelle classe d'actifs doit y être ajoutée pour éviter un `KeyError` :
```python
"Actions MSCI ACWI": 10,  # ETF très liquide, coût similaire aux actions US
```

---

## 5. Impact sur les autres composants

| Composant | Impact | Action requise |
|---|---|---|
| `data/generator.py` | Utilise `len(ASSET_CLASSES_ORDER)` | Aucune — dynamique |
| `models/efficient_frontier.py` | Utilise `mu`, `cov_matrix` | Aucune — dynamique |
| `risk/covariance.py` | Opère sur `returns_data` | Aucune — dynamique |
| Autres pages Streamlit | Utilisent `get_asset_names_fr()` etc. | Aucune — fonctions dynamiques |
| `page_rebalancing.py` | `TRANSACTION_COSTS_BPS` hardcodé | Ajouter l'entrée (voir section 4.3) |
| `CHART_COLORS` | 13 couleurs — 14e asset prendrait la couleur 0 | Ajouter une 14e couleur (voir section 4.1.h) |

---

## 6. Tests

- Vérifier `len(get_asset_names_fr()) == 14`
- Vérifier `get_expected_returns()[13] == 0.078`
- Vérifier `DEFAULT_CORRELATION_MATRIX.shape == (14, 14)`
- Vérifier que la matrice est symétrique : `np.allclose(M, M.T)`
- Vérifier `len(DEFAULT_CURRENT_WEIGHTS) == 14` et `sum == 1.0`
- Vérifier que chaque benchmark pèse 1.0
- Vérifier `13 in QuebecPensionRegulations.EQUITY_INDICES`
- Vérifier `13 in PortableAlphaRegulations.SHORT_ELIGIBLE_INDICES`

---

## 7. Non-objectifs (YAGNI)

- Ne pas modifier les poids par défaut du portefeuille actuel (reste 0 %)
- Ne pas migrer l'historique de données (le générateur est synthétique)
- Ne pas modifier les pages Streamlit (elles sont déjà dynamiques)

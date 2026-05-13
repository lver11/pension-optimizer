# Design : Ajout de la classe d'actifs Dette de pays émergents

**Date :** 2026-05-13  
**Projet :** pension_optimizer (Streamlit Cloud)  
**Fichiers modifiés :** `config.py`, `constraints/regulatory.py`, `pages/page_rebalancing.py`

---

## 1. Objectif

Ajouter une 15e classe d'actifs **"Dette pays emergents"** à l'univers existant de 14 classes. Il s'agit d'un mélange hybride 70 % dette en devise forte (JPMorgan EMBI+) / 30 % dette en devise locale (JPMorgan GBI-EM), profil standard institutionnel. Cette classe complète l'exposition aux marchés émergents du côté obligataire, en complément des Actions émergentes déjà présentes à l'index 3.

---

## 2. Paramètres financiers

| Paramètre | Valeur | Justification |
|---|---|---|
| Rendement attendu | 5,8 % | Blended 70 % EMBI / 30 % GBI-EM |
| Volatilité | 11 % | Entre obligations corp (8 %) et actions EM (22 %) |
| Duration | 7,0 ans | EMBI ~6,5 ans + GBI-EM ~5,5 ans pondérés |
| Score liquidité | 0,70 | Liquide mais moins que marchés développés |
| Score ESG | 45,0 | Risque gouvernance souveraine EM |
| Min allocation | 0,0 % | Pas de minimum imposé |
| Max allocation | 15,0 % | Plafond raisonnable pour fonds de pension |
| `is_alternative` | False | Marché public coté |
| Index | **14** (0-indexé) | Après Actions MSCI ACWI |

---

## 3. Matrice de corrélation — nouvelle ligne/colonne (index 14)

| Avec | Corrélation | Raisonnement |
|---|---|---|
| Actions CDN (0) | 0,15 | Exposition modeste au cycle mondial |
| Actions US (1) | 0,10 | Faible — dette obligataire vs équité US |
| Actions EAFE (2) | 0,15 | Faible corrélation avec marchés développés |
| Actions EM (3) | 0,55 | Facteur EM commun dominant |
| Oblig Gov CDN (4) | 0,35 | Sensibilité aux taux (duration commune) |
| Oblig Corp (5) | 0,45 | Corrélation de spread crédit |
| Oblig Inflation (6) | 0,25 | Sensibilité aux taux partiellement commune |
| Immobilier (7) | 0,15 | Modeste |
| Infrastructure (8) | 0,20 | Exposition à l'infrastructure EM |
| Capital inv. (9) | 0,25 | Modeste |
| Rendement absolu (10) | 0,25 | Modeste |
| Matières premières (11) | 0,30 | Pays exportateurs de commodités en EM |
| Encaisse (12) | 0,05 | Quasi-nul |
| Actions MSCI ACWI (13) | 0,40 | Exposition mondiale + facteur EM |
| Dette EM (14) | 1,00 | Diagonale |

Vecteur complet (ligne/colonne 14) :  
`[0.15, 0.10, 0.15, 0.55, 0.35, 0.45, 0.25, 0.15, 0.20, 0.25, 0.25, 0.30, 0.05, 0.40, 1.00]`

> **Note PSD :** Les corrélations seront vérifiées pour la propriété positive semi-définie avant commit. Si `min_eigenvalue < 0`, les valeurs les plus élevées (EM équity: 0,55 ; oblig corp: 0,45) seront réduites progressivement jusqu'à PSD.

---

## 4. Fichiers à modifier

### 4.1 `config.py`

**a) Enum `AssetClass`** — ajouter après `ACTIONS_ACWI` :
```python
DETTE_EMERGENTE = "dette_emergente"
```

**b) `ASSET_DEFAULTS`** — ajouter l'entrée :
```python
AssetClass.DETTE_EMERGENTE: AssetClassConfig(
    AssetClass.DETTE_EMERGENTE, "Dette pays emergents",
    0.058, 0.11, 0.70, 45.0, 0.00, 0.15, False, 7.0
),
```

**c) `ASSET_CLASSES_ORDER`** — appendre à la fin :
```python
AssetClass.DETTE_EMERGENTE,
```

**d) `DEFAULT_CORRELATION_MATRIX`** — passer de 14×14 à 15×15 :
- Ajouter la colonne EMD à chaque ligne existante (valeurs section 3)
- Ajouter la ligne EMD complète (15 éléments)

**e) `DEFAULT_CURRENT_WEIGHTS`** — appendre `0.00` :
```python
DEFAULT_CURRENT_WEIGHTS = np.array([
    0.12, 0.14, 0.08, 0.05, 0.19, 0.10, 0.05, 0.07, 0.07, 0.05, 0.03, 0.03, 0.02, 0.00, 0.00,
])
```

**f) `BENCHMARK_PORTFOLIOS`** — les 3 tableaux hardcodés reçoivent un `0.00` supplémentaire :
- `"60_40_equilibre"` : append `0.00` → 15 éléments
- `"obligations_pures"` : append `0.00` → 15 éléments
- `"croissance_70_30"` : append `0.00` → 15 éléments
- `"politique_placement"` : référence `DEFAULT_CURRENT_WEIGHTS.copy()` — automatique

**g) `ALPHA_ELIGIBLE_SHORT`** — ajouter index 14 :
```python
ALPHA_ELIGIBLE_SHORT = [0, 1, 2, 3, 4, 5, 6, 11, 13, 14]
```
(Liquidité 0,70 ≥ seuil, ETF EMD eligible aux positions courtes)

**h) `CHART_COLORS`** — ajouter une 15e couleur :
```python
"#20b2aa",  # Dette pays emergents (light sea green)
```

### 4.2 `constraints/regulatory.py`

**a) `QuebecPensionRegulations.BOND_INDICES`** — ajouter index 14 :
```python
BOND_INDICES = [4, 5, 6, 14]   # Oblig Gov, Corp, Inflation, Dette EM
```
(La dette émergente compte dans la contrainte "Obligations totales 10–70 %")

**b) `QuebecPensionRegulations.FOREIGN_INDICES`** — ajouter index 14 :
```python
FOREIGN_INDICES = [1, 2, 3, 13, 14]   # Actions intl + ACWI + Dette EM
```

**c) `PortableAlphaRegulations.SHORT_ELIGIBLE_INDICES`** — ajouter index 14 :
```python
SHORT_ELIGIBLE_INDICES = [0, 1, 2, 3, 4, 5, 6, 11, 13, 14]
```

### 4.3 `pages/page_rebalancing.py`

Ajouter dans `TRANSACTION_COSTS_BPS` :
```python
"Dette pays emergents": 25,   # Moins liquide que les marchés développés
```

---

## 5. Impact sur les autres composants

| Composant | Impact | Action requise |
|---|---|---|
| `data/generator.py` | `len(ASSET_CLASSES_ORDER)` | Aucune — dynamique |
| `models/efficient_frontier.py` | `mu`, `cov_matrix` | Aucune — dynamique |
| Autres pages Streamlit | `get_asset_names_fr()` etc. | Aucune — dynamique |
| `page_rebalancing.py` | `TRANSACTION_COSTS_BPS` | Ajouter entrée (section 4.3) |
| `CHART_COLORS` | 14 couleurs → 15 | Ajouter 15e couleur (section 4.1.h) |

---

## 6. Tests

- `len(get_asset_names_fr()) == 15`
- `get_expected_returns()[14] == 0.058`
- `DEFAULT_CORRELATION_MATRIX.shape == (15, 15)`
- `np.allclose(M, M.T)` — symétrie
- `np.linalg.eigvalsh(M).min() >= -1e-10` — PSD
- `len(DEFAULT_CURRENT_WEIGHTS) == 15` et `sum == 1.0`
- Chaque benchmark pèse 1.0
- `14 in QuebecPensionRegulations.BOND_INDICES`
- `14 in QuebecPensionRegulations.FOREIGN_INDICES`
- `14 in PortableAlphaRegulations.SHORT_ELIGIBLE_INDICES`

---

## 7. Non-objectifs (YAGNI)

- Ne pas modifier les poids par défaut du portefeuille actuel (reste 0 %)
- Ne pas créer de classes séparées EMBI et GBI-EM
- Ne pas modifier les pages Streamlit (déjà dynamiques)

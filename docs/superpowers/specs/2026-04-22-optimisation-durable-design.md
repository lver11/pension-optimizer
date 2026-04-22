# Design : Outil d'optimisation durable

**Date :** 2026-04-22
**Projet :** pension_optimizer (Streamlit Cloud)
**Source de données :** `WG_Catégories_actifs_v4.xlsx`

---

## 1. Objectif

Ajouter un outil d'optimisation de portefeuille institutionnel intégrant la durabilité de chaque classe d'actifs, basé sur les données cartographiques du fichier Excel. L'outil génère une **frontière Pareto rendement/risque ↔ durabilité** permettant à l'utilisateur de choisir explicitement son compromis entre performance financière et score de durabilité.

---

## 2. Architecture & navigation

Cinq nouvelles pages dans une section "🌱 Optimisation durable" dans la sidebar de l'app existante. La navigation existante reste intacte. Les deux outils partagent la page *Source de données* pour les rendements et covariances.

### Pages

| Page | Fichier | Rôle |
|------|---------|------|
| Univers durable | `page_durable_univers.py` | Configuration des actifs et pondérations de dimensions |
| Scores de durabilité | `page_durable_scores.py` | Tableau éditable des scores par classe d'actifs |
| Frontière durable | `page_durable_frontier.py` | Frontière Pareto interactive |
| Optimisation durable | `page_durable_optimization.py` | Résultats comparatifs et adoption du portefeuille |
| Rapport durable | `page_durable_rapport.py` | Export résumé pour gouvernance |

### Flux de données

```
Source de données (existant)
    ↓ rendements, covariances (session_state)
Univers durable + Scores de durabilité
    ↓ actifs sélectionnés, scores personnalisés, pondérations (session_state)
Frontière durable → Optimisation durable → Rapport durable
```

---

## 3. Modèle de données

### DurableAsset

```python
@dataclass
class DurableAsset:
    id: str                         # identifiant unique
    nom: str                        # ex. "Obligations univers"
    nom_durable: str                # ex. "Obligations vertes"
    rendement: float                # rendement attendu standard
    volatilite: float
    rendement_durable: float        # rendement variante durable
    volatilite_durable: float
    has_durable_variant: bool       # True si variante durable disponible
    # Scores standard (source: Excel, éditables)
    score_durabilite: float         # 1–4
    score_additionnalite: float
    score_disponibilite: float
    score_retombees_qc: float
    score_liquidite: float
    # Scores variante durable
    score_durabilite_durable: float
    score_additionnalite_durable: float
    score_disponibilite_durable: float
    score_retombees_qc_durable: float
    score_liquidite_durable: float
```

### Score composite du portefeuille

```
S_portefeuille = Σ wᵢ × [α×durabilité + β×additionnalité + γ×retombées_qc + δ×disponibilité + ε×liquidité]ᵢ
```

où α+β+γ+δ+ε = 1 (pondérations définies par l'utilisateur).

### Clés session_state

| Clé | Type | Contenu |
|-----|------|---------|
| `durable_universe` | dict | actif → `{variant: "standard"\|"durable", active: bool}` |
| `durable_scores` | DataFrame | une ligne par actif, 5 colonnes de scores éditables |
| `durable_dim_weights` | dict | `{durabilite: float, additionnalite: float, ...}` |
| `durable_gamma` | float | aversion au risque (défaut: 2.5) |
| `durable_lambda` | float | poids durabilité dans l'objectif |
| `durable_result` | DurableResult | résultat de l'optimisation courante |
| `durable_frontier` | List[DurableResult] | points de la frontière Pareto |

### Nouveaux fichiers

```
pension_optimizer/
├── sustainable/
│   ├── __init__.py
│   ├── config.py           ← 18 actifs avec scores Excel
│   └── optimizer.py        ← DurableOptimizer + DurableResult
└── pages/
    ├── page_durable_univers.py
    ├── page_durable_scores.py
    ├── page_durable_frontier.py
    ├── page_durable_optimization.py
    └── page_durable_rapport.py
```

---

## 4. Modèle d'optimisation

### Objectif bi-critère (convexe)

```
Maximiser :  μ'w  −  (γ/2) × w'Σw  +  λ × S'w

Sous contraintes :
  Σ wᵢ = 1
  wᵢ ≥ min_wᵢ,  wᵢ ≤ max_wᵢ
  (contraintes de groupe optionnelles)
```

- `μ` = rendements attendus des actifs sélectionnés
- `Σ` = matrice de covariance (depuis Source de données)
- `S` = vecteur des scores composites de durabilité
- `γ` = aversion au risque (défaut 2.5, ajustable)
- `λ` = poids durabilité (0 = purement financier, croissant = plus durable)

### Frontière Pareto

Varier `λ` de 0 à `λ_max` en N=50 points. Chaque point donne un portefeuille optimal → paire `(score_durabilité, sharpe)`. La courbe résultante est la frontière Pareto.

### DurableOptimizer

```python
class DurableOptimizer(BaseOptimizer):
    def optimize_durable(
        self, lam: float, gamma: float,
        sustainability_scores: np.ndarray,
        constraint_set=None
    ) -> DurableResult: ...

    def pareto_frontier(
        self, n_points: int = 50, gamma: float = 2.5,
        sustainability_scores: np.ndarray = None,
        constraint_set=None
    ) -> List[DurableResult]: ...
```

### DurableResult

Étend `OptimizationResult` avec :
- `sustainability_score: float` — score composite du portefeuille
- `sustainability_breakdown: dict` — contribution par dimension
- `lambda_used: float`
- `variant_used: dict` — actif → standard/durable

---

## 5. Détail des pages

### Univers durable

- Tableau : une ligne par actif, toggle "Variante durable" (si disponible), checkbox "Inclure"
- Curseurs pour pondérer les 5 dimensions (α, β, γ, δ, ε), normalisés à 100%
- Graphique radar des priorités de durabilité choisies
- Paramètre `γ` (aversion au risque)
- Bouton "Appliquer"

### Scores de durabilité

- `st.data_editor` : une ligne par actif actif, 5 colonnes de scores éditables
- Valeurs par défaut issues de `WG_Catégories_actifs_v4.xlsx` (Cartographie_complète_v1)
- Colonne calculée "Score composite" mise à jour en temps réel
- Graphique à barres horizontales pour comparaison visuelle
- Boutons : "Réinitialiser aux valeurs Excel" / "Appliquer"

### Frontière durable

- Scatter plot interactif : X = score durabilité, Y = ratio de Sharpe
- Chaque point = un λ différent (coloré selon λ)
- Curseur λ pour sélectionner un point sur la frontière
- Tooltip : allocation complète, rendement, volatilité, score
- Portefeuille actuel affiché comme point de référence (étoile)
- Bouton "Utiliser ce point" → pré-remplit la page Optimisation

### Optimisation durable

- λ pré-rempli depuis la frontière (ajustable manuellement)
- Résultats en 3 colonnes : Actuel / Optimal financier (λ=0) / Optimal durable (λ choisi)
- Graphique comparatif des allocations (barres groupées)
- Décomposition du score durabilité par dimension (graphique en pile)
- Bouton "Adopter comme portefeuille actuel"

### Rapport durable

- Sections : métriques financières, score de durabilité avec décomposition, allocation détaillée, hypothèses (pondérations, γ, λ)
- Export CSV de l'allocation
- Tableau récapitulatif prêt pour présentation au conseil

---

## 6. Gestion des erreurs

- Si l'univers sélectionné est vide : message d'erreur explicite
- Si l'optimisation est infaisable (contraintes trop strictes) : fallback sur portefeuille équipondéré avec avertissement
- Si `λ_max` ne produit pas de différence visible : avertissement "Les contraintes limitent la diversification durable"
- Scores manquants dans l'éditeur : validation avant application (0 < score ≤ 5, échelle Excel)

---

## 7. Tests

- Vérifier que λ=0 reproduit le portefeuille Markowitz standard
- Vérifier que score composite = moyenne pondérée des scores actifs
- Vérifier que la frontière est monotone (score durabilité croissant avec λ)
- Vérifier que les variantes durables ont des scores supérieurs aux standards

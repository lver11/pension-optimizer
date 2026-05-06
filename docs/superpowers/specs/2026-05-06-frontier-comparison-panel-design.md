# Design : Panneau de comparaison sur la frontière efficiente

**Date :** 2026-05-06
**Projet :** pension_optimizer (Streamlit Cloud)
**Fichier modifié :** `pages/page_frontier.py`

---

## 1. Objectif

Ajouter un panneau de comparaison côte à côte sur la page Frontière efficiente, permettant à l'utilisateur de voir simultanément les métriques et l'allocation du **portefeuille actuel** et du **point sélectionné sur la frontière**, avec les deltas entre les deux.

---

## 2. Layout

```
[Graphique frontière efficiente — inchangé]
[Curseur : rendement cible — inchangé]

┌─────────────────────────┬─────────────────────────┐
│  📍 Portefeuille actuel │  ★ Point sélectionné    │
│─────────────────────────│─────────────────────────│
│  Rendement :  X.XX%     │  X.XX%  (+Δ pp)         │
│  Volatilité : X.XX%     │  X.XX%  (−Δ pp)         │
│  Sharpe :     X.XXX     │  X.XXX  (+Δ)            │
│  VaR 95% :    X.XX%     │  X.XX%  (−Δ pp)         │
│  CVaR 95% :   X.XX%     │  X.XX%  (−Δ pp)         │
│─────────────────────────│─────────────────────────│
│  [Donut allocation]     │  [Donut allocation]     │
└─────────────────────────┴─────────────────────────┘

[Barres horizontales groupées — toutes les classes d'actifs]
[Tableau : actif / poids actuel / poids sélectionné / écart pp]
```

---

## 3. Composants

### 3.1 Métriques (5 par colonne)

Sources dans la page :
- `mu` : `get_expected_returns()` (déjà importé)
- `cov_matrix` : `CovarianceEstimator.estimate(returns_data, method)` (déjà calculé dans la page)
- `rf` : `st.session_state.get("pension_config", PensionFundConfig()).taux_sans_risque`
- `asset_names` : `get_asset_names_fr()` (déjà importé)
- `returns_data` : `st.session_state.returns_data`

| Métrique | Source actuel | Source sélectionné |
|----------|--------------|-------------------|
| Rendement attendu | `current_weights @ mu` | `selected_point['return']` |
| Volatilité | `sqrt(current_weights @ cov_matrix @ current_weights)` | `selected_point['volatility']` |
| Ratio de Sharpe | `(ret - rf) / vol` | `selected_point['sharpe']` |
| VaR 95% | `RiskMetrics.compute_all(returns_data.values @ current_weights)['VaR (historique)']` | idem avec `w_selected` |
| CVaR 95% | `RiskMetrics.compute_all(...)['CVaR']` | idem avec `w_selected` |

Les deltas sont affichés via `st.metric(label, value, delta=delta_val, delta_color=color)`.

**Coloration des deltas** :
- Rendement, Sharpe : `delta_color="normal"` (vert si positif, rouge si négatif)
- Volatilité, VaR, CVaR : `delta_color="inverse"` (vert si négatif = baisse = amélioration, rouge si positif)

Exemple d'appel :
```python
st.metric("Volatilité", f"{vol_sel:.2%}", delta=f"{vol_sel - vol_cur:+.2%}", delta_color="inverse")
```

### 3.2 Graphiques d'allocation

- Colonne gauche : `ChartBuilder.allocation_pie(current_weights, asset_names, "Actuel")`
- Colonne droite : `ChartBuilder.allocation_pie(w_selected, asset_names, "Point sélectionné")`

### 3.3 Barres horizontales groupées

`ChartBuilder.allocation_comparison_bar(current_weights, w_selected, asset_names)` — déjà disponible dans `visualization/charts.py`.

### 3.4 Tableau détaillé

DataFrame avec colonnes : `Classe d'actifs`, `Actuel (%)`, `Sélectionné (%)`, `Écart (pp)`.
Formatage : `{:+.1f}` pour l'écart, fond coloré selon direction.

---

## 4. Extraction des poids du point sélectionné

```python
weight_cols = [c for c in frontier_df.columns if c.startswith("w_")]
if weight_cols:
    w_selected = selected_point[weight_cols].values
    w_selected = w_selected / w_selected.sum()  # renormaliser au cas où
else:
    w_selected = None  # afficher message si pas de colonnes de poids
```

---

## 5. Gestion des erreurs

- Si `frontier_df` ne contient pas de colonnes `w_*` : afficher uniquement les métriques, masquer les graphiques d'allocation avec un `st.caption("Allocation non disponible pour ce type de frontière.")`
- Si `returns_data` absent : calculer VaR/CVaR par approximation gaussienne :
  `VaR ≈ vol × norm.ppf(0.95)` et `CVaR ≈ vol × norm.pdf(norm.ppf(0.05)) / 0.05`
  (importer `from scipy.stats import norm`)
- Si les poids sélectionnés sont nuls ou invalides : ne pas afficher le panneau

---

## 6. Position dans la page

Le panneau est inséré **après** le curseur de sélection et **avant** le tableau de la frontière existant (dans l'expander). Il remplace l'affichage actuel du donut seul pour le point sélectionné.

---

## 7. Tests

- Vérifier que le panneau s'affiche uniquement quand `frontier_data` est dans `session_state`
- Vérifier que les deltas sont correctement signés (baisse de vol = delta négatif = couleur verte)
- Vérifier que la renormalisation des poids ne provoque pas de division par zéro
- Vérifier le fallback si pas de colonnes `w_*`

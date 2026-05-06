# Frontier Comparison Panel Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ajouter un panneau de comparaison côte à côte (portefeuille actuel vs point sélectionné sur la frontière) avec métriques de risque/rendement, graphiques d'allocation et tableau détaillé.

**Architecture:** Une fonction pure `compute_portfolio_metrics()` est ajoutée à `risk/metrics.py` pour centraliser le calcul des 5 métriques (rendement, volatilité, Sharpe, VaR, CVaR). La page `pages/page_frontier.py` utilise cette fonction pour alimenter le panneau de comparaison côte à côte, qui remplace l'unique donut existant du point sélectionné.

**Tech Stack:** Python 3.9+, Streamlit, NumPy, SciPy (scipy.stats), CVXPY (via modèles existants), Plotly (via ChartBuilder)

---

## Chunk 1 : Helper function + panneau de comparaison

### Task 1 : Ajouter `compute_portfolio_metrics` dans `risk/metrics.py`

**Files:**
- Modify: `risk/metrics.py` (ajouter après la classe `RiskMetrics`, à la fin du fichier)
- Create: `tests/test_frontier_comparison.py`

**Contexte :** `risk/metrics.py` contient la classe `RiskMetrics` avec `compute_all()`. On ajoute une fonction module-level (pas une méthode de classe) qui prend des poids et retourne un dict de 5 métriques. Cette fonction sera importée par `page_frontier.py`.

La clé `"VaR (historique)"` est le nom exact retourné par `RiskMetrics.compute_all()` (visible dans le code existant). La clé `"CVaR"` est également exacte.

- [ ] **Step 1 : Écrire les tests (ils doivent échouer)**

Créer `tests/test_frontier_comparison.py` :

```python
"""Tests pour compute_portfolio_metrics (risk/metrics.py)."""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from risk.metrics import compute_portfolio_metrics


def _simple_setup():
    """Retourne mu, cov, rf, weights simples pour les tests."""
    n = 3
    mu = np.array([0.08, 0.04, 0.02])
    cov = np.diag([0.16**2, 0.08**2, 0.01**2])  # Pas de corrélation
    rf = 0.025
    w = np.array([0.5, 0.3, 0.2])
    return mu, cov, rf, w


def test_rendement_correct():
    mu, cov, rf, w = _simple_setup()
    m = compute_portfolio_metrics(w, mu, cov, rf)
    expected = float(w @ mu)
    assert abs(m["rendement"] - expected) < 1e-10


def test_volatilite_correct():
    mu, cov, rf, w = _simple_setup()
    m = compute_portfolio_metrics(w, mu, cov, rf)
    expected = float(np.sqrt(w @ cov @ w))
    assert abs(m["volatilite"] - expected) < 1e-10


def test_sharpe_correct():
    mu, cov, rf, w = _simple_setup()
    m = compute_portfolio_metrics(w, mu, cov, rf)
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ cov @ w))
    expected = (ret - rf) / vol
    assert abs(m["sharpe"] - expected) < 1e-10


def test_returns_data_none_gaussian_fallback():
    """Sans returns_data, VaR et CVaR doivent être calculées en Gaussien."""
    mu, cov, rf, w = _simple_setup()
    m = compute_portfolio_metrics(w, mu, cov, rf, returns_data=None)
    # VaR gaussienne = vol * norm.ppf(0.95)
    from scipy.stats import norm
    vol = float(np.sqrt(w @ cov @ w))
    expected_var = vol * norm.ppf(0.95)
    assert abs(m["var_95"] - expected_var) < 1e-8
    assert m["cvar_95"] > m["var_95"]  # CVaR >= VaR toujours


def test_returns_data_historical():
    """Avec returns_data, VaR et CVaR sont calculées historiquement."""
    mu, cov, rf, w = _simple_setup()
    np.random.seed(42)
    # Simuler 240 mois de rendements pour 3 actifs
    returns_data = pd.DataFrame(
        np.random.randn(240, 3) * [0.16, 0.08, 0.01] / np.sqrt(12),
        columns=["A", "B", "C"],
    )
    m = compute_portfolio_metrics(w, mu, cov, rf, returns_data=returns_data)
    assert m["var_95"] > 0
    assert m["cvar_95"] >= m["var_95"]


def test_all_keys_present():
    mu, cov, rf, w = _simple_setup()
    m = compute_portfolio_metrics(w, mu, cov, rf)
    for key in ("rendement", "volatilite", "sharpe", "var_95", "cvar_95"):
        assert key in m, f"Clé manquante: {key}"


def test_zero_vol_sharpe():
    """Volatilité nulle → Sharpe = 0.0 (pas de division par zéro)."""
    n = 2
    mu = np.array([0.05, 0.05])
    cov = np.zeros((2, 2))  # Vol nulle
    rf = 0.025
    w = np.array([0.5, 0.5])
    m = compute_portfolio_metrics(w, mu, cov, rf)
    assert m["sharpe"] == 0.0
```

- [ ] **Step 2 : Vérifier que les tests échouent**

```bash
cd /chemin/vers/pension_optimizer
python3 -m pytest tests/test_frontier_comparison.py -v 2>&1 | head -30
```

Résultat attendu : `ImportError: cannot import name 'compute_portfolio_metrics'`

- [ ] **Step 3 : Ajouter `compute_portfolio_metrics` à `risk/metrics.py`**

Ajouter à la **fin** de `risk/metrics.py`, après la classe `RiskMetrics` :

```python

def compute_portfolio_metrics(
    weights: np.ndarray,
    mu: np.ndarray,
    cov_matrix: np.ndarray,
    rf: float,
    returns_data=None,
) -> dict:
    """Calcule les métriques clés d'un portefeuille.

    Args:
        weights: Vecteur de poids (somme = 1).
        mu: Rendements attendus annualisés.
        cov_matrix: Matrice de covariance annualisée.
        rf: Taux sans risque annualisé.
        returns_data: DataFrame ou ndarray (T x N) de rendements historiques.
                      Si None, VaR et CVaR sont approximées par un modèle gaussien.

    Returns:
        Dict avec clés : rendement, volatilite, sharpe, var_95, cvar_95.
    """
    weights = np.asarray(weights, dtype=float)
    ret = float(weights @ mu)
    var_port = float(weights @ cov_matrix @ weights)
    vol = float(np.sqrt(max(var_port, 0.0)))
    sharpe = (ret - rf) / vol if vol > 1e-10 else 0.0

    if returns_data is not None:
        port_returns = np.asarray(returns_data) @ weights
        hist_metrics = RiskMetrics.compute_all(port_returns, rf)
        var_95 = float(hist_metrics.get("VaR (historique)", 0.0))
        cvar_95 = float(hist_metrics.get("CVaR", 0.0))
    else:
        from scipy.stats import norm as _norm
        var_95 = float(vol * _norm.ppf(0.95))
        cvar_95 = float(vol * _norm.pdf(_norm.ppf(0.05)) / 0.05)

    return {
        "rendement": ret,
        "volatilite": vol,
        "sharpe": sharpe,
        "var_95": var_95,
        "cvar_95": cvar_95,
    }
```

- [ ] **Step 4 : Vérifier que les tests passent**

```bash
python3 -m pytest tests/test_frontier_comparison.py -v
```

Résultat attendu : `7 passed`

- [ ] **Step 5 : Commit**

```bash
git add risk/metrics.py tests/test_frontier_comparison.py
git commit -m "feat(frontier): add compute_portfolio_metrics helper + tests"
```

---

### Task 2 : Panneau de comparaison dans `pages/page_frontier.py`

**Files:**
- Modify: `pages/page_frontier.py`

**Contexte :** La page actuelle affiche le point sélectionné avec un donut seul (lignes ~177-183) et une rangée de 3 métriques juste avant (lignes ~172-175). On remplace **les deux blocs** par le panneau complet. Les fonctions disponibles dans la page :
- `ChartBuilder.allocation_pie(weights, names, title)` → donut Plotly
- `ChartBuilder.allocation_comparison_bar(w1, w2, names)` → barres groupées
- `asset_names` = `get_asset_names_fr()` (déjà dans `render()`). Les colonnes `w_*` du frontier_df sont nommées `w_{nom_fr}` — ordre identique à `asset_names`, donc `len(weight_cols) == len(asset_names)` dans le cas normal.
- `current_weights` = `st.session_state.get("current_weights", DEFAULT_CURRENT_WEIGHTS)`
- `mu` = `get_expected_returns()` (déjà dans `render()`)
- `cov_matrix` = déjà calculé dans `render()`
- `rf` = déjà dans `render()`
- `returns_data` = `st.session_state.returns_data`

**Import à ajouter** en haut de `page_frontier.py` :
```python
from risk.metrics import compute_portfolio_metrics
```

- [ ] **Step 1 : Ajouter l'import dans `page_frontier.py`**

Dans les imports existants de `page_frontier.py` (après `from risk.covariance import CovarianceEstimator`), ajouter :

```python
from risk.metrics import compute_portfolio_metrics
```

- [ ] **Step 2 : Remplacer les deux blocs du point sélectionné**

Localiser et remplacer le bloc suivant dans `render()` (les deux blocs consécutifs — métriques 3-colonnes + donut) :

```python
            col1, col2, col3 = st.columns(3)
            col1.metric("Rendement", f"{selected_point['return']:.2%}")
            col2.metric("Volatilite", f"{selected_point['volatility']:.2%}")
            col3.metric("Sharpe", f"{selected_point.get('sharpe', 0):.3f}")

            # Poids du point selectionne
            weight_cols = [c for c in frontier_df.columns if c.startswith("w_")]
            if weight_cols:
                w_vals = selected_point[weight_cols].values
                w_names = [c.replace("w_", "") for c in weight_cols]
                fig_bar = ChartBuilder.allocation_pie(w_vals, w_names, "Allocation du point selectionne")
                st.plotly_chart(fig_bar, use_container_width=True)
```

Par le bloc suivant :

```python
            # --- Panneau de comparaison ---
            st.markdown("### Comparaison : portefeuille actuel vs point selectionne")

            # Extraire les poids du point sélectionné
            weight_cols = [c for c in frontier_df.columns if c.startswith("w_")]
            w_selected = None
            if weight_cols and len(weight_cols) == len(asset_names):
                w_raw = selected_point[weight_cols].values.astype(float)
                w_sum = w_raw.sum()
                if w_sum > 1e-10:
                    w_selected = w_raw / w_sum

            # Métriques des deux portefeuilles
            returns_data = st.session_state.get("returns_data")
            m_cur = compute_portfolio_metrics(current_weights, mu, cov_matrix, rf, returns_data)
            w_for_sel = w_selected if w_selected is not None else current_weights
            m_sel = compute_portfolio_metrics(w_for_sel, mu, cov_matrix, rf, returns_data)

            col_cur, col_sel = st.columns(2)

            with col_cur:
                st.markdown("**📍 Portefeuille actuel**")
                st.metric("Rendement attendu", f"{m_cur['rendement']:.2%}")
                st.metric("Volatilite", f"{m_cur['volatilite']:.2%}")
                st.metric("Ratio de Sharpe", f"{m_cur['sharpe']:.3f}")
                st.metric("VaR 95%", f"{m_cur['var_95']:.2%}")
                st.metric("CVaR 95%", f"{m_cur['cvar_95']:.2%}")
                fig_cur = ChartBuilder.allocation_pie(
                    current_weights, asset_names, "Allocation actuelle"
                )
                st.plotly_chart(fig_cur, use_container_width=True)

            with col_sel:
                st.markdown("**★ Point selectionne**")
                st.metric(
                    "Rendement attendu", f"{m_sel['rendement']:.2%}",
                    delta=f"{m_sel['rendement'] - m_cur['rendement']:+.2%}",
                    delta_color="normal",
                )
                st.metric(
                    "Volatilite", f"{m_sel['volatilite']:.2%}",
                    delta=f"{m_sel['volatilite'] - m_cur['volatilite']:+.2%}",
                    delta_color="inverse",
                )
                st.metric(
                    "Ratio de Sharpe", f"{m_sel['sharpe']:.3f}",
                    delta=f"{m_sel['sharpe'] - m_cur['sharpe']:+.3f}",
                    delta_color="normal",
                )
                st.metric(
                    "VaR 95%", f"{m_sel['var_95']:.2%}",
                    delta=f"{m_sel['var_95'] - m_cur['var_95']:+.2%}",
                    delta_color="inverse",
                )
                st.metric(
                    "CVaR 95%", f"{m_sel['cvar_95']:.2%}",
                    delta=f"{m_sel['cvar_95'] - m_cur['cvar_95']:+.2%}",
                    delta_color="inverse",
                )
                if w_selected is not None:
                    fig_sel = ChartBuilder.allocation_pie(
                        w_selected, asset_names, "Allocation selectionnee"
                    )
                    st.plotly_chart(fig_sel, use_container_width=True)
                else:
                    st.caption("Allocation non disponible pour ce type de frontiere.")

            # Barres groupées + tableau si poids disponibles
            if w_selected is not None:
                st.markdown("#### Comparaison des allocations")
                fig_comp = ChartBuilder.allocation_comparison_bar(
                    current_weights, w_selected, asset_names
                )
                st.plotly_chart(fig_comp, use_container_width=True)

                ecarts = w_selected - current_weights
                comp_df = pd.DataFrame({
                    "Classe d'actifs": asset_names,
                    "Actuel (%)": current_weights * 100,
                    "Selectionne (%)": w_selected * 100,
                    "Ecart (pp)": ecarts * 100,
                })
                st.dataframe(
                    comp_df.style.format({
                        "Actuel (%)": "{:.1f}",
                        "Selectionne (%)": "{:.1f}",
                        "Ecart (pp)": "{:+.1f}",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )
```

- [ ] **Step 3 : Vérifier que tous les tests passent encore**

```bash
python3 -m pytest tests/test_frontier_comparison.py tests/sustainable/ -v 2>&1 | tail -10
```

Résultat attendu : tous les tests passent (17 sustainable + 7 frontier = 24 total).

- [ ] **Step 4 : Vérifier que la page s'importe sans erreur**

```bash
python3 -c "
import sys; sys.path.insert(0, '.')
# Vérifier les imports de la page sans lancer Streamlit
from risk.metrics import compute_portfolio_metrics
from visualization.charts import ChartBuilder
import numpy as np
w = np.array([0.5, 0.3, 0.2])
mu = np.array([0.08, 0.04, 0.02])
cov = np.diag([0.16**2, 0.08**2, 0.01**2])
m = compute_portfolio_metrics(w, mu, cov, 0.025)
print('compute_portfolio_metrics OK:', m)
print('ChartBuilder OK')
"
```

Résultat attendu : affichage du dict de métriques sans erreur.

- [ ] **Step 5 : Commit et push**

```bash
git add pages/page_frontier.py
git commit -m "feat(frontier): add side-by-side comparison panel with risk metrics"
git push origin main
```

---

## Checklist de validation finale

- [ ] `python3 -m pytest tests/test_frontier_comparison.py -v` → 7 passed
- [ ] `python3 -m pytest tests/sustainable/ tests/test_frontier_comparison.py -v` → tous les tests passent (régression 0 failure)
- [ ] La page Frontière efficiente charge sans erreur sur Streamlit Cloud
- [ ] Le panneau affiche bien les métriques du portefeuille actuel sans cliquer (côté gauche statique)
- [ ] Les deltas sont verts/rouges correctement (baisse vol = vert)
- [ ] Si frontier type = Moyenne-CVaR et pas de colonnes w_* : message "Allocation non disponible" affiché

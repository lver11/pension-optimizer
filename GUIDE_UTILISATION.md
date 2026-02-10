# Guide d'utilisation - Optimiseur de Portefeuille Institutionnel

## Caisse de Retraite - Application Streamlit

---

## 1. Demarrage

### Lancement local

```bash
cd pension_optimizer
python3 -m streamlit run app.py
```

L'application s'ouvre dans le navigateur a l'adresse `http://localhost:8501`.

### Version en ligne

L'application est deployee sur Streamlit Community Cloud.
Le code source est sur GitHub : `lver11/pension-optimizer`.

### Prerequis techniques

- Python 3.9+
- Dependances : streamlit, numpy, pandas, scipy, cvxpy, plotly, scikit-learn, openpyxl, xlsxwriter, fpdf2

---

## 2. Configuration globale (sidebar)

La sidebar gauche est presente sur **toutes les pages**. Elle contient :

| Parametre | Description | Defaut |
|-----------|-------------|--------|
| Valeur de l'actif (M$) | Valeur marchande totale des actifs du fonds | 1 000 M$ |
| Valeur du passif (M$) | Valeur actualisee des obligations futures | 950 M$ |
| Taux sans risque (%) | Taux de reference pour le calcul des ratios | 2.5% |
| Horizon (annees) | Horizon de placement du fonds | 20 ans |

**Indicateur de capitalisation** : affiche en temps reel le ratio actif/passif avec un code couleur :
- Vert (>= 100%) : capitalisation integrale
- Jaune (85-100%) : surveillance
- Rouge (< 85%) : situation critique

### Source de donnees

- **Simulees** : genere des rendements synthetiques a partir d'une graine aleatoire (defaut : 42, 20 ans, frequence mensuelle). Bouton "Regenerer les donnees" pour creer un nouvel echantillon.
- **Importees** : utilisez la page Rapports pour importer vos propres donnees historiques (CSV ou Excel).

---

## 3. Pages de l'application

L'application comporte 10 pages organisees en 5 sections.

---

### 3.1 Vue d'ensemble - Tableau de bord

**Objectif** : vision synthetique de l'etat actuel du portefeuille.

**Indicateurs cles (ligne 1)** :
- Ratio de capitalisation (actif / passif)
- Rendement annualise du portefeuille
- Volatilite annualisee
- Ratio de Sharpe
- VaR a 95%

**Indicateurs cles (ligne 2)** :
- Valeur de l'actif et du passif (M$)
- Surplus ou deficit (M$)
- CVaR a 95%
- Perte maximale historique (drawdown)

**4 onglets** :

1. **Allocation et risque** : diagramme en anneau de l'allocation actuelle + barres horizontales des contributions au risque + tableau detaille par classe d'actifs (poids, rendement attendu, volatilite, score de liquidite, contribution au risque).

2. **Performance** : courbes de rendements cumules par classe d'actifs, rendement cumule du portefeuille, histogramme de la distribution des rendements avec lignes VaR/CVaR.

3. **Correlations** : matrice de correlation empirique (calculee sur les donnees historiques) + matrice theorique (en accordeon).

4. **Metriques detaillees** : tableau complet de toutes les metriques de risque, 24 derniers mois de rendements (codes par couleur), statistiques par classe d'actifs.

---

### 3.2 Optimisation - Moteur d'optimisation

**Objectif** : trouver l'allocation optimale selon differents modeles.

**Etape 1 - Choisir le modele** :

| Modele | Description | Quand l'utiliser |
|--------|-------------|------------------|
| Moyenne-Variance (Markowitz) | Optimise le couple rendement/risque sur la frontiere efficiente | Point de depart standard, hypothese de normalite |
| Black-Litterman | Combine les rendements d'equilibre avec vos vues d'investisseur | Vous avez des convictions sur certaines classes d'actifs |
| Parite de risque | Egalise la contribution au risque de chaque actif | Diversification maximale, pas de prevision de rendements |
| CVaR (Rockafellar-Uryasev) | Minimise les pertes extremes (queue gauche) | Focus sur la protection en cas de crise |

**Etape 2 - Configurer les parametres** :

- **Methode de covariance** : Ledoit-Wolf (recommande, regularise), Sample (classique), EWMA (reactive aux donnees recentes)
- **Contraintes reglementaires** : cocher pour appliquer les limites du Quebec (actions <= 70%, alternatives <= 40%, etc.)
- Parametres specifiques selon le modele choisi (voir ci-dessous)

**Parametres par modele** :

*Markowitz* :
- Objectif : maximiser Sharpe, minimiser variance, rendement cible, risque cible
- Curseur de rendement ou risque cible selon l'objectif

*Black-Litterman* :
- Aversion au risque (delta) : controle l'agressivite des rendements d'equilibre
- Incertitude (tau) : poids relatif des vues vs l'equilibre (plus petit = plus de poids a l'equilibre)
- Vues d'investisseur (jusqu'a 5) : selectionnez un actif, un type (absolue ou relative), le rendement attendu et votre niveau de confiance

*Parite de risque* :
- Option de budgets de risque personnalises (sinon equibudget 1/n)

*CVaR* :
- Objectif : minimiser CVaR, rendement cible, rendement max pour CVaR cible
- Niveau de confiance (90-99%)

**Etape 3 - Lancer l'optimisation** : cliquez sur "Lancer l'optimisation"

**Resultats** :
- Message de succes avec temps de resolution, rendement, volatilite, Sharpe
- Comparaison avant/apres : tableau et barres groupees (allocation actuelle vs optimisee)
- Diagramme en anneau de la nouvelle allocation
- Contributions au risque du portefeuille optimise
- Bouton **"Adopter le portefeuille optimise"** : remplace l'allocation actuelle dans toute l'application

---

### 3.3 Optimisation - Gestionnaire de contraintes

**Objectif** : definir les bornes et regles que l'optimiseur doit respecter.

**Bornes individuelles** :
- Pour chaque classe d'actifs (12 au total), definir un poids minimum et maximum
- Exemple : Actions canadiennes entre 5% et 30%

**Contraintes de groupe** :
- Actions totales <= 70%
- Actifs alternatifs <= 40%
- Capital investissement <= 20%
- Obligations >= 10%

**Contraintes reglementaires du Quebec** : active les limites legales en un clic (voir le lexique pour les details).

**Contraintes ESG** :
- Score ESG minimum du portefeuille (0-100)
- Affiche le score ESG actuel et l'intensite carbone

**Contraintes supplementaires** :
- Liquidite minimale : pourcentage minimum d'actifs liquides
- Rotation maximale : limite le volume de transactions

**Validation** : le bouton "Sauvegarder et valider" verifie que l'allocation actuelle respecte toutes les contraintes definies et signale les violations.

---

### 3.4 Optimisation - Frontiere efficiente

**Objectif** : visualiser l'ensemble des portefeuilles optimaux possibles.

**Configuration** :
- Type de frontiere : Moyenne-Variance ou Moyenne-CVaR
- Nombre de points (20 a 100) : precision de la courbe
- Methode de covariance
- Contraintes reglementaires (oui/non)

**Options d'affichage** :
- Portefeuille actuel (losange rouge)
- Portefeuille tangent (etoile doree = Sharpe maximal)
- Frontiere non contrainte (pour visualiser le cout des contraintes)
- Ligne du marche des capitaux (CML)

**Interaction** :
- Curseur de rendement cible pour selectionner un point sur la frontiere
- Affiche les metriques et l'allocation du point selectionne
- Diagramme en anneau de l'allocation du point choisi
- Tableau complet de tous les points de la frontiere (rendement, volatilite, Sharpe, poids)

---

### 3.5 Analyse de risque - Analytique de risque

**Objectif** : analyse approfondie du profil de risque du portefeuille.

**Metriques cles (8)** : VaR 95%, CVaR 95%, Sharpe, drawdown max, Sortino, Calmar, Omega, asymetrie

**Onglet 1 - Distribution et metriques** :
- Histogramme des rendements avec lignes VaR et CVaR
- Tableau complet de toutes les metriques (14+ indicateurs)

**Onglet 2 - Tests de tension historiques** :
- 6 scenarios predefinis : Crise 2008, COVID 2020, Hausse taux 2022, Bulle technologique 2000, Crise dette euro 2011, Stagflation
- Tableau recapitulatif (impact en % et en M$)
- Diagramme en cascade de l'impact
- Detail par classe d'actifs (choc applique et contribution a la perte)
- Impact sur le ratio de capitalisation

**Onglet 3 - Tests de tension parametriques** :
- Definir vos propres chocs : actions (-50% a +10%), taux (-200 a +300 bps), spreads (-50 a +300 bps), inflation (-2% a +5%)
- Resultat : impact total, perte en M$, detail par actif
- **Test de tension inverse** : trouvez le choc minimal necessaire pour provoquer une perte donnee (ex: -15%)

**Onglet 4 - Analyse du drawdown** :
- Graphique temporel du drawdown ("sous l'eau")
- Perte maximale, date du pic et du creux

---

### 3.6 Analyse de risque - Simulation Monte Carlo

**Objectif** : projeter l'evolution du fonds sur un horizon de plusieurs annees en tenant compte de l'incertitude.

**Parametres** :
- Horizon (5-40 ans), nombre de simulations (1 000 a 25 000)
- Valeur initiale de l'actif et du passif
- Cotisations annuelles et prestations annuelles (M$)
- Taux de croissance des prestations et du passif

**Resultats** :
- Metriques : ratio de capitalisation terminal median, probabilite de sous-capitalisation, VaR du surplus
- **Graphique en eventail de l'actif** : projection avec bandes de percentiles (5e/25e/50e/75e/95e) + ligne mediane du passif
- **Graphique en eventail du ratio de capitalisation** : zones colorees (rouge < 80%, jaune 80-100%, vert > 100%)
- **Histogramme du ratio terminal** : distribution de la capitalisation finale
- **Comparaison** : si une optimisation a ete effectuee, comparer les projections allocation actuelle vs optimisee

---

### 3.7 Strategies - Alpha portable

**Objectif** : separer la generation de beta (marche) et d'alpha (rendement excedentaire) pour ameliorer le rendement ajuste au risque.

**Concept** : le portefeuille est decompose en :
1. **Portefeuille beta** : replique passivement un benchmark (ex: 60/40)
2. **Overlay alpha** : positions longues et courtes visant a generer de l'alpha

**Configuration (sidebar)** :

| Parametre | Description |
|-----------|-------------|
| Benchmark beta | 60/40 Equilibre, Politique de placement, Obligations pures (LDI), Croissance 70/30 |
| Strategie | Max ratio d'information, Max alpha (budget TE), Min tracking error (alpha cible), Budget de risque |
| Levier brut maximal | 1.0x a 2.0x (limite reglementaire) |
| Position courte max/actif | 1% a 15% |
| Spread de financement | 0 a 100 bps (cout des emprunts) |

**Strategies disponibles** :

| Strategie | Objectif | Quand l'utiliser |
|-----------|----------|------------------|
| Max ratio d'information | Maximise alpha / tracking error | Equilibre entre alpha et risque actif |
| Max alpha (budget TE) | Maximise l'alpha brut sous un budget de tracking error | Vous avez un budget de risque actif fixe |
| Min tracking error (alpha cible) | Minimise le risque actif pour atteindre un alpha cible | Vous voulez un alpha precis avec le moins de risque possible |
| Budget de risque | Optimise le rendement total ajuste au risque | Approche globale, pas de decomposition beta/alpha stricte |

**Resultats** :
- **8 metriques cles** : alpha brut, alpha net (apres couts), tracking error, ratio d'information, rendement combine, volatilite, levier brut, exposition nette
- **Conformite reglementaire** : verification automatique des limites de levier
- **5 graphiques** :
  1. Decomposition beta/alpha par classe d'actifs (barres empilees)
  2. Carte de chaleur de l'overlay (surponderations/sous-ponderations)
  3. Cascade du levier et des couts (exposition -> financement -> alpha net)
  4. Decomposition du risque (beta vs alpha vs interaction)
  5. Frontiere efficiente alpha/tracking error
- **Tableaux** : allocations detaillees, comparaison vs benchmark

---

### 3.8 Gestion - Reequilibrage

**Objectif** : planifier le retour a l'allocation cible et estimer les couts de transaction.

**Strategies de reequilibrage** :

| Strategie | Declencheur | Avantage |
|-----------|-------------|----------|
| Calendrier | A date fixe (mensuel, trimestriel, semestriel, annuel) | Simple, previsible |
| Seuil | Quand un ecart depasse un seuil (1-10%) | Reagit quand necessaire |
| Hybride | Seuil + frequence minimale | Combine les deux avantages |

**Resultats** :
- **Tableau des ecarts** : actuel vs cible, deviation (pp), montant a negocier (M$), direction (achat/vente/maintien)
- **Graphique des deviations** par classe d'actifs
- **Estimation des couts de transaction** : par classe d'actifs avec les couts en points de base (1 bps pour l'encaisse jusqu'a 200 bps pour le capital investissement)
- **Simulation comparative** : performance (rendement, vol, Sharpe) et couts de chaque frequence de reequilibrage

**Bareme des couts de transaction** :

| Classe d'actifs | Cout (bps) |
|-----------------|------------|
| Encaisse | 1 |
| Obligations gouvernementales | 5 |
| Obligations corporatives | 8 |
| Obligations indexees inflation | 8 |
| Actions canadiennes | 10 |
| Actions americaines | 10 |
| Actions EAFE | 15 |
| Actions emergentes | 25 |
| Matieres premieres | 15 |
| Immobilier | 150 |
| Infrastructure | 200 |
| Capital investissement | 200 |

---

### 3.9 Gestion - Gestion actif-passif (ALM)

**Objectif** : gerer la relation entre les actifs et les engagements du fonds de pension.

**Configuration (sidebar)** :
- Valeur actualisee du passif, duration du passif, taux d'actualisation, taux de croissance du passif

**4 sections** :

1. **Indicateurs ALM** : ratio de capitalisation, surplus (M$), ecart de duration (annees), statut de capitalisation avec recommandation d'action

2. **Sensibilite aux taux** : impact sur l'actif, le passif, le surplus et le ratio de capitalisation pour des chocs de -200 a +200 bps

3. **Optimisation du surplus** :
   - Maximise le rendement du surplus tout en controlant sa volatilite
   - Comparaison allocation actuelle vs optimisee (barres groupees)
   - Recommandation de couverture : allocation obligataire pour atteindre un ratio de couverture cible

4. **Trajectoire de desensibilisation (glide path)** :
   - Plan de transition progressive vers une allocation plus defensive a mesure que le ratio de capitalisation s'ameliore
   - Graphique en aires empilees : actifs de croissance / actifs de couverture / encaisse sur l'horizon

5. **Projection des flux de tresorerie** : cotisations, prestations et flux net sur 30 ans

---

### 3.10 Gestion - Rapports

**Objectif** : generer des rapports professionnels et importer/exporter des donnees.

**Types de rapports disponibles** :
- Rapport d'optimisation complet
- Synthese executive
- Rapport de risque
- Rapport de conformite reglementaire
- Rapport ESG

**Options** :
- Format : Excel (.xlsx, multi-feuilles) ou CSV
- Inclure : tests de tension, analyse ESG, resultats d'optimisation, resultats Monte Carlo

**Import de donnees** :
- Formats acceptes : CSV, Excel (.xlsx)
- Apercu des 10 premieres lignes avant adoption
- Template Excel telechargeablable pour structurer vos donnees

---

## 4. Flux de travail recommande

### Scenario typique d'une seance de travail

```
1. Tableau de bord     -> Prendre connaissance de l'etat actuel
2. Contraintes         -> Definir/ajuster les bornes et regles
3. Optimisation        -> Lancer une optimisation (Markowitz ou BL)
4. Frontiere           -> Visualiser l'ensemble des possibilites
5. Risque              -> Analyser le profil de risque du portefeuille optimise
6. Monte Carlo         -> Projeter l'evolution sur l'horizon de placement
7. Alpha portable      -> Explorer une strategie avec levier (optionnel)
8. ALM                 -> Verifier l'adequation actif-passif
9. Reequilibrage       -> Planifier la transition et estimer les couts
10. Rapports           -> Generer les documents pour le comite de placement
```

### Flux decisonnel

```
Etat actuel (Dashboard)
    |
    v
Definir les contraintes
    |
    v
Choisir un modele d'optimisation
    |
    +---> Markowitz : si pas de vue specifique
    +---> Black-Litterman : si convictions sur certains marches
    +---> Parite de risque : si focus diversification
    +---> CVaR : si focus protection en crise
    |
    v
Valider sur la frontiere efficiente
    |
    v
Tester le risque (stress tests + Monte Carlo)
    |
    v
[Optionnel] Alpha portable pour ameliorer le rendement
    |
    v
Verifier l'ALM (actif-passif)
    |
    v
Planifier le reequilibrage
    |
    v
Generer le rapport
```

---

## 5. Classes d'actifs disponibles (12)

| # | Code | Nom | Rendement attendu | Volatilite | Alternatif |
|---|------|-----|-------------------|------------|------------|
| 0 | ACTIONS_CDN | Actions canadiennes | 7.5% | 16% | Non |
| 1 | ACTIONS_US | Actions americaines | 8.0% | 17% | Non |
| 2 | ACTIONS_EAFE | Actions EAFE | 7.0% | 18% | Non |
| 3 | ACTIONS_EMERGENTES | Actions emergentes | 9.0% | 22% | Non |
| 4 | OBLIGATIONS_GOV_CDN | Obligations gouvernementales CDN | 3.5% | 6% | Non |
| 5 | OBLIGATIONS_CORP | Obligations corporatives | 4.5% | 8% | Non |
| 6 | OBLIGATIONS_INFLATION | Obligations indexees inflation | 3.0% | 7% | Non |
| 7 | IMMOBILIER | Immobilier | 7.0% | 12% | Oui |
| 8 | INFRASTRUCTURE | Infrastructure | 7.5% | 10% | Oui |
| 9 | CAPITAL_INVESTISSEMENT | Capital investissement | 10.0% | 20% | Oui |
| 10 | MATIERES_PREMIERES | Matieres premieres | 4.0% | 18% | Non |
| 11 | ENCAISSE | Encaisse | 2.5% | 1% | Non |

---

## 6. Contraintes reglementaires du Quebec

| Contrainte | Limite | Classes concernees |
|------------|--------|--------------------|
| Actions totales | <= 70% | CDN + US + EAFE + Emergentes |
| Capital investissement | <= 20% | Capital investissement |
| Actifs alternatifs | <= 40% | Immobilier + Infrastructure + Capital investissement |
| Liquidite minimale | >= 2% | Encaisse |
| Obligations totales | >= 10%, <= 70% | Gov CDN + Corp + Inflation |

### Contraintes supplementaires pour l'alpha portable

| Contrainte | Limite |
|------------|--------|
| Levier brut maximal | <= 200% |
| Exposition courte totale | <= 50% |
| Position courte par actif | <= 15% |
| Actifs eligibles au short | Actions + Obligations + Matieres premieres uniquement |

---

## 7. Donnees et session

- Les donnees sont stockees dans `st.session_state` et partagees entre toutes les pages
- Le bouton "Adopter le portefeuille optimise" (page Optimisation) met a jour l'allocation actuelle globalement
- Les resultats de Monte Carlo et de la frontiere sont conserves dans la session pour comparaison
- **Reinitialiser** : cliquer sur "Regenerer les donnees" dans la sidebar

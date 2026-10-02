# Guide d'utilisation - Optimiseur de Portefeuille Institutionnel

## Application Streamlit pour tout fonds institutionnel

Regimes de retraite a prestations ou a cotisations determinees, fondations, fonds de travailleurs : les parametres propres a chaque organisation sont regroupes dans un **profil de fonds**.

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

## 2. Profil du fonds et configuration globale

### Profil du fonds (page Vue d'ensemble > Profil du fonds)

Le profil contient tout ce qui distingue votre fonds. Toutes les pages l'utilisent.

| Element | Contenu |
|---------|---------|
| Identification | Nom, type de fonds (PD, CD, fondation, fonds de travailleurs, autre), notes |
| Bilan | Valeur de l'actif ; passif actuariel facultatif (valeur, duration, taux d'actualisation, croissance) |
| Marche et horizon | Taux sans risque, horizon de placement |
| Flux annuels | Entrees (cotisations, dons, souscriptions) et sorties (prestations, decaissements, rachats), avec leur croissance |
| Portefeuille de politique | Poids cibles par classe d'actifs, bornes min/max |
| Limites de groupe | Limites de la politique de placement (ex. actions totales <= 70 %), par categorie ou par classe |
| Hypotheses de marche | Facultatif : rendements et volatilites propres au fonds |

**Profils types fournis** (valeurs illustratives a remplacer) : caisse PD generique (profil par defaut), caisse PD mature orientee LDI, regime CD fonds equilibre, fondation / fonds de dotation, exemple de fonds de travailleurs.

**Sauvegarde** : bouton *Exporter le profil actif (JSON)* ; pour le recharger plus tard, *Importer un profil (JSON)*. Le profil vit dans la session du navigateur : exportez-le avant de fermer l'onglet.

Le profil est valide avant d'etre applique : les poids doivent totaliser 100 %, les bornes doivent permettre au moins une allocation, et l'application signale si le portefeuille de politique viole ses propres bornes ou limites.

### Sidebar

La sidebar affiche le profil actif et permet d'ajuster rapidement, pour la session, la valeur de l'actif, le taux sans risque et l'horizon.

### Source de donnees

- **Simulees** : genere des rendements synthetiques a partir d'une graine aleatoire (defaut : 42, 20 ans, frequence mensuelle). Bouton "Regenerer les donnees" pour creer un nouvel echantillon.
- **Importees** : utilisez la page Rapports pour importer vos propres donnees historiques (CSV ou Excel).

---

## 3. Pages de l'application

L'application comporte 15 pages organisees en 6 sections.

> **Note navigation** : la sidebar Streamlit affiche les sections par defaut; si vous ne voyez pas la section "🌱 Optimisation durable", cliquez sur **"View more"** en bas de la liste.

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
- **Appliquer les contraintes** : bornes et limites de groupe du profil de fonds (ou celles sauvegardees dans le Gestionnaire de contraintes)
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

**Contraintes de groupe** : reprises des limites du profil de fonds actif, modifiables pour la session. Les categories regroupent toutes les classes concernees (ex. *Actions* inclut les actions canadiennes, americaines, EAFE, emergentes et MSCI ACWI).

**Actifs liquides minimum** : part minimale du portefeuille dans les classes dont le score de liquidite est d'au moins 0,75.

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
- Contraintes du profil (oui/non)

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
| Levier brut maximal | 1.0x a 2.0x (limite par defaut) |
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
- **Conformite** : verification automatique des limites de levier par defaut
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

**Objectif** : gerer la relation entre les actifs et les engagements d'un regime a prestations determinees.

La page est active seulement si le profil de fonds a un passif actuariel ; sinon elle l'indique.

**Configuration (sidebar)** : valeurs du profil (passif, duration, taux d'actualisation, croissance), modifiables pour la session.

**4 sections** :

1. **Indicateurs ALM** : ratio de capitalisation, surplus (M$), ecart de duration (annees), ratio de couverture du risque de taux, et statut. Le statut retient le plus severe de deux diagnostics : le niveau de capitalisation et la perte de capitalisation en cas de baisse des taux de 100 pb (seuils detailles dans l'encadre *Comment le statut est calcule*). Un regime capitalise a 100 % mais tres expose aux taux n'est donc plus affiche « adequat ».

   **Exposition par segment de courbe** : valeur d'un point de base (k$) de l'actif et du passif aux echeances 2, 5, 10, 20 et 30 ans, pour verifier que la couverture porte sur la bonne partie de la courbe.

   **Ratio de couverture** = duration-dollar de l'actif / duration-dollar du passif. Le poids obligataire requis pour une cible depend de la duration des obligations de couverture et du ratio de capitalisation ; l'application indique quand la cible est inatteignable sans levier.

   **Passif detaille (facultatif)** : dans Profil du fonds, importez les flux de prestations projetes de l'evaluation actuarielle (CSV `annee, nominal, indexe` ; un gabarit est telechargeable) et, au besoin, une courbe d'actualisation (2, 5, 10, 20, 30 ans). Le passif est alors reevalue exactement sur la courbe choquee : la convexite et les durations par echeance sont calculees. Sans flux, l'application utilise la valeur, la duration et une convexite estimee pour un passif de retraite typique.

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

### Lire les indicateurs : ex ante ou ex post

- **Ex ante** (tableau de bord, optimisation, frontiere, Monte Carlo) : calcules a partir des hypotheses de marche (rendements attendus, volatilites, correlations).
- **Ex post** (VaR, CVaR, perte maximale, page Analytique de risque) : mesures sur une serie de rendements mensuels - simulee par defaut, ou vos donnees importees. La source est indiquee a cote des chiffres.

Une serie simulee de 20 ans est un seul tirage, avec periodes de crise : son Sharpe peut etre tres different du Sharpe ex ante sans que ce soit une erreur.

### 3.10 Gestion - Rapports

**Objectif** : generer des rapports professionnels et importer/exporter des donnees.

**Types de rapports disponibles** :
- Rapport d'optimisation complet
- Synthese executive
- Rapport de risque
- Conformite a la politique de placement (bornes et limites de groupe du profil)
- Rapport ESG

**Options** :
- Format : Excel (.xlsx, multi-feuilles) ou CSV
- Inclure : tests de tension, analyse ESG, resultats d'optimisation, resultats Monte Carlo

**Import de donnees** :
- Formats acceptes : CSV, Excel (.xlsx)
- Apercu des 10 premieres lignes avant adoption
- Template Excel telechargeablable pour structurer vos donnees

---

## 3.11-3.15 🌱 Optimisation durable

> L'univers durable fourni est un **exemple** tire du contexte d'un fonds de travailleurs quebecois (variantes « admissibles », micro-capitalisations du Quebec, scores de retombees regionales). Ses hypotheses et scores sont a adapter a votre fonds dans `sustainable/config.py`.

Ces cinq pages forment un outil d'optimisation bi-critere **rendement/risque ↔ durabilite**. Elles s'utilisent en sequence.

---

### 3.11 🌱 Univers durable

**Objectif** : configurer l'univers d'actifs et les preferences de durabilite avant tout calcul.

**Actifs disponibles (15 classes)** :

| Categorie | Actif standard | Variante durable |
|-----------|---------------|-----------------|
| Revenu fixe | Obligations court terme | — |
| Revenu fixe | Obligations univers | Obligations vertes |
| Revenu fixe | Hypotheques commerciales | Hypotheques vertes |
| Revenu fixe | Obligations haut rendement | — |
| Revenu fixe | Dettes emergentes | Dettes emergentes vertes |
| Actions & liquidites | Actions canadiennes | Actions responsables CDN |
| Actions & liquidites | Actions mondiales | Actions mondiales ESG |
| Actions & liquidites | Actions petite cap | — |
| Actions & liquidites | Eq. prive Quebec | — |
| Actions & liquidites | Fonds de couverture | — |
| Actifs prives | Dette privee | Dette privee verte |
| Actifs prives | Immobilier prive | Immobilier vert |
| Actifs prives | Infrastructure privee | Infrastructure verte |
| Actifs prives | Buyout | — |
| Actifs prives | Capital risque | Capital risque impact |

**Configuration** :
- **Checkbox** : inclure ou exclure chaque actif de l'optimisation
- **Toggle "Variante durable"** : si disponible, utiliser la version durable de l'actif (rendement et scores differents)
- **Pondérations des 5 dimensions** (curseurs 0-100%, normalises a 100%) :
  - Durabilite : poids de la dimension environnementale/sociale
  - Additionnalite : poids de la contribution incrementale du financement
  - Disponibilite : poids de l'accessibilite du produit sur le marche
  - Retombees regionales : poids des benefices economiques locaux ou regionaux
  - Liquidite : poids de la facilite de negociation
- **Graphique radar** : visualisation des priorites choisies
- **Aversion au risque γ** : controle l'agressivite de l'optimisation (defaut : 2.5)
- **Bouton "Appliquer"** : sauvegarde la configuration (reinitialise la frontiere et les resultats)

---

### 3.12 🌱 Scores de durabilite

**Objectif** : consulter et ajuster les scores de durabilite par classe d'actifs.

**Tableau editable** :
- Source par defaut : fichier Excel `WG_Categories_actifs_v4.xlsx` (Cartographie_complete_v1)
- Une ligne par actif actif, 5 colonnes de scores (echelle 1.0 a 5.0, pas de 0.25)
- La colonne **"Score composite"** est recalculee en temps reel selon les pondérations choisies dans l'Univers
- **Graphique a barres horizontales** : rouge (< 2), orange (2-3), vert (>= 3)

**Actions** :
- **"Reinitialiser aux valeurs Excel"** : remet les scores officiels du groupe de travail
- **"Appliquer"** : sauvegarde les scores modifies

> Les scores doivent etre entre 1.0 et 5.0. Un actif avec score composite eleve sera favorise par l'optimiseur lorsque λ est grand.

---

### 3.13 🌱 Frontiere durable (Pareto)

**Objectif** : calculer et visualiser le compromis entre performance financiere et durabilite.

**Principe** : l'optimiseur fait varier le parametre λ (poids durabilite) de 0 a λ_max en 50 points. Chaque valeur de λ produit un portefeuille optimal different, formant une **frontiere Pareto** :

```
Maximiser :  μ'w  −  (γ/2) × w'Σw  +  λ × S'w

Avec :  μ = rendements attendus des actifs selectionnes
        Σ = matrice de covariance
        S = scores composites de durabilite (ponderes par les dimensions)
        γ = aversion au risque (definie dans l'Univers)
        λ = poids accordé a la durabilite
```

**Graphique scatter interactif** :
- Axe X : score de durabilite composite du portefeuille
- Axe Y : ratio de Sharpe
- Chaque point = un λ different (couleur Viridis, bleu = financier pur, jaune = durabilite max)
- **Curseur λ** : selectionner un point precis sur la frontiere
- **Metriques du point selectionne** : rendement, volatilite, Sharpe, score durabilite, λ

**Actions** :
- **"Calculer la frontiere Pareto"** : lance l'optimisation (50 points, ~2s)
- **"Utiliser ce portefeuille"** : transfere le point selectionne vers la page Optimisation

> Prerequis : avoir configure et applique l'Univers durable.

---

### 3.14 🌱 Optimisation durable

**Objectif** : comparer le portefeuille financier pur et le portefeuille durable et adopter l'un d'eux.

**Panneau de gauche** :
- Curseur λ (pre-rempli depuis la Frontiere, ajustable manuellement)
- Bouton "Lancer l'optimisation" : calcule deux portefeuilles en parallele

**Tableau comparatif (3 colonnes)** :

| Metrique | Optimal financier (λ=0) | Optimal durable (λ choisi) |
|----------|------------------------|---------------------------|
| Rendement attendu | ... | ... |
| Volatilite | ... | ... |
| Ratio de Sharpe | ... | ... |
| Score durabilite composite | ... | ... |

**3 onglets de visualisation** :
1. **Allocations** : barres groupees cote a cote (financier vs durable)
2. **Score par dimension** : barres horizontales groupees (Durabilite, Additionnalite, Disponibilite, Retombees QC, Liquidite)
3. **Tableau detaille** : poids (%) pour chaque actif dans les deux portefeuilles

**Action** :
- **"Enregistrer le portefeuille durable"** : sauvegarde l'allocation pour le Rapport

> Le "cout de la durabilite" est visible en comparant le Sharpe du portefeuille financier (λ=0) et du portefeuille durable. Un ecart faible signifie que la durabilite s'obtient presque gratuitement.

---

### 3.15 🌱 Rapport durable

**Objectif** : produire un recapitulatif pour presentation au conseil ou au comite de placement.

**5 sections** :
1. **Metriques financieres** : rendement, volatilite, Sharpe, score durabilite
2. **Score par dimension** : tableau des contributions par dimension de durabilite
3. **Allocation detaillee** : poids de chaque actif + indicateur variante durable (✅/—)
4. **Hypotheses** : pondérations des dimensions, γ, λ, liste des actifs actifs avec variante utilisee
5. **Export** : bouton de telechargement CSV de l'allocation

> Prerequis : avoir clique "Enregistrer le portefeuille durable" dans la page Optimisation.

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

### Flux de travail - Optimisation durable

```
1. Univers durable      -> Selectionner les actifs, activer variantes durables,
                           regler les pondérations et γ, cliquer Appliquer
2. Scores de durabilite -> Verifier/ajuster les scores Excel, cliquer Appliquer
3. Frontiere durable    -> Calculer la frontiere Pareto, selectionner un point λ,
                           cliquer "Utiliser ce portefeuille"
4. Optimisation durable -> Ajuster λ si besoin, lancer l'optimisation,
                           comparer financier vs durable, enregistrer
5. Rapport durable      -> Consulter le recapitulatif, exporter en CSV
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
[Optionnel] Optimisation durable -> Frontiere Pareto -> Rapport durable
    |
    v
Generer le rapport
```

---

## 5. Classes d'actifs disponibles (17)

Hypotheses par defaut, remplacables dans le profil de fonds ou la page Source de donnees.

| # | Code | Nom | Categorie | Rendement attendu | Volatilite |
|---|------|-----|-----------|-------------------|------------|
| 0 | actions_canadiennes | Actions canadiennes | Actions | 7.5% | 16% |
| 1 | actions_americaines | Actions americaines | Actions | 8.0% | 17% |
| 2 | actions_eafe | Actions EAFE | Actions | 7.0% | 18% |
| 3 | actions_emergentes | Actions emergentes | Actions | 9.0% | 22% |
| 4 | obligations_gouvernementales_cdn | Obligations gouvernementales CDN | Titres a revenu fixe | 3.5% | 6% |
| 5 | obligations_corporatives | Obligations corporatives | Titres a revenu fixe | 4.5% | 8% |
| 6 | obligations_indexees_inflation | Obligations indexees inflation | Titres a revenu fixe | 3.0% | 7% |
| 7 | immobilier | Immobilier | Placements alternatifs | 7.0% | 12% |
| 8 | infrastructure | Infrastructure | Placements alternatifs | 7.5% | 10% |
| 9 | capital_investissement | Capital investissement | Placements alternatifs | 10.0% | 20% |
| 10 | rendement_absolu | Rendement absolu | Placements alternatifs | 5.5% | 8% |
| 11 | matieres_premieres | Matieres premieres | Matieres premieres | 4.0% | 18% |
| 12 | encaisse | Encaisse | Liquidites | 2.5% | 1% |
| 13 | actions_acwi | Actions MSCI ACWI | Actions | 7.8% | 16% |
| 14 | dette_emergente | Dette pays emergents | Titres a revenu fixe | 5.8% | 11% |
| 15 | dette_privee | Dette privee | Placements alternatifs | 7.5% | 6% |
| 16 | obligations_hy | Obligations HY | Titres a revenu fixe | 5.5% | 9% |

---

## 6. Limites de politique de placement

Les limites de groupe viennent du profil de fonds : chaque organisation saisit celles de sa propre politique de placement. Les valeurs des profils types sont **illustratives** ; ce ne sont pas des exigences reglementaires. Les lois sur les regimes de retraite (Quebec, federal, autres provinces) reposent surtout sur la regle de la personne prudente et sur des limites par emetteur, que l'application ne modelise pas.

Exemple (profil type *Caisse de retraite PD - generique*) :

| Limite | Valeur | Classes concernees |
|--------|--------|--------------------|
| Actions totales | <= 70% | Actions CDN, US, EAFE, emergentes, MSCI ACWI |
| Titres a revenu fixe | 10% a 70% | Oblig. gouvernementales, corporatives, indexees, dette emergente, HY |
| Placements alternatifs | <= 40% | Immobilier, infrastructure, capital investissement, rendement absolu, dette privee |
| Capital investissement | <= 20% | Capital investissement |
| Liquidites minimales | >= 2% | Encaisse |

### Limites par defaut pour l'alpha portable

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

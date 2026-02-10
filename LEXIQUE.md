# Lexique - Concepts financiers et techniques

## Optimiseur de Portefeuille Institutionnel - Caisse de Retraite

Ce lexique explique en detail tous les concepts utilises dans l'application. Les termes sont organises par theme.

---

## A. Mesures de rendement

### Rendement attendu (Expected Return)
Le rendement moyen qu'on anticipe pour un actif ou un portefeuille sur une periode donnee. Pour un portefeuille, c'est la somme ponderee des rendements attendus de chaque actif :

`E[Rp] = somme(wi * E[Ri])`

ou `wi` est le poids de l'actif i et `E[Ri]` son rendement attendu.

*Exemple* : si un portefeuille contient 60% d'actions (rendement attendu 8%) et 40% d'obligations (3.5%), le rendement attendu du portefeuille est : 0.60 x 8% + 0.40 x 3.5% = 6.2%.

### Rendement annualise
Transforme un rendement cumule ou mensuel en equivalent annuel. L'annualisation geometrique est plus precise que l'arithmetique car elle tient compte de la capitalisation :

`Rannuel = (1 + Rmensuel)^12 - 1`

### Rendement excedentaire (Excess Return)
Le rendement d'un portefeuille au-dela du taux sans risque. C'est ce que l'investisseur gagne pour avoir accepte de prendre du risque :

`Rexcedentaire = Rportefeuille - Rsans_risque`

### Taux sans risque
Le rendement theorique d'un placement sans aucun risque de defaut. En pratique, on utilise le taux des obligations gouvernementales a court terme (bons du Tresor). Dans l'application, il est configurable (defaut : 2.5%).

---

## B. Mesures de risque

### Volatilite (ecart-type)
Mesure la dispersion des rendements autour de leur moyenne. Plus la volatilite est elevee, plus les rendements fluctuent et plus le placement est risque :

`sigma = racine(variance(R))`

La volatilite annualisee a partir de donnees mensuelles : `sigma_annuel = sigma_mensuel * racine(12)`

*Exemple* : une volatilite de 16% signifie que, dans environ 68% des cas, le rendement annuel se situera entre (rendement moyen - 16%) et (rendement moyen + 16%).

### Variance
Le carre de la volatilite. C'est la mesure mathematique fondamentale de la dispersion. La variance d'un portefeuille depend non seulement de la variance de chaque actif, mais aussi de leurs correlations :

`Var(Rp) = w' * Sigma * w`

ou `Sigma` est la matrice de covariance.

### Covariance
Mesure comment deux actifs evoluent ensemble. Positive = ils montent/descendent ensemble. Negative = quand l'un monte, l'autre tend a descendre. C'est la base de la diversification.

### Correlation
La covariance normalisee, comprise entre -1 et +1 :
- +1 : les actifs evoluent parfaitement ensemble (aucune diversification)
- 0 : pas de lien lineaire
- -1 : les actifs evoluent parfaitement a l'oppose (diversification maximale)

`rho(i,j) = Cov(Ri, Rj) / (sigma_i * sigma_j)`

### Matrice de covariance
Matrice carree (n x n) qui contient les covariances entre toutes les paires de classes d'actifs. C'est l'ingredient fondamental de l'optimisation de portefeuille. L'application propose 3 methodes d'estimation (voir section I).

### VaR (Value at Risk) - Valeur a risque
La perte maximale attendue sur une periode donnee avec un certain niveau de confiance. Par exemple, une VaR de 5% a 95% signifie : "dans 95% des cas, la perte ne depassera pas 5%". Autrement dit, il y a 5% de chance de perdre plus que 5%.

L'application utilise 3 methodes :
- **Historique** : le quantile empirique des rendements observes
- **Parametrique** : basee sur l'hypothese de normalite
- **Cornish-Fisher** : ajustee pour l'asymetrie et les queues epaisses

*Limitation* : la VaR ne dit rien sur l'ampleur des pertes au-dela du seuil.

### CVaR (Conditional Value at Risk) - Valeur a risque conditionnelle
Aussi appelee Expected Shortfall (ES). C'est la perte moyenne dans les cas ou la VaR est depassee. Elle repond a la question : "quand ca va mal (au-dela de la VaR), quelle est la perte moyenne ?"

`CVaR = E[perte | perte > VaR]`

*Exemple* : si la VaR 95% est 5% et la CVaR 95% est 8%, cela signifie que dans les 5% pires scenarios, la perte moyenne est de 8%.

La CVaR est consideree superieure a la VaR car elle est :
- Coherente (elle satisfait les axiomes de mesure de risque)
- Sensible a l'ampleur des pertes extremes
- Convexe (plus facile a optimiser)

### Drawdown (perte maximale)
La baisse cumulee entre un sommet et un creux. Le drawdown maximum est la pire perte qu'un investisseur aurait subie s'il avait investi au pire moment :

`DD = (Vpic - Vcreux) / Vpic`

*Exemple* : si la valeur du fonds est passee de 1 000 M$ a 750 M$, le drawdown est de 25%.

### Contribution au risque
La part de la volatilite totale du portefeuille attribuable a chaque actif. Ce n'est pas simplement la volatilite de l'actif, mais le produit de son poids, sa volatilite et sa correlation avec le reste du portefeuille :

`RC_i = w_i * (Sigma * w)_i / sigma_p`

La somme des contributions au risque egale la volatilite totale du portefeuille.

---

## C. Ratios de performance

### Ratio de Sharpe
Le rendement excedentaire par unite de risque. C'est le ratio le plus utilise pour comparer des portefeuilles :

`Sharpe = (Rp - Rf) / sigma_p`

*Interpretation* : un Sharpe de 0.5 signifie que pour chaque 1% de volatilite, le portefeuille genere 0.5% de rendement au-dessus du taux sans risque.

- < 0.3 : mediocre
- 0.3 - 0.5 : acceptable
- 0.5 - 0.8 : bon
- > 0.8 : excellent

### Ratio de Sortino
Comme le Sharpe, mais ne penalise que la volatilite a la baisse (downside deviation). Il est plus adapte pour les investisseurs qui ne s'inquietent que des pertes :

`Sortino = (Rp - Rf) / sigma_down`

ou `sigma_down` ne tient compte que des rendements negatifs. Un Sortino eleve signifie que les rendements positifs sont eleves par rapport aux pertes.

### Ratio de Calmar
Le rendement annualise divise par le drawdown maximum. Mesure le rendement par unite de pire perte :

`Calmar = Rannualise / |DDmax|`

*Utilite* : un investisseur qui craint les pertes importantes preferera un Calmar eleve.

### Ratio Omega
Le ratio des gains ponderes par leur probabilite sur les pertes ponderees par leur probabilite, au-dessus d'un seuil (typiquement 0) :

`Omega = E[max(R - seuil, 0)] / E[max(seuil - R, 0)]`

- Omega > 1 : les gains depassent les pertes en esperance
- Omega < 1 : les pertes dominent

L'avantage de l'Omega est qu'il tient compte de toute la distribution (pas seulement la moyenne et la variance).

### Ratio d'information (Information Ratio)
L'alpha genere par rapport a un benchmark, divise par le tracking error. C'est le Sharpe du rendement actif :

`IR = alpha / TE`

- > 0.5 : bonne gestion active
- > 1.0 : excellente gestion active

### Asymetrie (Skewness)
Mesure si la distribution des rendements est symetrique. Une asymetrie negative signifie que les pertes extremes sont plus frequentes que les gains extremes (c'est souvent le cas des actions).

### Aplatissement (Kurtosis)
Mesure l'epaisseur des queues de la distribution. Un kurtosis eleve (> 3, excedentaire > 0) signifie que les evenements extremes sont plus frequents qu'une loi normale ne le predit. C'est critique pour la gestion de risque : la loi normale sous-estime les crises.

---

## D. Modeles d'optimisation

### Optimisation Moyenne-Variance (Markowitz)
Le modele fondateur de la theorie moderne du portefeuille (1952). Il cherche le portefeuille qui offre le meilleur rendement pour un niveau de risque donne (ou inversement).

Le probleme mathematique :
```
Minimiser   w' * Sigma * w           (variance du portefeuille)
Sous        w' * mu >= Rcible         (rendement minimum)
            somme(wi) = 1             (investissement total)
            wmin <= w <= wmax          (bornes)
```

**Hypotheses** : les rendements suivent une loi normale, les investisseurs ne se soucient que de la moyenne et de la variance.

**Variantes dans l'application** :
- **Max Sharpe** : trouve le portefeuille avec le ratio de Sharpe le plus eleve
- **Min Variance** : le portefeuille le moins risque possible
- **Rendement cible** : minimise le risque pour atteindre un rendement donne
- **Risque cible** : maximise le rendement pour un niveau de risque donne

### Black-Litterman
Modele developpe par Fischer Black et Robert Litterman (Goldman Sachs, 1992). Il resout un probleme majeur de Markowitz : la sensibilite extreme aux rendements attendus.

**Principe en 3 etapes** :

1. **Equilibre** : au lieu de deviner les rendements, on part de ceux implicites par le marche. On "inverse" la formule de Markowitz pour deduire les rendements que le marche "anticipe" etant donne les poids actuels :
   `Pi = delta * Sigma * w_marche`

2. **Vues de l'investisseur** : on exprime ses convictions sous forme quantitative :
   - Vue absolue : "Les actions americaines rendront 10%"
   - Vue relative : "Les actions canadiennes surperformeront les actions EAFE de 2%"
   - Chaque vue a un niveau de confiance (0-100%)

3. **Combinaison bayesienne** : les rendements du marche et les vues sont combines de facon ponderee par leur incertitude respective. Le resultat est un vecteur de rendements "posterieurs" plus stable.

**Parametres cles** :
- `delta` (aversion au risque) : controle l'agressivite des rendements d'equilibre
- `tau` (incertitude) : plus petit = plus de poids a l'equilibre, plus grand = plus de poids aux vues

### Parite de risque (Risk Parity)
Strategie ou chaque classe d'actifs contribue de maniere egale au risque total du portefeuille. Popularisee par Bridgewater (All Weather Fund).

**Principe** : au lieu d'egaliser les poids (ex: 1/12 chaque), on egalise les contributions au risque. Les actifs peu risques (obligations) recevront un poids plus eleve et les actifs risques (actions) un poids plus faible.

**Avantage** : diversification veritable du risque, pas de prevision de rendements necessaire.

**Inconvenient** : peut sous-performer si les rendements sont tres inegaux entre les classes. Necessite souvent du levier pour atteindre un rendement cible.

**Variante** : budgets de risque personnalises (ex: 40% du risque aux actions, 40% aux obligations, 20% aux alternatifs).

### CVaR (Rockafellar-Uryasev)
Optimisation qui minimise la CVaR au lieu de la variance. Developpe par Rockafellar et Uryasev (2000).

**Avantage par rapport a Markowitz** :
- Tient compte des pertes extremes, pas seulement de la dispersion
- Plus robuste quand les rendements ne sont pas normaux (queues epaisses)
- Se formule comme un programme lineaire (efficace a resoudre)

**Formulation** : au lieu de minimiser la variance, on minimise la perte moyenne dans les pires scenarios (au-dela d'un quantile).

---

## E. Frontiere efficiente

### Frontiere efficiente
L'ensemble des portefeuilles qui offrent le rendement maximal pour chaque niveau de risque (ou le risque minimal pour chaque rendement). Tout portefeuille en dessous de la frontiere est sous-optimal : on pourrait obtenir plus de rendement sans risque supplementaire.

### Portefeuille tangent
Le portefeuille sur la frontiere efficiente qui a le ratio de Sharpe le plus eleve. C'est le point de tangence entre la frontiere et la ligne du marche des capitaux (CML).

### Ligne du marche des capitaux (CML)
La droite qui part du taux sans risque et passe par le portefeuille tangent. Tout investisseur rationnel devrait se situer sur cette droite en combinant le portefeuille tangent et l'actif sans risque.

`E[R] = Rf + ((E[Rtangent] - Rf) / sigma_tangent) * sigma`

### Portefeuille de variance minimale (GMV)
Le portefeuille avec la plus faible volatilite possible, sans aucune contrainte de rendement. Il se situe a l'extremite gauche de la frontiere efficiente.

### Frontiere non contrainte
La frontiere calculee sans les contraintes reglementaires ni les bornes d'allocation. Elle est toujours au-dessus (ou confondue avec) la frontiere contrainte. La difference entre les deux represente le "cout" des contraintes.

### Frontiere Moyenne-CVaR
Variante de la frontiere efficiente ou l'axe horizontal represente la CVaR au lieu de la volatilite. Utile quand on se preoccupe des pertes extremes plutot que de la dispersion symetrique.

---

## F. Gestion actif-passif (ALM)

### Ratio de capitalisation (Funded Ratio)
Le rapport entre la valeur des actifs et la valeur actualisee des engagements (passif) :

`FR = Actifs / Passif`

- FR > 100% : le fonds a plus d'actifs que d'obligations (excedentaire)
- FR = 100% : capitalisation integrale
- FR < 100% : deficit, le fonds ne peut pas couvrir toutes ses obligations
- FR < 80% : situation critique necessitant un plan de redressement

### Surplus
La difference entre les actifs et le passif :

`Surplus = Actifs - Passif`

Un surplus positif offre une marge de securite ; un surplus negatif (deficit) est un risque pour les beneficiaires.

### Duration
Mesure la sensibilite du prix d'une obligation (ou d'un portefeuille) aux variations de taux d'interet. Exprimee en annees, elle indique approximativement de combien le prix change pour un mouvement de 1% des taux :

`delta_Prix / Prix ≈ -Duration * delta_Taux`

*Exemple* : un portefeuille obligataire de duration 7.5 ans perdra environ 7.5% si les taux montent de 1%.

### Ecart de duration (Duration Gap)
La difference entre la duration des actifs et la duration du passif ajustee par le levier :

`DG = Dactifs - (Passif/Actif) * Dpassif`

Un ecart de duration positif signifie que les actifs sont plus sensibles aux taux que le passif : une hausse des taux cause un surplus plus grand, une baisse des taux cause un deficit plus grand.

### LDI (Liability-Driven Investing)
Strategie d'investissement ou l'objectif principal n'est pas de maximiser le rendement, mais de faire correspondre l'evolution des actifs a celle du passif. Cela passe par l'appariement des durations et la couverture du risque de taux.

### Couverture (Hedging)
Reduction du risque de taux en alignant la duration des actifs sur celle du passif. Un ratio de couverture de 100% signifie que les actifs et le passif reagissent de facon identique aux mouvements de taux.

### Optimisation du surplus
Au lieu de maximiser le rendement du portefeuille, on maximise le rendement du surplus tout en controlant sa volatilite :

`Maximiser   E[Rsurplus] - lambda * Var(Rsurplus)`

### Trajectoire de desensibilisation (Glide Path)
Plan progressif de transition d'une allocation de croissance (plus d'actions) vers une allocation de couverture (plus d'obligations) a mesure que le ratio de capitalisation s'ameliore. L'idee est de "verrouiller" les gains quand le fonds est bien capitalise.

### Convexite
Mesure de second ordre de la sensibilite aux taux. Alors que la duration est une approximation lineaire, la convexite capture la courbure de la relation prix/taux. Importante pour les grands mouvements de taux.

---

## G. Simulation Monte Carlo

### Simulation Monte Carlo
Methode statistique qui genere des milliers de scenarios aleatoires pour projeter l'evolution future du fonds. Chaque simulation tire des rendements aleatoires a partir de la distribution estimee (moyenne, volatilite, correlations).

**Processus dans l'application** :
1. Pour chaque annee t et chaque simulation s, tirer un rendement : `r(t,s) = exp(N(mu - sigma^2/2, sigma^2)) - 1`
2. Mettre a jour l'actif : `A(t) = A(t-1) * (1 + r(t)) + Cotisations - Prestations`
3. Mettre a jour le passif : `L(t) = L(t-1) * (1 + g) - Prestations`
4. Calculer le ratio : `FR(t) = A(t) / L(t)`
5. Repeter pour toutes les annees et toutes les simulations

### Graphique en eventail (Fan Chart)
Representation visuelle des trajectoires simulees sous forme de bandes de percentiles :
- **Bande large (5e-95e)** : intervalle contenant 90% des scenarios
- **Bande etroite (25e-75e)** : intervalle contenant 50% des scenarios
- **Ligne centrale (50e)** : scenario median

Plus les bandes sont larges, plus l'incertitude est grande.

### Probabilite de sous-capitalisation
Le pourcentage de simulations ou le ratio de capitalisation terminal est inferieur a 100%. C'est un indicateur cle pour le comite de placement.

### VaR du surplus
La perte maximale sur le surplus avec un certain niveau de confiance. Par exemple, "le surplus ne diminuera pas de plus de 200 M$ dans 95% des cas sur 20 ans."

---

## H. Alpha portable

### Alpha
Le rendement excedentaire d'un portefeuille par rapport a son benchmark, attribuable a la gestion active (selection d'actifs, timing). Un alpha positif signifie que le gestionnaire a ajoute de la valeur.

`Alpha = Rportefeuille - Rbenchmark`

### Beta
La sensibilite du portefeuille au marche (benchmark). Un beta de 1.0 signifie que le portefeuille evolue exactement comme le marche. Le portefeuille beta dans l'alpha portable replique passivement le benchmark.

### Alpha portable
Strategie qui separe la generation de beta (exposition au marche) et d'alpha (rendement excedentaire). L'idee fondamentale :

1. **Portefeuille beta** : replique passivement un benchmark choisi (ex: 60/40)
2. **Overlay alpha** : positions longues et courtes qui visent a generer un rendement excedentaire sans modifier l'exposition nette au marche
3. **Portefeuille combine** = beta + overlay alpha

**Pourquoi "portable" ?** Parce que l'alpha est "deplace" d'une source (ou le gestionnaire est competent) vers une exposition (ou il veut etre investi). On peut generer de l'alpha dans les actions tout en maintenant une exposition obligataire.

### Tracking Error
La volatilite de la difference de rendement entre le portefeuille et son benchmark. C'est le "risque actif" :

`TE = ecart-type(Rportefeuille - Rbenchmark)`

Un TE eleve signifie que le portefeuille s'ecarte significativement du benchmark (pour le meilleur ou le pire).

### Overlay alpha
Les positions qui s'ajoutent au portefeuille beta pour generer de l'alpha. Elles contiennent des surponderations (positions longues au-dessus du benchmark) et des sous-ponderations (positions courtes en dessous du benchmark). La somme nette de l'overlay est proche de zero pour ne pas modifier l'exposition totale.

### Position courte (Short Selling)
Vendre un actif qu'on ne possede pas en l'empruntant, avec l'intention de le racheter plus tard a un prix inferieur. Dans l'overlay alpha, les positions courtes financent les positions longues et profitent de la baisse des actifs sous-ponderes.

### Levier brut (Gross Leverage)
La somme des valeurs absolues de toutes les positions :

`Levier brut = somme(|wi|)`

Un levier brut de 1.5x signifie que le portefeuille a 150% d'exposition totale (par exemple, 125% long et 25% short). Plus le levier est eleve, plus le potentiel de rendement ET de perte est amplifie.

### Exposition nette (Net Exposure)
La somme algebrique de toutes les positions :

`Exposition nette = somme(wi)`

Typiquement maintenue proche de 100% pour conserver l'exposition au marche. Un portefeuille 125% long / 25% short a un levier brut de 150% mais une exposition nette de 100%.

### Cout de financement
Le cout d'emprunter des titres pour les vendre a decouvert et le cout du capital pour le levier excedentaire. Exprime en points de base (bps). Ce cout reduit l'alpha brut pour donner l'alpha net :

`Alpha net = Alpha brut - Couts de financement`

### Ratio d'information (dans le contexte alpha portable)
L'alpha divise par le tracking error. C'est la mesure principale de l'efficacite de la strategie alpha :

`IR = Alpha / TE`

Un IR de 0.5 signifie que pour chaque 1% de tracking error, la strategie genere 0.5% d'alpha.

### Decomposition du risque (beta vs alpha)
Le risque total du portefeuille combine se decompose en :
- **Risque beta** : le risque inherent au benchmark
- **Risque alpha** : le risque ajoute par l'overlay (tracking error)
- **Interaction** : la covariance entre beta et alpha

`Risque total^2 = Risque beta^2 + Risque alpha^2 + 2 * Cov(beta, alpha)`

---

## I. Methodes d'estimation de la covariance

### Covariance echantillonnale (Sample)
L'estimateur classique : `Sigma = (1/(T-1)) * X' * X`. Simple mais instable quand le nombre d'observations est petit par rapport au nombre d'actifs.

### Ledoit-Wolf (shrinkage)
Methode de regularisation qui combine la matrice echantillonnale avec une matrice structuree (typiquement une matrice proportionnelle a l'identite). Le coefficient de retrecissement (shrinkage) est calcule automatiquement pour minimiser l'erreur quadratique :

`Sigma_LW = alpha * Cible + (1 - alpha) * Sigma_echantillon`

**Avantage** : plus stable et mieux conditionnee que la matrice echantillonnale. C'est le choix recommande par defaut.

### EWMA (Exponential Weighted Moving Average)
Methode qui pondere les observations recentes plus fortement que les anciennes. Le parametre "halflife" (defaut : 60 periodes) controle la vitesse d'oubli :

**Avantage** : reagit plus vite aux changements de regime de marche.
**Inconvenient** : peut sur-reagir a des evenements temporaires.

### Debruitage Marcenko-Pastur
Methode basee sur la theorie des matrices aleatoires. Elle identifie les valeurs propres de la matrice de covariance qui ne contiennent que du bruit statistique et les remplace par leur moyenne. Seules les valeurs propres au-dessus du seuil de Marcenko-Pastur sont conservees.

### Correction PSD (Nearest PSD)
Algorithme de Higham (2002) qui projete une matrice sur l'ensemble des matrices positives semi-definies. Necessaire quand les methodes d'estimation produisent une matrice non PSD (ce qui rendrait l'optimisation impossible).

---

## J. Tests de tension (Stress Testing)

### Test de tension historique
Applique les chocs observes lors de crises passees au portefeuille actuel. L'application inclut 6 scenarios :

| Scenario | Annee | Caracteristique principale |
|----------|-------|---------------------------|
| Crise financiere | 2008 | Chute generalisee des actifs risques, fuite vers les obligations gouvernementales |
| COVID-19 | 2020 | Chute rapide et violente, recuperation en V |
| Hausse des taux | 2022 | Pertes simultanees actions et obligations (rare) |
| Bulle technologique | 2000 | Chute prolongee des actions, obligations refuge |
| Crise de la dette euro | 2011 | Contagion europeenne, actions EAFE touchees |
| Stagflation | - | Inflation elevee + recession : matieres premieres en hausse, le reste en baisse |

### Test de tension parametrique
L'utilisateur definit ses propres chocs hypothetiques sur 4 axes :
- **Choc actions** : baisse/hausse du marche des actions
- **Choc de taux** : hausse/baisse des taux d'interet (en points de base)
- **Choc de spreads** : ecartement/resserrement des ecarts de credit
- **Choc d'inflation** : hausse/baisse de l'inflation

L'impact est calcule via des vecteurs de sensibilite propres a chaque classe d'actifs (beta actions, duration, sensibilite au spread, sensibilite a l'inflation).

### Test de tension inverse (Reverse Stress Test)
Au lieu de definir un choc et voir la perte, on definit une perte et on cherche le choc minimal qui la cause. Utilise l'optimisation (SLSQP) pour trouver le vecteur de choc de norme minimale qui atteint le seuil de perte.

*Exemple* : "Quel est le plus petit choc qui causerait une perte de 15% du portefeuille ?"

---

## K. Reequilibrage

### Reequilibrage
Le processus de retour a l'allocation cible apres que les mouvements de marche ont fait deriver les poids. Sans reequilibrage, un portefeuille tend naturellement a surponderer les actifs qui ont le plus monte.

### Rotation (Turnover)
Le volume de transactions effectuees lors d'un reequilibrage, exprime en pourcentage de l'actif total :

`Turnover = somme(|wi_apres - wi_avant|) / 2`

Un turnover de 10% signifie que 10% du portefeuille a ete achete et 10% vendu.

### Cout de transaction
Le cout total de l'execution des transactions. Varie fortement selon la classe d'actifs :
- Actifs liquides (encaisse, obligations gouvernementales) : 1-5 bps
- Actions cotees : 10-25 bps
- Actifs alternatifs (immobilier, infrastructure, PE) : 150-200 bps

### Points de base (bps)
Unite de mesure financiere : 1 point de base = 0.01% = 0.0001.
100 bps = 1%. Utilise pour les couts, les spreads et les rendements de faible amplitude.

---

## L. ESG et contraintes

### Score ESG
Note composite evaluant les pratiques Environnementales, Sociales et de Gouvernance d'un actif ou d'un portefeuille. Echelle de 0 a 100 dans l'application.

Le score ESG du portefeuille est la moyenne ponderee des scores de chaque classe d'actifs :

`ESG_portefeuille = somme(wi * ESG_i)`

### Intensite carbone
Les emissions de gaz a effet de serre par million de dollars investi (tCO2e/M$). Utilisee pour mesurer l'empreinte environnementale du portefeuille.

### Contraintes reglementaires
Regles imposees par les autorites (Retraite Quebec, BSIF) qui limitent l'allocation. Elles visent a proteger les beneficiaires en evitant une concentration excessive ou des prises de risque deraisonnables.

---

## M. Termes techniques de l'optimisation

### Programmation convexe
Cadre mathematique utilise pour resoudre les problemes d'optimisation de portefeuille. Un probleme est convexe quand la fonction objectif est convexe et les contraintes forment un ensemble convexe. Cela garantit que toute solution locale est aussi la solution globale.

### CVXPY
Bibliotheque Python utilisee dans l'application pour formuler et resoudre les problemes d'optimisation convexe. Supporte la programmation quadratique (Markowitz), la programmation lineaire (CVaR), et les contraintes coniques (levier brut).

### Solveur
L'algorithme qui resout le probleme d'optimisation. L'application utilise principalement CLARABEL (pour les problemes coniques) et SCS (pour l'alpha portable). En cas d'echec, elle bascule automatiquement sur ECOS ou OSQP.

### Regularisation
Technique qui ajoute un petit terme a l'objectif pour stabiliser la solution. Par exemple, ajouter `lambda * ||w||^2` empeche les poids de prendre des valeurs extremes. C'est l'equivalent financier de "ne pas mettre tous ses oeufs dans le meme panier de facon extreme".

### PSD (Positive Semi-Definite)
Propriete mathematique requise pour la matrice de covariance. Une matrice PSD garantit que la variance du portefeuille est toujours positive ou nulle. Si la matrice estimee n'est pas PSD, l'optimisation echoue ; l'application corrige automatiquement ce probleme via l'algorithme de Higham.

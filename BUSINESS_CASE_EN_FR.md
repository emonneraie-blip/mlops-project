# NYC Taxi Demand Forecast | Business case & technical summary

## English

### Business question
How can a taxi operator anticipate pickups by New York zone for the next hour? Forecasts could help position drivers and plan capacity. Underestimating may mean longer waits or lost trips; overestimating may leave drivers idle. TLC records contain no cost or customer-outcome data, so the project does not claim measured savings.

### System
A deliberately simple `HistGradientBoostingRegressor` is supported by a reproducible MLOps workflow:

**TLC data → DVC pipeline → time-based training → MLflow tracking and registry → gated promotion → FastAPI/Docker service → monitoring → retraining or rollback.**

Git versions code, DVC identifies data, and `uv.lock` pins dependencies. MLflow records commit, data hash, parameters and metrics. Aliases (`@staging`, `@production`, `@previous`) support controlled promotion and recovery. GitHub Actions runs checks and builds a commit-tagged image.

### Evidence and interpretation

| Evidence | What it shows |
|---|---|
| February 2020 golden set: MAE 4.68 vs naïve baseline 9.26 pickups per zone-hour | The model reduces baseline error by about half on this held-out period. |
| Covid shift: monthly trips 6.3m → 0.24m (−96%); PSI `lag_1` 0.637, predictions 0.625 | Drift was detected; a candidate was retrained and evaluated through a gate. |
| Broken-data simulation: fares in cents, median $1,380, PSI 11.63; zone 132 went silent | Data-quality checks blocked retraining instead of teaching the model corrupted inputs. |
| Rollback v2 → v1: 1.5 seconds | A known-good model can be restored quickly. A diagnostic February request (zone 132: v1 predicted 87.0, Covid-trained v2 39.8) exposed a recovery-regime risk; it was not itself the actual rollback trigger. |

### Decisions, limits and value
The system distinguishes real-world drift, which may justify a candidate model, from broken input data, which should stop retraining. A gate compares candidates with the naïve baseline and production model, then requires human approval. A gate only assesses its evaluation window: v2 passed on a Covid-period window but could fail when demand recovered. MAE also changes with demand volume, so scores across regimes are not directly comparable.

This is a local course demonstration, not a measured production deployment. DVC/MLflow storage is local, drift checks are manual, and CI builds but does not start the container. The demonstrated value is operational trust: traceable versions, visible incidents and fast recovery. Cloud storage, scheduled monitoring and measured business outcomes remain future work.

**Jury takeaway:** the project operationalises a forecast; it does not claim that model accuracy alone creates business value.

---

## Français

### Question métier
Comment aider un opérateur de taxis à anticiper, pour l’heure suivante, les prises en charge par zone de New York ? Les prévisions pourraient aider à positionner les chauffeurs et à planifier la capacité. Sous-estimer peut allonger l’attente ou faire perdre des courses ; surestimer peut laisser des chauffeurs inactifs. Les données TLC ne contiennent ni coûts ni résultats clients : le projet ne revendique donc pas d’économies mesurées.

### Système
Un `HistGradientBoostingRegressor` volontairement simple s’inscrit dans un workflow MLOps reproductible :

**Données TLC → pipeline DVC → entraînement temporel → suivi et registre MLflow → promotion avec gate → service FastAPI/Docker → surveillance → réentraînement ou rollback.**

Git versionne le code, DVC identifie les données et `uv.lock` fixe les dépendances. MLflow conserve le commit, le hash des données, les paramètres et les métriques. Les alias (`@staging`, `@production`, `@previous`) permettent une promotion et un retour contrôlés. GitHub Actions exécute les contrôles et construit une image taguée avec le commit.

### Résultats et interprétation

| Résultat | Ce qu’il montre |
|---|---|
| Golden set de février 2020 : MAE 4,68 contre 9,26 pour la baseline naïve, par zone-heure | Le modèle réduit d’environ moitié l’erreur de la baseline sur cette période tenue à l’écart. |
| Choc Covid : courses mensuelles de 6,3 M à 0,24 M (−96 %) ; PSI `lag_1` 0,637, prédictions 0,625 | La dérive est détectée ; un candidat est réentraîné et évalué par un gate. |
| Simulation de données cassées : tarifs en centimes, médiane de 1 380 $, PSI 11,63 ; zone 132 muette | Les contrôles de qualité bloquent le réentraînement sur des entrées corrompues. |
| Rollback v2 → v1 : 1,5 seconde | Une version fiable peut être restaurée rapidement. Une requête diagnostique de février (zone 132 : 87,0 avec v1, 39,8 avec v2 entraîné pendant le Covid) a révélé un risque lors de la reprise ; ce n’était pas le déclencheur réel du rollback. |

### Arbitrages, limites et valeur
Le système distingue la dérive du monde réel, qui peut justifier un candidat, des données cassées, qui doivent bloquer le réentraînement. Le gate compare le candidat à la baseline naïve et au modèle en production, puis exige une validation humaine. Il n’évalue que la fenêtre choisie : v2 a passé un contrôle sur une période Covid, mais pouvait échouer lors de la reprise. La MAE varie aussi avec le volume et ne se compare pas directement entre régimes.

Il s’agit d’une démonstration locale de cours, pas d’un déploiement dont l’impact en production est mesuré. Le stockage DVC/MLflow est local, les contrôles de dérive sont manuels et la CI construit l’image sans démarrer le conteneur. La valeur démontrée est la confiance opérationnelle : versions traçables, incidents visibles et retour rapide. Le stockage cloud, la surveillance planifiée et la mesure des résultats métier restent à réaliser.

**À retenir pour le jury :** le projet rend une prévision exploitable ; il ne prétend pas que la seule précision du modèle crée de la valeur métier.

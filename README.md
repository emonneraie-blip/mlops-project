# Prévision de la demande de taxis à New York — couche MLOps

Le modèle prévoit le nombre de prises en charge dans une zone de taxi pour l'heure suivante.
Le sujet n'est pas le modèle (volontairement simple) mais son exploitation : on sait quelles données l'ont entraîné, on voit s'il fonctionne encore, et on sait revenir en arrière.

Données : NYC Taxi & Limousine Commission (TLC), Trip Record Data, Yellow Taxi —
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page. Les fichiers bruts ne sont pas dans Git : `src/download.py` + DVC les reproduisent.

## Le système

```mermaid
flowchart LR
    TLC[(TLC Parquet<br/>mensuels)] -->|download.py| RAW[data/raw]
    RAW -->|prepare.py<br/>zone x heure| FEAT[features.parquet]
    DVC[[DVC<br/>dvc.lock + remote]] -.versionne.- RAW
    DVC -.versionne.- FEAT
    FEAT -->|train.py| RUN[MLflow run<br/>params, métriques,<br/>git_commit, data_md5]
    RUN --> REG[(MLflow Registry<br/>@staging @production @previous)]
    REG -->|promote.py<br/>gate + validation| REG
    REG -->|models:/taxi-demand@production| API[FastAPI /v1/predict]
    FEAT -->|drift.py<br/>PSI, KS, qualité| MON{Alerte ?}
    MON -->|drift réel| RET[retrain.py<br/>cooldown + gate]
    RET --> REG
    MON -->|données cassées| RB[rollback.py<br/>@production -> @previous]
    RB --> REG
```

| Où | Ce qui est stocké |
|---|---|
| Git | code, `params.yaml`, `dvc.yaml`, `dvc.lock`, `uv.lock` — jamais les données ni les secrets |
| DVC | les octets des données (remote local `../dvc-storage` sur la machine de l'équipe, S3 en production) |
| MLflow tracking | paramètres, métriques, tags `git_commit`, `git_dirty`, `data_md5` |
| MLflow Registry | versions du modèle et alias `@staging`, `@production`, `@previous` |

## Démarrage depuis un clone vide

Prérequis : [uv](https://docs.astral.sh/uv/) et git. Docker est optionnel.

```bash
git clone https://github.com/emonneraie-blip/mlops-project && cd mlops-project
cp .env.example .env
uv sync --locked
```

Terminal 1 — serveur MLflow (UI sur http://localhost:5050) :
```bash
make mlflow
```

Terminal 2 — données, entraînement, mise en production :
```bash
make data       # uv run dvc repro : télécharge 6 mois TLC (~440 Mo) et construit les features
make train      # nouvelle version enregistrée avec l'alias @staging
make promote    # gate sur le golden set (février 2020), puis validation "o/N"
```

Terminal 3 — l'API (documentation sur http://localhost:8000/docs) :
```bash
make serve
make request    # depuis un autre terminal
```

Sans `make` (Windows) : chaque cible du `Makefile` contient la commande exacte à copier.

Tout en Docker : `docker compose up --build` lance MLflow et l'API. Les scripts (`make train`, `make promote`…) tournent ensuite depuis la machine, comme ci-dessus. L'API démarre en 503 tant qu'aucun modèle n'est en `@production`, puis `POST /v1/reload` (appelé automatiquement par `promote.py`) charge le modèle.

## Données et découpage

Une ligne par zone et par heure (`pickups`, `avg_distance`, `avg_fare`). Les heures sans course sont ajoutées avec 0 pour que les lags soient justes.

Features : `lag_1`, `lag_2`, `lag_24`, `lag_168`, heure, jour de semaine, week-end, zone. Modèle : `HistGradientBoostingRegressor` (scikit-learn).

Découpage par le temps, jamais au hasard (`params.yaml`) :

| Période | Rôle |
|---|---|
| 8 nov. 2019 → 31 janv. 2020 | référence : entraînement |
| février 2020 | golden set : test et gate, jamais vu à l'entraînement |
| 15 mars → 30 avril 2020 | période de drift : effondrement de la demande (Covid) |

Métriques : MAE (lisible : « on se trompe de N courses par heure ») et RMSE (pénalise les grosses erreurs). Chaque modèle est comparé à une baseline naïve : même heure la veille (`lag_24`).

## Reproduire un run

Chaque run MLflow porte `git_commit`, `git_dirty` et `data_md5`. Pour reconstruire un run :
```bash
git checkout <git_commit>
uv sync --locked
uv run dvc repro       # reconstruit les données de ce commit (retélécharge les fichiers TLC)
git diff --exit-code dvc.lock   # aucune différence = mêmes octets que ceux du run
make train             # même MAE à 1e-6 près (random_state fixe, pas d'early stopping aléatoire)
```
`uv run dvc pull` ne marche que sur la machine qui héberge le remote DVC : ici un dossier local (`../dvc-storage`). Depuis un clone neuf, il échoue (`Missing cache files`) ; `dvc repro` reconstruit alors les données depuis la source. Avec un remote partagé (S3 ou MinIO), `dvc pull` suffirait et ne dépendrait plus de la disponibilité des fichiers TLC.

`train.py` avertit si des fichiers ne sont pas commités : un run avec `git_dirty=True` n'est pas reproductible.

## API

| Route | Rôle |
|---|---|
| `POST /v1/predict` | prévision pour une zone et une heure, à partir des comptages récents |
| `GET /v1/health` | 200 si un modèle est chargé, 503 sinon (le load balancer arrête d'envoyer du trafic) |
| `GET /v1/model` | version servie |
| `POST /v1/reload` | recharge la version `@production` (après promotion ou rollback) |

Le contrat est strict (Pydantic, `extra="forbid"`) : un champ inconnu, une valeur négative ou une zone hors 1–265 renvoie 422. Chaque prédiction est loggée en JSON avec un `request_id` et la version du modèle, sans donnée personnelle.

### Latence sous charge

`make load-test` (API lancée, modèle en `@production`) envoie des requêtes sans pause avec 1, 4 puis 16 utilisateurs simultanés, et affiche P50, P95, P99, débit et erreurs. Le budget est celui de la séance 3 : P95 ≤ 200 ms. On retient le débit le plus haut qui tient ce budget sans erreur, pas le débit maximal atteint. Code de sortie 1 si aucun niveau ne tient.

## Monitoring du drift

`src/drift.py` compare une fenêtre récente à la période d'entraînement du modèle en production (lue dans MLflow : après un réentraînement, la référence suit).

- PSI sur les entrées (`lag_1`, `lag_24`), sur `avg_distance` / `avg_fare` et sur les prédictions : 0,10 à surveiller, 0,25 pour agir. La statistique KS est affichée à côté : sur des centaines de milliers de lignes, la p-value est toujours minuscule, on lit la taille de l'écart.
- Qualité des données : valeurs manquantes, médiane hors plage plausible, zone active devenue muette.
- Le PSI n'est pas calculé sur la cible : la cible se juge avec la MAE. Ici les labels arrivent une heure après la prévision (le nombre réel de courses), donc la MAE de la fenêtre, affichée par `drift.py`, est disponible presque aussi vite que le PSI. Elle ne se compare pas d'un régime à l'autre : elle baisse quand le volume baisse (incident 1).

Chaque contrôle écrit `monitoring/drift_report.json`, logge les PSI dans l'expérience MLflow `taxi-demand-monitoring`, et ajoute l'alerte dans `monitoring/events.log`.

## Réentraînement

`src/retrain.py` : contrôle de drift → décision → réentraînement → gate → validation humaine.

- Alerte de qualité de données : pas de réentraînement. Réentraîner sur une colonne cassée apprendrait l'erreur au modèle. On corrige la source, ou on fait un rollback.
- Drift réel (le monde a changé) : réentraînement sur les 3 dernières semaines, gate sur les 7 derniers jours.
- Le gate décide, pas l'alarme : la candidate doit battre la baseline naïve et la version en production sur le même golden set.
- Cooldown de 24 h entre deux réentraînements (`--force` pour passer outre).

## Rollback

Que faire pour chaque alerte, et qui décide : [docs/runbook.md](docs/runbook.md).

```bash
make rollback          # @production revient sur @previous, l'API recharge
```
- Une commande, moins de 5 minutes (environ 1 seconde mesurée). Aucun modèle n'est supprimé : la version fautive devient `@previous` et reste inspectable.
- Quand : erreur 5xx en hausse, alerte qualité de données sur des données qui ont servi au dernier réentraînement, ou MAE de la nouvelle version au-dessus de l'ancienne quand les labels arrivent.
- Qui : la personne d'astreinte nommée dans le [runbook](docs/runbook.md), sans attendre de validation (revenir en arrière ne fait jamais pire que la dernière version validée).
- Cible précise : `uv run python src/rollback.py --version 3`.

## Incidents simulés

Voir `incident/INCIDENT_REPORT.md`.
```bash
make drift         # drift réel : Covid
make retrain       # alerte -> réentraînement -> gate -> promotion
make rollback
make inject-zone   # la zone 132 (JFK) ne remonte plus rien
make inject-fare   # tarif en centimes au lieu de dollars
make events        # journal de tout ce qui s'est passé
```

## Tests et CI

`make test` : tests unitaires (lags sans fuite du futur, PSI) et tests d'intégration de l'API (vrai modèle chargé depuis un Registry MLflow temporaire, sans données ni serveur).

GitHub Actions (`.github/workflows/ci.yml`) sur chaque pull request et push sur `main` : installation depuis `uv.lock`, tests, puis build de l'image Docker taguée avec le commit. Actions épinglées par SHA, permissions en lecture seule, `concurrency` et `timeout-minutes`.

## Avec un mois de plus

- Contrôle de drift planifié chaque nuit (Airflow), réentraînement et gate dans GitHub Actions avec un environnement à validation obligatoire.
- Métriques Prometheus + tableau de bord Grafana sur les quatre couches (infra, API, données, modèle).
- Déploiement canary sur Kubernetes, rollback par `kubectl rollout undo`.
- Remote DVC et artefacts MLflow sur S3 / MinIO au lieu de dossiers locaux.
- Features météo et jours fériés ; un modèle par groupe de zones.

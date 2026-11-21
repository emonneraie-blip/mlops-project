# Résumé en une page — demande de taxis à New York, la couche MLOps

*English version: [SUMMARY_EN.md](SUMMARY_EN.md)*

**En une phrase.** Un service qui prévoit les prises en charge de taxis de l'heure suivante pour chaque zone de New York, construit pour savoir quelles données l'ont entraîné, voir quand il ne fonctionne plus, et annuler une mise en production en quelques secondes.

## Métier

- **Qui l'utilise :** un opérateur de taxis qui décide où positionner ses chauffeurs pour l'heure suivante, zone par zone.
- **Ce que coûte une erreur :** sous-estimer, les clients attendent et les courses sont perdues ; surestimer, les chauffeurs restent inactifs. Non chiffré : les données TLC ne contiennent aucun coût.
- **Valeur :** MAE de 4,68 courses par zone et par heure sur février 2020, contre 9,26 pour « même heure la veille » : environ deux fois moins d'erreur.
- **Risque que la couche gère :** de février à avril 2020, les courses baissent de 96 % (6,3 millions à 0,24 million par mois). Un modèle entraîné sur des mois normaux rencontre un monde qu'il n'a jamais vu.

## Technique

| Couche | Ce que nous avons construit | Outils |
|---|---|---|
| Données | TLC Yellow Taxi, nov. 2019 à avr. 2020, une ligne par zone et par heure (1 100 400 lignes) ; lags 1, 2, 24, 168 h et calendrier ; découpage par le temps : entraînement nov. à janv., golden set février, drift 15 mars au 30 avril | DVC, pandas |
| Entraînement | `HistGradientBoostingRegressor`, graine fixe ; chaque run porte `git_commit`, `git_dirty`, `data_md5` | scikit-learn, MLflow |
| Mise en production | Alias `@staging`, `@production`, `@previous` ; gate : battre la baseline et la production sur le même golden set, puis validation humaine | MLflow Registry |
| Service | `/v1/predict` avec un contrat strict (422), `/v1/health` (503 sans modèle), la version du modèle dans chaque réponse ; un reload raté garde le modèle en place | FastAPI, Pydantic, Docker |
| Monitoring | PSI par entrée et sur les prédictions (0,10 à surveiller, 0,25 pour agir), KS affiché ; qualité des données : valeurs manquantes, médianes implausibles, zones muettes | scipy, MLflow |
| Réaction | Drift réel : réentraînement sur 3 semaines, cooldown de 24 h, gate. Données cassées : pas de réentraînement. Rollback : une commande | `retrain.py`, `rollback.py` |
| Qualité | Tests lancés en CI : contrat de l'API, lags, PSI, rollback, gate, drift réel vs données cassées ; actions épinglées par SHA | pytest, GitHub Actions |

## Résultats (runs du 20 novembre 2026)

- **Drift Covid détecté :** PSI de 0,637 sur `lag_1` et de 0,625 sur les prédictions ; sur février (contrôle), tous les PSI restent sous 0,01.
- **Réentraînement, puis rollback :** la v2 (MAE 0,589 du 24 au 30 avril) a été promue, puis remplacée par la v1 en 1,5 s car elle divise par deux une prévision normale (39,8 contre 87,0 pour la même requête).
- **Données cassées, pas de réentraînement :** un tarif en centimes et une zone JFK muette ont été détectés et classés en qualité de données.

## Lancer le projet

```bash
git clone https://github.com/emonneraie-blip/mlops-project && cd mlops-project
cp .env.example .env && uv sync --locked
make mlflow                                   # terminal 1
make data && make train && make promote       # terminal 2
make serve                                    # terminal 3, puis : make request
```

## Limites et suite

Remote DVC local (`dvc pull` échoue depuis un clone neuf) → S3 ou MinIO. Contrôle de drift lancé à la main → tâche Airflow chaque nuit. Gate sur une seule fenêtre → une deuxième fenêtre du régime normal, et la WAPE. Pas de métriques en continu → Prometheus et Grafana. Pas de partage du trafic → canary sur Kubernetes.

**Équipe :** Eliott M., Clément, Lucas, Kelsy Nguyen (rôles dans le README). **Pour aller plus loin :** [business case](BUSINESS_CASE.md), [rapport d'incident](../incident/INCIDENT_REPORT.md).

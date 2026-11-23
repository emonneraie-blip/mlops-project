# Runbook — que faire quand une alerte part

Une fiche par alerte, en cinq lignes : symptôme, causes probables, première vérification, action (et qui décide), responsable.
Règle générale : on revient en arrière d'abord si des utilisateurs sont touchés, on enquête ensuite, on réentraîne en dernier et seulement si le gate passe.
Toutes les actions laissent une trace dans `monitoring/events.log` (`make events`).

| Rôle | Responsable |
|---|---|
| Rollback (décide seul, sans validation) | Lucas |
| API et service | Clément |
| Drift et réentraînement | Eliott |
| Qualité des données | Kelsy |
| Suppléant sur tous les rôles | Louis |

## 1. Drift réel — `drift.py` : STATUT ALERT, raisons `input` ou `context`

- **Symptôme** : PSI ≥ 0,25 sur `lag_1`, `lag_24` ou les prédictions, sans raison `data_quality`.
- **Causes probables** : la demande a changé (confinement, grève, météo, nouvel usage).
- **Première vérification** : le rapport `monitoring/drift_report.json` ; les médianes sont plausibles et aucune zone n'est muette ; la MAE de la fenêtre (les labels arrivent une heure après la prévision).
- **Action** : `make retrain` (cooldown 24 h, gate sur les 7 derniers jours, puis une personne valide). Le rollback n'aide pas : toutes les versions précédentes ont appris l'ancien régime.
- **Responsable** : drift et réentraînement. La promotion est validée par une deuxième personne.

## 2. Données cassées — `drift.py` : raison `data_quality`

- **Symptôme** : plus de 5 % de valeurs manquantes, médiane hors plage plausible, ou zone active devenue muette (`silent_zones`).
- **Causes probables** : changement d'unité ou de schéma en amont (tarif en centimes, nouvelle colonne), flux d'une zone coupé.
- **Première vérification** : la colonne citée dans le rapport, sur les fichiers bruts du mois (`data/raw`).
- **Action** : ne pas réentraîner (`retrain.py` refuse déjà). Corriger la source. Si une version a été entraînée sur ces données : rollback (fiche 4).
- **Responsable** : qualité des données.

## 3. API en erreur — `/v1/health` en 503, ou part de 5xx en hausse dans les logs `"event": "request"`

- **Symptôme** : 503 (pas de modèle chargé) ou 500 sur `/v1/predict`.
- **Causes probables** : MLflow injoignable au démarrage, alias `@production` absent, version incompatible avec le schéma.
- **Première vérification** : `GET /v1/model` (version servie), logs `model_not_loaded`, `curl $MLFLOW_TRACKING_URI/health`.
- **Action** : relancer MLflow puis `POST /v1/reload`. Si les 500 viennent d'une version récente : rollback (fiche 4).
- **Responsable** : API et service.

## 4. Nouvelle version moins bonne — décision de rollback

- **Symptôme** : quand les labels arrivent, la MAE de la nouvelle version dépasse celle de l'ancienne ; ou 5xx en hausse après une promotion ; ou données cassées apprises par la dernière version.
- **Causes probables** : la version a passé le gate sur une fenêtre trop étroite (incident 2), ou a appris sur des données fausses.
- **Première vérification** : `make events` (quelle version, promue quand) et la même requête sur les deux versions.
- **Action** : `make rollback`, sans attendre de validation : revenir à la dernière version validée ne fait jamais pire qu'elle. Vérifier ensuite `/v1/model` et `/v1/health`. Si la sortie dit « l'API n'a PAS rechargé », appeler `POST /v1/reload` jusqu'à ce qu'il réponde 200. Objectif < 5 minutes (1,5 s mesurées).
- **Responsable** : rollback. Un deuxième `make rollback` est refusé (la version retirée porte le tag `rolled_back`) : pour la remettre, `--version N` ou `make promote`.

## 5. Latence — P95 au-dessus de 200 ms

- **Symptôme** : `make load-test` hors budget, ou `latency_ms` des logs `"event": "request"` en hausse.
- **Causes probables** : saturation CPU (prédictions concurrentes qui se disputent les cœurs), machine trop petite, trop de requêtes.
- **Première vérification** : P95 par niveau de concurrence dans `make load-test` ; essayer `OMP_NUM_THREADS=1`.
- **Action** : plus de workers ou de réplicas (chaque worker charge sa copie du modèle : la mémoire suit), ou réduire le trafic.
- **Responsable** : API et service.

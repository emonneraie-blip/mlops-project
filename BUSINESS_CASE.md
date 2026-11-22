# Business case — prévision de la demande de taxis à New York

Un opérateur de taxis doit savoir, chaque heure, combien de prises en charge chaque zone de New York verra l'heure suivante, pour envoyer les chauffeurs là où seront les clients. Notre modèle divise par deux l'erreur de la règle naïve (MAE de 4,68 contre 9,26 courses par heure sur février 2020). Mais le modèle n'est pas le sujet : ce que nous avons construit, c'est la couche qui le rend fiable en production.

## Le besoin

**La décision que la prévision alimente.** Où positionner les chauffeurs pour l'heure suivante, zone par zone. Les deux erreurs n'ont pas le même coût :

- **Sous-estimer** : les clients attendent ou renoncent, les courses sont perdues.
- **Surestimer** : des chauffeurs restent inactifs dans une zone sans demande.

Les données TLC ne contiennent aucun coût : nous n'avons donc pas chiffré ces erreurs. Nous utilisons la MAE en courses par heure, qui se lit directement : « on se trompe de N courses par zone et par heure ».

**Pourquoi un bon modèle ne suffit pas.** Personne ne fait confiance à un modèle en production si personne ne peut dire quelles données l'ont entraîné, s'il fonctionne encore, ni comment revenir en arrière. Chacune de ces questions a une réponse dans le dépôt :

| Question de l'opérateur | Ce qui y répond | Preuve |
|---|---|---|
| Quelles données et quel code ont produit ce modèle ? | Chaque run MLflow porte `git_commit`, `git_dirty`, `data_md5` ; les données sont sous DVC (`dvc.lock`) | Run v1 : `git_dirty=False` |
| Fonctionne-t-il encore ? | `src/drift.py` : PSI sur les entrées et les prédictions, contrôles de qualité des données | Fenêtre Covid : PSI 0,637 sur `lag_1` ; février 2020 (contrôle) : tous les PSI sous 0,01 |
| Peut-on revenir en arrière ? | `src/rollback.py` : `@production` revient sur `@previous`, rien n'est supprimé | Rollback v2 → v1 en 1,5 s, objectif < 5 min |

**Le test de résistance que les données offrent.** De février à avril 2020, les courses passent de 6,3 millions à 0,24 million par mois (−96 %). Un modèle entraîné sur des mois normaux rencontre un monde qu'il n'a jamais vu : c'est l'incident que nous détectons, auquel nous réagissons, et que nous savons annuler.

Le schéma du système est dans le [README](../README.md#le-système).

## Choix de conception et compromis acceptés

| Domaine | Choix | Rejeté | Pourquoi | Prix accepté |
|---|---|---|---|---|
| Découpage | Par le temps : entraînement nov. à janv., golden set février | Découpage aléatoire | Un découpage aléatoire met des heures futures dans l'entraînement et flatte le score | Moins de saisons vues (pas d'été) |
| Modèle | `HistGradientBoostingRegressor`, `random_state=42`, `early_stopping=False` | Modèle profond ou optimisé | Rapide et déterministe : même MAE à 1e-6 près ; l'early stopping tirerait un jeu de validation au hasard | L'identifiant de zone est traité comme un nombre, pas comme une catégorie |
| Baseline | Même heure la veille (`lag_24`) | Pas de baseline | Un modèle doit battre la règle évidente pour mériter sa place | — |
| Métrique | MAE en courses par heure (RMSE aussi loggée) | MAPE | La MAPE divise par zéro sur les heures sans course, nombreuses | La MAE baisse quand le volume baisse (4,68 → 1,16 pendant le Covid) : pas comparable d'un régime à l'autre |
| Registry | Alias `@staging`, `@production`, `@previous` | Stages, chemins de fichiers | Les stages sont dépréciés dans MLflow ; promouvoir et revenir en arrière = déplacer un alias | — |
| Gate | La candidate doit battre la baseline ET la production sur le même golden set, puis une personne valide | Promotion automatique | Le gate décide sur des chiffres ; une personne assume la mise en production | Le gate ne voit que la fenêtre qu'on lui donne (incident 2) |
| Mesure du drift | Le PSI décide (0,10 à surveiller, 0,25 pour agir) ; KS affiché | p-value du test KS | Sur un million de lignes, toute p-value est minuscule ; le PSI mesure la taille de l'écart | Le PSI est une convention, pas un test statistique |
| Type d'alerte | Drift réel et données cassées sont séparés | Réentraîner à chaque alerte | Réentraîner sur une colonne cassée apprendrait l'erreur au modèle | Une plage plausible par colonne, fixée à la main |
| Référence | La période d'entraînement du modèle en production, lue dans son run MLflow | Référence fixe | Après un réentraînement, le drift se mesure par rapport à ce que le nouveau modèle a vu | — |
| Réentraînement | 3 dernières semaines, gate sur les 7 derniers jours, cooldown de 24 h | Tout l'historique, réentraîner à chaque alerte | S'adapte vite à un nouveau régime ; le cooldown évite les tempêtes d'alertes | Oublie le régime normal (incident 2) |
| Rollback | Une commande, sans validation, rien n'est supprimé | Rollback après accord | Revenir à la dernière version validée ne peut pas faire pire qu'elle ; la vitesse compte | — |
| Features au service | L'appelant envoie les comptages récents ; l'API réutilise `features.py` | Calculer les lags dans l'API | Même code de features des deux côtés : pas d'écart entraînement / service | Pas de feature store : l'appelant doit connaître les comptages récents |
| Remote des données | Dossier DVC local | S3 / MinIO | Pas d'identifiants cloud à gérer | `dvc pull` échoue depuis un clone neuf ; `dvc repro` retélécharge |
| Reload raté | L'API garde le modèle en place et répond 503 à l'appelant du reload | Passer en 503 partout | Un reload raté ne doit pas couper un service qui fonctionne | Pendant un rollback avec MLflow injoignable, le modèle fautif sert jusqu'au prochain essai |
| CI | Tests, puis image taguée avec le SHA du commit ; actions épinglées par SHA ; jeton en lecture seule | Push et déploiement depuis la CI | Chaque changement est testé et constructible | L'image est construite et testée au démarrage (smoke test), mais pas poussée vers un registre |

## Les incidents en bref

Trois incidents, trois bonnes réactions différentes : réentraîner quand le monde change, revenir en arrière quand une version se comporte mal, ne pas toucher au modèle quand la donnée est cassée. Détails et chronologie : [`incident/INCIDENT_REPORT.md`](../incident/INCIDENT_REPORT.md).

| | Ce qui s'est passé | Ce qui l'a vu | Réaction | Chiffres |
|---|---|---|---|---|
| 1. Drift réel | Effondrement de la demande à partir de mi-mars 2020 | PSI sur les entrées et les prédictions | Réentraînement (3 au 23 avril), gate (24 au 30 avril), promotion de la v2 | PSI `lag_1` 0,637, prédictions 0,625 ; MAE v2 0,589 |
| 2. Risque de version | La v2, apprise sur le confinement, divise par deux une prévision normale | Même requête envoyée à la v1 et à la v2 | Rollback vers la v1 | Zone 132, 20 février 18 h : v1 87,0, v2 39,8 ; rollback en 1,5 s |
| 3. Données cassées | Tarif en centimes ; zone 132 (JFK) muette | Médiane plausible ; contrôle par zone | Pas de réentraînement ; corriger la source | PSI tarif 11,63, médiane 1 380 $ ; la zone muette ne fait bouger aucun PSI |

**Limite assumée de l'incident 2.** En avril 2020, la demande *était* effondrée : la v2 collait au monde de ce moment-là. Le gate l'a validée sur 7 jours du même régime que son entraînement, il ne pouvait donc pas voir le risque d'une reprise de la demande. La requête de février illustre ce risque ; le vrai déclencheur du rollback, écrit dans le README, est une MAE de la nouvelle version au-dessus de l'ancienne quand les labels arrivent (par exemple sur une période de reprise). Leçon : un modèle qui passe le gate sur une fenêtre peut se tromper sur la suivante, donc le rollback doit toujours rester à une commande.

## Limites et suite

| Limite aujourd'hui | Conséquence | Avec un mois de plus |
|---|---|---|
| Remote DVC et artefacts MLflow dans des dossiers locaux | `dvc pull` échoue ailleurs ; un seul portable détient l'historique | S3 ou MinIO pour les deux |
| Contrôle de drift lancé à la main | Personne n'est prévenu la nuit | DAG Airflow chaque nuit avec `drift.py --fail-on-alert` ; réentraînement et gate dans GitHub Actions avec validation obligatoire |
| Gate sur une seule fenêtre du nouveau régime | Une candidate peut passer et échouer quand la demande reprend | Une deuxième fenêtre de référence du régime normal ; WAPE en plus de la MAE |
| Pas de métriques en continu | Latence et taux d'erreur non surveillés | Prometheus et Grafana sur les quatre couches : infra, API, données, modèle |
| Bascule d'alias, sans partage du trafic | Une nouvelle version reçoit 100 % du trafic d'un coup | Canary sur Kubernetes, rollback par `kubectl rollout undo` |
| L'appelant envoie les lags | Les clients doivent suivre les comptages récents | Feature store ou agrégation en ligne du flux de courses |
| Identifiant de zone traité comme un nombre | Les arbres traitent la zone 132 comme « plus grande » que la zone 50 | Zone en variable catégorielle ; un modèle par groupe de zones |
| `/v1/reload` ouvert à tous | N'importe qui sur le réseau peut déclencher un rechargement | Authentification sur les routes d'administration |
| La CI ne pousse l'image vers aucun registre | Pas d'image versionnée prête à déployer | Push vers un registre (GHCR) après les tests |
| Météo et jours fériés ignorés | Des facteurs connus de la demande manquent | Features météo et jours fériés |

## Sources

- Données : NYC Taxi & Limousine Commission (TLC), Trip Record Data, Yellow Taxi — https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- Chiffres des incidents : runs du 20 novembre 2026, [`incident/INCIDENT_REPORT.md`](../incident/INCIDENT_REPORT.md)
- Le cadrage métier de la première partie (qui utilise la prévision, ce que coûtent les erreurs) est notre lecture : les données TLC ne contiennent aucune information de coût.

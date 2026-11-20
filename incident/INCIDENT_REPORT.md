# Rapport d'incident — le jour où la demande s'est effondrée

Chiffres issus des runs du 20 novembre 2026 : `monitoring/events.log`, sorties de `drift.py` et UI MLflow (expériences `taxi-demand` et `taxi-demand-monitoring`).

## Résumé

| | |
|---|---|
| Incident 1 | Drift réel : effondrement de la demande à partir de mi-mars 2020 (confinement) |
| Détection | `drift.py` : PSI = 0,637 sur `lag_1`, 0,625 sur les prédictions (seuil d'action 0,25) |
| Réaction | Réentraînement sur 3 semaines récentes, gate passé, promotion de la v2 |
| Incident 2 | La v2, apprise sur le confinement, sous-estime de moitié une demande normale : rollback vers la v1 en 1,5 s |
| Incident 3 | Données cassées (tarif en centimes, zone 132 muette) : détectées, pas de réentraînement |

## Contexte

Données NYC TLC Yellow Taxi, agrégées par zone et par heure (1 100 400 lignes, novembre 2019 à avril 2020).

| Mois | Courses |
|---|---|
| Février 2020 | 6 299 064 |
| Mars 2020 | 3 007 261 |
| Avril 2020 | 237 941 (-96 % par rapport à février) |

Le modèle v1 est entraîné du 8 novembre 2019 au 31 janvier 2020. Sur le golden set de février 2020, sa MAE est de 4,68 courses par heure, contre 9,26 pour la baseline « même heure la veille ». Le run est propre (`git_dirty=False`) et porte son commit et l'empreinte des données.

## Incident 1 — drift réel (Covid)

**Contrôle témoin** (`make drift-ok`, février 2020) : tous les PSI sont sous 0,01, statut OK. Le monitoring ne donne pas de fausse alerte sur une période normale.

**Détection** (`make drift`, fenêtre 15 mars → 30 avril 2020) :

| Colonne | Type | PSI | KS | Niveau |
|---|---|---|---|---|
| lag_1 | entrée | 0,637 | 0,244 | ALERT |
| lag_24 | entrée | 0,568 | 0,241 | ALERT |
| avg_distance | contexte | 0,063 | 0,097 | OK |
| avg_fare | contexte | 0,402 | 0,260 | ALERT |
| prédiction | sortie | 0,625 | 0,256 | ALERT |

**Diagnostic.** Les entrées du modèle et ses prédictions changent de distribution, sans problème de qualité : pas de valeurs manquantes, médianes plausibles, aucune zone muette. C'est le monde qui a changé, donc réentraîner est la bonne réaction.

**Piège évité.** Sur la fenêtre Covid, la MAE de la v1 tombe à 1,16 (contre 4,68). Le modèle n'est pas meilleur : il y a 25 fois moins de courses, donc des erreurs plus petites en valeur absolue. Surveiller la MAE seule aurait masqué l'incident ; le PSI l'a vu.

**Réaction** (`make retrain`) :
- Réentraînement du 3 au 23 avril 2020 ; golden set du 24 au 30 avril 2020.
- Gate : candidate v2 MAE = 0,589 ; production v1 MAE = [valeur affichée par make retrain] ; la v2 bat la v1 et la baseline.
- Validation humaine, puis `@production` → v2 et `@previous` → v1. L'API recharge sans redémarrer.

## Incident 2 — le modèle réentraîné ne tient pas sur une demande normale

Même requête envoyée à l'API (zone 132, 20 février 2020 à 18 h, 85 courses l'heure précédente) :

| Version | Prévision |
|---|---|
| v1 (apprise sur une demande normale) | 87,0 courses |
| v2 (apprise sur le confinement) | 39,8 courses |

La v2 a appris que la demande était effondrée. Si la demande repart, elle sous-estime de moitié et l'opérateur envoie deux fois trop peu de chauffeurs.

**Rollback** (`make rollback`) : `@production` v2 → v1 en 1,5 seconde, objectif < 5 minutes. L'API sert la v1 dès le rechargement. La v2 n'est pas supprimée : elle reste dans le Registry sous l'alias `@previous`.

## Incident 3 — données cassées

| Incident injecté | Ce que voit `drift.py` | Décision |
|---|---|---|
| Tarif en centimes (`make inject-fare`) | PSI `avg_fare` = 11,63, médiane 1 380 $ hors plage plausible | Qualité de données : pas de réentraînement |
| Zone 132 (JFK) muette (`make inject-zone`) | 1 zone muette : [132], tous les PSI restent sous 0,01 | Qualité de données : pas de réentraînement |

Dans le cas de la zone muette, aucun PSI ne bouge : une zone sur 260 pèse trop peu sur la distribution globale. Seul le contrôle par zone l'a détectée.

Réentraîner sur ces données aurait appris l'erreur au modèle. La réaction est de corriger la source, puis de faire un rollback si un modèle a déjà appris sur ces données.

## Chronologie (`monitoring/events.log`)

| Heure | Événement |
|---|---|
| 16:22:44 | Promotion v1 en production (MAE 4,68) |
| 16:24:52 | Alerte drift, fenêtre Covid |
| 16:25:48 | Réentraînement déclenché |
| 16:25:55 | Promotion v2 en production (MAE 0,589 sur le golden set) |
| 16:26:15 | Rollback v2 → v1 |
| 16:26:21 | Alerte qualité : tarif hors plage |
| 16:26:26 | Alerte qualité : zone 132 muette |

## Ce que nous retenons

- Sans labels, on détecte qu'une distribution change, pas que le modèle se trompe. Et une MAE qui baisse peut cacher un incident.
- Une alerte ne dit pas quoi faire : il faut distinguer drift réel et données cassées.
- Un modèle qui passe le gate sur une fenêtre peut être mauvais sur la suivante. Le rollback doit rester disponible à tout moment.
- Un rollback qu'on n'a pas testé n'est pas un rollback : il a été joué en 1,5 seconde.

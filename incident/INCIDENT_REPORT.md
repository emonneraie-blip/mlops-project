# Rapport d'incident — le jour où la demande s'est effondrée

> Les valeurs entre crochets viennent de vos propres runs : `monitoring/drift_report.json`, `monitoring/events.log` et l'UI MLflow (http://localhost:5050).

## Résumé

| | |
|---|---|
| Incident 1 | Drift réel : effondrement de la demande à partir de mi-mars 2020 (confinement) |
| Détection | `drift.py`, PSI sur `lag_1` = [à compléter] (seuil d'action 0,25) |
| Réaction | Réentraînement sur les 3 dernières semaines, gate passé, promotion v[à compléter] |
| Incident 2 | Données cassées : tarif en centimes, puis zone 132 (JFK) muette |
| Réaction | Pas de réentraînement ; rollback testé en [à compléter] s |

## Incident 1 — drift réel (Covid)

**Contexte.** Le modèle v1 est entraîné du 8 novembre 2019 au 31 janvier 2020. Sur le golden set de février : MAE = [à compléter], baseline lag_24 = [à compléter].

**Détection** (`make drift`, fenêtre 15 mars → 30 avril 2020) :

| Colonne | PSI | KS | Niveau |
|---|---|---|---|
| lag_1 | [ ] | [ ] | [ ] |
| lag_24 | [ ] | [ ] | [ ] |
| avg_distance | [ ] | [ ] | [ ] |
| avg_fare | [ ] | [ ] | [ ] |
| prédiction | [ ] | [ ] | [ ] |

Contrôle témoin sur février 2020 (`make drift-ok`) : statut [OK], ce qui montre que l'alerte ne se déclenche pas sur une période normale.

**Diagnostic.** Les entrées du modèle et ses prédictions changent de distribution, sans problème de qualité (pas de valeurs manquantes, médianes plausibles, aucune zone muette). C'est le monde qui a changé : réentraîner est la bonne réaction.

**Réaction** (`make retrain`) :
- Réentraînement sur [dates train] ; golden set [dates golden].
- Gate : candidate v[ ] MAE = [ ] ; production v1 MAE = [ ] ; baseline = [ ]. Gate passé.
- Validation humaine, puis `@production` → v[ ], `@previous` → v1. L'API recharge le modèle sans redémarrer.
- Un second déclenchement juste après est bloqué par le cooldown de 24 h.

## Incident 2 — données cassées

**Tarif en centimes** (`make inject-fare`) : PSI sur `avg_fare` = [ ], médiane = [ ] $, hors plage plausible.
**Zone muette** (`make inject-zone`) : la zone 132 n'a remonté aucune course sur la fenêtre.

**Décision.** Ce sont des alertes de qualité de données : `retrain.py` refuse de réentraîner. Réentraîner aurait appris l'erreur au modèle. La bonne réaction est de corriger la source. Si un modèle avait déjà été réentraîné sur ces données, on fait un rollback.

**Rollback** (`make rollback`) : `@production` v[ ] → v[ ] en [ ] s. L'API sert l'ancienne version dès le rechargement. La version fautive reste dans le Registry, sous l'alias `@previous`.

## Chronologie (extrait de `monitoring/events.log`)

```
[coller ici les lignes du journal]
```

## Ce que nous retenons

- Sans labels, on détecte qu'une distribution a changé, pas que le modèle se trompe. La MAE ne se lit qu'une fois les labels arrivés.
- Une alerte ne dit pas quoi faire : il faut d'abord distinguer drift réel et données cassées.
- Le gate décide de la promotion, pas l'alarme. Une personne valide.
- Un rollback qu'on n'a pas testé n'est pas un rollback : il a été joué pendant la préparation et pendant la démo.

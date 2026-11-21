# Business case — forecasting taxi demand in New York

*Version française : [BUSINESS_CASE.md](BUSINESS_CASE.md)*

A taxi operator needs to know, every hour, how many pickups each New York zone will see in the next hour, so it can send drivers where riders will be. Our model halves the error of the naive rule (MAE of 4.68 vs 9.26 pickups per hour on February 2020). But the model is not the point: what we built is the layer that makes it trustworthy in production.

## The need

**The decision the forecast feeds.** Where to position drivers for the next hour, zone by zone. The two errors do not cost the same:

- **Under-forecast:** riders wait or give up, and the trips are lost.
- **Over-forecast:** drivers sit idle in a zone with no demand.

The TLC data contains no cost information, so we did not price these errors. We use MAE in pickups per hour, which reads directly: "we are wrong by N trips per zone per hour".

**Why a good model is not enough.** Nobody trusts a model in production if nobody can say which data trained it, whether it still works, or how to undo it. Each of these questions has an answer in the repository:

| Operator's question | What answers it | Evidence |
|---|---|---|
| Which data and code produced this model? | Every MLflow run carries `git_commit`, `git_dirty`, `data_md5`; data is under DVC (`dvc.lock`) | Run v1: `git_dirty=False` |
| Does it still work? | `src/drift.py`: PSI on inputs and predictions, data-quality checks | Covid window: PSI 0.637 on `lag_1`; February 2020 (control): every PSI below 0.01 |
| Can we undo it? | `src/rollback.py`: `@production` goes back to `@previous`, nothing is deleted | Rollback v2 → v1 in 1.5 s, target under 5 min |

**The stress test the data gives us.** From February to April 2020, trips fall from 6.3 million to 0.24 million a month (−96%). A model trained on normal months meets a world it has never seen: that is the incident we detect, react to, and know how to undo.

The system diagram is in the [README](../README.md#le-système).

## Design choices and the trade-offs we accepted

| Area | Choice | Rejected | Why | Price we accept |
|---|---|---|---|---|
| Split | By time: train Nov to Jan, golden set February | Random split | A random split puts future hours in training and flatters the score | Fewer seasons seen (no summer) |
| Model | `HistGradientBoostingRegressor`, `random_state=42`, `early_stopping=False` | A deep or tuned model | Fast and deterministic: same MAE to 1e-6; early stopping would draw a random validation set | The zone ID is treated as a number, not as a category |
| Baseline | Same hour the day before (`lag_24`) | No baseline | A model must beat the obvious rule to earn its place | — |
| Metric | MAE in pickups per hour (RMSE also logged) | MAPE | MAPE divides by zero on hours with no trip, which are frequent | MAE falls when volume falls (4.68 → 1.16 during Covid): not comparable across regimes |
| Registry | Aliases `@staging`, `@production`, `@previous` | Stages, file paths | Stages are deprecated in MLflow; promoting and rolling back = moving an alias | — |
| Gate | The candidate must beat the baseline AND production on the same golden set, then a person approves | Automatic promotion | The gate decides on numbers; a person owns the release | The gate only sees the window it is given (incident 2) |
| Drift measure | PSI decides (0.10 watch, 0.25 act); KS shown | KS test p-value | On a million rows every p-value is tiny; PSI measures the size of the shift | PSI is a convention, not a statistical test |
| Alert type | Real drift and broken data are kept apart | Retrain on every alert | Retraining on a broken column would teach the error to the model | A plausible range per column, set by hand |
| Reference | The production model's training period, read from its MLflow run | A fixed reference | After a retrain, drift is measured against what the new model saw | — |
| Retraining | Last 3 weeks, gate on the last 7 days, 24 h cooldown | Full history, retrain on every alert | Adapts fast to a new regime; the cooldown prevents alert storms | Forgets the normal regime (incident 2) |
| Rollback | One command, no approval, nothing deleted | Rollback after sign-off | Going back to the last validated version cannot do worse than it; speed matters | — |
| Features at serving | The caller sends recent counts; the API reuses `features.py` | Compute lags inside the API | Same feature code on both sides: no training/serving skew | No feature store: the caller must know the recent counts |
| Data remote | Local DVC folder | S3 / MinIO | No cloud credentials to manage | `dvc pull` fails from a fresh clone; `dvc repro` downloads again |
| Failed reload | The API keeps its current model and answers 503 to the reload caller | Fall to 503 everywhere | A failed reload must not take down a working service | During a rollback with MLflow unreachable, the faulty model serves until the next attempt |
| CI | Tests, then an image tagged with the commit SHA; actions pinned by SHA; read-only token | Push and deploy from CI | Every change is tested and buildable | The image is built but neither pushed nor started in CI |

## The incidents in brief

Three incidents, three different right answers: retrain when the world changes, roll back when a version misbehaves, leave the model alone when the data is broken. Details and timeline: [`incident/INCIDENT_REPORT.md`](../incident/INCIDENT_REPORT.md).

| | What happened | What saw it | Reaction | Figures |
|---|---|---|---|---|
| 1. Real drift | Demand collapses from mid-March 2020 | PSI on inputs and predictions | Retrain (3 to 23 April), gate (24 to 30 April), promote v2 | PSI `lag_1` 0.637, predictions 0.625; v2 MAE 0.589 |
| 2. Release risk | v2, trained on the lockdown, halves a normal forecast | Same request sent to v1 and v2 | Roll back to v1 | Zone 132, 20 February 18:00: v1 87.0, v2 39.8; rollback in 1.5 s |
| 3. Broken data | Fare in cents; zone 132 (JFK) silent | Plausible median; per-zone check | No retraining; fix the source | Fare PSI 11.63, median $1,380; the silent zone moves no PSI |

**An acknowledged limit of incident 2.** In April 2020 demand *was* collapsed: v2 matched the world at that moment. The gate validated it on 7 days from the same regime as its training, so it could not see the risk of demand recovering. The February request illustrates that risk; the real rollback trigger, written in the README, is the new version's MAE rising above the old one's once labels arrive (for example on a recovery period). Lesson: a model that passes the gate on one window can be wrong on the next, so rollback must always stay one command away.

## Limits and next steps

| Limit today | Consequence | With another month |
|---|---|---|
| DVC remote and MLflow artifacts in local folders | `dvc pull` fails elsewhere; one laptop holds the history | S3 or MinIO for both |
| Drift check run by hand | Nobody is warned at night | Nightly Airflow DAG with `drift.py --fail-on-alert`; retraining and gate in GitHub Actions with required approval |
| Gate on a single window from the new regime | A candidate can pass and fail when demand recovers | A second reference window from the normal regime; WAPE alongside MAE |
| No continuous metrics | Latency and error rate are not watched | Prometheus and Grafana on the four layers: infrastructure, API, data, model |
| Alias switch, no traffic split | A new version gets 100% of traffic at once | Canary on Kubernetes, rollback with `kubectl rollout undo` |
| The caller sends the lags | Clients must track recent counts | A feature store or online aggregation of the trip stream |
| Zone ID treated as a number | Trees treat zone 132 as "bigger" than zone 50 | Zone as a categorical variable; one model per group of zones |
| `/v1/reload` open to anyone | Anyone on the network can trigger a reload | Authentication on admin routes |
| CI builds the image without starting it | A container that crashes at startup goes unnoticed | Smoke test (`/v1/health` must answer 503 without a model), push to a registry |
| Weather and holidays ignored | Known demand drivers are missing | Weather and public-holiday features |

## Sources

- Data: NYC Taxi & Limousine Commission (TLC), Trip Record Data, Yellow Taxi — https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- Incident figures: runs of 20 November 2026, [`incident/INCIDENT_REPORT.md`](../incident/INCIDENT_REPORT.md)
- The business framing in the first part (who uses the forecast, what errors cost) is our own reading: the TLC data contains no cost information.

# One-page summary — NYC taxi demand, the MLOps layer

*Version française : [SUMMARY.md](SUMMARY.md)*

**In one sentence.** A service that forecasts next-hour taxi pickups for each New York zone, built so that we can say which data trained it, see when it stops working, and undo a release in seconds.

## Business

- **Who uses it:** a taxi operator deciding where to position drivers for the next hour, zone by zone.
- **What an error costs:** under-forecast, riders wait and trips are lost; over-forecast, drivers sit idle. Not priced: the TLC data has no cost information.
- **Value:** MAE of 4.68 pickups per zone per hour on February 2020, against 9.26 for "same hour yesterday": about half the error.
- **Risk the layer manages:** from February to April 2020, trips fell by 96% (6.3 million to 0.24 million a month). A model trained on normal months met a world it had never seen.

## Technical

| Layer | What we built | Tools |
|---|---|---|
| Data | TLC Yellow Taxi, Nov 2019 to Apr 2020, one row per zone and hour (1,100,400 rows); lags 1, 2, 24, 168 h and calendar features; split by time: train Nov to Jan, golden set Feb, drift 15 Mar to 30 Apr | DVC, pandas |
| Training | `HistGradientBoostingRegressor`, fixed seed; every run tagged `git_commit`, `git_dirty`, `data_md5` | scikit-learn, MLflow |
| Release | Aliases `@staging`, `@production`, `@previous`; gate: beat the baseline and production on the same golden set, then a person approves | MLflow Registry |
| Serving | `/v1/predict` with a strict contract (422), `/v1/health` (503 without a model), the model version in every answer; a failed reload keeps the current model | FastAPI, Pydantic, Docker |
| Monitoring | PSI per input and on predictions (0.10 watch, 0.25 act), KS shown; data quality: missing values, implausible medians, silent zones | scipy, MLflow |
| Reaction | Real drift: retrain on 3 weeks, 24 h cooldown, gate. Broken data: no retraining. Rollback: one command | `retrain.py`, `rollback.py` |
| Quality | Tests run in CI: API contract, lags, PSI, rollback, gate, drift vs broken data; actions pinned by SHA | pytest, GitHub Actions |

## Results (runs of 20 November 2026)

- **Covid drift detected:** PSI 0.637 on `lag_1` and 0.625 on predictions; on February (control) every PSI stays under 0.01.
- **Retrain, then rollback:** v2 (MAE 0.589 on 24 to 30 April) was promoted, then rolled back to v1 in 1.5 s because it halves a normal forecast (39.8 vs 87.0 for the same request).
- **Broken data, no retraining:** a fare in cents and a silent JFK zone were both detected and classified as data quality.

## Run it

```bash
git clone https://github.com/emonneraie-blip/mlops-project && cd mlops-project
cp .env.example .env && uv sync --locked
make mlflow                                   # terminal 1
make data && make train && make promote       # terminal 2
make serve                                    # terminal 3, then: make request
```

## Limits and next steps

Local DVC remote (`dvc pull` fails from a fresh clone) → S3 or MinIO. Drift check run by hand → nightly Airflow job. Gate on a single window → a second golden window from the normal regime, and WAPE. No live metrics → Prometheus and Grafana. No traffic split → canary on Kubernetes.

**Team:** Eliott M., Clément, Lucas, Kelsy Nguyen (roles in the README). **More:** [business case](BUSINESS_CASE_EN.md), [incident report](../incident/INCIDENT_REPORT.md).

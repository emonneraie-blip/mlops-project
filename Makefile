# Raccourcis. Sans make (Windows), copier la commande de la ligne du dessous.
PY = uv run python

.PHONY: setup mlflow data train promote serve request test drift drift-ok inject-zone inject-fare retrain rollback events up

setup:        ## installe l'environnement figé par uv.lock
	uv sync --locked

mlflow:       ## serveur MLflow local (terminal 1)
	mkdir -p mlflow_data && uv run mlflow server --host 127.0.0.1 --port 5050 --backend-store-uri sqlite:///mlflow_data/mlflow.db --artifacts-destination ./mlflow_data/artifacts

data:         ## télécharge les fichiers TLC et construit les features (pipeline DVC)
	uv run dvc repro

train:        ## entraîne et enregistre une version @staging
	$(PY) src/train.py

promote:      ## gate + validation humaine, puis @staging -> @production
	$(PY) src/promote.py

serve:        ## API sur http://localhost:8000/docs (terminal 2)
	uv run uvicorn api.main:app --port 8000

request:      ## une requête de démo
	curl -s -X POST http://localhost:8000/v1/predict -H "Content-Type: application/json" -d '{"pu_location_id": 132, "pickup_hour": "2020-02-20T18:00:00", "lag_1": 85, "lag_2": 90, "lag_24": 80, "lag_168": 78}'

test:
	uv run pytest -q

drift-ok:     ## contrôle sur février 2020 : pas de drift attendu
	$(PY) src/drift.py --start 2020-02-01 --end "2020-02-29 23:00"

drift:        ## contrôle sur la période Covid (mi-mars à avril 2020)
	$(PY) src/drift.py

inject-zone:  ## incident : la zone 132 (JFK) ne remonte plus rien
	$(PY) src/inject_drift.py zone_silent && $(PY) src/drift.py --injected

inject-fare:  ## incident : tarif en centimes au lieu de dollars
	$(PY) src/inject_drift.py broken_fare && $(PY) src/drift.py --injected

retrain:      ## déclencheur : drift -> réentraînement -> gate -> promotion
	$(PY) src/retrain.py

rollback:     ## remet @production sur @previous
	$(PY) src/rollback.py

events:       ## journal des promotions, alertes et rollbacks
	cat monitoring/events.log

up:           ## tout en Docker (MLflow + API)
	docker compose up --build

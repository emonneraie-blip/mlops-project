"""API de prévision : nombre de prises en charge dans une zone pour l'heure demandée.
Le modèle est chargé par nom et alias (models:/taxi-demand@production), jamais par chemin de fichier.
"""
import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

# MLflow réessaie chaque requête HTTP avec un backoff exponentiel (environ 4 min au total par défaut).
# Pour l'API, c'est trop long : MLflow injoignable bloquerait le démarrage (et donc le 503 attendu)
# et chaque /v1/reload pendant plusieurs minutes. On échoue vite ; reste réglable par variable d'environnement.
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", "1")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")

import mlflow
from fastapi import FastAPI, HTTPException, Request
from mlflow import MlflowClient
from pydantic import BaseModel, ConfigDict, Field

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from config import MLFLOW_TRACKING_URI, MODEL_NAME  # noqa: E402
from features import request_to_frame  # noqa: E402

# Logs structurés : une ligne JSON par prédiction, sans donnée personnelle
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("taxi-api")

state = {"model": None, "version": None}


def load_production_model():
    """Charge la version qui porte l'alias @production. Renvoie True si le chargement a réussi.
    - On charge le numéro de version lu, pas l'alias une deuxième fois : si l'alias bouge entre
      les deux appels, la version annoncée serait différente du modèle réellement servi.
    - Si le chargement échoue (MLflow injoignable...), on garde le modèle déjà en place :
      un reload raté ne doit pas couper une API qui fonctionne. Sans modèle du tout, l'API reste en 503.
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()
        version = str(client.get_model_version_by_alias(MODEL_NAME, "production").version)
        model = mlflow.pyfunc.load_model("models:/" + MODEL_NAME + "/" + version)
    except Exception as error:
        logger.info(json.dumps({"event": "model_not_loaded", "kept_version": state["version"],
                                "error": str(error)[:200]}))
        return False
    state["model"] = model
    state["version"] = version
    logger.info(json.dumps({"event": "model_loaded", "version": version}))
    return True


@asynccontextmanager
async def lifespan(app):
    load_production_model()
    yield


app = FastAPI(title="NYC Taxi Demand API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Couche API du monitoring : une ligne JSON par requête, avec le code HTTP et la latence.
    C'est le signal du rollback « erreurs 5xx en hausse » (README) et de la latence P95."""
    started = time.perf_counter()
    status = 500  # une exception non gérée dans la route finit en 500 : on la compte aussi
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Process-Time-Ms"] = str(round((time.perf_counter() - started) * 1000, 1))
        return response
    finally:
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        logger.info(json.dumps({"event": "request", "method": request.method, "path": request.url.path,
                                "status": status, "latency_ms": latency_ms, "model_version": state["version"]}))


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")  # un champ inconnu (faute de frappe) = erreur 422

    pu_location_id: int = Field(ge=1, le=265, examples=[132], description="zone TLC de prise en charge")
    pickup_hour: datetime = Field(examples=["2020-02-20T18:00:00"], description="heure à prévoir")
    lag_1: float = Field(ge=0, examples=[85], description="prises en charge 1 h avant")
    lag_2: float = Field(ge=0, examples=[90], description="prises en charge 2 h avant")
    lag_24: float = Field(ge=0, examples=[80], description="même heure la veille")
    lag_168: float = Field(ge=0, examples=[78], description="même heure la semaine précédente")


class PredictionResponse(BaseModel):
    request_id: str
    pu_location_id: int
    pickup_hour: datetime
    predicted_pickups: float
    model_version: str


@app.get("/v1/health")
def health():
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Aucun modèle @production chargé")
    return {"status": "ok", "model_name": MODEL_NAME, "model_version": state["version"]}


@app.get("/v1/model")
def model_info():
    return {"model_name": MODEL_NAME, "alias": "production", "model_version": state["version"],
            "tracking_uri": MLFLOW_TRACKING_URI}


@app.post("/v1/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Aucun modèle @production chargé")

    features = request_to_frame(request.pu_location_id, request.pickup_hour,
                                request.lag_1, request.lag_2, request.lag_24, request.lag_168)
    prediction = float(state["model"].predict(features)[0])
    prediction = max(0.0, round(prediction, 1))  # pas de demande négative

    request_id = str(uuid.uuid4())
    logger.info(json.dumps({
        "event": "prediction", "request_id": request_id, "model_version": state["version"],
        "pu_location_id": request.pu_location_id, "pickup_hour": request.pickup_hour.isoformat(),
        "prediction": prediction,
    }))
    return PredictionResponse(request_id=request_id, pu_location_id=request.pu_location_id,
                              pickup_hour=request.pickup_hour, predicted_pickups=prediction,
                              model_version=str(state["version"]))


@app.post("/v1/reload")
def reload_model():
    """Appelé après une promotion ou un rollback : recharge la version @production."""
    old_version = state["version"]
    if not load_production_model():
        # 503 : l'appelant (promote.py, rollback.py) doit savoir que le changement n'est PAS en ligne
        raise HTTPException(status_code=503, detail={"error": "reload impossible, l'API garde sa version",
                                                     "model_version": old_version})
    return {"old_version": old_version, "new_version": state["version"]}

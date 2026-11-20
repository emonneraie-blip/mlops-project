"""API de prévision : nombre de prises en charge dans une zone pour l'heure demandée.
Le modèle est chargé par nom et alias (models:/taxi-demand@production), jamais par chemin de fichier.
"""
import json
import logging
import os
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

import mlflow
from fastapi import FastAPI, HTTPException
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
    """Charge la version qui porte l'alias @production. En cas d'échec, l'API reste en 503."""
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = MlflowClient()
        version = str(client.get_model_version_by_alias(MODEL_NAME, "production").version)
        state["model"] = mlflow.pyfunc.load_model("models:/" + MODEL_NAME + "@production")
        state["version"] = version
        logger.info(json.dumps({"event": "model_loaded", "version": version}))
    except Exception as error:
        state["model"] = None
        state["version"] = None
        logger.info(json.dumps({"event": "model_not_loaded", "error": str(error)[:200]}))


@asynccontextmanager
async def lifespan(app):
    load_production_model()
    yield


app = FastAPI(title="NYC Taxi Demand API", version="1.0.0", lifespan=lifespan)


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
    load_production_model()
    return {"old_version": old_version, "new_version": state["version"]}

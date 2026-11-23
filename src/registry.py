"""Fonctions communes autour du MLflow Model Registry (alias, chargement, évaluation, journal)."""
import json
import os
from datetime import datetime

import mlflow
import numpy as np
import requests
from mlflow import MlflowClient

from config import API_URL, MLFLOW_TRACKING_URI, MODEL_NAME, MONITORING_DIR
from features import TARGET, to_model_input


def get_client():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return MlflowClient()


def get_alias_version(client, alias):
    """Renvoie le numéro de version derrière un alias (@production, @staging...), ou None."""
    try:
        return str(client.get_model_version_by_alias(MODEL_NAME, alias).version)
    except Exception:
        return None


def load_model(version):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    return mlflow.sklearn.load_model("models:/" + MODEL_NAME + "/" + str(version))


def evaluate(model, df):
    y_true = df[TARGET].values
    y_pred = model.predict(to_model_input(df))
    errors = y_true - y_pred
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    return {"mae": mae, "rmse": rmse}


def naive_mae(df):
    """Baseline naïve : même heure la veille (lag_24)."""
    return float(np.mean(np.abs(df[TARGET].values - df["lag_24"].values)))


def log_event(action, details):
    """Journal des opérations (promotion, rollback, alerte). Une ligne JSON par événement."""
    os.makedirs(MONITORING_DIR, exist_ok=True)
    line = {"time": datetime.now().isoformat(timespec="seconds"), "action": action}
    line.update(details)
    with open(os.path.join(MONITORING_DIR, "events.log"), "a") as f:
        f.write(json.dumps(line) + "\n")


def notify_api():
    """Demande à l'API de recharger le modèle @production.
    Renvoie True seulement si l'API sert la nouvelle version. Un 503 de /v1/reload veut dire
    que l'API n'a PAS rechargé et sert toujours l'ancienne version : il ne faut pas l'afficher comme un succès.
    """
    try:
        response = requests.post(API_URL + "/v1/reload", timeout=30)
    except requests.RequestException:
        print("API non joignable : relancez-la ou appelez POST /v1/reload")
        return False
    if response.status_code != 200:
        print("ATTENTION : l'API n'a PAS rechargé (HTTP " + str(response.status_code) + ") :", response.text[:200])
        print("Elle sert toujours l'ancienne version. Relancez POST /v1/reload quand MLflow répond.")
        return False
    print("API rechargée :", response.json())
    return True

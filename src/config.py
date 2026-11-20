"""Lecture de params.yaml et du .env, partagée par tous les scripts."""
import os

import yaml
from dotenv import load_dotenv

load_dotenv()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")
PROCESSED_FILE = os.path.join(ROOT, "data", "processed", "features.parquet")
DRIFT_FILE = os.path.join(ROOT, "data", "drift", "current.parquet")
MONITORING_DIR = os.path.join(ROOT, "monitoring")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5050")
MODEL_NAME = os.getenv("MODEL_NAME", "taxi-demand")
API_URL = os.getenv("API_URL", "http://localhost:8000")


def load_params():
    with open(os.path.join(ROOT, "params.yaml")) as f:
        return yaml.safe_load(f)

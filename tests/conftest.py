"""Les tests n'ont besoin ni des vraies données ni d'un serveur MLflow :
on entraîne un petit modèle sur des données synthétiques, dans un MLflow temporaire."""
import os
import tempfile

TMP_DIR = tempfile.mkdtemp()
os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///" + os.path.join(TMP_DIR, "mlflow.db")
os.environ["MODEL_NAME"] = "taxi-demand-test"

import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from mlflow import MlflowClient  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

from features import TARGET, build_features, to_model_input  # noqa: E402


def make_hourly(n_days=21, zones=(1, 2, 132), seed=0):
    """Données zone x heure synthétiques avec un cycle journalier."""
    rng = np.random.default_rng(seed)
    hours = pd.date_range("2020-01-01", periods=24 * n_days, freq="h")
    rows = []
    for zone in zones:
        for h in hours:
            base = 20 + 15 * np.sin(2 * np.pi * h.hour / 24) + zone % 7
            rows.append({"pu_location_id": zone, "pickup_hour": h,
                         "pickups": max(0, round(base + rng.normal(0, 3))),
                         "avg_distance": 2.5 + rng.normal(0, 0.3), "avg_fare": 12 + rng.normal(0, 1)})
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def features_df():
    return build_features(make_hourly())


@pytest.fixture(scope="session")
def registered_model(features_df):
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    experiment_id = mlflow.create_experiment("tests", artifact_location=os.path.join(TMP_DIR, "artifacts"))
    model = HistGradientBoostingRegressor(max_iter=20, random_state=0)
    model.fit(to_model_input(features_df), features_df[TARGET])
    x = to_model_input(features_df.head(5))
    signature = infer_signature(x, model.predict(x))  # même signature qu'en vrai : le test vérifie les types
    with mlflow.start_run(experiment_id=experiment_id):
        info = mlflow.sklearn.log_model(model, name="model", registered_model_name=os.environ["MODEL_NAME"],
                                        signature=signature,
                                        skops_trusted_types=["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"])
    MlflowClient().set_registered_model_alias(os.environ["MODEL_NAME"], "production", info.registered_model_version)
    return info.registered_model_version

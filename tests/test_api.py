"""Tests d'intégration de l'API : vrai modèle chargé depuis le Registry, vraies requêtes HTTP."""
from fastapi.testclient import TestClient

VALID = {"pu_location_id": 132, "pickup_hour": "2020-02-20T18:00:00",
         "lag_1": 85, "lag_2": 90, "lag_24": 80, "lag_168": 78}


def get_client(registered_model):
    from api.main import app
    return TestClient(app)


def test_health_ok(registered_model):
    with get_client(registered_model) as client:
        response = client.get("/v1/health")
        assert response.status_code == 200
        assert response.json()["model_version"] == str(registered_model)


def test_predict_ok(registered_model):
    with get_client(registered_model) as client:
        response = client.post("/v1/predict", json=VALID)
        assert response.status_code == 200
        body = response.json()
        assert body["predicted_pickups"] >= 0
        assert body["model_version"] == str(registered_model)
        assert len(body["request_id"]) > 0


def test_unknown_field_is_refused(registered_model):
    with get_client(registered_model) as client:
        request = dict(VALID)
        request["lag_one"] = 3  # faute de frappe
        assert client.post("/v1/predict", json=request).status_code == 422


def test_invalid_values_are_refused(registered_model):
    with get_client(registered_model) as client:
        request = dict(VALID)
        request["lag_1"] = -5
        assert client.post("/v1/predict", json=request).status_code == 422
        request = dict(VALID)
        request["pu_location_id"] = 999
        assert client.post("/v1/predict", json=request).status_code == 422


def test_missing_field_is_refused(registered_model):
    with get_client(registered_model) as client:
        request = dict(VALID)
        del request["lag_168"]
        assert client.post("/v1/predict", json=request).status_code == 422


def test_503_without_model(registered_model):
    from api import main
    with get_client(registered_model) as client:
        main.state["model"] = None
        assert client.get("/v1/health").status_code == 503
        assert client.post("/v1/predict", json=VALID).status_code == 503
        # le reload recharge @production et l'API repart
        assert client.post("/v1/reload").json()["new_version"] == str(registered_model)
        assert client.get("/v1/health").status_code == 200


def test_failed_reload_keeps_serving(registered_model, monkeypatch):
    """MLflow injoignable pendant un reload : l'API garde l'ancien modèle au lieu de tomber en 503."""
    from api import main
    with get_client(registered_model) as client:
        def mlflow_down(*args, **kwargs):
            raise ConnectionError("MLflow injoignable")
        monkeypatch.setattr(main.mlflow.pyfunc, "load_model", mlflow_down)

        assert client.post("/v1/reload").status_code == 503  # l'appelant sait que rien n'a changé
        assert client.get("/v1/health").status_code == 200   # mais l'API sert toujours
        assert client.post("/v1/predict", json=VALID).json()["model_version"] == str(registered_model)





def test_unreachable_mlflow_fails_fast(registered_model, monkeypatch):
    """Serveur MLflow éteint : le reload doit répondre 503 en quelques secondes, pas après ~4 min de retries."""
    import os
    import time

    from api import main
    real_uri = os.environ["MLFLOW_TRACKING_URI"]  # set_tracking_uri réécrit aussi la variable d'environnement
    with get_client(registered_model) as client:
        monkeypatch.setattr(main, "MLFLOW_TRACKING_URI", "http://127.0.0.1:1")  # port fermé
        start = time.monotonic()
        response = client.post("/v1/reload")
        elapsed = time.monotonic() - start
        main.mlflow.set_tracking_uri(real_uri)  # rendre le vrai MLflow aux autres tests

        assert response.status_code == 503
        assert elapsed < 30, f"reload bloqué {elapsed:.0f} s"
        assert client.get("/v1/health").status_code == 200  # l'ancien modèle sert toujours

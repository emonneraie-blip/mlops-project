"""Tests des deux décisions opérationnelles : le rollback et le gate de promotion.
Même MLflow temporaire que conftest.py : pas de serveur, pas de vraies données."""
import os

import mlflow
from mlflow import MlflowClient

import promote as promote_module
import registry
import rollback as rollback_module

MODEL = os.environ["MODEL_NAME"]


def register_copy(registered_model):
    """Enregistre une deuxième version du même modèle, pour avoir deux versions à échanger."""
    source = MlflowClient().get_model_version(MODEL, registered_model).source
    return str(mlflow.register_model(source, MODEL).version)


def test_rollback_swaps_production_and_previous(registered_model, monkeypatch):
    monkeypatch.setattr(rollback_module, "notify_api", lambda: None)  # pas d'API lancée pendant les tests
    client = MlflowClient()
    old = str(registered_model)
    new = register_copy(registered_model)
    client.set_registered_model_alias(MODEL, "previous", old)
    client.set_registered_model_alias(MODEL, "production", new)

    assert rollback_module.rollback(reason="test") is True
    assert registry.get_alias_version(client, "production") == old
    assert registry.get_alias_version(client, "previous") == new  # la version fautive reste inspectable

    client.set_registered_model_alias(MODEL, "production", old)  # remet l'état attendu par les autres tests


def test_gate_refuses_a_candidate_worse_than_the_baseline(registered_model, features_df, monkeypatch):
    monkeypatch.setattr(promote_module, "notify_api", lambda: None)
    monkeypatch.setattr(promote_module, "evaluate", lambda model, df: {"mae": 1e9, "rmse": 1e9})
    client = MlflowClient()
    production_before = registry.get_alias_version(client, "production")
    candidate = register_copy(registered_model)

    data_file = os.path.join(os.path.dirname(os.environ["MLFLOW_TRACKING_URI"].removeprefix("sqlite:///")),
                             "features.parquet")
    features_df.to_parquet(data_file, index=False)
    start, end = str(features_df["pickup_hour"].min()), str(features_df["pickup_hour"].max())

    assert promote_module.promote(candidate, start, end, yes=True, data_file=data_file) is False
    assert registry.get_alias_version(client, "production") == production_before  # rien n'a bougé

"""repro_check.py : un run se reproduit à 1e-6 depuis le même commit et les mêmes données,
et des données différentes sont détectées."""
import os

import mlflow

import repro_check
import train


def train_test_run(data_file):
    """Un run d'entraînement sur les données synthétiques, artefacts dans le dossier temporaire des tests."""
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    if mlflow.get_experiment_by_name("taxi-demand") is None:
        tmp_dir = os.path.dirname(os.environ["MLFLOW_TRACKING_URI"].removeprefix("sqlite:///"))
        mlflow.create_experiment("taxi-demand", artifact_location=os.path.join(tmp_dir, "artifacts-train"))
    train.train_model("2020-01-08", "2020-01-15 23:00", "2020-01-16", "2020-01-21 23:00",
                      data_file=data_file, reason="test")
    runs = mlflow.search_runs(experiment_names=["taxi-demand"], order_by=["start_time DESC"])
    return runs.iloc[0]["run_id"]


def test_run_is_reproduced_and_changed_data_is_detected(features_df, tmp_path, monkeypatch):
    monkeypatch.setattr(train, "git_info", lambda: ("abc123", False))
    monkeypatch.setattr(repro_check, "git_info", lambda: ("abc123", False))
    data_file = str(tmp_path / "features.parquet")
    features_df.to_parquet(data_file, index=False)

    run_id = train_test_run(data_file)

    assert repro_check.check(run_id, data_file=data_file) is True

    changed = features_df.copy()
    changed.loc[0, "pickups"] += 1  # une seule valeur différente
    changed.to_parquet(data_file, index=False)
    assert repro_check.check(run_id, data_file=data_file) is False


def test_other_commit_is_not_reproduced(features_df, tmp_path, monkeypatch):
    monkeypatch.setattr(train, "git_info", lambda: ("abc123", False))
    data_file = str(tmp_path / "features.parquet")
    features_df.to_parquet(data_file, index=False)
    run_id = train_test_run(data_file)

    monkeypatch.setattr(repro_check, "git_info", lambda: ("def456", False))
    assert repro_check.check(run_id, data_file=data_file) is False

"""Entraîne le modèle, logge tout dans MLflow et enregistre une nouvelle version dans le Registry.
La nouvelle version reçoit l'alias @staging. Le passage en @production se fait avec promote.py.
"""
import argparse
import hashlib
import subprocess

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingRegressor

from config import MODEL_NAME, PROCESSED_FILE, load_params
from features import TARGET, select_period, to_model_input
from registry import evaluate, get_client, naive_mae

# MLflow sérialise le modèle avec skops (format sûr) : on déclare le type interne du modèle comme fiable
TRUSTED_TYPES = ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]


def git_info():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = status != ""
    except Exception:
        commit = "unknown"
        dirty = True
    return commit, dirty


def file_md5(path):
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            md5.update(chunk)
    return md5.hexdigest()


def fit_and_evaluate(df, train_start, train_end, test_start, test_end, model_params):
    """Entraînement + évaluation, sans MLflow. Utilisé par train_model ET par repro_check.py :
    le même code produit le modèle et le vérifie."""
    train = select_period(df, train_start, train_end)
    test = select_period(df, test_start, test_end)
    model = HistGradientBoostingRegressor(
        max_iter=model_params["max_iter"],
        learning_rate=model_params["learning_rate"],
        max_depth=model_params["max_depth"],
        random_state=model_params["random_state"],
        early_stopping=False,
    )
    model.fit(to_model_input(train), train[TARGET])
    metrics = evaluate(model, test)
    metrics["baseline_mae_lag24"] = naive_mae(test)
    return model, metrics, train, test


def train_model(train_start, train_end, test_start, test_end, data_file=PROCESSED_FILE, reason="initial"):
    params = load_params()
    model_params = params["model"]
    client = get_client()

    commit, dirty = git_info()
    if dirty:
        print("ATTENTION : des fichiers ne sont pas commités, le run ne sera pas reproductible (git_dirty=True)")

    df = pd.read_parquet(data_file)
    model, metrics, train, test = fit_and_evaluate(df, train_start, train_end, test_start, test_end, model_params)
    print("Train :", len(train), "lignes | Test :", len(test), "lignes")

    mlflow.set_experiment("taxi-demand")
    with mlflow.start_run(run_name="train-" + reason):
        mlflow.log_params(model_params)
        mlflow.log_params({
            "train_start": train_start, "train_end": train_end,
            "test_start": test_start, "test_end": test_end,
            "n_train": len(train), "n_test": len(test),
        })
        mlflow.log_metrics({
            "test_mae": metrics["mae"],
            "test_rmse": metrics["rmse"],
            "baseline_mae_lag24": metrics["baseline_mae_lag24"],
        })
        mlflow.set_tags({
            "git_commit": commit,
            "git_dirty": str(dirty),
            "data_md5": file_md5(data_file),
            "reason": reason,
        })
        x_example = to_model_input(test.head(5))
        signature = infer_signature(x_example, model.predict(x_example))
        info = mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature,
            input_example=x_example,
            registered_model_name=MODEL_NAME,
            skops_trusted_types=TRUSTED_TYPES,
        )

    version = str(info.registered_model_version)
    client.set_registered_model_alias(MODEL_NAME, "staging", version)
    print("Version", version, "enregistrée avec l'alias @staging")
    print("MAE test :", round(metrics["mae"], 3), "| RMSE :", round(metrics["rmse"], 3),
          "| baseline lag_24 :", round(metrics["baseline_mae_lag24"], 3))
    return version, metrics


def main():
    params = load_params()
    split = params["split"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-start", default=split["train_start"])
    parser.add_argument("--train-end", default=split["train_end"])
    parser.add_argument("--test-start", default=split["test_start"])
    parser.add_argument("--test-end", default=split["test_end"])
    args = parser.parse_args()
    train_model(args.train_start, args.train_end, args.test_start, args.test_end)


if __name__ == "__main__":
    main()

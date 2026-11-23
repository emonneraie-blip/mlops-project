"""Test de reproductibilité d'un run MLflow, à 1e-6 près.
Usage : python src/repro_check.py RUN_ID        (ou : make repro-check RUN=<run_id>)

Réentraîne depuis le checkout ACTUEL avec les paramètres enregistrés dans le run (pas ceux de
params.yaml : on veut ce qui a vraiment servi), puis compare à ce que MLflow a gardé :
  code    : le commit actuel est-il celui du run (git_commit), sans fichier modifié ?
  données : l'empreinte de features.parquet est-elle celle du run (data_md5) ?
  modèle  : test_mae, test_rmse et baseline_mae_lag24 identiques à 1e-6 près ?
Ne crée ni run ni version de modèle. Code de sortie 1 si une vérification échoue (utilisable en CI).
"""
import argparse
import sys

import mlflow
import pandas as pd

from config import MLFLOW_TRACKING_URI, PROCESSED_FILE, load_params
from train import file_md5, fit_and_evaluate, git_info

TOLERANCE = 1e-6
LOGGED_METRICS = {"test_mae": "mae", "test_rmse": "rmse", "baseline_mae_lag24": "baseline_mae_lag24"}


def run_model_params(run_params):
    """Les paramètres sont stockés en texte dans MLflow : on les remet dans le type de params.yaml."""
    reference = load_params()["model"]
    return {key: type(default)(run_params[key]) for key, default in reference.items() if key in run_params}


def check(run_id, data_file=PROCESSED_FILE):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    run = mlflow.get_run(run_id)
    tags, run_params, logged = run.data.tags, run.data.params, run.data.metrics

    commit, dirty = git_info()
    checks = {
        "code     (git_commit " + tags.get("git_commit", "?")[:8] + ")": commit == tags.get("git_commit"),
        "code     (aucun fichier modifié)": not dirty,
        "données  (data_md5)": file_md5(data_file) == tags.get("data_md5"),
    }

    _, metrics, _, _ = fit_and_evaluate(pd.read_parquet(data_file),
                                        run_params["train_start"], run_params["train_end"],
                                        run_params["test_start"], run_params["test_end"],
                                        run_model_params(run_params))
    for logged_name, name in LOGGED_METRICS.items():
        delta = abs(metrics[name] - logged[logged_name])
        checks["modèle   " + logged_name.ljust(18) + " écart = " + format(delta, ".1e")] = delta < TOLERANCE

    for label, ok in checks.items():
        print("PASS" if ok else "FAIL", label)
    reproduced = all(checks.values())
    print("\nREPRODUIT" if reproduced else "\nNON REPRODUCTIBLE")
    return reproduced


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", help="identifiant du run MLflow (onglet du run dans l'UI, ou tag run_id de la version)")
    args = parser.parse_args()
    sys.exit(0 if check(args.run_id) else 1)


if __name__ == "__main__":
    main()

"""Monitoring du drift : compare une fenêtre récente à la période de référence (celle de l'entraînement).
- PSI (0,10 à surveiller, 0,25 pour agir) et statistique KS sur chaque colonne
- PSI sur les prédictions du modèle en production
- qualité de données : valeurs manquantes, médiane hors plage plausible, zones devenues muettes
Les deux types d'alerte n'appellent pas la même réaction :
  drift "input"/"context" -> le monde a changé -> réentraîner
  "data_quality"          -> la donnée est cassée -> corriger la source / rollback, surtout pas réentraîner
Sans labels, on détecte que quelque chose a changé, pas que le modèle se trompe.
"""
import argparse
import json
import os
import sys

import mlflow
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from config import DRIFT_FILE, MODEL_NAME, MONITORING_DIR, PROCESSED_FILE, load_params
from features import select_period, to_model_input
from registry import evaluate, get_alias_version, get_client, load_model, log_event

INPUT_COLUMNS = ["lag_1", "lag_24"]             # entrées du modèle
CONTEXT_COLUMNS = ["avg_distance", "avg_fare"]  # pas utilisées par le modèle, mais elles décrivent les courses
# Plage plausible de la médiane : en dehors, c'est un problème de données (unité, colonne cassée), pas un vrai changement
PLAUSIBLE_MEDIAN = {"avg_distance": (0.1, 50), "avg_fare": (2, 150)}


def psi(reference, current, n_bins=10):
    """Population Stability Index, avec des bins calés sur les quantiles de la référence."""
    edges = np.quantile(reference, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    edges[0] = -np.inf
    edges[-1] = np.inf
    ref_counts, _ = np.histogram(reference, edges)
    cur_counts, _ = np.histogram(current, edges)
    ref_pct = np.clip(ref_counts / len(reference), 1e-6, None)
    cur_pct = np.clip(cur_counts / len(current), 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def level(value, params):
    if value >= params["drift"]["psi_act"]:
        return "ALERT"
    if value >= params["drift"]["psi_watch"]:
        return "WATCH"
    return "OK"


def check_drift(current, reference, params, model=None):
    n_bins = params["drift"]["n_bins"]
    report = {"checks": [], "silent_zones": [], "status": "OK", "reasons": []}

    # 1. Drift sur les entrées du modèle et sur les colonnes de contexte
    for column in INPUT_COLUMNS + CONTEXT_COLUMNS:
        ref_values = reference[column]
        cur_values = current[column]
        if column in CONTEXT_COLUMNS:
            # distance et tarif n'existent que sur les heures avec au moins une course
            ref_values = ref_values[reference["pickups"] > 0]
            cur_values = cur_values[current["pickups"] > 0]
        missing_rate = float(cur_values.isna().mean())
        ref_values = ref_values.dropna()
        cur_values = cur_values.dropna()
        value = psi(ref_values, cur_values, n_bins)
        ks = float(ks_2samp(ref_values, cur_values).statistic)
        kind = "input" if column in INPUT_COLUMNS else "context"
        report["checks"].append({"column": column, "kind": kind, "psi": value, "ks": ks,
                                 "missing_rate": missing_rate, "level": level(value, params)})
        if level(value, params) == "ALERT":
            report["reasons"].append({"kind": kind, "text": column + " : PSI = " + str(round(value, 3))})
        if missing_rate > params["drift"]["max_missing_rate"]:
            report["reasons"].append({"kind": "data_quality",
                                      "text": column + " : " + str(round(100 * missing_rate, 1)) + " % de valeurs manquantes"})
        if column in PLAUSIBLE_MEDIAN and len(cur_values) > 0:
            low, high = PLAUSIBLE_MEDIAN[column]
            median = float(cur_values.median())
            if median < low or median > high:
                report["reasons"].append({"kind": "data_quality",
                                          "text": column + " : médiane " + str(round(median, 2)) + " hors plage plausible"})

    # 2. Drift sur les prédictions du modèle en production
    if model is not None:
        ref_pred = model.predict(to_model_input(reference))
        cur_pred = model.predict(to_model_input(current))
        value = psi(ref_pred, cur_pred, n_bins)
        ks = float(ks_2samp(ref_pred, cur_pred).statistic)
        report["checks"].append({"column": "prediction", "kind": "input", "psi": value, "ks": ks,
                                 "missing_rate": 0.0, "level": level(value, params)})
        if level(value, params) == "ALERT":
            report["reasons"].append({"kind": "input", "text": "prediction : PSI = " + str(round(value, 3))})
        # Performance : possible seulement si les labels de la fenêtre sont arrivés
        report["current_mae"] = evaluate(model, current)["mae"]
        report["reference_mae"] = evaluate(model, reference)["mae"]

    # 3. Zones devenues muettes : une zone active dans la référence ne remonte plus aucune course
    ref_mean = reference.groupby("pu_location_id")["pickups"].mean()
    cur_total = current.groupby("pu_location_id")["pickups"].sum()
    for zone in ref_mean.index:
        if ref_mean[zone] < params["drift"]["silent_zone_min_mean"]:
            continue
        if zone in cur_total.index and cur_total[zone] == 0:
            report["silent_zones"].append(int(zone))
    if len(report["silent_zones"]) > 0:
        report["reasons"].append({"kind": "data_quality",
                                  "text": str(len(report["silent_zones"])) + " zone(s) muette(s) : "
                                  + str(report["silent_zones"][:10])})

    # Statut global
    for check in report["checks"]:
        if check["level"] == "WATCH" and report["status"] == "OK":
            report["status"] = "WATCH"
    if len(report["reasons"]) > 0:
        report["status"] = "ALERT"
    return report


def print_report(report):
    print("")
    print("colonne        type          PSI      KS     manquants  niveau")
    for c in report["checks"]:
        print(c["column"].ljust(14), c["kind"].ljust(13), str(round(c["psi"], 3)).ljust(8),
              str(round(c["ks"], 3)).ljust(6), (str(round(100 * c["missing_rate"], 1)) + " %").ljust(10), c["level"])
    if "current_mae" in report:
        print("MAE production : référence", round(report["reference_mae"], 2),
              "| fenêtre", round(report["current_mae"], 2))
    print("")
    print("STATUT :", report["status"])
    for reason in report["reasons"]:
        print("  -", reason["kind"], ":", reason["text"])


def reference_window(client, production, split):
    """La référence = la période sur laquelle le modèle en production a été entraîné (lue dans MLflow).
    Après un réentraînement, la référence suit donc le nouveau modèle."""
    if production is not None:
        try:
            run_id = client.get_model_version(MODEL_NAME, production).run_id
            run_params = client.get_run(run_id).data.params
            return run_params["train_start"], run_params["train_end"]
        except Exception:
            pass
    return split["train_start"], split["train_end"]


def run_drift_check(start=None, end=None, data_file=None):
    params = load_params()
    split = params["split"]
    client = get_client()
    production = get_alias_version(client, "production")
    ref_start, ref_end = reference_window(client, production, split)
    reference = select_period(pd.read_parquet(PROCESSED_FILE), ref_start, ref_end)
    if data_file is not None:
        current = pd.read_parquet(data_file)
        window = data_file
    else:
        current = select_period(pd.read_parquet(PROCESSED_FILE), start, end)
        window = start + " -> " + end
    print("Référence :", ref_start, "->", ref_end, "| Fenêtre :", window, "(", len(current), "lignes )")

    model = None
    if production is not None:
        model = load_model(production)
    report = check_drift(current, reference, params, model)
    report["window"] = window
    report["production_version"] = production
    print_report(report)

    # Sauvegarde du rapport + suivi dans MLflow
    os.makedirs(MONITORING_DIR, exist_ok=True)
    with open(os.path.join(MONITORING_DIR, "drift_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    mlflow.set_experiment("taxi-demand-monitoring")
    with mlflow.start_run(run_name="drift-check"):
        mlflow.set_tags({"status": report["status"], "window": window, "production_version": str(production)})
        for c in report["checks"]:
            mlflow.log_metric("psi_" + c["column"], c["psi"])
            mlflow.log_metric("missing_" + c["column"], c["missing_rate"])
        mlflow.log_metric("n_silent_zones", len(report["silent_zones"]))
    if report["status"] == "ALERT":
        log_event("drift_alert", {"window": window, "reasons": [r["text"] for r in report["reasons"]]})
    return report


def main():
    split = load_params()["split"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=split["drift_start"])
    parser.add_argument("--end", default=split["drift_end"])
    parser.add_argument("--injected", action="store_true", help="contrôler le fichier produit par inject_drift.py")
    parser.add_argument("--fail-on-alert", action="store_true", help="code de sortie 1 si ALERT (pour un cron / la CI)")
    args = parser.parse_args()

    data_file = None
    if args.injected:
        data_file = DRIFT_FILE
    report = run_drift_check(args.start, args.end, data_file)
    if args.fail_on_alert and report["status"] == "ALERT":
        sys.exit(1)


if __name__ == "__main__":
    main()

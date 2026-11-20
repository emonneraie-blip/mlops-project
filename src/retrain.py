"""Déclencheur de réentraînement.
1. contrôle de drift sur la fenêtre récente
2. si l'alerte vient de la qualité des données -> on ne réentraîne PAS (on corrigerait le modèle sur des données fausses)
3. si l'alerte vient d'un vrai changement de distribution -> cooldown, puis réentraînement sur les semaines récentes
4. gate sur les derniers jours (golden set) + validation humaine avant @production
"""
import argparse
import json
import os
from datetime import datetime, timedelta

import pandas as pd

from config import DRIFT_FILE, MONITORING_DIR, load_params
from drift import run_drift_check
from promote import promote
from registry import log_event
from train import train_model

STATE_FILE = os.path.join(MONITORING_DIR, "state.json")


def in_cooldown(hours):
    if not os.path.exists(STATE_FILE):
        return False
    with open(STATE_FILE) as f:
        state = json.load(f)
    last = datetime.fromisoformat(state["last_retrain"])
    return datetime.now() - last < timedelta(hours=hours)


def save_state():
    os.makedirs(MONITORING_DIR, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump({"last_retrain": datetime.now().isoformat(timespec="seconds")}, f)


def main():
    params = load_params()
    split = params["split"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=split["drift_start"])
    parser.add_argument("--end", default=split["drift_end"])
    parser.add_argument("--injected", action="store_true")
    parser.add_argument("--force", action="store_true", help="ignorer le cooldown")
    parser.add_argument("--yes", action="store_true", help="promotion sans question")
    args = parser.parse_args()

    data_file = None
    if args.injected:
        data_file = DRIFT_FILE
    report = run_drift_check(args.start, args.end, data_file)

    if report["status"] != "ALERT":
        print("\nPas d'alerte : pas de réentraînement")
        return

    quality_problem = False
    for reason in report["reasons"]:
        if reason["kind"] == "data_quality":
            quality_problem = True
    if quality_problem:
        print("\nINCIDENT QUALITÉ DE DONNÉES : pas de réentraînement.")
        print("Action : corriger la source, ou rollback si une version récente a appris sur ces données.")
        log_event("retrain_skipped", {"why": "data_quality"})
        return

    if in_cooldown(params["retrain"]["cooldown_hours"]) and not args.force:
        print("\nCooldown actif : un réentraînement a déjà eu lieu récemment (--force pour passer outre)")
        return

    # Fenêtres : [train sur N semaines] puis [golden set sur les derniers jours]
    end = pd.Timestamp(args.end)
    golden_start = end.floor("D") - timedelta(days=params["retrain"]["golden_days"] - 1)
    train_start = golden_start - timedelta(weeks=params["retrain"]["train_weeks"])
    train_end = golden_start - timedelta(hours=1)
    print("\nRéentraînement : train", train_start, "->", train_end, "| golden", golden_start, "->", end)

    log_event("retrain_triggered", {"reasons": [r["text"] for r in report["reasons"]]})
    version, metrics = train_model(str(train_start), str(train_end), str(golden_start), str(end), reason="drift")
    save_state()
    promote(version, str(golden_start), str(end), args.yes)


if __name__ == "__main__":
    main()

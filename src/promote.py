"""Fait passer une version en @production, après un gate et une validation humaine.
Gate : la candidate doit battre la baseline naïve ET la version en production, sur un golden set.
L'ancienne version de production garde l'alias @previous (utilisé par rollback.py).
"""
import argparse

import pandas as pd

from config import MODEL_NAME, PROCESSED_FILE, load_params
from features import select_period
from registry import evaluate, get_alias_version, get_client, load_model, log_event, naive_mae, notify_api


def promote(version, start, end, yes=False, data_file=PROCESSED_FILE):
    client = get_client()
    version = str(version)
    production = get_alias_version(client, "production")
    if production == version:
        print("La version", version, "est déjà en production")
        return False

    golden = select_period(pd.read_parquet(data_file), start, end)
    print("Golden set :", start, "->", end, "(", len(golden), "lignes )")

    candidate_mae = evaluate(load_model(version), golden)["mae"]
    baseline_mae = naive_mae(golden)
    print("Candidate v" + version, "MAE =", round(candidate_mae, 3))
    print("Baseline lag_24      MAE =", round(baseline_mae, 3))

    if candidate_mae >= baseline_mae:
        print("GATE REFUSÉ : la candidate ne bat pas la baseline naïve")
        log_event("gate_refused", {"version": version, "candidate_mae": candidate_mae, "baseline_mae": baseline_mae})
        return False

    if production is not None:
        production_mae = evaluate(load_model(production), golden)["mae"]
        print("Production v" + production, "MAE =", round(production_mae, 3))
        if candidate_mae >= production_mae:
            print("GATE REFUSÉ : la candidate ne bat pas la production")
            log_event("gate_refused", {"version": version, "candidate_mae": candidate_mae, "production_mae": production_mae})
            return False

    print("GATE OK")
    if not yes:
        answer = input("Promouvoir la version " + version + " en production ? [o/N] ")
        if answer.strip().lower() != "o":
            print("Promotion annulée")
            return False

    if production is not None:
        client.set_registered_model_alias(MODEL_NAME, "previous", production)
    client.set_registered_model_alias(MODEL_NAME, "production", version)
    if "rolled_back" in client.get_model_version(MODEL_NAME, version).tags:
        client.delete_model_version_tag(MODEL_NAME, version, "rolled_back")  # repassée par le gate
    print("@production -> v" + version, "| @previous -> v" + str(production))
    api_reloaded = notify_api()
    log_event("promote", {"from_version": production, "to_version": version, "candidate_mae": candidate_mae,
                          "api_reloaded": api_reloaded})
    return True


def main():
    split = load_params()["split"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", help="numéro de version ; par défaut celle qui porte @staging")
    parser.add_argument("--start", default=split["test_start"], help="début du golden set")
    parser.add_argument("--end", default=split["test_end"], help="fin du golden set")
    parser.add_argument("--yes", action="store_true", help="valider sans question (à éviter)")
    args = parser.parse_args()

    version = args.version
    if version is None:
        version = get_alias_version(get_client(), "staging")
    if version is None:
        print("Aucune version @staging : lancez d'abord train.py")
        return
    promote(version, args.start, args.end, args.yes)


if __name__ == "__main__":
    main()

"""Injecte un incident dans une fenêtre de données saine, pour tester le monitoring.
Scénarios :
  zone_silent  : une zone ne remonte plus aucune course (capteur / flux coupé)
  broken_fare  : le tarif arrive en centimes au lieu de dollars (changement d'unité en amont)
Le drift réel (Covid, mars-avril 2020) n'a pas besoin d'injection : drift.py le lit directement.
"""
import argparse
import os

import numpy as np
import pandas as pd

from config import DRIFT_FILE, PROCESSED_FILE
from features import select_period


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", choices=["zone_silent", "broken_fare"])
    parser.add_argument("--zone", type=int, default=132, help="zone à couper (132 = JFK Airport)")
    parser.add_argument("--start", default="2020-02-22")
    parser.add_argument("--end", default="2020-02-29 23:00")
    args = parser.parse_args()

    df = select_period(pd.read_parquet(PROCESSED_FILE), args.start, args.end)

    if args.scenario == "zone_silent":
        rows = df["pu_location_id"] == args.zone
        for column in ["pickups", "lag_1", "lag_2", "lag_24", "lag_168"]:
            df.loc[rows, column] = 0
        df.loc[rows, "avg_distance"] = np.nan
        df.loc[rows, "avg_fare"] = np.nan
        print("Zone", args.zone, "coupée sur", int(rows.sum()), "lignes")

    if args.scenario == "broken_fare":
        df["avg_fare"] = df["avg_fare"] * 100
        print("avg_fare multiplié par 100 (dollars -> centimes)")

    os.makedirs(os.path.dirname(DRIFT_FILE), exist_ok=True)
    df.to_parquet(DRIFT_FILE, index=False)
    print("Incident écrit dans", DRIFT_FILE, "- lancez : python src/drift.py --injected")


if __name__ == "__main__":
    main()

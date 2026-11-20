"""Agrège les courses en une ligne par zone et par heure, puis construit les features."""
import os

import pandas as pd

from config import PROCESSED_FILE, RAW_DIR, load_params
from features import build_features

COLUMNS = ["tpep_pickup_datetime", "PULocationID", "trip_distance", "fare_amount"]


def aggregate_month(month):
    path = os.path.join(RAW_DIR, "yellow_tripdata_" + month + ".parquet")
    trips = pd.read_parquet(path, columns=COLUMNS)

    # Le fichier contient quelques dates hors du mois : on les retire
    start = pd.Timestamp(month + "-01")
    end = start + pd.offsets.MonthBegin(1)
    trips = trips[(trips["tpep_pickup_datetime"] >= start) & (trips["tpep_pickup_datetime"] < end)]

    trips["pickup_hour"] = trips["tpep_pickup_datetime"].dt.floor("h")
    hourly = trips.groupby(["PULocationID", "pickup_hour"]).agg(
        pickups=("PULocationID", "size"),
        avg_distance=("trip_distance", "mean"),
        avg_fare=("fare_amount", "mean"),
    ).reset_index()
    hourly = hourly.rename(columns={"PULocationID": "pu_location_id"})
    print(month, ":", len(trips), "courses ->", len(hourly), "lignes zone x heure")
    return hourly


def complete_grid(hourly):
    """Ajoute les heures sans course (pickups = 0) pour que chaque zone ait toutes les heures."""
    zones = sorted(hourly["pu_location_id"].unique())
    hours = pd.date_range(hourly["pickup_hour"].min(), hourly["pickup_hour"].max(), freq="h")
    grid = pd.MultiIndex.from_product([zones, hours], names=["pu_location_id", "pickup_hour"])
    hourly = hourly.set_index(["pu_location_id", "pickup_hour"]).reindex(grid).reset_index()
    hourly["pickups"] = hourly["pickups"].fillna(0)
    return hourly


def main():
    params = load_params()
    parts = []
    for month in params["data"]["months"]:
        parts.append(aggregate_month(month))
    hourly = pd.concat(parts, ignore_index=True)
    hourly = complete_grid(hourly)
    df = build_features(hourly)

    os.makedirs(os.path.dirname(PROCESSED_FILE), exist_ok=True)
    df.to_parquet(PROCESSED_FILE, index=False)
    print("Écrit :", PROCESSED_FILE, "-", len(df), "lignes")


if __name__ == "__main__":
    main()

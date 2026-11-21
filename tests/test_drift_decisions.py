"""Tests de la règle centrale du monitoring : une alerte ne dit pas quoi faire.
- drift réel (le monde change)      -> raisons "input" : réentraîner
- données cassées (unité, flux coupé) -> raisons "data_quality" : NE PAS réentraîner
Plus la grille zone x heure : une heure sans course doit valoir 0, sinon les lags sont décalés."""
import pandas as pd

from conftest import make_hourly
from config import load_params
from drift import check_drift
from features import build_features
from prepare import complete_grid

PARAMS = load_params()


def split_two_weeks(features_df):
    """Référence = 1re semaine disponible, fenêtre courante = la suivante (même régime)."""
    middle = features_df["pickup_hour"].min() + pd.Timedelta(days=7)
    reference = features_df[features_df["pickup_hour"] < middle].reset_index(drop=True)
    current = features_df[features_df["pickup_hour"] >= middle].reset_index(drop=True)
    return reference, current


def kinds(report):
    return {reason["kind"] for reason in report["reasons"]}


def test_same_regime_gives_no_alert(features_df):
    reference, current = split_two_weeks(features_df)
    report = check_drift(current, reference, PARAMS)
    assert report["status"] != "ALERT"  # pas de fausse alerte sur une période normale


def test_demand_collapse_is_drift_not_data_quality(features_df):
    reference, current = split_two_weeks(features_df)
    current = current.copy()
    for column in ["pickups", "lag_1", "lag_2", "lag_24", "lag_168"]:
        current[column] = current[column] * 0.1  # demande divisée par 10, comme au confinement
    report = check_drift(current, reference, PARAMS)
    assert report["status"] == "ALERT"
    assert "input" in kinds(report)
    assert "data_quality" not in kinds(report)  # le monde a changé : réentraîner est permis


def test_fare_in_cents_is_data_quality(features_df):
    reference, current = split_two_weeks(features_df)
    current = current.copy()
    current["avg_fare"] = current["avg_fare"] * 100  # tarif en centimes au lieu de dollars
    report = check_drift(current, reference, PARAMS)
    assert "data_quality" in kinds(report)  # retrain.py refusera de réentraîner


def test_silent_zone_is_data_quality(features_df):
    reference, current = split_two_weeks(features_df)
    current = current.copy()
    current.loc[current["pu_location_id"] == 132, "pickups"] = 0  # la zone 132 ne remonte plus rien
    report = check_drift(current, reference, PARAMS)
    assert report["silent_zones"] == [132]
    assert "data_quality" in kinds(report)


def test_missing_hour_is_filled_with_zero():
    hourly = make_hourly(n_days=9, zones=(1,))
    gap = hourly["pickup_hour"].iloc[200]
    with_gap = hourly[hourly["pickup_hour"] != gap]  # une heure sans aucune course : absente du fichier TLC

    grid = complete_grid(with_gap)
    assert len(grid) == len(hourly)
    assert grid.loc[grid["pickup_hour"] == gap, "pickups"].item() == 0

    features = build_features(grid)
    next_hour = features[features["pickup_hour"] == gap + pd.Timedelta(hours=1)]
    assert next_hour["lag_1"].item() == 0  # sans la grille, lag_1 vaudrait les courses d'il y a 2 h

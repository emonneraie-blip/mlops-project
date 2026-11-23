"""Tests unitaires : features (pas de fuite du futur) et PSI."""
import numpy as np
import pandas as pd

from drift import psi
from features import FEATURES


def test_lags_use_only_the_past(features_df):
    zone = features_df[features_df["pu_location_id"] == 1].reset_index(drop=True)
    # lag_1 de l'heure t = pickups de l'heure t-1
    assert zone.loc[10, "lag_1"] == zone.loc[9, "pickups"]
    assert zone.loc[30, "lag_24"] == zone.loc[6, "pickups"]


def test_no_missing_feature(features_df):
    assert features_df[FEATURES].isna().sum().sum() == 0


def test_psi_same_distribution_is_small():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(0, 1, 5000), rng.normal(0, 1, 5000)) < 0.10


def test_psi_shifted_distribution_is_large():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(0, 1, 5000), rng.normal(1.5, 1, 5000)) > 0.25


def test_zone_missing_from_window_is_silent(features_df):
    """Le flux d'une zone peut s'arrêter de deux façons : des lignes à 0, ou plus de lignes du tout.
    Les deux sont une zone muette."""
    from config import load_params
    from drift import check_drift

    middle = features_df["pickup_hour"].min() + pd.Timedelta(days=7)
    reference = features_df[features_df["pickup_hour"] < middle].reset_index(drop=True)
    current = features_df[features_df["pickup_hour"] >= middle]
    current = current[current["pu_location_id"] != 132].reset_index(drop=True)  # la zone 132 a disparu

    report = check_drift(current, reference, load_params())
    assert report["silent_zones"] == [132]
    assert report["status"] == "ALERT"

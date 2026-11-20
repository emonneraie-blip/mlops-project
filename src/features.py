"""Construction des features. Utilisé à l'entraînement ET par l'API (même code des deux côtés)."""
import pandas as pd

LAGS = [1, 2, 24, 168]
FEATURES = ["pu_location_id", "hour", "day_of_week", "is_weekend", "lag_1", "lag_2", "lag_24", "lag_168"]
TARGET = "pickups"


def add_calendar_features(df):
    df = df.copy()
    df["hour"] = df["pickup_hour"].dt.hour
    df["day_of_week"] = df["pickup_hour"].dt.day_of_week
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def add_lag_features(df):
    # La grille zone x heure doit être complète (pas de trou), sinon shift() décale mal.
    df = df.sort_values(["pu_location_id", "pickup_hour"]).reset_index(drop=True)
    for lag in LAGS:
        df["lag_" + str(lag)] = df.groupby("pu_location_id")["pickups"].shift(lag)
    return df


def build_features(hourly):
    """hourly : une ligne par zone et par heure (pickups, avg_distance, avg_fare)."""
    df = add_lag_features(hourly)
    df = add_calendar_features(df)
    df = df.dropna(subset=["lag_168"]).reset_index(drop=True)
    return df


def request_to_frame(pu_location_id, pickup_hour, lag_1, lag_2, lag_24, lag_168):
    """Transforme une requête de l'API en une ligne de features."""
    df = pd.DataFrame({
        "pu_location_id": [pu_location_id],
        "pickup_hour": [pd.Timestamp(pickup_hour)],
        "lag_1": [float(lag_1)],
        "lag_2": [float(lag_2)],
        "lag_24": [float(lag_24)],
        "lag_168": [float(lag_168)],
    })
    df = add_calendar_features(df)
    return to_model_input(df)


def select_period(df, start, end):
    mask = (df["pickup_hour"] >= pd.Timestamp(start)) & (df["pickup_hour"] <= pd.Timestamp(end))
    return df[mask].reset_index(drop=True)


def to_model_input(df):
    """Le modèle reçoit toujours les mêmes colonnes, dans le même ordre, en float."""
    return df[FEATURES].astype(float)

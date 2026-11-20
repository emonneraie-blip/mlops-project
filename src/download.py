"""Télécharge les fichiers mensuels Yellow Taxi.
Source : NYC Taxi & Limousine Commission (TLC), Trip Record Data
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
Les fichiers bruts ne sont jamais commités : ce script + DVC les reproduisent.
"""
import os

import requests

from config import RAW_DIR, load_params

BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{}.parquet"


def download_month(month):
    path = os.path.join(RAW_DIR, "yellow_tripdata_" + month + ".parquet")
    if os.path.exists(path):
        print("Déjà présent :", path)
        return
    url = BASE_URL.format(month)
    print("Téléchargement :", url)
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    with open(path + ".tmp", "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
    os.rename(path + ".tmp", path)


def main():
    params = load_params()
    os.makedirs(RAW_DIR, exist_ok=True)
    for month in params["data"]["months"]:
        download_month(month)


if __name__ == "__main__":
    main()

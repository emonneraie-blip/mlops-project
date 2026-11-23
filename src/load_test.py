"""Test de charge de l'API : latence P50 / P95 / P99 et débit, comparés au budget de la séance 3
(P95 <= 200 ms). Pas de dépendance en plus : requests + threads.
Usage (API lancée, modèle en @production) :
    python src/load_test.py --users 1 4 16 --seconds 15
Pour chaque niveau de concurrence : requêtes envoyées pendant N secondes, sans pause.
On retient le débit le plus haut dont la P95 reste sous le budget, pas le débit maximal atteint.
Code de sortie 1 si aucun niveau ne tient le budget, ou si une requête échoue.
"""
import argparse
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

from config import API_URL

BUDGET_MS = 200
ZONES = [132, 138, 161, 162, 230, 236, 237, 48, 79, 186]  # zones parmi les plus actives


def random_request():
    lag_1 = random.randint(0, 300)
    return {"pu_location_id": random.choice(ZONES),
            "pickup_hour": "2020-02-" + str(random.randint(10, 28)) + "T" + str(random.randint(0, 23)).zfill(2) + ":00:00",
            "lag_1": lag_1, "lag_2": max(0, lag_1 + random.randint(-20, 20)),
            "lag_24": random.randint(0, 300), "lag_168": random.randint(0, 300)}


def user_loop(url, deadline):
    """Un utilisateur : envoie des requêtes sans pause jusqu'à la fin. Renvoie (latences en ms, nb d'erreurs)."""
    latencies, errors = [], 0
    with requests.Session() as session:
        while time.monotonic() < deadline:
            started = time.perf_counter()
            try:
                response = session.post(url, json=random_request(), timeout=5)
                ok = response.status_code == 200
            except requests.RequestException:
                ok = False
            latencies.append((time.perf_counter() - started) * 1000)
            if not ok:
                errors += 1
    return latencies, errors


def summarize(latencies, errors, seconds):
    values = np.array(latencies)
    return {"requests": len(values), "errors": errors, "rps": len(values) / seconds,
            "p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99))}


def run_level(url, users, seconds):
    deadline = time.monotonic() + seconds
    with ThreadPoolExecutor(max_workers=users) as pool:
        results = list(pool.map(lambda _: user_loop(url, deadline), range(users)))
    latencies = [value for user_latencies, _ in results for value in user_latencies]
    return summarize(latencies, sum(errors for _, errors in results), seconds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, nargs="+", default=[1, 4, 16], help="niveaux de concurrence")
    parser.add_argument("--seconds", type=float, default=15)
    parser.add_argument("--url", default=API_URL + "/v1/predict")
    args = parser.parse_args()

    requests.post(args.url, json=random_request(), timeout=30)  # échauffement : la première requête est plus lente
    print("utilisateurs  requêtes  erreurs  req/s    P50 ms  P95 ms  P99 ms  budget P95 " + str(BUDGET_MS) + " ms")
    best = None
    failed = False
    for users in args.users:
        r = run_level(args.url, users, args.seconds)
        within = r["p95"] <= BUDGET_MS and r["errors"] == 0
        failed = failed or r["errors"] > 0
        if within and (best is None or r["rps"] > best["rps"]):
            best = dict(r, users=users)
        print(str(users).ljust(13), str(r["requests"]).ljust(9), str(r["errors"]).ljust(8),
              str(round(r["rps"], 1)).ljust(8), str(round(r["p50"], 1)).ljust(7), str(round(r["p95"], 1)).ljust(7),
              str(round(r["p99"], 1)).ljust(7), "OK" if within else "HORS BUDGET")
    print("")
    if best is None:
        print("Aucun niveau ne tient P95 <= " + str(BUDGET_MS) + " ms sans erreur")
        sys.exit(1)
    print("Capacité retenue :", round(best["rps"], 1), "req/s avec", best["users"],
          "utilisateurs (P95", round(best["p95"], 1), "ms)")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

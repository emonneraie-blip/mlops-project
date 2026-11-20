"""Rollback : remet @production sur la version précédente. Une commande, pas de réentraînement.
Le modèle fautif n'est pas supprimé : il garde son numéro de version et devient @previous.
"""
import argparse
import time

from config import MODEL_NAME
from registry import get_alias_version, get_client, log_event, notify_api


def rollback(target=None, reason="manual"):
    started = time.time()
    client = get_client()
    production = get_alias_version(client, "production")
    if target is None:
        target = get_alias_version(client, "previous")
    if target is None:
        print("Pas de version @previous : rien vers quoi revenir")
        return False
    target = str(target)
    if target == production:
        print("La version", target, "est déjà en production")
        return False

    client.set_registered_model_alias(MODEL_NAME, "production", target)
    if production is not None:
        client.set_registered_model_alias(MODEL_NAME, "previous", production)
    print("ROLLBACK : @production v" + str(production), "-> v" + target)
    log_event("rollback", {"from_version": production, "to_version": target, "reason": reason})
    notify_api()
    print("Durée :", round(time.time() - started, 1), "s")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", help="version cible ; par défaut celle qui porte @previous")
    parser.add_argument("--reason", default="manual")
    args = parser.parse_args()
    rollback(args.version, args.reason)


if __name__ == "__main__":
    main()

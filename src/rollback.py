"""Rollback : remet @production sur la version précédente. Une commande, pas de réentraînement.
Le modèle fautif n'est pas supprimé : il garde son numéro de version, devient @previous,
et reçoit le tag rolled_back (visible dans l'UI MLflow, avec la raison).
Un deuxième rollback ne remet donc pas automatiquement en production la version qu'on vient de retirer :
pour la remettre, il faut la nommer (--version N) ou la repromouvoir avec promote.py (gate).
"""
import argparse
import time

from config import MODEL_NAME
from registry import get_alias_version, get_client, log_event, notify_api

ROLLED_BACK_TAG = "rolled_back"


def rolled_back_reason(client, version):
    """La raison du rollback si cette version a été retirée de la production, sinon None."""
    return client.get_model_version(MODEL_NAME, version).tags.get(ROLLED_BACK_TAG)


def rollback(target=None, reason="manual"):
    started = time.time()
    client = get_client()
    production = get_alias_version(client, "production")
    explicit = target is not None
    if target is None:
        target = get_alias_version(client, "previous")
    if target is None:
        print("Pas de version @previous : rien vers quoi revenir")
        return False
    target = str(target)
    if target == production:
        print("La version", target, "est déjà en production")
        return False

    # @previous peut être la version qu'un rollback précédent vient de retirer : on n'y revient pas en aveugle
    previous_reason = rolled_back_reason(client, target)
    if previous_reason is not None and not explicit:
        print("REFUSÉ : @previous (v" + target + ") a déjà été retirée par un rollback (raison : " + previous_reason + ")")
        print("Pour la remettre quand même : python src/rollback.py --version " + target)
        log_event("rollback_refused", {"production": production, "previous": target})
        return False

    client.set_registered_model_alias(MODEL_NAME, "production", target)
    if previous_reason is not None:
        client.delete_model_version_tag(MODEL_NAME, target, ROLLED_BACK_TAG)  # remise en production assumée
    if production is not None:
        client.set_registered_model_alias(MODEL_NAME, "previous", production)
        client.set_model_version_tag(MODEL_NAME, production, ROLLED_BACK_TAG, reason)
    print("ROLLBACK : @production v" + str(production), "-> v" + target)
    api_reloaded = notify_api()
    log_event("rollback", {"from_version": production, "to_version": target, "reason": reason,
                           "api_reloaded": api_reloaded})
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

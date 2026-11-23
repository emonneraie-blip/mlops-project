#!/usr/bin/env bash
# Démo en 9 étapes, jouée automatiquement avec des pauses, pour la vidéo de secours (légendes en anglais, comme les slides).
# Prérequis : make mlflow (onglet 1) et make serve (onglet 2) tournent déjà.
# Usage : bash scripts/demo.sh            (PAUSE=5 pour des pauses plus longues)
#         RECORD=1 bash scripts/demo.sh   enregistre aussi l'écran tout seul (macOS) dans ~/Desktop/demo_backup.mov
set -u
cd "$(dirname "$0")/.."
PAUSE="${PAUSE:-3}"
RECORD="${RECORD:-0}"
VIDEO="$HOME/Desktop/demo_backup.mov"
MAX_SECONDS="${MAX_SECONDS:-240}"   # durée max de l'enregistrement auto ; la fin est coupée au montage
UI="http://127.0.0.1:5050/#/models/taxi-demand"
case "${TERM_PROGRAM:-}" in
  Apple_Terminal) TERM_APP="Terminal" ;;
  iTerm.app) TERM_APP="iTerm" ;;
  *) TERM_APP="" ;;
esac

type_out() {  # affiche une commande lettre par lettre, comme tapée à la main
  printf '\033[1;32m$ \033[0m'
  local s="$1" i
  for ((i = 0; i < ${#s}; i++)); do printf '%s' "${s:i:1}"; sleep 0.04; done
  printf '\n'
}
caption() { printf '\n\033[1;33m# %s\033[0m\n' "$1"; sleep 1.5; }
run() { type_out "$1"; eval "$1"; sleep "$PAUSE"; }
version() { curl -s localhost:8000/v1/health | uv run python -c 'import sys, json; print(json.load(sys.stdin)["model_version"])'; }
show_mlflow() {  # ouvre MLflow, laisse le temps de lire, revient au terminal
  caption "$1"
  open "$2"
  sleep 12
  [ -n "$TERM_APP" ] && osascript -e "tell application \"$TERM_APP\" to activate"
  sleep 1
}

# Préparation, avant l'enregistrement
curl -sf localhost:8000/v1/health > /dev/null || { echo "API injoignable : lancez make mlflow puis make serve"; exit 1; }
uv run python src/rollback.py --version 1 --reason "demo reset" > /dev/null
rm -f monitoring/state.json
sleep 2
[ "$(version)" = "1" ] || { echo "La production n'est pas sur v1 : vérifiez l'onglet de make serve"; exit 1; }
uv run python -c "print('ok')" > /dev/null   # réchauffe uv pour que make test soit rapide
if [ "$RECORD" = "1" ]; then
  echo "Prêt. Appuyez sur Entrée : l'écran entier sera enregistré dans $VIDEO (au plus ${MAX_SECONDS} s)."
  read -r
  rm -f "$VIDEO"
  screencapture -v -x -V "$MAX_SECONDS" "$VIDEO" &
  REC_PID=$!
  sleep 2
else
  echo "Prêt. Lancez l'enregistrement (Cmd+Shift+5), puis appuyez sur Entrée."
  read -r
fi
clear

caption "1. The tests, live (no MLflow, no network needed)"
run "make test"
caption "2. Every answer says which model version produced it"
run "make request"
caption "3. February: same regime, no alert"
run "make drift-ok"
caption "4. March-April 2020: Covid"
run "make drift"
caption "5. Real drift: retrain, gate, then a person approves (o)"
type_out "make retrain"
echo o | make retrain
echo
sleep "$PAUSE"
NEW="$(version)"
show_mlflow "MLflow: v$NEW carries @production, v1 carries @previous" "$UI"
caption "6. The new version learned the lockdown"
run "make request"
caption "7. Rollback: one command, a few seconds"
run "make rollback"
caption "8. A second rollback by mistake is refused"
run "make rollback"
show_mlflow "MLflow: @production is back on v1, v$NEW is tagged rolled_back" "$UI/versions/$NEW"
caption "9. Everything is logged"
run "make events | tail"
caption "End of demo"
if [ "$RECORD" = "1" ]; then
  echo
  echo "Démo terminée. L'enregistrement s'arrête tout seul au bout de ${MAX_SECONDS} s : ne touchez à rien, puis la vidéo sera dans $VIDEO"
  wait "$REC_PID"
  echo "Vidéo enregistrée : $VIDEO"
fi

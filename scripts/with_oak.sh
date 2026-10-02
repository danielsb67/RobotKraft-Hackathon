#!/usr/bin/env bash
# Lance le pont OAK-D -> ZMQ (camera1) en arrière-plan, attend sa première image,
# exécute la commande passée en arguments (lerobot-record, robot_client...), puis arrête le pont.
# Usage : bash scripts/with_oak.sh lerobot-record --robot.type=... ...
set -euo pipefail
cd "$(dirname "$0")/.."

export OAK_READY_FILE=${OAK_READY_FILE:-/tmp/oak_zmq_ready}
rm -f "$OAK_READY_FILE"
python3 scripts/oak_zmq_server.py &
OAK_PID=$!
trap 'kill "$OAK_PID" 2>/dev/null || true; wait "$OAK_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 60); do
    [ -e "$OAK_READY_FILE" ] && break
    kill -0 "$OAK_PID" 2>/dev/null || { echo "ERREUR : le pont OAK-D s'est arrêté (caméra branchée ? déjà utilisée par un autre programme ?)" >&2; exit 1; }
    sleep 0.5
done
[ -e "$OAK_READY_FILE" ] || { echo "ERREUR : aucune image de l'OAK-D après 30 s" >&2; exit 1; }

"$@"

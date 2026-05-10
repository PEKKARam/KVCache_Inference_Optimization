#!/usr/bin/env bash
set -euo pipefail

HOST=${HOST:-"http://127.0.0.1:8000"}
USERS=${USERS:-20}
SPAWN_RATE=${SPAWN_RATE:-2}
RUN_TIME=${RUN_TIME:-"2m"}

locust -f locustfile.py --headless -u "$USERS" -r "$SPAWN_RATE" --run-time "$RUN_TIME" --host "$HOST"

#!/usr/bin/env bash
set -euo pipefail

URL=${URL:-"http://127.0.0.1:8000"}
DURATION=${DURATION:-"30s"}
CONNECTIONS=${CONNECTIONS:-64}
THREADS=${THREADS:-4}

wrk -t"$THREADS" -c"$CONNECTIONS" -d"$DURATION" -s post.lua "$URL"

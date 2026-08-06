#!/bin/sh
set -eu

ROOT=${ROOT:-$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)}
RUN_ROOT=${1:-$ROOT/runs/production_001}
PYTHON=${PYTHON:-python3}
cd "$ROOT"
"$PYTHON" -u scripts/monitor.py "$RUN_ROOT"

#!/bin/sh
set -eu

ROOT=${ROOT:-$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)}
RUN_ROOT=${RUN_ROOT:-$ROOT/runs/production_001}
CONFIG=${CONFIG:-$ROOT/config/search.json}
PYTHON=${PYTHON:-python3}
NTASKS=${NTASKS:-32}
WORKERS=${WORKERS:-6}
INCAR=${INCAR:?Set INCAR to the immutable master INCAR path}
KPOINTS=${KPOINTS:?Set KPOINTS to the immutable master KPOINTS path}
POTCAR=${POTCAR:?Set POTCAR to the immutable master POTCAR path}
TEMPLATE=${TEMPLATE:-$ROOT/inputs/POSCAR_CeO2_111_3x3_1TL}
VASP_EXEC=${VASP_EXEC:?Set VASP_EXEC to the VASP executable}

cd "$ROOT"
mkdir -p logs "$RUN_ROOT"

case "$WORKERS" in
  ''|*[!0-9]*) echo "WORKERS must be a positive integer"; exit 2 ;;
esac
if [ "$WORKERS" -lt 1 ]; then
  echo "WORKERS must be at least one"
  exit 2
fi

if [ -f "$RUN_ROOT/state/controller_state.json" ] && [ "${ALLOW_RESUME:-0}" != 1 ]; then
  echo "Run state already exists: $RUN_ROOT"
  echo "Use ALLOW_RESUME=1 only after checking squeue and the run state."
  exit 2
fi

"$PYTHON" -u scripts/preflight.py \
  --template "$TEMPLATE" --incar "$INCAR" --kpoints "$KPOINTS" --potcar "$POTCAR"

"$PYTHON" -u scripts/capture_provenance.py \
  --run-root "$RUN_ROOT" --config "$CONFIG" --template "$TEMPLATE" \
  --incar "$INCAR" --kpoints "$KPOINTS" --potcar "$POTCAR" \
  --vasp-exec "$VASP_EXEC"

JOB_ID=$(sbatch --parsable \
  --array="0-$((WORKERS - 1))%$WORKERS" \
  --ntasks="$NTASKS" \
  --export=ALL,ROOT="$ROOT",RUN_ROOT="$RUN_ROOT",CONFIG="$CONFIG",PYTHON="$PYTHON",INCAR="$INCAR",KPOINTS="$KPOINTS",POTCAR="$POTCAR",TEMPLATE="$TEMPLATE",VASP_EXEC="$VASP_EXEC" \
  slurm/worker_array.sbatch)
echo "Submitted asynchronous search array: $JOB_ID"
echo "Run root: $RUN_ROOT"
echo "Monitor: $PYTHON -u scripts/monitor.py $RUN_ROOT"

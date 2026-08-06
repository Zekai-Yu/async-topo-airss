#!/bin/sh
set -eu

ROOT=${ROOT:-$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)}
RUN_ROOT=${RUN_ROOT:-$ROOT/runs/smoke_001}
CONFIG=${CONFIG:-$ROOT/config/smoke.json}
PYTHON=${PYTHON:-python3}
NTASKS=${NTASKS:-32}
INCAR=${INCAR:?Set INCAR to the immutable master INCAR path}
KPOINTS=${KPOINTS:?Set KPOINTS to the immutable master KPOINTS path}
POTCAR=${POTCAR:?Set POTCAR to the immutable master POTCAR path}
VASP_EXEC=${VASP_EXEC:?Set VASP_EXEC to the VASP executable}
TEMPLATE=${TEMPLATE:-$ROOT/inputs/POSCAR_CeO2_111_3x3_1TL}

cd "$ROOT"
mkdir -p logs "$RUN_ROOT"
if [ -f "$RUN_ROOT/state/controller_state.json" ]; then
  echo "Refusing to overwrite existing smoke run: $RUN_ROOT"
  exit 2
fi
"$PYTHON" -u scripts/preflight.py \
  --template "$TEMPLATE" --incar "$INCAR" --kpoints "$KPOINTS" --potcar "$POTCAR"
"$PYTHON" -u scripts/capture_provenance.py \
  --run-root "$RUN_ROOT" --config "$CONFIG" --template "$TEMPLATE" \
  --incar "$INCAR" --kpoints "$KPOINTS" --potcar "$POTCAR" \
  --vasp-exec "$VASP_EXEC"
JOB_ID=$(sbatch --parsable --array=0-1%2 \
  --ntasks="$NTASKS" \
  --export=ALL,ROOT="$ROOT",RUN_ROOT="$RUN_ROOT",CONFIG="$CONFIG",PYTHON="$PYTHON",INCAR="$INCAR",KPOINTS="$KPOINTS",POTCAR="$POTCAR",TEMPLATE="$TEMPLATE",VASP_EXEC="$VASP_EXEC" \
  slurm/worker_array.sbatch)
echo "Submitted two-worker smoke test: $JOB_ID"
echo "Monitor: $PYTHON -u scripts/monitor.py $RUN_ROOT"

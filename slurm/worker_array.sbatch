#!/bin/sh
#SBATCH -J pd5ox_async
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --array=0-5%6
#SBATCH -o logs/%A_%a.out
#SBATCH -e logs/%A_%a.err

set -eu

job_log() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*"
}

CONTROLLER_PID=

cleanup() {
  code=$?
  trap - EXIT
  if [ "$code" -ne 0 ]; then
    touch "$RUN_ROOT/WORKER_FAILED" "$RUN_ROOT/STOP_WORKERS"
    if [ -n "$CONTROLLER_PID" ]; then
      kill "$CONTROLLER_PID" 2>/dev/null || true
    fi
  fi
  exit "$code"
}

ROOT=${ROOT:?ROOT was not exported by the submit script}
RUN_ROOT=${RUN_ROOT:-$ROOT/runs/production_001}
CONFIG=${CONFIG:-$ROOT/config/search.json}
PYTHON=${PYTHON:-python3}
VASP_EXEC=${VASP_EXEC:?VASP_EXEC was not exported by the submit script}
INCAR=${INCAR:?INCAR was not exported by the submit script}
KPOINTS=${KPOINTS:?KPOINTS was not exported by the submit script}
POTCAR=${POTCAR:?POTCAR was not exported by the submit script}
TEMPLATE=${TEMPLATE:-$ROOT/inputs/POSCAR_CeO2_111_3x3_1TL}
NPROC=${SLURM_NTASKS:-32}

mkdir -p "$ROOT/logs" "$RUN_ROOT"
trap cleanup EXIT
trap 'exit 130' HUP INT TERM

if [ -n "${SITE_ENV:-}" ]; then
  . "$SITE_ENV"
fi
ulimit -s unlimited
ulimit -l unlimited
ulimit -c 0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_WAIT_POLICY=PASSIVE
export MALLOC_ARENA_MAX=2
export PYTHONUNBUFFERED=1

cd "$ROOT"
job_log "[SLURM] START job=${SLURM_ARRAY_JOB_ID:-unknown}_${SLURM_ARRAY_TASK_ID:-unknown} host=$(hostname) ranks=$NPROC"

if [ "${SLURM_ARRAY_TASK_ID:-0}" -eq 0 ]; then
  if ! "$PYTHON" -u scripts/preflight.py \
    --template "$TEMPLATE" --incar "$INCAR" --kpoints "$KPOINTS" --potcar "$POTCAR"; then
    touch "$RUN_ROOT/STOP_WORKERS"
    exit 2
  fi
  (nice -n 10 "$PYTHON" -u scripts/controller.py \
    --run-root "$RUN_ROOT" --config "$CONFIG" --template "$TEMPLATE" \
    || { touch "$RUN_ROOT/CONTROLLER_FAILED" "$RUN_ROOT/STOP_WORKERS"; exit 3; }) &
  CONTROLLER_PID=$!
  job_log "[SLURM] controller_pid=$CONTROLLER_PID"
fi

"$PYTHON" -u scripts/worker.py \
  --run-root "$RUN_ROOT" \
  --config "$CONFIG" \
  --worker-id "${SLURM_ARRAY_TASK_ID:-0}" \
  --incar "$INCAR" \
  --kpoints "$KPOINTS" \
  --potcar "$POTCAR" \
  --vasp-exec "$VASP_EXEC" \
  --nproc "$NPROC" \
  --mpi-launcher mpirun

if [ -n "$CONTROLLER_PID" ]; then
  wait "$CONTROLLER_PID"
fi
job_log "[SLURM] END job=${SLURM_ARRAY_JOB_ID:-unknown}_${SLURM_ARRAY_TASK_ID:-unknown}"

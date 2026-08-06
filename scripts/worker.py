#!/usr/bin/env python3
"""Long-lived VASP worker for the atomic filesystem queue."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
import traceback

import numpy as np
from ase.constraints import FixAtoms
from ase.io import read, write

from chemistry import blocks, fingerprint, relaxed_classification, topology_signature


def timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(worker_id, message):
    print(f"{timestamp()} [WORKER {worker_id}] {message}", flush=True)


def atomic_json(path: Path, payload, durable=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        if durable:
            handle.flush()
            os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_active(text, key, value):
    pattern = rf"(?im)^(\s*{re.escape(key)}\s*=\s*)([^#;!\r\n]+)(.*)$"
    matches = list(re.finditer(pattern, text))
    if len(matches) != 1:
        raise ValueError(f"Master INCAR must contain exactly one active {key} line")
    return re.sub(pattern, rf"\g<1>{value}\g<3>", text, count=1)


def oszicar_progress(path):
    """Read OSZICAR once for heartbeat and final convergence metadata."""
    if not path.is_file():
        return {"ionic": 0, "last_electronic": None,
                "status": "waiting for OSZICAR"}
    ionic = 0
    current = 0
    last_electronic = None
    last_ionic_line = None
    last_electronic_line = None
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if re.match(r"^\s*(DAV|RMM|CG|DMP|EDDIAG):\s*\d+", line):
                current += 1
                last_electronic_line = line.strip()
            elif re.match(r"^\s*\d+\s+F=", line):
                ionic += 1
                last_electronic = current
                current = 0
                last_ionic_line = line.strip()
    if current > 0 and last_electronic_line is not None:
        status = (f"ionic={ionic} current_electronic={current} "
                  f"{last_electronic_line}")
    elif last_ionic_line is not None:
        status = f"ionic={ionic} {last_ionic_line}"
    elif last_electronic_line is not None:
        status = f"electronic={current} {last_electronic_line}"
    else:
        status = "OSZICAR has no ionic/electronic line yet"
    return {"ionic": ionic, "last_electronic": last_electronic,
            "status": status}


def mobile_fmax(atoms, forces):
    index = blocks(atoms)
    mobile = np.concatenate((index["top_o"], index["cluster_o"], index["pd"]))
    return float(np.linalg.norm(np.asarray(forces)[mobile], axis=1).max())


def fixed_indices(atoms):
    values = []
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            values.extend(map(int, constraint.get_indices()))
    return sorted(set(values))


def prepare_workdir(task, args, config, workdir):
    workdir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(task["input_poscar"], workdir / "POSCAR")
    shutil.copy2(args.kpoints, workdir / "KPOINTS")
    shutil.copy2(args.potcar, workdir / "POTCAR")
    text = Path(args.incar).read_text(encoding="utf-8")
    text = replace_active(text, "NELM", str(int(config["nelm"])))
    nsw = {
        "SHORT": int(config["short_nsw"]),
        "MEDIUM": int(config["medium_nsw"]),
        "FULL": int(config["full_nsw"]),
    }[task["stage"]]
    text = replace_active(text, "NSW", str(nsw))
    (workdir / "INCAR").write_text(text, encoding="utf-8")
    atoms = read(workdir / "POSCAR", format="vasp")
    blocks(atoms, int(task["x"]))
    return nsw


def run_task(task, args, config, worker_id):
    run_root = Path(args.run_root).resolve()
    workdir = run_root / "calculations" / task["candidate_id"] / \
        f"{task['stage'].lower()}_a{task['attempt']}"
    result_path = workdir / "result.json"
    if result_path.is_file():
        return result_path
    if workdir.exists():
        raise FileExistsError(f"Refusing to overwrite incomplete workdir {workdir}")
    master_before = {name: sha256(path) for name, path in
                     (("INCAR", args.incar), ("KPOINTS", args.kpoints),
                      ("POTCAR", args.potcar))}
    requested_nsw = prepare_workdir(task, args, config, workdir)
    command = [args.mpi_launcher, "-np", str(args.nproc),
               str(Path(args.vasp_exec).resolve())]
    start = time.monotonic()
    log(worker_id, f"START task={task['task_id']} x={task['x']} "
        f"stage={task['stage']} NSW={requested_nsw} ranks={args.nproc}")
    stop = threading.Event()

    def heartbeat(process):
        while not stop.wait(int(config["heartbeat_seconds"])):
            elapsed = time.monotonic() - start
            oszicar = workdir / "OSZICAR"
            progress = oszicar_progress(oszicar)
            completed_ionic = progress["ionic"]
            eta = (elapsed / completed_ionic * max(0, requested_nsw - completed_ionic)
                   if completed_ionic else None)
            eta_text = "estimating" if eta is None else f"{eta:.0f}s"
            log(worker_id, f"RUNNING task={task['task_id']} elapsed={elapsed:.0f}s "
                f"ETA_stage={eta_text} {progress['status']}")
            atomic_json(run_root / "workers" / f"worker_{worker_id}.json", {
                "worker": worker_id, "task": task["task_id"], "pid": process.pid,
                "updated": timestamp(), "elapsed_s": elapsed,
            }, durable=False)

    with open(workdir / "vasp.stdout", "w", encoding="utf-8") as stdout, \
            open(workdir / "vasp.stderr", "w", encoding="utf-8") as stderr:
        process = subprocess.Popen(command, cwd=workdir, stdout=stdout, stderr=stderr)
        reporter = threading.Thread(target=heartbeat, args=(process,), daemon=True)
        reporter.start()
        returncode = process.wait()
        stop.set()
        reporter.join(timeout=2)
    elapsed = time.monotonic() - start
    master_after = {name: sha256(path) for name, path in
                    (("INCAR", args.incar), ("KPOINTS", args.kpoints),
                     ("POTCAR", args.potcar))}
    payload = {
        "task_id": task["task_id"], "candidate_id": task["candidate_id"],
        "x": task["x"], "stage": task["stage"], "attempt": task["attempt"],
        "elapsed_s": elapsed, "returncode": returncode, "nproc": args.nproc,
        "requested_nsw": requested_nsw, "master_sha256_before": master_before,
        "master_sha256_after": master_after, "status": "FAILED",
    }
    try:
        if master_before != master_after:
            raise RuntimeError("Master VASP inputs changed during a worker task")
        if returncode != 0:
            raise RuntimeError(f"VASP returned exit code {returncode}")
        for name in ("OUTCAR", "OSZICAR", "CONTCAR"):
            if not (workdir / name).is_file():
                raise RuntimeError(f"Missing {name}")
        frame = read(workdir / "OUTCAR", index=-1, format="vasp-out")
        final_atoms = read(workdir / "CONTCAR", format="vasp")
        index = blocks(final_atoms, int(task["x"]))
        expected_fixed = sorted(list(index["bottom_o"]) + list(index["ce"]))
        if fixed_indices(final_atoms) != expected_fixed:
            raise RuntimeError("CONTCAR selective-dynamics flags changed")
        if frame.get_chemical_symbols() != final_atoms.get_chemical_symbols():
            raise RuntimeError("OUTCAR/CONTCAR atom order mismatch")
        if not np.allclose(frame.cell.array, final_atoms.cell.array, atol=1.0e-6):
            raise RuntimeError("OUTCAR/CONTCAR cell mismatch")
        position_delta = (frame.get_scaled_positions(wrap=False)
                          - final_atoms.get_scaled_positions(wrap=False))
        position_delta[:, :2] -= np.rint(position_delta[:, :2])
        position_error = float(np.linalg.norm(
            position_delta @ final_atoms.cell.array, axis=1).max()
        )
        energy = float(frame.get_potential_energy())
        free_energy = float(frame.get_potential_energy(force_consistent=True))
        forces = np.asarray(frame.get_forces(apply_constraint=False), dtype=float)
        if (forces.shape != (len(final_atoms), 3) or not np.isfinite(energy)
                or not np.isfinite(free_energy)
                or not np.all(np.isfinite(forces))):
            raise RuntimeError("Non-finite or mis-shaped VASP energy/forces")
        progress = oszicar_progress(workdir / "OSZICAR")
        ionic_frames = progress["ionic"]
        if ionic_frames < 1:
            raise RuntimeError("OSZICAR contains no completed ionic frame")
        electronic = progress["last_electronic"]
        if electronic is None or electronic >= int(config["nelm"]):
            raise RuntimeError(f"Last ionic frame electronic steps={electronic}")
        evaluated_atoms = frame.copy()
        evaluated_atoms.set_constraint(final_atoms.constraints)
        blocks(evaluated_atoms, int(task["x"]))
        evaluated_poscar = workdir / "POSCAR.evaluated"
        write(evaluated_poscar, evaluated_atoms, format="vasp", direct=True,
              vasp5=True, sort=False)
        fmax = mobile_fmax(evaluated_atoms, forces)
        classification = relaxed_classification(evaluated_atoms, int(task["x"]))
        payload.update({
            "status": "OK", "energy_eV": energy,
            "free_energy_eV": free_energy,
            "fmax_mobile_eVA": fmax,
            "force_converged": bool(fmax <= float(config["full_fmax_eVA"])),
            "electronic_steps_last_ionic": electronic,
            "ionic_frames": ionic_frames,
            "contcar": str((workdir / "CONTCAR").resolve()),
            "evaluated_poscar": str(evaluated_poscar.resolve()),
            "outcar_contcar_position_rms_max_A": position_error,
            "topology": topology_signature(evaluated_atoms, int(task["x"])),
            "fingerprint": fingerprint(evaluated_atoms, int(task["x"])).round(8).tolist(),
            "classification": classification,
        })
    except Exception as exc:
        payload["error"] = str(exc)
        payload["traceback"] = traceback.format_exc()
    atomic_json(result_path, payload)
    atomic_json(run_root / "workers" / f"worker_{worker_id}.json", {
        "worker": worker_id, "task": task["task_id"], "state": "DONE",
        "updated": timestamp(), "elapsed_s": elapsed,
        "status": payload["status"],
    }, durable=False)
    log(worker_id, f"DONE task={task['task_id']} status={payload['status']} "
        f"elapsed={elapsed:.1f}s E={payload.get('energy_eV')} "
        f"fmax={payload.get('fmax_mobile_eVA')}")
    return result_path


def claim_task(run_root, worker_id):
    pending = run_root / "queue" / "pending"
    running = run_root / "queue" / "running"
    for source in sorted(pending.glob("*.json")):
        target = running / source.name
        try:
            os.replace(source, target)
            return target
        except FileNotFoundError:
            continue
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--incar", required=True)
    parser.add_argument("--kpoints", required=True)
    parser.add_argument("--potcar", required=True)
    parser.add_argument("--vasp-exec", required=True)
    parser.add_argument("--nproc", type=int, default=32)
    parser.add_argument("--mpi-launcher", default="mpirun")
    args = parser.parse_args()
    worker_id = str(args.worker_id)
    run_root = Path(args.run_root).resolve()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    (run_root / "workers").mkdir(parents=True, exist_ok=True)
    log(worker_id, f"READY host={os.uname().nodename} ranks={args.nproc}")
    idle_since = time.monotonic()
    while True:
        task_file = claim_task(run_root, worker_id)
        if task_file is None:
            if (run_root / "SEARCH_COMPLETE").exists() or (run_root / "STOP_WORKERS").exists():
                log(worker_id, "EXIT marker detected")
                return
            if time.monotonic() - idle_since >= int(config["heartbeat_seconds"]):
                log(worker_id, "IDLE waiting for a queued task")
                idle_since = time.monotonic()
            time.sleep(int(config["idle_seconds"]))
            continue
        idle_since = time.monotonic()
        task = json.loads(task_file.read_text(encoding="utf-8"))
        task["claimed_by"] = worker_id
        task["claimed"] = timestamp()
        atomic_json(task_file, task)
        try:
            result_path = run_task(task, args, config, worker_id)
        except Exception as exc:
            failure_dir = run_root / "failures" / task["task_id"]
            failure_dir.mkdir(parents=True, exist_ok=True)
            result_path = failure_dir / "result.json"
            atomic_json(result_path, {
                "task_id": task["task_id"], "candidate_id": task["candidate_id"],
                "x": task["x"], "stage": task["stage"], "attempt": task["attempt"],
                "status": "FAILED", "error": str(exc),
                "traceback": traceback.format_exc(),
            })
            log(worker_id, f"FAILED task={task['task_id']} error={exc}")
        task["result_path"] = str(Path(result_path).resolve())
        task["completed"] = timestamp()
        atomic_json(task_file, task)
        os.replace(task_file, run_root / "queue" / "completed" / task_file.name)


if __name__ == "__main__":
    main()

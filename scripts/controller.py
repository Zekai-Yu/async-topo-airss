#!/usr/bin/env python3
"""Single-writer asynchronous controller for topology-GA plus AIRSS."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import time

import numpy as np
from ase.io import read, write

from chemistry import (
    AIRSSBuilder,
    crossover_candidate,
    fingerprint,
    fingerprint_rms,
    ladder_candidate,
    mutate_candidate,
    structure_key,
    topology_signature,
    validate_initial,
)


def timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(message):
    print(f"{timestamp()} [CONTROLLER] {message}", flush=True)


def atomic_json(path: Path, payload, compact=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        if compact:
            json.dump(payload, handle, separators=(",", ":"), sort_keys=True)
        else:
            json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def initial_state(config):
    return {
        "schema_version": 1,
        "created": timestamp(),
        "next_candidate": 1,
        "next_task": 1,
        "processed_tasks": [],
        "candidates": {},
        "rounds": {str(x): [] for x in config["compositions"]},
        "populations": {str(x): [] for x in config["compositions"]},
        "archives": {str(x): [] for x in config["compositions"]},
    }


class Controller:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.run_root).resolve()
        self.config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        self.template = read(args.template, format="vasp")
        self.rng = np.random.default_rng(int(self.config["seed"]))
        self.fp_cache = {}
        self.initial_index = {}
        self.full_history = {str(x): [] for x in self.config["compositions"]}
        self.metrics = {"generation_count": 0, "generation_s": 0.0,
                        "save_count": 0, "save_s": 0.0}
        self.state_path = self.root / "state" / "controller_state.json"
        self.queue = self.root / "queue"
        for name in ("pending", "running", "completed", "ingested", "failed"):
            (self.queue / name).mkdir(parents=True, exist_ok=True)
        (self.root / "candidates").mkdir(parents=True, exist_ok=True)
        if self.state_path.is_file():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.rng = np.random.default_rng()
            if "rng_state" in self.state:
                self.rng.bit_generator.state = self.state["rng_state"]
            self.reconcile_counters()
            log(f"RESUME candidates={len(self.state['candidates'])}")
        else:
            self.state = initial_state(self.config)
            self.save()
            log("NEW SEARCH state initialized")
        self.rebuild_indexes()
        self.archive_processed_task_files()

    def rebuild_indexes(self):
        """Build compact in-memory indexes once when starting or resuming."""
        self.initial_index = {}
        # A resumed state can contain completed compositions excluded from the
        # extension config. Keep their history while scheduling only configured x.
        self.full_history = {str(x): [] for x in self.state["rounds"]}
        for candidate_id, record in self.state["candidates"].items():
            key = (int(record["x"]), record["initial_topology"])
            self.initial_index.setdefault(key, []).append(
                record["initial_fingerprint_file"])
            if record.get("full_converged"):
                self.full_history[str(record["x"])].append(candidate_id)
        for identifiers in self.full_history.values():
            identifiers.sort(key=lambda item: self.state["candidates"][item].get(
                "final_completed", self.state["candidates"][item]["created"]))

    def reconcile_counters(self):
        """Prevent ID reuse if a crash occurred between file and state commits."""
        candidate_numbers = []
        for directory in (self.root / "candidates").glob("x*_c*"):
            try:
                candidate_numbers.append(int(directory.name.rsplit("c", 1)[1]))
            except (IndexError, ValueError):
                continue
        if candidate_numbers:
            self.state["next_candidate"] = max(
                int(self.state["next_candidate"]), max(candidate_numbers) + 1)
        task_numbers = []
        for queue_name in ("pending", "running", "completed", "ingested", "failed"):
            for filename in (self.queue / queue_name).glob("*.json"):
                try:
                    task_numbers.append(int(filename.name.split("_", 2)[1]))
                except (IndexError, ValueError):
                    continue
        if task_numbers:
            self.state["next_task"] = max(
                int(self.state["next_task"]), max(task_numbers) + 1)

    def save(self):
        start = time.monotonic()
        self.state["updated"] = timestamp()
        self.state["rng_state"] = self.rng.bit_generator.state
        atomic_json(self.state_path, self.state, compact=True)
        self.metrics["save_count"] += 1
        self.metrics["save_s"] += time.monotonic() - start

    def candidate_path(self, candidate_id, name="POSCAR.initial"):
        return self.root / "candidates" / candidate_id / name

    def save_fingerprint(self, candidate_id, label, values):
        path = self.root / "candidates" / candidate_id / f"fingerprint.{label}.npy"
        values = np.asarray(values, dtype=float)
        np.save(path, values, allow_pickle=False)
        self.fp_cache[str(path)] = values
        return str(path)

    def load_fingerprint(self, filename):
        filename = str(filename)
        if filename not in self.fp_cache:
            self.fp_cache[filename] = np.load(filename, allow_pickle=False)
        return self.fp_cache[filename]

    def read_candidate_atoms(self, candidate_id):
        record = self.state["candidates"][candidate_id]
        filename = record.get("final_poscar") or record.get("latest_poscar") \
            or str(self.candidate_path(candidate_id))
        return read(filename, format="vasp")

    def is_duplicate(self, x_value, topology, values):
        threshold = float(self.config["duplicate_fingerprint_rms_A"])
        for filename in self.initial_index.get((x_value, topology), []):
            if fingerprint_rms(values, self.load_fingerprint(filename)) < threshold:
                return True
        return False

    def choose_source(self, x_value):
        population = self.state["populations"][str(x_value)]
        neighbours = []
        for neighbour in (x_value - 1, x_value + 1):
            if str(neighbour) in self.state["populations"]:
                neighbours.extend(self.state["populations"][str(neighbour)])
        base_airss = float(self.config["airss_fraction"])
        airss_weight = base_airss
        completed = [self.state["candidates"][item]
                     for item in self.full_history[str(x_value)]]
        stagnation_count = int(self.config["stagnation_full_count"])
        if len(completed) > stagnation_count:
            earlier_best = min(item["final_energy_eV"]
                               for item in completed[:-stagnation_count])
            recent_best = min(item["final_energy_eV"]
                              for item in completed[-stagnation_count:])
            if recent_best >= earlier_best - 0.10:
                airss_weight = float(self.config["stagnation_airss_fraction"])
        scale = ((1.0 - airss_weight) / (1.0 - base_airss)
                 if base_airss < 1.0 else 0.0)
        choices = ["airss"]
        weights = [airss_weight]
        if population:
            choices.append("mutation")
            weights.append(float(self.config["mutation_fraction"]) * scale)
        if len(population) >= 2:
            choices.append("crossover")
            weights.append(float(self.config["crossover_fraction"]) * scale)
        if neighbours:
            choices.append("ladder")
            weights.append(float(self.config["ladder_fraction"]) * scale)
        weights = np.asarray(weights, dtype=float)
        weights /= weights.sum()
        return str(self.rng.choice(np.asarray(choices), p=weights))

    def generate_candidate(self, x_value, round_index):
        start = time.monotonic()
        population = self.state["populations"][str(x_value)]
        attempts = int(self.config["generation_attempts"])
        for _ in range(attempts):
            source = self.choose_source(x_value)
            parents = []
            try:
                if source == "airss":
                    atoms, detail = AIRSSBuilder(self.template, x_value, self.rng).build()
                    operation = "airss:" + ",".join(detail)
                    roots = []
                elif source == "mutation":
                    parent_id = str(self.rng.choice(population))
                    atoms, operation = mutate_candidate(
                        self.read_candidate_atoms(parent_id), self.template, self.rng)
                    parents = [parent_id]
                    roots = self.state["candidates"][parent_id]["roots"]
                elif source == "crossover":
                    parent_ids = list(map(str, self.rng.choice(
                        population, size=2, replace=False)))
                    atoms, operation = crossover_candidate(
                        self.read_candidate_atoms(parent_ids[0]),
                        self.read_candidate_atoms(parent_ids[1]), self.template, self.rng)
                    parents = parent_ids
                    roots = sorted(set(sum((self.state["candidates"][item]["roots"]
                                            for item in parents), [])))
                else:
                    options = []
                    for neighbour in (x_value - 1, x_value + 1):
                        options.extend(self.state["populations"].get(str(neighbour), []))
                    parent_id = str(self.rng.choice(options))
                    atoms, operation = ladder_candidate(
                        self.read_candidate_atoms(parent_id), x_value, self.template, self.rng)
                    parents = [parent_id]
                    roots = self.state["candidates"][parent_id]["roots"]
                report = validate_initial(atoms, x_value)
                if not report["valid"]:
                    continue
                topology = topology_signature(atoms, x_value)
                values = fingerprint(atoms, x_value)
                if self.is_duplicate(x_value, topology, values):
                    continue
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                continue

            number = int(self.state["next_candidate"])
            candidate_id = f"x{x_value:02d}_c{number:07d}"
            self.state["next_candidate"] = number + 1
            if not roots:
                roots = [candidate_id]
            directory = self.root / "candidates" / candidate_id
            directory.mkdir(parents=True, exist_ok=False)
            poscar = directory / "POSCAR.initial"
            write(poscar, atoms, format="vasp", direct=True, vasp5=True, sort=False)
            initial_fingerprint_file = self.save_fingerprint(
                candidate_id, "initial", values)
            record = {
                "id": candidate_id,
                "x": x_value,
                "round": round_index,
                "source": source,
                "operation": operation,
                "parents": parents,
                "roots": roots,
                "created": timestamp(),
                "initial_topology": topology,
                "initial_fingerprint_file": initial_fingerprint_file,
                "initial_key": structure_key(atoms, x_value),
                "latest_poscar": str(poscar),
                "stages": {},
                "full_converged": False,
            }
            self.state["candidates"][candidate_id] = record
            self.initial_index.setdefault((x_value, topology), []).append(
                initial_fingerprint_file)
            atomic_json(directory / "metadata.json", record)
            elapsed = time.monotonic() - start
            self.metrics["generation_count"] += 1
            self.metrics["generation_s"] += elapsed
            if elapsed >= 5.0:
                log(f"SLOW_GENERATION x={x_value} elapsed={elapsed:.2f}s "
                    f"source={source}")
            return candidate_id
        raise RuntimeError(f"Failed to generate a unique candidate for x={x_value}")

    def enqueue(self, candidate_id, stage, input_poscar, attempt=1):
        sequence = int(self.state["next_task"])
        self.state["next_task"] = sequence + 1
        task_id = f"{candidate_id}_{stage.lower()}_a{attempt}"
        priority = {"FULL": 10, "MEDIUM": 20, "SHORT": 30}[stage]
        filename = f"{priority:03d}_{sequence:09d}_{task_id}.json"
        task = {
            "task_id": task_id,
            "candidate_id": candidate_id,
            "x": self.state["candidates"][candidate_id]["x"],
            "stage": stage,
            "attempt": attempt,
            "input_poscar": str(Path(input_poscar).resolve()),
            "created": timestamp(),
        }
        atomic_json(self.queue / "pending" / filename, task)
        return task_id

    def ensure_rounds(self):
        changed = False
        target = int(self.config["rounds_per_composition"])
        for x_value in self.config["compositions"]:
            rounds = self.state["rounds"][str(x_value)]
            previous_resolved = False
            if rounds:
                previous = rounds[-1]
                previous_resolved = previous["medium_selected"] and all(
                    self.state["candidates"][candidate_id].get("full_converged")
                    or self.state["candidates"][candidate_id].get("full_exhausted")
                    for candidate_id in previous["full_ids"]
                )
            if len(rounds) < target and (not rounds or previous_resolved):
                rounds.append({
                    "index": len(rounds) + 1,
                    "short_ids": [],
                    "short_selected": False,
                    "medium_ids": [],
                    "medium_selected": False,
                    "full_ids": [],
                })
                changed = True
        return changed

    def fill_pending(self):
        changed = False
        pending_count = len(list((self.queue / "pending").glob("*.json")))
        target = int(self.config["pending_target"])
        while pending_count < target:
            choices = []
            for x_value in self.config["compositions"]:
                rounds = self.state["rounds"][str(x_value)]
                if not rounds:
                    continue
                current = rounds[-1]
                missing = int(self.config["short_per_round"]) - len(current["short_ids"])
                if missing > 0:
                    completed = sum(len(item["short_ids"]) for item in rounds)
                    choices.append((completed, x_value, current))
            if not choices:
                break
            _, x_value, current = min(choices, key=lambda item: (item[0], item[1]))
            candidate_id = self.generate_candidate(x_value, current["index"])
            current["short_ids"].append(candidate_id)
            self.enqueue(candidate_id, "SHORT", self.candidate_path(candidate_id))
            pending_count += 1
            changed = True
        return changed

    def ingest_completed(self):
        changed = False
        processed = set(self.state["processed_tasks"])
        for task_file in sorted((self.queue / "completed").glob("*.json")):
            task = json.loads(task_file.read_text(encoding="utf-8"))
            task_id = task["task_id"]
            if task_id in processed:
                continue
            result = json.loads(Path(task["result_path"]).read_text(encoding="utf-8"))
            record = self.state["candidates"][task["candidate_id"]]
            stage_key = task["stage"].lower()
            if stage_key == "full":
                stage_key = f"full_a{task['attempt']}"
            fingerprint_file = None
            if result.get("fingerprint") is not None:
                fingerprint_file = self.save_fingerprint(
                    record["id"], stage_key, result["fingerprint"])
            summary = {
                key: result.get(key) for key in (
                    "status", "energy_eV", "free_energy_eV", "fmax_mobile_eVA",
                    "force_converged",
                    "elapsed_s", "contcar", "topology", "error",
                    "electronic_steps_last_ionic", "ionic_frames", "classification",
                    "evaluated_poscar", "outcar_contcar_position_rms_max_A")
            }
            summary["result_path"] = task["result_path"]
            summary["fingerprint_file"] = fingerprint_file
            record["stages"][stage_key] = summary
            record["latest_poscar"] = result.get("contcar") or record["latest_poscar"]
            record["latest_topology"] = (
                result.get("topology") or record["initial_topology"])
            if fingerprint_file:
                record["latest_fingerprint_file"] = fingerprint_file
            self.state["processed_tasks"].append(task_id)
            processed.add(task_id)
            changed = True
            log(f"INGEST {task_id} status={result['status']} "
                f"E={result.get('energy_eV')} fmax={result.get('fmax_mobile_eVA')}")

            if task["stage"] == "FULL":
                if result["status"] == "OK" and result.get("force_converged", False):
                    record["full_converged"] = True
                    record["final_poscar"] = (
                        result.get("evaluated_poscar") or result["contcar"])
                    record["final_energy_eV"] = result["energy_eV"]
                    record["final_fmax_eVA"] = result["fmax_mobile_eVA"]
                    record["final_completed"] = timestamp()
                    if record["id"] not in self.full_history[str(record["x"])]:
                        self.full_history[str(record["x"])].append(record["id"])
                    self.add_to_archive(record["id"])
                elif task["attempt"] < int(self.config["full_max_attempts"]):
                    continuation = result.get("contcar") or task["input_poscar"]
                    self.enqueue(record["id"], "FULL", continuation,
                                 attempt=task["attempt"] + 1)
                else:
                    record["full_exhausted"] = True
        return changed

    def archive_processed_task_files(self):
        """Move committed queue records out of the high-frequency scan path."""
        processed = set(self.state["processed_tasks"])
        destination = self.queue / "ingested"
        moved = 0
        for task_file in (self.queue / "completed").glob("*.json"):
            task = json.loads(task_file.read_text(encoding="utf-8"))
            if task.get("task_id") not in processed:
                continue
            os.replace(task_file, destination / task_file.name)
            moved += 1
        if moved:
            log(f"QUEUE archived_ingested={moved}")

    def stage_result(self, candidate_id, stage):
        return self.state["candidates"][candidate_id]["stages"].get(stage)

    def valid_stage_ids(self, candidate_ids, stage):
        valid = []
        for candidate_id in candidate_ids:
            result = self.stage_result(candidate_id, stage)
            energy = None if result is None else result.get("energy_eV")
            if (result is not None and result.get("status") == "OK"
                    and energy is not None and np.isfinite(energy)
                    and result.get("contcar")):
                valid.append(candidate_id)
        return valid

    def select_diverse(self, candidate_ids, stage, count):
        if count <= 0:
            return []
        valid = self.valid_stage_ids(candidate_ids, stage)
        if len(valid) <= count:
            return valid
        valid.sort(key=lambda item: self.stage_result(item, stage)["energy_eV"])
        energy_count = max(1, count // 2)
        selected = valid[:energy_count]
        remaining = valid[energy_count:]
        while remaining and len(selected) < count:
            best_id = None
            best_score = None
            energy_min = self.stage_result(valid[0], stage)["energy_eV"]
            for candidate_id in remaining:
                values = self.load_fingerprint(
                    self.stage_result(candidate_id, stage)["fingerprint_file"])
                novelty = min(
                    fingerprint_rms(
                        values,
                        self.load_fingerprint(
                            self.stage_result(chosen, stage)["fingerprint_file"]),
                    )
                    for chosen in selected
                )
                energy_penalty = 0.01 * max(
                    0.0, self.stage_result(candidate_id, stage)["energy_eV"] - energy_min)
                score = (novelty - energy_penalty,
                         self.stage_result(candidate_id, stage)["topology"])
                if best_score is None or score > best_score:
                    best_score = score
                    best_id = candidate_id
            selected.append(best_id)
            remaining.remove(best_id)
        return selected

    def process_promotions(self):
        changed = False
        for x_value in self.config["compositions"]:
            for round_record in self.state["rounds"][str(x_value)]:
                short_ids = round_record["short_ids"]
                if (len(short_ids) >= int(self.config["short_per_round"])
                        and not round_record["short_selected"]
                        and all(self.stage_result(item, "short") is not None
                                for item in short_ids)):
                    required = int(self.config["medium_per_round"])
                    valid_short = self.valid_stage_ids(short_ids, "short")
                    missing = required - len(valid_short)
                    if missing > 0:
                        limit = int(self.config.get("short_replacement_limit", 0))
                        used = len(short_ids) - int(self.config["short_per_round"])
                        available = max(0, limit - used)
                        if available == 0:
                            raise RuntimeError(
                                f"x={x_value} round={round_record['index']} has only "
                                f"{len(valid_short)}/{required} valid SHORT results after "
                                f"{used} replacements")
                        added = min(missing, available)
                        for _ in range(added):
                            candidate_id = self.generate_candidate(
                                x_value, round_record["index"])
                            short_ids.append(candidate_id)
                            self.enqueue(candidate_id, "SHORT",
                                         self.candidate_path(candidate_id))
                        log(f"REPLACE x={x_value} round={round_record['index']} "
                            f"SHORT added={added} valid={len(valid_short)}/{required}")
                        changed = True
                        continue
                    selected = self.select_diverse(
                        short_ids, "short", required)
                    if len(selected) != required:
                        raise RuntimeError("SHORT promotion did not meet its exact quota")
                    round_record["medium_ids"] = selected
                    round_record["short_selected"] = True
                    for candidate_id in selected:
                        result = self.stage_result(candidate_id, "short")
                        self.enqueue(candidate_id, "MEDIUM", result["contcar"])
                    log(f"PROMOTE x={x_value} round={round_record['index']} "
                        f"SHORT->{len(selected)} MEDIUM")
                    changed = True

                medium_ids = round_record["medium_ids"]
                if (round_record["short_selected"] and not round_record["medium_selected"]
                        and all(self.stage_result(item, "medium") is not None
                                for item in medium_ids)):
                    required = int(self.config["full_per_round"])
                    valid_medium = self.valid_stage_ids(medium_ids, "medium")
                    missing = required - len(valid_medium)
                    if missing > 0:
                        limit = int(self.config.get("medium_replacement_limit", 0))
                        used = len(medium_ids) - int(self.config["medium_per_round"])
                        available_slots = max(0, limit - used)
                        unused_short = [
                            item for item in self.valid_stage_ids(
                                round_record["short_ids"], "short")
                            if item not in medium_ids
                        ]
                        added_ids = self.select_diverse(
                            unused_short, "short", min(missing, available_slots))
                        if not added_ids:
                            raise RuntimeError(
                                f"x={x_value} round={round_record['index']} has only "
                                f"{len(valid_medium)}/{required} valid MEDIUM results and "
                                "no bounded replacement remains")
                        for candidate_id in added_ids:
                            medium_ids.append(candidate_id)
                            result = self.stage_result(candidate_id, "short")
                            self.enqueue(candidate_id, "MEDIUM", result["contcar"])
                        log(f"REPLACE x={x_value} round={round_record['index']} "
                            f"MEDIUM added={len(added_ids)} "
                            f"valid={len(valid_medium)}/{required}")
                        changed = True
                        continue
                    selected = self.select_diverse(
                        medium_ids, "medium", required)
                    if len(selected) != required:
                        raise RuntimeError("MEDIUM promotion did not meet its exact quota")
                    round_record["full_ids"] = selected
                    round_record["medium_selected"] = True
                    for candidate_id in selected:
                        result = self.stage_result(candidate_id, "medium")
                        self.enqueue(candidate_id, "FULL", result["contcar"])
                    log(f"PROMOTE x={x_value} round={round_record['index']} "
                        f"MEDIUM->{len(selected)} FULL")
                    changed = True
        return changed

    def add_to_archive(self, candidate_id):
        record = self.state["candidates"][candidate_id]
        archive = self.state["archives"][str(record["x"])]
        new_values = self.load_fingerprint(record["latest_fingerprint_file"])
        duplicate_rms = float(self.config["archive_duplicate_fingerprint_rms_A"])
        duplicate_energy = float(self.config["archive_duplicate_energy_eV"])
        for existing_id in list(archive):
            existing = self.state["candidates"][existing_id]
            if existing.get("latest_topology") != record.get("latest_topology"):
                continue
            existing_values = self.load_fingerprint(
                existing["latest_fingerprint_file"])
            if (fingerprint_rms(new_values, existing_values) < duplicate_rms
                    and abs(existing["final_energy_eV"] - record["final_energy_eV"])
                    < duplicate_energy):
                if record["final_energy_eV"] < existing["final_energy_eV"]:
                    archive.remove(existing_id)
                    existing["duplicate_of"] = candidate_id
                    break
                record["duplicate_of"] = existing_id
                self.update_population(record["x"])
                return
        if candidate_id not in archive:
            archive.append(candidate_id)
        self.update_population(record["x"])

    def update_population(self, x_value):
        archive = [item for item in self.state["archives"][str(x_value)]
                   if self.state["candidates"][item].get("full_converged")]
        if not archive:
            return
        archive.sort(key=lambda item: self.state["candidates"][item]["final_energy_eV"])
        elite_count = min(int(self.config["elite_size"]), len(archive))
        selected = archive[:elite_count]
        energy_min = self.state["candidates"][archive[0]]["final_energy_eV"]
        window = float(self.config["population_energy_window_eV"])
        remaining = [item for item in archive[elite_count:]
                     if self.state["candidates"][item]["final_energy_eV"] <= energy_min + window]
        while remaining and len(selected) < int(self.config["population_size"]):
            best_id = None
            best_score = None
            for candidate_id in remaining:
                values = self.load_fingerprint(
                    self.state["candidates"][candidate_id]["latest_fingerprint_file"])
                novelty = min(
                    fingerprint_rms(
                        values,
                        self.load_fingerprint(
                            self.state["candidates"][chosen]["latest_fingerprint_file"]),
                    )
                    for chosen in selected
                )
                score = (novelty, -self.state["candidates"][candidate_id]["final_energy_eV"])
                if best_score is None or score > best_score:
                    best_score = score
                    best_id = candidate_id
            selected.append(best_id)
            remaining.remove(best_id)
        self.state["populations"][str(x_value)] = selected
        log(f"POPULATION x={x_value} size={len(selected)} archive={len(archive)} "
            f"best={energy_min:.8f} eV")

    def active_tasks(self):
        return (len(list((self.queue / "pending").glob("*.json")))
                + len(list((self.queue / "running").glob("*.json"))))

    def all_done(self):
        target_rounds = int(self.config["rounds_per_composition"])
        for x_value in self.config["compositions"]:
            rounds = self.state["rounds"][str(x_value)]
            if len(rounds) < target_rounds:
                return False
            for round_record in rounds:
                if not round_record["medium_selected"]:
                    return False
                for candidate_id in round_record["full_ids"]:
                    record = self.state["candidates"][candidate_id]
                    if not record.get("full_converged") and not record.get("full_exhausted"):
                        full_results = [key for key in record["stages"] if key.startswith("full_")]
                        if not full_results or len(full_results) < int(self.config["full_max_attempts"]):
                            return False
        return self.active_tasks() == 0

    def heartbeat(self, start):
        completed = len(self.state["processed_tasks"])
        total_candidates = len(self.state["candidates"])
        pending = len(list((self.queue / "pending").glob("*.json")))
        running = len(list((self.queue / "running").glob("*.json")))
        full = sum(record.get("full_converged", False)
                   for record in self.state["candidates"].values())
        elapsed = time.monotonic() - start
        stage_defaults = {
            "SHORT": int(self.config["short_nsw"]) * 60.0,
            "MEDIUM": int(self.config["medium_nsw"]) * 60.0,
            "FULL": 30.0 * 60.0,
        }
        durations = {stage: [] for stage in stage_defaults}
        done = {stage: 0 for stage in stage_defaults}
        for record in self.state["candidates"].values():
            for key, result in record["stages"].items():
                stage = "SHORT" if key == "short" else "MEDIUM" if key == "medium" else "FULL"
                done[stage] += 1
                if result.get("elapsed_s"):
                    durations[stage].append(float(result["elapsed_s"]))
        per_round = {
            "SHORT": int(self.config["short_per_round"]),
            "MEDIUM": int(self.config["medium_per_round"]),
            "FULL": int(self.config["full_per_round"]),
        }
        planned = {stage: len(self.config["compositions"])
                   * int(self.config["rounds_per_composition"]) * count
                   for stage, count in per_round.items()}
        remaining_node_seconds = 0.0
        for stage in stage_defaults:
            mean = (sum(durations[stage]) / len(durations[stage])
                    if durations[stage] else stage_defaults[stage])
            remaining_node_seconds += max(0, planned[stage] - done[stage]) * mean
        eta_hours = remaining_node_seconds / (6.0 * 3600.0)
        generation_average = (self.metrics["generation_s"]
                              / max(1, self.metrics["generation_count"]))
        save_average = (self.metrics["save_s"]
                        / max(1, self.metrics["save_count"]))
        log(f"RUNNING elapsed={elapsed / 3600:.2f}h candidates={total_candidates} "
            f"tasks_done={completed} pending={pending} running={running} full={full} "
            f"ETA_DFT_pool={eta_hours:.2f}h "
            f"controller_gen_avg={generation_average:.3f}s "
            f"state_save_avg={save_average:.3f}s")

    def run(self):
        start = time.monotonic()
        last_heartbeat = 0.0
        while True:
            if (self.root / "WORKER_FAILED").exists():
                self.save()
                (self.root / "STOP_WORKERS").touch()
                raise RuntimeError("A worker process failed; search stopped for recovery")
            if (self.root / "STOP").exists():
                self.save()
                (self.root / "STOP_WORKERS").touch()
                log("STOP requested; state saved and workers asked to exit")
                return
            changed = self.ingest_completed()
            changed |= self.process_promotions()
            changed |= self.ensure_rounds()
            changed |= self.fill_pending()
            if changed:
                self.save()
                self.archive_processed_task_files()
            if self.all_done():
                self.save()
                (self.root / "SEARCH_COMPLETE").touch()
                log("SEARCH COMPLETE")
                return
            now = time.monotonic()
            if now - last_heartbeat >= int(self.config["heartbeat_seconds"]):
                self.heartbeat(start)
                last_heartbeat = now
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--template", required=True)
    args = parser.parse_args()
    Controller(args).run()


if __name__ == "__main__":
    main()

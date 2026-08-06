#!/usr/bin/env python3
"""Non-destructively close abandoned queue tasks after all workers are stopped."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root")
    parser.add_argument("--confirm-workers-stopped", action="store_true")
    args = parser.parse_args()
    if not args.confirm_workers_stopped:
        raise SystemExit("Refusing recovery without --confirm-workers-stopped")
    root = Path(args.run_root).resolve()
    for task_file in sorted((root / "queue" / "running").glob("*.json")):
        task = json.loads(task_file.read_text(encoding="utf-8"))
        failure = root / "failures" / task["task_id"] / "result.json"
        failure.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(failure, {
            "task_id": task["task_id"], "candidate_id": task["candidate_id"],
            "x": task["x"], "stage": task["stage"], "attempt": task["attempt"],
            "status": "FAILED", "error": "Worker stopped before terminal result",
            "recovered": datetime.now().astimezone().isoformat(timespec="seconds"),
        })
        task["result_path"] = str(failure)
        task["completed"] = datetime.now().astimezone().isoformat(timespec="seconds")
        atomic_json(task_file, task)
        os.replace(task_file, root / "queue" / "completed" / task_file.name)
        print(f"Recovered {task['task_id']} as a preserved failure record")


if __name__ == "__main__":
    main()

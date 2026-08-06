#!/usr/bin/env python3
"""Print a compact snapshot of an asynchronous search."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root")
    args = parser.parse_args()
    root = Path(args.run_root)
    state_file = root / "state" / "controller_state.json"
    if not state_file.is_file():
        raise FileNotFoundError(state_file)
    state = json.loads(state_file.read_text(encoding="utf-8"))
    queue_counts = {name: len(list((root / "queue" / name).glob("*.json")))
                    for name in ("pending", "running", "completed", "failed")}
    queue_counts["committed"] = len(state["processed_tasks"])
    print(f"updated={state.get('updated')} queue={queue_counts}")
    print("x generated short medium full best_eV population archive")
    for x_key in sorted(state["rounds"], key=int):
        records = [record for record in state["candidates"].values()
                   if str(record["x"]) == x_key]
        short = sum("short" in record["stages"] for record in records)
        medium = sum("medium" in record["stages"] for record in records)
        full = [record for record in records if record.get("full_converged")]
        best = min((record["final_energy_eV"] for record in full), default=None)
        best_text = "NA" if best is None else f"{best:.8f}"
        print(f"{int(x_key):2d} {len(records):9d} {short:5d} {medium:6d} "
              f"{len(full):4d} {best_text:>14} "
              f"{len(state['populations'][x_key]):10d} {len(state['archives'][x_key]):7d}")
    workers = sorted((root / "workers").glob("worker_*.json"))
    for worker in workers:
        data = json.loads(worker.read_text(encoding="utf-8"))
        print(f"worker={data.get('worker')} task={data.get('task')} "
              f"updated={data.get('updated')} elapsed_s={data.get('elapsed_s')}")
    for marker in ("STOP", "STOP_WORKERS", "SEARCH_COMPLETE"):
        if (root / marker).exists():
            print(f"marker={marker}")


if __name__ == "__main__":
    main()

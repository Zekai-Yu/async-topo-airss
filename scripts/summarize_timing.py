#!/usr/bin/env python3
"""Summarize measured VASP wall time per completed ionic frame."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root")
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    state = json.loads((root / "state" / "controller_state.json").read_text(
        encoding="utf-8"))
    grouped = {}
    for record in state["candidates"].values():
        for summary in record["stages"].values():
            if summary.get("status") != "OK" or not summary.get("ionic_frames"):
                continue
            result = json.loads(Path(summary["result_path"]).read_text(
                encoding="utf-8"))
            stage = result["stage"]
            nproc = int(result["nproc"])
            seconds_per_frame = (float(result["elapsed_s"])
                                 / int(result["ionic_frames"]))
            grouped.setdefault((nproc, stage), []).append(seconds_per_frame)
    if not grouped:
        raise RuntimeError("No successful timed VASP result was found")
    print("ranks stage calls mean_s_per_ionic median_s_per_ionic")
    for (nproc, stage), values in sorted(grouped.items()):
        print(f"{nproc:5d} {stage:6s} {len(values):5d} "
              f"{statistics.mean(values):16.3f} "
              f"{statistics.median(values):18.3f}")


if __name__ == "__main__":
    main()

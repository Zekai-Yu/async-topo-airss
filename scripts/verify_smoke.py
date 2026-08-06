#!/usr/bin/env python3
"""Verify the complete two-worker smoke-test data path."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root")
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    if not (root / "SEARCH_COMPLETE").is_file():
        raise RuntimeError("Smoke run has no SEARCH_COMPLETE marker")
    state = json.loads((root / "state" / "controller_state.json").read_text(
        encoding="utf-8"))
    records = [item for item in state["candidates"].values() if item["x"] == 5]
    stage_counts = {"short": 0, "medium": 0, "full": 0}
    for record in records:
        for stage, summary in record["stages"].items():
            family = "full" if stage.startswith("full_a") else stage
            stage_counts[family] += 1
            result = json.loads(Path(summary["result_path"]).read_text(
                encoding="utf-8"))
            if result.get("status") != "OK":
                raise RuntimeError(f"Smoke task failed: {result.get('task_id')}")
            if result.get("master_sha256_before") != result.get("master_sha256_after"):
                raise RuntimeError("A master VASP input changed during smoke testing")
            for key in ("contcar", "evaluated_poscar"):
                if not Path(result[key]).is_file():
                    raise FileNotFoundError(result[key])
            if (result.get("energy_eV") is None
                    or result.get("free_energy_eV") is None
                    or result.get("fmax_mobile_eVA") is None):
                raise RuntimeError("Smoke result lacks a parsed energy or mobile force")
    expected = {"short": 2, "medium": 1, "full": 1}
    if stage_counts != expected:
        raise RuntimeError(f"Smoke stage counts are {stage_counts}; expected {expected}")
    if sum(item.get("full_converged", False) for item in records) != 1:
        raise RuntimeError("Smoke run did not produce exactly one terminal FULL record")
    print(f"SMOKE VERIFICATION PASSED stage_counts={stage_counts}")


if __name__ == "__main__":
    main()

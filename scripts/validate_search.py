#!/usr/bin/env python3
"""Hard validation gate for completed search state and exported POSCAR files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import read

from chemistry import blocks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root")
    parser.add_argument("--minimum-full", type=int, default=10)
    parser.add_argument("--require-complete-marker", action="store_true")
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    if args.require_complete_marker and not (root / "SEARCH_COMPLETE").is_file():
        raise RuntimeError("SEARCH_COMPLETE marker is absent")
    state = json.loads((root / "state" / "controller_state.json").read_text(
        encoding="utf-8"))
    errors = []
    for x_value in range(11):
        key = str(x_value)
        identifiers = [item for item in state["archives"].get(key, [])
                       if state["candidates"][item].get("full_converged")]
        if len(identifiers) < args.minimum_full:
            errors.append(f"x={x_value} has only {len(identifiers)} converged FULL structures")
        for candidate_id in identifiers:
            record = state["candidates"][candidate_id]
            atoms = read(record["final_poscar"], format="vasp")
            blocks(atoms, x_value)
            if not np.isfinite(record["final_energy_eV"]):
                errors.append(f"{candidate_id} has non-finite energy")
            if record["final_fmax_eVA"] > 0.05 + 1.0e-8:
                errors.append(f"{candidate_id} fmax={record['final_fmax_eVA']}")
    if errors:
        print("VALIDATION FAILED")
        for error in errors:
            print("- " + error)
        raise SystemExit(2)
    print("SEARCH VALIDATION PASSED")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Assess independent-lineage reproduction and recent energy stability."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root")
    parser.add_argument("--minimum-full", type=int, default=20)
    parser.add_argument("--recent", type=int, default=10)
    parser.add_argument("--reproduction-eV", type=float, default=0.05)
    parser.add_argument("--stability-eV", type=float, default=0.10)
    parser.add_argument("--ignore-reproduction", action="store_true",
                        help="Assess only FULL count and recent energy stability.")
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    state = json.loads((root / "state" / "controller_state.json").read_text(
        encoding="utf-8"))
    all_passed = True
    print("x full best_eV reproduced stable status")
    for x_value in range(11):
        records = [item for item in state["candidates"].values()
                   if item["x"] == x_value and item.get("full_converged")]
        records.sort(key=lambda item: item.get("final_completed", item["created"]))
        if not records:
            print(f"{x_value:2d} 0 NA False False INCOMPLETE")
            all_passed = False
            continue
        best = min(records, key=lambda item: item["final_energy_eV"])
        best_roots = set(best["roots"])
        reproduced = any(
            item["id"] != best["id"]
            and abs(item["final_energy_eV"] - best["final_energy_eV"])
            <= args.reproduction_eV
            and best_roots.isdisjoint(item["roots"])
            for item in records
        )
        stable = False
        if len(records) > args.recent:
            earlier_best = min(item["final_energy_eV"] for item in records[:-args.recent])
            recent_best = min(item["final_energy_eV"] for item in records[-args.recent:])
            stable = recent_best >= earlier_best - args.stability_eV
        passed = (len(records) >= args.minimum_full and stable
                  and (args.ignore_reproduction or reproduced))
        all_passed &= passed
        print(f"{x_value:2d} {len(records):4d} {best['final_energy_eV']:.10f} "
              f"{reproduced!s:10s} {stable!s:6s} {'PASS' if passed else 'EXTEND'}")
    if not all_passed:
        gate = "count/stability" if args.ignore_reproduction else "full convergence"
        print(f"CONVERGENCE GATE NOT PASSED ({gate}): extend selected compositions/rounds")
        raise SystemExit(2)
    print("CONVERGENCE GATE PASSED")


if __name__ == "__main__":
    main()

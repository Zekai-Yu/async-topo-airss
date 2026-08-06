#!/usr/bin/env python3
"""Export low-energy diverse converged structures in strict O/Pd/Ce order."""
from __future__ import annotations

import argparse
import csv
import json
import itertools
from pathlib import Path

import numpy as np
from ase.io import read, write

from chemistry import blocks


def surface_operations():
    rotations = (
        np.array([[1, 0], [0, 1]], dtype=int),
        np.array([[0, 1], [-1, -1]], dtype=int),
        np.array([[-1, -1], [1, 0]], dtype=int),
    )
    reflection = np.array([[1, 0], [-1, -1]], dtype=int)
    for operation in list(rotations) + [rotation @ reflection for rotation in rotations]:
        for first, second in itertools.product(range(3), repeat=2):
            yield operation, np.array([first / 3.0, second / 3.0])


def symmetry_rmsd(first, second, x_value):
    """Exact element-matched mobile-atom RMSD over the CeO2(111) C3v group."""
    from scipy.optimize import linear_sum_assignment

    first_blocks = blocks(first, x_value)
    second_blocks = blocks(second, x_value)
    cell = first.cell.array
    if not np.allclose(cell, second.cell.array, atol=1.0e-6):
        return np.inf
    first_fractional = first.get_scaled_positions(wrap=False)
    second_fractional = second.get_scaled_positions(wrap=False)
    groups_first = (first_blocks["top_o"], first_blocks["cluster_o"], first_blocks["pd"])
    groups_second = (second_blocks["top_o"], second_blocks["cluster_o"], second_blocks["pd"])
    best = np.inf
    for operation, translation in surface_operations():
        transformed = first_fractional.copy()
        transformed[:, :2] = transformed[:, :2] @ operation + translation
        transformed[:, :2] %= 1.0
        squared = 0.0
        count = 0
        for first_group, second_group in zip(groups_first, groups_second):
            cost = np.empty((len(first_group), len(second_group)), dtype=float)
            for row, first_index in enumerate(first_group):
                for column, second_index in enumerate(second_group):
                    delta = second_fractional[second_index] - transformed[first_index]
                    delta[:2] -= np.rint(delta[:2])
                    cartesian = delta @ cell
                    cost[row, column] = np.dot(cartesian, cartesian)
            if len(first_group):
                rows, columns = linear_sum_assignment(cost)
                squared += float(np.sum(cost[rows, columns]))
                count += len(rows)
        if count:
            best = min(best, np.sqrt(squared / count))
    return float(best)


def select(records, count, threshold):
    records = sorted(records, key=lambda item: item["final_energy_eV"])
    if not records:
        return []
    for record in records:
        record["_atoms"] = read(record["final_poscar"], format="vasp")
    selected = [records[0]]
    remaining = records[1:]
    while remaining and len(selected) < count:
        eligible = []
        for record in remaining:
            distances = [
                symmetry_rmsd(record["_atoms"], item["_atoms"], record["x"])
                if record.get("latest_topology") == item.get("latest_topology") else np.inf
                for item in selected
            ]
            novelty = min(distances)
            if novelty >= threshold:
                eligible.append((record["final_energy_eV"], -novelty, record))
        if eligible:
            chosen = min(eligible, key=lambda item: (item[0], item[1]))[2]
        else:
            chosen = max(remaining, key=lambda record: min(
                symmetry_rmsd(record["_atoms"], item["_atoms"], record["x"])
                if record.get("latest_topology") == item.get("latest_topology") else np.inf
                for item in selected))
        selected.append(chosen)
        remaining.remove(chosen)
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root")
    parser.add_argument("--output", default=None)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--fingerprint-rms", type=float, default=0.35)
    args = parser.parse_args()
    root = Path(args.run_root).resolve()
    state = json.loads((root / "state" / "controller_state.json").read_text(
        encoding="utf-8"))
    output = Path(args.output).resolve() if args.output else root / "final_candidates"
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for x_key in sorted(state["archives"], key=int):
        records = [state["candidates"][item] for item in state["archives"][x_key]
                   if state["candidates"][item].get("full_converged")]
        chosen = select(records, args.count, args.fingerprint_rms)
        directory = output / f"x{int(x_key):02d}"
        directory.mkdir(parents=True, exist_ok=True)
        energy_min = min((item["final_energy_eV"] for item in chosen), default=0.0)
        previous_exported = []
        for rank, record in enumerate(chosen, 1):
            atoms = record.pop("_atoms")
            blocks(atoms, int(x_key))
            filename = directory / f"POSCAR_rank{rank:02d}"
            write(filename, atoms, format="vasp", direct=True, vasp5=True, sort=False)
            minimum_rmsd = min((
                symmetry_rmsd(atoms, previous_atoms, int(x_key))
                for previous_topology, previous_atoms in previous_exported
                if previous_topology == record.get("latest_topology")
            ), default=None)
            rows.append({
                "x": int(x_key), "rank": rank, "candidate": record["id"],
                "energy_eV": record["final_energy_eV"],
                "relative_eV": record["final_energy_eV"] - energy_min,
                "fmax_mobile_eVA": record["final_fmax_eVA"],
                "topology": record.get("latest_topology"),
                "source": record["source"], "operation": record["operation"],
                "parents": ",".join(record["parents"]),
                "roots": ",".join(record["roots"]), "poscar": str(filename),
                "min_same_topology_rmsd_A": minimum_rmsd,
            })
            previous_exported.append((record.get("latest_topology"), atoms))
    fields = ["x", "rank", "candidate", "energy_eV", "relative_eV",
              "fmax_mobile_eVA", "topology", "source", "operation", "parents",
              "roots", "poscar", "min_same_topology_rmsd_A"]
    with open(output / "summary.tsv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} converged structures to {output}")


if __name__ == "__main__":
    main()

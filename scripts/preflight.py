#!/usr/bin/env python3
"""Validate the slab and immutable VASP input contract before submission."""
from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

import numpy as np
from ase.constraints import FixAtoms
from ase.io import read


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def active_value(text, key):
    values = re.findall(rf"(?im)^\s*{re.escape(key)}\s*=\s*([^#;!\r\n]+)", text)
    if len(values) != 1:
        raise ValueError(f"Expected exactly one active {key}; found {len(values)}")
    return values[0].strip()


def fixed_indices(atoms):
    values = []
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            values.extend(map(int, constraint.get_indices()))
    return sorted(set(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True)
    parser.add_argument("--incar", required=True)
    parser.add_argument("--kpoints", required=True)
    parser.add_argument("--potcar", required=True)
    args = parser.parse_args()

    slab = read(args.template, format="vasp")
    if slab.get_chemical_symbols() != ["O"] * 18 + ["Ce"] * 9:
        raise ValueError("Template must use O18/Ce9 order")
    if fixed_indices(slab) != list(range(9)) + list(range(18, 27)):
        raise ValueError(f"Wrong template fixed atoms: {fixed_indices(slab)}")
    lengths = slab.cell.lengths()
    angles = slab.cell.angles()
    expected_lateral = 3.0 * 5.456 / np.sqrt(2.0)
    if not np.allclose(lengths[:2], expected_lateral, atol=1.0e-7):
        raise ValueError("Wrong 3x3 CeO2(111) lateral cell")
    if not np.allclose(angles, [90.0, 90.0, 120.0], atol=1.0e-7):
        raise ValueError("Wrong CeO2(111) cell angles")
    vacuum = lengths[2] - np.ptp(slab.positions[:, 2])
    if vacuum < 15.0:
        raise ValueError(f"Vacuum is only {vacuum:.6f} A")
    expected_ce_o = 5.456 * np.sqrt(3.0) / 4.0
    distances = slab.get_all_distances(mic=True)
    ce_coordination = [int(np.sum(np.abs(distances[i, :18] - expected_ce_o) < 1.0e-6))
                       for i in range(18, 27)]
    o_coordination = [int(np.sum(np.abs(distances[i, 18:27] - expected_ce_o) < 1.0e-6))
                      for i in range(18)]
    if ce_coordination != [6] * 9 or o_coordination != [3] * 18:
        raise ValueError("Template is not the expected fluorite CeO2(111) trilayer")

    paths = {"INCAR": Path(args.incar), "KPOINTS": Path(args.kpoints),
             "POTCAR": Path(args.potcar)}
    for label, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty {label}: {path}")
    incar = paths["INCAR"].read_text(encoding="utf-8")
    values = {key: active_value(incar, key) for key in ("IBRION", "NSW", "NELM", "EDIFFG")}
    potcar = paths["POTCAR"].read_text(encoding="utf-8", errors="ignore")
    order = re.findall(r"TITEL\s*=.*?\b(O|Pd|Ce)\b", potcar)
    if order != ["O", "Pd", "Ce"]:
        raise ValueError(f"POTCAR order is {order}; expected O/Pd/Ce")
    kpoints = paths["KPOINTS"].read_text(encoding="utf-8", errors="replace")
    if not re.search(r"(?im)^\s*G", kpoints) or not re.search(r"(?m)^\s*1\s+1\s+1\s*$", kpoints):
        raise ValueError("KPOINTS must be Gamma-centered 1x1x1")

    print("PREFLIGHT PASSED", flush=True)
    print(f"CeO2 Fm-3m a0=5.456 A; Ce-O={expected_ce_o:.10f} A", flush=True)
    print(f"cell={lengths.tolist()} angles={angles.tolist()} vacuum={vacuum:.6f} A", flush=True)
    print("fixed=bottom-O(9)+Ce(9); mobile=top-O(9)+Pd5Ox", flush=True)
    print(f"master INCAR values={values}; workers replace only NSW and NELM", flush=True)
    for label, path in paths.items():
        print(f"MASTER_SHA256 {label} {digest(path)}", flush=True)


if __name__ == "__main__":
    main()

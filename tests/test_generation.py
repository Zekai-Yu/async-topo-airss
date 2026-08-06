"""Offline checks for the public input template and search operators."""
from pathlib import Path

import numpy as np
from ase.constraints import FixAtoms
from ase.io import read

from chemistry import AIRSSBuilder, blocks, topology_signature, validate_initial


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "inputs" / "POSCAR_CeO2_111_3x3_1TL"


def test_template_contract():
    atoms = read(TEMPLATE, format="vasp")
    assert atoms.get_chemical_symbols() == ["O"] * 18 + ["Ce"] * 9
    fixed = []
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            fixed.extend(constraint.get_indices())
    assert sorted(fixed) == list(range(9)) + list(range(18, 27))
    assert np.allclose(atoms.cell.lengths()[:2], 3.0 * 5.456 / np.sqrt(2.0))
    assert np.allclose(atoms.cell.angles(), [90.0, 90.0, 120.0])


def test_seeded_airss_candidates_obey_contract():
    template = read(TEMPLATE, format="vasp")
    rng = np.random.default_rng(20260806)
    for x_value in range(11):
        atoms, _ = AIRSSBuilder(template, x_value, rng).build()
        assert validate_initial(atoms, x_value)["valid"]
        assert blocks(atoms, x_value)["x"] == x_value
        assert len(topology_signature(atoms, x_value)) == 64

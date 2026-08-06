#!/usr/bin/env python3
"""Fast chemistry, topology and structure operators for Pd5Ox/CeO2(111)."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms


@dataclass(frozen=True)
class Rules:
    """Initial-generation bounds; relaxed structures are classified, not rejected."""

    oo_hard: float = 1.80
    oo_generation: float = 2.00
    pdo_hard: float = 1.60
    pdo_bond: float = 2.35
    pdpd_hard: float = 2.15
    pdpd_bond: float = 3.10
    oce_hard: float = 1.85
    pdce_hard: float = 2.20


RULES = Rules()


def blocks(atoms: Atoms, expected_x: int | None = None) -> dict:
    """Return strict O/Pd/Ce blocks for the production POSCAR contract."""
    symbols = atoms.get_chemical_symbols()
    x_value = symbols.count("O") - 18
    expected = ["O"] * (18 + x_value) + ["Pd"] * 5 + ["Ce"] * 9
    if symbols != expected or not 0 <= x_value <= 10:
        raise ValueError("Structure must have strict O(18+x)/Pd5/Ce9 order")
    if expected_x is not None and x_value != expected_x:
        raise ValueError(f"Expected Pd5O{expected_x}, got Pd5O{x_value}")
    return {
        "x": x_value,
        "bottom_o": np.arange(0, 9, dtype=int),
        "top_o": np.arange(9, 18, dtype=int),
        "cluster_o": np.arange(18, 18 + x_value, dtype=int),
        "pd": np.arange(18 + x_value, 23 + x_value, dtype=int),
        "ce": np.arange(23 + x_value, 32 + x_value, dtype=int),
    }


def mic_delta(first, second, cell):
    """Minimum-image displacement with periodicity in the surface plane only."""
    delta = np.asarray(second, dtype=float) - np.asarray(first, dtype=float)
    fractional = np.linalg.solve(np.asarray(cell, dtype=float).T, delta)
    fractional[:2] -= np.rint(fractional[:2])
    return fractional @ np.asarray(cell, dtype=float)


def mic_distance(first, second, cell):
    return float(np.linalg.norm(mic_delta(first, second, cell)))


def wrap_xy(position, cell):
    fractional = np.linalg.solve(np.asarray(cell, dtype=float).T,
                                 np.asarray(position, dtype=float))
    fractional[:2] %= 1.0
    return fractional @ np.asarray(cell, dtype=float)


def unwrap_positions(positions, cell, reference=None):
    """Return one coherent cluster image around a reference atom."""
    positions = np.asarray(positions, dtype=float).reshape((-1, 3))
    if not len(positions):
        return positions.copy()
    reference = positions[0] if reference is None else np.asarray(reference, dtype=float)
    return np.asarray([reference + mic_delta(reference, position, cell)
                       for position in positions])


def make_ordered_atoms(template: Atoms, pd_positions, oxygen_positions) -> Atoms:
    """Build O/Pd/Ce order and restore the exact fixed-atom contract."""
    x_value = len(oxygen_positions)
    pd_positions = np.asarray([wrap_xy(item, template.cell.array)
                               for item in np.asarray(pd_positions).reshape((5, 3))])
    oxygen_positions = np.asarray([wrap_xy(item, template.cell.array)
                                   for item in np.asarray(oxygen_positions).reshape((-1, 3))],
                                  dtype=float).reshape((-1, 3))
    symbols = ["O"] * (18 + x_value) + ["Pd"] * 5 + ["Ce"] * 9
    positions = np.vstack((template.positions[:18],
                           oxygen_positions,
                           pd_positions,
                           template.positions[18:27]))
    atoms = Atoms(symbols=symbols, positions=positions,
                  cell=template.cell.copy(), pbc=(True, True, False))
    fixed = list(range(9)) + list(range(23 + x_value, 32 + x_value))
    atoms.set_constraint(FixAtoms(indices=fixed))
    return atoms


def extract_cluster(atoms: Atoms):
    index = blocks(atoms)
    pd_positions = unwrap_positions(atoms.positions[index["pd"]], atoms.cell.array)
    oxygen_positions = unwrap_positions(
        atoms.positions[index["cluster_o"]], atoms.cell.array,
        reference=pd_positions[0])
    return pd_positions, oxygen_positions


def _connected(pd_positions, oxygen_positions, cell):
    n_oxygen = len(oxygen_positions)
    adjacency = {i: set() for i in range(5 + n_oxygen)}
    for first, second in itertools.combinations(range(5), 2):
        if mic_distance(pd_positions[first], pd_positions[second], cell) <= RULES.pdpd_bond:
            adjacency[first].add(second)
            adjacency[second].add(first)
    for offset, oxygen in enumerate(oxygen_positions):
        node = 5 + offset
        for metal, position in enumerate(pd_positions):
            if mic_distance(oxygen, position, cell) <= RULES.pdo_bond:
                adjacency[node].add(metal)
                adjacency[metal].add(node)
    visited = {0}
    stack = [0]
    while stack:
        current = stack.pop()
        for neighbour in adjacency[current] - visited:
            visited.add(neighbour)
            stack.append(neighbour)
    return len(visited) == len(adjacency)


def validate_initial(atoms: Atoms, expected_x: int | None = None) -> dict:
    """Validate a generated candidate before any DFT call."""
    index = blocks(atoms, expected_x)
    errors = []
    pd_positions = atoms.positions[index["pd"]]
    cluster_o = atoms.positions[index["cluster_o"]]
    surface_o = atoms.positions[np.arange(18)]
    ce_positions = atoms.positions[index["ce"]]
    cell = atoms.cell.array
    top_z = float(np.max(atoms.positions[index["top_o"], 2]))

    if np.any(pd_positions[:, 2] < top_z + 0.30) or np.any(pd_positions[:, 2] > top_z + 6.20):
        errors.append("Pd lies outside the initial search height")
    if len(cluster_o) and (np.any(cluster_o[:, 2] < top_z + 0.30)
                           or np.any(cluster_o[:, 2] > top_z + 6.20)):
        errors.append("Generated O lies outside the initial search height")

    for first, second in itertools.combinations(pd_positions, 2):
        distance = mic_distance(first, second, cell)
        if distance < RULES.pdpd_hard:
            errors.append(f"Pd-Pd hard overlap {distance:.3f} A")
    for oxygen_index, oxygen in enumerate(cluster_o):
        all_other_o = list(surface_o) + [item for item_index, item in enumerate(cluster_o)
                                         if item_index != oxygen_index]
        if any(mic_distance(oxygen, item, cell) < RULES.oo_generation
               for item in all_other_o):
            errors.append("Generated O is closer than 2.00 A to another O")
        pd_distances = [mic_distance(oxygen, metal, cell) for metal in pd_positions]
        if min(pd_distances, default=999.0) < RULES.pdo_hard:
            errors.append("Pd-O hard overlap")
        if min(pd_distances, default=999.0) > RULES.pdo_bond:
            errors.append("Generated O has no primary Pd-O bond")
        if any(mic_distance(oxygen, cerium, cell) < RULES.oce_hard
               for cerium in ce_positions):
            errors.append("Generated O-Ce hard overlap")
    for metal in pd_positions:
        if any(mic_distance(metal, oxygen, cell) < RULES.pdo_hard
               for oxygen in surface_o):
            errors.append("Pd-surface-O hard overlap")
        if any(mic_distance(metal, cerium, cell) < RULES.pdce_hard
               for cerium in ce_positions):
            errors.append("Pd-Ce hard overlap")
    if not _connected(pd_positions, cluster_o, cell):
        errors.append("Pd5Ox primary-bond graph is disconnected")
    if min(mic_distance(metal, oxygen, cell)
           for metal in pd_positions for oxygen in atoms.positions[index["top_o"]]) > 2.80:
        errors.append("Cluster is not anchored to the mobile top-O layer")
    return {"valid": not errors, "errors": sorted(set(errors))}


def relaxed_classification(atoms: Atoms, expected_x: int | None = None) -> dict:
    """Describe relaxed chemistry without deleting unexpected DFT products."""
    index = blocks(atoms, expected_x)
    pd_positions = atoms.positions[index["pd"]]
    cluster_o = atoms.positions[index["cluster_o"]]
    surface_o = atoms.positions[np.arange(18)]
    cell = atoms.cell.array
    oo_min = min((mic_distance(first, second, cell)
                  for first in cluster_o for second in np.vstack((surface_o, cluster_o))
                  if np.linalg.norm(first - second) > 1.0e-10), default=None)
    free_o = sum(min(mic_distance(oxygen, metal, cell) for metal in pd_positions)
                 > RULES.pdo_bond for oxygen in cluster_o)
    return {
        "connected": _connected(pd_positions, cluster_o, cell),
        "cluster_oo_min_A": oo_min,
        "has_oo_bond": bool(oo_min is not None and oo_min < RULES.oo_hard),
        "oxygen_without_primary_pd": int(free_o),
        "pd_z_span_A": float(np.ptp(pd_positions[:, 2])),
    }


def topology_signature(atoms: Atoms, expected_x: int | None = None) -> str:
    """Return an exact Pd-label-invariant Pd-Pd/Pd-O graph key."""
    index = blocks(atoms, expected_x)
    pd_positions = atoms.positions[index["pd"]]
    oxygen_positions = atoms.positions[index["cluster_o"]]
    cell = atoms.cell.array
    pd_edges = [(i, j) for i, j in itertools.combinations(range(5), 2)
                if mic_distance(pd_positions[i], pd_positions[j], cell) <= RULES.pdpd_bond]
    masks = []
    for oxygen in oxygen_positions:
        mask = 0
        for metal, position in enumerate(pd_positions):
            if mic_distance(oxygen, position, cell) <= RULES.pdo_bond:
                mask |= 1 << metal
        masks.append(mask)
    canonical = None
    for permutation in itertools.permutations(range(5)):
        inverse = {old: new for new, old in enumerate(permutation)}
        edges = sorted(tuple(sorted((inverse[a], inverse[b]))) for a, b in pd_edges)
        permuted_masks = []
        for mask in masks:
            value = 0
            for old in range(5):
                if mask & (1 << old):
                    value |= 1 << inverse[old]
            permuted_masks.append(value)
        representation = json.dumps({"pp": edges, "op": sorted(permuted_masks)},
                                    separators=(",", ":"), sort_keys=True)
        if canonical is None or representation < canonical:
            canonical = representation
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def fingerprint(atoms: Atoms, expected_x: int | None = None) -> np.ndarray:
    """Cheap variable-atom fingerprint; constant support-support pairs are absent."""
    index = blocks(atoms, expected_x)
    groups = [
        (index["pd"], index["pd"], True),
        (index["pd"], index["cluster_o"], False),
        (index["cluster_o"], index["cluster_o"], True),
        (index["pd"], index["top_o"], False),
        (index["cluster_o"], index["top_o"], False),
    ]
    values = []
    positions = atoms.positions
    cell = atoms.cell.array
    inverse_cell = np.linalg.inv(cell)
    for first_group, second_group, same in groups:
        first_group = np.asarray(first_group, dtype=int)
        second_group = np.asarray(second_group, dtype=int)
        delta = (positions[second_group][None, :, :]
                 - positions[first_group][:, None, :])
        fractional = delta @ inverse_cell
        fractional[:, :, :2] -= np.rint(fractional[:, :, :2])
        distances = np.linalg.norm(fractional @ cell, axis=2)
        if same:
            triangle = np.triu_indices(len(first_group), k=1)
            block = distances[triangle]
        else:
            block = distances.ravel()
        values.extend(np.sort(block).tolist())
    return np.asarray(values, dtype=float)


def fingerprint_rms(first, second):
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.shape != second.shape or not len(first):
        return np.inf
    return float(np.sqrt(np.mean((first - second) ** 2)))


def structure_key(atoms: Atoms, expected_x: int | None = None, quantum=0.05) -> str:
    values = np.rint(fingerprint(atoms, expected_x) / quantum).astype(np.int32)
    return hashlib.sha256(topology_signature(atoms, expected_x).encode("ascii")
                          + values.tobytes()).hexdigest()


class AIRSSBuilder:
    """Chemically constrained random search without an energy model."""

    def __init__(self, template: Atoms, x_value: int, rng: np.random.Generator):
        self.template = template
        self.x_value = x_value
        self.rng = rng
        self.cell = template.cell.array
        self.top_o = template.positions[9:18]
        self.ce = template.positions[18:27]
        self.top_z = float(np.max(self.top_o[:, 2]))

    def unit_vector(self):
        vector = self.rng.normal(size=3)
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 1.0e-12 else np.array([0.0, 0.0, 1.0])

    def build_pd(self):
        for _ in range(600):
            anchor = self.top_o[int(self.rng.integers(len(self.top_o)))]
            first = wrap_xy(anchor + np.array([self.rng.normal(0, 0.25),
                                                self.rng.normal(0, 0.25),
                                                self.rng.uniform(1.85, 2.20)]), self.cell)
            positions = [first]
            for _atom in range(4):
                for _trial in range(400):
                    parent = positions[int(self.rng.integers(len(positions)))]
                    trial = wrap_xy(parent + self.unit_vector()
                                    * self.rng.uniform(2.25, 3.00), self.cell)
                    if not self.top_z + 0.55 <= trial[2] <= self.top_z + 5.80:
                        continue
                    distances = [mic_distance(trial, item, self.cell) for item in positions]
                    if min(distances) >= RULES.pdpd_hard and min(distances) <= RULES.pdpd_bond:
                        positions.append(trial)
                        break
                else:
                    break
            if len(positions) != 5:
                continue
            positions = np.asarray(positions)
            coherent = unwrap_positions(positions, self.cell)
            require_3d = self.rng.random() < 0.75
            singular = np.linalg.svd(coherent - coherent.mean(axis=0),
                                     compute_uv=False)[-1]
            if require_3d and (np.ptp(coherent[:, 2]) < 0.70 or singular < 0.12):
                continue
            return positions
        raise RuntimeError("AIRSS failed to construct a connected Pd5 backbone")

    def oxygen_trial(self, pd_positions, mode):
        pd_positions = unwrap_positions(pd_positions, self.cell)
        center = pd_positions.mean(axis=0)
        if mode == "terminal":
            index = int(self.rng.integers(5))
            radial = mic_delta(center, pd_positions[index], self.cell)
            radial = radial / max(np.linalg.norm(radial), 1.0e-12)
            direction = radial + 0.35 * self.unit_vector()
            direction /= np.linalg.norm(direction)
            return pd_positions[index] + direction * self.rng.uniform(1.78, 2.18)
        if mode == "bridge":
            pairs = list(itertools.combinations(range(5), 2))
            self.rng.shuffle(pairs)
            for first, second in pairs:
                vector = mic_delta(pd_positions[first], pd_positions[second], self.cell)
                distance = np.linalg.norm(vector)
                radius = self.rng.uniform(1.88, 2.20)
                if RULES.pdpd_hard <= distance < 2.0 * radius:
                    midpoint = pd_positions[first] + 0.5 * vector
                    axis = vector / distance
                    perpendicular = self.unit_vector()
                    perpendicular -= np.dot(perpendicular, axis) * axis
                    if np.linalg.norm(perpendicular) < 1.0e-8:
                        continue
                    perpendicular /= np.linalg.norm(perpendicular)
                    if np.dot(perpendicular, mic_delta(center, midpoint, self.cell)) < 0:
                        perpendicular *= -1
                    height = np.sqrt(radius * radius - 0.25 * distance * distance)
                    return midpoint + height * perpendicular
        indices = self.rng.choice(5, size=int(self.rng.choice([2, 3, 4])), replace=False)
        reference = pd_positions[indices[0]]
        points = [reference] + [reference + mic_delta(reference, pd_positions[i], self.cell)
                                for i in indices[1:]]
        return np.mean(points, axis=0) + 0.18 * self.unit_vector()

    def place_oxygen(self, pd_positions, existing, preferred_mode=None):
        modes = np.array(["terminal", "bridge", "internal"])
        probabilities = np.array([0.50, 0.35, 0.15])
        for _ in range(1000):
            mode = (str(preferred_mode) if preferred_mode is not None
                    else str(self.rng.choice(modes, p=probabilities)))
            trial = wrap_xy(self.oxygen_trial(pd_positions, mode), self.cell)
            if not self.top_z + 0.30 <= trial[2] <= self.top_z + 6.20:
                continue
            all_o = list(self.template.positions[:18]) + list(existing)
            if all_o and min(mic_distance(trial, item, self.cell) for item in all_o) < RULES.oo_generation:
                continue
            pd_distances = [mic_distance(trial, item, self.cell) for item in pd_positions]
            if min(pd_distances) < RULES.pdo_hard or min(pd_distances) > RULES.pdo_bond:
                continue
            if min(mic_distance(trial, item, self.cell) for item in self.ce) < RULES.oce_hard:
                continue
            return trial, mode
        raise RuntimeError("AIRSS failed to place an oxygen atom")

    def build(self):
        for _ in range(200):
            pd_positions = self.build_pd()
            oxygen_positions = []
            operations = []
            try:
                for _ in range(self.x_value):
                    oxygen, operation = self.place_oxygen(pd_positions, oxygen_positions)
                    oxygen_positions.append(oxygen)
                    operations.append(operation)
            except RuntimeError:
                continue
            atoms = make_ordered_atoms(self.template, pd_positions, oxygen_positions)
            if validate_initial(atoms, self.x_value)["valid"]:
                return atoms, operations
        raise RuntimeError(f"AIRSS failed for Pd5O{self.x_value}")


def _rotation_matrix(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    cross = np.array([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])
    return (np.eye(3) * np.cos(angle) + (1.0 - np.cos(angle)) * np.outer(axis, axis)
            + np.sin(angle) * cross)


def mutate_candidate(parent: Atoms, template: Atoms, rng: np.random.Generator):
    """Apply graph-changing or rigid cluster mutations; reject invalid products."""
    x_value = blocks(parent)["x"]
    original_pd, original_o = extract_cluster(parent)
    builder = AIRSSBuilder(template, x_value, rng)
    operations = ["oxygen_reinsert", "pd_reinsert", "rigid_move", "rattle"]
    if x_value == 0:
        operations.remove("oxygen_reinsert")
    for _ in range(100):
        operation = str(rng.choice(operations, p=None))
        pd_positions = original_pd.copy()
        oxygen_positions = original_o.copy()
        try:
            if operation == "oxygen_reinsert":
                remove = int(rng.integers(x_value))
                retained = np.delete(oxygen_positions, remove, axis=0)
                trial, _ = builder.place_oxygen(pd_positions, retained)
                oxygen_positions = np.vstack((retained, trial))
            elif operation == "pd_reinsert":
                remove = int(rng.integers(5))
                retained = np.delete(pd_positions, remove, axis=0)
                parent_index = int(rng.integers(4))
                trial = wrap_xy(retained[parent_index] + builder.unit_vector()
                                * rng.uniform(2.25, 3.00), parent.cell.array)
                pd_positions = np.vstack((retained, trial))
            elif operation == "rigid_move":
                cluster = np.vstack((pd_positions, oxygen_positions))
                center = cluster.mean(axis=0)
                rotation = _rotation_matrix(np.array([0.0, 0.0, 1.0]),
                                            rng.uniform(-np.pi, np.pi))
                shift = rng.normal(0.0, 0.45, size=3)
                shift[2] = rng.normal(0.0, 0.20)
                cluster = (cluster - center) @ rotation.T + center + shift
                cluster = np.asarray([wrap_xy(item, parent.cell.array) for item in cluster])
                pd_positions = cluster[:5]
                oxygen_positions = cluster[5:]
            else:
                pd_positions += rng.normal(0.0, 0.22, size=pd_positions.shape)
                if x_value:
                    oxygen_positions += rng.normal(0.0, 0.18, size=oxygen_positions.shape)
                pd_positions = np.asarray([wrap_xy(item, parent.cell.array) for item in pd_positions])
                oxygen_positions = np.asarray([wrap_xy(item, parent.cell.array)
                                               for item in oxygen_positions])
            atoms = make_ordered_atoms(template, pd_positions, oxygen_positions)
            if validate_initial(atoms, x_value)["valid"]:
                return atoms, operation
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            continue
    raise RuntimeError("No valid topology mutation was generated")


def crossover_candidate(first: Atoms, second: Atoms, template: Atoms,
                        rng: np.random.Generator):
    """Cross aligned Pd skeletons and parent oxygen-coordination recipes."""
    x_value = blocks(first)["x"]
    if blocks(second)["x"] != x_value:
        raise ValueError("Crossover requires equal composition")
    first_pd, first_o = extract_cluster(first)
    second_pd, second_o = extract_cluster(second)
    first_center = first_pd.mean(axis=0)
    second_center = second_pd.mean(axis=0)
    best = None
    for permutation in itertools.permutations(range(5)):
        moving = second_pd[list(permutation)] - second_center
        target = first_pd - first_center
        left, _, right = np.linalg.svd(moving.T @ target)
        rotation = left @ right
        if np.linalg.det(rotation) < 0:
            left[:, -1] *= -1
            rotation = left @ right
        aligned = moving @ rotation + first_center
        score = float(np.sum((aligned - first_pd) ** 2))
        if best is None or score < best[0]:
            best = (score, rotation, aligned)
    aligned_pd = best[2]
    builder = AIRSSBuilder(template, x_value, rng)

    def coordination_modes(oxygen_positions, pd_positions):
        result = []
        for oxygen in oxygen_positions:
            degree = sum(mic_distance(oxygen, metal, builder.cell)
                         <= RULES.pdo_bond for metal in pd_positions)
            if degree <= 1:
                result.append("terminal")
            elif degree == 2:
                result.append("bridge")
            else:
                result.append("internal")
        return result

    first_modes = coordination_modes(first_o, first_pd)
    second_modes = coordination_modes(second_o, second_pd)

    for attempt in range(40):
        if attempt < 20:
            pd_positions = first_pd.copy()
        else:
            mixing = rng.uniform(0.15, 0.60)
            pd_positions = ((1.0 - mixing) * first_pd
                            + mixing * aligned_pd)
        if x_value == 0:
            atoms = make_ordered_atoms(template, pd_positions, [])
            if attempt >= 20 and validate_initial(atoms, 0)["valid"]:
                return atoms, "pd_skeleton_crossover"
            continue

        if x_value == 1:
            modes = [second_modes[0] if attempt % 2 == 0 else first_modes[0]]
            first_count = int(attempt % 2 == 1)
        else:
            first_count = int(rng.integers(1, x_value))
            first_indices = rng.choice(x_value, size=first_count, replace=False)
            second_indices = rng.choice(
                x_value, size=x_value - first_count, replace=False)
            modes = [first_modes[int(item)] for item in first_indices]
            modes.extend(second_modes[int(item)] for item in second_indices)
            rng.shuffle(modes)
        accepted = []
        try:
            for mode in modes:
                try:
                    trial, _ = builder.place_oxygen(
                        pd_positions, accepted, preferred_mode=mode)
                except RuntimeError:
                    trial, _ = builder.place_oxygen(pd_positions, accepted)
                accepted.append(trial)
        except RuntimeError:
            continue
        atoms = make_ordered_atoms(template, pd_positions, accepted)
        if validate_initial(atoms, x_value)["valid"]:
            return atoms, (f"coordination_recipe_crossover:first={first_count},"
                           f"second={x_value - first_count}")
    raise RuntimeError("No valid crossover was generated")


def ladder_candidate(parent: Atoms, target_x: int, template: Atoms,
                     rng: np.random.Generator):
    """Add or remove one O while comparing energies only in the target island."""
    source_x = blocks(parent)["x"]
    pd_positions, oxygen_positions = extract_cluster(parent)
    if target_x == source_x + 1:
        builder = AIRSSBuilder(template, target_x, rng)
        trial, mode = builder.place_oxygen(pd_positions, oxygen_positions)
        oxygen_positions = np.vstack((oxygen_positions, trial))
        operation = f"ladder_add_{mode}"
    elif target_x == source_x - 1 and source_x > 0:
        removal_order = list(rng.permutation(source_x))
        for remove in removal_order:
            trial_oxygen = np.delete(oxygen_positions, int(remove), axis=0)
            trial_atoms = make_ordered_atoms(template, pd_positions, trial_oxygen)
            if validate_initial(trial_atoms, target_x)["valid"]:
                return trial_atoms, "ladder_remove"
        raise RuntimeError("No connected ladder-removal product was found")
    else:
        raise ValueError("Ladder mutation requires an adjacent composition")
    atoms = make_ordered_atoms(template, pd_positions, oxygen_positions)
    report = validate_initial(atoms, target_x)
    if not report["valid"]:
        raise RuntimeError("Adjacent-composition mutation produced an invalid candidate")
    return atoms, operation

#!/usr/bin/env python3
"""No-DFT acceptance test for AIRSS and topology operators with ETA."""
from __future__ import annotations

import argparse
from collections import Counter
import time

import numpy as np
from ase.io import read

from chemistry import (
    AIRSSBuilder,
    crossover_candidate,
    ladder_candidate,
    mutate_candidate,
    validate_initial,
)


def duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", default="inputs/POSCAR_CeO2_111_3x3_1TL")
    parser.add_argument("--samples-per-x", type=int, default=100)
    parser.add_argument("--operator-samples", type=int, default=20)
    parser.add_argument("--operator-progress-every", type=int, default=20)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--dft-step-seconds", type=float, default=60.0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--short-nsw", type=int, default=3)
    parser.add_argument("--minimum-queue-margin", type=float, default=2.0)
    args = parser.parse_args()
    template = read(args.template, format="vasp")
    rng = np.random.default_rng(args.seed)
    total = 11 * args.samples_per_x
    completed = 0
    start = time.monotonic()
    operation_counts = Counter()
    parents = {}
    for x_value in range(11):
        parents[x_value] = []
        for _ in range(args.samples_per_x):
            atoms, operations = AIRSSBuilder(template, x_value, rng).build()
            report = validate_initial(atoms, x_value)
            if not report["valid"]:
                raise RuntimeError(f"AIRSS x={x_value}: {report['errors']}")
            operation_counts.update(operations)
            if len(parents[x_value]) < 2:
                parents[x_value].append(atoms)
            completed += 1
            if completed % args.progress_every == 0 or completed == total:
                elapsed = time.monotonic() - start
                eta = elapsed / completed * (total - completed)
                print(f"GENERATION completed={completed}/{total} "
                      f"elapsed={duration(elapsed)} ETA={duration(eta)}", flush=True)

    generation_elapsed = time.monotonic() - start
    generation_rate = total / max(generation_elapsed, 1.0e-12)
    dft_demand = args.workers / max(
        args.short_nsw * args.dft_step_seconds, 1.0e-12)
    queue_margin = generation_rate / dft_demand
    print(f"NON_DFT_THROUGHPUT candidates_per_s={generation_rate:.6f} "
          f"DFT_demand_per_s={dft_demand:.6f} "
          f"queue_margin={queue_margin:.2f}x", flush=True)
    if queue_margin < args.minimum_queue_margin:
        raise RuntimeError(
            f"Candidate generation queue margin {queue_margin:.2f}x is below "
            f"the required {args.minimum_queue_margin:.2f}x")

    operator_total = 22 * args.operator_samples + 20
    operator_completed = 0
    operator_start = time.monotonic()
    print(f"OPERATOR_TEST START total={operator_total}", flush=True)

    def operator_progress():
        nonlocal operator_completed
        operator_completed += 1
        if (operator_completed % args.operator_progress_every == 0
                or operator_completed == operator_total):
            elapsed = time.monotonic() - operator_start
            eta = elapsed / operator_completed * (operator_total - operator_completed)
            print(f"OPERATOR_TEST completed={operator_completed}/{operator_total} "
                  f"elapsed={duration(elapsed)} ETA={duration(eta)}", flush=True)

    for x_value in range(11):
        first = parents[x_value][0]
        second = parents[x_value][-1]
        crossover_successes = 0
        for _ in range(args.operator_samples):
            mutated, _ = mutate_candidate(first, template, rng)
            if not validate_initial(mutated, x_value)["valid"]:
                raise RuntimeError(f"Mutation failed validation for x={x_value}")
            operator_progress()
            try:
                crossed, _ = crossover_candidate(first, second, template, rng)
            except RuntimeError:
                operation_counts["crossover_rejected"] += 1
            else:
                if not validate_initial(crossed, x_value)["valid"]:
                    raise RuntimeError(f"Crossover failed validation for x={x_value}")
                crossover_successes += 1
                operation_counts["crossover_accepted"] += 1
            operator_progress()
        if crossover_successes == 0:
            raise RuntimeError(f"No valid crossover was produced for x={x_value}")
        if x_value < 10:
            added, _ = ladder_candidate(first, x_value + 1, template, rng)
            if not validate_initial(added, x_value + 1)["valid"]:
                raise RuntimeError(f"Ladder add failed for x={x_value}")
            operator_progress()
        if x_value > 0:
            removed, _ = ladder_candidate(first, x_value - 1, template, rng)
            if not validate_initial(removed, x_value - 1)["valid"]:
                raise RuntimeError(f"Ladder remove failed for x={x_value}")
            operator_progress()
    print(f"GENERATION ACCEPTANCE PASSED: {total} AIRSS candidates", flush=True)
    print("operation_counts=" + ",".join(
        f"{key}:{operation_counts[key]}" for key in sorted(operation_counts)), flush=True)


if __name__ == "__main__":
    main()

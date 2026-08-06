# Asynchronous topology-guided random and genetic structure search with staged DFT relaxation.

AsyncTopoAIRSS is a lightweight global-structure-search workflow for atomistic systems with expensive first-principles energies. It combines chemically constrained random generation, topology-aware mutation and crossover, asynchronous file-queue scheduling, and staged DFT relaxation. Candidates are screened through short, intermediate, and full relaxations, while low-energy structurally distinct minima are retained for subsequent analysis.

This repository implements a file-queue-based global search for `Pd5Ox`
(`x = 0...10`) clusters supported on a one-trilayer `3 x 3` CeO2(111) slab.
It combines chemically constrained random structure generation (AIRSS),
topology-preserving mutations, crossover, and adjacent-composition moves with
short, intermediate, and full VASP relaxations.

The code is intended for fixed-composition structural searches. It does not
assign thermodynamic weights across oxygen chemical potentials and does not
prove a mathematical global minimum. Search completeness should be assessed
from the number of converged structures, the energy history, and independent
restarts where required.

## Model and constraints

- Fluorite lattice constant: `a0 = 5.456 A`.
- Support: one O-Ce-O CeO2(111) trilayer in a `3 x 3` surface cell.
- Fixed atoms: bottom O layer (9 atoms) and all Ce atoms (9 atoms).
- Mobile atoms: top O layer and the Pd5Ox cluster.
- Atom order: `O(18+x) Pd5 Ce9` in all generated and exported POSCAR files.
- The provided template is validated against the ideal fluorite nearest-neighbour
  geometry before a calculation is submitted.

The search generator rejects initial O-O contacts below 2.00 A, requires each
added oxygen to be coordinated to Pd, and enforces a connected Pd5Ox graph.
These are generation rules, not constraints on the final DFT-relaxed product.

## Requirements

Python 3.9 or later with ASE, NumPy, and SciPy is required. VASP, a valid VASP
license, and O/Pd/Ce PAW datasets are required for electronic-structure runs;
neither VASP nor `POTCAR` is distributed here.

```sh
conda env create -f environment.yml
conda activate pd5ox-ceo2-search
python -u tests/run_offline_checks.py
```

The checks are offline. They verify the slab template and seeded structure
generator, but do not execute VASP. `pytest` is optional and enables the same
checks through `python -m pytest -q`.

Before creating a source archive, refresh and verify the checksums:

```sh
python -u scripts/write_checksums.py
sha256sum -c SHA256SUMS
```

## Site configuration

Set the paths to the *immutable* VASP input files and executable in the shell.
The workflow copies these files into task directories. It changes only `NSW`
and `NELM` in each copied INCAR and records hashes before and after every task.

```sh
export INCAR=/path/to/master/INCAR
export KPOINTS=/path/to/master/KPOINTS
export POTCAR=/path/to/licensed/POTCAR
export VASP_EXEC=/path/to/vasp_gam
export PYTHON=$(command -v python)
```

If a cluster requires modules or a scheduler-specific environment, place those
commands in a small shell file and set `SITE_ENV` to its absolute path:

```sh
export SITE_ENV=$PWD/slurm/site.env
```

`slurm/site.env.example` is a template. It is not sourced unless `SITE_ENV` is
explicitly set. Choose the partition, account, time limit, and MPI launcher for
your cluster by editing a local copy of `slurm/worker_array.sbatch` or by
passing site-specific `sbatch` options through your local submission wrapper.

## Preflight and generator checks

```sh
python -u scripts/preflight.py \
  --template inputs/POSCAR_CeO2_111_3x3_1TL \
  --incar "$INCAR" --kpoints "$KPOINTS" --potcar "$POTCAR"

python -u scripts/test_generation.py \
  --samples-per-x 100 --operator-samples 10 --progress-every 100
```

The first command verifies the structural and VASP-input contracts. The second
reports the candidate generation rate and tests AIRSS, mutation, crossover,
and composition-ladder operators without running VASP.

## Smoke test

Run the two-worker smoke test before a production search:

```sh
NTASKS=32 sh slurm/submit_smoke.sh
watch -n 15 'PYTHON=python3 sh slurm/monitor.sh runs/smoke_001'
python -u scripts/verify_smoke.py runs/smoke_001
```

Use benchmark smoke runs to select a rank count rather than assuming that more
ranks are faster:

```sh
NTASKS=16 RUN_ROOT=$PWD/runs/smoke_n16 sh slurm/submit_smoke.sh
NTASKS=32 RUN_ROOT=$PWD/runs/smoke_n32 sh slurm/submit_smoke.sh
python -u scripts/summarize_timing.py runs/smoke_n16
python -u scripts/summarize_timing.py runs/smoke_n32
```

## Production run

The default configuration uses five rounds per composition. Each round is
`24 SHORT -> 8 MEDIUM -> 4 FULL` candidates. `SHORT`, `MEDIUM`, and `FULL`
use 3, 5, and up to 80 ionic steps, respectively. A FULL relaxation can be
continued twice if the force threshold has not been reached.

```sh
NTASKS=32 WORKERS=6 RUN_ROOT=$PWD/runs/production_001 \
  sh slurm/submit_all.sh
```

Each array element receives `NTASKS` MPI ranks. Thus `WORKERS=6` and
`NTASKS=32` requests six independent 32-rank VASP jobs, rather than one
192-rank job. All submitted worker processes emit a 30 s progress line based
on OSZICAR while VASP is running.

The run directory contains the queue, candidate metadata, VASP task folders,
and a new `provenance/submission_*.json` record for every submission. The
record preserves hashes of the configuration, template, INCAR, KPOINTS, and
POTCAR, as well as the Git revision when the repository is a Git checkout.

## Monitoring, recovery, and export

```sh
python -u scripts/monitor.py runs/production_001
touch runs/production_001/STOP                     # request a safe stop
python -u scripts/recover_running.py runs/production_001 \
  --confirm-workers-stopped                         # only after workers stop
```

Resuming retains all previous state and never overwrites task directories:

```sh
rm -f runs/production_001/STOP runs/production_001/STOP_WORKERS
ALLOW_RESUME=1 RUN_ROOT=$PWD/runs/production_001 sh slurm/submit_all.sh
```

After `SEARCH_COMPLETE` is present, distinguish two counts:

- `assess_convergence.py` counts all force-converged FULL relaxations.
- `validate_search.py` counts topology/fingerprint-deduplicated archive members.

For example, a practical count-and-plateau criterion is:

```sh
python -u scripts/assess_convergence.py runs/production_001 \
  --minimum-full 30 --recent 10 --stability-eV 0.10 --ignore-reproduction

python -u scripts/validate_search.py runs/production_001 \
  --minimum-full 10 --require-complete-marker

python -u scripts/export_final.py runs/production_001 \
  --output runs/production_001/final_candidates \
  --count 10 --fingerprint-rms 0.35
```

The export contains `POSCAR_rankNN` files for each composition and a
`summary.tsv` containing energy, force, topology, lineage, and structural
distance metadata.

## Reproducibility notes

Reproducibility requires the same VASP executable version, PAW datasets,
INCAR/KPOINTS, hardware/MPI stack, random seed, and search configuration. The
repository records the input hashes but cannot redistribute licensed PAW data.
Finite stochastic searches can reproduce a low-energy family without yielding
identical candidate order or identical floating-point energies on different
platforms. Report the search budget and convergence criterion with any
published structure set.

## References

1. C. J. Pickard and R. J. Needs, *J. Phys.: Condens. Matter* **23**, 053201
   (2011). https://doi.org/10.1088/0953-8984/23/5/053201
2. R. L. Johnston, *Dalton Trans.* 4193-4207 (2003).
   https://doi.org/10.1039/B305333P
3. G. Kresse and J. Furthmuller, *Phys. Rev. B* **54**, 11169 (1996).
   https://doi.org/10.1103/PhysRevB.54.11169

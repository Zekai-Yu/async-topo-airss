# Reproducibility record

For each reported search, archive the repository commit, the configuration
file, the generated `provenance/submission_*.json` files, and the final
`summary.tsv`. Keep licensed VASP inputs in the local archive permitted by
your VASP licence; publish their SHA256 values and `TITEL` lines instead of
publishing `POTCAR` itself.

Record the following alongside each run:

- VASP version, compiler, MPI implementation, and MPI launcher;
- `INCAR`, `KPOINTS`, and `POTCAR` hashes;
- Slurm resource request and module environment;
- random seed and complete search configuration;
- number of SHORT, MEDIUM, FULL, and force-converged FULL evaluations;
- the stopping criterion and the archive de-duplication threshold.

The workflow uses deterministic pseudorandom sequences for a fixed NumPy
version and saved controller state. Exact task order can still vary when
workers complete asynchronously. For a repeated production run, compare the
energy range and structural families, rather than requiring byte-identical
trajectory order.

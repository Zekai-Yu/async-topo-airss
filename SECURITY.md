# Security and data handling

Do not commit `POTCAR`, VASP binaries, queue-account strings, host names, or
calculation directories. `POTCAR` is licensed material and must be supplied by
each user from an authorised VASP installation.

The queue is implemented with atomic file renames. Use a filesystem that
supports atomic rename within a directory. Before recovering `queue/running`
tasks, verify that all workers from the interrupted submission have stopped.

Please report suspected data-loss or queue-corruption issues privately to the
repository maintainers before opening a public issue.

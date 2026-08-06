#!/usr/bin/env python3
"""Run the repository's offline checks without requiring pytest."""
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def load_checks():
    target = ROOT / "tests" / "test_generation.py"
    specification = importlib.util.spec_from_file_location("offline_generation_tests", target)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Cannot load {target}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def main():
    module = load_checks()
    checks = (module.test_template_contract,
              module.test_seeded_airss_candidates_obey_contract)
    for check in checks:
        check()
        print(f"PASSED {check.__name__}", flush=True)
    print(f"OFFLINE CHECKS PASSED ({len(checks)} checks)", flush=True)


if __name__ == "__main__":
    main()

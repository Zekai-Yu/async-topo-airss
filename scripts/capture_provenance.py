#!/usr/bin/env python3
"""Write a compact provenance record before a smoke or production submission."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--template", required=True)
    parser.add_argument("--incar", required=True)
    parser.add_argument("--kpoints", required=True)
    parser.add_argument("--potcar", required=True)
    parser.add_argument("--vasp-exec", default=None)
    args = parser.parse_args()

    tracked = {
        "config": Path(args.config), "template": Path(args.template),
        "INCAR": Path(args.incar), "KPOINTS": Path(args.kpoints),
        "POTCAR": Path(args.potcar),
    }
    for label, path in tracked.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    root = Path(args.run_root).resolve()
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    destination = root / "provenance" / f"submission_{stamp}.json"
    executable = Path(args.vasp_exec) if args.vasp_exec else None
    if executable is not None and not executable.is_file():
        raise FileNotFoundError(f"Missing VASP executable: {executable}")
    payload = {
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "status", "--porcelain"]),
        "files": {label: {"path": str(path.resolve()), "sha256": sha256(path)}
                  for label, path in tracked.items()},
    }
    if executable is not None:
        payload["vasp_executable"] = {
            "path": str(executable.resolve()), "sha256": sha256(executable),
            "version": command_output([str(executable.resolve()), "--version"]),
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(f"Wrote provenance: {destination}")


if __name__ == "__main__":
    main()

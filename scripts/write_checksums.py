#!/usr/bin/env python3
"""Write SHA256SUMS for tracked source files in a source distribution."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


SKIP_PARTS = {".git", ".github", "__pycache__", "runs", "logs"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", default="SHA256SUMS")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = root / args.output
    files = []
    for path in root.rglob("*"):
        if not path.is_file() or path == output or SKIP_PARTS.intersection(path.parts):
            continue
        files.append(path)
    lines = [f"{digest(path)}  {path.relative_to(root).as_posix()}" for path in sorted(files)]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} checksums to {output}")


if __name__ == "__main__":
    main()

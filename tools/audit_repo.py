#!/usr/bin/env python3
"""Audit source-control hygiene: fail if local assets or pipeline outputs got tracked."""

from __future__ import annotations

import argparse
import fnmatch
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# Assets (weights, datasets) and pipeline OUTPUTS (data/derived/) are local-only.
# Benchmarking lives on the `benchmark` branch, not here. See docs/shared-data.md.
FORBIDDEN_TRACKED_PATTERNS = [
    "models/*/weights/*",
    "models/*/checksums/*",
    "data/raw/*",
    "data/derived/*",
    "drive/dataset/*",
    "checkpoints/*",
    "__pycache__/*",
    "*.pyc",
    "*.mp4",
]

ALLOWED_TRACKED_PATTERNS = [
    "models/*/weights/.gitkeep",
    "models/*/checksums/.gitkeep",
    "data/raw/.gitkeep",
    "data/derived/.gitkeep",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fail", action="store_true", help="Exit non-zero when tracked hygiene violations exist")
    return parser.parse_args()


def git_ls_files() -> list[str]:
    completed = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return [line for line in completed.stdout.splitlines() if line.strip()]


def matches(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def violations(paths: list[str]) -> list[str]:
    bad = []
    for path in paths:
        if matches(path, ALLOWED_TRACKED_PATTERNS):
            continue
        if matches(path, FORBIDDEN_TRACKED_PATTERNS):
            bad.append(path)
    return sorted(bad)


def main() -> int:
    args = parse_args()
    bad = violations(git_ls_files())
    if bad:
        print("Tracked generated/local artifacts:")
        for path in bad:
            print(f"  {path}")
    else:
        print("Repository hygiene audit passed")
    return 1 if args.fail and bad else 0


if __name__ == "__main__":
    raise SystemExit(main())

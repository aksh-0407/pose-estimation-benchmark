#!/usr/bin/env python3
"""Fail if committed visual QA folders exceed the per-camera image quota."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", help="Benchmark run directories to check.")
    parser.add_argument("--max-per-camera", type=int, default=5)
    return parser.parse_args()


def image_count(path: Path) -> int:
    return sum(1 for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS)


def main() -> int:
    args = parse_args()
    if args.max_per_camera < 0:
        raise SystemExit("--max-per-camera must be >= 0")

    violations: list[str] = []
    checked = 0
    for run_dir_value in args.run_dirs:
        run_dir = Path(run_dir_value)
        visual_root = run_dir / "visualizations"
        if not visual_root.exists():
            continue
        for camera_dir in sorted(path for path in visual_root.glob("*/*/cam_*") if path.is_dir()):
            count = image_count(camera_dir)
            checked += 1
            if count > args.max_per_camera:
                violations.append(f"{camera_dir}: {count} images > {args.max_per_camera}")

    if violations:
        print("Visualization quota exceeded:", file=sys.stderr)
        for violation in violations:
            print(f"  {violation}", file=sys.stderr)
        return 1
    print(f"Visualization quota OK: checked {checked} camera folders, max_per_camera={args.max_per_camera}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

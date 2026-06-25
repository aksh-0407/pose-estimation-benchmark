#!/usr/bin/env python3
"""Rewrite machine-specific absolute paths in run artifacts to repo-root-relative.

Older Phase 1 runs baked the absolute repo path (e.g. ``/home/aksh/quidich/
Pose_Estimation/...``) into manifests, metrics, and prediction JSONL records. The
inference scripts now emit repo-root-relative paths; this one-off migration brings
existing ``benchmarks/runs/`` outputs onto the same scheme **without re-running
inference**.

It simply strips the repo-root prefix (``<ROOT>/``) from every JSON/JSONL value
under the given run dirs. Idempotent and safe to re-run.

    python scripts/setup/relativize_run_paths.py                 # all runs
    python scripts/setup/relativize_run_paths.py --runs-dir benchmarks/runs
    python scripts/setup/relativize_run_paths.py --dry-run
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", default="benchmarks/runs", help="Directory of run folders (default: benchmarks/runs)")
    parser.add_argument("--root", default=str(ROOT), help="Repo root prefix to strip (default: detected repo root)")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing")
    return parser.parse_args()


def iter_target_files(runs_dir: Path):
    yield from runs_dir.glob("*/run_manifest.json")
    yield from runs_dir.glob("*/p1_metrics.json")
    yield from runs_dir.glob("*/delivery_metrics/*/run_manifest.json")
    yield from runs_dir.glob("*/delivery_metrics/*/p1_metrics.json")
    yield from runs_dir.glob("*/predictions/*.jsonl")
    yield from runs_dir.glob("*/visualizations/visual_qa_manifest.json")
    yield from runs_dir.glob("*/visualizations/videos/video_manifest.json")


def main() -> int:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_absolute():
        runs_dir = ROOT / runs_dir
    # Strip both "<root>/" and a bare "<root>" occurrence; the trailing slash form
    # is what produces clean relative paths.
    root_prefix = str(Path(args.root).resolve()).rstrip("/") + "/"

    changed = 0
    scanned = 0
    for path in sorted(iter_target_files(runs_dir)):
        scanned += 1
        text = path.read_text(encoding="utf-8")
        if root_prefix not in text:
            continue
        new_text = text.replace(root_prefix, "")
        changed += 1
        print(f"{'would update' if args.dry_run else 'updated'}: {path.relative_to(ROOT)}")
        if not args.dry_run:
            path.write_text(new_text, encoding="utf-8")

    print(f"\nScanned {scanned} file(s); {changed} contained the absolute prefix.")
    if args.dry_run and changed:
        print("Re-run without --dry-run to apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

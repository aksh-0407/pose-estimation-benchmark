#!/usr/bin/env python3
"""Check benchmark rows against the project acceptance thresholds."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def parse_float(value: str) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/manual_benchmark_matrix.csv")
    parser.add_argument("--latency-ms", type=float, default=200.0)
    parser.add_argument("--reprojection-px", type=float, default=10.0)
    parser.add_argument("--mpjpe-mm", type=float, default=25.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    failures = []
    with Path(args.input).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            model_id = row.get("model_id", "")
            run_id = row.get("run_id", "")
            latency = parse_float(row.get("latency_p95_ms", ""))
            reprojection = parse_float(row.get("mean_reprojection_error_px", ""))
            mpjpe = parse_float(row.get("mpjpe_mm", ""))
            if latency is not None and latency > args.latency_ms:
                failures.append(f"{model_id}/{run_id}: p95 latency {latency} ms > {args.latency_ms} ms")
            if reprojection is not None and reprojection > args.reprojection_px:
                failures.append(f"{model_id}/{run_id}: reprojection {reprojection} px > {args.reprojection_px} px")
            if mpjpe is not None and mpjpe > args.mpjpe_mm:
                failures.append(f"{model_id}/{run_id}: MPJPE {mpjpe} mm > {args.mpjpe_mm} mm")

    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1
    print("All populated benchmark thresholds pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Rank benchmark rows using the project selection weights."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pose_estimation.metrics import weighted_model_score


def parse_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/manual_benchmark_matrix.csv")
    parser.add_argument("--output", default="results/model_ranking.csv")
    parser.add_argument("--latency-budget-ms", type=float, default=200.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    rows = []
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row.get("model_id"):
                continue
            score = weighted_model_score(
                cricket_2d_accuracy=parse_float(row.get("cricket_2d_score", "")),
                occlusion_robustness=parse_float(row.get("occlusion_score", "")),
                latency_p95_ms=parse_float(row.get("latency_p95_ms", ""), default=args.latency_budget_ms),
                jitter_score=parse_float(row.get("jitter_score", "")),
                integration_effort=parse_float(row.get("integration_effort_score", "")),
                latency_budget_ms=args.latency_budget_ms,
            )
            row["weighted_score"] = f"{score:.4f}"
            rows.append(row)

    rows.sort(key=lambda item: parse_float(item.get("weighted_score", "")), reverse=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["model_id", "weighted_score"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {output_path} with {len(rows)} ranked rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

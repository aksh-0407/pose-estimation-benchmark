#!/usr/bin/env python3
"""Create reproducible benchmark manifests and optional placeholder rows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pose_estimation.registry import get_model, list_models
from pose_estimation.results_io import append_result, collect_environment, utc_now, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=None, help="Model IDs from configs/model_registry.yaml")
    parser.add_argument("--registry", default="configs/model_registry.yaml")
    parser.add_argument("--results", default="results/manual_benchmark_matrix.csv")
    parser.add_argument("--runs-dir", default="benchmarks/runs")
    parser.add_argument("--dataset", default="TBD")
    parser.add_argument("--split", default="TBD")
    parser.add_argument("--backend", default="tensorrt_fp16")
    parser.add_argument("--hardware", default="nvidia_gpu")
    parser.add_argument("--dry-run", action="store_true", help="Only write manifests and placeholder rows")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected_ids = args.models or [model["id"] for model in list_models(args.registry)]
    run_id = utc_now().replace(":", "").replace("+", "_").replace(".", "_")
    command = " ".join(sys.argv)
    environment = collect_environment(command=command, workdir=ROOT)

    for model_id in selected_ids:
        model = get_model(model_id, args.registry)
        manifest = {
            "run_id": run_id,
            "dry_run": bool(args.dry_run),
            "model": model,
            "dataset": args.dataset,
            "split": args.split,
            "backend": args.backend,
            "hardware": args.hardware,
            "environment": environment,
            "next_action": "Integrate the framework-specific runner and replace placeholder metrics.",
        }
        manifest_path = Path(args.runs_dir) / run_id / f"{model_id}_manifest.json"
        write_manifest(manifest_path, manifest)

        if args.dry_run:
            append_result(
                args.results,
                {
                    "model_id": model_id,
                    "run_id": run_id,
                    "benchmark_type": "dry_run_manifest",
                    "dataset": args.dataset,
                    "split": args.split,
                    "skeleton": model.get("skeleton", ""),
                    "num_keypoints": 133 if "133" in str(model.get("skeleton", "")) else "",
                    "precision": "fp16" if "fp16" in args.backend else "",
                    "backend": args.backend,
                    "hardware": args.hardware,
                    "batch_size": 1,
                    "input_size": model.get("input_size", ""),
                    "git_sha": environment.get("git_sha") or "",
                    "command": command,
                    "notes": f"Dry-run manifest written to {manifest_path}",
                    "created_at": environment["created_at"],
                },
            )
        print(f"Wrote {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

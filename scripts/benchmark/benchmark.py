#!/usr/bin/env python3
"""Master CLI for preparing, smoking, running, aggregating, and reporting benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pose_estimation.datasets import CocoKeypointDataset
from pose_estimation.hardware import collect_hardware_report, collect_software_report
from pose_estimation.registry import get_model, list_models
from pose_estimation.results_io import file_sha256, utc_now, write_manifest
from pose_estimation.visualization import write_html_report

RUNS_DIR = ROOT / "benchmarks" / "runs"
ARTIFACTS_DIR = ROOT / "benchmarks" / "artifacts"
REPORTS_DIR = ROOT / "benchmarks" / "reports"
AGGREGATE_CSV = ROOT / "results" / "aggregate_metrics.csv"
DEFAULT_MODEL_ENVS = ROOT / "configs" / "model_envs.yaml"
DEFAULT_DATASETS = ROOT / "configs" / "datasets.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Prepare model assets and datasets")
    prepare.add_argument("--models", nargs="+", default=["all"])
    prepare.add_argument("--datasets", nargs="+", default=[])
    prepare.add_argument("--download-large-assets", action="store_true")
    prepare.add_argument("--remove-archives", action="store_true")
    prepare.add_argument("--build-openpose", action="store_true", help="Clone/configure/build official CMU OpenPose")
    prepare.add_argument("--dry-run", action="store_true")

    smoke = subparsers.add_parser("smoke", help="Run model smoke checks")
    smoke.add_argument("--models", nargs="+", default=["all"])
    smoke.add_argument("--device", default="cuda:0")
    smoke.add_argument("--allow-heavy", action="store_true")

    run = subparsers.add_parser("run", help="Create an immutable benchmark run")
    run.add_argument("--models", nargs="+", required=True)
    run.add_argument("--datasets", nargs="+", required=True)
    run.add_argument("--run-id", default=None)
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--start-index", type=int, default=0)
    run.add_argument("--batch-size", type=int, default=8)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--imgsz", type=int, default=640)
    run.add_argument("--conf", type=float, default=0.001)
    run.add_argument("--iou", type=float, default=0.7)
    run.add_argument("--precision", default="fp32")
    run.add_argument("--backend", default="framework_native")
    run.add_argument("--no-resume", dest="resume", action="store_false")
    run.set_defaults(resume=True)
    run.add_argument("--dry-run", action="store_true")

    aggregate = subparsers.add_parser("aggregate", help="Aggregate immutable metric files")
    aggregate.add_argument("--runs-dir", default=str(RUNS_DIR))
    aggregate.add_argument("--output", default=str(AGGREGATE_CSV))

    report = subparsers.add_parser("report", help="Generate static HTML report from aggregate CSV")
    report.add_argument("--aggregate", default=str(AGGREGATE_CSV))
    report.add_argument("--output", default=None)
    return parser.parse_args()


def resolve_models(models: list[str]) -> list[str]:
    if models == ["all"]:
        return [model["id"] for model in list_models()]
    return models


def run_command(command: list[str], *, dry_run: bool = False) -> int:
    print("+ " + " ".join(shlex.quote(part) for part in command))
    if dry_run:
        return 0
    cache_dir = Path("/tmp") / "pose_benchmark_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "MPLCONFIGDIR": str(cache_dir / "matplotlib"),
        "XDG_CACHE_HOME": str(cache_dir / "xdg"),
        "PYTHONNOUSERSITE": "1",
    }
    return subprocess.run(command, cwd=ROOT, check=False, env=env).returncode


def command_prepare(args: argparse.Namespace) -> int:
    failures = 0
    selected_models = resolve_models(args.models)
    if selected_models:
        command = [sys.executable, "scripts/setup_model_envs.py", "--models", *selected_models, "--download-assets"]
        if args.download_large_assets:
            command.append("--download-large-assets")
        if args.dry_run:
            command.append("--dry-run")
        failures += int(run_command(command, dry_run=False) != 0)
        if args.build_openpose and "openpose_body25" in selected_models:
            command = [sys.executable, "scripts/setup_openpose.py"]
            if args.dry_run:
                command.append("--dry-run")
            failures += int(run_command(command, dry_run=False) != 0)

    for dataset_id in args.datasets:
        if dataset_id == "coco17_val2017":
            command = [sys.executable, "scripts/download_coco_keypoints.py"]
            if args.remove_archives:
                command.append("--remove-archives")
            failures += int(run_command(command, dry_run=args.dry_run) != 0)
        else:
            print(f"{dataset_id}: dataset preparation is registered but not automated yet")
    return 1 if failures else 0


def command_smoke(args: argparse.Namespace) -> int:
    selected = resolve_models(args.models)
    command = [sys.executable, "scripts/smoke_model_envs.py", "--models", *selected, "--device", args.device]
    if args.allow_heavy:
        command.append("--allow-heavy")
    return run_command(command)


def command_run(args: argparse.Namespace) -> int:
    selected_models = resolve_models(args.models)
    datasets = args.datasets
    run_id = args.run_id or _run_id(
        models=selected_models,
        datasets=datasets,
        limit=args.limit,
        start_index=args.start_index,
    )
    run_dir = RUNS_DIR / run_id
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (run_dir / "visualizations").mkdir(exist_ok=True)

    write_manifest(run_dir / "hardware.json", collect_hardware_report())
    write_manifest(run_dir / "software.json", collect_software_report(ROOT))

    run_manifest = {
        "schema_version": "pose_benchmark_run/v1",
        "run_id": run_id,
        "created_at": utc_now(),
        "command": " ".join(sys.argv),
        "models": selected_models,
        "datasets": datasets,
        "precision": args.precision,
        "backend": args.backend,
        "limit": args.limit,
        "dry_run": args.dry_run,
    }
    write_manifest(run_dir / "run_manifest.json", run_manifest)

    dataset_manifests = {}
    for dataset_id in datasets:
        dataset_manifest = dataset_status(dataset_id)
        dataset_manifests[dataset_id] = dataset_manifest
    write_manifest(run_dir / "dataset_manifest.json", dataset_manifests)

    model_manifests = {}
    metric_rows = []
    failures = 0
    for model_id in selected_models:
        model = get_model(model_id)
        model_manifests[model_id] = model
        for dataset_id in datasets:
            metric_path = run_dir / "metrics" / f"{model_id}__{dataset_id}.json"
            if can_run_native_benchmark(model_id, dataset_id) and not args.dry_run:
                returncode = run_native_benchmark(run_id, run_dir, model_id, dataset_id, args)
                if returncode != 0:
                    failures += 1
                    metrics = build_pending_metrics(run_id, model, dataset_manifests[dataset_id], args)
                    metrics["status"] = "runner_failed"
                    write_manifest(metric_path, metrics)
                else:
                    with metric_path.open("r", encoding="utf-8") as handle:
                        metrics = json.load(handle)
            else:
                metrics = build_pending_metrics(run_id, model, dataset_manifests[dataset_id], args)
                write_manifest(metric_path, metrics)
            metric_rows.append(flatten_metrics(metrics))
    write_manifest(run_dir / "model_manifest.json", model_manifests)
    write_html_report(run_dir / "visualizations" / "summary.html", metric_rows, title=f"Benchmark Run {run_id}")
    print(f"Wrote immutable benchmark run: {run_dir}")
    return 1 if failures else 0


# Datasets that have an automated runner today. A model is benchmark-ready on one of
# these when its config entry declares a `benchmark_runner` (see configs/model_envs.yaml).
# Registering an adapter is therefore a per-model config edit, not a shared-code edit,
# so contributors adding different adapters do not conflict. Adding a brand-new runner
# *kind* (rare) is the only change that touches this map.
BENCHMARK_DATASETS = {"coco17_val2017"}
RUNNER_SCRIPTS = {
    "yolo": "scripts/run_yolo_coco_benchmark.py",  # end-to-end detector+pose
    "mmpose_topdown": "scripts/run_mmpose_coco_benchmark.py",  # GT-bbox top-down
}


def native_runner(model_id: str, dataset_id: str) -> str | None:
    """Runner kind for a (model, dataset), or None if it has no full adapter."""
    if dataset_id not in BENCHMARK_DATASETS:
        return None
    model = load_yaml(DEFAULT_MODEL_ENVS).get("models", {}).get(model_id, {})
    runner = model.get("benchmark_runner")
    return runner if runner in RUNNER_SCRIPTS else None


def can_run_native_benchmark(model_id: str, dataset_id: str) -> bool:
    return native_runner(model_id, dataset_id) is not None


def run_native_benchmark(
    run_id: str,
    run_dir: Path,
    model_id: str,
    dataset_id: str,
    args: argparse.Namespace,
) -> int:
    env_config = load_yaml(DEFAULT_MODEL_ENVS)
    env_name = env_config["models"][model_id]["env_name"]
    conda = env_config.get("defaults", {}).get("conda_executable", "conda")
    runner = native_runner(model_id, dataset_id)
    runner_script = RUNNER_SCRIPTS[runner]
    command = [
        conda,
        "run",
        "-n",
        env_name,
        "python",
        runner_script,
        "--run-id",
        run_id,
        "--run-dir",
        str(run_dir),
        "--artifact-dir",
        str(ARTIFACTS_DIR / run_id),
        "--model-id",
        model_id,
        "--dataset-id",
        dataset_id,
        "--model-config",
        str(DEFAULT_MODEL_ENVS),
        "--dataset-config",
        str(DEFAULT_DATASETS),
        "--registry",
        "configs/model_registry.yaml",
        "--start-index",
        str(args.start_index),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
    ]
    if runner == "yolo":
        command += ["--imgsz", str(args.imgsz), "--conf", str(args.conf), "--iou", str(args.iou)]
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if not args.resume:
        command.append("--no-resume")
    return run_command(command)


def dataset_status(dataset_id: str) -> dict[str, Any]:
    if dataset_id == "coco17_val2017":
        dataset = CocoKeypointDataset(dataset_id)
        manifest = dataset.manifest()
        annotation = Path(manifest["annotation_file"])
        manifest["dataset_hash"] = file_sha256(annotation) if annotation.exists() else ""
        return manifest
    return {
        "dataset_id": dataset_id,
        "ready": False,
        "missing": ["dataset loader not implemented"],
    }


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_pending_metrics(run_id: str, model: dict[str, Any], dataset: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    status = "dry_run" if args.dry_run else "adapter_pending"
    if not dataset.get("ready"):
        status = "dataset_missing"
    return {
        "schema_version": "pose_metrics/v1",
        "run_id": run_id,
        "model_id": model["id"],
        "model_name": model.get("name"),
        "framework": model.get("framework"),
        "source_skeleton": model.get("skeleton"),
        "dataset_id": dataset.get("dataset_id", dataset.get("name", "")),
        "target_skeleton": dataset.get("target_skeleton", "coco_17"),
        "status": status,
        "precision": args.precision,
        "backend": args.backend,
        "limit": args.limit,
        "metrics": {
            "coco_oks_ap": None,
            "coco_oks_ar": None,
            "pck@0.05": None,
            "pck@0.10": None,
            "pck@0.20": None,
            "mean_pixel_error": None,
            "detection_rate": None,
            "missing_keypoint_rate": None,
            "latency_p50_ms": None,
            "latency_p90_ms": None,
            "latency_p95_ms": None,
            "latency_p99_ms": None,
            "fps_per_camera": None,
            "gpu_memory_peak_mb": None,
            "cpu_memory_peak_mb": None,
            "mpjpe_mm": None,
            "pa_mpjpe_mm": None,
            "pck3d": None,
            "acceleration_error": None,
            "triangulation_success_rate": None,
        },
        "dataset": dataset,
    }


def command_aggregate(args: argparse.Namespace) -> int:
    rows = []
    for metrics_path in sorted(Path(args.runs_dir).glob("*/metrics/*.json")):
        with metrics_path.open("r", encoding="utf-8") as handle:
            rows.append(flatten_metrics(json.load(handle)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row}) if rows else ["status"]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output} with {len(rows)} rows")
    return 0


def command_report(args: argparse.Namespace) -> int:
    aggregate = Path(args.aggregate)
    rows = []
    if aggregate.exists():
        with aggregate.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    output = Path(args.output) if args.output else REPORTS_DIR / "aggregate" / "index.html"
    write_html_report(output, rows, title="Pose Benchmark Aggregate Report")
    print(f"Wrote {output}")
    return 0


def flatten_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    row = {
        "run_id": metrics.get("run_id", ""),
        "model_id": metrics.get("model_id", ""),
        "model_name": metrics.get("model_name", ""),
        "framework": metrics.get("framework", ""),
        "dataset_id": metrics.get("dataset_id", ""),
        "source_skeleton": metrics.get("source_skeleton", ""),
        "target_skeleton": metrics.get("target_skeleton", ""),
        "status": metrics.get("status", ""),
        "precision": metrics.get("precision", ""),
        "backend": metrics.get("backend", ""),
    }
    for key, value in metrics.get("metrics", {}).items():
        row[key] = "" if value is None else value
    return row


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return slug or "unknown"


def _sequence_slug(values: list[str] | None) -> str:
    values = values or []
    if not values:
        return "none"
    if len(values) <= 3:
        return "+".join(_safe_slug(value) for value in values)
    head = "+".join(_safe_slug(value) for value in values[:3])
    return f"{head}+{len(values) - 3}more"


def _timestamp_slug() -> str:
    timestamp = utc_now().split("+", 1)[0].split(".", 1)[0]
    return timestamp.replace("-", "").replace(":", "") + "Z"


def _run_id(
    *,
    models: list[str] | None = None,
    datasets: list[str] | None = None,
    limit: int | None = None,
    start_index: int = 0,
    prefix: str = "run",
) -> str:
    if not models and not datasets:
        return f"{_safe_slug(prefix)}__{_timestamp_slug()}"
    scope = "full" if limit is None else f"n{limit}"
    if start_index:
        scope = f"start{start_index}-{scope}"
    return (
        f"models-{_sequence_slug(models)}"
        f"__bench-{_sequence_slug(datasets)}"
        f"__scope-{_safe_slug(scope)}"
        f"__{_timestamp_slug()}"
    )


def main() -> int:
    args = parse_args()
    if args.command == "prepare":
        return command_prepare(args)
    if args.command == "smoke":
        return command_smoke(args)
    if args.command == "run":
        return command_run(args)
    if args.command == "aggregate":
        return command_aggregate(args)
    if args.command == "report":
        return command_report(args)
    raise RuntimeError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

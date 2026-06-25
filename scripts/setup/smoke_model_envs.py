#!/usr/bin/env python3
"""Run smoke checks through each model's Conda environment and log results."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pose_estimation.registry import get_model
from pose_estimation.results_io import append_result, collect_environment, utc_now, write_manifest

DEFAULT_CONFIG = ROOT / "configs" / "model_envs.yaml"
PASSING_SMOKE_STATUSES = {"ok", "ready_heavy_skipped", "ready_runtime_limited"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--registry", default="configs/model_registry.yaml")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--results", default="results/smoke_results.csv")
    parser.add_argument("--runs-dir", default="benchmarks/runs")
    parser.add_argument("--dataset", default="smoke_image")
    parser.add_argument("--split", default="local")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-heavy", action="store_true")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_smoke_json(output: str) -> dict[str, Any]:
    start = output.find("{")
    end = output.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {"status": "error", "error": output.strip()}
    return json.loads(output[start : end + 1])


def run_smoke(conda: str, model_id: str, env_name: str, args: argparse.Namespace) -> tuple[int, dict[str, Any], str]:
    command = [
        conda,
        "run",
        "-n",
        env_name,
        "python",
        "scripts/run_model_smoke.py",
        "--model",
        model_id,
        "--config",
        args.config,
        "--device",
        args.device,
    ]
    if args.allow_heavy:
        command.append("--allow-heavy")
    print("+ " + " ".join(shlex.quote(part) for part in command))
    completed = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={
            **os.environ,
            "MPLCONFIGDIR": str(Path("/tmp") / "matplotlib"),
            "PYTHONNOUSERSITE": "1",
        },
    )
    parsed = parse_smoke_json(completed.stdout)
    return completed.returncode, parsed, completed.stdout


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    conda = config.get("defaults", {}).get("conda_executable", "conda")
    selected = args.models or list(config["models"])
    run_id = utc_now().replace(":", "").replace("+", "_").replace(".", "_")
    environment = collect_environment(command=" ".join(sys.argv), workdir=ROOT)
    failures = 0

    for model_id in selected:
        model_env = config["models"][model_id]
        registry_model = get_model(model_id, args.registry)
        returncode, smoke, raw_output = run_smoke(conda, model_id, model_env["env_name"], args)
        if returncode != 0 or smoke.get("status") not in PASSING_SMOKE_STATUSES:
            failures += 1

        manifest = {
            "run_id": run_id,
            "model_id": model_id,
            "env": model_env,
            "smoke": smoke,
            "returncode": returncode,
            "raw_output": raw_output,
            "environment": environment,
        }
        manifest_path = Path(args.runs_dir) / run_id / f"{model_id}_smoke.json"
        write_manifest(manifest_path, manifest)

        latency = smoke.get("latency_ms")
        append_result(
            args.results,
            {
                "model_id": model_id,
                "run_id": run_id,
                "benchmark_type": "smoke_inference",
                "dataset": args.dataset,
                "split": args.split,
                "skeleton": registry_model.get("skeleton", ""),
                "num_keypoints": smoke.get("keypoints", ""),
                "precision": "fp32",
                "backend": model_env.get("smoke_profile", ""),
                "hardware": environment.get("nvidia_smi") or "local",
                "batch_size": 1,
                "input_size": registry_model.get("input_size", ""),
                "latency_p50_ms": latency or "",
                "latency_p95_ms": latency or "",
                "fps_per_camera": (1000.0 / latency) if latency else "",
                "gpu_memory_mb": "",
                "git_sha": environment.get("git_sha") or "",
                "command": " ".join(sys.argv),
                "notes": f"status={smoke.get('status')} manifest={manifest_path}",
                "created_at": environment["created_at"],
            },
        )
        print(f"{model_id}: {smoke.get('status')} -> {manifest_path}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

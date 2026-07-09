#!/usr/bin/env python3
"""Sweep RTMPose detector/pose batch sizes and rank benchmark-only runs.

Run this inside the same Conda environment used for RTMPose inference. The
script calls ``scripts/inference/run_phase1_rtmpose_inference.py`` repeatedly, reuses
completed manifests by default, and writes both a per-run CSV and a ranked JSON
summary by ``det_batch_size`` / ``pose_batch_size``.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
INFERENCE_SCRIPT = ROOT / "scripts" / "inference" / "run_phase1_rtmpose_inference.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    selection = parser.add_argument_group("dataset selection")
    selection.add_argument("--drive-root", default="drive")
    selection.add_argument("--groups", nargs="+", default=None)
    selection.add_argument("--deliveries", nargs="+", default=None)
    selection.add_argument("--cameras", nargs="+", default=None)
    selection.add_argument("--frame-limit", type=int, default=300,
                           help="Frames per selected camera for each run (default: 300)")
    selection.add_argument("--start-index", type=int, default=0)
    selection.add_argument("--stride", type=int, default=1)

    tuning = parser.add_argument_group("batch sweep")
    tuning.add_argument("--det-batches", nargs="+", type=int,
                        default=[8, 12, 16, 20, 24, 32])
    tuning.add_argument("--pose-batches", nargs="+", type=int,
                        default=[96, 128, 160, 192, 224, 256, 320, 384, 512])
    tuning.add_argument("--io-workers-list", nargs="+", type=int, default=None,
                        help="Optional sweep over --io-workers values (decode threads). "
                             "Defaults to just [--io-workers]. On I/O-bound machines this "
                             "is often the biggest lever, so sweep e.g. 8 12 16 24.")
    tuning.add_argument("--prefetch-list", nargs="+", type=int, default=None,
                        help="Optional sweep over --prefetch-batches values (read-ahead "
                             "depth). Defaults to [3].")
    tuning.add_argument("--repeats", type=int, default=2,
                        help="Repeat each combination this many times (default: 2)")
    tuning.add_argument("--prefix", default="tune-rtmpose-body8-prof")
    tuning.add_argument("--force", action="store_true",
                        help="Rerun even when a run_manifest.json already exists")
    tuning.add_argument("--stop-on-error", action="store_true")

    runtime = parser.add_argument_group("runtime")
    runtime.add_argument("--model-id", default="rtmpose_l_body8")
    runtime.add_argument("--device", default="cuda:0")
    runtime.add_argument("--io-workers", type=int, default=8)
    runtime.add_argument("--runs-dir", default="benchmarks/runs")
    runtime.add_argument("--results", default="results/rtmpose_batch_tuning.csv")
    runtime.add_argument("--summary", default=None,
                         help="Ranked JSON summary path (default: <results>.summary.json)")
    runtime.add_argument("--show-progress", action="store_true",
                         help="Keep per-run tqdm bars enabled")
    runtime.add_argument("--no-sync-cuda-timing", dest="sync_cuda_timing", action="store_false",
                         help="Do not synchronize CUDA around timed regions")
    runtime.add_argument("--dry-run", action="store_true")
    runtime.add_argument("--extra-args", nargs=argparse.REMAINDER,
                         help="Additional args passed to run_phase1_rtmpose_inference.py")
    runtime.set_defaults(sync_cuda_timing=True)

    return parser.parse_args()


def extend_optional(command: list[str], flag: str, values: list[str] | None) -> None:
    if values:
        command.append(flag)
        command.extend(values)


def build_command(args: argparse.Namespace, det_batch: int, pose_batch: int,
                  io_workers: int, prefetch: int, run_id: str) -> list[str]:
    command = [
        sys.executable,
        str(INFERENCE_SCRIPT),
        "--drive-root", args.drive_root,
        "--model-id", args.model_id,
        "--device", args.device,
        "--run-id", run_id,
        "--det-batch-size", str(det_batch),
        "--pose-batch-size", str(pose_batch),
        "--io-workers", str(io_workers),
        "--prefetch-batches", str(prefetch),
        "--frame-limit", str(args.frame_limit),
        "--start-index", str(args.start_index),
        "--stride", str(args.stride),
        "--benchmark-only",
        "--no-resume",
    ]
    if not args.show_progress:
        command.append("--no-progress")
    if args.sync_cuda_timing:
        command.append("--sync-cuda-timing")
    extend_optional(command, "--groups", args.groups)
    extend_optional(command, "--deliveries", args.deliveries)
    extend_optional(command, "--cameras", args.cameras)
    if args.extra_args:
        command.extend(args.extra_args)
    return command


def manifest_path(args: argparse.Namespace, run_id: str) -> Path:
    runs_dir = Path(args.runs_dir)
    if not runs_dir.is_absolute():
        runs_dir = ROOT / runs_dir
    return runs_dir / run_id / "run_manifest.json"


def row_from_manifest(
    manifest: dict[str, Any],
    *,
    run_id: str,
    repeat: int,
    det_batch: int,
    pose_batch: int,
    io_workers: int,
    prefetch: int,
    manifest_file: Path,
    command: list[str],
    status: str = "success",
    returncode: int = 0,
) -> dict[str, Any]:
    timings = manifest.get("timings", {})
    return {
        "run_id": run_id,
        "status": status,
        "returncode": returncode,
        "repeat": repeat,
        "det_batch_size": det_batch,
        "pose_batch_size": pose_batch,
        "io_workers": io_workers,
        "prefetch_batches": prefetch,
        "fps": manifest.get("fps"),
        "frames_processed": manifest.get("frames_processed"),
        "failed_frames": manifest.get("failed_frames"),
        "total_people": manifest.get("total_people"),
        "elapsed_seconds": manifest.get("elapsed_seconds"),
        "decode_seconds": timings.get("decode_seconds"),
        "detect_seconds": timings.get("detect_seconds"),
        "pose_seconds": timings.get("pose_seconds"),
        "write_seconds": timings.get("write_seconds"),
        "overlay_seconds": timings.get("overlay_seconds"),
        "other_seconds": timings.get("other_seconds"),
        "manifest": str(manifest_file),
        "command": " ".join(shlex.quote(part) for part in command),
    }


def failed_row(
    *,
    run_id: str,
    repeat: int,
    det_batch: int,
    pose_batch: int,
    io_workers: int,
    prefetch: int,
    manifest_file: Path,
    command: list[str],
    returncode: int,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "status": "failed",
        "returncode": returncode,
        "repeat": repeat,
        "det_batch_size": det_batch,
        "pose_batch_size": pose_batch,
        "io_workers": io_workers,
        "prefetch_batches": prefetch,
        "fps": None,
        "frames_processed": None,
        "failed_frames": None,
        "total_people": None,
        "elapsed_seconds": None,
        "decode_seconds": None,
        "detect_seconds": None,
        "pose_seconds": None,
        "write_seconds": None,
        "overlay_seconds": None,
        "other_seconds": None,
        "manifest": str(manifest_file),
        "command": " ".join(shlex.quote(part) for part in command),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id", "status", "returncode", "repeat", "det_batch_size", "pose_batch_size",
        "io_workers", "prefetch_batches",
        "fps", "frames_processed", "failed_frames", "total_people", "elapsed_seconds",
        "decode_seconds", "detect_seconds", "pose_seconds", "write_seconds",
        "overlay_seconds", "other_seconds", "manifest", "command",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ranked_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        if row["status"] != "success" or row["fps"] in (None, ""):
            continue
        key = (
            int(row["det_batch_size"]),
            int(row["pose_batch_size"]),
            int(row.get("io_workers") or 0),
            int(row.get("prefetch_batches") or 0),
        )
        grouped.setdefault(key, []).append(row)

    summary: list[dict[str, Any]] = []
    for (det_batch, pose_batch, io_workers, prefetch), group in grouped.items():
        fps_values = [float(row["fps"]) for row in group]
        failed_frames = sum(int(row.get("failed_frames") or 0) for row in group)
        summary.append({
            "det_batch_size": det_batch,
            "pose_batch_size": pose_batch,
            "io_workers": io_workers,
            "prefetch_batches": prefetch,
            "runs": len(group),
            "median_fps": round(statistics.median(fps_values), 2),
            "best_fps": round(max(fps_values), 2),
            "worst_fps": round(min(fps_values), 2),
            "failed_frames": failed_frames,
            "median_detect_seconds": round(statistics.median(float(row["detect_seconds"] or 0) for row in group), 3),
            "median_pose_seconds": round(statistics.median(float(row["pose_seconds"] or 0) for row in group), 3),
            "run_ids": [row["run_id"] for row in group],
        })

    summary.sort(key=lambda item: (item["median_fps"], item["best_fps"]), reverse=True)
    return summary


def main() -> int:
    args = parse_args()
    if args.frame_limit <= 0:
        raise SystemExit("--frame-limit must be positive")
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    if any(value <= 0 for value in args.det_batches):
        raise SystemExit("--det-batches must all be positive")
    if any(value <= 0 for value in args.pose_batches):
        raise SystemExit("--pose-batches must all be positive")

    io_list = args.io_workers_list or [args.io_workers]
    pf_list = args.prefetch_list or [3]
    combos = [
        (det_batch, pose_batch, io_workers, prefetch)
        for det_batch in args.det_batches
        for pose_batch in args.pose_batches
        for io_workers in io_list
        for prefetch in pf_list
    ]
    total = len(combos) * args.repeats
    rows: list[dict[str, Any]] = []
    counter = 0
    aborted = False
    for det_batch, pose_batch, io_workers, prefetch in combos:
        for repeat in range(1, args.repeats + 1):
            counter += 1
            run_id = f"{args.prefix}-db{det_batch}-pb{pose_batch}-io{io_workers}-pf{prefetch}-r{repeat}"
            command = build_command(args, det_batch, pose_batch, io_workers, prefetch, run_id)
            mpath = manifest_path(args, run_id)
            print(f"\n[{counter}/{total}] det={det_batch} pose={pose_batch} "
                  f"io={io_workers} prefetch={prefetch} repeat={repeat}")
            print("+ " + " ".join(shlex.quote(part) for part in command), flush=True)

            if args.dry_run:
                continue

            if mpath.exists() and not args.force:
                print(f"Reusing existing manifest: {mpath}", flush=True)
                with mpath.open("r", encoding="utf-8") as handle:
                    manifest = json.load(handle)
                rows.append(row_from_manifest(
                    manifest, run_id=run_id, repeat=repeat, det_batch=det_batch,
                    pose_batch=pose_batch, io_workers=io_workers, prefetch=prefetch,
                    manifest_file=mpath, command=command,
                ))
                continue

            completed = subprocess.run(command, cwd=ROOT, check=False)
            if completed.returncode != 0:
                rows.append(failed_row(
                    run_id=run_id, repeat=repeat, det_batch=det_batch, pose_batch=pose_batch,
                    io_workers=io_workers, prefetch=prefetch, manifest_file=mpath,
                    command=command, returncode=completed.returncode,
                ))
                if args.stop_on_error:
                    aborted = True
                    break
                continue

            with mpath.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            rows.append(row_from_manifest(
                manifest, run_id=run_id, repeat=repeat, det_batch=det_batch,
                pose_batch=pose_batch, io_workers=io_workers, prefetch=prefetch,
                manifest_file=mpath, command=command, returncode=completed.returncode,
            ))
        if aborted:
            break

    if args.dry_run:
        print("\nDry run only; no benchmarks were executed.")
        return 0

    results_path = Path(args.results)
    if not results_path.is_absolute():
        results_path = ROOT / results_path
    summary_path = Path(args.summary) if args.summary else results_path.with_suffix(".summary.json")
    if not summary_path.is_absolute():
        summary_path = ROOT / summary_path

    write_csv(results_path, rows)
    summary = ranked_summary(rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nWrote per-run results: {results_path}")
    print(f"Wrote ranked summary: {summary_path}")
    if summary:
        print("\nTop settings by median FPS:")
        for item in summary[:10]:
            print(
                f"  fps={item['median_fps']:8.2f} "
                f"best={item['best_fps']:8.2f} "
                f"det_batch={item['det_batch_size']:3} "
                f"pose_batch={item['pose_batch_size']:4} "
                f"io={item['io_workers']:3} "
                f"prefetch={item['prefetch_batches']:2} "
                f"runs={item['runs']} failed={item['failed_frames']}"
            )
    return 1 if any(row["status"] != "success" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())

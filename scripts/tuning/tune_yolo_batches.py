#!/usr/bin/env python3
"""Quickly tune YOLO Phase 1 batch size on local cricket frames.

Run this inside the YOLO Conda environment. It loads the model once, decodes a
small frame sample once, sweeps candidate batch sizes, catches CUDA OOM, and
writes a ranked JSON summary.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pose_estimation.cricket.dataset import camera_label, frame_paths_for_camera, resolve_delivery_camera_dirs
from pose_estimation.cricket.phase1_runner import clear_cuda_cache, read_batch_images
from pose_estimation.cricket.phase1_yolo_adapter import YOLOPoseAdapter, load_model_config


DEFAULT_MODEL_CONFIG = ROOT / "configs" / "model_envs.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-root", default="drive")
    parser.add_argument("--delivery-id", default="CCPL080626M1_1_14_2")
    parser.add_argument("--cameras", nargs="+", default=["cam_01"])
    parser.add_argument("--frame-limit", type=int, default=96)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[8, 16, 24, 32, 48, 64])
    parser.add_argument("--model-id", default="yolo26x_pose")
    parser.add_argument("--model-config", default=str(DEFAULT_MODEL_CONFIG))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--no-half", dest="half", action="store_false")
    parser.set_defaults(half=True)
    parser.add_argument(
        "--resize-long-side",
        type=int,
        default=None,
        help="Resize decoded frames before inference; defaults to --imgsz",
    )
    parser.add_argument("--decode-workers", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1, help="Warmup batches per candidate")
    parser.add_argument("--output", default="results/yolo_batch_tuning.json")
    return parser.parse_args()


def normalize_camera(value: str) -> str:
    digits = "".join(ch for ch in value if ch.isdigit())
    if not digits:
        raise ValueError(f"cannot parse camera id: {value!r}")
    return camera_label(digits)


def resolve_model_paths(model_config: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(model_config)
    for key in ["model_name", "checkpoint"]:
        value = resolved.get(key)
        if not value:
            continue
        path = Path(value)
        if not path.is_absolute():
            resolved[key] = str((ROOT / path).resolve())
    return resolved


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def synchronize_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def reset_cuda_peak_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def cuda_peak_mib() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
    except Exception:
        pass
    return None


def is_oom(exc: Exception) -> bool:
    text = str(exc).lower()
    return "cuda out of memory" in text or "outofmemoryerror" in text


def chunked(values: list[Any], size: int) -> list[list[Any]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def collect_sample_frames(args: argparse.Namespace) -> list[Path]:
    drive_root = Path(args.drive_root)
    if not drive_root.is_absolute():
        drive_root = ROOT / drive_root
    camera_dirs = resolve_delivery_camera_dirs(drive_root, args.delivery_id)
    wanted = {normalize_camera(camera) for camera in args.cameras}

    frames: list[Path] = []
    per_camera_limit = max(1, args.frame_limit // max(1, len(wanted)))
    for camera_id in sorted(wanted):
        camera_dir = camera_dirs.get(camera_id)
        if camera_dir is None:
            raise SystemExit(f"{args.delivery_id} does not contain {camera_id}")
        selected = frame_paths_for_camera(camera_dir)[args.start_index :]
        frames.extend(selected[:per_camera_limit])

    if not frames:
        raise SystemExit("No frames selected for tuning")
    return frames[: args.frame_limit]


def tune_batch_size(adapter: YOLOPoseAdapter, images: list[Any], batch_size: int, warmup: int) -> dict[str, Any]:
    clear_cuda_cache()
    try:
        for _ in range(max(0, warmup)):
            adapter.predict_batch(images[: min(batch_size, len(images))], batch_size=batch_size)
        synchronize_cuda()
        reset_cuda_peak_memory()
        start = time.perf_counter()
        detections = 0
        for batch in chunked(images, batch_size):
            predictions = adapter.predict_batch(batch, batch_size=batch_size)
            detections += sum(len(frame_predictions) for frame_predictions in predictions)
        synchronize_cuda()
        elapsed = time.perf_counter() - start
    except Exception as exc:  # noqa: BLE001 - reported in tuning output.
        clear_cuda_cache()
        status = "oom" if is_oom(exc) else "error"
        return {
            "batch_size": batch_size,
            "status": status,
            "error": str(exc),
        }

    fps = len(images) / elapsed if elapsed > 0 else 0.0
    return {
        "batch_size": batch_size,
        "status": "ok",
        "images": len(images),
        "elapsed_seconds": round(elapsed, 4),
        "fps": round(fps, 2),
        "ms_per_image": round((elapsed * 1000) / len(images), 3),
        "detections": detections,
        "cuda_peak_mib": cuda_peak_mib(),
    }


def main() -> int:
    args = parse_args()
    if args.model_id != "yolo26x_pose":
        raise SystemExit("This tuner is for yolo26x_pose only.")
    if args.frame_limit <= 0:
        raise SystemExit("--frame-limit must be positive")
    if args.decode_workers <= 0:
        raise SystemExit("--decode-workers must be positive")
    if any(size <= 0 for size in args.batch_sizes):
        raise SystemExit("--batch-sizes must all be positive")
    if args.device.startswith("cuda") and not cuda_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false in this environment")

    resize_long_side = args.resize_long_side if args.resize_long_side is not None else args.imgsz
    frame_paths = collect_sample_frames(args)
    print(
        f"Decoding {len(frame_paths)} frame(s) from {args.delivery_id} "
        f"with resize_long_side={resize_long_side} decode_workers={args.decode_workers}",
        flush=True,
    )
    decoded_frames, failures, decode_ms = read_batch_images(
        frame_paths,
        resize_long_side=resize_long_side,
        decode_workers=args.decode_workers,
    )
    if failures:
        raise SystemExit(f"Decode failures during tuning: {failures[:3]}")
    images = [frame.image for frame in decoded_frames]

    model_config = resolve_model_paths(load_model_config(args.model_id, args.model_config))
    adapter = YOLOPoseAdapter(
        model_id=args.model_id,
        model_config=model_config,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        half=args.half and args.device != "cpu",
    )

    results: list[dict[str, Any]] = []
    for batch_size in args.batch_sizes:
        print(f"Testing batch_size={batch_size} ...", flush=True)
        result = tune_batch_size(adapter, images, batch_size, args.warmup)
        results.append(result)
        if result["status"] == "ok":
            print(
                f"  ok: {result['fps']} fps, {result['ms_per_image']} ms/image, "
                f"peak={result['cuda_peak_mib']} MiB",
                flush=True,
            )
        else:
            print(f"  {result['status']}: {result.get('error', '')}", flush=True)

    ranked = sorted(
        (result for result in results if result["status"] == "ok"),
        key=lambda item: item["fps"],
        reverse=True,
    )
    recommended = ranked[0]["batch_size"] if ranked else None
    output = {
        "schema_version": "yolo_batch_tuning/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(shlex.quote(part) for part in sys.argv),
        "model_id": args.model_id,
        "device": args.device,
        "imgsz": args.imgsz,
        "resize_long_side": resize_long_side,
        "half": args.half and args.device != "cpu",
        "decode_workers": args.decode_workers,
        "sample_frames": len(images),
        "decode_seconds": round(decode_ms / 1000, 4),
        "results": results,
        "recommended_batch_size": recommended,
    }
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print(f"\nWrote tuning summary: {output_path}")
    if recommended is not None:
        print(f"Recommended --batch-size {recommended}")
    else:
        print("No successful batch size found.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

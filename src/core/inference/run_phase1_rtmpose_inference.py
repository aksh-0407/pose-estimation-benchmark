#!/usr/bin/env python3
"""Run RTMPose (top-down) 2D pose inference on the internal cricket frame dataset.

RTMPose is a top-down model: a person detector (RTMDet) produces boxes, then
RTMPose predicts keypoints per box. This script walks ``drive/dataset/<group>/
<delivery>/camera<NN>/`` frame folders, runs detection + pose, and writes one
JSONL prediction file per camera under ``data/derived/runs/<run-id>/predictions/``.

Everything is filterable from the CLI so you can run the whole dataset, a single
capture group, one delivery, one camera, or a short frame slice -- see --help.

Run inside the model's Conda env, e.g.:

    conda activate pose-lab
    python src/core/inference/run_phase1_rtmpose_inference.py --list
    python src/core/inference/run_phase1_rtmpose_inference.py \
        --groups bt_01 --deliveries CCPL080626M1_1_14_1 --cameras camera01 \
        --frame-limit 50 --overlay
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import warnings
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import yaml

from core.contract import (
    SCHEMA_VERSION as P1_SCHEMA_VERSION,
    SKELETON as P1_SKELETON,
    validate_group1_frame,
)
from core.dataset import FRAME_RE, camera_label, parse_frame_id, repo_relative
from core.datasets import derived_root, raw_root

# Detailed per-frame failure records are only for the manifest dump (the real
# counts live in cam_failed). Cap the retained list so a pathological run cannot
# grow it without bound.
MAX_FAILURE_RECORDS = 2000

MMPOSE_DIR = ROOT / "external" / "mmpose"
DEFAULT_MODEL_CONFIG = ROOT / "configs" / "model_envs.yaml"
DEFAULT_DET_CONFIG = ROOT / "external" / "mmpose" / "demo" / "mmdetection_cfg" / "rtmdet_m_640-8xb32_coco-person.py"
DEFAULT_DET_CHECKPOINT = ROOT / "models" / "rtmdet_m_person" / "weights" / "rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Dataset selection / filtering
    sel = parser.add_argument_group("dataset selection")
    sel.add_argument("--data-root", default=None,
                     help="Base dir for raw/derived/viz (default: $PIPETRACK_DATA or 'data'; L40S: ~/bits-pose-data)")
    sel.add_argument("--dataset", default=None,
                     help="Dataset from configs/datasets.yaml (e.g. 8_init, 40_full); derives --drive-root + --run-dir")
    sel.add_argument("--version", default=None,
                     help="Run version -> pipetrack_v<version> (with --dataset, P1 writes <derived>/p1)")
    sel.add_argument("--drive-root", default=None,
                     help="Dataset raw/footage root (contains bt_0X/, calibration-data/, events-data/). "
                          "Default with --dataset: <DATA_ROOT>/raw/<dataset>.")
    sel.add_argument("--groups", nargs="+", default=None,
                     help="Capture groups to include, e.g. bt_01 bt_02 (default: all found)")
    sel.add_argument("--deliveries", nargs="+", default=None,
                     help="Delivery IDs to include; matched as exact or substring (default: all)")
    sel.add_argument("--cameras", nargs="+", default=None,
                     help="Cameras to include, accepts camera01 / cam_01 / 01 / 1 (default: all)")
    sel.add_argument("--frame-limit", type=int, default=None,
                     help="Max frames per camera after start/stride (default: all)")
    sel.add_argument("--start-index", type=int, default=0, help="Skip this many frames per camera first")
    sel.add_argument("--stride", type=int, default=1, help="Take every Nth frame (default: 1)")
    sel.add_argument("--list", action="store_true", help="List the cameras/frames that would run, then exit")

    # Model / weights
    mdl = parser.add_argument_group("model")
    mdl.add_argument("--model-id", default="rtmpose_x_body8",
                     help="Key in configs/model_envs.yaml supplying pose config+checkpoint. "
                          "Default rtmpose_x_body8 (Halpe-26, accuracy-first); all 26 keypoints "
                          "(COCO-17 in [0:17] + head/neck/hip + feet) are emitted as pose_2d.")
    mdl.add_argument("--model-config", default=str(DEFAULT_MODEL_CONFIG))
    mdl.add_argument("--pose-config", default=None, help="Override pose config path")
    mdl.add_argument("--pose-checkpoint", default=None, help="Override pose checkpoint path")
    mdl.add_argument("--det-config", default=str(DEFAULT_DET_CONFIG))
    mdl.add_argument("--det-checkpoint", default=str(DEFAULT_DET_CHECKPOINT))
    mdl.add_argument("--det-cat-id", type=int, default=0, help="Detector category id for person (default: 0)")
    mdl.add_argument("--bbox-thr", type=float, default=0.3, help="Min detection score to keep a box")
    mdl.add_argument("--nms-thr", type=float, default=0.3, help="IoU threshold for box NMS")
    mdl.add_argument("--kpt-thr", type=float, default=0.3, help="Keypoint score threshold for overlays")
    mdl.add_argument("--max-people", type=int, default=None, help="Keep only top-N highest-scoring boxes per frame")

    # Runtime
    rt = parser.add_argument_group("runtime")
    rt.add_argument("--device", default="cuda:0", help="cuda:0 / cpu (default: cuda:0)")
    rt.add_argument("--allow-cpu", action="store_true", help="Permit CPU when CUDA is unavailable")
    rt.add_argument("--run-id", default=None, help="Run identifier (default: auto timestamp)")
    rt.add_argument("--run-dir", default=None, help="Output dir (default: data/derived/runs/<run-id>)")
    rt.add_argument("--no-resume", dest="resume", action="store_false",
                    help="Recompute frames already present in the camera JSONL")
    rt.add_argument("--no-progress", dest="show_progress", action="store_false",
                    help="Disable the tqdm progress bar")
    rt.add_argument("--det-batch-size", type=int, default=8,
                    help="Frames per batched detector call (default: 8)")
    rt.add_argument("--pose-batch-size", type=int, default=64,
                    help="Person crops per batched RTMPose call (default: 64)")
    rt.add_argument("--io-workers", type=int, default=4,
                    help="CPU worker threads for image decode prefetch (default: 4)")
    rt.add_argument("--cv2-threads", type=int, default=2,
                    help="OpenCV internal threads per decode call (default: 2). Kept low "
                         "so N io-workers x cv2 threads does not oversubscribe the CPU and "
                         "starve the GPU; the GPU does the pose/detect compute.")
    rt.add_argument("--prefetch-batches", type=int, default=3,
                    help="How many detector batches to read+decode ahead on the io-worker "
                         "threads while the GPU runs the current batch (default: 3). This "
                         "overlaps cold-disk frame reads with GPU compute so the GPU stays "
                         "fed. Set 0/1 to effectively disable look-ahead.")
    rt.add_argument("--benchmark-only", action="store_true",
                    help="Run inference and timing without writing prediction JSONL or overlays")
    rt.add_argument("--sync-cuda-timing", action="store_true",
                    help="Synchronize CUDA around timed regions for more accurate benchmark timings")
    rt.add_argument("--show-torch-warnings", action="store_true",
                    help="Show known upstream torch warnings that are hidden by default")
    rt.set_defaults(resume=True)
    rt.set_defaults(show_progress=True)

    # Overlays
    ov = parser.add_argument_group("overlays")
    ov.add_argument("--overlay", action="store_true", help="Render keypoint overlay images")
    ov.add_argument("--no-smoke-overlay", dest="smoke_overlay", action="store_false",
                    help="Do not auto-enable overlays when --run-id contains 'smoke'")
    ov.add_argument("--overlay-every", type=int, default=1, help="Render every Nth processed frame (default: 1)")
    ov.add_argument("--overlay-frame-ids", nargs="+", type=int, default=None,
                    help="Render overlays only for these exact frame IDs from filenames, e.g. 1 150 300")
    ov.add_argument("--overlay-row-indices", nargs="+", type=int, default=None,
                    help="Render overlays for these one-based selected-frame positions per camera, e.g. 1 150 300")
    ov.add_argument("--overlay-limit", type=int, default=30, help="Max overlays per camera (default: 30)")
    ov.set_defaults(smoke_overlay=True)

    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_device(device: str, *, allow_cpu: bool) -> str:
    if device == "cpu" and not allow_cpu:
        raise SystemExit("CPU requested without --allow-cpu")
    if device.startswith("cuda") and not cuda_available():
        if allow_cpu:
            print("WARN: CUDA unavailable, falling back to CPU", flush=True)
            return "cpu"
        raise SystemExit("CUDA device requested but torch.cuda.is_available() is false. Use --allow-cpu to override.")
    return device


def resolve_model_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.pose_config and args.pose_checkpoint:
        return Path(args.pose_config), Path(args.pose_checkpoint)
    with Path(args.model_config).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    models = cfg.get("models", {})
    if args.model_id not in models:
        raise SystemExit(f"model-id '{args.model_id}' not found in {args.model_config}")
    entry = models[args.model_id]
    pose_config = Path(args.pose_config or entry["config"])
    pose_checkpoint = Path(args.pose_checkpoint or entry["checkpoint"])
    return pose_config, pose_checkpoint


def abspath(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (ROOT / path).resolve()


def rel(path: str | Path) -> str:
    """Repo-root-relative string for portable run artifacts."""
    return repo_relative(path, ROOT)


def build_progress_bar(total: int, enabled: bool):
    if not enabled:
        return None
    try:
        from tqdm import tqdm
    except ImportError:
        print("WARN: tqdm is not installed; progress bar disabled.", flush=True)
        return None

    return tqdm(
        total=total,
        desc="rtmpose",
        unit="frame",
        dynamic_ncols=True,
        smoothing=0.05,
        mininterval=0.5,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )


def progress_write(progress, message: str) -> None:
    if progress is None:
        print(message, flush=True)
    else:
        progress.write(message)


def progress_step(
    progress,
    *,
    camera: str,
    cam_processed: int,
    cam_skipped: int,
    cam_failed: int,
    cam_people: int,
    overlays: int,
) -> None:
    if progress is None:
        return
    progress.set_description_str(camera, refresh=False)
    progress.set_postfix(
        ok=cam_processed,
        skip=cam_skipped,
        fail=cam_failed,
        people=cam_people,
        overlays=overlays,
        refresh=False,
    )
    progress.update(1)


def chunked(items: list[Any], size: int):
    if size <= 0:
        raise ValueError("batch size must be positive")
    for start in range(0, len(items), size):
        yield items[start:start + size]


def sync_cuda_if_requested(device: str, enabled: bool) -> None:
    if not enabled or not device.startswith("cuda"):
        return
    try:
        import torch
        torch.cuda.synchronize()
    except Exception:
        pass


def timed_call(timings: dict[str, float], key: str, device: str, sync_cuda: bool, fn, *args, **kwargs):
    sync_cuda_if_requested(device, sync_cuda)
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    sync_cuda_if_requested(device, sync_cuda)
    timings[key] = timings.get(key, 0.0) + time.perf_counter() - start
    return result


def load_frame_for_batch(item: tuple[int, Path]) -> dict[str, Any]:
    idx, frame_path = item
    import cv2

    img = cv2.imread(str(frame_path))
    if img is None:
        raise RuntimeError("image failed to decode")
    height, width = img.shape[:2]
    return {"idx": idx, "frame_path": frame_path, "img": img, "height": height, "width": width}


def load_frame_batch(items: list[tuple[int, Path]], workers: int) -> tuple[list[dict[str, Any]], list[tuple[int, Path, Exception]]]:
    loaded: list[dict[str, Any]] = []
    failed: list[tuple[int, Path, Exception]] = []
    if workers <= 1 or len(items) <= 1:
        for item in items:
            try:
                loaded.append(load_frame_for_batch(item))
            except Exception as exc:  # noqa: BLE001
                failed.append((item[0], item[1], exc))
        return loaded, failed

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [(item, pool.submit(load_frame_for_batch, item)) for item in items]
        for item, future in futures:
            try:
                loaded.append(future.result())
            except Exception as exc:  # noqa: BLE001
                failed.append((item[0], item[1], exc))
    loaded.sort(key=lambda entry: entry["idx"])
    return loaded, failed


def prefetch_decoded_batches(
    pending: list[tuple[int, Path]],
    executor: ThreadPoolExecutor,
    det_batch_size: int,
    depth: int,
    timings: dict[str, float],
    loader=None,
):
    """Yield ``(loaded, load_failures)`` per detector batch while decoding ahead.

    Disk read + JPEG decode (CPU) for the next ``depth`` batches run on the shared
    ``executor`` while the caller runs detection/pose on the GPU for the current
    batch. This overlaps the CPU/IO stage with the GPU stage so the GPU is not left
    idle waiting on cold-disk frame reads. ``decode_seconds`` only accrues the time
    actually spent *blocked* waiting for a future, so a well-fed pipeline reports it
    near zero.
    """
    from collections import deque

    batches = list(chunked(pending, det_batch_size))
    inflight: deque = deque()
    nxt = 0

    load = loader or load_frame_for_batch

    def submit(index: int):
        return [
            (idx, path, executor.submit(load, (idx, path)))
            for idx, path in batches[index]
        ]

    while nxt < len(batches) and len(inflight) < max(1, depth):
        inflight.append(submit(nxt))
        nxt += 1

    while inflight:
        futures = inflight.popleft()
        # Submit the next batch's decode BEFORE we block, so it overlaps the GPU
        # work the caller does with the batch we are about to yield.
        if nxt < len(batches):
            inflight.append(submit(nxt))
            nxt += 1
        start = time.perf_counter()
        loaded: list[dict[str, Any]] = []
        failed: list[tuple[int, Path, Exception]] = []
        for idx, path, future in futures:
            try:
                loaded.append(future.result())
            except Exception as exc:  # noqa: BLE001
                failed.append((idx, path, exc))
        loaded.sort(key=lambda entry: entry["idx"])
        timings["decode_seconds"] += time.perf_counter() - start
        yield loaded, failed


def normalize_camera_filters(values: list[str] | None) -> set[str] | None:
    """Accept camera01 / cam_01 / 01 / 1 and normalize to cam_NN labels."""
    if not values:
        return None
    out: set[str] = set()
    for value in values:
        digits = "".join(ch for ch in value if ch.isdigit())
        if not digits:
            continue
        out.add(camera_label(digits))
    return out or None


def discover_targets(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Return one record per selected camera: paths + filtered frame list."""
    dataset_root = abspath(args.drive_root)
    if not dataset_root.exists():
        raise SystemExit(f"dataset root not found: {dataset_root}")

    group_filter = set(args.groups) if args.groups else None
    camera_filter = normalize_camera_filters(args.cameras)
    delivery_filter = args.deliveries

    targets: list[dict[str, Any]] = []
    for group_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        if group_filter and group_dir.name not in group_filter:
            continue
        if not group_dir.name.startswith("bt_"):
            continue
        for delivery_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
            if delivery_filter and not any(
                f == delivery_dir.name or f in delivery_dir.name for f in delivery_filter
            ):
                continue
            for camera_dir in sorted(p for p in delivery_dir.iterdir() if p.is_dir()):
                if not camera_dir.name.startswith("camera"):
                    continue
                cam_id = camera_label(camera_dir.name.replace("camera", ""))
                if camera_filter and cam_id not in camera_filter:
                    continue
                frames = sorted(
                    (f for f in camera_dir.glob("*.jpg") if FRAME_RE.match(f.name)),
                    key=lambda f: parse_frame_id(f),
                )
                frames = frames[args.start_index:]
                if args.stride > 1:
                    frames = frames[:: args.stride]
                if args.frame_limit is not None:
                    frames = frames[: args.frame_limit]
                if not frames:
                    continue
                targets.append({
                    "group": group_dir.name,
                    "delivery_id": delivery_dir.name,
                    "camera_dir": camera_dir,
                    "camera_id": cam_id,
                    "camera_number": camera_dir.name.replace("camera", ""),
                    "frames": frames,
                })
    return targets


def match_id_from_delivery(delivery_id: str) -> str:
    head = delivery_id.split("_", 1)[0]
    return head[:-2] if head[-2:].startswith("M") else head


def git_sha(workdir: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(workdir), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def camera_metric_key(group: str, delivery_id: str, camera_id: str) -> str:
    return f"{group}/{delivery_id}/{camera_id}"


def append_capped(items: list[Any], value: Any, *, cap: int = MAX_FAILURE_RECORDS) -> None:
    if len(items) < cap:
        items.append(value)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def build_models(args: argparse.Namespace, device: str):
    try:
        from mmdet.apis import inference_detector, init_detector
        from mmpose.apis import init_model
        from mmpose.utils import adapt_mmdet_pipeline
    except ModuleNotFoundError as exc:
        missing = exc.name or "required OpenMMLab package"
        raise SystemExit(
            f"Missing {missing!r} in the active environment. Rebuild/repair this model env with:\n"
            f"  python3 tools/setup_model_envs.py --models {args.model_id} --force-install --download-assets\n"
            f"Then activate the configured Conda env and rerun this script."
        ) from exc

    pose_config, pose_checkpoint = resolve_model_paths(args)
    pose_config, pose_checkpoint = abspath(pose_config), abspath(pose_checkpoint)
    det_config, det_checkpoint = abspath(args.det_config), abspath(args.det_checkpoint)
    for label, path in [("pose config", pose_config), ("pose checkpoint", pose_checkpoint),
                        ("detector config", det_config), ("detector checkpoint", det_checkpoint)]:
        if not path.exists():
            raise SystemExit(f"missing {label}: {path}")

    # MMPose/MMDet configs resolve _base_ includes relative to the mmpose tree;
    # chdir there for init only, then restore so output paths stay correct.
    previous_cwd = Path.cwd()
    os.chdir(MMPOSE_DIR)
    try:
        detector = init_detector(str(det_config), str(det_checkpoint), device=device)
        detector.cfg = adapt_mmdet_pipeline(detector.cfg)
        pose_model = init_model(str(pose_config), str(pose_checkpoint), device=device)
    finally:
        os.chdir(previous_cwd)
    return detector, pose_model, inference_detector, str(pose_config), str(pose_checkpoint)


def boxes_from_det_result(det_result, args: argparse.Namespace):
    import numpy as np
    from mmpose.evaluation.functional import nms

    pred = det_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate((pred.bboxes, pred.scores[:, None]), axis=1)
    keep = np.logical_and(pred.labels == args.det_cat_id, pred.scores > args.bbox_thr)
    bboxes = bboxes[keep]
    if bboxes.shape[0]:
        bboxes = bboxes[nms(bboxes, args.nms_thr)]
    if args.max_people is not None and bboxes.shape[0] > args.max_people:
        order = np.argsort(bboxes[:, 4])[::-1][: args.max_people]
        bboxes = bboxes[order]
    return bboxes  # (N, 5): x1,y1,x2,y2,score


def detect_person_boxes(detector, inference_detector, image_path: str, args: argparse.Namespace):
    det_result = inference_detector(detector, image_path)
    return boxes_from_det_result(det_result, args)


def detect_person_boxes_batch(detector, inference_detector, entries: list[dict[str, Any]], args: argparse.Namespace):
    if not entries:
        return []
    images = [entry["img"] for entry in entries]
    try:
        det_results = inference_detector(detector, images)
        if not isinstance(det_results, list):
            det_results = [det_results]
        if len(det_results) != len(entries):
            raise RuntimeError(f"detector returned {len(det_results)} results for {len(entries)} inputs")
        return [boxes_from_det_result(result, args) for result in det_results]
    except Exception:
        # Some MMDetection versions/plugins do not accept a list input here.
        return [detect_person_boxes(detector, inference_detector, entry["img"], args) for entry in entries]


def build_pose_pipeline(pose_model):
    from mmengine.dataset import Compose

    return Compose(pose_model.cfg.test_dataloader.dataset.pipeline)


def inference_topdown_batch(pose_model, pose_pipeline, entries: list[dict[str, Any]],
                            pose_batch_size: int) -> list[list[Any]]:
    import numpy as np
    import torch
    from mmengine.dataset import pseudo_collate
    from mmengine.registry import init_default_scope

    grouped_results: list[list[Any]] = [[] for _ in entries]
    flat_samples: list[dict[str, Any]] = []
    owners: list[int] = []

    scope = pose_model.cfg.get("default_scope", "mmpose")
    if scope is not None:
        init_default_scope(scope)

    for owner_idx, entry in enumerate(entries):
        boxes = entry["boxes"]
        if boxes is None or len(boxes) == 0:
            continue
        for box in boxes:
            data_info = {
                "img": entry["img"],
                "bbox": np.asarray(box[:4], dtype=np.float32)[None],
                "bbox_score": np.asarray([box[4]], dtype=np.float32),
            }
            data_info.update(pose_model.dataset_meta)
            flat_samples.append(pose_pipeline(data_info))
            owners.append(owner_idx)

    for sample_batch, owner_batch in zip(chunked(flat_samples, pose_batch_size), chunked(owners, pose_batch_size)):
        batch = pseudo_collate(sample_batch)
        with torch.no_grad():
            results = pose_model.test_step(batch)
        for owner_idx, result in zip(owner_batch, results):
            grouped_results[owner_idx].append(result)

    return grouped_results


def coco17_source_indices(source_skeleton: str) -> list[int] | None:
    """Indices into the source skeleton that yield COCO-17 order, or None if unmapped."""
    from core.keypoints import load_keypoint_mappings

    mapping = load_keypoint_mappings()["source_to_coco_17"].get(source_skeleton)
    return mapping["source_indices"] if mapping else None


def finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def select_coco17_pose(
    keypoints_px: list[list[float]],
    confidence: list[float],
    coco17_indices: list[int] | None,
) -> tuple[list[list[float]], list[float]]:
    source_indices = coco17_indices or list(range(17))
    output_keypoints: list[list[float]] = []
    output_confidence: list[float] = []
    for source_index in source_indices[:17]:
        if source_index < len(keypoints_px):
            x, y = keypoints_px[source_index]
            output_keypoints.append([finite_float(x), finite_float(y)])
        else:
            output_keypoints.append([0.0, 0.0])
        output_confidence.append(
            finite_float(confidence[source_index]) if source_index < len(confidence) else 0.0
        )
    while len(output_keypoints) < 17:
        output_keypoints.append([0.0, 0.0])
        output_confidence.append(0.0)
    return output_keypoints, output_confidence


def bbox_score_from_instance(inst) -> float | None:
    for name in ("bbox_scores", "bbox_score"):
        if not hasattr(inst, name):
            continue
        values = getattr(inst, name)
        try:
            if len(values):
                return finite_float(values[0])
        except TypeError:
            return finite_float(values)
    return None


def player_records(results, source_skeleton: str, width: int, height: int,
                   coco17_indices: list[int] | None) -> list[dict[str, Any]]:
    """Convert mmpose top-down results into player dicts matching the cricket schema."""
    players: list[dict[str, Any]] = []
    for res in results:
        inst = res.pred_instances
        kpts = inst.keypoints[0]            # (K, 2)
        scores = inst.keypoint_scores[0]    # (K,)
        bbox = inst.bboxes[0] if hasattr(inst, "bboxes") and len(inst.bboxes) else None
        if bbox is not None:
            x1, y1, x2, y2 = (float(v) for v in bbox)
        else:
            import numpy as np

            xs, ys = np.asarray(kpts[:, 0], dtype=float), np.asarray(kpts[:, 1], dtype=float)
            if np.isfinite(xs).any() and np.isfinite(ys).any():
                x1, y1 = float(np.nanmin(xs)), float(np.nanmin(ys))
                x2, y2 = float(np.nanmax(xs)), float(np.nanmax(ys))
            else:
                x1 = y1 = x2 = y2 = 0.0
        bw, bh = x2 - x1, y2 - y1

        source_keypoints_px = [[finite_float(x), finite_float(y)] for x, y in kpts]
        source_confidence = [finite_float(s) for s in scores]
        # Halpe-26 is the canonical pipeline skeleton: pose_2d carries every native
        # joint the model emits (for RTMPose-x that is all 26 — COCO-17 in indices
        # [0:17] plus head/neck/hip and the 6 foot joints).
        source_keypoints_norm = [[x / width, y / height] for x, y in source_keypoints_px]

        players.append({
            "bbox_xywh_px": [x1, y1, bw, bh],
            "bbox_xywh_norm": [x1 / width, y1 / height, bw / width, bh / height],
            "global_player_id": None,
            "local_track_id": None,
            "detection_confidence": bbox_score_from_instance(inst),
            "track_confidence": None,
            "role": "unknown",
            "pose_2d": {
                "skeleton": source_skeleton,
                "keypoints_px": source_keypoints_px,
                "keypoints_norm": source_keypoints_norm,
                "confidence": source_confidence,
            },
            "pose_3d": None,
        })
    return players


def build_visualizer(pose_model, kpt_thr: float):
    from mmpose.registry import VISUALIZERS
    pose_model.cfg.visualizer.radius = 4
    pose_model.cfg.visualizer.line_width = 2
    visualizer = VISUALIZERS.build(pose_model.cfg.visualizer)
    visualizer.set_dataset_meta(pose_model.dataset_meta, skeleton_style="mmpose")
    return visualizer


def render_overlay(visualizer, image_path: str, results, out_path: Path, kpt_thr: float) -> None:
    import mmcv

    img = mmcv.imread(image_path, channel_order="rgb")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        mmcv.imwrite(img[..., ::-1], str(out_path))
        return

    from mmpose.structures import merge_data_samples

    data_samples = merge_data_samples(results)
    visualizer.add_datasample(
        "result", img, data_sample=data_samples, draw_gt=False,
        draw_bbox=True, show=False, kpt_thr=kpt_thr,
    )
    mmcv.imwrite(visualizer.get_image()[..., ::-1], str(out_path))


def should_render_overlay(args: argparse.Namespace, entry: dict[str, Any], overlays: int) -> bool:
    if overlays >= args.overlay_limit:
        return False
    if args.overlay_frame_ids is not None:
        return parse_frame_id(entry["frame_path"]) in args.overlay_frame_ids
    if args.overlay_row_indices is not None:
        return (entry["idx"] + 1) in args.overlay_row_indices
    return entry["idx"] % args.overlay_every == 0


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    args = parse_args()
    # Dataset abstraction: --dataset (+ --version) derive the footage root and the
    # P1 output dir (<derived>/pipetrack_v<version>/p1); explicit flags win.
    if args.dataset:
        if not args.version:
            raise SystemExit("--dataset requires --version (-> pipetrack_v<version>)")
        if args.drive_root is None:
            args.drive_root = str(raw_root(args.data_root, args.dataset))
        if args.run_dir is None:
            args.run_dir = str(derived_root(args.data_root, args.dataset, args.version) / "p1")
    if args.drive_root is None:
        args.drive_root = "drive"
    if not args.show_torch_warnings:
        warnings.filterwarnings(
            "ignore",
            message=r"torch\.meshgrid: in an upcoming release.*",
            category=UserWarning,
        )
    if args.det_batch_size <= 0:
        raise SystemExit("--det-batch-size must be positive")
    if args.pose_batch_size <= 0:
        raise SystemExit("--pose-batch-size must be positive")
    if args.io_workers < 0:
        raise SystemExit("--io-workers must be >= 0")
    if args.cv2_threads < 1:
        raise SystemExit("--cv2-threads must be >= 1")
    if args.prefetch_batches < 0:
        raise SystemExit("--prefetch-batches must be >= 0")
    # Cap CPU-side parallelism so decode threads don't oversubscribe all cores and
    # thrash (which starves the GPU). OpenCV otherwise defaults to every core, so
    # io_workers x cores threads fight; keep each decode light and let the GPU carry
    # the detect/pose compute.
    try:
        import cv2
        cv2.setNumThreads(args.cv2_threads)
    except Exception:
        pass
    try:
        import torch
        torch.set_num_threads(max(1, args.cv2_threads))
    except Exception:
        pass
    if args.overlay_every <= 0:
        raise SystemExit("--overlay-every must be positive")
    if args.overlay_limit < 0:
        raise SystemExit("--overlay-limit must be >= 0")
    if args.overlay_frame_ids is not None:
        if any(frame_id < 0 for frame_id in args.overlay_frame_ids):
            raise SystemExit("--overlay-frame-ids must be non-negative")
        args.overlay_frame_ids = set(args.overlay_frame_ids)
    if args.overlay_row_indices is not None:
        if any(row_index <= 0 for row_index in args.overlay_row_indices):
            raise SystemExit("--overlay-row-indices must be positive one-based positions")
        args.overlay_row_indices = set(args.overlay_row_indices)
    if args.overlay_frame_ids is not None and args.overlay_row_indices is not None:
        raise SystemExit("Use either --overlay-frame-ids or --overlay-row-indices, not both")
    targets = discover_targets(args)
    if not targets:
        raise SystemExit("No frames matched the given filters. Try --list to inspect selection.")

    total_frames = sum(len(t["frames"]) for t in targets)
    print(f"Selected {len(targets)} camera(s), {total_frames} frame(s):", flush=True)
    for t in targets:
        print(f"  {t['group']}/{t['delivery_id']}/{t['camera_id']}: {len(t['frames'])} frames", flush=True)
    if args.list:
        return 0

    device = resolve_device(args.device, allow_cpu=args.allow_cpu)
    run_id = args.run_id or f"p1-rtmpose-{datetime.now().strftime('%Y%m%dT%H%M%SZ')}"
    if args.benchmark_only:
        args.resume = False
        args.overlay = False
    if args.smoke_overlay and not args.benchmark_only and "smoke" in run_id.lower() and not args.overlay:
        args.overlay = True
        print("Smoke run detected; enabling visualizations.", flush=True)
    run_dir = abspath(args.run_dir) if args.run_dir else (ROOT / "data" / "derived" / "runs" / run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    def inference_dir(delivery: str) -> Path:
        """Per-delivery P1 stage dir: <run>/<delivery>/00_inference/ (sits beside 01..06)."""
        return run_dir / delivery / "00_inference"

    print(f"Loading detector + RTMPose on {device} ...", flush=True)
    detector, pose_model, inference_detector, pose_config, pose_checkpoint = build_models(args, device)
    pose_pipeline = build_pose_pipeline(pose_model)
    meta_get = getattr(pose_model.dataset_meta, "get", None)
    source_skeleton = (meta_get("dataset_name") if callable(meta_get) else None) or "coco_wholebody_133"
    pose_config_lower = pose_config.lower()
    if "wholebody" in pose_config_lower:
        source_skeleton = "coco_wholebody_133"
    elif "halpe" in pose_config_lower:
        # Halpe-26 configs (e.g. RTMPose-x body8-halpe26) emit 26 keypoints whose
        # first 17 are COCO-17; the keypoint mapping slices them back to COCO-17.
        source_skeleton = "halpe26"
    elif any(token in pose_config_lower for token in ("coco", "body8", "body7")):
        source_skeleton = "coco_17"
    coco17_indices = coco17_source_indices(source_skeleton)
    visualizer = build_visualizer(pose_model, args.kpt_thr) if args.overlay else None

    summary = {
        "schema_version": "cricket_phase1_run/v2",
        "run_id": run_id, "model_id": args.model_id, "device": device,
        "prediction_schema_version": P1_SCHEMA_VERSION,
        "pose_config": rel(pose_config), "pose_checkpoint": rel(pose_checkpoint),
        "det_config": rel(abspath(args.det_config)), "det_checkpoint": rel(abspath(args.det_checkpoint)),
        "skeleton": source_skeleton, "started_at": utc_now(),
        "inference_mode": "topdown_detector_pose",
        "input_mode": "opencv_bgr_mmdet_mmpose_batch",
        "det_batch_size": args.det_batch_size, "pose_batch_size": args.pose_batch_size,
        "io_workers": args.io_workers, "cv2_threads": args.cv2_threads,
        "prefetch_batches": args.prefetch_batches,
        "benchmark_only": args.benchmark_only,
        "sync_cuda_timing": args.sync_cuda_timing,
        "overlay_enabled": args.overlay, "overlay_limit_per_camera": args.overlay_limit,
        "overlay_frame_ids": sorted(args.overlay_frame_ids) if args.overlay_frame_ids is not None else None,
        "overlay_row_indices": sorted(args.overlay_row_indices) if args.overlay_row_indices is not None else None,
        "visualizations": rel(run_dir / "visualizations") if args.overlay else None,
        "run_dir": rel(run_dir),
        "cameras": [], "frames_processed": 0, "frames_skipped": 0,
        "total_people": 0, "failed_frames": 0,
    }

    timings: dict[str, float] = {
        "decode_seconds": 0.0,
        "detect_seconds": 0.0,
        "pose_seconds": 0.0,
        "write_seconds": 0.0,
        "overlay_seconds": 0.0,
    }
    failures: list[dict[str, Any]] = []
    overlay_failures: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    progress = build_progress_bar(total_frames, args.show_progress)
    # One decode thread-pool for the whole run (not one per batch): io-worker threads
    # read + decode frames ahead of the GPU. cv2 is capped to --cv2-threads so the pool
    # does not oversubscribe the CPU.
    decode_pool = ThreadPoolExecutor(max_workers=max(1, args.io_workers))
    for t in targets:
        cam_id, delivery_id = t["camera_id"], t["delivery_id"]
        camera_name = f"{t['group']}/{delivery_id}/{cam_id}"
        out_jsonl = inference_dir(delivery_id) / "predictions" / f"{t['group']}__{delivery_id}__{cam_id}.jsonl"
        if not args.benchmark_only:
            out_jsonl.parent.mkdir(parents=True, exist_ok=True)
        done: set[str] = set()
        existing_people = 0
        if args.resume and not args.benchmark_only and out_jsonl.exists():
            with out_jsonl.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                        done.add(row["frame_name"])
                        existing_people += len(row.get("players", []))
                    except Exception:
                        pass
        mode = "a" if (args.resume and out_jsonl.exists()) else "w"

        cam_processed = cam_skipped = cam_failed = overlays = 0
        cam_people = existing_people
        output_context = nullcontext(None) if args.benchmark_only else out_jsonl.open(mode, encoding="utf-8")
        with output_context as handle:
            # Resolve resume-skips up front, then stream the remaining frames through
            # the prefetch pipeline so cold-disk read + decode (CPU/io-workers) overlaps
            # detection + pose (GPU) instead of blocking it.
            pending_all: list[tuple[int, Path]] = []
            for idx, frame_path in enumerate(t["frames"]):
                if frame_path.name in done:
                    cam_skipped += 1
                    progress_step(
                        progress,
                        camera=camera_name,
                        cam_processed=cam_processed,
                        cam_skipped=cam_skipped,
                        cam_failed=cam_failed,
                        cam_people=cam_people,
                        overlays=overlays,
                    )
                else:
                    pending_all.append((idx, frame_path))

            for loaded, load_failures in prefetch_decoded_batches(
                pending_all,
                decode_pool,
                args.det_batch_size,
                args.prefetch_batches,
                timings,
            ):
                for _, frame_path, exc in load_failures:
                    cam_failed += 1
                    append_capped(failures, {
                        "group": t["group"],
                        "delivery_id": delivery_id,
                        "camera_id": cam_id,
                        "frame_name": frame_path.name,
                        "stage": "decode",
                        "error": str(exc),
                    })
                    progress_write(progress, f"  ! {camera_name}/{frame_path.name}: {exc}")
                    progress_step(
                        progress,
                        camera=camera_name,
                        cam_processed=cam_processed,
                        cam_skipped=cam_skipped,
                        cam_failed=cam_failed,
                        cam_people=cam_people,
                        overlays=overlays,
                    )

                if not loaded:
                    continue

                try:
                    batch_boxes = timed_call(
                        timings,
                        "detect_seconds",
                        device,
                        args.sync_cuda_timing,
                        detect_person_boxes_batch,
                        detector,
                        inference_detector,
                        loaded,
                        args,
                    )
                    for entry, boxes in zip(loaded, batch_boxes):
                        entry["boxes"] = boxes
                    batch_results = timed_call(
                        timings,
                        "pose_seconds",
                        device,
                        args.sync_cuda_timing,
                        inference_topdown_batch,
                        pose_model,
                        pose_pipeline,
                        loaded,
                        args.pose_batch_size,
                    )
                except Exception as exc:  # noqa: BLE001
                    for entry in loaded:
                        cam_failed += 1
                        append_capped(failures, {
                            "group": t["group"],
                            "delivery_id": delivery_id,
                            "camera_id": cam_id,
                            "frame_name": entry["frame_path"].name,
                            "stage": "inference",
                            "error": str(exc),
                        })
                        progress_write(progress, f"  ! {camera_name}/{entry['frame_path'].name}: {exc}")
                        progress_step(
                            progress,
                            camera=camera_name,
                            cam_processed=cam_processed,
                            cam_skipped=cam_skipped,
                            cam_failed=cam_failed,
                            cam_people=cam_people,
                            overlays=overlays,
                        )
                    continue

                for entry, results in zip(loaded, batch_results):
                    frame_path = entry["frame_path"]
                    players = player_records(results, source_skeleton, entry["width"], entry["height"], coco17_indices)

                    record = {
                        "schema_version": P1_SCHEMA_VERSION,
                        "camera_id": cam_id,
                        "delivery_id": delivery_id,
                        "capture_group": t["group"],
                        "frame_index": parse_frame_id(frame_path),
                        "frame_name": frame_path.name,
                        "match_id": match_id_from_delivery(delivery_id),
                        "metadata": {
                            "model_id": args.model_id, "run_id": run_id, "device": device,
                            "capture_group": t["group"],
                            "image_size_px": [entry["width"], entry["height"]],
                            "inference_mode": "topdown_detector_pose",
                            "input_mode": "opencv_bgr_mmdet_mmpose_batch",
                            "det_batch_size_requested": args.det_batch_size,
                            "det_batch_size_effective": args.det_batch_size,
                            "pose_batch_size_requested": args.pose_batch_size,
                            "pose_batch_size_effective": args.pose_batch_size,
                            "io_workers": args.io_workers,
                            "detector": "rtmdet_m_person", "bbox_thr": args.bbox_thr,
                            "nms_thr": args.nms_thr,
                            "model_specific": {
                                "rtmpose": {
                                    "source_skeleton": source_skeleton,
                                    "output_skeleton": source_skeleton,
                                    "pose_config": rel(pose_config),
                                    "pose_checkpoint": rel(pose_checkpoint),
                                    "det_config": rel(abspath(args.det_config)),
                                    "det_checkpoint": rel(abspath(args.det_checkpoint)),
                                }
                            },
                        },
                        "players": players,
                    }
                    validate_group1_frame(record, final_handoff=False)
                    if handle is not None:
                        start = time.perf_counter()
                        handle.write(json.dumps(record) + "\n")
                        timings["write_seconds"] += time.perf_counter() - start
                    cam_people += len(players)
                    cam_processed += 1

                    if visualizer is not None and should_render_overlay(args, entry, overlays):
                        vis_path = run_dir / "visualizations" / t["group"] / delivery_id / cam_id / f"{frame_path.stem}.jpg"
                        try:
                            start = time.perf_counter()
                            render_overlay(visualizer, str(frame_path), results, vis_path, args.kpt_thr)
                            timings["overlay_seconds"] += time.perf_counter() - start
                            overlays += 1
                        except Exception as exc:  # noqa: BLE001
                            append_capped(overlay_failures, {
                                "group": t["group"],
                                "delivery_id": delivery_id,
                                "camera_id": cam_id,
                                "frame_name": frame_path.name,
                                "stage": "overlay",
                                "error": str(exc),
                            })
                            progress_write(progress, f"  ! overlay {camera_name}/{frame_path.name}: {exc}")

                    progress_step(
                        progress,
                        camera=camera_name,
                        cam_processed=cam_processed,
                        cam_skipped=cam_skipped,
                        cam_failed=cam_failed,
                        cam_people=cam_people,
                        overlays=overlays,
                    )

                    if progress is None and cam_processed % 100 == 0:
                        print(f"  {t['group']}/{delivery_id}/{cam_id}: {cam_processed} frames", flush=True)

        progress_write(progress, f"+ {t['group']}/{delivery_id}/{cam_id}: processed={cam_processed} "
                       f"skipped={cam_skipped} people={cam_people} failed={cam_failed} -> {out_jsonl.name}")
        summary["cameras"].append({
            "group": t["group"], "delivery_id": delivery_id, "camera_id": cam_id,
            "frames_processed": cam_processed, "frames_skipped": cam_skipped,
            "people": cam_people, "failed": cam_failed, "overlays": overlays,
            "predictions": rel(out_jsonl),
        })
        summary["frames_processed"] += cam_processed
        summary["frames_skipped"] += cam_skipped
        summary["total_people"] += cam_people
        summary["failed_frames"] += cam_failed

    decode_pool.shutdown(wait=True)
    if progress is not None:
        progress.close()

    elapsed = time.perf_counter() - t0
    summary["finished_at"] = utc_now()
    summary["elapsed_seconds"] = round(elapsed, 2)
    summary["fps"] = round(summary["frames_processed"] / elapsed, 2) if elapsed > 0 else None
    summary["timings"] = {key: round(value, 3) for key, value in timings.items()}
    summary["timings"]["other_seconds"] = round(
        elapsed - sum(timings.values()),
        3,
    )
    summary["failures"] = failures[:500]
    summary["overlay_failures"] = overlay_failures[:500]
    summary["summary"] = {
        "delivery_count": len({camera["delivery_id"] for camera in summary["cameras"]}),
        "camera_count": len(summary["cameras"]),
        "records_written": summary["frames_processed"] + summary["frames_skipped"],
        "records_written_this_run": summary["frames_processed"],
        "records_reused": summary["frames_skipped"],
        "total_players_detected": summary["total_people"],
        "failed_frames": summary["failed_frames"],
        "wall_clock_s": elapsed,
        "fps_overall": summary["fps"],
        "timings": summary["timings"],
        "status": "pass" if summary["failed_frames"] == 0 else "partial",
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    p1_metrics = {
        "schema_version": "cricket_phase1_metrics/v2",
        "prediction_schema_version": P1_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": summary["finished_at"],
        "delivery_ids": sorted({camera["delivery_id"] for camera in summary["cameras"]}),
        "match_ids": sorted({match_id_from_delivery(camera["delivery_id"]) for camera in summary["cameras"]}),
        "model_id": args.model_id,
        "device": device,
        "inference_mode": summary["inference_mode"],
        "input_mode": summary["input_mode"],
        "skeleton": source_skeleton,
        "detector": "rtmdet_m_person",
        "det_batch_size": args.det_batch_size,
        "pose_batch_size": args.pose_batch_size,
        "io_workers": args.io_workers,
        "bbox_thr": args.bbox_thr,
        "nms_thr": args.nms_thr,
        "kpt_thr": args.kpt_thr,
        "git_sha": git_sha(ROOT),
        "run_dir": rel(run_dir),
        "visualizations": summary["visualizations"],
        "summary": {
            "delivery_count": len({camera["delivery_id"] for camera in summary["cameras"]}),
            "camera_count": len(summary["cameras"]),
            "records_written": summary["frames_processed"],
            "records_written_this_run": summary["frames_processed"],
            "records_reused": summary["frames_skipped"],
            "total_players_detected": summary["total_people"],
            "failed_frames": summary["failed_frames"],
            "wall_clock_s": elapsed,
            "fps_overall": summary["fps"],
            "timings": summary["timings"],
            "status": "pass" if summary["failed_frames"] == 0 else "partial",
        },
        "per_camera": {
            camera_metric_key(camera["group"], camera["delivery_id"], camera["camera_id"]): {
                "capture_group": camera["group"],
                "delivery_id": camera["delivery_id"],
                "camera_id": camera["camera_id"],
                "prediction_jsonl": camera["predictions"],
                "frames_selected": camera["frames_processed"] + camera["frames_skipped"] + camera["failed"],
                "records_written": camera["frames_processed"] + camera["frames_skipped"],
                "records_written_this_run": camera["frames_processed"],
                "records_reused": camera["frames_skipped"],
                "total_players_detected": camera["people"],
                "failed_frames": camera["failed"],
                "overlays": camera["overlays"],
            }
            for camera in summary["cameras"]
        },
        "failures": failures[:500],
        "overlay_failures": overlay_failures[:500],
    }
    (run_dir / "p1_metrics.json").write_text(json.dumps(p1_metrics, indent=2), encoding="utf-8")
    if not args.benchmark_only:
        for delivery_id in sorted({camera["delivery_id"] for camera in summary["cameras"]}):
            delivery_cameras = [
                camera for camera in summary["cameras"]
                if camera["delivery_id"] == delivery_id
            ]
            delivery_failures = [
                failure for failure in failures
                if failure.get("delivery_id") == delivery_id
            ]
            delivery_overlay_failures = [
                failure for failure in overlay_failures
                if failure.get("delivery_id") == delivery_id
            ]
            delivery_dir = inference_dir(delivery_id)
            delivery_dir.mkdir(parents=True, exist_ok=True)
            delivery_metrics = {
                "schema_version": "cricket_phase1_metrics/v2",
                "prediction_schema_version": P1_SCHEMA_VERSION,
                "run_id": run_id,
                "created_at": summary["finished_at"],
                "delivery_id": delivery_id,
                "match_id": match_id_from_delivery(delivery_id),
                "model_id": args.model_id,
                "device": device,
                "inference_mode": summary["inference_mode"],
                "input_mode": summary["input_mode"],
                "skeleton": source_skeleton,
                "detector": "rtmdet_m_person",
                "det_batch_size": args.det_batch_size,
                "pose_batch_size": args.pose_batch_size,
                "io_workers": args.io_workers,
                "bbox_thr": args.bbox_thr,
                "nms_thr": args.nms_thr,
                "kpt_thr": args.kpt_thr,
                "prediction_dir": rel(inference_dir(delivery_id) / "predictions"),
                "visualizations": summary["visualizations"],
                "summary": {
                    "camera_count": len(delivery_cameras),
                    "records_written": sum(camera["frames_processed"] + camera["frames_skipped"] for camera in delivery_cameras),
                    "records_written_this_run": sum(camera["frames_processed"] for camera in delivery_cameras),
                    "records_reused": sum(camera["frames_skipped"] for camera in delivery_cameras),
                    "total_players_detected": sum(camera["people"] for camera in delivery_cameras),
                    "failed_frames": sum(camera["failed"] for camera in delivery_cameras),
                    "status": "pass" if not delivery_failures else "partial",
                },
                "per_camera": {
                    camera_metric_key(camera["group"], camera["delivery_id"], camera["camera_id"]): {
                        "capture_group": camera["group"],
                        "delivery_id": camera["delivery_id"],
                        "camera_id": camera["camera_id"],
                        "prediction_jsonl": camera["predictions"],
                        "frames_selected": camera["frames_processed"] + camera["frames_skipped"] + camera["failed"],
                        "records_written": camera["frames_processed"] + camera["frames_skipped"],
                        "records_written_this_run": camera["frames_processed"],
                        "records_reused": camera["frames_skipped"],
                        "total_players_detected": camera["people"],
                        "failed_frames": camera["failed"],
                        "overlays": camera["overlays"],
                    }
                    for camera in delivery_cameras
                },
                "failures": delivery_failures[:200],
                "overlay_failures": delivery_overlay_failures[:200],
            }
            delivery_manifest = {
                "schema_version": "cricket_phase1_delivery_run/v1",
                "prediction_schema_version": P1_SCHEMA_VERSION,
                "run_id": run_id,
                "created_at": summary["finished_at"],
                "drive_root": rel(abspath(args.drive_root)),
                "delivery_id": delivery_id,
                "model_id": args.model_id,
                "device": device,
                "inference_mode": summary["inference_mode"],
                "input_mode": summary["input_mode"],
                "det_batch_size": args.det_batch_size,
                "pose_batch_size": args.pose_batch_size,
                "io_workers": args.io_workers,
                "prediction_dir": rel(inference_dir(delivery_id) / "predictions"),
                "visualizations": summary["visualizations"],
                "summary": delivery_metrics["summary"],
            }
            (delivery_dir / "p1_metrics.json").write_text(
                json.dumps(delivery_metrics, indent=2),
                encoding="utf-8",
            )
            (delivery_dir / "run_manifest.json").write_text(
                json.dumps(delivery_manifest, indent=2),
                encoding="utf-8",
            )

    print(
        f"\nDone in {elapsed:.1f}s | processed={summary['frames_processed']} "
        f"skipped={summary['frames_skipped']} people={summary['total_people']} "
        f"failed={summary['failed_frames']} | {summary['fps']} fps\nRun dir: {run_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

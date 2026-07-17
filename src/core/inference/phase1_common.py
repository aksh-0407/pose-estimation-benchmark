"""Shared building blocks for the Phase 1 (2D pose inference) runners.

Both P1 entry points import from this module:

- ``run_phase1_rtmpose_inference.py``: the standard runner for the repo-local
  dataset layout (``<drive-root>/bt_0X/<delivery>/camera<NN>/``).
- ``run_phase1_l40s.py``: the remote-GPU runner for the native capture layout
  (``bt1/bt2/bt3``), with tiled detection, mixed precision, and batch sweeps.

Everything here is runner-agnostic: model construction (RTMDet detector +
RTMPose top-down pose), batched detection and pose inference, frame decode
prefetching, the per-frame prediction record, and small path/time helpers.
The runners keep only their CLI, dataset discovery, run loop, and metrics.

Notes for maintainers:

- ``build_frame_record`` is the single source of the per-frame JSONL schema.
  Key order is part of the byte-level output contract; do not reorder keys.
- The detector presets let a run swap the person detector (RTMDet-m baseline,
  RTMDet-l/x, DINO) while the pose model stays RTMPose; presets other than the
  vendored baseline are downloaded by ``tools/detector_bakeoff/fetch_detectors.py``.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from core.contract import SCHEMA_VERSION as P1_SCHEMA_VERSION
from core.frames import FRAME_RE, camera_label, parse_frame_id, repo_relative

ROOT = Path(__file__).resolve().parents[3]

# Detailed per-frame failure records are only for the manifest dump (the real
# counts live in the per-camera summaries). Cap the retained list so a
# pathological run cannot grow it without bound.
MAX_FAILURE_RECORDS = 2000

MMPOSE_DIR = ROOT / "external" / "mmpose"
DEFAULT_MODEL_CONFIG = ROOT / "configs" / "model_envs.yaml"
DEFAULT_DET_CONFIG = ROOT / "external" / "mmpose" / "demo" / "mmdetection_cfg" / "rtmdet_m_640-8xb32_coco-person.py"
DEFAULT_DET_CHECKPOINT = ROOT / "models" / "rtmdet_m_person" / "weights" / "rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"

# --------------------------------------------------------------------------- #
# Swappable-detector presets. The pose model stays RTMPose (project mandate);
# only the person detector underneath it changes. ``--detector <name>`` selects
# one; the default (None) is the vendored RTMDet-m baseline. All COCO variants
# keep person as category 0 (the obj365-person checkpoint is single-class with
# person = 0, so cat_id = 0 works uniformly).
#
# Presets without explicit paths are fetched with mim into ``models/<dir>/``
# (config .py plus checkpoint .pth) by tools/detector_bakeoff/fetch_detectors.py.
# The resolver globs that directory so an mmdet version bump that renames the
# checkpoint still resolves.
# --------------------------------------------------------------------------- #
DETECTOR_PRESETS: dict[str, dict[str, Any]] = {
    "rtmdet_m": {  # production baseline (person-only obj365 weights, vendored)
        "config": DEFAULT_DET_CONFIG,
        "checkpoint": DEFAULT_DET_CHECKPOINT,
        "cat_id": 0,
    },
    "rtmdet_l": {"dir": ROOT / "models" / "rtmdet_l_coco",
                 "mim": "rtmdet_l_8xb32-300e_coco", "cat_id": 0},
    "rtmdet_x": {"dir": ROOT / "models" / "rtmdet_x_coco",
                 "mim": "rtmdet_x_8xb32-300e_coco", "cat_id": 0},
    "dino": {"dir": ROOT / "models" / "dino_4scale_coco",
             "mim": "dino-4scale_r50_8xb2-12e_coco", "cat_id": 0},
}


def resolve_detector_preset(name: str) -> tuple[Path, Path, int]:
    """Return (det_config, det_checkpoint, cat_id) for a --detector preset name."""
    preset = DETECTOR_PRESETS.get(name)
    if preset is None:
        raise SystemExit(
            f"unknown --detector {name!r}; choices: {sorted(DETECTOR_PRESETS)}")
    if "config" in preset:  # vendored, explicit paths
        return Path(preset["config"]), Path(preset["checkpoint"]), int(preset["cat_id"])
    # mim-fetched: glob the download dir for the config and the weight
    ddir = Path(preset["dir"])
    cfgs = sorted(ddir.glob("*.py"))
    ckpts = sorted(ddir.glob("*.pth"))
    if not cfgs or not ckpts:
        raise SystemExit(
            f"detector preset {name!r} not downloaded (looked in {ddir}). "
            f"Fetch it with:\n"
            f"  python tools/detector_bakeoff/fetch_detectors.py --detectors {name}")
    return cfgs[0], ckpts[0], int(preset["cat_id"])


# --------------------------------------------------------------------------- #
# Small helpers
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


def resolve_model_paths(args) -> tuple[Path, Path]:
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


def match_id_from_delivery(delivery_id: str) -> str:
    head = delivery_id.split("_", 1)[0]
    return head[:-2] if head[-2:].startswith("M") else head


def git_sha(workdir: Path = ROOT) -> str | None:
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


def iter_camera_targets(delivery_dir: Path, camera_filter: set[str] | None,
                        start_index: int, stride: int, frame_limit: int | None):
    """Yield (camera_dir, cam_id, frames) per selected camera of one delivery.

    Shared by both runners' discovery: filters camera dirs, collects frame files
    in frame-id order, and applies the start/stride/limit slice.
    """
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
        frames = frames[start_index:]
        if stride > 1:
            frames = frames[::stride]
        if frame_limit is not None:
            frames = frames[:frame_limit]
        if not frames:
            continue
        yield camera_dir, cam_id, frames


# --------------------------------------------------------------------------- #
# Frame decode and prefetch
# --------------------------------------------------------------------------- #
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

    Disk read plus JPEG decode (CPU) for the next ``depth`` batches run on the
    shared ``executor`` while the caller runs detection/pose on the GPU for the
    current batch. This overlaps the CPU/IO stage with the GPU stage so the GPU
    is not left idle waiting on cold-disk frame reads. ``decode_seconds`` only
    accrues the time actually spent blocked waiting for a future, so a well-fed
    pipeline reports it near zero.
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
        # Submit the next batch's decode BEFORE blocking, so it overlaps the GPU
        # work the caller does with the batch about to be yielded.
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


# --------------------------------------------------------------------------- #
# Models and inference
# --------------------------------------------------------------------------- #
def build_models(args, device: str):
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
    # A `--detector <preset>` selection overrides the raw --det-* paths (and the
    # person cat-id). Default (None) leaves the RTMDet-m baseline untouched.
    detector_name = getattr(args, "detector", None)
    if detector_name:
        cfg, ckpt, cat_id = resolve_detector_preset(detector_name)
        args.det_config, args.det_checkpoint, args.det_cat_id = str(cfg), str(ckpt), cat_id
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


def boxes_from_det_result(det_result, args):
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


def detect_person_boxes(detector, inference_detector, image_path: str, args):
    det_result = inference_detector(detector, image_path)
    return boxes_from_det_result(det_result, args)


def detect_person_boxes_batch(detector, inference_detector, entries: list[dict[str, Any]], args):
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


# --------------------------------------------------------------------------- #
# Skeleton resolution and per-frame records
# --------------------------------------------------------------------------- #
def coco17_source_indices(source_skeleton: str) -> list[int] | None:
    """Indices into the source skeleton that yield COCO-17 order, or None if unmapped."""
    from core.keypoints import load_keypoint_mappings

    mapping = load_keypoint_mappings()["source_to_coco_17"].get(source_skeleton)
    return mapping["source_indices"] if mapping else None


def resolve_skeleton(pose_config: str, pose_model) -> tuple[str, list[int] | None]:
    """Infer the model's native skeleton name from its config path and metadata.

    Returns ``(source_skeleton, coco17_indices)``. The pipeline contract is
    Halpe-26 (COCO-17 in indices 0-16 plus head/neck/hip-mid and six foot
    joints); the production model (RTMPose-x body8-halpe26) emits exactly that.
    """
    meta_get = getattr(pose_model.dataset_meta, "get", None)
    source = (meta_get("dataset_name") if callable(meta_get) else None) or "coco_wholebody_133"
    low = pose_config.lower()
    if "wholebody" in low:
        source = "coco_wholebody_133"
    elif "halpe" in low:
        source = "halpe26"
    elif any(tok in low for tok in ("coco", "body8", "body7")):
        source = "coco_17"
    return source, coco17_source_indices(source)


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
    """Slice a native pose down to COCO-17 order.

    Currently unused by the runners: since the Halpe-26 migration the record
    carries the model's native keypoints unsliced (``player_records``). Kept as
    the documented COCO-17 slicing reference; see docs/reference/legacy-and-dead-code.md.
    """
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
        # joint the model emits (for RTMPose-x that is all 26: COCO-17 in indices
        # 0-16 plus head/neck/hip-mid and the 6 foot joints).
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


def build_frame_record(
    *,
    camera_id: str,
    delivery_id: str,
    capture_group: str,
    frame_path: Path,
    width: int,
    height: int,
    players: list[dict[str, Any]],
    model_id: str,
    run_id: str,
    device: str,
    det_batch_size: int,
    pose_batch_size: int,
    io_workers: int,
    detector_label: str,
    bbox_thr: float,
    nms_thr: float,
    source_skeleton: str,
    output_skeleton: str,
    pose_config_rel: str,
    pose_checkpoint_rel: str,
    det_config_rel: str,
    det_checkpoint_rel: str,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one per-frame prediction record (the P1 JSONL line).

    This is the single definition of the P1 output schema for both runners.
    ``extra_metadata`` entries are inserted after ``nms_thr`` (the L40S runner
    adds its tiled-detection fields there). Key order is part of the byte-level
    output contract; do not reorder.
    """
    metadata: dict[str, Any] = {
        "model_id": model_id, "run_id": run_id, "device": device,
        "capture_group": capture_group,
        "image_size_px": [width, height],
        "inference_mode": "topdown_detector_pose",
        "input_mode": "opencv_bgr_mmdet_mmpose_batch",
        "det_batch_size_requested": det_batch_size,
        "det_batch_size_effective": det_batch_size,
        "pose_batch_size_requested": pose_batch_size,
        "pose_batch_size_effective": pose_batch_size,
        "io_workers": io_workers,
        "detector": detector_label,
        "bbox_thr": bbox_thr,
        "nms_thr": nms_thr,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    metadata["model_specific"] = {
        "rtmpose": {
            "source_skeleton": source_skeleton,
            "output_skeleton": output_skeleton,
            "pose_config": pose_config_rel,
            "pose_checkpoint": pose_checkpoint_rel,
            "det_config": det_config_rel,
            "det_checkpoint": det_checkpoint_rel,
        }
    }
    return {
        "schema_version": P1_SCHEMA_VERSION,
        "camera_id": camera_id,
        "delivery_id": delivery_id,
        "capture_group": capture_group,
        "frame_index": parse_frame_id(frame_path),
        "frame_name": frame_path.name,
        "match_id": match_id_from_delivery(delivery_id),
        "metadata": metadata,
        "players": players,
    }


def write_record(handle, record: dict[str, Any], timings: dict[str, float]) -> None:
    """Serialize one record to an open JSONL handle, accruing write time."""
    start = time.perf_counter()
    handle.write(json.dumps(record) + "\n")
    timings["write_seconds"] += time.perf_counter() - start


def read_resume_state(out_jsonl: Path, warn_label: str = "") -> tuple[set[str], int]:
    """Return (frame names already present, people already counted) for --resume.

    A line that fails to parse is counted and reported instead of silently
    shrinking the resume set (a corrupt line would otherwise cause silent
    recompute of already-done frames).
    """
    done: set[str] = set()
    existing_people = 0
    corrupt = 0
    with out_jsonl.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
                done.add(row["frame_name"])
                existing_people += len(row.get("players", []))
            except Exception:
                corrupt += 1
    if corrupt:
        print(f"WARN: {warn_label or out_jsonl.name}: {corrupt} corrupt resume line(s) "
              "ignored; those frames will be recomputed", flush=True)
    return done, existing_people

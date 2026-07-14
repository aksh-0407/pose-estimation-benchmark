#!/usr/bin/env python3
"""Phase 1 (RTMPose-L / Body8) 2D-pose inference for the L40S benchmark machine.

This is a self-contained runner for the *new* capture machine whose dataset lives at
``/home/ubuntu/pose_data/{bt1,bt2,bt3}/<delivery>/camera<NN>/frame_camera<NN>_*.jpg``
(134,400 frames = 3 groups x 32 deliveries x {2,3,2} cameras x 600). The stock runner
(``run_phase1_rtmpose_inference.py``) hard-requires ``bt_01/bt_02/bt_03`` under
``<drive-root>/dataset/`` and writes under ``benchmarks/runs/``; this file discovers the
``bt1/bt2/bt3`` layout natively, is tuned for the L40S (46 GB, Ada), and writes to a caller
chosen output dir. It does NOT copy/symlink data and changes no existing file.

It reuses the validated mmdet/mmpose building blocks and the cricket P1 schema from the stock
runner (imported, not reimplemented) so the per-frame JSONL is byte-compatible with downstream
P2/P3/P4.

Key fact this design leans on: top-down RTMDet + RTMPose are batch-invariant in eval -- batch
size changes *speed only, never output*. So the ``--sweep`` mode writes nothing, and the full
run touches every frame exactly once (``--resume`` makes restarts free).

Typical usage on the L40S box (inside the ``cricket-rtmpose-l`` conda env):

    # 0) what would run + preflight
    python scripts/inference/run_phase1_l40s.py --list

    # 1) fine in-process batch sweep (~1-2 min, writes nothing) -> best.json
    python scripts/inference/run_phase1_l40s.py --sweep

    # 2) full resumable run at the winning config
    python scripts/inference/run_phase1_l40s.py \
        --det-batch-size <B> --pose-batch-size <P> --io-workers <W>
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "inference"))

# Reuse the proven internals from the stock runner (module import has no main() side effects).
import run_phase1_rtmpose_inference as p1  # noqa: E402
from pose_estimation.cricket.contract import (  # noqa: E402
    SCHEMA_VERSION as P1_SCHEMA_VERSION,
    SKELETON as P1_SKELETON,
    validate_group1_frame,
)
from pose_estimation.cricket.dataset import (  # noqa: E402
    FRAME_RE,
    camera_label,
    parse_frame_id,
)

DEFAULT_POSE_DATA = "/home/ubuntu/pose_data"
DEFAULT_OUTPUT_DIR = "/home/ubuntu/pose-data-inference-output"
GROUP_RE = re.compile(r"^bt_?0*(?P<num>\d+)$", re.IGNORECASE)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    sel = parser.add_argument_group("dataset selection")
    sel.add_argument("--pose-data", default=DEFAULT_POSE_DATA,
                     help=f"Root holding bt1/bt2/bt3 (default: {DEFAULT_POSE_DATA})")
    sel.add_argument("--groups", nargs="+", default=None,
                     help="Groups to include; accepts bt1 / bt_01 / 1 (default: all)")
    sel.add_argument("--deliveries", nargs="+", default=None,
                     help="Delivery IDs; exact or substring match (default: all)")
    sel.add_argument("--cameras", nargs="+", default=None,
                     help="Cameras; accepts camera01 / cam_01 / 01 / 1 (default: all)")
    sel.add_argument("--frame-limit", type=int, default=None,
                     help="Max frames per camera after start/stride (default: all)")
    sel.add_argument("--start-index", type=int, default=0, help="Skip N frames per camera first")
    sel.add_argument("--stride", type=int, default=1, help="Take every Nth frame (default: 1)")
    sel.add_argument("--list", action="store_true",
                     help="List cameras/frames that would run, then exit (no GPU needed)")

    mdl = parser.add_argument_group("model")
    mdl.add_argument("--model-id", default="rtmpose_x_body8",
                     help="Key in configs/model_envs.yaml (default: rtmpose_x_body8, Halpe-26)")
    mdl.add_argument("--model-config", default=str(p1.DEFAULT_MODEL_CONFIG))
    mdl.add_argument("--pose-config", default=None, help="Override pose config path")
    mdl.add_argument("--pose-checkpoint", default=None, help="Override pose checkpoint path")
    mdl.add_argument("--det-config", default=str(p1.DEFAULT_DET_CONFIG))
    mdl.add_argument("--det-checkpoint", default=str(p1.DEFAULT_DET_CHECKPOINT))
    mdl.add_argument("--det-cat-id", type=int, default=0, help="Detector person category id")
    mdl.add_argument("--bbox-thr", type=float, default=0.3, help="Min detection score")
    mdl.add_argument("--nms-thr", type=float, default=0.3, help="Box NMS IoU threshold")
    mdl.add_argument("--max-people", type=int, default=None, help="Keep top-N boxes per frame")
    # Wave-5 tiled (SAHI-style) detection: overlapping tiles + a full-frame pass,
    # merged with NMS + containment suppression. Keeps people at RTMDet's trained
    # object scale, recovering the small/distant band (bake-off: +0.8..+6 boxes/frame,
    # zero lost boxes vs the plain 640 pass). Pose stage unchanged.
    mdl.add_argument("--tiled-det", action="store_true",
                     help="Tile-based detection for small/distant-player recall")
    mdl.add_argument("--tile-cols", type=int, default=4)
    mdl.add_argument("--tile-rows", type=int, default=2)
    mdl.add_argument("--tile-overlap", type=float, default=0.25)
    mdl.add_argument("--no-tiled-fast", dest="tiled_fast", action="store_false",
                     help="Disable the fast tiled path (crop prep in prefetch workers + "
                          "direct data_preprocessor/predict). Fast path is parity-checked "
                          "against the generic path (W5-PERF).")
    parser.set_defaults(tiled_fast=True)

    rt = parser.add_argument_group("runtime")
    rt.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                    help=f"Output root (default: {DEFAULT_OUTPUT_DIR})")
    rt.add_argument("--device", default="cuda:0", help="cuda:0 / cpu (default: cuda:0)")
    rt.add_argument("--allow-cpu", action="store_true", help="Permit CPU when CUDA is unavailable")
    rt.add_argument("--run-id", default=None, help="Run identifier (default: auto timestamp)")
    rt.add_argument("--det-batch-size", type=int, default=24, help="Frames per detector call")
    rt.add_argument("--pose-batch-size", type=int, default=256, help="Crops per RTMPose call")
    rt.add_argument("--io-workers", type=int, default=12, help="Decode prefetch threads")
    rt.add_argument("--cv2-threads", type=int, default=2,
                    help="OpenCV internal threads per decode call (default: 2). Kept low so "
                         "io-workers x cv2 threads does not oversubscribe the CPU.")
    rt.add_argument("--prefetch-batches", type=int, default=3,
                    help="Detector batches to read+decode ahead on the io-workers while the "
                         "GPU runs the current batch (default: 3). Overlaps cold-disk reads "
                         "with GPU compute so the GPU stays fed.")
    rt.add_argument("--no-resume", dest="resume", action="store_false",
                    help="Recompute frames already present in the camera JSONL")
    rt.add_argument("--no-progress", dest="show_progress", action="store_false",
                    help="Disable the tqdm progress bar")
    rt.add_argument("--no-perf", dest="perf", action="store_false",
                    help="Disable cudnn.benchmark + TF32 (on by default for CUDA)")
    rt.add_argument("--no-amp", dest="amp", action="store_false",
                    help="Disable fp16 autocast for detector+pose forwards (on by default; "
                         "W5-PERF: verified box/keypoint parity within tolerance on L40S)")
    rt.add_argument("--show-torch-warnings", action="store_true")
    rt.set_defaults(resume=True, show_progress=True, perf=True, amp=True)

    sw = parser.add_argument_group("sweep (batch autotune, writes nothing)")
    sw.add_argument("--sweep", action="store_true",
                    help="Run an in-process det/pose batch sweep and write best.json, then exit")
    sw.add_argument("--grid", action="store_true",
                    help="Rigorous mode: measure REAL end-to-end pipeline FPS for every "
                         "det x pose combo (like the old tuner), ranked. Use for the real pick.")
    sw.add_argument("--sweep-delivery", default=None,
                    help="Delivery to benchmark on (default: first discovered)")
    sw.add_argument("--sweep-frames", type=int, default=128,
                    help="Frames decoded once into RAM for the sweep (default: 128)")
    sw.add_argument("--det-batches", nargs="+", type=int, default=[8, 16, 24, 32, 48])
    sw.add_argument("--pose-batches", nargs="+", type=int,
                    default=[128, 192, 256, 320, 384, 512, 768])
    sw.add_argument("--io-workers-probe", nargs="+", type=int, default=None,
                    help="io_workers values to probe (default: 4 8 16 nproc)")
    sw.add_argument("--repeats", type=int, default=3,
                    help="Timed repeats per config; first is warmup (default: 3)")

    return parser.parse_args()


# --------------------------------------------------------------------------- #
# discovery (native bt1/bt2/bt3 layout)
# --------------------------------------------------------------------------- #
def group_label(name: str) -> str | None:
    """'bt1' / 'bt_01' / 'BT3' -> canonical 'bt_0N'; None if not a group dir."""
    match = GROUP_RE.match(name)
    if not match:
        return None
    return f"bt_{int(match.group('num')):02d}"


def normalize_group_filters(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    out: set[str] = set()
    for value in values:
        label = group_label(value) or group_label(f"bt{value}")
        if label:
            out.add(label)
    return out or None


def discover_targets(args: argparse.Namespace) -> list[dict[str, Any]]:
    """One record per selected camera: paths + filtered frame list (native layout)."""
    pose_root = Path(args.pose_data).expanduser()
    if not pose_root.is_dir():
        raise SystemExit(f"pose-data root not found: {pose_root}")

    group_filter = normalize_group_filters(args.groups)
    camera_filter = p1.normalize_camera_filters(args.cameras)
    delivery_filter = args.deliveries

    targets: list[dict[str, Any]] = []
    for group_dir in sorted(p for p in pose_root.iterdir() if p.is_dir()):
        label = group_label(group_dir.name)
        if label is None:
            continue
        if group_filter and label not in group_filter:
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
                    "group": label,
                    "source_group": group_dir.name,
                    "delivery_id": delivery_dir.name,
                    "camera_dir": camera_dir,
                    "camera_id": cam_id,
                    "camera_number": camera_dir.name.replace("camera", ""),
                    "frames": frames,
                })
    return targets


# --------------------------------------------------------------------------- #
# perf / preflight / skeleton
# --------------------------------------------------------------------------- #
def apply_perf(device: str, enabled: bool) -> None:
    if not enabled or not device.startswith("cuda"):
        return
    import torch

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("perf: cudnn.benchmark=on, TF32=on", flush=True)


def preflight(args: argparse.Namespace, device: str) -> None:
    """Fail fast with actionable errors before loading anything heavy."""
    # model + detector assets resolve and exist
    pose_config, pose_checkpoint = p1.resolve_model_paths(args)
    checks = [
        ("pose config", p1.abspath(pose_config)),
        ("pose checkpoint", p1.abspath(pose_checkpoint)),
        ("detector config", p1.abspath(args.det_config)),
        ("detector checkpoint", p1.abspath(args.det_checkpoint)),
    ]
    for label, path in checks:
        if not path.exists():
            raise SystemExit(
                f"missing {label}: {path}\n"
                f"Fetch assets with:\n"
                f"  python3 scripts/setup/setup_model_envs.py --models {args.model_id} "
                f"--force-install --download-assets"
            )

    # GPU
    if device.startswith("cuda"):
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "torch is not importable -- activate the model env "
                "(e.g. `conda activate cricket-rtmpose-l`)."
            ) from exc
        if not torch.cuda.is_available():
            raise SystemExit(
                "CUDA requested but torch.cuda.is_available() is false. "
                "Use --device cpu --allow-cpu to override (slow)."
            )
        name = torch.cuda.get_device_name(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"gpu: {name} ({total_gb:.0f} GB)", flush=True)

    # host resources
    nproc = _nproc()
    out_parent = Path(args.output_dir).expanduser()
    probe = out_parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free_gb = shutil.disk_usage(probe).free / (1024 ** 3)
    print(f"host: {nproc} cpus, {free_gb:.0f} GB free at {probe}", flush=True)
    if free_gb < 2:
        print("WARN: less than 2 GB free at output location", flush=True)


def _nproc() -> int:
    import os

    return os.cpu_count() or 4


def resolve_skeleton(pose_config: str, pose_model) -> tuple[str, list[int] | None]:
    """Mirror the stock runner's source-skeleton inference exactly."""
    meta_get = getattr(pose_model.dataset_meta, "get", None)
    source = (meta_get("dataset_name") if callable(meta_get) else None) or "coco_wholebody_133"
    low = pose_config.lower()
    if "wholebody" in low:
        source = "coco_wholebody_133"
    elif "halpe" in low:
        # Halpe-26 (e.g. RTMPose-x body8-halpe26): 26 kpts whose first 17 are COCO-17.
        source = "halpe26"
    elif any(tok in low for tok in ("coco", "body8", "body7")):
        source = "coco_17"
    return source, p1.coco17_source_indices(source)


# --------------------------------------------------------------------------- #
# sweep (in-process, single load, writes nothing)
# --------------------------------------------------------------------------- #
def detect_person_boxes_tiled_batch(detector, inference_detector, entries, args):
    """Wave-5 tiled detection: per frame, tile crops + full frame in ONE detector call.

    Returns the same (N, 5) x1y1x2y2score arrays as p1.detect_person_boxes_batch, after
    the SAHI hygiene passes: interior-border clip drop (partial-person fragments),
    cross-tile NMS (args.nms_thr) and IoM containment suppression.
    """
    import numpy as np

    from detector_bakeoff import (
        _drop_tile_clipped,
        nms_xyxy,
        suppress_contained,
        tile_layout,
    )

    if not entries:
        return []
    crops: list[Any] = []
    plan: list[tuple[int, tuple[int, int, int, int] | None]] = []  # (entry idx, tile or None=full)
    layouts = []
    for idx, entry in enumerate(entries):
        img = entry["img"]
        h, w = img.shape[:2]
        tiles = tile_layout(w, h, (args.tile_cols, args.tile_rows), args.tile_overlap)
        layouts.append((w, h))
        for tile in tiles:
            x, y, tw, th = tile
            crops.append(img[y:y + th, x:x + tw])
            plan.append((idx, tile))
        crops.append(img)
        plan.append((idx, None))
    det_results = inference_detector(detector, crops)
    if not isinstance(det_results, list):
        det_results = [det_results]
    per_entry: list[list[np.ndarray]] = [[] for _ in entries]
    for result, (idx, tile) in zip(det_results, plan):
        pred = result.pred_instances.cpu().numpy()
        keep = np.logical_and(pred.labels == args.det_cat_id, pred.scores > args.bbox_thr)
        boxes = np.concatenate((pred.bboxes[keep], pred.scores[keep, None]), axis=1)
        if tile is not None:
            boxes = _drop_tile_clipped(boxes, tile, layouts[idx])
            if boxes.shape[0]:
                boxes = boxes.copy()
                boxes[:, [0, 2]] += tile[0]
                boxes[:, [1, 3]] += tile[1]
        if boxes.shape[0]:
            per_entry[idx].append(boxes)
    out = []
    for idx, chunks in enumerate(per_entry):
        if not chunks:
            out.append(np.zeros((0, 5), dtype=np.float32))
            continue
        merged = nms_xyxy(np.concatenate(chunks), iou_thr=args.nms_thr)
        merged = suppress_contained(merged, iom_thr=0.7)
        if args.max_people is not None and merged.shape[0] > args.max_people:
            merged = merged[np.argsort(merged[:, 4])[::-1][: args.max_people]]
        out.append(merged.astype(np.float32))
    return out


DET_INPUT_LONG_SIDE = 640  # matches the RTMDet test pipeline Resize scale


def make_tiled_loader(args):
    """Prefetch-worker loader: decode + slice tiles + resize crops off the GPU thread.

    Produces entry["tile_crops"] (uint8 BGR, pipeline-scale) and entry["tile_metas"]
    (offset/ori_shape/img_shape/scale_factor per crop; full frame is the last crop),
    so the GPU loop never runs per-crop Python preprocessing (W5-PERF: the generic
    inference_detector pipeline was the tiled-mode bottleneck, not GPU compute).
    """
    import cv2
    import numpy as np

    from detector_bakeoff import tile_layout

    def load(item):
        entry = p1.load_frame_for_batch(item)
        img = entry["img"]
        h, w = img.shape[:2]
        tiles = tile_layout(w, h, (args.tile_cols, args.tile_rows), args.tile_overlap)
        crops, metas = [], []
        for (x, y, tw, th) in tiles + [(0, 0, w, h)]:
            crop = img[y:y + th, x:x + tw]
            scale = DET_INPUT_LONG_SIDE / max(tw, th)
            rw, rh = int(round(tw * scale)), int(round(th * scale))
            resized = cv2.resize(crop, (rw, rh), interpolation=cv2.INTER_LINEAR)
            # Pad bottom/right to /32 like the generic pipeline's Pad transform:
            # RTMDet's FPN needs stride-divisible inputs; extreme aspect crops
            # (cam_07 full frame -> 640x163) otherwise crash the neck concat.
            ph = (rh + 31) // 32 * 32
            pw = (rw + 31) // 32 * 32
            if (ph, pw) != (rh, rw):
                canvas = np.zeros((ph, pw, 3), dtype=resized.dtype)
                canvas[:rh, :rw] = resized
                resized = canvas
            crops.append(resized)
            metas.append({
                "offset": (x, y),
                "tile": (x, y, tw, th),
                "ori_shape": (th, tw),
                "img_shape": (rh, rw),
                "pad_shape": (ph, pw),
                "scale_factor": (rw / tw, rh / th),
            })
        entry["tile_crops"] = crops
        entry["tile_metas"] = metas
        return entry

    return load


def detect_person_boxes_tiled_fast(detector, entries, args):
    """Tiled detection without the generic per-image pipeline: pre-resized crops from
    the prefetch workers go straight through data_preprocessor -> predict (one batched
    forward for all crops of all frames in the batch). Post-processing (person filter,
    interior-clip drop, offset, NMS + IoM containment) matches the generic tiled path.
    """
    import numpy as np
    import torch
    from mmdet.structures import DetDataSample

    from detector_bakeoff import _drop_tile_clipped, nms_xyxy, suppress_contained

    if not entries:
        return []
    inputs, samples, plan = [], [], []
    for idx, entry in enumerate(entries):
        for crop, meta in zip(entry["tile_crops"], entry["tile_metas"]):
            inputs.append(torch.from_numpy(np.ascontiguousarray(crop.transpose(2, 0, 1))))
            samples.append(DetDataSample(metainfo={
                "img_shape": meta["img_shape"],
                "ori_shape": meta["ori_shape"],
                "pad_shape": meta["pad_shape"],
                "scale_factor": meta["scale_factor"],
            }))
            plan.append((idx, meta))
    with torch.no_grad():
        data = detector.data_preprocessor({"inputs": inputs, "data_samples": samples}, False)
        results = detector._run_forward(data, mode="predict")
    per_entry: list[list[np.ndarray]] = [[] for _ in entries]
    frame_shapes = [entry["img"].shape[:2] for entry in entries]
    for result, (idx, meta) in zip(results, plan):
        pred = result.pred_instances.cpu().numpy()
        keep = np.logical_and(pred.labels == args.det_cat_id, pred.scores > args.bbox_thr)
        boxes = np.concatenate((pred.bboxes[keep], pred.scores[keep, None]), axis=1)
        x, y, tw, th = meta["tile"]
        fh, fw = frame_shapes[idx]
        if (tw, th) != (fw, fh):  # tile crop (full frame is exempt from clip-drop)
            boxes = _drop_tile_clipped(boxes, meta["tile"], (fw, fh))
            if boxes.shape[0]:
                boxes = boxes.copy()
                boxes[:, [0, 2]] += x
                boxes[:, [1, 3]] += y
        if boxes.shape[0]:
            per_entry[idx].append(boxes)
    out = []
    for chunks in per_entry:
        if not chunks:
            out.append(np.zeros((0, 5), dtype=np.float32))
            continue
        merged = nms_xyxy(np.concatenate(chunks), iou_thr=args.nms_thr)
        merged = suppress_contained(merged, iom_thr=0.7)
        if args.max_people is not None and merged.shape[0] > args.max_people:
            merged = merged[np.argsort(merged[:, 4])[::-1][: args.max_people]]
        out.append(merged.astype(np.float32))
    return out


def _amp_context(args, device: str):
    """fp16 autocast for inference forwards (W5-PERF). No-op on CPU or --no-amp."""
    if getattr(args, "amp", False) and device.startswith("cuda"):
        import torch

        return torch.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize()


def _median(values: list[float]) -> float:
    import statistics

    return statistics.median(values) if values else float("nan")


def _grid_sweep(args, loaded, n, best_workers, best_decode_fps, best_decode_pf,
                detector, inference_detector, pose_model, pose_pipeline, device) -> int:
    """Measure REAL end-to-end pipeline FPS for every det x pose combo, ranked.

    This mirrors the production run loop (per det-chunk: detect -> pose) exactly, on a
    cached slice, so the winner reflects true interleaved throughput -- the same basis on
    which the laptop's det=32/pose=96 was chosen. Decode is measured once (probe) and added
    analytically, since decode cost is independent of the det/pose batch sizes.
    """
    import itertools

    print(f"\n[grid] end-to-end det x pose on {n} frames "
          f"(decode {best_decode_fps:.0f} f/s @ io_workers={best_workers}), "
          f"repeats={args.repeats} (first is warmup)")
    print(f"  {'det':>4} {'pose':>5} {'gpu f/s':>9} {'e2e f/s':>9} {'peak GB':>8}")
    rows: list[dict[str, Any]] = []
    for det_batch, pose_batch in itertools.product(args.det_batches, args.pose_batches):
        samples: list[float] = []
        peak = 0.0
        for r in range(args.repeats):
            if device.startswith("cuda"):
                import torch
                torch.cuda.reset_peak_memory_stats()
            for entry in loaded:
                entry.pop("boxes", None)
            _sync(device)
            start = time.perf_counter()
            for chunk in p1.chunked(loaded, det_batch):
                boxes = p1.detect_person_boxes_batch(detector, inference_detector, chunk, args)
                for entry, box in zip(chunk, boxes):
                    entry["boxes"] = box
                p1.inference_topdown_batch(pose_model, pose_pipeline, chunk, pose_batch)
            _sync(device)
            dt = time.perf_counter() - start
            if device.startswith("cuda"):
                import torch
                peak = max(peak, torch.cuda.max_memory_allocated() / (1024 ** 3))
            if r > 0 or args.repeats == 1:
                samples.append(dt)
        gpu_pf = _median(samples) / n if n else float("inf")
        e2e_pf = gpu_pf + best_decode_pf
        gpu_fps = 1.0 / gpu_pf if gpu_pf > 0 else float("nan")
        e2e_fps = 1.0 / e2e_pf if e2e_pf > 0 else float("nan")
        rows.append({
            "det_batch": det_batch, "pose_batch": pose_batch,
            "gpu_fps": round(gpu_fps, 1), "e2e_fps": round(e2e_fps, 1),
            "e2e_per_frame_s": round(e2e_pf, 5),
            "peak_gpu_mem_gb": round(peak, 1) if peak else None,
        })
        print(f"  {det_batch:>4} {pose_batch:>5} {gpu_fps:>9.1f} {e2e_fps:>9.1f} "
              f"{(peak if peak else 0):>8.1f}")

    rows.sort(key=lambda x: x["e2e_fps"], reverse=True)
    win = rows[0]
    best = {
        "created_at": p1.utc_now(),
        "mode": "grid_end_to_end",
        "model_id": args.model_id, "device": device,
        "sweep_frames": n, "repeats": args.repeats,
        "det_batch_size": win["det_batch"],
        "pose_batch_size": win["pose_batch"],
        "io_workers": best_workers,
        "decode_fps": round(best_decode_fps, 1),
        "gpu_fps": win["gpu_fps"],
        "e2e_fps": win["e2e_fps"],
        "projected_full_run_minutes": round(134400 * win["e2e_per_frame_s"] / 60.0, 1),
        "peak_gpu_mem_gb": win["peak_gpu_mem_gb"],
        "grid": rows,
    }
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "best.json").write_text(json.dumps(best, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"BEST (end-to-end)  det_batch={win['det_batch']}  pose_batch={win['pose_batch']}  "
          f"io_workers={best_workers}")
    print(f"      {win['e2e_fps']:.1f} f/s e2e  "
          f"(~{best['projected_full_run_minutes']:.0f} min for 134,400 frames)  "
          f"peak {win['peak_gpu_mem_gb']} GB")
    print(f"wrote {out_dir / 'best.json'}")
    print("=" * 60)
    print("\nRun the full job with:")
    print(f"  python scripts/inference/run_phase1_l40s.py "
          f"--det-batch-size {win['det_batch']} --pose-batch-size {win['pose_batch']} "
          f"--io-workers {best_workers}")
    return 0


def run_sweep(args: argparse.Namespace, targets: list[dict[str, Any]], device: str) -> int:
    delivery = args.sweep_delivery or targets[0]["delivery_id"]
    selected = [t for t in targets if t["delivery_id"] == delivery]
    if not selected:
        raise SystemExit(f"sweep delivery not found: {delivery}")
    paths: list[Path] = []
    for t in selected:
        paths.extend(t["frames"])
    paths = paths[: args.sweep_frames]
    items = list(enumerate(paths))
    print(f"\nSweep on delivery {delivery}: {len(items)} frames, "
          f"{len(selected)} cameras, device {device}", flush=True)

    print("Loading detector + RTMPose (once) ...", flush=True)
    detector, pose_model, inference_detector, pose_config, _ = p1.build_models(args, device)
    pose_pipeline = p1.build_pose_pipeline(pose_model)

    # ---- decode probe: pick io_workers ---------------------------------- #
    probe_values = args.io_workers_probe or sorted({4, 8, 16, _nproc()})
    print("\n[decode] io_workers -> frames/s (decode only)")
    decode_rows: list[tuple[int, float]] = []
    best_workers, best_decode_fps, best_decode_pf = probe_values[0], 0.0, float("inf")
    for workers in probe_values:
        samples = []
        for r in range(args.repeats):
            start = time.perf_counter()
            loaded, _ = p1.load_frame_batch(items, workers)
            dt = time.perf_counter() - start
            if r > 0 or args.repeats == 1:
                samples.append(dt)
        med = _median(samples)
        fps = len(items) / med if med > 0 else float("nan")
        decode_rows.append((workers, fps))
        print(f"  workers={workers:<3} {fps:8.1f} f/s")
        if med > 0 and (med / len(items)) < best_decode_pf:
            best_decode_pf = med / len(items)
            best_decode_fps, best_workers = fps, workers

    # decode a clean cached copy once (at best workers) for the GPU sweeps
    loaded, load_failures = p1.load_frame_batch(items, best_workers)
    if load_failures:
        print(f"WARN: {len(load_failures)} frames failed to decode", flush=True)
    n = len(loaded)
    if n == 0:
        raise SystemExit("no frames decoded for sweep")

    if args.grid:
        return _grid_sweep(args, loaded, n, best_workers, best_decode_fps, best_decode_pf,
                           detector, inference_detector, pose_model, pose_pipeline, device)

    # ---- detector sweep -------------------------------------------------- #
    # det and pose are independent sequential stages, so optimise each in 1-D.
    print("\n[detect] det_batch -> frames/s (detection only)")
    best_det, best_det_pf = args.det_batches[0], float("inf")
    det_rows: list[tuple[int, float]] = []
    for det_batch in args.det_batches:
        samples = []
        for r in range(args.repeats):
            _sync(device)
            start = time.perf_counter()
            for chunk in p1.chunked(loaded, det_batch):
                p1.detect_person_boxes_batch(detector, inference_detector, chunk, args)
            _sync(device)
            dt = time.perf_counter() - start
            if r > 0 or args.repeats == 1:
                samples.append(dt)
        med = _median(samples)
        pf = med / n if n else float("inf")
        det_rows.append((det_batch, n / med if med > 0 else float("nan")))
        print(f"  det_batch={det_batch:<4} {n / med if med > 0 else float('nan'):8.1f} f/s")
        if pf < best_det_pf:
            best_det_pf, best_det = pf, det_batch

    # populate boxes once (needed by the pose stage) using the best det batch
    for chunk in p1.chunked(loaded, best_det):
        boxes = p1.detect_person_boxes_batch(detector, inference_detector, chunk, args)
        for entry, box in zip(chunk, boxes):
            entry["boxes"] = box
    total_boxes = sum(len(e.get("boxes", [])) for e in loaded)
    print(f"  ({total_boxes} person boxes across {n} frames)")

    # ---- pose sweep ------------------------------------------------------ #
    print("\n[pose] pose_batch -> frames/s (pose only)")
    best_pose, best_pose_pf = args.pose_batches[0], float("inf")
    pose_rows: list[tuple[int, float]] = []
    for pose_batch in args.pose_batches:
        samples = []
        for r in range(args.repeats):
            _sync(device)
            start = time.perf_counter()
            p1.inference_topdown_batch(pose_model, pose_pipeline, loaded, pose_batch)
            _sync(device)
            dt = time.perf_counter() - start
            if r > 0 or args.repeats == 1:
                samples.append(dt)
        med = _median(samples)
        pf = med / n if n else float("inf")
        pose_rows.append((pose_batch, n / med if med > 0 else float("nan")))
        print(f"  pose_batch={pose_batch:<4} {n / med if med > 0 else float('nan'):8.1f} f/s")
        if pf < best_pose_pf:
            best_pose_pf, best_pose = pf, pose_batch

    # ---- combine + report ----------------------------------------------- #
    total_pf = best_decode_pf + best_det_pf + best_pose_pf
    projected_fps = 1.0 / total_pf if total_pf > 0 else float("nan")
    peak_mem_gb = None
    if device.startswith("cuda"):
        import torch

        peak_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)

    best = {
        "created_at": p1.utc_now(),
        "model_id": args.model_id,
        "device": device,
        "sweep_delivery": delivery,
        "sweep_frames": n,
        "det_batch_size": best_det,
        "pose_batch_size": best_pose,
        "io_workers": best_workers,
        "per_frame_seconds": {
            "decode": round(best_decode_pf, 5),
            "detect": round(best_det_pf, 5),
            "pose": round(best_pose_pf, 5),
            "total": round(total_pf, 5),
        },
        "projected_fps": round(projected_fps, 1),
        "projected_full_run_minutes": round(134400 * total_pf / 60.0, 1),
        "peak_gpu_mem_gb": round(peak_mem_gb, 1) if peak_mem_gb is not None else None,
        "grids": {
            "decode": [{"io_workers": w, "fps": round(f, 1)} for w, f in decode_rows],
            "detect": [{"det_batch": b, "fps": round(f, 1)} for b, f in det_rows],
            "pose": [{"pose_batch": b, "fps": round(f, 1)} for b, f in pose_rows],
        },
    }
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best.json"
    best_path.write_text(json.dumps(best, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"BEST  det_batch={best_det}  pose_batch={best_pose}  io_workers={best_workers}")
    print(f"      projected {projected_fps:.1f} f/s  "
          f"(~{best['projected_full_run_minutes']:.0f} min for 134,400 frames)")
    if peak_mem_gb is not None:
        print(f"      peak GPU mem seen: {peak_mem_gb:.1f} GB")
    print(f"wrote {best_path}")
    print("=" * 60)
    print("\nRun the full job with:")
    print(f"  python scripts/inference/run_phase1_l40s.py "
          f"--det-batch-size {best_det} --pose-batch-size {best_pose} "
          f"--io-workers {best_workers}")
    return 0


# --------------------------------------------------------------------------- #
# run (writes predictions, resume-safe, no overlays)
# --------------------------------------------------------------------------- #
def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return None


def build_progress_bar(total: int, enabled: bool):
    """One bar for the WHOLE run: dataset %, throughput, elapsed<ETA."""
    if not enabled:
        return None
    try:
        from tqdm import tqdm
    except ImportError:
        print("WARN: tqdm not installed; progress bar disabled.", flush=True)
        return None
    return tqdm(
        total=total, desc="P1 total", unit="f", dynamic_ncols=True,
        smoothing=0.05, mininterval=0.5,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} ({percentage:3.0f}%) "
                   "[{elapsed}<{remaining}, {rate_fmt}]{postfix}",
    )


def run_inference(args: argparse.Namespace, targets: list[dict[str, Any]], device: str) -> int:
    run_id = args.run_id or f"p1-l40s-rtmpose-{datetime.now().strftime('%Y%m%dT%H%M%SZ')}"
    out_dir = Path(args.output_dir).expanduser()
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading detector + RTMPose on {device} ...", flush=True)
    detector, pose_model, inference_detector, pose_config, pose_checkpoint = p1.build_models(args, device)
    pose_pipeline = p1.build_pose_pipeline(pose_model)
    source_skeleton, coco17_indices = resolve_skeleton(pose_config, pose_model)

    timings = {"decode_seconds": 0.0, "detect_seconds": 0.0,
               "pose_seconds": 0.0, "write_seconds": 0.0}
    failures: list[dict[str, Any]] = []
    total_frames = sum(len(t["frames"]) for t in targets)
    progress = build_progress_bar(total_frames, args.show_progress)

    cameras_summary: list[dict[str, Any]] = []
    frames_processed = frames_skipped = failed_frames = total_people = seen = 0
    t0 = time.perf_counter()
    print(f"Processing {total_frames} frames across {len(targets)} cameras -> {pred_dir}",
          flush=True)

    # One decode thread-pool for the whole run: io-worker threads read + decode the next
    # --prefetch-batches batches while the GPU runs detection/pose on the current one.
    decode_pool = ThreadPoolExecutor(max_workers=max(1, args.io_workers))
    for t in targets:
        cam_id, delivery_id, group = t["camera_id"], t["delivery_id"], t["group"]
        camera_name = f"{group}/{delivery_id}/{cam_id}"
        out_jsonl = pred_dir / f"{group}__{delivery_id}__{cam_id}.jsonl"
        if progress is not None:
            progress.set_postfix_str(camera_name, refresh=False)

        done: set[str] = set()
        existing_people = 0
        if args.resume and out_jsonl.exists():
            with out_jsonl.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                        done.add(row["frame_name"])
                        existing_people += len(row.get("players", []))
                    except Exception:
                        pass
        mode = "a" if (args.resume and out_jsonl.exists()) else "w"

        cam_processed = cam_skipped = cam_failed = 0
        cam_people = existing_people
        with out_jsonl.open(mode, encoding="utf-8") as handle:
            # Resolve resume-skips up front, then stream the rest through the prefetch
            # pipeline so cold-disk read + decode overlaps detection + pose on the GPU.
            pending_all = [(idx, fp) for idx, fp in enumerate(t["frames"]) if fp.name not in done]
            skipped_here = len(t["frames"]) - len(pending_all)
            cam_skipped += skipped_here
            if progress is not None and skipped_here:
                progress.update(skipped_here)

            tiled_loader = (
                make_tiled_loader(args)
                if args.tiled_det and args.tiled_fast else None
            )
            for loaded, load_failures in p1.prefetch_decoded_batches(
                pending_all, decode_pool, args.det_batch_size, args.prefetch_batches, timings,
                loader=tiled_loader,
            ):
                for _, frame_path, exc in load_failures:
                    cam_failed += 1
                    _record_failure(failures, t, frame_path.name, "decode", exc)
                    if progress is not None:
                        progress.update(1)
                if not loaded:
                    continue

                try:
                    start = time.perf_counter()
                    with _amp_context(args, device):
                        if args.tiled_det and args.tiled_fast:
                            batch_boxes = detect_person_boxes_tiled_fast(
                                detector, loaded, args)
                        elif args.tiled_det:
                            batch_boxes = detect_person_boxes_tiled_batch(
                                detector, inference_detector, loaded, args)
                        else:
                            batch_boxes = p1.detect_person_boxes_batch(
                                detector, inference_detector, loaded, args)
                    _sync(device)
                    timings["detect_seconds"] += time.perf_counter() - start
                    for entry, boxes in zip(loaded, batch_boxes):
                        entry["boxes"] = boxes
                    start = time.perf_counter()
                    with _amp_context(args, device):
                        batch_results = p1.inference_topdown_batch(
                            pose_model, pose_pipeline, loaded, args.pose_batch_size)
                    _sync(device)
                    timings["pose_seconds"] += time.perf_counter() - start
                except Exception as exc:  # noqa: BLE001
                    for entry in loaded:
                        cam_failed += 1
                        _record_failure(failures, t, entry["frame_path"].name, "inference", exc)
                        if progress is not None:
                            progress.update(1)
                    continue

                for entry, results in zip(loaded, batch_results):
                    frame_path = entry["frame_path"]
                    players = p1.player_records(
                        results, source_skeleton, entry["width"], entry["height"], coco17_indices)
                    # --- record dict copied verbatim from run_phase1_rtmpose_inference.py --- #
                    record = {
                        "schema_version": P1_SCHEMA_VERSION,
                        "camera_id": cam_id,
                        "delivery_id": delivery_id,
                        "capture_group": group,
                        "frame_index": parse_frame_id(frame_path),
                        "frame_name": frame_path.name,
                        "match_id": p1.match_id_from_delivery(delivery_id),
                        "metadata": {
                            "model_id": args.model_id, "run_id": run_id, "device": device,
                            "capture_group": group,
                            "image_size_px": [entry["width"], entry["height"]],
                            "inference_mode": "topdown_detector_pose",
                            "input_mode": "opencv_bgr_mmdet_mmpose_batch",
                            "det_batch_size_requested": args.det_batch_size,
                            "det_batch_size_effective": args.det_batch_size,
                            "pose_batch_size_requested": args.pose_batch_size,
                            "pose_batch_size_effective": args.pose_batch_size,
                            "io_workers": args.io_workers,
                            "detector": "rtmdet_m_person_tiled" if args.tiled_det else "rtmdet_m_person",
                            "bbox_thr": args.bbox_thr,
                            "nms_thr": args.nms_thr,
                            "tiled_det": bool(args.tiled_det),
                            "model_specific": {
                                "rtmpose": {
                                    "source_skeleton": source_skeleton,
                                    "output_skeleton": P1_SKELETON,
                                    "pose_config": p1.rel(pose_config),
                                    "pose_checkpoint": p1.rel(pose_checkpoint),
                                    "det_config": p1.rel(p1.abspath(args.det_config)),
                                    "det_checkpoint": p1.rel(p1.abspath(args.det_checkpoint)),
                                }
                            },
                        },
                        "players": players,
                    }
                    validate_group1_frame(record, final_handoff=False)
                    # ---------------------------------------------------------------------- #
                    start = time.perf_counter()
                    handle.write(json.dumps(record) + "\n")
                    timings["write_seconds"] += time.perf_counter() - start
                    cam_people += len(players)
                    cam_processed += 1
                    if progress is not None:
                        progress.update(1)

        seen += cam_processed + cam_skipped + cam_failed
        if progress is not None:
            progress.write(f"+ {camera_name}: processed={cam_processed} skipped={cam_skipped} "
                           f"people={cam_people} failed={cam_failed} -> {out_jsonl.name}")
        else:
            pct = 100.0 * seen / total_frames if total_frames else 100.0
            rate = seen / (time.perf_counter() - t0) if seen else 0.0
            eta_min = (total_frames - seen) / rate / 60.0 if rate > 0 else float("nan")
            print(f"+ {camera_name}: processed={cam_processed} skipped={cam_skipped} "
                  f"people={cam_people} failed={cam_failed} | overall {seen}/{total_frames} "
                  f"({pct:.0f}%) {rate:.1f} f/s ETA {eta_min:.0f}m", flush=True)
        cameras_summary.append({
            "group": group, "delivery_id": delivery_id, "camera_id": cam_id,
            "frames_processed": cam_processed, "frames_skipped": cam_skipped,
            "people": cam_people, "failed": cam_failed,
            "predictions": p1.rel(out_jsonl),
        })
        frames_processed += cam_processed
        frames_skipped += cam_skipped
        failed_frames += cam_failed
        total_people += cam_people

    decode_pool.shutdown(wait=True)
    if progress is not None:
        progress.close()

    elapsed = time.perf_counter() - t0
    _write_metrics(args, out_dir, pred_dir, run_id, device, source_skeleton,
                   cameras_summary, timings, failures, elapsed,
                   frames_processed, frames_skipped, failed_frames, total_people)

    fps = round(frames_processed / elapsed, 2) if elapsed > 0 else None
    print(f"\nDone in {elapsed:.1f}s | processed={frames_processed} skipped={frames_skipped} "
          f"people={total_people} failed={failed_frames} | {fps} fps", flush=True)
    print(f"Output: {out_dir}", flush=True)
    return 0 if failed_frames == 0 else 1


def _record_failure(failures: list[dict[str, Any]], t: dict[str, Any],
                    frame_name: str, stage: str, exc: Exception) -> None:
    if len(failures) < 2000:
        failures.append({
            "group": t["group"], "delivery_id": t["delivery_id"],
            "camera_id": t["camera_id"], "frame_name": frame_name,
            "stage": stage, "error": str(exc),
        })


def _write_metrics(args, out_dir: Path, pred_dir: Path, run_id: str, device: str,
                   source_skeleton: str, cameras: list[dict[str, Any]], timings: dict[str, float],
                   failures: list[dict[str, Any]], elapsed: float,
                   processed: int, skipped: int, failed: int, people: int) -> None:
    fps = round(processed / elapsed, 2) if elapsed > 0 else None
    summary = {
        "records_written": processed + skipped,
        "records_written_this_run": processed,
        "records_reused": skipped,
        "total_players_detected": people,
        "failed_frames": failed,
        "wall_clock_s": round(elapsed, 2),
        "fps_overall": fps,
        "timings": {k: round(v, 3) for k, v in timings.items()},
        "status": "pass" if failed == 0 else "partial",
    }
    manifest = {
        "schema_version": "cricket_phase1_run/v2",
        "prediction_schema_version": P1_SCHEMA_VERSION,
        "run_id": run_id, "created_at": p1.utc_now(),
        "model_id": args.model_id, "device": device, "skeleton": source_skeleton,
        "pose_data_root": str(Path(args.pose_data).expanduser()),
        "prediction_dir": p1.rel(pred_dir),
        "detector": "rtmdet_m_person_tiled" if args.tiled_det else "rtmdet_m_person",
        "tiled_det": bool(args.tiled_det),
        "amp": bool(getattr(args, "amp", False)),
        "tiled_fast": bool(getattr(args, "tiled_fast", False)) if args.tiled_det else None,
        "tile_grid": [args.tile_cols, args.tile_rows] if args.tiled_det else None,
        "tile_overlap": args.tile_overlap if args.tiled_det else None,
        "det_batch_size": args.det_batch_size, "pose_batch_size": args.pose_batch_size,
        "io_workers": args.io_workers, "cv2_threads": args.cv2_threads,
        "prefetch_batches": args.prefetch_batches, "perf": args.perf,
        "bbox_thr": args.bbox_thr, "nms_thr": args.nms_thr,
        "git_sha": git_sha(),
        "summary": summary,
        "delivery_count": len({c["delivery_id"] for c in cameras}),
        "camera_count": len(cameras),
        "cameras": cameras,
        "failures": failures[:500],
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "p1_metrics.json").write_text(json.dumps({
        "schema_version": "cricket_phase1_metrics/v2",
        "prediction_schema_version": P1_SCHEMA_VERSION,
        "run_id": run_id, "created_at": p1.utc_now(),
        "model_id": args.model_id, "device": device, "skeleton": source_skeleton,
        "det_batch_size": args.det_batch_size, "pose_batch_size": args.pose_batch_size,
        "io_workers": args.io_workers,
        "delivery_ids": sorted({c["delivery_id"] for c in cameras}),
        "summary": summary,
    }, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    args = parse_args()
    if not args.show_torch_warnings:
        warnings.filterwarnings(
            "ignore", message=r"torch\.meshgrid: in an upcoming release.*", category=UserWarning)
    if args.det_batch_size <= 0 or args.pose_batch_size <= 0:
        raise SystemExit("--det-batch-size and --pose-batch-size must be positive")
    if args.io_workers < 0:
        raise SystemExit("--io-workers must be >= 0")
    if args.cv2_threads < 1:
        raise SystemExit("--cv2-threads must be >= 1")
    if args.prefetch_batches < 0:
        raise SystemExit("--prefetch-batches must be >= 0")
    # Stop OpenCV/torch from each spawning a per-core pool inside every io-worker (which
    # thrashes the CPU and starves the GPU); the GPU carries detect/pose, the CPU decodes.
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

    targets = discover_targets(args)
    if not targets:
        raise SystemExit("No frames matched the given filters. Try --list to inspect selection.")

    total_frames = sum(len(t["frames"]) for t in targets)
    print(f"Selected {len(targets)} camera(s), {total_frames} frame(s) "
          f"across {len({t['delivery_id'] for t in targets})} deliveries:", flush=True)
    for t in targets[:12]:
        print(f"  {t['group']}/{t['delivery_id']}/{t['camera_id']}: {len(t['frames'])} frames",
              flush=True)
    if len(targets) > 12:
        print(f"  ... (+{len(targets) - 12} more cameras)", flush=True)
    if args.list:
        return 0

    device = args.device
    if device.startswith("cuda"):
        try:
            import torch
            if not torch.cuda.is_available():
                if not args.allow_cpu:
                    raise SystemExit("CUDA unavailable; pass --device cpu --allow-cpu to override.")
                device = "cpu"
        except ModuleNotFoundError:
            raise SystemExit("torch not importable -- activate the model env first.")
    elif device == "cpu" and not args.allow_cpu:
        raise SystemExit("CPU requested without --allow-cpu")

    preflight(args, device)
    apply_perf(device, args.perf)

    if args.sweep:
        return run_sweep(args, targets, device)
    return run_inference(args, targets, device)


if __name__ == "__main__":
    raise SystemExit(main())

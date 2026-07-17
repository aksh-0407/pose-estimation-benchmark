#!/usr/bin/env python3
"""Run RTMPose (top-down) 2D pose inference on the internal cricket frame dataset.

RTMPose is a top-down model: a person detector (RTMDet) produces boxes, then
RTMPose predicts keypoints per box. This script walks ``<drive-root>/bt_0X/
<delivery>/camera<NN>/`` frame folders, runs detection plus pose, and writes one
JSONL prediction file per camera under ``<run>/<DELIVERY>/00_inference/predictions/``.

Everything is filterable from the CLI so you can run the whole dataset, a single
capture group, one delivery, one camera, or a short frame slice; see --help.

Run inside the model's Conda env, e.g.:

    conda activate pose-lab
    python src/core/inference/run_phase1_rtmpose_inference.py --list
    python src/core/inference/run_phase1_rtmpose_inference.py \
        --groups bt_01 --deliveries CCPL080626M1_1_14_1 --cameras camera01 \
        --frame-limit 50 --overlay

Shared model/inference building blocks live in ``phase1_common`` (also used by
the remote-GPU runner ``run_phase1_l40s.py``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from core.contract import SCHEMA_VERSION as P1_SCHEMA_VERSION, validate_group1_frame  # noqa: E402
from core.frames import parse_frame_id  # noqa: E402
from core.datasets import derived_root, raw_root  # noqa: E402
from core.inference.phase1_common import (  # noqa: E402
    DEFAULT_DET_CHECKPOINT,
    DEFAULT_DET_CONFIG,
    DEFAULT_MODEL_CONFIG,
    DETECTOR_PRESETS,
    abspath,
    append_capped,
    build_frame_record,
    build_models,
    build_pose_pipeline,
    camera_metric_key,
    detect_person_boxes_batch,
    git_sha,
    inference_topdown_batch,
    iter_camera_targets,
    match_id_from_delivery,
    normalize_camera_filters,
    player_records,
    prefetch_decoded_batches,
    read_resume_state,
    rel,
    resolve_device,
    resolve_skeleton,
    timed_call,
    utc_now,
    write_record,
)


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
    mdl.add_argument("--detector", default=None,
                     help="Detector preset overriding --det-config/--det-checkpoint/--det-cat-id: "
                          f"{sorted(DETECTOR_PRESETS)}. Default (unset) = the vendored RTMDet-m "
                          "baseline. mim presets need tools/detector_bakeoff/fetch_detectors.py first.")
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
# Progress reporting
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Discovery (repo dataset layout: <drive-root>/bt_0X/<delivery>/camera<NN>/)
# --------------------------------------------------------------------------- #
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
            for camera_dir, cam_id, frames in iter_camera_targets(
                delivery_dir, camera_filter, args.start_index, args.stride, args.frame_limit
            ):
                targets.append({
                    "group": group_dir.name,
                    "delivery_id": delivery_dir.name,
                    "camera_dir": camera_dir,
                    "camera_id": cam_id,
                    "camera_number": camera_dir.name.replace("camera", ""),
                    "frames": frames,
                })
    return targets


# --------------------------------------------------------------------------- #
# Overlays
# --------------------------------------------------------------------------- #
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
    # P1 output dir; explicit flags win.
    if args.dataset:
        if not args.version:
            raise SystemExit("--dataset requires --version (-> pipetrack_v<version>)")
        if args.drive_root is None:
            args.drive_root = str(raw_root(args.data_root, args.dataset))
        if args.run_dir is None:
            # The run root; P1 writes per delivery under <run>/<DELIVERY>/00_inference/.
            args.run_dir = str(derived_root(args.data_root, args.dataset, args.version))
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
        """Per-delivery P1 stage dir: <run>/<delivery>/00_inference/ (sits beside 01..07)."""
        return run_dir / delivery / "00_inference"

    print(f"Loading detector + RTMPose on {device} ...", flush=True)
    detector, pose_model, inference_detector, pose_config, pose_checkpoint = build_models(args, device)
    pose_pipeline = build_pose_pipeline(pose_model)
    source_skeleton, coco17_indices = resolve_skeleton(pose_config, pose_model)
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
            done, existing_people = read_resume_state(out_jsonl, camera_name)
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
                    record = build_frame_record(
                        camera_id=cam_id,
                        delivery_id=delivery_id,
                        capture_group=t["group"],
                        frame_path=frame_path,
                        width=entry["width"],
                        height=entry["height"],
                        players=players,
                        model_id=args.model_id,
                        run_id=run_id,
                        device=device,
                        det_batch_size=args.det_batch_size,
                        pose_batch_size=args.pose_batch_size,
                        io_workers=args.io_workers,
                        detector_label=(getattr(args, "detector", None) or "rtmdet_m_person"),
                        bbox_thr=args.bbox_thr,
                        nms_thr=args.nms_thr,
                        source_skeleton=source_skeleton,
                        output_skeleton=source_skeleton,
                        pose_config_rel=rel(pose_config),
                        pose_checkpoint_rel=rel(pose_checkpoint),
                        det_config_rel=rel(abspath(args.det_config)),
                        det_checkpoint_rel=rel(abspath(args.det_checkpoint)),
                    )
                    validate_group1_frame(record, final_handoff=False)
                    if handle is not None:
                        write_record(handle, record, timings)
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
        "detector": (getattr(args, "detector", None) or "rtmdet_m_person"),
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
                "detector": (getattr(args, "detector", None) or "rtmdet_m_person"),
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

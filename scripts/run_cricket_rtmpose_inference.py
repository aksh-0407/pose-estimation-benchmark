#!/usr/bin/env python3
"""Run RTMPose (top-down) 2D pose inference on the internal cricket frame dataset.

RTMPose is a top-down model: a person detector (RTMDet) produces boxes, then
RTMPose predicts keypoints per box. This script walks ``drive/dataset/<group>/
<delivery>/camera<NN>/`` frame folders, runs detection + pose, and writes one
JSONL prediction file per camera under ``benchmarks/runs/<run-id>/predictions/``.

Everything is filterable from the CLI so you can run the whole dataset, a single
capture group, one delivery, one camera, or a short frame slice -- see --help.

Run inside the model's Conda env, e.g.:

    conda activate cricket-rtmpose-l-wholebody
    python scripts/run_cricket_rtmpose_inference.py --list
    python scripts/run_cricket_rtmpose_inference.py \
        --groups bt_01 --deliveries CCPL080626M1_1_14_1 --cameras camera01 \
        --frame-limit 50 --overlay
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml

from pose_estimation.cricket.dataset import FRAME_RE, camera_label, parse_frame_id

MMPOSE_DIR = ROOT / "external" / "mmpose"
DEFAULT_MODEL_CONFIG = ROOT / "configs" / "model_envs.yaml"
DEFAULT_DET_CONFIG = ROOT / "external" / "mmpose" / "demo" / "mmdetection_cfg" / "rtmdet_m_640-8xb32_coco-person.py"
DEFAULT_DET_CHECKPOINT = ROOT / "models" / "rtmdet_m_person" / "weights" / "rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
SCHEMA_VERSION = "rtmpose_topdown.v1"


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
    sel.add_argument("--drive-root", default="drive", help="Root containing dataset/ (default: drive)")
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
    mdl.add_argument("--model-id", default="rtmpose_l_wholebody",
                     help="Key in configs/model_envs.yaml supplying pose config+checkpoint")
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
    rt.add_argument("--run-dir", default=None, help="Output dir (default: benchmarks/runs/<run-id>)")
    rt.add_argument("--no-resume", dest="resume", action="store_false",
                    help="Recompute frames already present in the camera JSONL")
    rt.add_argument("--no-progress", dest="show_progress", action="store_false",
                    help="Disable the tqdm progress bar")
    rt.set_defaults(resume=True)
    rt.set_defaults(show_progress=True)

    # Overlays
    ov = parser.add_argument_group("overlays")
    ov.add_argument("--overlay", action="store_true", help="Render keypoint overlay images")
    ov.add_argument("--overlay-every", type=int, default=1, help="Render every Nth processed frame (default: 1)")
    ov.add_argument("--overlay-limit", type=int, default=30, help="Max overlays per camera (default: 30)")

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
    dataset_root = abspath(args.drive_root) / "dataset"
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


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def build_models(args: argparse.Namespace, device: str):
    from mmdet.apis import inference_detector, init_detector
    from mmpose.apis import inference_topdown, init_model
    from mmpose.utils import adapt_mmdet_pipeline

    pose_config, pose_checkpoint = resolve_model_paths(args)
    pose_config, pose_checkpoint = abspath(pose_config), abspath(pose_checkpoint)
    det_config, det_checkpoint = abspath(args.det_config), abspath(args.det_checkpoint)
    for label, path in [("pose config", pose_config), ("pose checkpoint", pose_checkpoint),
                        ("detector config", det_config), ("detector checkpoint", det_checkpoint)]:
        if not path.exists():
            raise SystemExit(f"missing {label}: {path}")

    # MMPose/MMDet configs resolve _base_ includes relative to the mmpose tree.
    os.chdir(MMPOSE_DIR)
    detector = init_detector(str(det_config), str(det_checkpoint), device=device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)
    pose_model = init_model(str(pose_config), str(pose_checkpoint), device=device)
    return detector, pose_model, inference_detector, inference_topdown, str(pose_config), str(pose_checkpoint)


def detect_person_boxes(detector, inference_detector, image_path: str, args: argparse.Namespace):
    import numpy as np
    from mmpose.evaluation.functional import nms

    det_result = inference_detector(detector, image_path)
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


def coco17_source_indices(source_skeleton: str) -> list[int] | None:
    """Indices into the source skeleton that yield COCO-17 order, or None if unmapped."""
    from pose_estimation.keypoints import load_keypoint_mappings

    mapping = load_keypoint_mappings()["source_to_coco_17"].get(source_skeleton)
    return mapping["source_indices"] if mapping else None


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
            xs, ys = kpts[:, 0], kpts[:, 1]
            x1, y1, x2, y2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
        bw, bh = x2 - x1, y2 - y1

        keypoints_px = [[float(x), float(y)] for x, y in kpts]
        confidence = [float(s) for s in scores]
        keypoints_norm = [[x / width, y / height] for x, y in keypoints_px]

        if coco17_indices is not None:
            coco17_px = [keypoints_px[i] if i < len(keypoints_px) else [None, None] for i in coco17_indices]
            coco17_conf = [confidence[i] if i < len(confidence) else None for i in coco17_indices]
        else:
            coco17_px, coco17_conf = None, None

        players.append({
            "bbox_xywh_px": [x1, y1, bw, bh],
            "bbox_xywh_norm": [x1 / width, y1 / height, bw / width, bh / height],
            "global_player_id": None,
            "local_track_id": None,
            "track_confidence": None,
            "role": None,
            "pose_2d": {
                "skeleton": source_skeleton,
                "num_keypoints": len(keypoints_px),
                "keypoints_px": keypoints_px,
                "keypoints_norm": keypoints_norm,
                "confidence": confidence,
                "coco17_keypoints_px": coco17_px,
                "coco17_confidence": coco17_conf,
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
    from mmpose.structures import merge_data_samples

    img = mmcv.imread(image_path, channel_order="rgb")
    data_samples = merge_data_samples(results)
    visualizer.add_datasample(
        "result", img, data_sample=data_samples, draw_gt=False,
        draw_bbox=True, show=False, kpt_thr=kpt_thr,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mmcv.imwrite(visualizer.get_image()[..., ::-1], str(out_path))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    args = parse_args()
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
    run_dir = abspath(args.run_dir) if args.run_dir else (ROOT / "benchmarks" / "runs" / run_id)
    pred_dir = run_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading detector + RTMPose on {device} ...", flush=True)
    detector, pose_model, inference_detector, inference_topdown, pose_config, pose_checkpoint = build_models(args, device)
    source_skeleton = getattr(pose_model.dataset_meta, "get", lambda *a: None)("dataset_name") or "coco_wholebody_133"
    if "wholebody" in pose_config.lower():
        source_skeleton = "coco_wholebody_133"
    elif "coco" in pose_config.lower():
        source_skeleton = "coco_17"
    coco17_indices = coco17_source_indices(source_skeleton)
    visualizer = build_visualizer(pose_model, args.kpt_thr) if args.overlay else None

    summary = {
        "run_id": run_id, "model_id": args.model_id, "device": device,
        "pose_config": pose_config, "pose_checkpoint": pose_checkpoint,
        "det_config": str(abspath(args.det_config)), "det_checkpoint": str(abspath(args.det_checkpoint)),
        "skeleton": source_skeleton, "started_at": utc_now(),
        "cameras": [], "frames_processed": 0, "frames_skipped": 0,
        "total_people": 0, "failed_frames": 0,
    }

    t0 = time.perf_counter()
    progress = build_progress_bar(total_frames, args.show_progress)
    for t in targets:
        cam_id, delivery_id = t["camera_id"], t["delivery_id"]
        camera_name = f"{t['group']}/{delivery_id}/{cam_id}"
        out_jsonl = pred_dir / f"{t['group']}__{delivery_id}__{cam_id}.jsonl"
        done: set[str] = set()
        if args.resume and out_jsonl.exists():
            with out_jsonl.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        done.add(json.loads(line)["frame_name"])
                    except Exception:
                        pass
        mode = "a" if (args.resume and out_jsonl.exists()) else "w"

        cam_people = cam_processed = cam_skipped = cam_failed = overlays = 0
        with out_jsonl.open(mode, encoding="utf-8") as handle:
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
                    continue
                try:
                    import cv2
                    img = cv2.imread(str(frame_path))
                    if img is None:
                        raise RuntimeError("image failed to decode")
                    height, width = img.shape[:2]
                    boxes = detect_person_boxes(detector, inference_detector, str(frame_path), args)
                    results = []
                    if len(boxes):
                        results = inference_topdown(pose_model, str(frame_path), boxes[:, :4], bbox_format="xyxy")
                    players = player_records(results, source_skeleton, width, height, coco17_indices)
                except Exception as exc:  # noqa: BLE001
                    cam_failed += 1
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
                    continue

                record = {
                    "schema_version": SCHEMA_VERSION,
                    "camera_id": cam_id,
                    "delivery_id": delivery_id,
                    "capture_group": t["group"],
                    "frame_index": parse_frame_id(frame_path),
                    "frame_name": frame_path.name,
                    "match_id": match_id_from_delivery(delivery_id),
                    "metadata": {
                        "model_id": args.model_id, "run_id": run_id, "device": device,
                        "image_size_px": [width, height], "skeleton": source_skeleton,
                        "detector": "rtmdet_m_person", "bbox_thr": args.bbox_thr,
                        "nms_thr": args.nms_thr,
                    },
                    "players": players,
                }
                handle.write(json.dumps(record) + "\n")
                cam_people += len(players)
                cam_processed += 1

                if visualizer is not None and overlays < args.overlay_limit and idx % args.overlay_every == 0:
                    vis_path = run_dir / "visualizations" / t["group"] / delivery_id / cam_id / f"{frame_path.stem}.jpg"
                    try:
                        render_overlay(visualizer, str(frame_path), results, vis_path, args.kpt_thr)
                        overlays += 1
                    except Exception as exc:  # noqa: BLE001
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
            "people": cam_people, "failed": cam_failed, "predictions": str(out_jsonl),
        })
        summary["frames_processed"] += cam_processed
        summary["frames_skipped"] += cam_skipped
        summary["total_people"] += cam_people
        summary["failed_frames"] += cam_failed

    if progress is not None:
        progress.close()

    elapsed = time.perf_counter() - t0
    summary["finished_at"] = utc_now()
    summary["elapsed_seconds"] = round(elapsed, 2)
    summary["fps"] = round(summary["frames_processed"] / elapsed, 2) if elapsed > 0 else None
    (run_dir / "run_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"\nDone in {elapsed:.1f}s | processed={summary['frames_processed']} "
        f"skipped={summary['frames_skipped']} people={summary['total_people']} "
        f"failed={summary['failed_frames']} | {summary['fps']} fps\nRun dir: {run_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the 01 (stabilization) 2D stabilization stage over a canonical run directory.

Reads a P1 run dir (``predictions/<capture_group>__<delivery>__cam_NN.jsonl``), smooths
each camera's per-keypoint pixel trajectories, and writes a new run dir in the same
canonical format (``local_track_id`` stays null, so the output is a drop-in P2 input).
Reports mean 2D jitter (px) before vs after per camera.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from core.contract import validate_group1_frame
from identity.p1_stabilization.config import StabilizationConfig
from identity.p1_stabilization.linker import link_micro_tracks
from identity.p1_stabilization.smoothing import mean_jitter_px, smooth_track_keypoints

CANONICAL_PREDICTION_RE = re.compile(
    r"^(?P<capture_group>bt_\d{2})__(?P<delivery_id>.+)__(?P<camera_id>cam_\d{2})\.jsonl$"
)


def _read_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _bbox_xywh(player: dict) -> tuple[float, float, float, float] | None:
    box = player.get("bbox_xywh_px")
    if not box or len(box) != 4:
        return None
    x, y, w, h = (float(v) for v in box)
    if not (np.isfinite([x, y, w, h]).all()) or w <= 0 or h <= 0:
        return None
    return x, y, w, h


def _image_size(records: list[dict]) -> tuple[float, float]:
    for rec in records:
        size = rec.get("metadata", {}).get("image_size_px")
        if size and len(size) == 2 and size[0] and size[1]:
            return float(size[0]), float(size[1])
    return 2560.0, 1440.0  # rig default


def _smooth_block(members, records, block_key, config: StabilizationConfig,
                  fps: float, jitter_acc: dict) -> None:
    """Smooth one keypoint block (``pose_2d``, the Halpe-26 skeleton) for one micro-track."""
    frames = [(fp, pi) for (fp, pi) in members]
    series_kpts: list[list[list[float]]] = []
    series_conf: list[list[float]] = []
    series_bboxdiag: list[float] = []
    series_dt: list[float] = []
    prev_pos = None
    valid_refs: list[tuple[dict, int]] = []
    for fp, pi in frames:
        player = records[fp]["players"][pi]
        block = player.get(block_key)
        if not block or not block.get("keypoints_px"):
            continue
        series_kpts.append([[float(x), float(y)] for x, y in block["keypoints_px"]])
        series_conf.append([float(c) for c in block.get("confidence", [])])
        box = _bbox_xywh(player)
        series_bboxdiag.append(float(np.hypot(box[2], box[3])) if box else 0.0)
        series_dt.append(1.0 / fps if prev_pos is None else max(fp - prev_pos, 1) / fps)
        prev_pos = fp
        valid_refs.append((block, len(series_kpts) - 1))
    if len(series_kpts) < 2:
        return
    # Ragged guard: only smooth if every frame has the same keypoint count.
    kcount = {len(row) for row in series_kpts}
    if len(kcount) != 1:
        return
    kpts = np.asarray(series_kpts, dtype=float)          # (T, K, 2)
    conf = np.asarray(series_conf, dtype=float)
    if conf.shape != kpts.shape[:2]:
        return
    bboxdiag = np.asarray(series_bboxdiag, dtype=float)
    dt = np.asarray(series_dt, dtype=float)

    before = mean_jitter_px(kpts, conf, config.gating.confidence_min)
    smoothed = smooth_track_keypoints(kpts, conf, bboxdiag, dt, config.smoothing, config.gating)
    after = mean_jitter_px(smoothed, conf, config.gating.confidence_min)

    if block_key == "pose_2d":
        weight = kpts.shape[0]
        jitter_acc["before"] += before * weight
        jitter_acc["after"] += after * weight
        jitter_acc["weight"] += weight

    img_w, img_h = jitter_acc["img_size"]
    for block, ti in valid_refs:
        new_px = [[float(smoothed[ti, k, 0]), float(smoothed[ti, k, 1])]
                  for k in range(smoothed.shape[1])]
        block["keypoints_px"] = new_px
        block["keypoints_norm"] = [[x / img_w, y / img_h] for x, y in new_px]


def stabilize_camera_file(input_path: Path, output_path: Path, camera_id: str,
                          config: StabilizationConfig) -> dict:
    records = _read_records(input_path)
    for rec in records:
        validate_group1_frame(rec)
    img_w, img_h = _image_size(records)
    jitter_acc = {"before": 0.0, "after": 0.0, "weight": 0.0, "img_size": (img_w, img_h)}

    if config.enabled:
        frame_boxes: list[list[tuple[int, tuple[float, float, float, float]]]] = []
        for rec in records:
            dets = []
            for pi, player in enumerate(rec.get("players", [])):
                box = _bbox_xywh(player)
                if box is not None and player.get("pose_2d"):
                    dets.append((pi, box))
            frame_boxes.append(dets)
        micro_tracks = link_micro_tracks(frame_boxes, config.link)
        for members in micro_tracks:
            # pose_2d is the canonical Halpe-26 block (feet included) — one smoothing pass.
            _smooth_block(members, records, "pose_2d", config, config.frame_rate_fps, jitter_acc)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for rec in records:
            validate_group1_frame(rec)
            # Preserve the parsed key order (no sort_keys): with the stage disabled
            # the output is byte-identical to ANY input regardless of how P1 ordered
            # its keys, and with it enabled the diff shows only the smoothed values.
            out.write(json.dumps(rec) + "\n")

    w = jitter_acc["weight"] or 1.0
    return {
        "camera_id": camera_id,
        "status": "ok",
        "frames": len(records),
        "jitter_px_before": jitter_acc["before"] / w,
        "jitter_px_after": jitter_acc["after"] / w,
    }


def run_stabilization(input_run_dir: str | Path, output_run_dir: str | Path,
                      delivery_id: str, config: StabilizationConfig,
                      cameras: list[str] | None = None) -> dict:
    input_run_dir = Path(input_run_dir)
    output_run_dir = Path(output_run_dir)
    prediction_dir = input_run_dir / "predictions"
    if not prediction_dir.exists():
        raise RuntimeError(f"missing predictions directory: {prediction_dir}")

    wanted = set(cameras) if cameras else None
    per_camera: dict[str, dict] = {}
    for path in sorted(prediction_dir.glob("*.jsonl")):
        m = CANONICAL_PREDICTION_RE.match(path.name)
        if not m or m.group("delivery_id") != delivery_id:
            continue
        cam = m.group("camera_id")
        if wanted is not None and cam not in wanted:
            continue
        out_path = output_run_dir / "predictions" / path.name
        per_camera[cam] = stabilize_camera_file(path, out_path, cam, config)

    if not per_camera:
        raise RuntimeError(f"no canonical prediction files for delivery {delivery_id} in {prediction_dir}")

    created_at = datetime.now(timezone.utc).isoformat()
    output_run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "stabilization_run/v1",
        "created_at": created_at,
        "task": "2d_stabilization",
        "input_run_dir": str(input_run_dir),
        "output_run_dir": str(output_run_dir),
        "delivery_id": delivery_id,
        "config": config.to_dict(),
        "cameras": sorted(per_camera),
    }
    weights = sum(1 for _ in per_camera)
    metrics = {
        "schema_version": "stabilization_metrics/v1",
        "created_at": created_at,
        "delivery_id": delivery_id,
        "status": "pass",
        "enabled": config.enabled,
        "mean_jitter_px_before": sum(c["jitter_px_before"] for c in per_camera.values()) / max(weights, 1),
        "mean_jitter_px_after": sum(c["jitter_px_after"] for c in per_camera.values()) / max(weights, 1),
        "per_camera": per_camera,
    }
    with (output_run_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (output_run_dir / "stabilization_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return metrics

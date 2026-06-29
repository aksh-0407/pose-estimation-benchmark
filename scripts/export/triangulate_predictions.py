#!/usr/bin/env python3
"""Triangulate multi-view poses, including canonical PipeTrack P4 runs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pose_estimation.cricket.contract import validate_group1_frame  # noqa: E402
from pose_estimation.schemas import CameraCalibration  # noqa: E402
from pose_estimation.triangulation import (  # noqa: E402
    confidence_ema_smooth,
    triangulate_skeleton_ransac,
)
from scripts.association.jsonl_io import load_synchronized_records  # noqa: E402
from scripts.tracking.calibration import load_projection_matrices_from_drive  # noqa: E402
from scripts.tracking.runner import discover_prediction_files, infer_match_id  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    canonical = parser.add_argument_group("canonical PipeTrack run")
    canonical.add_argument("--input-run-dir")
    canonical.add_argument("--output-run-dir")
    canonical.add_argument("--drive-root")
    canonical.add_argument("--delivery-id")
    canonical.add_argument("--camera", action="append", default=None)
    legacy = parser.add_argument_group("legacy flat JSONL")
    legacy.add_argument("--predictions")
    legacy.add_argument("--calibration")
    legacy.add_argument("--output")
    parser.add_argument("--reprojection-threshold-px", type=float, default=10.0)
    parser.add_argument("--min-views", type=int, default=2)
    parser.add_argument("--ema-alpha", type=float, default=0.65)
    return parser.parse_args(argv)


def load_calibration(path: str | Path) -> dict[str, CameraCalibration]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    cameras = payload.get("cameras", payload)
    if isinstance(cameras, dict):
        cameras = list(cameras.values())
    return {camera["camera_id"]: CameraCalibration.from_dict(camera) for camera in cameras}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def triangulate_canonical_run(
    *,
    input_run_dir: str | Path,
    output_run_dir: str | Path,
    drive_root: str | Path,
    delivery_id: str,
    cameras: list[str] | None,
    reprojection_threshold_px: float,
    min_views: int,
    ema_alpha: float,
) -> dict[str, Any]:
    input_run_dir, output_run_dir = Path(input_run_dir), Path(output_run_dir)
    prediction_files = discover_prediction_files(input_run_dir, delivery_id, cameras)
    records_by_frame = load_synchronized_records(prediction_files, delivery_id)
    match_id = infer_match_id(delivery_id)
    all_projections = load_projection_matrices_from_drive(drive_root, match_id)

    observations: dict[tuple[int, str], list[tuple[str, dict]]] = defaultdict(list)
    for frame_index, camera_records in records_by_frame.items():
        for camera_id, record in camera_records.items():
            if camera_id not in all_projections:
                continue
            for player in record.get("players", []):
                player_id = player.get("global_player_id")
                if player_id and player.get("pose_2d") is not None:
                    observations[(frame_index, player_id)].append((camera_id, player))

    raw: dict[tuple[int, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for key, views in sorted(observations.items()):
        if len(views) < min_views:
            continue
        keypoints = np.asarray(
            [
                np.column_stack((player["pose_2d"]["keypoints_px"], player["pose_2d"]["confidence"]))
                for _, player in views
            ],
            dtype=float,
        )
        projections = np.asarray([all_projections[camera] for camera, _ in views], dtype=float)
        raw[key] = triangulate_skeleton_ransac(
            keypoints,
            projections,
            reprojection_threshold_px=reprojection_threshold_px,
            min_views=min_views,
        )

    # Smooth each identity in temporal order, then stamp the same 3D pose into
    # every contributing camera record for that synchronized frame.
    per_identity: dict[str, list[int]] = defaultdict(list)
    for frame_index, player_id in raw:
        per_identity[player_id].append(frame_index)
    smoothed: dict[tuple[int, str], np.ndarray] = {}
    for player_id, frames in per_identity.items():
        frames = sorted(set(frames))
        sequence = np.asarray([raw[(frame, player_id)][0] for frame in frames], dtype=float)
        confidences = np.asarray([raw[(frame, player_id)][1] for frame in frames], dtype=float)
        filtered = confidence_ema_smooth(sequence, confidences, alpha=ema_alpha)
        smoothed.update({(frame, player_id): filtered[index] for index, frame in enumerate(frames)})

    successful = 0
    for key, views in observations.items():
        if key not in raw:
            continue
        points = smoothed[key]
        confidences, errors = raw[key][1], raw[key][2]
        if not (np.isfinite(points).all() and np.isfinite(confidences).all() and np.isfinite(errors).all()):
            continue
        pose_3d = {
            "keypoints_world_m": points.tolist(),
            "confidence": np.clip(confidences, 0.0, 1.0).tolist(),
            "mean_reprojection_error_px": errors.tolist(),
        }
        for _, player in views:
            player["pose_3d"] = pose_3d
        successful += 1

    output_prediction_dir = output_run_dir / "predictions"
    output_prediction_dir.mkdir(parents=True, exist_ok=True)
    for item in prediction_files:
        with (output_prediction_dir / item.path.name).open("w", encoding="utf-8") as handle:
            for frame_index in sorted(records_by_frame):
                record = records_by_frame[frame_index].get(item.camera_id)
                if record is None:
                    continue
                validate_group1_frame(record, final_handoff=False)
                handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

    created_at = datetime.now(timezone.utc).isoformat()
    metrics = {
        "schema_version": "triangulation_metrics/v1",
        "created_at": created_at,
        "match_id": match_id,
        "delivery_id": delivery_id,
        "candidate_identity_frames": len(observations),
        "eligible_multiview_identity_frames": sum(
            len(views) >= min_views for views in observations.values()
        ),
        "triangulated_identity_frames": len(raw),
        "fully_valid_identity_frames": successful,
        "triangulation_success_rate": successful / len(raw) if raw else 0.0,
    }
    manifest = {
        "schema_version": "triangulation_run/v1",
        "created_at": created_at,
        "task": "multi_view_3d_lift",
        "input_run_dir": str(input_run_dir),
        "output_run_dir": str(output_run_dir),
        "drive_root": str(drive_root),
        "match_id": match_id,
        "delivery_id": delivery_id,
        "config": {
            "reprojection_threshold_px": reprojection_threshold_px,
            "min_views": min_views,
            "ema_alpha": ema_alpha,
        },
    }
    _write_json(output_run_dir / "run_manifest.json", manifest)
    _write_json(output_run_dir / "triangulation_metrics.json", metrics)
    return metrics


def triangulate_legacy(args: argparse.Namespace) -> None:
    calibrations = load_calibration(args.calibration)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.predictions):
        player_id = row.get("global_player_id", row.get("player_id"))
        if player_id is None:
            continue
        grouped[(str(row["frame_id"]), str(player_id))].append(row)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for (frame_id, player_id), views in sorted(grouped.items()):
            usable = [view for view in views if view["camera_id"] in calibrations]
            if len(usable) < args.min_views:
                continue
            points, confidence, errors = triangulate_skeleton_ransac(
                np.asarray([view["keypoints"] for view in usable], dtype=float),
                np.asarray([calibrations[view["camera_id"]].projection_matrix for view in usable]),
                reprojection_threshold_px=args.reprojection_threshold_px,
                min_views=args.min_views,
            )
            payload = {
                "frame_id": frame_id,
                "timestamp_ns": int(usable[0].get("timestamp_ns", 0)),
                "player_id": player_id,
                "global_player_id": player_id,
                "calibration_id": usable[0].get("calibration_id"),
                "camera_ids": [view["camera_id"] for view in usable],
                "keypoints3d_world_m": np.column_stack([points, confidence]).tolist(),
                "mean_reprojection_error_px": float(np.nanmean(errors)) if np.isfinite(errors).any() else None,
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    canonical_values = (args.input_run_dir, args.output_run_dir, args.drive_root, args.delivery_id)
    if any(value is not None for value in canonical_values):
        if not all(value is not None for value in canonical_values):
            raise SystemExit("canonical mode requires --input-run-dir, --output-run-dir, --drive-root, --delivery-id")
        metrics = triangulate_canonical_run(
            input_run_dir=args.input_run_dir,
            output_run_dir=args.output_run_dir,
            drive_root=args.drive_root,
            delivery_id=args.delivery_id,
            cameras=args.camera,
            reprojection_threshold_px=args.reprojection_threshold_px,
            min_views=args.min_views,
            ema_alpha=args.ema_alpha,
        )
        print(f"P6: triangulated {metrics['fully_valid_identity_frames']} identity-frames")
        return 0
    if not all((args.predictions, args.calibration, args.output)):
        raise SystemExit("legacy mode requires --predictions, --calibration, and --output")
    triangulate_legacy(args)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

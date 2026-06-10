#!/usr/bin/env python3
"""Triangulate multi-view 2D keypoint JSONL predictions into 3D JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pose_estimation.schemas import CameraCalibration
from pose_estimation.triangulation import triangulate_skeleton_ransac


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="JSONL with frame_id,camera_id,player_id,keypoints")
    parser.add_argument("--calibration", required=True, help="JSON with cameras[].projection_matrix")
    parser.add_argument("--output", required=True)
    parser.add_argument("--reprojection-threshold-px", type=float, default=10.0)
    parser.add_argument("--min-views", type=int, default=2)
    return parser.parse_args()


def load_calibration(path: str | Path) -> dict[str, CameraCalibration]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    cameras = payload.get("cameras", payload)
    if isinstance(cameras, dict):
        cameras = list(cameras.values())
    return {camera["camera_id"]: CameraCalibration.from_dict(camera) for camera in cameras}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    args = parse_args()
    calibrations = load_calibration(args.calibration)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.predictions):
        grouped[(str(row["frame_id"]), str(row["player_id"]))].append(row)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for (frame_id, player_id), views in sorted(grouped.items()):
            usable = [view for view in views if view["camera_id"] in calibrations]
            if len(usable) < args.min_views:
                continue
            keypoints_by_view = np.asarray([view["keypoints"] for view in usable], dtype=float)
            projection_matrices = np.asarray(
                [calibrations[view["camera_id"]].projection_matrix for view in usable],
                dtype=float,
            )
            points3d, confidences, reprojection_errors = triangulate_skeleton_ransac(
                keypoints_by_view,
                projection_matrices,
                reprojection_threshold_px=args.reprojection_threshold_px,
                min_views=args.min_views,
            )
            payload = {
                "frame_id": frame_id,
                "timestamp_ns": int(usable[0].get("timestamp_ns", 0)),
                "player_id": player_id,
                "calibration_id": usable[0].get("calibration_id"),
                "camera_ids": [view["camera_id"] for view in usable],
                "keypoints3d_world_m": np.column_stack([points3d, confidences]).tolist(),
                "mean_reprojection_error_px": np.nanmean(reprojection_errors).item()
                if np.isfinite(reprojection_errors).any()
                else None,
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


"""Canonical JSONL boundary for P3 cross-camera association."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from pose_estimation.cricket.contract import KEYPOINT_COUNT, validate_group1_frame
from pose_estimation.cricket.geometry import ground_contact_pixel
from scripts.association.appearance import extract_appearance_descriptor
from scripts.association.associator import Correspondence, Detection3
from scripts.tracking.calibration import GroundPlaneCalibrator
from scripts.tracking.jsonl_io import read_prediction_frames
from scripts.tracking.runner import PredictionFile


RecordsByFrame = dict[int, dict[str, dict]]


def load_synchronized_records(
    prediction_files: Iterable[PredictionFile],
    delivery_id: str,
) -> RecordsByFrame:
    """Load canonical camera streams and group them by true synchronized frame index."""

    records: RecordsByFrame = {}
    for item in prediction_files:
        previous_frame: int | None = None
        for record in read_prediction_frames(item.path):
            validate_group1_frame(record)
            if record.get("delivery_id") != delivery_id:
                raise ValueError(f"{item.path} contains an unexpected delivery_id")
            if record.get("camera_id") != item.camera_id:
                raise ValueError(f"{item.path} contains an unexpected camera_id")
            if record.get("capture_group") != item.capture_group:
                raise ValueError(f"{item.path} contains an unexpected capture_group")
            frame_index = record.get("frame_index")
            if type(frame_index) is not int:
                raise ValueError(f"{item.path} contains a non-integer frame_index")
            if previous_frame is not None and frame_index <= previous_frame:
                raise ValueError(f"{item.path} frame_index values must be strictly increasing")
            previous_frame = frame_index
            camera_records = records.setdefault(frame_index, {})
            if item.camera_id in camera_records:
                raise ValueError(f"duplicate {item.camera_id} record at frame {frame_index}")
            camera_records[item.camera_id] = record
    if not records:
        raise RuntimeError(f"no prediction records found for delivery {delivery_id}")
    return records


def record_to_detections(
    record: dict,
    calibrator: GroundPlaneCalibrator | None = None,
    image_bgr: np.ndarray | None = None,
    *,
    ankle_confidence_min: float = 0.6,
    max_ankle_above_bbox_fraction: float = 0.25,
    foot_contact_mode: str = "legacy",
    ankle_height_m: float = 0.10,
    foot_horizontal_margin_frac: float = 0.15,
    foot_level_frac: float = 0.15,
) -> list[Detection3]:
    """Convert detection-bearing contract players to the association domain type."""

    detections: list[Detection3] = []
    camera_id = record["camera_id"]
    for player_index, player in enumerate(record.get("players", [])):
        pose = player.get("pose_2d")
        bbox = player.get("bbox_xywh_px")
        if pose is None or bbox is None:
            continue
        keypoints = np.asarray(pose.get("keypoints_px"), dtype=float)
        confidences = np.asarray(pose.get("confidence"), dtype=float)
        if keypoints.shape != (KEYPOINT_COUNT, 2) or confidences.shape != (KEYPOINT_COUNT,):
            continue
        if not np.isfinite(keypoints).all() or not np.isfinite(confidences).all():
            continue
        bbox_values = [float(value) for value in bbox]
        if len(bbox_values) != 4 or not np.isfinite(bbox_values).all():
            continue
        confidence = player.get("detection_confidence")
        confidence = float(np.mean(confidences)) if confidence is None else float(confidence)
        if not math.isfinite(confidence):
            continue
        ground_xy = None
        if calibrator is not None:
            contact = ground_contact_pixel(
                bbox_values,
                keypoints,
                confidences,
                ankle_confidence_min=ankle_confidence_min,
                max_ankle_above_bbox_fraction=max_ankle_above_bbox_fraction,
                mode=foot_contact_mode,
                ankle_height_m=ankle_height_m,
                horizontal_margin_frac=foot_horizontal_margin_frac,
                level_frac=foot_level_frac,
            )
            ground_xy = calibrator.image_to_ground_xy(contact)
        appearance = (
            extract_appearance_descriptor(image_bgr, player)
            if image_bgr is not None else None
        )
        detections.append(
            Detection3(
                cam_id=camera_id,
                player_index=player_index,
                bbox_xywh_px=bbox_values,
                keypoints_px=keypoints,
                keypoint_conf=confidences,
                confidence=float(np.clip(confidence, 0.0, 1.0)),
                local_track_id=player.get("local_track_id"),
                ground_xy=ground_xy,
                appearance=appearance,
            )
        )
    return detections


def apply_correspondences(
    camera_records: dict[str, dict],
    correspondences: Iterable[Correspondence],
) -> None:
    """Stamp association fields back by camera/player index without changing identity fields."""

    touched: set[tuple[str, int]] = set()
    for corr in correspondences:
        for camera_id, detection in corr.members.items():
            players = camera_records[camera_id].get("players", [])
            if detection.player_index >= len(players):
                raise IndexError(f"player index {detection.player_index} is invalid for {camera_id}")
            player = players[detection.player_index]
            player["single_camera"] = bool(corr.single_camera)
            player["track_confidence"] = float(np.clip(corr.track_confidence, 0.0, 1.0))
            player["global_player_id"] = None
            touched.add((camera_id, detection.player_index))

    # Malformed/null-pose players are not associable. Preserve the record and mark
    # their P3 status explicitly rather than silently retaining stale confidence.
    for camera_id, record in camera_records.items():
        for player_index, player in enumerate(record.get("players", [])):
            if (camera_id, player_index) not in touched:
                player["single_camera"] = True
                player["track_confidence"] = 0.0
                player["global_player_id"] = None
        validate_group1_frame(record)


def correspondence_row(frame_index: int, correspondences: Iterable[Correspondence]) -> dict:
    clusters = []
    for corr in correspondences:
        ground_xy = (
            [float(value) for value in corr.ground_xy]
            if np.asarray(corr.ground_xy).shape == (2,) and np.isfinite(corr.ground_xy).all()
            else None
        )
        clusters.append(
            {
                "cluster_id": int(corr.cluster_id),
                "members": [
                    {
                        "cam_id": camera_id,
                        "player_index": int(det.player_index),
                        "local_track_id": det.local_track_id,
                        "bbox_xywh_px": [float(value) for value in det.bbox_xywh_px],
                    }
                    for camera_id, det in sorted(corr.members.items())
                ],
                "ground_xy": ground_xy,
                "track_confidence": float(corr.track_confidence),
                "single_camera": bool(corr.single_camera),
                "mean_reprojection_error_px": corr.mean_reprojection_error_px,
                "cycle_consistent": bool(corr.cycle_consistent),
                "ground_spread_m": corr.ground_spread_m,
                "pose_descriptor": (
                    corr.pose_descriptor.to_json() if corr.pose_descriptor is not None else None
                ),
                "binding_id": corr.binding_id,
            }
        )
    return {"frame_index": int(frame_index), "clusters": clusters}


def write_prediction_streams(
    records_by_frame: RecordsByFrame,
    prediction_files: Iterable[PredictionFile],
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in prediction_files:
        output_path = output_dir / item.path.name
        with output_path.open("w", encoding="utf-8") as handle:
            for frame_index in sorted(records_by_frame):
                record = records_by_frame[frame_index].get(item.camera_id)
                if record is None:
                    continue
                validate_group1_frame(record)
                handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def write_correspondence_rows(rows: Iterable[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")

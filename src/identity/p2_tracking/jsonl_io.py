"""Tracking JSONL I/O, retroactive write buffer, and diagnostics."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterator

import numpy as np

from core.contract import validate_group1_frame
from identity.common.geometry import ground_contact_pixel
from core.calibration import GroundPlaneCalibrator
from identity.p2_tracking.config import TrackingConfig
from identity.p2_tracking.pose_vector import build_pose_vector
from identity.p2_tracking.tracker import CameraTracker, Detection


def read_prediction_frames(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _new_io_diagnostics() -> dict[str, int]:
    return {
        "malformed_detections_skipped": 0,
        "calibration_projection_failures": 0,
        "detection_confidence_pose_fallbacks": 0,
    }


def _detection_confidence(player: dict, diagnostics: dict[str, int]) -> float:
    if player.get("track_confidence") is not None:
        raise ValueError("tracking input must have track_confidence null")
    raw_confidence = player.get("detection_confidence")
    if raw_confidence is None:
        pose_confidences = np.asarray(player.get("pose_2d", {}).get("confidence", []), dtype=float)
        usable = pose_confidences[np.isfinite(pose_confidences)]
        if usable.size == 0:
            raise ValueError("tracking input has neither detection nor pose confidence")
        diagnostics["detection_confidence_pose_fallbacks"] += 1
        confidence = float(np.mean(usable))
    else:
        confidence = float(raw_confidence)
    if not math.isfinite(confidence) or confidence < 0.0 or confidence > 1.0:
        raise ValueError("detection_confidence must be in [0.0, 1.0]")
    return confidence


def frame_to_detections(
    record: dict,
    config: TrackingConfig,
    calibrator: GroundPlaneCalibrator | None = None,
    diagnostics: dict[str, int] | None = None,
) -> list[Detection]:
    detections: list[Detection] = []
    diagnostics = diagnostics if diagnostics is not None else _new_io_diagnostics()
    for player in record.get("players", []):
        if player.get("local_track_id") is not None:
            raise ValueError("tracking input must not already contain local_track_id")
        confidence = _detection_confidence(player, diagnostics)
        try:
            bbox_xywh = [float(value) for value in player.get("bbox_xywh_px", [])]
            if (
                len(bbox_xywh) != 4
                or not all(math.isfinite(value) for value in bbox_xywh)
                or bbox_xywh[2] <= 0.0
                or bbox_xywh[3] <= 0.0
            ):
                diagnostics["malformed_detections_skipped"] += 1
                continue
            pose_block = player.get("pose_2d", {})
            pose = build_pose_vector(
                pose_block.get("keypoints_px", []),
                pose_block.get("confidence", []),
                bbox_xywh,
                config,
            )
        except (TypeError, ValueError):
            diagnostics["malformed_detections_skipped"] += 1
            continue
        ground_xy = None
        if calibrator is not None:
            contact_pixel = ground_contact_pixel(
                bbox_xywh,
                np.asarray(pose_block.get("keypoints_px", []), dtype=float),
                np.asarray(pose_block.get("confidence", []), dtype=float),
                ankle_confidence_min=config.ankle_confidence_min,
                max_ankle_above_bbox_fraction=config.max_ankle_above_bbox_fraction,
            )
            ground_xy = calibrator.image_to_ground_xy(contact_pixel)
            if ground_xy is None:
                diagnostics["calibration_projection_failures"] += 1
        detections.append(
            Detection(
                bbox_xywh=bbox_xywh,
                pose=pose,
                confidence=confidence,
                player=player,
                ground_xy=ground_xy,
            )
        )
    return detections


def track_camera_file(
    input_path: str | Path,
    output_path: str | Path,
    diagnostics_path: str | Path,
    camera_id: str,
    capture_group: str,
    delivery_id: str,
    config: TrackingConfig,
    expected_frames: int = 600,
    calibrator: GroundPlaneCalibrator | None = None,
) -> dict:
    output_path = Path(output_path)
    diagnostics_path = Path(diagnostics_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)

    tracker = CameraTracker(camera_id, config)
    io_diagnostics = _new_io_diagnostics()
    # Buffer holds (record) until its tentative tracks resolve; players are stamped in place,
    # so we can flush after `tentative_confirm_window` frames of look-ahead.
    buffer: list[dict] = []
    frames_read = 0

    with output_path.open("w", encoding="utf-8") as out:
        def flush(up_to: int) -> None:
            while len(buffer) > up_to:
                record = buffer.pop(0)
                validate_group1_frame(record)
                out.write(json.dumps(record, sort_keys=True) + "\n")

        # Drive the tracker with a per-camera ordinal (0, 1, 2, ...) so confirmation-window and
        # dormancy math are independent of the source frame_index (which may be large/sparse).
        # The output record keeps its own true frame_index untouched.
        for ordinal, record in enumerate(read_prediction_frames(input_path)):
            validate_group1_frame(record)
            if record.get("camera_id") != camera_id:
                raise ValueError(f"{input_path} contains camera_id={record.get('camera_id')}, expected {camera_id}")
            if record.get("delivery_id") != delivery_id:
                raise ValueError(f"{input_path} contains delivery_id={record.get('delivery_id')}, expected {delivery_id}")
            if record.get("capture_group") != capture_group:
                raise ValueError(
                    f"{input_path} contains capture_group={record.get('capture_group')}, expected {capture_group}"
                )
            frames_read += 1
            detections = frame_to_detections(record, config, calibrator, io_diagnostics)
            tracker.update(detections, frame_index=ordinal)
            buffer.append(record)
            flush(config.tentative_confirm_window)  # keep a look-ahead window buffered

        tracker.finalize()
        flush(0)  # drain remaining buffer at EOF

    combined_tracker_diagnostics = dict(tracker.diagnostics)
    for key, value in io_diagnostics.items():
        combined_tracker_diagnostics[key] = combined_tracker_diagnostics.get(key, 0) + value

    diagnostics = {
        "camera_id": camera_id,
        "delivery_id": delivery_id,
        "status": "ok",
        "error": None,
        "frames_expected": expected_frames,
        "frames_read": frames_read,
        "input_jsonl": str(Path(input_path)),
        "output_jsonl": str(output_path),
        "capture_group": capture_group,
        "calibration_enabled": calibrator is not None,
        **combined_tracker_diagnostics,
    }
    with diagnostics_path.open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return diagnostics

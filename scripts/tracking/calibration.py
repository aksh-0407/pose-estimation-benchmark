"""Calibration helpers for per-camera tracking."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GroundPlaneCalibrator:
    """Project image points to the z=0 world plane for one calibrated camera."""

    camera_id: str
    image_to_ground_h: np.ndarray

    def image_to_ground_xy(self, pixel_xy: np.ndarray) -> np.ndarray | None:
        point = np.asarray(pixel_xy, dtype=float)
        if point.shape != (2,) or not np.isfinite(point).all():
            return None
        homogeneous = self.image_to_ground_h @ np.array([point[0], point[1], 1.0], dtype=float)
        if abs(float(homogeneous[2])) < 1e-12:
            return None
        xy = homogeneous[:2] / homogeneous[2]
        if not np.isfinite(xy).all():
            return None
        return xy.astype(float)

    def bbox_bottom_center_ground_xy(self, bbox_xywh_px: list[float]) -> np.ndarray | None:
        if len(bbox_xywh_px) != 4:
            return None
        x, y, w, h = [float(v) for v in bbox_xywh_px]
        if not np.isfinite([x, y, w, h]).all() or w <= 0.0 or h <= 0.0:
            return None
        return self.image_to_ground_xy(np.array([x + w / 2.0, y + h], dtype=float))


def current_calibration_dir(drive_root: str | Path, match_id: str) -> Path:
    return Path(drive_root) / "dataset" / "calibration-data" / match_id / "calibration_data"


def load_projection_matrices_from_drive(drive_root: str | Path, match_id: str) -> dict[str, np.ndarray]:
    calibration_dir = current_calibration_dir(drive_root, match_id)
    path = calibration_dir / "Bundle_Adjusted_extrinsics.json"
    if not path.exists():
        raise FileNotFoundError(f"missing calibration file: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)
    raw_matrices = payload.get("projection_matrices")
    if not isinstance(raw_matrices, dict):
        raise ValueError(f"projection_matrices missing from {path}")
    matrices: dict[str, np.ndarray] = {}
    for key, value in raw_matrices.items():
        if not key.startswith("C"):
            continue
        try:
            camera_id = f"cam_{int(key[1:]):02d}"
        except ValueError:
            continue
        matrix = np.asarray(value, dtype=float)
        if matrix.shape != (3, 4) or not np.isfinite(matrix).all():
            raise ValueError(f"invalid projection matrix for {key} in {path}")
        matrices[camera_id] = matrix
    if not matrices:
        raise ValueError(f"no usable projection matrices in {path}")
    return matrices


def build_ground_calibrators(
    drive_root: str | Path,
    match_id: str,
    camera_ids: list[str] | None = None,
) -> dict[str, GroundPlaneCalibrator]:
    projection_matrices = load_projection_matrices_from_drive(drive_root, match_id)
    wanted = set(camera_ids) if camera_ids else set(projection_matrices)
    calibrators: dict[str, GroundPlaneCalibrator] = {}
    for camera_id in sorted(wanted):
        matrix = projection_matrices.get(camera_id)
        if matrix is None:
            raise ValueError(f"calibration missing projection matrix for {camera_id}")
        ground_to_image_h = matrix[:, [0, 1, 3]]
        try:
            image_to_ground_h = np.linalg.solve(ground_to_image_h, np.eye(3, dtype=float))
        except np.linalg.LinAlgError as exc:
            raise ValueError(f"ground-plane homography is singular for {camera_id}") from exc
        if not np.isfinite(image_to_ground_h).all():
            raise ValueError(f"ground-plane homography is non-finite for {camera_id}")
        calibrators[camera_id] = GroundPlaneCalibrator(
            camera_id=camera_id,
            image_to_ground_h=image_to_ground_h,
        )
    return calibrators

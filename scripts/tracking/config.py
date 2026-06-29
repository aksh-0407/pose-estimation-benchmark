"""Per-camera tracking configuration."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclass(frozen=True)
class TrackingConfig:
    # Stage thresholds
    stage1_confidence_threshold: float = 0.5
    stage2_confidence_min: float = 0.1
    cost_accept_threshold: float = 0.7
    lowconf_can_spawn: bool = True
    # Cost matrix weights
    iou_alpha: float = 0.6
    pose_beta: float = 0.4
    # Pose vector
    pose_keypoint_confidence_min: float = 0.3
    min_shared_keypoints: int = 6
    scale_min_frac_bbox_h: float = 0.05
    # Spatial / motion gating
    chi2_gate: float = 9.21
    gate_bbox_factor: float = 1.5
    gate_max_distance_px: float = 600.0
    v_max_px_per_frame: float = 120.0
    # Calibration-assisted ground-plane gating
    frame_rate_fps: float = 50.0
    ground_vmax_mps: float = 9.0
    ground_gate_base_m: float = 1.5
    ground_cost_weight: float = 0.2
    ankle_confidence_min: float = 0.6
    max_ankle_above_bbox_fraction: float = 0.25
    # Dormant re-ID
    pose_cosine_reid_threshold: float = 0.25
    reid_ambiguity_margin: float = 0.05
    dormant_max_frames: int = 60
    # Kalman stability
    kalman_cov_trace_max: float = 1.0e6
    # Track confirmation
    tentative_confirm_hits: int = 3
    tentative_confirm_window: int = 5
    # Gallery
    pose_gallery_size: int = 30
    gallery_repr: str = "medoid"

    def __post_init__(self) -> None:
        _require_range("stage1_confidence_threshold", self.stage1_confidence_threshold, 0.0, 1.0)
        _require_range("stage2_confidence_min", self.stage2_confidence_min, 0.0, 1.0)
        _require_range("cost_accept_threshold", self.cost_accept_threshold, 0.0, 2.0)
        if self.stage2_confidence_min > self.stage1_confidence_threshold:
            raise ValueError("stage2_confidence_min must be <= stage1_confidence_threshold")

        for name in ("iou_alpha", "pose_beta", "chi2_gate", "gate_bbox_factor",
                     "gate_max_distance_px", "v_max_px_per_frame", "frame_rate_fps",
                     "ground_vmax_mps", "ground_gate_base_m", "kalman_cov_trace_max"):
            _require_positive(name, getattr(self, name))
        _require_range("pose_keypoint_confidence_min", self.pose_keypoint_confidence_min, 0.0, 1.0)
        _require_range("pose_cosine_reid_threshold", self.pose_cosine_reid_threshold, 0.0, 2.0)
        _require_range("reid_ambiguity_margin", self.reid_ambiguity_margin, 0.0, 2.0)
        _require_range("ground_cost_weight", self.ground_cost_weight, 0.0, 1.0)
        _require_range("ankle_confidence_min", self.ankle_confidence_min, 0.0, 1.0)
        _require_range(
            "max_ankle_above_bbox_fraction", self.max_ankle_above_bbox_fraction, 0.0, 1.0
        )

        for name in ("min_shared_keypoints", "dormant_max_frames", "tentative_confirm_hits",
                     "tentative_confirm_window", "pose_gallery_size"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_shared_keypoints > 17:
            raise ValueError("min_shared_keypoints must be <= 17")
        if self.gallery_repr not in {"medoid", "first"}:
            raise ValueError("gallery_repr must be one of: medoid, first")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_positive(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive number")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be a positive number")


def _require_range(name: str, value: float, low: float, high: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < low or numeric > high:
        raise ValueError(f"{name} must be in [{low}, {high}]")


def _coerce_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"{field_name} must be a boolean")


def load_tracking_config(path: str | Path | None) -> TrackingConfig:
    if path is None:
        return TrackingConfig()
    with Path(path).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}
    type_hints = get_type_hints(TrackingConfig)
    unknown = set(raw) - set(type_hints)
    if unknown:
        raise ValueError(f"unknown tracking config keys: {sorted(unknown)}")

    coerced: dict[str, Any] = {}
    for name, val in raw.items():
        field_type = type_hints[name]
        if field_type in {float, int} and isinstance(val, bool):
            raise ValueError(f"{name} must be numeric, not boolean")
        if field_type is float and not isinstance(val, float):
            coerced[name] = float(val)
        elif field_type is int and type(val) is not int:
            coerced[name] = int(val)
        elif field_type is bool:
            coerced[name] = _coerce_bool(val, field_name=name)
        else:
            coerced[name] = val
    return TrackingConfig(**coerced)

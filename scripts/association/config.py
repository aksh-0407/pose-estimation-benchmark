"""Cross-camera association (P3) configuration.

Same frozen-dataclass + validated-YAML-loader pattern as
scripts/tracking/config.py. Every magic number Vedant hard-coded is exposed
here so the run manifest records exactly what produced a result.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, get_type_hints

import yaml

_MATCHING_MODES = {"pairwise_anchor", "multiway_cycle"}
_DEFAULT_ANCHOR_PRIORITY = ["cam_01", "cam_04", "cam_02", "cam_03", "cam_05", "cam_06", "cam_07"]
# FACING (co-observing) pairs, per configs/facing.jpeg and verified against the
# calibration optical axes: each pair looks at the SAME ground strip from opposite
# sides. NOT the diametrically-opposite *positions* (C2/C5, C3/C6) — those look at
# different strips and never co-observe. The association runner re-derives these from
# the projection matrices and overrides this default, so calibration is the source of
# truth even if this list is edited wrongly.
_DEFAULT_OPPOSITE_PAIRS = [
    ["cam_01", "cam_04"],
    ["cam_02", "cam_06"],
    ["cam_03", "cam_05"],
]


@dataclass(frozen=True)
class P3AssociationConfig:
    # Image size (for epipole-in-image degeneracy test)
    image_w: int = 2560
    image_h: int = 1440
    # Matching strategy
    matching_mode: str = "multiway_cycle"  # or "pairwise_anchor"
    # Degeneracy detection
    baseline_angle_degen_deg: float = 20.0
    degenerate_pairs: list = field(default_factory=list)  # explicit [[cam_a, cam_b], ...]
    # Pair fine-score weights
    w_epi: float = 0.6
    w_tri: float = 0.4
    parallax_min_deg: float = 10.0
    parallax_full_deg: float = 25.0
    # Calibration stats (replace Vedant's fake auto-compute; empirically tuned)
    mu_fine_score: float = 15.0
    sigma_fine_score: float = 5.0
    dummy_cost_scale: float = 3.0
    # Foot / keypoint confidence gates
    ankle_conf_min: float = 0.6
    keypoint_match_conf_min: float = 0.5
    max_ankle_above_bbox_fraction: float = 0.25
    # Ground-plane association (the primary cue for calibrated cricket views)
    ground_distance_gate_m: float = 3.5
    opposite_pair_ground_gate_m: float = 2.5
    ground_cluster_gate_m: float = 2.75
    ground_weight: float = 0.65
    epipolar_weight: float = 0.15
    epipolar_scale_px: float = 12.0
    appearance_enabled: bool = True
    appearance_weight: float = 0.20
    pair_unmatched_cost: float = 0.75
    temporal_link_bonus: float = 0.25
    temporal_confirm_frames: int = 3
    opposite_camera_pairs: list = field(default_factory=lambda: [list(pair) for pair in _DEFAULT_OPPOSITE_PAIRS])
    # Multi-way cycle-consistency reconciliation
    cycle_xy_tol_m: float = 0.5
    cycle_reproj_tol_px: float = 12.0
    triangulation_min_views: int = 2
    triangulation_reproj_threshold_px: float = 10.0
    # Confidence / gating
    chi2_gate_2dof: float = 5.991
    confidence_high: float = 0.7
    confidence_discard: float = 0.3
    single_camera_confidence: float = 0.3
    untracked_single_camera_confidence: float = 0.2
    # Anchor selection (sticky, with hysteresis)
    anchor_hysteresis_margin: int = 2
    anchor_hysteresis_frames: int = 3
    anchor_priority: list = field(default_factory=lambda: list(_DEFAULT_ANCHOR_PRIORITY))

    def __post_init__(self) -> None:
        for name in ("image_w", "image_h"):
            _require_positive_int(name, getattr(self, name))
        if self.matching_mode not in _MATCHING_MODES:
            raise ValueError(f"matching_mode must be one of {sorted(_MATCHING_MODES)}")
        for name in ("baseline_angle_degen_deg", "parallax_full_deg", "mu_fine_score",
                     "sigma_fine_score", "dummy_cost_scale", "cycle_xy_tol_m",
                     "cycle_reproj_tol_px", "triangulation_reproj_threshold_px",
                     "chi2_gate_2dof", "ground_distance_gate_m",
                     "opposite_pair_ground_gate_m", "ground_cluster_gate_m",
                     "epipolar_scale_px", "pair_unmatched_cost"):
            _require_positive(name, getattr(self, name))
        if self.parallax_min_deg < 0.0 or self.parallax_min_deg >= self.parallax_full_deg:
            raise ValueError("require 0 <= parallax_min_deg < parallax_full_deg")
        for name in ("w_epi", "w_tri", "ankle_conf_min", "keypoint_match_conf_min",
                     "confidence_high", "confidence_discard", "single_camera_confidence",
                     "untracked_single_camera_confidence",
                     "max_ankle_above_bbox_fraction", "ground_weight", "epipolar_weight",
                     "appearance_weight", "temporal_link_bonus"):
            _require_range(name, getattr(self, name), 0.0, 1.0)
        if self.confidence_discard > self.confidence_high:
            raise ValueError("confidence_discard must be <= confidence_high")
        for name in ("triangulation_min_views", "anchor_hysteresis_margin", "anchor_hysteresis_frames",
                     "temporal_confirm_frames"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.triangulation_min_views < 2:
            raise ValueError("triangulation_min_views must be >= 2")
        _validate_pair_list("degenerate_pairs", self.degenerate_pairs)
        _validate_pair_list("opposite_camera_pairs", self.opposite_camera_pairs)
        if not all(isinstance(cam, str) for cam in self.anchor_priority):
            raise ValueError("anchor_priority must be a list of camera-id strings")

    @property
    def image_wh(self) -> tuple[int, int]:
        return (self.image_w, self.image_h)

    def huber_delta(self) -> float:
        """Huber transition calibrated as the ~90th percentile correct-match fine score."""
        return float(self.mu_fine_score + 1.645 * self.sigma_fine_score)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_positive(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a positive number")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{name} must be a positive number")


def _require_positive_int(name: str, value: Any) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_range(name: str, value: float, low: float, high: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < low or numeric > high:
        raise ValueError(f"{name} must be in [{low}, {high}]")


def _validate_pair_list(name: str, value: Any) -> None:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list of [cam_a, cam_b] pairs")
    for pair in value:
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or not all(isinstance(cam, str) for cam in pair)):
            raise ValueError(f"{name} entries must be [cam_a, cam_b] string pairs")


def load_association_config(path: str | Path | None) -> P3AssociationConfig:
    if path is None:
        return P3AssociationConfig()
    with Path(path).open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}
    type_hints = get_type_hints(P3AssociationConfig)
    unknown = set(raw) - set(type_hints)
    if unknown:
        raise ValueError(f"unknown association config keys: {sorted(unknown)}")

    coerced: dict[str, Any] = {}
    for name, val in raw.items():
        field_type = type_hints[name]
        if field_type in {float, int} and isinstance(val, bool):
            raise ValueError(f"{name} must be numeric, not boolean")
        if field_type is float and not isinstance(val, float):
            coerced[name] = float(val)
        elif field_type is int and type(val) is not int:
            coerced[name] = int(val)
        else:
            coerced[name] = val
    return P3AssociationConfig(**coerced)

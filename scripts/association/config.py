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
_ASSOCIATION_MODES = {"per_frame", "tracklet_graph"}
_CALIBRATION_MODES = {"auto", "file", "defaults"}
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
    # Pose-shape descriptor (view-invariant 3D bone-length ratios). Emitted per
    # multi-view correspondence as a SOFT signal: it feeds the P4a temporal
    # tie-breaker and (via torso plausibility) a small confidence down-weight for
    # likely chimera clusters. Never a hard gate -- see changes_tbd.md.
    pose_descriptor_enabled: bool = True
    pose_min_conf: float = 0.3
    pose_parallax_min_deg: float = 12.0
    pose_shoulder_width_m: list = field(default_factory=lambda: [0.25, 0.65])
    pose_hip_width_m: list = field(default_factory=lambda: [0.15, 0.55])
    pose_torso_len_m: list = field(default_factory=lambda: [0.35, 0.85])
    pose_torso_tilt_max_deg: float = 75.0
    pose_confidence_penalty: float = 0.15
    # --- tracklet-graph identity (association_mode: tracklet_graph) ---------
    # per_frame reproduces the historical per-frame clustering byte-for-byte;
    # tracklet_graph decides identity once per P2-tracklet pair over the whole
    # delivery and emits per-frame correspondences from those bindings.
    association_mode: str = "per_frame"
    graph_sample_gate_m: float = 6.0          # wide evidence gate (also feeds calibration)
    graph_min_covis_frames: int = 10          # min gated co-visible frames for an edge
    graph_covis_full_frames: int = 40         # support saturation for full edge weight
    # Min aggregated LLR to merge. Above the single-cue positive cap by design:
    # capped ground agreement alone can never merge two tracklets — at least one
    # corroborating cue (posture, motion, appearance) must also vote "same".
    graph_llr_merge_threshold: float = 2.0
    graph_llr_veto: float = -4.5              # only a CONFIDENT contradiction blocks a merge
    graph_llr_positive_cap: float = 1.5       # per-cue cap on "same" evidence
    graph_llr_ground_neg_clip: float = 2.0    # ground's negative clip (bias tolerance)
    graph_move_margin: float = 0.5            # refinement move hysteresis
    graph_refine_passes: int = 2
    graph_cannot_link_overlap_frames: int = 3  # same-camera overlap => different people
    graph_rescue_min_covis: int = 30          # evidence floor for constraint rescues
    graph_hard_dist_gate_m: float = 2.75      # median ground residual ceiling for edges
    graph_motion_enabled: bool = True
    graph_min_app_samples: int = 5
    # Per-detection ground covariance (pixel noise through the ray-plane Jacobian).
    # The floor absorbs cross-camera calibration bias: measured same-player ground
    # residuals on this rig are ~0.7-1.2 m median, far above pure pixel noise.
    ground_sigma_px_base: float = 2.0
    ground_sigma_px_bbox_frac: float = 0.01   # + frac * bbox_height_px
    ground_var_floor_m: float = 0.4
    # Feet-unusable recovery: when a bbox reaches the frame bottom with no
    # confident ankle, re-anchor the ground point on an upper-body landmark's
    # height plane (hips -> shoulders -> bbox-top-as-head). approx_var_floor_m is
    # the honest positional sigma of that estimate.
    approx_feet_enabled: bool = True
    approx_hip_height_m: float = 0.93
    approx_shoulder_height_m: float = 1.42
    approx_head_height_m: float = 1.78
    approx_var_floor_m: float = 0.8
    # Synthetic tracklets: chain persistent untracked detections (e.g. umpires P2
    # never tracked) by ground continuity so the graph can bind them.
    synthetic_tracklets_enabled: bool = True
    syn_chain_gate_m: float = 1.2
    syn_chain_max_gap_frames: int = 150
    syn_min_confidence: float = 0.2
    # Tracklet purity: split a P2 tracklet at kinematically impossible ground jumps
    purity_split_enabled: bool = True
    purity_jump_slack: float = 1.5
    purity_jump_floor_m: float = 1.5
    frame_rate_fps: float = 50.0
    kinematic_v_max_mps: float = 9.0
    # Ground-anchored (billboard) posture cue -- the pose-shape identity layer
    posture_enabled: bool = True
    posture_min_samples: int = 8
    # Cue calibration: auto = bootstrap same/diff populations from this delivery,
    # file = load cue_calibration_path, defaults = conservative built-ins.
    calibration_mode: str = "auto"
    cue_calibration_path: str = ""
    anchor_pair_min_frames: int = 30
    anchor_pair_dist_m: float = 1.5
    anchor_pair_isolation_m: float = 3.0
    diff_pair_min_dist_m: float = 3.0
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
        if self.association_mode not in _ASSOCIATION_MODES:
            raise ValueError(f"association_mode must be one of {sorted(_ASSOCIATION_MODES)}")
        if self.calibration_mode not in _CALIBRATION_MODES:
            raise ValueError(f"calibration_mode must be one of {sorted(_CALIBRATION_MODES)}")
        if not isinstance(self.cue_calibration_path, str):
            raise ValueError("cue_calibration_path must be a string (may be empty)")
        for name in ("graph_sample_gate_m", "graph_covis_full_frames",
                     "graph_hard_dist_gate_m", "graph_llr_positive_cap",
                     "graph_llr_ground_neg_clip",
                     "ground_sigma_px_base",
                     "ground_var_floor_m", "purity_jump_slack", "purity_jump_floor_m",
                     "frame_rate_fps", "kinematic_v_max_mps",
                     "anchor_pair_dist_m", "anchor_pair_isolation_m",
                     "diff_pair_min_dist_m",
                     "approx_hip_height_m", "approx_shoulder_height_m",
                     "approx_head_height_m", "approx_var_floor_m",
                     "syn_chain_gate_m"):
            _require_positive(name, getattr(self, name))
        for name in ("graph_min_covis_frames", "graph_refine_passes", "graph_rescue_min_covis",
                     "graph_cannot_link_overlap_frames", "graph_min_app_samples",
                     "posture_min_samples", "anchor_pair_min_frames",
                     "syn_chain_max_gap_frames"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        _require_range("syn_min_confidence", self.syn_min_confidence, 0.0, 1.0)
        for name in ("approx_feet_enabled", "synthetic_tracklets_enabled"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        for name in ("graph_llr_merge_threshold", "graph_llr_veto", "graph_move_margin"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be a finite number")
        _require_range("ground_sigma_px_bbox_frac", self.ground_sigma_px_bbox_frac, 0.0, 1.0)
        for name in ("graph_motion_enabled", "purity_split_enabled", "posture_enabled"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if self.anchor_pair_dist_m >= self.diff_pair_min_dist_m:
            raise ValueError("anchor_pair_dist_m must be < diff_pair_min_dist_m")
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
                     "appearance_weight", "temporal_link_bonus",
                     "pose_min_conf", "pose_confidence_penalty"):
            _require_range(name, getattr(self, name), 0.0, 1.0)
        if type(self.pose_descriptor_enabled) is not bool:
            raise ValueError("pose_descriptor_enabled must be a boolean")
        _require_positive("pose_parallax_min_deg", self.pose_parallax_min_deg)
        _require_positive("pose_torso_tilt_max_deg", self.pose_torso_tilt_max_deg)
        for name in ("pose_shoulder_width_m", "pose_hip_width_m", "pose_torso_len_m"):
            _require_meter_range(name, getattr(self, name))
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


def _require_meter_range(name: str, value: Any) -> None:
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in value)):
        raise ValueError(f"{name} must be a [low, high] pair of metres")
    low, high = float(value[0]), float(value[1])
    if not (math.isfinite(low) and math.isfinite(high)) or low <= 0.0 or low >= high:
        raise ValueError(f"{name} must satisfy 0 < low < high")


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

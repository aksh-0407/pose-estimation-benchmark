"""Cross-camera association (P3) configuration.

Same frozen-dataclass + validated-YAML-loader pattern as
src/identity/p2_tracking/config.py. Every magic number Vedant hard-coded is exposed
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
    # H4 (per-frame mode only): exponential decay of temporal-link evidence per
    # frame; 1.0 = legacy no-decay.
    temporal_link_decay: float = 1.0
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
    # Corroboration-aware single-cue merge (ID-1 under-merge on the facing pairs).
    # On the low-parallax facing pairs appearance abstains, motion abstains for
    # static players, and posture can abstain for crouched/oblique bodies, leaving
    # ground alone -- which the positive cap (1.5) holds below the 2.0 threshold, so
    # a genuine same-player pair never merges (the 0.50 cross-camera agreement on
    # _7). When enabled, a second pass merges an edge in [single, threshold) ONLY if
    # it has full co-visible support, NO observable cue disagrees, it is the mutual
    # unambiguous best for both endpoints, and it passes the cannot-link/veto check.
    graph_corrob_merge: bool = False
    graph_llr_merge_single: float = 1.2
    # Parallax-adaptive facing-pair gate: the facing (opposite) pairs use the tighter
    # opposite_pair_ground_gate_m (2.5) which, under foot-projection noise, can itself
    # split a correct 2-view merge. When enabled the graph hard-distance gate is
    # widened by this factor for facing pairs (lean on posture/appearance there),
    # never below the general gate. 1.0 = disabled (byte-identical).
    graph_facing_gate_scale: float = 1.0
    graph_llr_ground_neg_clip: float = 2.0    # ground's negative clip (bias tolerance)
    graph_move_margin: float = 0.5            # refinement move hysteresis
    graph_refine_passes: int = 2
    graph_cannot_link_overlap_frames: int = 3  # same-camera overlap => different people
    graph_rescue_min_covis: int = 30          # evidence floor for constraint rescues
    # Fragment recovery + binding hygiene (the anti-id-explosion controls):
    # fragments riding a binding's fused trajectory get attached to it; clusters
    # that are neither multi-camera nor one long stable track get NO binding id.
    graph_traj_attach_gate_m: float = 1.5
    binding_min_single_frames: int = 150
    graph_hard_dist_gate_m: float = 2.75      # median ground residual ceiling for edges
    graph_motion_enabled: bool = True
    # G6: the motion-cue shape parameters were hard-coded magic numbers while every
    # other cue is calibrated/configurable; promoted here (values unchanged).
    graph_motion_speed_full_mps: float = 2.0     # both clearly moving above this
    graph_motion_speed_still_mps: float = 0.7    # clearly standing below this
    graph_motion_gain: float = 1.5               # llr = gain * (cos - offset)
    graph_motion_cos_offset: float = 0.35
    graph_motion_llr_min: float = -2.5
    graph_motion_llr_max: float = 0.75
    graph_motion_still_llr: float = -1.5         # one sprints, one stands => "different"
    graph_min_app_samples: int = 5
    # Per-detection ground covariance (pixel noise through the ray-plane Jacobian).
    # The floor absorbs cross-camera calibration bias: measured same-player ground
    # residuals on this rig are ~0.7-1.2 m median, far above pure pixel noise.
    ground_sigma_px_base: float = 2.0
    ground_sigma_px_bbox_frac: float = 0.01   # + frac * bbox_height_px
    ground_var_floor_m: float = 0.4
    # Wave-5b: contested-camera evidence down-weighting. When two detections in the
    # SAME camera overlap heavily (bowler/non-striker crossing in a facing pair),
    # that camera cannot separate the two players: its ground evidence is unreliable
    # and its appearance/posture crops bleed one identity into the other. Marking
    # both detections "contested" makes the z0_reproj ground solve and the per-view
    # covariances ride on the cameras where the players ARE distinct (e.g. C1/C4
    # while C2-C6 overlap), and mutes identity-descriptor sampling from the merged
    # boxes. The merge GATE is untouched. 0.0 disables (byte-identical).
    contested_iou: float = 0.0
    contested_conf_scale: float = 0.25      # z0_reproj weight multiplier for contested members
    contested_sigma_scale: float = 2.5      # foot-pixel sigma multiplier -> per-view ground cov
    contested_mute_appearance: bool = True  # skip appearance/posture/kp samples when contested
    # V2-L3: when a majority of a cluster's views flag airborne, emit the vertical
    # ground projection of the triangulated hip midpoint instead of the biased z=0
    # foot solve. Emit-only (gate unchanged). Default off (byte-identical).
    airborne_pelvis_emit: bool = False
    # Cross-camera ground fusion for the EMITTED cluster position (feeds P4 Kalman +
    # ground_tracks). The merge GATE (max pairwise spread) is UNCHANGED across all
    # modes, so clustering/identity is byte-identical; only the reported position moves.
    #   "median"     - historical unweighted median of per-camera homography points.
    #   "z0_reproj"  - joint z=0-constrained robust (Huber) reprojection minimisation
    #                  over every member's full projection matrix. Uses the calibration
    #                  directly; well-posed on low-parallax facing pairs; lands on the
    #                  reprojection-optimal foot to ~cm. RECOMMENDED. Empirically ~0.016 m
    #                  from the triangulated foot vs 0.176 m for the median (delivery 1).
    #   "robust_cov" - inverse ground-covariance IRLS fusion (kept for A/B; on this rig
    #                  it under-performs z0_reproj because pixel-space reprojection, not
    #                  metric variance, is the criterion the calibration nails).
    ground_fusion_mode: str = "median"        # or "z0_reproj" | "robust_cov"
    ground_reproj_huber_px: float = 8.0       # px residual before a view is down-weighted
    ground_fusion_huber_delta: float = 2.5    # (robust_cov only) Mahalanobis-sqrt cutoff
    # Foot-contact pixel (ISSUE-7). "legacy" = historical (lower confident ankle else
    # bbox bottom-centre, projected as if on the ground). "v2" = ankle MIDPOINT as the
    # cross-camera-consistent reference when both feet are down (F4/F6), tighter vertical
    # + new horizontal plausibility (F3), and the ankle height reported so the z0_reproj
    # solver back-projects onto z=ankle_height instead of z=0 (removes the ~10 cm bias, F2).
    # "v3" (campaign fix F4) = prefer the Halpe-26 heel/toe keypoints from
    # pose_2d_native — true ground-contact landmarks (~2 cm above ground vs the
    # ankle's ~10 cm) — falling back to the v2 ankle stack when unavailable.
    foot_contact_mode: str = "legacy"         # or "v2" / "v3"
    ankle_height_m: float = 0.10
    foot_horizontal_margin_frac: float = 0.15
    foot_level_frac: float = 0.15
    foot_kp_conf_min: float = 0.5             # v3: min confidence for a heel/toe landmark
    foot_height_m: float = 0.02               # v3: heel/toe landmark height above ground
    # C5: the documented single-camera ankle-height emit (~0.94 m grazing-angle bias
    # fix, methods-log M5) was never wired to the production emit path — single-member
    # clusters emitted the legacy z=0 back-projection. When enabled, the EMITTED
    # position back-projects the emit-path foot pixel onto its landmark height plane;
    # the clustering gate keeps the legacy foot (identity invariant). Off = legacy.
    single_cam_height_emit: bool = False
    # Temporal smoothing of the EMITTED foot pixel per (camera, tracklet), F7. Odd
    # window for a centred median (robust to single-frame ankle spikes); 1 = disabled.
    # Emit-only, so identity is unchanged.
    foot_smooth_window: int = 1
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
    # F6b: emit each binding's pooled posture aggregate in correspondences.jsonl so
    # P4a can veto teleports / gate re-entries on the facing-pair-capable body-shape
    # cue. Off = rows are byte-identical to the baseline.
    emit_posture: bool = False
    # H3 policy (see PostureAccumulator.add): keep stature samples for players whose
    # posture could not be determined (feet cut off). Off = legacy strict policy.
    posture_keep_upright_unknown: bool = False
    # F9a: emit the 2x2 ground covariance per cluster (GN posterior for multi-view
    # z0_reproj, inflated per-view Jacobian model for single-camera). Feeds the P4
    # uncertainty-aware Kalman R (F10). Off = rows byte-identical to the baseline.
    emit_ground_cov: bool = False
    single_cam_cov_inflation: float = 4.0
    airborne_cov_scale: float = 4.0
    airborne_ankle_bbox_frac: float = 0.25
    # F9c: after the graph solve, lift each binding's multi-view frames to 3D and
    # log the per-binding purity report (torso-residual chimera signature, pooled
    # bone-ratio descriptor, stature) into diagnostics + metrics. Read-only —
    # nothing feeds back into clustering yet (that is Wave 3/4). Off = no extra
    # compute, byte-identical outputs.
    graph_lift_feedback: bool = False
    graph_lift_stride: int = 5                # lift every Nth frame (cost control)
    graph_chimera_torso_residual_px: float = 20.0
    graph_chimera_frame_fraction: float = 0.3
    # F11: pose-shape as a PRIMARY cluster-level cue. After the pairwise merge
    # rounds, each multi-camera cluster is lifted to 3D and its bone-ratio
    # descriptor + metric stature become a second corroboration round: two
    # compatible clusters with agreeing geometry AND agreeing body shape merge
    # even where the per-cue positive cap holds the pairwise round below the
    # threshold (the facing-pair under-merge, ID-1). The shape LLR is
    # self-calibrated per delivery (same = temporal halves of one cluster,
    # diff = co-visible distinct clusters) and abstains when starved.
    # W9 union-lift merge: same ground location + one coherent 3D skeleton across
    # ALL member views => one person. The geometric fix for facing-pair split
    # identities (ghost-under-player swaps). Default off (byte-identical).
    graph_union_lift_merge: bool = False
    graph_union_colocate_m: float = 1.0       # median co-frame distance gate
    graph_union_min_co_frames: int = 25       # co-located frames required
    graph_union_min_lift_frames: int = 6      # union lifts required for the test
    graph_union_torso_p50_px: float = 20.0    # union torso reproj residual gate
    graph_union_posture_max_z: float = 3.0    # billboard stature agreement gate
    graph_shape_enabled: bool = False
    graph_shape_min_frames: int = 8           # min lifted frames for a descriptor
    graph_shape_min_segments: int = 4         # min shared bone segments to compare
    graph_shape_stature_max_m: float = 0.15   # hard stature disagreement gate
    # F13 splittable clustering: before refinement, lift each multi-camera cluster
    # and, where the torso-residual chimera signature fires, veto the intruding
    # (worst) camera's within-cluster pair LLRs down to graph_chimera_veto_llr —
    # the existing refine move/split machinery then evicts the intruder. This is
    # the merge-only clustering's missing UNDO (ID-5 permanent chimeras).
    graph_split_enabled: bool = False
    graph_chimera_veto_llr: float = -6.0
    # Cue calibration: auto = bootstrap same/diff populations from this delivery,
    # file = load cue_calibration_path, defaults = conservative built-ins.
    calibration_mode: str = "auto"
    cue_calibration_path: str = ""
    anchor_pair_min_frames: int = 30
    anchor_pair_dist_m: float = 1.5
    anchor_pair_isolation_m: float = 3.0
    diff_pair_min_dist_m: float = 3.0
    # F8 cold-start robustness: when the strict gates find < 3 anchor pairs, retry
    # once with these relaxed same-player gates; if still starved, load a prior
    # calibration fitted on a clean delivery of the same match instead of silently
    # reverting to the default Gaussians. All off by default (byte-identical).
    anchor_relax_enabled: bool = False
    anchor_relax_dist_m: float = 2.0
    anchor_relax_isolation_m: float = 2.0
    calibration_fallback_path: str = ""
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
                     "diff_pair_min_dist_m", "graph_traj_attach_gate_m",
                     "approx_hip_height_m", "approx_shoulder_height_m",
                     "approx_head_height_m", "approx_var_floor_m",
                     "syn_chain_gate_m"):
            _require_positive(name, getattr(self, name))
        for name in ("graph_min_covis_frames", "graph_refine_passes", "graph_rescue_min_covis",
                     "graph_cannot_link_overlap_frames", "graph_min_app_samples",
                     "posture_min_samples", "anchor_pair_min_frames",
                     "syn_chain_max_gap_frames", "binding_min_single_frames"):
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
        _require_range("contested_iou", self.contested_iou, 0.0, 1.0)
        _require_range("contested_conf_scale", self.contested_conf_scale, 0.0, 1.0)
        if self.contested_sigma_scale < 1.0:
            raise ValueError("contested_sigma_scale must be >= 1.0")
        if type(self.contested_mute_appearance) is not bool:
            raise ValueError("contested_mute_appearance must be a boolean")
        for name in ("graph_motion_enabled", "purity_split_enabled", "posture_enabled",
                     "graph_corrob_merge"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        _require_positive("graph_llr_merge_single", self.graph_llr_merge_single)
        if self.graph_facing_gate_scale < 1.0:
            raise ValueError("graph_facing_gate_scale must be >= 1.0")
        if self.anchor_pair_dist_m >= self.diff_pair_min_dist_m:
            raise ValueError("anchor_pair_dist_m must be < diff_pair_min_dist_m")
        if self.anchor_relax_dist_m >= self.diff_pair_min_dist_m:
            raise ValueError("anchor_relax_dist_m must be < diff_pair_min_dist_m")
        if not isinstance(self.calibration_fallback_path, str):
            raise ValueError("calibration_fallback_path must be a string (may be empty)")
        for name in ("single_cam_cov_inflation", "airborne_cov_scale",
                     "airborne_ankle_bbox_frac", "graph_chimera_torso_residual_px",
                     "graph_chimera_frame_fraction"):
            _require_positive(name, getattr(self, name))
        if type(self.graph_lift_feedback) is not bool:
            raise ValueError("graph_lift_feedback must be a boolean")
        if type(self.graph_lift_stride) is not int or self.graph_lift_stride <= 0:
            raise ValueError("graph_lift_stride must be a positive integer")
        if type(self.graph_shape_enabled) is not bool:
            raise ValueError("graph_shape_enabled must be a boolean")
        if type(self.graph_split_enabled) is not bool:
            raise ValueError("graph_split_enabled must be a boolean")
        if self.graph_chimera_veto_llr >= 0:
            raise ValueError("graph_chimera_veto_llr must be negative")
        for name in ("graph_shape_min_frames", "graph_shape_min_segments"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        _require_positive("graph_shape_stature_max_m", self.graph_shape_stature_max_m)
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

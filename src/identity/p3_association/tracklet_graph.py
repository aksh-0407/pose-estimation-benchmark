"""Tracklet-graph cross-camera identity for P3 (``association_mode: tracklet_graph``).

Per-frame clustering decides identity where every cue is noisiest: one frame at a
time. This module decides it once per **tracklet pair** instead. P2 tracklets are
long (hundreds of frames) and clean, so aggregating cue evidence over the whole
co-visible span divides the per-frame noise by sqrt(n) and lets weak-but-honest
cues (kit colour, ground-anchored posture) contribute meaningfully.

Pipeline (all offline over one delivery, deterministic):

1. ``observe_frame`` — per detection: ground point + covariance + billboard posture
   sample; per cross-camera detection pair within a wide sample gate: ground
   residual/Mahalanobis, appearance distance, isolation. Tracklets are split into
   *chunks* at kinematically impossible ground jumps so a P2 identity switch can
   never weld two players together.
2. ``harvest_calibration`` — bootstrap same-player anchors (tight ground agreement
   + spatial isolation) and different-player pairs (consistently metres apart),
   fit per-cue LLR distributions (see :mod:`identity.p3_association.cue_calibration`).
3. ``solve`` — fuse per-pair aggregated cues into edge LLRs, then constrained
   agglomerative clustering (cannot-link: same-camera temporal overlap) with a
   local move-refinement pass. Every cluster becomes a persistent ``binding_id``.
4. ``emit_frame`` — rebuild per-frame correspondences from the bindings, so the
   correspondence stream P4 consumes is temporally stable by construction.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from itertools import combinations

import numpy as np

from dataclasses import replace as dc_replace

from identity.common.geometry import (
    camera_center_from_P,
    derive_facing_pairs,
    ground_covariance,
    ground_mahalanobis_sq,
    upper_body_ground_estimate,
)
from identity.common.pose_shape import (
    STATURE_QUANTITIES,
    PostureAccumulator,
    PostureAggregate,
    ground_anchored_skeleton,
    posture_distance_z,
    posture_from_skeleton,
)
from identity.p3_association.appearance import appearance_distance
from identity.p3_association.associator import (
    Correspondence,
    Detection3,
    _build_correspondence,
    _foot_pixel,
)
from identity.p3_association.config import P3AssociationConfig
from identity.p3_association.cue_calibration import (
    CueCalibration,
    fit_cue_calibration,
    fit_pair_distribution,
)

ChunkKey = tuple[str, str, int]  # (cam_id, local_track_id, chunk_index)


def _feet_unusable(det: Detection3, image_h: int, config: P3AssociationConfig) -> bool:
    bbox_bottom = float(det.bbox_xywh_px[1] + det.bbox_xywh_px[3])
    ankle_conf = float(np.max(det.keypoint_conf[[15, 16]]))
    return bbox_bottom >= image_h - 4 and ankle_conf < config.ankle_conf_min


def _approximated(det: Detection3, projection: np.ndarray, config: P3AssociationConfig) -> Detection3:
    estimate = upper_body_ground_estimate(
        det.keypoints_px, det.keypoint_conf, det.bbox_xywh_px, projection,
        hip_height_m=config.approx_hip_height_m,
        shoulder_height_m=config.approx_shoulder_height_m,
        head_height_m=config.approx_head_height_m,
    )
    if estimate is None:
        # Feet unusable AND no upper-body anchor this frame: no position at all
        # beats a garbage bbox-bottom projection — one garbage frame is enough to
        # shatter a chunk at a purity split and orphan everything after it.
        return dc_replace(det, ground_xy=np.full(2, np.nan), ground_approx=True)
    return dc_replace(det, ground_xy=estimate[0], ground_approx=True)


def apply_feet_approximation(
    detections_by_frame: dict[int, dict[str, list[Detection3]]],
    projections: dict[str, np.ndarray],
    image_h_by_cam: dict[str, int],
    config: P3AssociationConfig,
) -> dict[int, dict[str, list[Detection3]]]:
    """Re-anchor detections whose feet are unusable — STICKY per P2 tracklet.

    A bbox that reaches the frame's bottom edge with no confident ankle means the
    feet are cut off: the bbox-bottom ground projection is garbage (it projects
    the FRAME edge, not the player). An upper-body landmark of known typical
    height intersected with its height plane lands directly above the feet —
    most accurate exactly for the close-to-camera subjects that get cut off.

    The decision is made ONCE per tracklet (majority vote over its frames), never
    per frame: a tracklet whose ankle confidence hovers around the threshold must
    not flip between foot- and hip-anchored grounds (~1 m apart), which reads as
    teleporting and shatters the purity splitter. Untracked detections have no
    tracklet to vote over, so they stay per-frame.
    """

    if not config.approx_feet_enabled:
        return detections_by_frame
    votes: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for frames in detections_by_frame.values():
        for cam_id, dets in frames.items():
            image_h = image_h_by_cam.get(cam_id, config.image_h)
            for det in dets:
                if det.local_track_id is None:
                    continue
                tally = votes[(cam_id, det.local_track_id)]
                tally[0] += 1
                tally[1] += int(_feet_unusable(det, image_h, config))
    sticky = {
        key for key, (total, unusable) in votes.items()
        if total > 0 and unusable / total >= 0.6
    }
    out: dict[int, dict[str, list[Detection3]]] = {}
    for frame_index, frames in detections_by_frame.items():
        new_frames: dict[str, list[Detection3]] = {}
        for cam_id, dets in frames.items():
            projection = projections.get(cam_id)
            if projection is None:
                new_frames[cam_id] = dets
                continue
            image_h = image_h_by_cam.get(cam_id, config.image_h)
            new_dets = []
            for det in dets:
                if det.local_track_id is not None:
                    approximate = (cam_id, det.local_track_id) in sticky
                else:
                    approximate = _feet_unusable(det, image_h, config)
                new_dets.append(
                    _approximated(det, projection, config) if approximate else det
                )
            new_frames[cam_id] = new_dets
        out[frame_index] = new_frames
    return out


@dataclass
class _ChunkState:
    key: ChunkKey
    frames: list[int] = field(default_factory=list)
    ground_by_frame: dict[int, np.ndarray] = field(default_factory=dict)
    cov_by_frame: dict[int, np.ndarray] = field(default_factory=dict)
    posture: PostureAccumulator = field(default_factory=PostureAccumulator)
    # F11: sampled (17, 3) [x, y, conf] keypoints per frame for the cluster-level
    # shape lift; populated only when graph_shape_enabled (every graph_lift_stride
    # frames), so memory stays bounded.
    kp_samples: dict[int, np.ndarray] = field(default_factory=dict)
    posture_samples: int = 0
    upright_samples: int = 0
    approx_frames: int = 0

    @property
    def is_synthetic(self) -> bool:
        return "_syn_" in self.key[1]

    @property
    def calibration_grade(self) -> bool:
        """Only foot-anchored real tracklets may teach the calibration."""

        return (
            not self.is_synthetic
            and (not self.frames or self.approx_frames / len(self.frames) <= 0.3)
        )

    @property
    def upright_fraction(self) -> float:
        """Fraction of DETERMINABLE samples that were standing.

        With no determinable sample (feet never visible) the player is assumed
        standing: "unknown" must not trigger the crouch restrictions that
        "measured as crouching" does.
        """

        if not self.posture_samples:
            return 1.0
        return self.upright_samples / self.posture_samples

    @property
    def first_frame(self) -> int:
        return self.frames[0] if self.frames else 1 << 60

    @property
    def last_frame(self) -> int:
        return self.frames[-1] if self.frames else -1


@dataclass
class _PairSamples:
    frames: list[int] = field(default_factory=list)
    dist_m: list[float] = field(default_factory=list)
    maha: list[float] = field(default_factory=list)         # sqrt of Mahalanobis^2
    appearance: list[float] = field(default_factory=list)
    isolation_m: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class GraphEdge:
    key_a: ChunkKey
    key_b: ChunkKey
    llr_total: float
    llr_ground: float
    llr_appearance: float
    llr_posture: float
    llr_motion: float
    gated_frames: int
    median_dist_m: float
    posture_z: float | None

    def to_json(self) -> dict:
        return {
            "a": list(self.key_a), "b": list(self.key_b),
            "llr_total": round(self.llr_total, 3),
            "llr_ground": round(self.llr_ground, 3),
            "llr_appearance": round(self.llr_appearance, 3),
            "llr_posture": round(self.llr_posture, 3),
            "llr_motion": round(self.llr_motion, 3),
            "gated_frames": self.gated_frames,
            "median_dist_m": round(self.median_dist_m, 3),
            "posture_z": None if self.posture_z is None else round(self.posture_z, 3),
        }


@dataclass
class GraphSolution:
    binding_of_chunk: dict[ChunkKey, str]
    clusters: dict[str, list[ChunkKey]]
    edges: list[GraphEdge]
    calibration: CueCalibration
    diagnostics: dict

    def to_json(self) -> dict:
        return {
            "clusters": {
                binding: [list(key) for key in sorted(keys)]
                for binding, keys in sorted(self.clusters.items())
            },
            "edges": [edge.to_json() for edge in self.edges],
            "calibration": self.calibration.summary(),
            "diagnostics": self.diagnostics,
        }


def _pair_key(a: ChunkKey, b: ChunkKey) -> tuple[ChunkKey, ChunkKey]:
    return (a, b) if a <= b else (b, a)


def _median(values: list[float]) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.median(finite)) if finite else float("nan")


class TrackletGraphBuilder:
    """Accumulate evidence per frame, then solve identity once for the delivery."""

    def __init__(
        self,
        config: P3AssociationConfig,
        projections: dict[str, np.ndarray],
    ) -> None:
        self.config = config
        self.projections = projections
        self._chunks: dict[ChunkKey, _ChunkState] = {}
        self._current_chunk: dict[tuple[str, str], int] = {}
        self._last_seen: dict[tuple[str, str], tuple[int, np.ndarray]] = {}
        # (frame, cam, local_track_id) -> chunk, so emission is independent of
        # per-frame player ordering.
        self._det_chunk: dict[tuple[int, str, str], ChunkKey] = {}
        self._pairs: dict[tuple[ChunkKey, ChunkKey], _PairSamples] = defaultdict(_PairSamples)
        self._velocity_cache: dict[ChunkKey, dict[int, np.ndarray | None]] = {}
        self._azimuth_cache: dict[ChunkKey, np.ndarray | None] = {}
        # Per-binding pooled posture, computed once per solve for emit (F6b).
        self._emit_posture_cache: dict[str, PostureAggregate] | None = None
        # Synthetic tracklets: persistent untracked detections (e.g. umpires whose
        # cut-off dark figures P2 never tracked) chained by ground continuity so
        # the graph can bind them like any P2 tracklet.
        self._syn_chains: dict[str, list[dict]] = defaultdict(list)
        self._syn_counter: dict[str, int] = defaultdict(int)
        self._syn_of_det: dict[tuple[int, str, int], str] = {}
        self._support_lookup: dict[tuple[ChunkKey, ChunkKey], int] = {}
        self.diagnostics: dict[str, int] = defaultdict(int)
        # Facing (co-observing, low-parallax) camera pairs derived from calibration —
        # the same relationship the runner uses. Used to widen the hard-distance gate
        # on exactly those pairs where a tight gate splits a correct merge (ID-1).
        self._facing_pairs: set[frozenset[str]] = set()
        if self.config.graph_facing_gate_scale > 1.0 and len(projections) >= 2:
            try:
                self._facing_pairs = {
                    frozenset(pair) for pair in derive_facing_pairs(projections)
                }
            except Exception:
                self._facing_pairs = set()

    def _hard_dist_gate(self, cam_a: str, cam_b: str) -> float:
        gate = float(self.config.graph_hard_dist_gate_m)
        if frozenset((cam_a, cam_b)) in self._facing_pairs:
            gate *= float(self.config.graph_facing_gate_scale)
        return gate

    # ------------------------------------------------------------- pass A

    def _chunk_for(self, cam_id: str, local_track_id: str, frame_index: int,
                   ground: np.ndarray) -> ChunkKey:
        tracklet = (cam_id, local_track_id)
        index = self._current_chunk.get(tracklet, 0)
        previous = self._last_seen.get(tracklet)
        if previous is not None and self.config.purity_split_enabled:
            gap = max(1, frame_index - previous[0])
            allowed = (
                self.config.kinematic_v_max_mps * gap / self.config.frame_rate_fps
            ) * self.config.purity_jump_slack + self.config.purity_jump_floor_m
            if float(np.linalg.norm(ground - previous[1])) > allowed:
                index += 1
                self._current_chunk[tracklet] = index
                self.diagnostics["purity_splits"] += 1
        self._last_seen[tracklet] = (frame_index, ground.copy())
        return (cam_id, local_track_id, index)

    def observe_frame(self, frame_index: int, dets_per_cam: dict[str, list[Detection3]]) -> None:
        placed: list[tuple[ChunkKey, Detection3, np.ndarray, np.ndarray]] = []
        grounds_by_cam: dict[str, list[np.ndarray]] = defaultdict(list)
        for cam_id in sorted(dets_per_cam):
            projection = self.projections.get(cam_id)
            for index, det in enumerate(dets_per_cam[cam_id]):
                ground = det.ground_xy
                if ground is None or not np.isfinite(np.asarray(ground, dtype=float)).all():
                    continue
                ground = np.asarray(ground, dtype=float)
                grounds_by_cam[cam_id].append(ground)
                if projection is None:
                    continue
                tid = det.local_track_id
                if tid is None:
                    tid = self._synthetic_tid(cam_id, frame_index, ground, det)
                    if tid is None:
                        continue
                    self._syn_of_det[(frame_index, cam_id, det.player_index)] = tid
                key = self._chunk_for(cam_id, tid, frame_index, ground)
                chunk = self._chunks.setdefault(key, _ChunkState(key=key))
                chunk.frames.append(frame_index)
                chunk.ground_by_frame[frame_index] = ground
                chunk.approx_frames += int(det.ground_approx)
                foot = _foot_pixel(det, self.config)
                if det.ground_approx:
                    # Height-plane anchored position: honest isotropic uncertainty
                    # from the height-prior error, no foot-pixel Jacobian.
                    cov = float(self.config.approx_var_floor_m) ** 2 * np.eye(2)
                else:
                    cov = ground_covariance(
                        foot, projection,
                        sigma_px=self._sigma_px(det),
                        var_floor_m=self.config.ground_var_floor_m,
                    )
                chunk.cov_by_frame[frame_index] = cov
                # Wave-5b: overlapping same-camera boxes produce crops/skeletons that
                # mix the two players — poison for every identity descriptor. Skip
                # descriptor sampling for contested detections (the geometry above
                # keeps its own, separately down-weighted, covariance).
                mute_descriptors = (
                    getattr(det, "contested", False)
                    and self.config.contested_mute_appearance
                )
                if self.config.posture_enabled and not mute_descriptors:
                    points3d, valid = ground_anchored_skeleton(
                        det.keypoints_px, det.keypoint_conf, foot, projection,
                        min_conf=self.config.pose_min_conf,
                        ground_xy=ground if det.ground_approx else None,
                    )
                    sample = posture_from_skeleton(points3d, valid)
                    chunk.posture.add(
                        sample,
                        keep_upright_unknown=self.config.posture_keep_upright_unknown,
                    )
                    if sample is not None and sample.upright_known:
                        chunk.posture_samples += 1
                        chunk.upright_samples += int(sample.upright)
                if (
                    (self.config.graph_shape_enabled or self.config.graph_split_enabled)
                    and frame_index % self.config.graph_lift_stride == 0
                    and not mute_descriptors
                ):
                    chunk.kp_samples[frame_index] = np.column_stack(
                        (det.keypoints_px, det.keypoint_conf)
                    )
                self._det_chunk[(frame_index, cam_id, tid)] = key
                placed.append((key, det, ground, cov))

        for (key_a, det_a, ground_a, cov_a), (key_b, det_b, ground_b, cov_b) in combinations(placed, 2):
            if key_a[0] == key_b[0]:
                continue
            distance = float(np.linalg.norm(ground_a - ground_b))
            if distance > self.config.graph_sample_gate_m:
                continue
            samples = self._pairs[_pair_key(key_a, key_b)]
            samples.frames.append(frame_index)
            samples.dist_m.append(distance)
            m2 = ground_mahalanobis_sq(ground_a, cov_a, ground_b, cov_b)
            samples.maha.append(float(np.sqrt(m2)) if np.isfinite(m2) else float("nan"))
            # Wave-5b: appearance from a merged two-player crop is identity poison on
            # BOTH sides of the pair sample — skip it (geometry samples stay).
            if not (
                self.config.contested_mute_appearance
                and (getattr(det_a, "contested", False) or getattr(det_b, "contested", False))
            ):
                app = appearance_distance(det_a.appearance, det_b.appearance)
                if app is not None:
                    samples.appearance.append(float(app))
            # Isolation for anchor bootstrapping: nearest OTHER PERSON. Only the
            # pair's own two cameras can vouch for that — a detection in a third
            # camera may be this very player, but one camera never sees one person
            # twice, so same-camera neighbours are guaranteed different people.
            midpoint = 0.5 * (ground_a + ground_b)
            others = [
                float(np.linalg.norm(g - midpoint))
                for cam_id, own in ((key_a[0], ground_a), (key_b[0], ground_b))
                for g in grounds_by_cam[cam_id]
                if g is not own
            ]
            samples.isolation_m.append(min(others) if others else float("inf"))

    def _synthetic_tid(
        self, cam_id: str, frame_index: int, ground: np.ndarray, det: Detection3
    ) -> str | None:
        """Chain a persistent untracked detection into a synthetic tracklet."""

        if not self.config.synthetic_tracklets_enabled:
            return None
        if det.confidence < self.config.syn_min_confidence:
            return None
        chains = self._syn_chains[cam_id]
        best, best_dist = None, float("inf")
        for chain in chains:
            gap = frame_index - chain["last_frame"]
            if gap <= 0 or gap > self.config.syn_chain_max_gap_frames:
                continue
            # Re-acquisition radius grows kinematically but is capped: a chain
            # surviving a long occlusion must resume near where it stopped, never
            # leap to a different person.
            allowed = min(
                max(
                    self.config.syn_chain_gate_m,
                    self.config.kinematic_v_max_mps * gap / self.config.frame_rate_fps,
                ),
                self.config.graph_hard_dist_gate_m,
            )
            distance = float(np.linalg.norm(ground - chain["last_ground"]))
            if distance <= allowed and distance < best_dist:
                best, best_dist = chain, distance
        if best is None:
            self._syn_counter[cam_id] += 1
            best = {"tid": f"{cam_id}_syn_{self._syn_counter[cam_id]:03d}"}
            chains.append(best)
            self.diagnostics["synthetic_tracklets"] += 1
        best["last_frame"] = frame_index
        best["last_ground"] = ground.copy()
        return best["tid"]

    def _sigma_px(self, det: Detection3) -> float:
        bbox_h = float(det.bbox_xywh_px[3]) if len(det.bbox_xywh_px) == 4 else 0.0
        sigma = (
            self.config.ground_sigma_px_base
            + self.config.ground_sigma_px_bbox_frac * max(bbox_h, 0.0)
        )
        if getattr(det, "contested", False):
            # Wave-5b: a merged-box foot pixel is unreliable — widen this view's
            # ground covariance so pair Mahalanobis evidence leans on clean cameras.
            sigma *= float(self.config.contested_sigma_scale)
        return sigma

    # -------------------------------------------------------- calibration

    def _collect_calibration_pairs(
        self,
        postures: dict,
        anchor_dist_m: float,
        anchor_isolation_m: float,
    ) -> tuple[dict, dict, dict, dict, dict, list, list]:
        """One anchor-selection sweep at the given same-player gates.

        Diff pairs always use ``diff_pair_min_dist_m``; only the same-player
        gates vary between the strict pass and the F8 relaxation pass.
        """

        same_samples: dict[str, list[float]] = defaultdict(list)
        diff_samples: dict[str, list[float]] = defaultdict(list)
        pair_app_same: dict[str, list[float]] = defaultdict(list)
        pair_app_diff: dict[str, list[float]] = defaultdict(list)
        posture_same_deltas: dict[str, list[float]] = defaultdict(list)
        same_pairs: list[tuple[ChunkKey, ChunkKey]] = []
        diff_pairs: list[tuple[ChunkKey, ChunkKey]] = []

        for (key_a, key_b), samples in sorted(self._pairs.items()):
            if len(samples.frames) < self.config.anchor_pair_min_frames:
                continue
            # Synthetic chains and approximation-anchored tracklets carry
            # model-based position error; they must not teach the calibration.
            if not (self._chunks[key_a].calibration_grade
                    and self._chunks[key_b].calibration_grade):
                continue
            med_dist = _median(samples.dist_m)
            med_iso = _median(samples.isolation_m)
            if not np.isfinite(med_dist):
                continue
            camera_pair = CueCalibration.camera_pair_key(key_a[0], key_b[0])
            if med_dist <= anchor_dist_m and med_iso >= anchor_isolation_m:
                same_pairs.append((key_a, key_b))
                same_samples["ground_dist_m"].extend(samples.dist_m)
                same_samples["ground_maha"].extend(samples.maha)
                same_samples["appearance"].extend(samples.appearance)
                pair_app_same[camera_pair].extend(samples.appearance)
                agg_a, agg_b = postures.get(key_a), postures.get(key_b)
                if agg_a is not None and agg_b is not None:
                    for name in set(agg_a.median) & set(agg_b.median):
                        posture_same_deltas[name].append(
                            abs(agg_a.median[name] - agg_b.median[name])
                        )
            elif med_dist >= self.config.diff_pair_min_dist_m:
                diff_pairs.append((key_a, key_b))
                diff_samples["ground_dist_m"].extend(samples.dist_m)
                diff_samples["ground_maha"].extend(samples.maha)
                diff_samples["appearance"].extend(samples.appearance)
                pair_app_diff[camera_pair].extend(samples.appearance)

        return (same_samples, diff_samples, pair_app_same, pair_app_diff,
                posture_same_deltas, same_pairs, diff_pairs)

    def harvest_calibration(self) -> CueCalibration:
        """Bootstrap same/different populations from geometry and fit cue LLRs."""

        postures = self._chunk_postures()
        (same_samples, diff_samples, pair_app_same, pair_app_diff,
         posture_same_deltas, same_pairs, diff_pairs) = self._collect_calibration_pairs(
            postures, self.config.anchor_pair_dist_m, self.config.anchor_pair_isolation_m
        )

        if len(same_pairs) < 3 and self.config.anchor_relax_enabled:
            # F8: a crowded delivery may hold no player isolated by 3 m for long
            # enough; retry once with looser same-player gates before losing every
            # delivery-fitted cue. Adopted only when it actually finds more anchors.
            relaxed = self._collect_calibration_pairs(
                postures, self.config.anchor_relax_dist_m, self.config.anchor_relax_isolation_m
            )
            if len(relaxed[5]) > len(same_pairs):
                (same_samples, diff_samples, pair_app_same, pair_app_diff,
                 posture_same_deltas, same_pairs, diff_pairs) = relaxed
                self.diagnostics["calibration_anchor_relaxed"] = 1

        if len(same_pairs) < 3:
            # Per-frame samples within one pair are correlated; fewer than three
            # independent anchor pairs cannot support a trustworthy same-player
            # distribution. Prefer a cross-delivery prior calibration fitted on a
            # clean clip of the same match (F8) when configured; else keep the
            # conservative physical defaults.
            if self.config.calibration_fallback_path:
                calibration = CueCalibration.load(self.config.calibration_fallback_path)
                self.diagnostics["calibration_used_prior"] = 1
            else:
                calibration = CueCalibration()
                self.diagnostics["calibration_fell_back_to_defaults"] = 1
            calibration.anchor_pair_count = len(same_pairs)
            calibration.diff_pair_count = len(diff_pairs)
            return calibration

        calibration = fit_cue_calibration(
            same_samples={k: v for k, v in same_samples.items()},
            diff_samples={k: v for k, v in diff_samples.items()},
            posture_same_deltas={k: v for k, v in posture_same_deltas.items()},
            anchor_pair_count=len(same_pairs),
            diff_pair_count=len(diff_pairs),
        )
        for camera_pair in sorted(set(pair_app_same) & set(pair_app_diff)):
            fitted = fit_pair_distribution(
                pair_app_same[camera_pair], pair_app_diff[camera_pair]
            )
            if fitted is not None:
                calibration.appearance_by_pair[camera_pair] = fitted

        # Second pass: posture z-scores need the fitted systematic sigmas.
        z_same = [
            z for pair in same_pairs
            if (z := self._posture_z(postures, pair, calibration)) is not None
        ]
        z_diff = [
            z for pair in diff_pairs
            if (z := self._posture_z(postures, pair, calibration)) is not None
        ]
        if len(z_same) >= 8 and len(z_diff) >= 8:
            refit = fit_cue_calibration(
                same_samples={"posture_z": z_same},
                diff_samples={"posture_z": z_diff},
            )
            calibration.distributions["posture_z"] = refit.distributions["posture_z"]
        return calibration

    def _posture_z(
        self,
        postures: dict[ChunkKey, PostureAggregate | None],
        pair: tuple[ChunkKey, ChunkKey],
        calibration: CueCalibration,
    ) -> float | None:
        # A bent/crouched body (e.g. the keeper) is not billboard-planar, so its
        # SHAPE quantities (torso length, widths) are only comparable between
        # near-parallel or near-antiparallel views; between oblique views of a
        # non-upright player only the foreshortening-free verticals may vote.
        quantities = None
        chunk_a, chunk_b = self._chunks[pair[0]], self._chunks[pair[1]]
        if min(chunk_a.upright_fraction, chunk_b.upright_fraction) < 0.5:
            az_a, az_b = self._view_azimuth(pair[0]), self._view_azimuth(pair[1])
            if az_a is not None and az_b is not None and abs(float(az_a @ az_b)) < 0.7:
                quantities = STATURE_QUANTITIES
        result = posture_distance_z(
            postures.get(pair[0]), postures.get(pair[1]),
            sigma_sys=calibration.posture_sigma_sys,
            quantities=quantities,
        )
        return None if result is None else result[0]

    def _view_azimuth(self, key: ChunkKey) -> np.ndarray | None:
        """Horizontal unit vector from the chunk's camera to its mean position."""

        cached = self._azimuth_cache.get(key)
        if cached is not None or key in self._azimuth_cache:
            return cached
        chunk = self._chunks[key]
        projection = self.projections.get(key[0])
        azimuth = None
        if projection is not None and chunk.ground_by_frame:
            center = camera_center_from_P(np.asarray(projection, dtype=float))
            mean_ground = np.mean(np.asarray(list(chunk.ground_by_frame.values())), axis=0)
            direction = np.array([mean_ground[0] - center[0], mean_ground[1] - center[1]])
            norm = float(np.linalg.norm(direction))
            if np.isfinite(norm) and norm > 1e-6:
                azimuth = direction / norm
        self._azimuth_cache[key] = azimuth
        return azimuth

    def _chunk_postures(self) -> dict[ChunkKey, PostureAggregate | None]:
        return {
            key: chunk.posture.aggregate(min_samples=self.config.posture_min_samples)
            for key, chunk in self._chunks.items()
        }

    def binding_postures(self, solution: "GraphSolution") -> dict[str, PostureAggregate]:
        """Pooled billboard posture per binding, from every member chunk's samples (F6b).

        Pooling the raw samples (rather than merging per-chunk aggregates) keeps the
        robust median/SE semantics of :class:`PostureAccumulator` for the combined
        population. This is the facing-pair-capable body-shape key P4 uses for its
        teleport veto and re-entry gate.
        """

        merged: dict[str, PostureAggregate] = {}
        for binding, keys in solution.clusters.items():
            pooled = PostureAccumulator()
            for key in keys:
                chunk = self._chunks.get(key)
                if chunk is None:
                    continue
                for name, values in chunk.posture.samples.items():
                    pooled.samples.setdefault(name, []).extend(values)
            aggregate = pooled.aggregate(min_samples=self.config.posture_min_samples)
            if aggregate is not None and aggregate.is_defined():
                merged[binding] = aggregate
        return merged

    # -------------------------------------------------------------- solve

    def _motion_llr(self, key_a: ChunkKey, key_b: ChunkKey, frames: list[int]) -> float:
        """Motion agreement over the shared window, on heavily-smoothed velocities.

        Raw foot projections carry ~0.3 m of noise, so short-lag finite differences
        are useless (~5 m/s of speed noise). Velocities here come from median-
        smoothed positions with a 10-frame central difference. Two verdicts:

        * both clearly moving -> direction cosine (parallel = weak "same",
          opposite = strong "different");
        * one sprinting while the other stands -> strong "different" (this is the
          bowler-vs-non-striker case: co-located during the run-up, but only one
          of them is running).
        """

        if not self.config.graph_motion_enabled:
            return 0.0
        velocities_a = self._smoothed_velocities(key_a)
        velocities_b = self._smoothed_velocities(key_b)
        cosines: list[float] = []
        asymmetric = 0
        comparable = 0
        for frame in frames:
            va, vb = velocities_a.get(frame), velocities_b.get(frame)
            if va is None or vb is None:
                continue
            speed_a, speed_b = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
            comparable += 1
            low, high = min(speed_a, speed_b), max(speed_a, speed_b)
            if (high > self.config.graph_motion_speed_full_mps
                    and low < self.config.graph_motion_speed_still_mps):
                asymmetric += 1
            elif (speed_a > 0.6 * self.config.graph_motion_speed_full_mps
                    and speed_b > 0.6 * self.config.graph_motion_speed_full_mps):
                cosines.append(float(va @ vb) / (speed_a * speed_b))
        if comparable < 10:
            return 0.0
        llr = 0.0
        if len(cosines) >= 10:
            llr += self.config.graph_motion_gain * (
                float(np.mean(cosines)) - self.config.graph_motion_cos_offset
            )
        if asymmetric >= 10 and asymmetric / comparable >= 0.2:
            llr += self.config.graph_motion_still_llr
        return float(np.clip(
            llr, self.config.graph_motion_llr_min, self.config.graph_motion_llr_max
        ))

    def _smoothed_velocities(self, key: ChunkKey) -> dict[int, np.ndarray | None]:
        """Per-frame ground velocity from median-smoothed positions (cached)."""

        cached = self._velocity_cache.get(key)
        if cached is not None:
            return cached
        chunk = self._chunks[key]
        frames = sorted(chunk.ground_by_frame)
        half, lag = 5, 10
        smoothed: dict[int, np.ndarray] = {}
        for frame in frames:
            window = [
                chunk.ground_by_frame[f]
                for f in range(frame - half, frame + half + 1)
                if f in chunk.ground_by_frame
            ]
            if len(window) >= 3:
                smoothed[frame] = np.median(np.asarray(window), axis=0)
        fps = self.config.frame_rate_fps
        velocities: dict[int, np.ndarray | None] = {}
        for frame in frames:
            before, after = smoothed.get(frame - lag), smoothed.get(frame + lag)
            velocities[frame] = (
                (after - before) * (fps / (2.0 * lag))
                if before is not None and after is not None else None
            )
        self._velocity_cache[key] = velocities
        return velocities

    def _build_edges(self, calibration: CueCalibration) -> list[GraphEdge]:
        postures = self._chunk_postures()
        edges: list[GraphEdge] = []
        positive_cap = self.config.graph_llr_positive_cap
        # Appearance is rig-dependent (inter-camera colour processing differs more
        # than kits differ between people on some pairs of this rig), so it is
        # scored PER CAMERA PAIR from that pair's own anchors and abstains for
        # pairs without a separable fit of their own.
        if not calibration.appearance_by_pair:
            self.diagnostics["appearance_cue_abstained"] = 1
        for (key_a, key_b), samples in sorted(self._pairs.items()):
            gated = len(samples.frames)
            if gated < self.config.graph_min_covis_frames:
                continue
            med_maha = _median(samples.maha)
            med_dist = _median(samples.dist_m)
            if not np.isfinite(med_dist) or med_dist > self._hard_dist_gate(key_a[0], key_b[0]):
                # Consistently farther apart than any observed same-player
                # cross-camera residual: not mergeable, and no negative edge is
                # needed (absence already blocks merging).
                continue
            # Every cue is clipped asymmetrically: agreement is weak evidence of
            # identity (players can share position, kit, and build) but strong
            # disagreement is near-conclusive evidence of difference. Ground gets a
            # milder NEGATIVE clip than the other cues: inside the hard distance
            # gate a large residual can also mean a large per-camera-pair
            # calibration bias (measured up to ~2 m on some facing pairs vs ~1 m
            # globally), so ground alone must never outvote posture/motion.
            llr_ground = calibration.llr(
                "ground_maha", med_maha,
                clip=self.config.graph_llr_ground_neg_clip, clip_pos=positive_cap,
            )
            med_app = (
                _median(samples.appearance)
                if len(samples.appearance) >= self.config.graph_min_app_samples
                else None
            )
            llr_app = calibration.appearance_llr(
                key_a[0], key_b[0], med_app, clip_pos=positive_cap
            )
            z = self._posture_z(postures, (key_a, key_b), calibration)
            llr_posture = calibration.llr("posture_z", z, clip_pos=positive_cap)
            llr_motion = self._motion_llr(key_a, key_b, samples.frames)
            support = min(1.0, gated / max(self.config.graph_covis_full_frames, 1))
            total = support * (llr_ground + llr_app + llr_posture + llr_motion)
            edges.append(GraphEdge(
                key_a=key_a, key_b=key_b, llr_total=total,
                llr_ground=llr_ground, llr_appearance=llr_app,
                llr_posture=llr_posture, llr_motion=llr_motion,
                gated_frames=gated, median_dist_m=med_dist, posture_z=z,
            ))
        # More co-visible evidence wins ties created by the positive cap.
        edges.sort(key=lambda e: (-e.llr_total, -e.gated_frames, e.key_a, e.key_b))
        return edges

    def _overlap_frames(self, key_a: ChunkKey, key_b: ChunkKey) -> int:
        return len(set(self._chunks[key_a].frames) & set(self._chunks[key_b].frames))

    def _cluster_compatible(self, members_a: list[ChunkKey], members_b: list[ChunkKey],
                            llr_lookup: dict[tuple[ChunkKey, ChunkKey], float]) -> bool:
        """Merging is a vote, not a veto by the noisiest pair.

        Hard blocks: same-camera temporal overlap (two simultaneous tracklets in
        one camera are two people) and any pair below ``graph_llr_veto`` (a
        confident contradiction, e.g. posture separating the crouching keeper from
        the striker). Otherwise the SUM of cross-pair evidence decides, so one
        noisy 22-frame pair cannot cancel a 600-frame agreement.
        """

        total = 0.0
        for a in members_a:
            for b in members_b:
                if a[0] == b[0]:
                    if self._overlap_frames(a, b) > self.config.graph_cannot_link_overlap_frames:
                        return False
                else:
                    llr = llr_lookup.get(_pair_key(a, b))
                    if llr is not None:
                        if llr < self.config.graph_llr_veto:
                            return False
                        total += llr
        return total >= 0.0

    def solve(self, calibration: CueCalibration | None = None) -> GraphSolution:
        calibration = calibration or self.harvest_calibration()
        edges = self._build_edges(calibration)
        llr_lookup = {_pair_key(e.key_a, e.key_b): e.llr_total for e in edges}
        self._support_lookup = {
            _pair_key(e.key_a, e.key_b): e.gated_frames for e in edges
        }

        parent: dict[ChunkKey, ChunkKey] = {key: key for key in self._chunks}
        members: dict[ChunkKey, list[ChunkKey]] = {key: [key] for key in self._chunks}

        def find(key: ChunkKey) -> ChunkKey:
            root = key
            while parent[root] != root:
                root = parent[root]
            while parent[key] != root:
                parent[key], key = root, parent[key]
            return root

        merges = 0
        for edge in edges:
            if edge.llr_total < self.config.graph_llr_merge_threshold:
                break
            root_a, root_b = find(edge.key_a), find(edge.key_b)
            if root_a == root_b:
                continue
            if not self._cluster_compatible(members[root_a], members[root_b], llr_lookup):
                self.diagnostics["merges_blocked"] += 1
                continue
            parent[root_b] = root_a
            members[root_a] = sorted(members[root_a] + members[root_b])
            del members[root_b]
            merges += 1
        self.diagnostics["merges_accepted"] = merges

        if self.config.graph_corrob_merge:
            self.diagnostics["corroboration_merges"] = self._corroboration_merge_pass(
                edges, parent, members, find, llr_lookup
            )

        if self.config.graph_shape_enabled:
            self.diagnostics["shape_merges"] = self._shape_corroboration_pass(
                parent, members, find, llr_lookup
            )

        if self.config.graph_union_lift_merge:
            self.diagnostics["union_lift_merges"] = self._union_lift_merge_pass(
                parent, members, find, llr_lookup
            )

        cluster_members: dict[int, list[ChunkKey]] = {
            index: sorted(keys)
            for index, keys in enumerate(sorted(members.values(), key=lambda keys: keys[0]))
        }
        cluster_of: dict[ChunkKey, int] = {
            key: cid for cid, keys in cluster_members.items() for key in keys
        }
        if self.config.graph_split_enabled:
            self.diagnostics["chimera_evictions"] = self._chimera_veto_pass(
                cluster_members, cluster_of, llr_lookup
            )
        moves = self._refine(cluster_members, cluster_of, llr_lookup)
        self.diagnostics["refine_moves"] = moves
        self._rescue_singletons(cluster_members, cluster_of, edges, llr_lookup)
        self._attach_fragments_by_trajectory(cluster_members, cluster_of)

        # Only clusters with real identity evidence EARN a binding id: seen by
        # two cameras, or one long stable single-camera track (an umpire). Short
        # fragments emit as unbound low-confidence detections — P4 absorbs or
        # ignores them instead of minting a fresh global id per fragment (the
        # mechanism that produced 54 ids for ~9 people on shattered-P2 clips).
        clusters_raw = sorted(
            cluster_members.values(),
            key=lambda keys: (min(self._chunks[k].first_frame for k in keys), keys[0]),
        )
        binding_of_chunk: dict[ChunkKey, str] = {}
        clusters: dict[str, list[ChunkKey]] = {}
        index = 0
        for keys in clusters_raw:
            cameras = {key[0] for key in keys}
            longest = max(len(self._chunks[key].frames) for key in keys)
            if len(cameras) < 2 and longest < self.config.binding_min_single_frames:
                self.diagnostics["clusters_demoted"] += 1
                continue
            index += 1
            binding = f"B{index:03d}"
            clusters[binding] = sorted(keys)
            for key in keys:
                binding_of_chunk[key] = binding
        return GraphSolution(
            binding_of_chunk=binding_of_chunk,
            clusters=clusters,
            edges=edges,
            calibration=calibration,
            diagnostics=dict(sorted(self.diagnostics.items())),
        )

    def _union_lift_merge_pass(self, parent, members, find, llr_lookup) -> int:
        """W9: same ground location + one coherent 3D skeleton across all views
        => one person (the ghost-under-player / split-identity fix).

        A facing-pair split leaves TWO clusters for one physical player (e.g.
        {C1,C4} and {C2,C6}) whose cross edges were killed by the facing-pair
        ground bias — no pairwise pass can ever rejoin them. This pass finds
        cluster pairs that are CO-LOCATED over many frames, checks the billboard
        stature agrees, then runs the decisive geometric test the user
        prescribed: triangulate the UNION of both clusters' member views. A
        genuine single person yields a valid low-residual skeleton in EVERY
        view (from any angle); two different people produce the one-sided
        chimera signature, which rejects the merge. Occluded views cannot poison
        the test: ``lift_frame`` weights joints by confidence and RANSAC drops
        outlier views.
        """

        from identity.common.pose_shape import (
            STATURE_QUANTITIES,
            posture_distance_z,
        )
        from identity.p3_association.cluster_lift import cluster_purity

        cfg = self.config

        def cluster_ground(keys: list[ChunkKey]) -> dict[int, np.ndarray]:
            points: dict[int, list[np.ndarray]] = {}
            for key in keys:
                chunk = self._chunks.get(key)
                if chunk is None:
                    continue
                for frame_index, ground in chunk.ground_by_frame.items():
                    points.setdefault(frame_index, []).append(np.asarray(ground, float))
            return {fi: np.mean(vals, axis=0) for fi, vals in points.items()}

        def cluster_posture(keys: list[ChunkKey]):
            pooled = PostureAccumulator()
            for key in keys:
                chunk = self._chunks.get(key)
                if chunk is None:
                    continue
                for name, values in chunk.posture.samples.items():
                    pooled.samples.setdefault(name, []).extend(values)
            return pooled.aggregate(min_samples=cfg.posture_min_samples)

        grounds = {root: cluster_ground(keys) for root, keys in members.items()}
        candidates = []
        for root_a, root_b in combinations(sorted(members), 2):
            ga, gb = grounds[root_a], grounds[root_b]
            co_frames = sorted(set(ga) & set(gb))
            if len(co_frames) < cfg.graph_union_min_co_frames:
                continue
            dists = np.asarray(
                [np.linalg.norm(ga[f] - gb[f]) for f in co_frames], dtype=float
            )
            med = float(np.median(dists))
            if med > cfg.graph_union_colocate_m:
                continue
            candidates.append((med, root_a, root_b))

        merges = 0
        reasons = {"overlap": 0, "veto": 0, "posture": 0, "lift_frames": 0,
                   "chimera": 0, "residual": 0}
        for med, root_a, root_b in sorted(candidates):
            root_a, root_b = find(root_a), find(root_b)
            if root_a == root_b or root_a not in members or root_b not in members:
                continue
            # Hard blocks ONLY (same-camera overlap; confident cue veto). The usual
            # sum>=0 evidence vote is deliberately NOT required here: the facing-pair
            # ground bias makes exactly these cross edges sum negative, and the
            # union-lift reprojection test below is the decisive replacement evidence.
            blocked = False
            for key_a in members[root_a]:
                for key_b in members[root_b]:
                    if key_a[0] == key_b[0]:
                        overlap = self._overlap_frames(key_a, key_b)
                        if overlap > self.config.graph_cannot_link_overlap_frames:
                            blocked = True
                            reasons["overlap"] += 1
                            reasons.setdefault("overlap_detail", []).append({
                                "cam": key_a[0], "frames": int(overlap),
                                "len_a": len(self._chunks[key_a].frames),
                                "len_b": len(self._chunks[key_b].frames),
                            })
                            break
                    else:
                        llr = llr_lookup.get(_pair_key(key_a, key_b))
                        if llr is not None and llr < self.config.graph_llr_veto:
                            blocked = True
                            reasons["veto"] += 1
                            break
                if blocked:
                    break
            if blocked:
                continue
            posture_a = cluster_posture(members[root_a])
            posture_b = cluster_posture(members[root_b])
            if posture_a is not None and posture_b is not None:
                result = posture_distance_z(
                    posture_a, posture_b, quantities=STATURE_QUANTITIES
                )
                if result is not None and result[0] > cfg.graph_union_posture_max_z:
                    reasons["posture"] += 1
                    continue
            lifts = self._cluster_lifts(sorted(members[root_a] + members[root_b]))
            if len(lifts) < cfg.graph_union_min_lift_frames:
                reasons["lift_frames"] += 1
                continue
            purity = cluster_purity(
                lifts,
                chimera_torso_residual_px=cfg.graph_chimera_torso_residual_px,
                chimera_frame_fraction=cfg.graph_chimera_frame_fraction,
            )
            if purity.chimera_suspect:
                reasons["chimera"] += 1
                continue
            if (
                purity.torso_residual_p50 is None
                or purity.torso_residual_p50 > cfg.graph_union_torso_p50_px
            ):
                reasons["residual"] += 1
                continue
            parent[root_b] = root_a
            members[root_a] = sorted(members[root_a] + members[root_b])
            del members[root_b]
            merges += 1
        self.diagnostics["union_lift_rejects"] = dict(reasons)
        return merges

    def _corroboration_merge_pass(
        self,
        edges: list[GraphEdge],
        parent: dict[ChunkKey, ChunkKey],
        members: dict[ChunkKey, list[ChunkKey]],
        find,
        llr_lookup: dict[tuple[ChunkKey, ChunkKey], float],
    ) -> int:
        """Merge strong-but-single-cue facing-pair edges when unambiguous (ID-1).

        A pair whose only discriminative cue is ground (appearance/motion/posture
        structurally abstain on the low-parallax facing pairs) caps at
        ``graph_llr_positive_cap`` < ``graph_llr_merge_threshold`` and so never
        merges, even when it is a genuine same-player pair with no contradicting
        evidence. This second pass admits such an edge only under strict conditions:
        full co-visible support, NO observable cue disagreeing (every present cue
        >= 0), the edge is the mutual unambiguous best for BOTH endpoints' clusters,
        and the merge passes the cannot-link/veto compatibility check. Conservative
        by construction, so it cannot manufacture the chimeras a blanket threshold
        drop would.
        """

        threshold = self.config.graph_llr_merge_threshold
        single = self.config.graph_llr_merge_single
        full = max(self.config.graph_covis_full_frames, 1)
        candidates = [
            edge for edge in edges
            if single <= edge.llr_total < threshold
            and edge.gated_frames >= full
            # nothing observable disagrees: abstaining cues are 0, only a negative
            # cue (a real contradiction) disqualifies corroboration.
            and edge.llr_ground >= 0.0 and edge.llr_appearance >= 0.0
            and edge.llr_posture >= 0.0 and edge.llr_motion >= 0.0
        ]
        candidates.sort(key=lambda e: (-e.llr_total, -e.gated_frames, e.key_a, e.key_b))

        def best_partner(root: ChunkKey) -> tuple[ChunkKey | None, float]:
            """Best candidate-edge partner root for ``root`` (unambiguity check)."""
            best_root, best_llr = None, -np.inf
            for edge in candidates:
                ra, rb = find(edge.key_a), find(edge.key_b)
                if ra == rb:
                    continue
                other = rb if ra == root else (ra if rb == root else None)
                if other is None:
                    continue
                if edge.llr_total > best_llr:
                    best_root, best_llr = other, edge.llr_total
            return best_root, best_llr

        merged = 0
        for edge in candidates:
            root_a, root_b = find(edge.key_a), find(edge.key_b)
            if root_a == root_b:
                continue
            # Mutual unambiguous best: each cluster's strongest corroboration edge
            # must point at the other, so we never grab the wrong nearby cluster.
            pa, _ = best_partner(root_a)
            pb, _ = best_partner(root_b)
            if pa != root_b or pb != root_a:
                self.diagnostics["corroboration_ambiguous"] += 1
                continue
            if not self._cluster_compatible(members[root_a], members[root_b], llr_lookup):
                continue
            parent[root_b] = root_a
            members[root_a] = sorted(members[root_a] + members[root_b])
            del members[root_b]
            merged += 1
        return merged

    def _cluster_shape(
        self, keys: list[ChunkKey]
    ) -> tuple["PoseProportions | None", float | None, int]:
        """Lifted bone-ratio descriptor + stature for one cluster (F11).

        Builds per-frame multi-view keypoint sets from the member chunks'
        sampled keypoints and pools the per-frame descriptors. Returns
        ``(descriptor, stature_m, frames_lifted)``; descriptor is None when the
        cluster never has two sampled views of the same frame.
        """

        from identity.p3_association.cluster_lift import cluster_purity, lift_frame

        lifts = self._cluster_lifts(keys)
        if len(lifts) < self.config.graph_shape_min_frames:
            return None, None, len(lifts)
        purity = cluster_purity(lifts)
        return purity.descriptor, purity.stature_m, len(lifts)

    def _cluster_lifts(self, keys: list[ChunkKey]) -> list:
        """Per-frame multi-view lifts for one cluster from the sampled keypoints."""

        from identity.p3_association.cluster_lift import lift_frame

        per_frame: dict[int, dict[str, np.ndarray]] = {}
        for key in keys:
            chunk = self._chunks.get(key)
            if chunk is None:
                continue
            cam_id = key[0]
            for frame_index, kp in chunk.kp_samples.items():
                per_frame.setdefault(frame_index, {}).setdefault(cam_id, kp)
        lifts = []
        for frame_index in sorted(per_frame):
            members = per_frame[frame_index]
            if len(members) < 2:
                continue
            lift = lift_frame(
                members, self.projections,
                reprojection_threshold_px=self.config.triangulation_reproj_threshold_px,
                min_views=self.config.triangulation_min_views,
            )
            if lift is not None:
                lifts.append(lift)
        return lifts

    def _shape_corroboration_pass(self, parent, members, find, llr_lookup) -> int:
        """F11: merge compatible cluster pairs on cluster-level body-shape agreement.

        The pairwise round caps every cue's positive evidence, so on the facing
        pairs (colour dead, motion static, posture sometimes crouched) a genuine
        same-player pair can never reach the merge threshold. This round compares
        whole clusters instead: a lifted bone-ratio descriptor + metric stature,
        self-calibrated on this delivery (same = temporal halves of one cluster;
        diff = simultaneously co-visible distinct clusters). Merges only when
        geometry does not disagree, shape actively agrees, stature is consistent,
        and the pair is the mutual best — and abstains entirely when the shape
        distributions cannot be fitted.
        """

        from identity.common.pose_shape import descriptor_distance

        shapes: dict[ChunkKey, tuple] = {}
        for root, keys in members.items():
            if len({key[0] for key in keys}) < 1:
                continue
            descriptor, stature, frames = self._cluster_shape(keys)
            if descriptor is not None and descriptor.is_defined():
                shapes[root] = (descriptor, stature)
        if len(shapes) < 2:
            return 0

        def shape_distance(root_a: ChunkKey, root_b: ChunkKey) -> float | None:
            return descriptor_distance(
                shapes[root_a][0], shapes[root_b][0],
                min_shared=self.config.graph_shape_min_segments,
            )

        # --- self-calibration ------------------------------------------------
        same_samples: list[float] = []
        for root in shapes:
            keys = members[root]
            frames_sorted = sorted(
                frame for key in keys for frame in self._chunks[key].kp_samples
            )
            if len(frames_sorted) < 2 * self.config.graph_shape_min_frames:
                continue
            midpoint = frames_sorted[len(frames_sorted) // 2]
            # Build two half-window descriptors by masking kp_samples temporally.
            halves = []
            for lo, hi in ((min(frames_sorted), midpoint), (midpoint, max(frames_sorted) + 1)):
                saved = {key: self._chunks[key].kp_samples for key in keys}
                try:
                    for key in keys:
                        self._chunks[key].kp_samples = {
                            f: kp for f, kp in saved[key].items() if lo <= f < hi
                        }
                    half_desc, _stature, _n = self._cluster_shape(keys)
                finally:
                    for key in keys:
                        self._chunks[key].kp_samples = saved[key]
                halves.append(half_desc)
            if halves[0] is not None and halves[1] is not None:
                value = descriptor_distance(
                    halves[0], halves[1], min_shared=self.config.graph_shape_min_segments
                )
                if value is not None:
                    same_samples.append(float(value))

        def cluster_median_ground(keys: list[ChunkKey]) -> np.ndarray:
            points = [
                ground for key in keys
                for ground in self._chunks[key].ground_by_frame.values()
            ]
            return np.median(np.asarray(points, dtype=float), axis=0)

        roots = sorted(shapes, key=lambda r: r)
        medians = {root: cluster_median_ground(members[root]) for root in roots}
        diff_samples: list[float] = []
        for i, root_a in enumerate(roots):
            for root_b in roots[i + 1:]:
                if float(np.linalg.norm(medians[root_a] - medians[root_b]))                         >= self.config.diff_pair_min_dist_m:
                    value = shape_distance(root_a, root_b)
                    if value is not None:
                        diff_samples.append(float(value))
        if len(same_samples) < 4 or len(diff_samples) < 4:
            self.diagnostics["shape_calibration_starved"] = 1
            return 0
        shape_calibration = fit_cue_calibration(
            same_samples={"shape_dist": same_samples},
            diff_samples={"shape_dist": diff_samples},
        )
        if shape_calibration.d_prime("shape_dist") < 0.5:
            self.diagnostics["shape_cue_abstained"] = 1
            return 0  # body shape cannot separate players on this footage

        # --- mutual-best conservative merges ---------------------------------
        merges = 0
        while True:
            candidates: dict[ChunkKey, tuple[float, ChunkKey]] = {}
            live_roots = [root for root in roots if root in members and root in shapes]
            for i, root_a in enumerate(live_roots):
                for root_b in live_roots[i + 1:]:
                    if not self._cluster_compatible(
                        members[root_a], members[root_b], llr_lookup
                    ):
                        continue
                    cross = [
                        llr_lookup[_pair_key(a, b)]
                        for a in members[root_a] for b in members[root_b]
                        if _pair_key(a, b) in llr_lookup
                    ]
                    if not cross or max(cross) <= 0:
                        continue  # never co-visible, or geometry disagrees
                    stature_a, stature_b = shapes[root_a][1], shapes[root_b][1]
                    if (
                        stature_a is not None and stature_b is not None
                        and abs(stature_a - stature_b) > self.config.graph_shape_stature_max_m
                    ):
                        continue
                    value = shape_distance(root_a, root_b)
                    llr_shape = shape_calibration.llr(
                        "shape_dist", value,
                        clip_pos=self.config.graph_llr_positive_cap,
                    )
                    if llr_shape <= 0:
                        continue
                    total = max(cross) + llr_shape
                    if total < self.config.graph_llr_merge_threshold:
                        continue
                    for source, target in ((root_a, root_b), (root_b, root_a)):
                        best = candidates.get(source)
                        if best is None or total > best[0]:
                            candidates[source] = (total, target)
            merged_this_round = False
            for root_a in sorted(candidates):
                total, root_b = candidates[root_a]
                partner = candidates.get(root_b)
                if partner is None or partner[1] != root_a:
                    continue  # not mutual best
                if root_a not in members or root_b not in members:
                    continue
                keep, absorb = sorted((root_a, root_b))
                parent[absorb] = keep
                members[keep] = sorted(members[keep] + members[absorb])
                del members[absorb]
                descriptor, stature, _frames = self._cluster_shape(members[keep])
                shapes.pop(absorb, None)
                if descriptor is not None and descriptor.is_defined():
                    shapes[keep] = (descriptor, stature)
                else:
                    shapes.pop(keep, None)
                merges += 1
                merged_this_round = True
            if not merged_this_round:
                break
        return merges

    def _chimera_veto_pass(
        self,
        cluster_members: dict[int, list[ChunkKey]],
        cluster_of: dict[ChunkKey, int],
        llr_lookup: dict[tuple[ChunkKey, ChunkKey], float],
    ) -> int:
        """F13: surgically split chimera clusters on the lifted purity signature.

        A cluster whose lifted torso residuals carry the chimera signature has an
        intruder; the per-camera residual bias names the intruding camera. That
        camera's chunks are EVICTED into fresh singleton clusters directly (a
        deterministic split — leaving it to refinement would let the innocent
        chunks scatter first, since the intruder poisons their within-cluster
        scores too), and the offending pair LLRs are vetoed down to
        ``graph_chimera_veto_llr`` so no later pass (refine / rescue / attach /
        shape round) can weld the pieces back together.
        """

        from identity.p3_association.cluster_lift import cluster_purity

        evictions = 0
        next_cluster_id = max(cluster_members, default=-1) + 1
        for cluster_id in sorted(cluster_members):
            keys = cluster_members[cluster_id]
            if len({key[0] for key in keys}) < 2:
                continue
            lifts = self._cluster_lifts(keys)
            if len(lifts) < self.config.graph_shape_min_frames:
                continue
            purity = cluster_purity(
                lifts,
                chimera_torso_residual_px=self.config.graph_chimera_torso_residual_px,
                chimera_frame_fraction=self.config.graph_chimera_frame_fraction,
            )
            if not purity.chimera_suspect or purity.worst_camera is None:
                continue
            intruders = [key for key in keys if key[0] == purity.worst_camera]
            others = [key for key in keys if key[0] != purity.worst_camera]
            if not intruders or not others:
                continue
            for intruder in intruders:
                for other in others:
                    pair = _pair_key(intruder, other)
                    llr_lookup[pair] = min(
                        llr_lookup.get(pair, 0.0), self.config.graph_chimera_veto_llr
                    )
                cluster_members[next_cluster_id] = [intruder]
                cluster_of[intruder] = next_cluster_id
                next_cluster_id += 1
                evictions += 1
            cluster_members[cluster_id] = sorted(others)
        return evictions

    def _refine(
        self,
        cluster_members: dict[int, list[ChunkKey]],
        cluster_of: dict[ChunkKey, int],
        llr_lookup: dict[tuple[ChunkKey, ChunkKey], float],
    ) -> int:
        """Move single chunks between clusters while total LLR improves.

        This is the escape hatch greedy single-linkage lacks: an early bad merge
        (or a chunk welded in by a since-outvoted edge) can be reconsidered. A
        chunk whose affinity to its own cluster is clearly negative may also split
        out into a fresh singleton.
        """

        moves = 0
        next_cluster_id = max(cluster_members, default=-1) + 1
        for _ in range(max(0, self.config.graph_refine_passes)):
            changed = False
            for key in sorted(self._chunks):
                current_id = cluster_of[key]
                rest = [m for m in cluster_members[current_id] if m != key]
                current_score = sum(
                    llr_lookup.get(_pair_key(key, other), 0.0) for other in rest
                )
                best_id, best_gain = None, 0.0
                for candidate_id in sorted(cluster_members):
                    if candidate_id == current_id:
                        continue
                    candidate = cluster_members[candidate_id]
                    # A move needs real evidence volume behind it, same floor as
                    # rescues — a thin edge must not relocate a tracklet.
                    support = sum(
                        self._support_lookup.get(_pair_key(key, other), 0)
                        for other in candidate
                    )
                    if support < self.config.graph_rescue_min_covis:
                        continue
                    if not self._cluster_compatible([key], candidate, llr_lookup):
                        continue
                    score = sum(
                        llr_lookup.get(_pair_key(key, other), 0.0) for other in candidate
                    )
                    gain = score - current_score
                    if gain > best_gain + self.config.graph_move_margin:
                        best_id, best_gain = candidate_id, gain
                if best_id is None and rest and current_score < -self.config.graph_move_margin:
                    cluster_members[next_cluster_id] = []
                    best_id = next_cluster_id
                    next_cluster_id += 1
                if best_id is None:
                    continue
                if rest:
                    cluster_members[current_id] = rest
                else:
                    del cluster_members[current_id]
                cluster_members[best_id] = sorted(cluster_members[best_id] + [key])
                cluster_of[key] = best_id
                moves += 1
                changed = True
            if not changed:
                break
        return moves

    def _rescue_singletons(
        self,
        cluster_members: dict[int, list[ChunkKey]],
        cluster_of: dict[ChunkKey, int],
        edges: list[GraphEdge],
        llr_lookup: dict[tuple[ChunkKey, ChunkKey], float],
    ) -> None:
        """Attach a leftover singleton when the constraints leave ONE explanation.

        The archetype is the wicketkeeper in a camera whose posture is polluted by
        occlusion: it cannot reach the merge threshold on its own, but every other
        cluster at its position is impossible (cannot-link with that camera's
        striker/non-striker/bowler tracklets), ground agrees with the keeper
        cluster, and nothing contradicts it. Requiring EXACTLY one compatible
        candidate keeps ambiguous cases (e.g. an umpire between two clusters)
        untouched.
        """

        edge_partners: dict[ChunkKey, dict[int, float]] = defaultdict(dict)
        rescue_grade: dict[tuple[ChunkKey, ChunkKey], bool] = {}
        for edge in edges:
            # A rescue is a below-threshold attachment justified by constraint
            # structure; it still needs real evidence volume — a handful of
            # co-visible frames must not move a tracklet between identities.
            rescue_grade[_pair_key(edge.key_a, edge.key_b)] = (
                edge.llr_ground > 0.0
                and edge.gated_frames >= self.config.graph_rescue_min_covis
            )
        for singleton in sorted(key for key, cid in cluster_of.items()
                                if len(cluster_members[cluster_of[key]]) == 1):
            candidates: list[int] = []
            for candidate_id in sorted(cluster_members):
                if candidate_id == cluster_of[singleton]:
                    continue
                members = cluster_members[candidate_id]
                pair_llrs = [
                    llr_lookup.get(_pair_key(singleton, member)) for member in members
                ]
                if not any(v is not None for v in pair_llrs):
                    continue
                if not any(
                    rescue_grade.get(_pair_key(singleton, member), False)
                    for member in members
                ):
                    continue
                if not self._cluster_compatible([singleton], members, llr_lookup):
                    continue
                candidates.append(candidate_id)
            if len(candidates) != 1:
                continue
            old_id = cluster_of[singleton]
            target = candidates[0]
            cluster_members[target] = sorted(cluster_members[target] + [singleton])
            del cluster_members[old_id]
            cluster_of[singleton] = target
            self.diagnostics["rescued_singletons"] += 1

    def _attach_fragments_by_trajectory(
        self,
        cluster_members: dict[int, list[ChunkKey]],
        cluster_of: dict[ChunkKey, int],
    ) -> None:
        """Attach leftover fragments to the binding whose TRAJECTORY they lie on.

        When P2 shatters (low light, sprints), a player's camera-track breaks into
        many short chunks with too little pairwise co-visibility for normal edges.
        But the player's multi-camera binding still traces a fused trajectory
        through every one of those fragments — the same information the ghost
        markers draw. A fragment is attached when it rides one binding's
        trajectory (median distance within the gate for most of its life),
        no same-camera overlap contradicts it, posture does not veto it, and no
        second binding competes.
        """

        def is_anchor_cluster(keys: list[ChunkKey]) -> bool:
            cameras = {key[0] for key in keys}
            longest = max(len(self._chunks[key].frames) for key in keys)
            return len(cameras) >= 2 or longest >= self.config.binding_min_single_frames

        postures = self._chunk_postures()
        for _ in range(2):  # attached fragments extend trajectories; one repeat
            targets: dict[int, dict[int, np.ndarray]] = {}
            for cid, keys in cluster_members.items():
                if not is_anchor_cluster(keys):
                    continue
                trajectory: dict[int, list[np.ndarray]] = defaultdict(list)
                for key in keys:
                    for frame, ground in self._chunks[key].ground_by_frame.items():
                        trajectory[frame].append(ground)
                targets[cid] = {
                    frame: np.mean(np.asarray(points), axis=0)
                    for frame, points in trajectory.items()
                }
            changed = False
            for cid in sorted(cluster_members):
                keys = cluster_members.get(cid)
                if keys is None or is_anchor_cluster(keys):
                    continue
                candidates: list[tuple[float, int]] = []
                for target_id, trajectory in sorted(targets.items()):
                    if target_id == cid:
                        continue
                    if not self._cluster_compatible(keys, cluster_members[target_id], {}):
                        continue  # same-camera overlap: a different person
                    distances: list[float] = []
                    fragment_frames = 0
                    for key in keys:
                        chunk = self._chunks[key]
                        fragment_frames += len(chunk.frames)
                        for frame, ground in chunk.ground_by_frame.items():
                            reference = trajectory.get(frame)
                            if reference is not None:
                                distances.append(float(np.linalg.norm(ground - reference)))
                    if len(distances) < max(10, fragment_frames // 2):
                        continue
                    median_distance = float(np.median(distances))
                    if median_distance > self.config.graph_traj_attach_gate_m:
                        continue
                    # Posture veto: a fragment that clearly belongs to a different
                    # build must not ride a nearby trajectory. Use the fragment's
                    # best-supported posture aggregate (most samples), not just its
                    # first chunk, whose aggregate is often undefined for a short
                    # fragment — so the veto actually gets a chance to fire.
                    fragment_posture = max(
                        (postures.get(k) for k in keys),
                        key=lambda agg: (
                            sum(agg.count.values()) if agg is not None and agg.is_defined() else -1
                        ),
                        default=None,
                    )
                    veto = False
                    for target_key in cluster_members[target_id]:
                        z = posture_distance_z(
                            fragment_posture, postures.get(target_key),
                        )
                        if z is not None and z[0] > 3.5:
                            veto = True
                            break
                    if not veto:
                        candidates.append((median_distance, target_id))
                if not candidates:
                    continue
                candidates.sort()
                best_distance, best_target = candidates[0]
                if len(candidates) > 1 and candidates[1][0] < max(
                    best_distance * 1.5, best_distance + 0.5
                ):  # H6: multiplicative-only margin degenerates at ~0 distance
                    self.diagnostics["fragment_attach_ambiguous"] += 1
                    continue
                cluster_members[best_target] = sorted(cluster_members[best_target] + keys)
                for key in keys:
                    cluster_of[key] = best_target
                del cluster_members[cid]
                self.diagnostics["fragments_attached"] += 1
                changed = True
            if not changed:
                break

    # --------------------------------------------------------------- emit

    def emit_frame(
        self,
        frame_index: int,
        dets_per_cam: dict[str, list[Detection3]],
        solution: GraphSolution,
        proj_matrices: dict[str, np.ndarray],
        camera_centers: dict[str, np.ndarray] | None = None,
    ) -> list[Correspondence]:
        """Group one frame's detections by binding and build correspondences."""

        groups: dict[str, dict[str, int]] = defaultdict(dict)
        leftovers: list[tuple[str, int]] = []
        for cam_id in sorted(dets_per_cam):
            for det in dets_per_cam[cam_id]:
                tid = det.local_track_id or self._syn_of_det.get(
                    (frame_index, cam_id, det.player_index)
                )
                chunk = (
                    self._det_chunk.get((frame_index, cam_id, tid))
                    if tid is not None else None
                )
                binding = solution.binding_of_chunk.get(chunk) if chunk else None
                if binding is None:
                    leftovers.append((cam_id, det.player_index))
                    continue
                existing = groups[binding].get(cam_id)
                if existing is not None:
                    # <=cannot_link_overlap_frames same-camera overlap can reach here;
                    # keep the higher-confidence detection, spill the other.
                    current = dets_per_cam[cam_id][existing]
                    if det.confidence > current.confidence:
                        groups[binding][cam_id] = det.player_index
                        leftovers.append((cam_id, existing))
                    else:
                        leftovers.append((cam_id, det.player_index))
                    self.diagnostics["emit_same_camera_conflicts"] += 1
                else:
                    groups[binding][cam_id] = det.player_index

        index_of = {
            cam_id: {det.player_index: i for i, det in enumerate(dets)}
            for cam_id, dets in dets_per_cam.items()
        }
        binding_posture: dict[str, PostureAggregate] = {}
        if self.config.emit_posture:
            if self._emit_posture_cache is None:
                self._emit_posture_cache = self.binding_postures(solution)
            binding_posture = self._emit_posture_cache

        correspondences: list[Correspondence] = []
        cluster_id = 0
        for binding in sorted(groups):
            member_map = {
                cam_id: index_of[cam_id][player_index]
                for cam_id, player_index in sorted(groups[binding].items())
            }
            corr = _build_correspondence(
                cluster_id, member_map, dets_per_cam, proj_matrices, self.config,
                camera_centers or {},
            )
            correspondences.append(replace(
                corr, binding_id=binding, posture=binding_posture.get(binding),
            ))
            cluster_id += 1
        for cam_id, player_index in sorted(leftovers):
            member_map = {cam_id: index_of[cam_id][player_index]}
            corr = _build_correspondence(
                cluster_id, member_map, dets_per_cam, proj_matrices, self.config,
                camera_centers or {},
            )
            correspondences.append(corr)
            cluster_id += 1
        return correspondences

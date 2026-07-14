"""Per-frame cross-camera association (P3).

Two modes (config ``matching_mode``):

* ``pairwise_anchor`` — Vedant's sticky-anchor star matching, kept for A/B
  comparison and as a fallback.
* ``multiway_cycle`` (default) — geometry-guided multi-view clustering. All
  pairwise epipolar/triangulation matches feed a *constrained agglomerative
  clustering*: components merge only when the foot point triangulated over the
  whole candidate cluster reprojects within tolerance into every member view
  (a RANSAC-based cycle-consistency check) and no camera appears twice. This
  replaces the anchor-star "average the ground XY" step, which never closed the
  B<->C loop and silently merged players under identical kits.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from itertools import combinations

import numpy as np
from scipy.optimize import linear_sum_assignment

from pose_estimation.cricket.geometry import (
    bbox_bottom_center_px,
    ground_contact_pixel,
    ground_contact_pixel_ex,
    ground_covariance,
    ground_from_reprojection,
    ground_from_reprojection_ex,
    pixel_to_plane_xy,
    parallax_angle_deg,
    parallax_weight,
    pixel_to_ground_xy,
    reprojection_error_px,
    robust_fuse_ground,
    sampson_distance,
    triangulate_dlt,
)
from pose_estimation.triangulation import (
    ransac_triangulate_point,
    reprojection_errors_for_point,
    triangulate_skeleton_ransac,
)
from pose_estimation.cricket.pose_shape import (
    PoseProportions,
    PostureAggregate,
    limb_proportion_descriptor,
    torso_anthropometric_ok,
)
from scripts.association.config import P3AssociationConfig
from scripts.association.appearance import appearance_distance
from scripts.association.geometry_cache import GeometryCache, PairGeometry

_L_ANKLE, _R_ANKLE = 15, 16  # COCO-17 ankle indices
# Body joints used by the pose-shape descriptor (skip the 5 face joints 0-4).
_BODY_JOINTS = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


@dataclass(frozen=True)
class Detection3:
    cam_id: str
    player_index: int          # index into the frame record's players[] (for write-back)
    bbox_xywh_px: list[float]
    keypoints_px: np.ndarray   # (17, 2)
    keypoint_conf: np.ndarray  # (17,)
    confidence: float
    local_track_id: str | None = None
    ground_xy: np.ndarray | None = None
    appearance: np.ndarray | None = None
    # True when ground_xy came from an upper-body height-plane estimate because
    # the feet were cut off / occluded — carries extra positional uncertainty.
    ground_approx: bool = False
    # Temporally-smoothed foot-contact pixel for the EMITTED position only (F7). Set by
    # a per-(camera, tracklet) smoothing pre-pass; None => use the per-frame foot. Never
    # feeds the clustering gate, so identity stays invariant.
    emit_foot_px: np.ndarray | None = None
    # Native-skeleton keypoints (Halpe-26 when the P1 model provides them). Carries the
    # heel/toe ground-contact landmarks for foot_contact_mode v3; None on COCO-17-only
    # runs. Emit-path only — never feeds the clustering gate.
    native_keypoints_px: np.ndarray | None = None
    native_keypoint_conf: np.ndarray | None = None
    # Wave-5b: True when another detection in the SAME camera overlaps this one
    # (bbox IoU >= contested_iou) — the camera cannot separate the two players, so
    # its ground/appearance/posture evidence for both is down-weighted or muted.
    contested: bool = False


@dataclass(frozen=True)
class Correspondence:
    cluster_id: int
    members: dict[str, Detection3]   # cam_id -> detection (1..7 members)
    ground_xy: np.ndarray            # world (x, y) of the foot, or NaNs
    track_confidence: float
    single_camera: bool
    mean_reprojection_error_px: float | None = None
    cycle_consistent: bool = True
    ground_spread_m: float | None = None
    pose_descriptor: PoseProportions | None = None
    # Persistent tracklet-graph identity (``association_mode: tracklet_graph``).
    # Same physical player => same binding_id for the whole delivery, across all
    # cameras. None in per-frame mode and for unbound leftovers.
    binding_id: str | None = None
    # Billboard (ground-anchored monocular) posture aggregate of the binding — the
    # facing-pair-capable body-shape cue, unlike pose_descriptor which needs
    # triangulation parallax. Emitted only when P3 ``emit_posture`` is on (F6b);
    # consumed by the P4a teleport veto and re-entry gate.
    posture: "PostureAggregate | None" = None
    # 2x2 world-frame covariance of ground_xy (F9a): Gauss-Newton posterior for
    # multi-view z0_reproj clusters, inflated homography-Jacobian model for
    # single-camera ones. Emitted only when P3 ``emit_ground_cov`` is on; the
    # measurement-noise input for P4's uncertainty-aware Kalman R (F10).
    ground_cov: np.ndarray | None = None


@dataclass
class TemporalLinkMemory:
    """Short-term evidence that two P2 tracklets represent the same player.

    H4: with ``decay`` < 1, pair counts are multiplied by it every frame so the
    support reflects RECENT agreement — without decay one early wrong
    co-clustering self-reinforces forever (per-frame mode only; the default
    tracklet-graph mode does not use this memory). 1.0 = legacy, no decay.
    """

    confirm_frames: int = 3
    decay: float = 1.0
    counts: dict[tuple[str, str, str, str], int] = field(default_factory=lambda: defaultdict(int))

    @staticmethod
    def _key(left: Detection3, right: Detection3) -> tuple[str, str, str, str] | None:
        if not left.local_track_id or not right.local_track_id or left.cam_id == right.cam_id:
            return None
        first, second = (left, right) if left.cam_id < right.cam_id else (right, left)
        return first.cam_id, first.local_track_id, second.cam_id, second.local_track_id

    def support(self, left: Detection3, right: Detection3) -> float:
        key = self._key(left, right)
        if key is None:
            return 0.0
        return min(1.0, self.counts.get(key, 0) / max(self.confirm_frames, 1))

    def update(self, correspondences: list[Correspondence], config: P3AssociationConfig) -> None:
        if self.decay < 1.0:
            # H4: age all pair evidence so support reflects recent agreement and an
            # early wrong co-clustering cannot self-reinforce indefinitely.
            for key in list(self.counts):
                aged = self.counts[key] * self.decay
                if aged < 0.05:
                    del self.counts[key]
                else:
                    self.counts[key] = aged
        for correspondence in correspondences:
            if correspondence.single_camera:
                continue
            if (
                correspondence.ground_spread_m is None
                or correspondence.ground_spread_m > config.ground_cluster_gate_m
            ):
                continue
            for left, right in combinations(correspondence.members.values(), 2):
                key = self._key(left, right)
                if key is not None:
                    self.counts[key] += 1


@dataclass(frozen=True)
class AnchorState:
    anchor_id: str
    frames_since_switch: int


def select_anchor(
    dets_per_cam: dict[str, list[Detection3]],
    prev: AnchorState | None,
    config: P3AssociationConfig,
) -> AnchorState:
    """Sticky anchor: switch only if a challenger leads by ``anchor_hysteresis_margin``."""

    counts = {cam: len(dets) for cam, dets in dets_per_cam.items() if dets}
    if not counts:
        return prev if prev is not None else AnchorState(config.anchor_priority[0], 0)

    if prev is None:
        best = max(counts.items(), key=lambda kv: (kv[1], -_priority_rank(kv[0], config)))
        return AnchorState(best[0], 0)

    current_count = counts.get(prev.anchor_id, 0)
    challenger_id, challenger_count = max(
        ((cam, cnt) for cam, cnt in counts.items() if cam != prev.anchor_id),
        key=lambda kv: (kv[1], -_priority_rank(kv[0], config)),
        default=(None, -1),
    )
    if (
        challenger_id is not None
        and challenger_count > current_count + config.anchor_hysteresis_margin
        and prev.frames_since_switch >= config.anchor_hysteresis_frames
    ):
        return AnchorState(challenger_id, 0)
    return AnchorState(prev.anchor_id, prev.frames_since_switch + 1)


def _priority_rank(cam_id: str, config: P3AssociationConfig) -> int:
    try:
        return config.anchor_priority.index(cam_id)
    except ValueError:
        return len(config.anchor_priority)


def _foot_pixel(det: Detection3, config: P3AssociationConfig) -> np.ndarray:
    """Ground-contact reference shared with P2."""

    # The GATE / cost / triangulation-consistency path always uses the LEGACY foot so
    # that which detections cluster is byte-identical regardless of foot_contact_mode.
    # Only the emitted cluster position (z0_reproj) uses the v2 foot -- see
    # _emit_foot_and_height. This decoupling keeps identity stable (verified: foot v2
    # feeding the gate inflated cluster counts +25% on some deliveries).
    return ground_contact_pixel(
        det.bbox_xywh_px,
        det.keypoints_px,
        det.keypoint_conf,
        ankle_confidence_min=config.ankle_conf_min,
        max_ankle_above_bbox_fraction=config.max_ankle_above_bbox_fraction,
        mode="legacy",
    )


def _emit_foot_and_height(det: Detection3, config: P3AssociationConfig) -> tuple[np.ndarray, float]:
    """Foot pixel + landmark height for the EMITTED position only (honours foot_contact_mode).

    Uses the temporally-smoothed foot pixel (F7) when a smoothing pre-pass has set it,
    keeping the per-frame landmark height/source.
    """

    pixel, height, _source = ground_contact_pixel_ex(
        det.bbox_xywh_px,
        det.keypoints_px,
        det.keypoint_conf,
        ankle_confidence_min=config.ankle_conf_min,
        max_ankle_above_bbox_fraction=config.max_ankle_above_bbox_fraction,
        mode=config.foot_contact_mode,
        ankle_height_m=config.ankle_height_m,
        horizontal_margin_frac=config.foot_horizontal_margin_frac,
        level_frac=config.foot_level_frac,
        native_keypoints_px=det.native_keypoints_px,
        native_confidence=det.native_keypoint_conf,
        foot_kp_conf_min=config.foot_kp_conf_min,
        foot_height_m=config.foot_height_m,
    )
    smoothed = getattr(det, "emit_foot_px", None)
    if smoothed is not None:
        smoothed = np.asarray(smoothed, dtype=float)
        if smoothed.shape == (2,) and np.isfinite(smoothed).all():
            pixel = smoothed
    return pixel, height


def smooth_emit_feet(
    detections_by_frame: dict[int, dict[str, list[Detection3]]],
    config: P3AssociationConfig,
) -> dict[int, dict[str, list[Detection3]]]:
    """Attach a temporally-smoothed emit-foot pixel per (camera, tracklet) time series (F7).

    Per-frame pose jitter (a foot bouncing ±5-15 px, an occasional hallucinated ankle)
    turns into ground jitter and the odd teleport. Here each (camera, local_track_id)
    foot-pixel series is passed through a short **median** filter (window
    ``config.foot_smooth_window``, robust to single-frame spikes) and, optionally, a
    light EMA; the result is stored on the detection as ``emit_foot_px`` for the EMITTED
    position only. Untracked detections and windows of 1 are left untouched. Nothing here
    feeds the clustering gate, so identity is unchanged.
    """

    from dataclasses import replace as _dc_replace

    window = int(getattr(config, "foot_smooth_window", 1) or 1)
    if window <= 1:
        return detections_by_frame
    half = window // 2

    # Collect each (camera, tracklet) foot series in frame order.
    series: dict[tuple[str, str], list[tuple[int, int, np.ndarray]]] = {}
    for frame_index in sorted(detections_by_frame):
        for camera_id, dets in detections_by_frame[frame_index].items():
            for det_idx, det in enumerate(dets):
                if not det.local_track_id:
                    continue
                foot, _height = _emit_foot_and_height(det, config)
                foot = np.asarray(foot, dtype=float)
                if foot.shape == (2,) and np.isfinite(foot).all():
                    series.setdefault((camera_id, det.local_track_id), []).append(
                        (frame_index, det_idx, foot)
                    )

    # Median-smooth each series -> lookup keyed by (frame, camera, det_idx).
    smoothed: dict[tuple[int, str, int], np.ndarray] = {}
    for (camera_id, _track_id), entries in series.items():
        entries.sort(key=lambda item: item[0])
        feet = np.asarray([entry[2] for entry in entries], dtype=float)
        count = len(entries)
        for position, (frame_index, det_idx, _foot) in enumerate(entries):
            lo = max(0, position - half)
            hi = min(count, position + half + 1)
            smoothed[(frame_index, camera_id, det_idx)] = np.median(feet[lo:hi], axis=0)

    if not smoothed:
        return detections_by_frame

    out: dict[int, dict[str, list[Detection3]]] = {}
    for frame_index, cams in detections_by_frame.items():
        out[frame_index] = {}
        for camera_id, dets in cams.items():
            new_dets = []
            for det_idx, det in enumerate(dets):
                key = (frame_index, camera_id, det_idx)
                if key in smoothed:
                    new_dets.append(_dc_replace(det, emit_foot_px=smoothed[key]))
                else:
                    new_dets.append(det)
            out[frame_index][camera_id] = new_dets
    return out


def _detection_ground_xy(
    detection: Detection3,
    projection: np.ndarray,
    config: P3AssociationConfig,
) -> np.ndarray:
    if detection.ground_xy is not None:
        point = np.asarray(detection.ground_xy, dtype=float)
        if point.shape == (2,) and np.isfinite(point).all():
            return point
    return pixel_to_ground_xy(_foot_pixel(detection, config), projection)


def _is_opposite_pair(cam_a: str, cam_b: str, config: P3AssociationConfig) -> bool:
    wanted = frozenset((cam_a, cam_b))
    return any(frozenset(pair) == wanted for pair in config.opposite_camera_pairs)


def _directed_pair(pg: PairGeometry, cam_a: str, cam_b: str) -> PairGeometry:
    """Return pair geometry oriented so F satisfies ``x_b^T F x_a = 0`` for (cam_a, cam_b)."""

    F = pg.F if pg.cam_id_a == cam_a else pg.F.T
    return PairGeometry(cam_a, cam_b, F, pg.is_degenerate, pg.w_epi, pg.w_tri, pg.huber_delta)


def build_cost_matrix(
    dets_a: list[Detection3],
    dets_b: list[Detection3],
    P_a: np.ndarray,
    P_b: np.ndarray,
    C_a: np.ndarray,
    C_b: np.ndarray,
    pg: PairGeometry,
    config: P3AssociationConfig,
    temporal_memory: TemporalLinkMemory | None = None,
) -> np.ndarray:
    """Real-pair cost matrix; impossible geometry remains a finite sentinel."""

    M, N = len(dets_a), len(dets_b)
    cost = np.full((M, N), 1e6, dtype=float)
    ground_gate = (
        config.opposite_pair_ground_gate_m
        if _is_opposite_pair(pg.cam_id_a, pg.cam_id_b, config)
        else config.ground_distance_gate_m
    )
    # Degenerate (near-collinear / facing) pairs have ill-conditioned epipolar
    # geometry; geometry_cache already flags them and zeroes ``w_epi``. Honour that
    # here by dropping the epipolar term for such pairs and reallocating its weight
    # to the trustworthy ground cue, so the row keeps its tuned 0..1 scale against
    # ``pair_unmatched_cost``. (These flags were previously computed but never read,
    # so the co-observing facing pairs -- exactly the pairs that must cluster -- were
    # scored with a full-weight, unreliable Sampson term, causing under-merges.)
    epipolar_factor = 0.0 if pg.is_degenerate else 1.0
    epipolar_weight = config.epipolar_weight * epipolar_factor
    ground_weight = config.ground_weight + config.epipolar_weight * (1.0 - epipolar_factor)

    for i, da in enumerate(dets_a):
        ground_a = _detection_ground_xy(da, P_a, config)
        if not np.isfinite(ground_a).all():
            continue
        for j, db in enumerate(dets_b):
            ground_b = _detection_ground_xy(db, P_b, config)
            if not np.isfinite(ground_b).all():
                continue
            ground_distance = float(np.linalg.norm(ground_a - ground_b))
            if ground_distance > ground_gate:
                continue
            shared_confidence = np.minimum(da.keypoint_conf, db.keypoint_conf)
            top5 = np.argsort(shared_confidence)[-5:]
            usable = [
                k for k in top5
                if da.keypoint_conf[k] > config.keypoint_match_conf_min
                and db.keypoint_conf[k] > config.keypoint_match_conf_min
            ]
            if usable:
                # sampson_distance is squared pixels; compare its square-root to
                # the configured pixel scale.
                epipolar_px = float(np.median([
                    np.sqrt(max(0.0, sampson_distance(da.keypoints_px[k], pg.F, db.keypoints_px[k])))
                    for k in usable
                ]))
            else:
                epipolar_px = config.epipolar_scale_px
            appearance = appearance_distance(da.appearance, db.appearance)
            appearance_cost = 0.5 if appearance is None else appearance
            continuity = temporal_memory.support(da, db) if temporal_memory is not None else 0.0
            cost[i, j] = max(
                0.0,
                ground_weight * (ground_distance / ground_gate)
                + epipolar_weight * min(epipolar_px / config.epipolar_scale_px, 1.0)
                + config.appearance_weight * appearance_cost
                - config.temporal_link_bonus * continuity,
            )
    return cost


def solve_optional_assignment(cost: np.ndarray, unmatched_cost: float) -> list[tuple[int, int]]:
    """Hungarian assignment with a private no-match choice for every detection."""

    cost = np.asarray(cost, dtype=float)
    if cost.ndim != 2:
        raise ValueError("cost must be a 2D matrix")
    rows_count, columns_count = cost.shape
    if rows_count == 0 or columns_count == 0:
        return []
    size = rows_count + columns_count
    augmented = np.full((size, size), 1e6, dtype=float)
    augmented[:rows_count, :columns_count] = cost
    # Leaving both endpoints unmatched should cost exactly unmatched_cost.
    half = unmatched_cost / 2.0
    for row in range(rows_count):
        augmented[row, columns_count + row] = half
    for column in range(columns_count):
        augmented[rows_count + column, column] = half
    augmented[rows_count:, columns_count:] = 0.0
    rows, columns = linear_sum_assignment(augmented)
    return [
        (int(row), int(column))
        for row, column in zip(rows, columns)
        if row < rows_count and column < columns_count and cost[row, column] < unmatched_cost
    ]


def associate_frame(
    dets_per_cam: dict[str, list[Detection3]],
    proj_matrices: dict[str, np.ndarray],
    geo: GeometryCache,
    anchor: AnchorState | None,
    config: P3AssociationConfig,
    temporal_memory: TemporalLinkMemory | None = None,
) -> tuple[list[Correspondence], AnchorState]:
    """Associate all detections in one synchronized frame into player clusters."""

    new_anchor = select_anchor(dets_per_cam, anchor, config)
    if config.matching_mode == "pairwise_anchor":
        clusters = _associate_pairwise_anchor(
            dets_per_cam, proj_matrices, geo, new_anchor, config, temporal_memory
        )
    else:
        clusters = _associate_multiway_cycle(
            dets_per_cam, proj_matrices, geo, config, temporal_memory
        )
    ordered = _order_clusters(clusters, new_anchor.anchor_id, config)
    correspondences = [
        _build_correspondence(idx, members, dets_per_cam, proj_matrices, config, geo.camera_centers)
        for idx, members in enumerate(ordered)
    ]
    if temporal_memory is not None:
        temporal_memory.update(correspondences, config)
    return correspondences, new_anchor


# --- pairwise-anchor (fallback mode) -----------------------------------------

def _associate_pairwise_anchor(
    dets_per_cam, proj_matrices, geo, anchor, config, temporal_memory=None
):
    anchor_id = anchor.anchor_id
    dets_anchor = dets_per_cam.get(anchor_id)
    if not dets_anchor:  # anchor empty this frame: fall back to the busiest camera
        anchor_id = max(
            (c for c, d in dets_per_cam.items() if d),
            key=lambda c: (len(dets_per_cam[c]), -_priority_rank(c, config)),
            default=None,
        )
        if anchor_id is None:
            return []
        dets_anchor = dets_per_cam[anchor_id]

    member_sets: list[dict[str, int]] = [{anchor_id: i} for i in range(len(dets_anchor))]
    claimed: set[tuple[str, int]] = set()
    for partner_id, dets_partner in dets_per_cam.items():
        if partner_id == anchor_id or not dets_partner:
            continue
        pg = geo.pairs.get(_sorted_pair(anchor_id, partner_id))
        if pg is None:
            continue
        pg_dir = _directed_pair(pg, anchor_id, partner_id)
        cost = build_cost_matrix(
            dets_anchor, dets_partner, proj_matrices[anchor_id], proj_matrices[partner_id],
            geo.camera_centers[anchor_id], geo.camera_centers[partner_id], pg_dir, config,
            temporal_memory,
        )
        for r, c in solve_optional_assignment(cost, config.pair_unmatched_cost):
            member_sets[r][partner_id] = c
            claimed.add((partner_id, c))

    clusters = [m for m in member_sets]
    # Partner detections never matched to the anchor become single-camera clusters.
    for cam, dets in dets_per_cam.items():
        if cam == anchor_id:
            continue
        for idx in range(len(dets)):
            if (cam, idx) not in claimed:
                clusters.append({cam: idx})
    return clusters


# --- multi-way cycle-consistent clustering (default mode) --------------------

def _associate_multiway_cycle(
    dets_per_cam, proj_matrices, geo, config, temporal_memory=None
):
    cams = sorted(c for c, d in dets_per_cam.items() if d)
    edges: list[tuple[float, tuple[str, int], tuple[str, int]]] = []
    for cam_a, cam_b in combinations(cams, 2):
        pg = geo.pairs.get(_sorted_pair(cam_a, cam_b))
        if pg is None:
            continue
        dets_a, dets_b = dets_per_cam[cam_a], dets_per_cam[cam_b]
        pg_dir = _directed_pair(pg, cam_a, cam_b)
        cost = build_cost_matrix(
            dets_a, dets_b, proj_matrices[cam_a], proj_matrices[cam_b],
            geo.camera_centers[cam_a], geo.camera_centers[cam_b], pg_dir, config,
            temporal_memory,
        )
        for r, c in solve_optional_assignment(cost, config.pair_unmatched_cost):
            edges.append((float(cost[r, c]), (cam_a, r), (cam_b, c)))

    return _constrained_cluster(edges, dets_per_cam, proj_matrices, config)


def _constrained_cluster(edges, dets_per_cam, proj_matrices, config):
    """Single-linkage clustering gated by one-per-camera and ground consensus."""

    nodes = [(cam, i) for cam, dets in dets_per_cam.items() for i in range(len(dets))]
    parent = {n: n for n in nodes}
    members: dict[tuple, dict[str, int]] = {n: {n[0]: n[1]} for n in nodes}
    point: dict[tuple, np.ndarray | None] = {
        n: _detection_ground_xy(dets_per_cam[n[0]][n[1]], proj_matrices[n[0]], config)
        for n in nodes
    }

    def find(node):
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    for _cost, na, nb in sorted(edges, key=lambda e: e[0]):
        ra, rb = find(na), find(nb)
        if ra == rb:
            continue
        ma, mb = members[ra], members[rb]
        if set(ma) & set(mb):  # one detection per camera per cluster
            continue
        merged = {**ma, **mb}
        merged_point, max_ground_residual, _merged_cov = _ground_consensus_members(
            merged, dets_per_cam, proj_matrices, config
        )
        if merged_point is None or max_ground_residual > config.ground_cluster_gate_m:
            continue
        if len(merged) >= 3:
            _point_3d, max_reprojection = _multiview_reprojection_consistency(
                merged, dets_per_cam, proj_matrices, config
            )
            if _point_3d is None or max_reprojection > config.cycle_reproj_tol_px:
                continue
        parent[rb] = ra
        members[ra] = merged
        point[ra] = merged_point

    seen: set[tuple] = set()
    clusters: list[dict[str, int]] = []
    for node in nodes:
        root = find(node)
        if root not in seen:
            seen.add(root)
            clusters.append(members[root])
    return clusters


def _gather_member_arrays(members, dets_per_cam, proj_matrices, config):
    feet, projections, confidences = [], [], []
    for cam_id, idx in members.items():
        det = dets_per_cam[cam_id][idx]
        feet.append(_foot_pixel(det, config))
        projections.append(proj_matrices[cam_id])
        confidences.append(max(det.confidence, 1e-3))
    return np.asarray(feet, float), np.asarray(projections, float), np.asarray(confidences, float)


def _bbox_iou_xywh(a, b) -> float:
    ax1, ay1, aw, ah = float(a[0]), float(a[1]), float(a[2]), float(a[3])
    bx1, by1, bw, bh = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    ix = max(0.0, min(ax1 + aw, bx1 + bw) - max(ax1, bx1))
    iy = max(0.0, min(ay1 + ah, by1 + bh) - max(ay1, by1))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def mark_contested_detections(
    detections_by_cam: dict[str, list[Detection3]], iou_thr: float
) -> dict[str, list[Detection3]]:
    """Flag same-camera detection pairs whose bboxes overlap >= iou_thr (Wave-5b).

    Cross-camera overlap is meaningless (different viewpoints), so the comparison is
    strictly within each camera's detection list for one frame.
    """

    if iou_thr <= 0.0:
        return detections_by_cam
    out: dict[str, list[Detection3]] = {}
    for cam_id, dets in detections_by_cam.items():
        flags = [False] * len(dets)
        for i, j in combinations(range(len(dets)), 2):
            if _bbox_iou_xywh(dets[i].bbox_xywh_px, dets[j].bbox_xywh_px) >= iou_thr:
                flags[i] = flags[j] = True
        out[cam_id] = [
            replace(det, contested=True) if flag else det
            for det, flag in zip(dets, flags)
        ]
    return out


def _member_ground_sigma_px(detection, config: P3AssociationConfig) -> float:
    """Foot-pixel noise (px) for this detection: base + bbox-scaled + confidence-scaled.

    A small, well-detected foot near the camera is sharp; a large/low-confidence one is
    fuzzy. This sigma feeds the ray-plane Jacobian in :func:`ground_covariance`, so a
    grazing far foot correctly gets a large, elongated ground covariance.
    """

    bbox = np.asarray(getattr(detection, "bbox_xywh_px", None), dtype=float)
    bbox_h = float(bbox[3]) if bbox.shape == (4,) and np.isfinite(bbox[3]) else 0.0
    sigma = float(config.ground_sigma_px_base) + float(config.ground_sigma_px_bbox_frac) * bbox_h
    if getattr(detection, "contested", False):
        sigma *= float(config.contested_sigma_scale)
    return max(sigma, 1e-3)


def _ground_consensus_members(members, dets_per_cam, proj_matrices, config):
    points = []
    covs = []
    for camera_id, index in members.items():
        detection = dets_per_cam[camera_id][index]
        projection = proj_matrices[camera_id]
        point = _detection_ground_xy(detection, projection, config)
        if not np.isfinite(point).all():
            return None, float("inf"), None
        points.append(point)
        if config.ground_fusion_mode == "robust_cov":
            covs.append(
                ground_covariance(
                    _foot_pixel(detection, config),
                    projection,
                    sigma_px=_member_ground_sigma_px(detection, config),
                    var_floor_m=float(config.ground_var_floor_m),
                )
            )
    if not points:
        return None, float("inf"), None
    values = np.asarray(points, dtype=float)
    # The merge GATE is always the max pairwise spread across members -- unchanged, so
    # which clusters form is byte-identical regardless of fusion mode.
    pairwise = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=2)
    max_spread = float(np.max(pairwise))

    # z0_reproj also handles the SINGLE-camera case (len == 1): projecting the ankle
    # onto its z = ankle_height plane instead of z = 0 removes the ~0.94 m grazing-angle
    # bias that a lone camera cannot otherwise correct (measured: mean 0.94 m, p95 1.3 m).
    # A single-member cluster has max_spread == 0, so this never affects the merge gate.
    if config.ground_fusion_mode == "z0_reproj" and len(values) >= 1:
        feet_heights = [_emit_foot_and_height(dets_per_cam[cam][idx], config) for cam, idx in members.items()]
        feet = np.asarray([fh[0] for fh in feet_heights], dtype=float)
        heights = np.asarray([fh[1] for fh in feet_heights], dtype=float)
        projections = np.asarray([proj_matrices[cam] for cam in members], dtype=float)
        confidences = np.asarray(
            [
                max(getattr(det, "confidence", 1.0), 1e-3)
                * (float(config.contested_conf_scale) if getattr(det, "contested", False) else 1.0)
                for det in (dets_per_cam[cam][idx] for cam, idx in members.items())
            ],
            dtype=float,
        )
        solved, solved_cov = ground_from_reprojection_ex(
            feet,
            projections,
            confidences,
            plane_heights=heights,
            huber_delta_px=float(config.ground_reproj_huber_px),
        )
        if np.isfinite(solved).all():
            centre, centre_cov = solved, solved_cov
        else:
            centre, centre_cov = np.median(values, axis=0), None
    elif config.ground_fusion_mode == "robust_cov" and len(values) >= 2:
        fused_xy, fused_cov, _w = robust_fuse_ground(
            values,
            np.asarray(covs, dtype=float),
            huber_delta=float(config.ground_fusion_huber_delta),
        )
        if np.isfinite(fused_xy).all():
            centre, centre_cov = fused_xy, fused_cov
        else:
            centre, centre_cov = np.median(values, axis=0), None
    else:
        centre, centre_cov = np.median(values, axis=0), None
    return centre, max_spread, centre_cov


def _triangulate_members(members, dets_per_cam, proj_matrices, config):
    """Estimate a z=0 foot point and report its maximum reprojection error."""

    ground, _spread, _cov = _ground_consensus_members(members, dets_per_cam, proj_matrices, config)
    if ground is None:
        return None, float("inf")
    point = np.array([ground[0], ground[1], 0.0], dtype=float)
    feet, projections, _confidences = _gather_member_arrays(
        members, dets_per_cam, proj_matrices, config
    )
    errors = reprojection_errors_for_point(point, feet, projections)
    max_reproj = float(np.nanmax(errors)) if np.isfinite(errors).any() else float("inf")
    return point, max_reproj


def _multiview_reprojection_consistency(members, dets_per_cam, proj_matrices, config):
    """RANSAC loop-closure check used once a cluster has at least three views."""

    feet, projections, confidences = _gather_member_arrays(
        members, dets_per_cam, proj_matrices, config
    )
    result = ransac_triangulate_point(
        feet,
        projections,
        confidences,
        reprojection_threshold_px=config.triangulation_reproj_threshold_px,
        min_views=config.triangulation_min_views,
    )
    if not np.isfinite(result.point_xyz).all():
        return None, float("inf")
    errors = reprojection_errors_for_point(result.point_xyz, feet, projections)
    maximum = float(np.nanmax(errors)) if np.isfinite(errors).any() else float("inf")
    return result.point_xyz, maximum


def _joint_parallax_ok(points3d, camera_centers, min_deg):
    """Per-joint mask: True when some member-camera pair triangulates it with >= min_deg parallax.

    Segments spanning only low-parallax (near-collinear / facing) views are noisy in
    depth, so the descriptor must exclude them -- this is what keeps the pose cue from
    firing on the very pairs where cross-camera triangulation is unreliable.
    """

    points3d = np.asarray(points3d, dtype=float).reshape(-1, 3)
    joints = points3d.shape[0]
    ok = np.zeros(joints, dtype=bool)
    centers = [np.asarray(c, dtype=float) for c in camera_centers
               if c is not None and np.isfinite(np.asarray(c, dtype=float)).all()]
    if len(centers) < 2:
        return ok
    for joint in range(joints):
        point = points3d[joint]
        if not np.isfinite(point).all():
            continue
        best = 0.0
        for left in range(len(centers)):
            for right in range(left + 1, len(centers)):
                best = max(best, parallax_angle_deg(centers[left], centers[right], point))
            if best >= min_deg:
                break
        ok[joint] = best >= min_deg
    return ok


def _pose_descriptor_for_members(members, dets_per_cam, proj_matrices, camera_centers, config):
    """Triangulate a reduced skeleton for a cluster and return (descriptor, torso_ok).

    ``torso_ok`` is a soft plausibility flag (None when the torso is not confidently
    co-observed) used only to down-weight confidence for likely chimeras.
    """

    cam_order = list(members.keys())
    keypoints, projections, centers = [], [], []
    for cam_id in cam_order:
        detection = dets_per_cam[cam_id][members[cam_id]]
        kp = np.asarray(detection.keypoints_px, dtype=float).reshape(-1, 2)
        conf = np.asarray(detection.keypoint_conf, dtype=float).reshape(-1, 1)
        keypoints.append(np.concatenate([kp[_BODY_JOINTS], conf[_BODY_JOINTS]], axis=1))
        projections.append(np.asarray(proj_matrices[cam_id], dtype=float))
        centers.append(camera_centers.get(cam_id) if camera_centers else None)

    body_points, body_conf, _reproj = triangulate_skeleton_ransac(
        np.asarray(keypoints, dtype=float),
        np.asarray(projections, dtype=float),
        reprojection_threshold_px=config.triangulation_reproj_threshold_px,
        min_views=config.triangulation_min_views,
    )
    points3d = np.full((17, 3), np.nan, dtype=float)
    joint_conf = np.zeros(17, dtype=float)
    points3d[_BODY_JOINTS] = body_points
    joint_conf[_BODY_JOINTS] = body_conf

    parallax_ok = _joint_parallax_ok(points3d, centers, config.pose_parallax_min_deg)
    descriptor = limb_proportion_descriptor(
        points3d, joint_conf, parallax_ok,
        min_conf=config.pose_min_conf, n_views=len(members),
    )
    torso_ok = torso_anthropometric_ok(
        points3d, joint_conf,
        shoulder_width_m=tuple(config.pose_shoulder_width_m),
        hip_width_m=tuple(config.pose_hip_width_m),
        torso_len_m=tuple(config.pose_torso_len_m),
        torso_tilt_max_deg=config.pose_torso_tilt_max_deg,
        min_conf=config.pose_min_conf,
    )
    return descriptor, torso_ok


def _airborne_2d_proxy(detection: Detection3, config) -> bool:
    """Cheap causal airborne flag (F9a): both confident ankles well above the bbox
    bottom means the feet are off the ground and the z=0 projection lands long —
    inflate the emitted covariance rather than trusting the grazing-angle point.
    (Replaced by the 3D-lift ankle-height flag once P3.5 runs.)"""

    bbox = detection.bbox_xywh_px
    if not bbox or bbox[3] <= 0:
        return False
    conf = detection.keypoint_conf
    pts = detection.keypoints_px
    bottom = float(bbox[1]) + float(bbox[3])
    lift = float(config.airborne_ankle_bbox_frac) * float(bbox[3])
    flags = []
    for i in (15, 16):
        if not (np.isfinite(pts[i]).all() and np.isfinite(conf[i]) and conf[i] >= 0.5):
            return False  # can't tell -> not airborne
        flags.append(float(pts[i][1]) < bottom - lift)
    return all(flags)


def _finalize_ground_cov(cov, members_detections, config) -> np.ndarray | None:
    """Gate + airborne-inflate the emitted ground covariance (F9a)."""

    if not config.emit_ground_cov or cov is None:
        return None
    cov = np.asarray(cov, dtype=float)
    if cov.shape != (2, 2) or not np.isfinite(cov).all():
        return None
    airborne = [_airborne_2d_proxy(det, config) for det in members_detections]
    if airborne and sum(airborne) * 2 > len(airborne):  # majority of views airborne
        cov = cov * float(config.airborne_cov_scale)
    # Wave-5b: when EVERY member view is contested the relative solve weights cancel
    # out (uniform scaling), so the posterior cov alone does not reflect the merged-box
    # ambiguity — inflate it explicitly so P4's measurement-R treats it as uncertain.
    contested = [getattr(det, "contested", False) for det in members_detections]
    if contested and all(contested):
        cov = cov * float(config.contested_sigma_scale) ** 2
    return cov


_L_HIP, _R_HIP = 11, 12  # COCO-17 hip indices


def _triangulated_pelvis_xy(members, dets_per_cam, proj_matrices, config) -> "np.ndarray | None":
    """Vertical ground projection of the triangulated hip midpoint (V2-L3).

    Used for airborne frames where the z=0 foot ray is biased. Requires two-plus
    member views with both hips confident; falls back to None otherwise.
    """

    pixels, projections = [], []
    for cam_id, idx in members.items():
        det = dets_per_cam[cam_id][idx]
        conf = det.keypoint_conf
        if len(conf) <= _R_HIP:
            continue
        if float(conf[_L_HIP]) < config.pose_min_conf or float(conf[_R_HIP]) < config.pose_min_conf:
            continue
        mid = 0.5 * (np.asarray(det.keypoints_px[_L_HIP], float)
                     + np.asarray(det.keypoints_px[_R_HIP], float))
        pixels.append(mid)
        projections.append(proj_matrices[cam_id])
    if len(pixels) < 2:
        return None
    point = triangulate_dlt(np.asarray(pixels, float), np.asarray(projections, float))
    if point is None or not np.isfinite(point).all():
        return None
    # a pelvis must be at plausible body height; reject wild triangulations
    if not (0.3 <= float(point[2]) <= 2.0):
        return None
    return np.asarray(point[:2], dtype=float)


def _build_correspondence(cluster_id, members, dets_per_cam, proj_matrices, config, camera_centers=None):
    detections = {cam_id: dets_per_cam[cam_id][idx] for cam_id, idx in members.items()}
    if len(members) == 1:
        camera_id, index = next(iter(members.items()))
        detection = dets_per_cam[camera_id][index]
        ground_xy = _detection_ground_xy(
            detection, proj_matrices[camera_id], config
        )
        if config.single_cam_height_emit:
            # C5: back-project the emit foot onto its landmark-height plane — the
            # single lone-camera case where z=0 projection overshoots by ~0.94 m
            # mean / 1.3 m p95 at grazing angles (methods-log M5). Emit-only.
            pixel, height = _emit_foot_and_height(detection, config)
            if height > 1e-9:
                corrected = pixel_to_plane_xy(pixel, proj_matrices[camera_id], height)
                if np.isfinite(corrected).all():
                    ground_xy = corrected
        single_cov = None
        if config.emit_ground_cov and np.isfinite(ground_xy).all():
            # Homography-Jacobian model along the lone camera's ray, inflated: a
            # single grazing view is the least-trustworthy ground estimate (F9a).
            single_cov = ground_covariance(
                _foot_pixel(detection, config),
                proj_matrices[camera_id],
                sigma_px=_member_ground_sigma_px(detection, config),
                var_floor_m=float(config.ground_var_floor_m),
            ) * float(config.single_cam_cov_inflation)
            single_cov = _finalize_ground_cov(single_cov, [detection], config)
        return Correspondence(
            cluster_id=cluster_id,
            members=detections,
            ground_xy=ground_xy,
            track_confidence=(
                config.single_camera_confidence
                if detection.local_track_id is not None
                else config.untracked_single_camera_confidence
            ),
            single_camera=True,
            mean_reprojection_error_px=None,
            cycle_consistent=True,
            ground_spread_m=0.0,
            ground_cov=single_cov,
        )
    ground_xy, max_ground_residual, ground_cov = _ground_consensus_members(
        members, dets_per_cam, proj_matrices, config
    )
    if ground_xy is None:
        ground_xy = np.full(2, np.nan)
        max_ground_residual = float("inf")
    elif getattr(config, "airborne_pelvis_emit", False):
        # V2-L3: an airborne player's feet are off the plane, so the z=0 foot solve
        # lands PAST the player along every ray. When a majority of member views
        # flag airborne and two-plus views see confident hips, triangulate the hip
        # midpoint and emit its vertical ground projection instead. EMIT-ONLY: the
        # clustering gate still uses the per-detection legacy foot points.
        airborne_votes = [
            _airborne_2d_proxy(dets_per_cam[cam][idx], config)
            for cam, idx in members.items()
        ]
        if airborne_votes and sum(airborne_votes) * 2 > len(airborne_votes):
            pelvis_xy = _triangulated_pelvis_xy(members, dets_per_cam, proj_matrices, config)
            if pelvis_xy is not None:
                ground_xy = pelvis_xy
    feet, projections, _confs = _gather_member_arrays(members, dets_per_cam, proj_matrices, config)
    point, _max_reprojection = _multiview_reprojection_consistency(
        members, dets_per_cam, proj_matrices, config
    )
    errors = reprojection_errors_for_point(point, feet, projections) if point is not None else np.full(len(feet), np.nan)
    mean_reproj = float(np.nanmean(errors)) if np.isfinite(errors).any() else float("inf")
    geo_term = float(
        np.clip(1.0 - max_ground_residual / config.ground_cluster_gate_m, 0.0, 1.0)
    )
    support = min(len(members), 4) / 4.0
    track_confidence = float(np.clip(0.45 + 0.35 * geo_term + 0.20 * support, 0.0, 1.0))

    pose_descriptor: PoseProportions | None = None
    if config.pose_descriptor_enabled:
        pose_descriptor, torso_ok = _pose_descriptor_for_members(
            members, dets_per_cam, proj_matrices, camera_centers or {}, config
        )
        # Fail-open: only a *confidently implausible* torso (a likely chimera merging
        # two different people) trims confidence; None (unobservable) never penalizes.
        if torso_ok is False:
            track_confidence = max(0.0, track_confidence - config.pose_confidence_penalty)

    return Correspondence(
        cluster_id=cluster_id,
        members=detections,
        ground_xy=ground_xy,
        track_confidence=track_confidence,
        single_camera=False,
        mean_reprojection_error_px=(mean_reproj if np.isfinite(mean_reproj) else None),
        cycle_consistent=bool(
            np.isfinite(max_ground_residual)
            and max_ground_residual <= config.ground_cluster_gate_m
        ),
        ground_spread_m=(max_ground_residual if np.isfinite(max_ground_residual) else None),
        pose_descriptor=pose_descriptor,
        ground_cov=_finalize_ground_cov(ground_cov, list(detections.values()), config),
    )


def _order_clusters(clusters, anchor_id, config):
    """Deterministic per-frame ordering: anchor-anchored clusters first, then by camera priority."""

    def sort_key(members):
        has_anchor = anchor_id in members
        anchor_idx = members.get(anchor_id, 1_000_000)
        min_rank = min(_priority_rank(cam, config) for cam in members)
        min_idx = min(members.values())
        return (0 if has_anchor else 1, anchor_idx, min_rank, min_idx)

    return sorted(clusters, key=sort_key)


def _sorted_pair(cam_a: str, cam_b: str) -> tuple[str, str]:
    return (cam_a, cam_b) if cam_a <= cam_b else (cam_b, cam_a)

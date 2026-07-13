"""P4b post-delivery tracklet stitching without a graph-library dependency."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from pose_estimation.cricket.pose_shape import (
    PoseProportions,
    PostureAggregate,
    descriptor_distance,
    posture_distance_z,
)
from scripts.global_id.config import P4Config


@dataclass(frozen=True)
class Segment:
    seg_id: int
    global_player_id: str
    start_frame: int
    end_frame: int
    first_ground_pos: np.ndarray
    last_ground_pos: np.ndarray
    dominant_role: str
    exit_velocity: np.ndarray
    entry_velocity: np.ndarray | None = None
    pose: PoseProportions | None = None
    posture: PostureAggregate | None = None


@dataclass(frozen=True)
class Edge:
    source_seg_id: int
    target_seg_id: int
    cost: float
    temporal_gap_frames: int
    spatial_gap_m: float


def _role_penalty(role_a: str, role_b: str, config: P4Config) -> float:
    if "unknown" in {role_a, role_b} or role_a == role_b:
        return 0.0
    incompatible = {frozenset(pair) for pair in config.p4b.incompatible_role_pairs}
    return config.p4b.w_role if frozenset((role_a, role_b)) in incompatible else 0.3 * config.p4b.w_role


def velocity_continuity_cost(segment_a: Segment, segment_b: Segment) -> float:
    link = segment_b.first_ground_pos - segment_a.last_ground_pos
    link_norm = float(np.linalg.norm(link))
    velocity_norm = float(np.linalg.norm(segment_a.exit_velocity))
    if link_norm < 0.1 or velocity_norm < 0.01:
        return 0.0
    cosine = float(np.clip((segment_a.exit_velocity / velocity_norm) @ (link / link_norm), -1.0, 1.0))
    return (1.0 - cosine) / 2.0


def _mean_velocity(positions: list[np.ndarray], *, window: int = 5, tail: bool = True) -> np.ndarray:
    """Short-window mean velocity (per frame) at the tail or head of a run.

    Uses the last/first ``window`` positions rather than the raw last-two-frame
    difference, so a single noisy foot projection at a segment boundary does not set
    a wild exit/entry heading (the input to the velocity-continuity stitch cost).
    """

    if len(positions) < 2:
        return np.zeros(2)
    seg = positions[-window:] if tail else positions[:window]
    if len(seg) < 2:
        seg = positions[-2:] if tail else positions[:2]
    diffs = [seg[i + 1] - seg[i] for i in range(len(seg) - 1)]
    return np.mean(np.asarray(diffs, dtype=float), axis=0)


def extract_segments(
    records: Iterable[dict[str, Any]],
    ground_positions: dict[tuple[str, int], np.ndarray],
    pose_by_id: dict[str, PoseProportions | None] | None = None,
    posture_by_id: dict[str, PostureAggregate | None] | None = None,
) -> list[Segment]:
    """Extract maximal contiguous confirmed runs using a separate ground-position table.

    ``pose_by_id`` optionally maps a P4a global id to its accumulated pose-shape
    descriptor, attached to every segment so stitching can require body-shape
    agreement before merging two fragments (ghost-verification, ID-2/WS1d).
    ``posture_by_id`` is the billboard-posture analogue (F12): it matures on the
    facing pairs where the triangulated descriptor cannot.
    """

    pose_by_id = pose_by_id or {}
    posture_by_id = posture_by_id or {}
    per_identity: dict[str, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        frame_index = int(record["frame_index"])
        for player in record.get("players", []):
            player_id = player.get("global_player_id")
            if player_id and player.get("track_state") == "confirmed":
                per_identity[player_id][frame_index].append(player.get("role", "unknown"))

    segments: list[Segment] = []
    for player_id in sorted(per_identity):
        usable_frames = sorted(
            frame for frame in per_identity[player_id]
            if (player_id, frame) in ground_positions
            and np.isfinite(ground_positions[(player_id, frame)]).all()
        )
        start = 0
        while start < len(usable_frames):
            end = start
            while end + 1 < len(usable_frames) and usable_frames[end + 1] == usable_frames[end] + 1:
                end += 1
            run = usable_frames[start : end + 1]
            positions = [np.asarray(ground_positions[(player_id, frame)], dtype=float) for frame in run]
            roles = [role for frame in run for role in per_identity[player_id][frame]]
            dominant_role = Counter(roles).most_common(1)[0][0] if roles else "unknown"
            segments.append(
                Segment(
                    seg_id=len(segments),
                    global_player_id=player_id,
                    start_frame=run[0],
                    end_frame=run[-1],
                    first_ground_pos=positions[0].copy(),
                    last_ground_pos=positions[-1].copy(),
                    dominant_role=dominant_role,
                    exit_velocity=_mean_velocity(positions, tail=True),
                    entry_velocity=_mean_velocity(positions, tail=False),
                    pose=pose_by_id.get(player_id),
                    posture=posture_by_id.get(player_id),
                )
            )
            start = end + 1
    return segments


def build_link_costs(
    segments: Iterable[Segment],
    config: P4Config,
    occupancy_by_id: dict[str, set[tuple[str, int]]] | None = None,
) -> list[Edge]:
    """Feasible stitch edges between temporally ordered fragments.

    ``occupancy_by_id`` (identity -> set of (camera, frame) cells) enables the F6
    occupancy-licensed long bridge: a gap beyond ``temporal_gate_frames`` is still
    admissible up to ``temporal_gate_frames_occupancy`` when the two identities'
    occupancies are fully disjoint (they can never have been two simultaneous
    people) — by default additionally requiring mature pose-shape agreement.
    """

    segments = list(segments)
    edges: list[Edge] = []
    pose_gate = config.p4b.pose_stitch_max_distance
    w_pose = config.p4b.w_pose
    bridge_enabled = config.p4b.occupancy_bridge and occupancy_by_id is not None
    for source in segments:
        for target in segments:
            gap = target.start_frame - source.end_frame
            if gap <= 0:
                continue
            long_bridge = gap > config.p4b.temporal_gate_frames
            if long_bridge:
                if not bridge_enabled or gap > config.p4b.temporal_gate_frames_occupancy:
                    continue
                source_cells = occupancy_by_id.get(source.global_player_id, set())
                target_cells = occupancy_by_id.get(target.global_player_id, set())
                if source_cells & target_cells:
                    continue
            distance = float(np.linalg.norm(target.first_ground_pos - source.last_ground_pos))
            maximum = (
                config.kinematic_v_max_mps * gap / config.frame_rate_fps * config.p4b.kinematic_slack
            )
            if distance > maximum:
                continue
            pose_term = 0.0
            pose_distance = None
            if pose_gate > 0.0 or w_pose > 0.0:
                pose_distance = descriptor_distance(
                    source.pose, target.pose,
                    min_shared=config.p4b.pose_min_shared_segments,
                )
                if pose_distance is not None:
                    # Hard gate: two fragments of clearly different build are not the
                    # same person, regardless of how kinematically reachable they are.
                    if pose_gate > 0.0 and pose_distance > pose_gate:
                        continue
                    pose_term = w_pose * pose_distance
            posture_term = 0.0
            posture_gate = config.p4b.posture_stitch_max_z
            if posture_gate > 0.0 or config.p4b.w_posture > 0.0:
                posture_z = posture_distance_z(source.posture, target.posture)
                if posture_z is not None:
                    # F12: a stitch between clearly different builds is forbidden on
                    # the billboard stature/shape too — this key matures on the
                    # facing pairs where the triangulated descriptor stays immature.
                    if posture_gate > 0.0 and posture_z[0] > posture_gate:
                        continue
                    posture_term = config.p4b.w_posture * posture_z[0]
            if long_bridge and config.p4b.occupancy_bridge_require_pose:
                # A long bridge on kinematics alone is how chimeras happen; demand
                # positive body-shape evidence (abstaining descriptors don't count).
                if pose_distance is None or pose_gate <= 0.0 or pose_distance > pose_gate:
                    continue
            if config.p4b.normalized_costs:
                # G7: commensurate unit-free terms — each divided by its own gate,
                # so a plausible stitch anywhere inside the gates can actually win
                # against the new-trajectory dummy (see P4BConfig.normalized_costs).
                temporal_term = config.p4b.w_temporal * (
                    gap / max(config.p4b.temporal_gate_frames, 1)
                )
                spatial_term = config.p4b.w_spatial * (distance / max(maximum, 1e-6))
            else:
                temporal_term = config.p4b.w_temporal * gap
                spatial_term = config.p4b.w_spatial * distance
            cost = (
                temporal_term
                + spatial_term
                + _role_penalty(source.dominant_role, target.dominant_role, config)
                + config.p4b.velocity_continuity_weight * velocity_continuity_cost(source, target)
                + pose_term
                + posture_term
            )
            edges.append(Edge(source.seg_id, target.seg_id, cost, gap, distance))
    return edges


def solve_flow(segments: Iterable[Segment], edges: Iterable[Edge], config: P4Config) -> dict[int, int]:
    """Solve the path-cover assignment with one private no-link dummy per tail."""

    segments = sorted(segments, key=lambda item: item.seg_id)
    if not segments:
        return {}
    ids = [segment.seg_id for segment in segments]
    index = {segment_id: offset for offset, segment_id in enumerate(ids)}
    count = len(ids)
    new_trajectory_cost = config.p4b.w_spatial * config.p4b.new_traj_cost_factor
    large = max(1e9, new_trajectory_cost * 1e6)
    cost = np.full((count, 2 * count), large, dtype=float)
    for row in range(count):
        cost[row, count + row] = new_trajectory_cost
    edge_lookup: dict[tuple[int, int], Edge] = {}
    for edge in edges:
        if edge.source_seg_id in index and edge.target_seg_id in index:
            row, column = index[edge.source_seg_id], index[edge.target_seg_id]
            if edge.cost < cost[row, column]:
                cost[row, column] = edge.cost
                edge_lookup[(row, column)] = edge
    rows, columns = linear_sum_assignment(cost)
    links = {}
    for row, column in zip(rows, columns):
        edge = edge_lookup.get((int(row), int(column)))
        if edge is not None and edge.cost < new_trajectory_cost:
            links[edge.source_seg_id] = edge.target_seg_id
    return links


def remap_ids(
    records: Iterable[dict[str, Any]],
    segments: Iterable[Segment],
    links: dict[int, int],
) -> list[dict[str, Any]]:
    """Merge linked identities to the earliest segment ID and patch contract records."""

    records = list(records)
    segments = list(segments)
    by_segment = {segment.seg_id: segment for segment in segments}
    identity_parent = {segment.global_player_id: segment.global_player_id for segment in segments}
    occupancy: dict[str, set[tuple[str, int]]] = {
        player_id: set() for player_id in identity_parent
    }
    for record in records:
        frame_index = int(record["frame_index"])
        camera_id = str(record.get("camera_id", "unknown"))
        for player in record.get("players", []):
            player_id = player.get("global_player_id")
            if player_id in occupancy:
                occupancy[player_id].add((camera_id, frame_index))

    def find(player_id: str) -> str:
        while identity_parent[player_id] != player_id:
            identity_parent[player_id] = identity_parent[identity_parent[player_id]]
            player_id = identity_parent[player_id]
        return player_id

    identity_first = {}
    for segment in segments:
        identity_first[segment.global_player_id] = min(
            identity_first.get(segment.global_player_id, segment.start_frame), segment.start_frame
        )
    for source_id, target_id in links.items():
        source, target = by_segment[source_id], by_segment[target_id]
        left, right = find(source.global_player_id), find(target.global_player_id)
        if left == right:
            continue
        # A global merge is forbidden when the two histories ever occupy the
        # same camera frame. This is the invariant the old whole-history remap
        # violated, producing two visibly different people with one ID.
        if occupancy[left] & occupancy[right]:
            continue
        winner, loser = sorted((left, right), key=lambda pid: (identity_first[pid], pid))
        identity_parent[loser] = winner
        identity_first[winner] = min(identity_first[winner], identity_first[loser])
        occupancy[winner].update(occupancy[loser])

    remap = {player_id: find(player_id) for player_id in identity_parent}
    first_segment = {
        player_id: min(
            (segment.start_frame for segment in segments if segment.global_player_id == player_id),
            default=None,
        )
        for player_id in remap
    }
    report = [
        {"merged_id": old_id, "into_id": new_id, "at_frame": first_segment[old_id]}
        for old_id, new_id in sorted(remap.items(), key=lambda item: (first_segment[item[0]], item[0]))
        if old_id != new_id
    ]
    for record in records:
        for player in record.get("players", []):
            player_id = player.get("global_player_id")
            if player_id in remap:
                player["global_player_id"] = remap[player_id]
    return report

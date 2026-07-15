"""Global-track state and lifecycle for P4a."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from identity.p5_global_id.ground_kalman import SingerGroundKalman
from identity.common.pose_shape import PoseProportions, PostureAggregate, merge_descriptor

TENTATIVE = "tentative"
CONFIRMED = "confirmed"
LOST = "lost"
DELETED = "deleted"


@dataclass
class GlobalTrack:
    global_player_id: str | None
    state: str
    kalman: SingerGroundKalman
    first_frame: int
    last_frame: int
    prediction_frame: int
    first_ground_pos: np.ndarray
    last_ground_pos: np.ndarray
    dominant_role: str = "unknown"
    role_latch_count: int = 0
    role_candidate: str = field(default="unknown", repr=False)
    hits: int = 1
    frames_since_update: int = 0
    last_bbox_xywh_px: list[float] | None = None
    pose_proportions: PoseProportions | None = None
    pose_update_count: int = 0
    # Binding-level billboard posture (F6b): whole-delivery pooled aggregate from
    # P3, so the latest observation's aggregate simply replaces the stored one.
    posture: PostureAggregate | None = None
    single_camera: bool = False
    # Confirmed neighbours within the density radius when this track was last
    # marked missed while CONFIRMED — a crowded loss reads as occlusion and earns
    # a longer lost window (wip/to_do.md: density-scaled adaptive window).
    density_at_loss: int = 0
    local_track_ids_by_cam: dict[str, str] = field(default_factory=dict)
    local_track_id_history: set[tuple[str, str]] = field(default_factory=set, repr=False)

    def __post_init__(self) -> None:
        self._register_local_ids(self.local_track_ids_by_cam)

    def _register_local_ids(self, values: dict[str, str]) -> None:
        for camera_id, local_track_id in values.items():
            if local_track_id:
                self.local_track_id_history.add((camera_id, local_track_id))
                self.local_track_ids_by_cam[camera_id] = local_track_id

    def matches_local_ids(self, values: dict[str, str]) -> bool:
        return any((camera_id, local_track_id) in self.local_track_id_history
                   for camera_id, local_track_id in values.items())

    def register_local_ids(self, values: dict[str, str]) -> None:
        self._register_local_ids(values)

    def predict_to(self, frame_index: int, *, max_pos_var: float) -> None:
        if frame_index < self.prediction_frame:
            raise ValueError("track prediction cannot move backwards in time")
        steps = frame_index - self.prediction_frame
        for _ in range(steps):
            self.kalman.predict()
            self.kalman.cap_covariance(max_pos_var)
        self.frames_since_update += steps
        self.prediction_frame = frame_index

    def mark_missed(self) -> None:
        if self.state == CONFIRMED:
            self.state = LOST

    def apply_hit(
        self,
        ground_xy: np.ndarray,
        bbox_xywh_px: list[float] | None,
        pose_descriptor: PoseProportions | None,
        frame_index: int,
        *,
        single_camera: bool,
        local_track_ids_by_cam: dict[str, str],
        pose_ema_rate: float = 0.15,
        posture: PostureAggregate | None = None,
        measurement_R: np.ndarray | None = None,
    ) -> None:
        self.kalman.update(np.asarray(ground_xy, dtype=float), R=measurement_R)
        self.last_ground_pos = np.asarray(ground_xy, dtype=float).copy()
        self.last_frame = frame_index
        self.prediction_frame = frame_index
        self.frames_since_update = 0
        self.hits += 1
        self.last_bbox_xywh_px = bbox_xywh_px
        if pose_descriptor is not None and pose_descriptor.is_defined():
            self.pose_proportions = merge_descriptor(
                self.pose_proportions, pose_descriptor, rate=pose_ema_rate
            )
            self.pose_update_count += 1
        if posture is not None and posture.is_defined():
            self.posture = posture
        self.single_camera = single_camera
        self._register_local_ids(local_track_ids_by_cam)
        if self.state in {LOST, DELETED}:
            self.state = CONFIRMED

    def apply_identity_only_hit(
        self,
        bbox_xywh_px: list[float] | None,
        frame_index: int,
        *,
        local_track_ids_by_cam: dict[str, str],
    ) -> None:
        """Record exact P2 tracklet continuity without a synthetic position update."""

        self.last_frame = frame_index
        self.prediction_frame = frame_index
        self.frames_since_update = 0
        self.hits += 1
        self.last_bbox_xywh_px = bbox_xywh_px
        self.single_camera = True
        self._register_local_ids(local_track_ids_by_cam)
        if self.state == LOST:
            self.state = CONFIRMED

    def maybe_confirm(self, confirm_hits: int) -> bool:
        if self.state == TENTATIVE and self.hits >= confirm_hits:
            self.state = CONFIRMED
            return True
        return False

    def should_delete(
        self,
        *,
        confirm_hits: int,
        lost_window_frames: int,
        bowler_lost_window_frames: int,
        adaptive_lost_window: bool = False,
        lost_window_max_frames: int = 90,
        density_bonus_frames: int = 0,
    ) -> bool:
        if self.state == TENTATIVE:
            return self.frames_since_update >= confirm_hits
        base = bowler_lost_window_frames if self.dominant_role == "bowler" else lost_window_frames
        window = base
        if adaptive_lost_window:
            # A well-established track (many confirmed hits) has earned a longer
            # occlusion tolerance: grow the window one frame per hit-beyond-confirm,
            # capped, so a briefly-hidden regular is re-acquired instead of re-born.
            window = min(int(lost_window_max_frames), base + max(0, self.hits - confirm_hits))
            window = max(window, base)
            if density_bonus_frames > 0 and self.density_at_loss > 0:
                # Lost inside a scrum/pack: almost certainly occlusion, not exit —
                # lengthen the window per crowding neighbour (same cap).
                window = min(
                    int(lost_window_max_frames),
                    window + density_bonus_frames * self.density_at_loss,
                )
        return self.state == LOST and self.frames_since_update > window

    def velocity_toward_crease(self, crease_y: float = 0.0) -> bool:
        """Dormant P5 hook: whether current velocity points toward the crease."""

        vy = float(self.kalman.velocity_xy[1])
        pos_y = float(self.kalman.pos_world_xy[1])
        return (pos_y > crease_y and vy < -0.1) or (pos_y < crease_y and vy > 0.1)

"""Global-track state and lifecycle for P4a."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pose_estimation.cricket.ground_kalman import SingerGroundKalman

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
    last_pose_2d: dict | None = None
    single_camera: bool = False
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
        pose_2d: dict | None,
        frame_index: int,
        *,
        single_camera: bool,
        local_track_ids_by_cam: dict[str, str],
    ) -> None:
        self.kalman.update(np.asarray(ground_xy, dtype=float))
        self.last_ground_pos = np.asarray(ground_xy, dtype=float).copy()
        self.last_frame = frame_index
        self.prediction_frame = frame_index
        self.frames_since_update = 0
        self.hits += 1
        self.last_bbox_xywh_px = bbox_xywh_px
        self.last_pose_2d = pose_2d
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
    ) -> bool:
        if self.state == TENTATIVE:
            return self.frames_since_update >= confirm_hits
        window = bowler_lost_window_frames if self.dominant_role == "bowler" else lost_window_frames
        return self.state == LOST and self.frames_since_update > window

    def velocity_toward_crease(self, crease_y: float = 0.0) -> bool:
        """Dormant P5 hook: whether current velocity points toward the crease."""

        vy = float(self.kalman.velocity_xy[1])
        pos_y = float(self.kalman.pos_world_xy[1])
        return (pos_y > crease_y and vy < -0.1) or (pos_y < crease_y and vy > 0.1)

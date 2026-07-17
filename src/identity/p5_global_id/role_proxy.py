"""Online role proxy (F5): feed role-aware Singer dynamics DURING tracking.

P5 assigns roles offline, after P4 - too late for the Kalman, which therefore
tracks every player with the generic ``unknown`` manoeuvre model. This proxy
watches the confirmed tracks' ground trajectories causally and, once the
evidence is unambiguous, calls :meth:`TrackManager.propose_role` (which latches
for ``role_latch_frames`` before switching the filter), so a bowler gets agile
dynamics for the delivery and an umpire near-static ones.

Deliberately conservative - it proposes only the three role families whose
dynamics differ most and whose online signatures are near-unambiguous:

* **bowler** - the first track with a sustained fast run along the pitch axis;
  the run's sign also fixes the bowling direction for the other rules.
* **umpire / wicketkeeper** - near-static, on the pitch line, behind the
  bowling-end / striker's-end stumps (only after the direction is known).

Everything else keeps ``unknown``; batsmen/fielders' dynamics are close enough
to the default that a wrong guess would cost more than the right one gains.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from identity.p5_global_id.track_manager import TrackManager
from identity.p6_roles.assigner import STUMPS_FROM_CENTRE_M


class OnlineRoleProxy:
    def __init__(
        self,
        pitch_axis: np.ndarray | None,
        *,
        frame_rate_fps: float,
        min_track_frames: int = 50,
        bowler_min_speed_mps: float = 3.5,
        static_speed_max_mps: float = 0.6,
        pitch_halfwidth_m: float = 2.5,
        eval_every_frames: int = 10,
        speed_window_frames: int = 25,
    ) -> None:
        self.axis = None
        if pitch_axis is not None:
            axis = np.asarray(pitch_axis, dtype=float)
            norm = float(np.linalg.norm(axis))
            if norm > 1e-9:
                self.axis = axis / norm
        self.frame_rate_fps = frame_rate_fps
        self.min_track_frames = min_track_frames
        self.bowler_min_speed_mps = bowler_min_speed_mps
        self.static_speed_max_mps = static_speed_max_mps
        self.pitch_halfwidth_m = pitch_halfwidth_m
        self.eval_every_frames = eval_every_frames
        self.speed_window_frames = speed_window_frames
        # +1/-1 once the bowler's run fixes which way "toward the striker" is.
        self.direction_sign: float = 0.0
        self.bowler_id: str | None = None
        self._history: dict[str, deque[tuple[int, np.ndarray]]] = defaultdict(
            lambda: deque(maxlen=600)
        )

    def observe(self, manager: TrackManager, frame_index: int) -> None:
        """Accumulate this frame's confirmed positions and propose latched roles."""

        if self.axis is None:
            return
        for track in manager.tracks:
            if (
                track.global_player_id is not None
                and track.frames_since_update == 0
                and np.isfinite(track.last_ground_pos).all()
            ):
                self._history[track.global_player_id].append(
                    (frame_index, track.last_ground_pos.copy())
                )
        if frame_index % self.eval_every_frames != 0:
            return
        for player_id, series in self._history.items():
            if len(series) < self.min_track_frames:
                continue
            role = self._classify(player_id, series)
            if role is not None:
                manager.propose_role(player_id, role, frame_index)

    def _windowed_axis_speed(self, series: deque[tuple[int, np.ndarray]]) -> float:
        """Signed axis-projected speed over the most recent ~window span."""

        recent = [item for item in series if item[0] >= series[-1][0] - self.speed_window_frames]
        if len(recent) < 2:
            return 0.0
        (frame_a, point_a), (frame_b, point_b) = recent[0], recent[-1]
        gap = frame_b - frame_a
        if gap < self.speed_window_frames // 2:
            return 0.0
        return float((point_b - point_a) @ self.axis) * self.frame_rate_fps / gap

    def _recent_speed(self, series: deque[tuple[int, np.ndarray]]) -> float:
        recent = [item for item in series if item[0] >= series[-1][0] - self.speed_window_frames]
        if len(recent) < 2:
            return 0.0
        (frame_a, point_a), (frame_b, point_b) = recent[0], recent[-1]
        gap = frame_b - frame_a
        if gap <= 0:
            return 0.0
        return float(np.linalg.norm(point_b - point_a)) * self.frame_rate_fps / gap

    def _classify(self, player_id: str, series: deque[tuple[int, np.ndarray]]) -> str | None:
        axis_speed = self._windowed_axis_speed(series)
        if self.bowler_id is None and abs(axis_speed) >= self.bowler_min_speed_mps:
            self.bowler_id = player_id
            self.direction_sign = float(np.sign(axis_speed))
            return "bowler"
        if player_id == self.bowler_id:
            return "bowler"
        if self.direction_sign == 0.0:
            return None  # umpire/keeper ends are ambiguous until the run fixes them
        if self._recent_speed(series) > self.static_speed_max_mps:
            return None
        points = np.asarray([point for _, point in series])
        # "toward the striker's end" is +along by construction
        along = float(np.median(points @ (self.axis * self.direction_sign)))
        lateral = np.array([-self.axis[1], self.axis[0]])
        across = float(np.median(points @ lateral))
        if abs(across) > self.pitch_halfwidth_m:
            return None
        if along > STUMPS_FROM_CENTRE_M + 0.3:
            return "wicketkeeper"
        if along < -(STUMPS_FROM_CENTRE_M + 0.3):
            return "umpire"
        return None

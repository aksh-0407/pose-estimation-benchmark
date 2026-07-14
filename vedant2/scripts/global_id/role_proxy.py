"""Online role proxy (F5): feed role-aware Singer dynamics DURING tracking.

P5 assigns roles offline, after P4 — too late for the Kalman, which therefore
tracks every player with the generic ``unknown`` manoeuvre model. This proxy
watches the confirmed tracks' ground trajectories causally and, once the
evidence is unambiguous, calls :meth:`TrackManager.propose_role` (which latches
for ``role_latch_frames`` before switching the filter), so a bowler gets agile
dynamics for the delivery and an umpire near-static ones.

Deliberately conservative — it proposes only the role families whose
dynamics differ most and whose online signatures are near-unambiguous:

* **bowler** — the first track with a sustained fast run along the pitch axis;
  the run's sign also fixes the bowling direction for the other rules.
  Self-decays if the bowler stops (A.1); relaxed fallback for spin bowlers (A.1).
* **striker / non_striker** — near the crease positions, velocity-delta gated
  after release_proxy_frame (A.3 / A.4).
* **umpire / wicketkeeper** — near-static, on the pitch line, behind the
  bowling-end / striker's-end stumps (only after the direction is known).

Everything else keeps ``unknown``; fielders' dynamics are close enough
to the default that a wrong guess would cost more than the right one gains.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from scripts.global_id.track_manager import TrackManager
from scripts.roles.assigner import STUMPS_FROM_CENTRE_M


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
        bowler_decay_speed_mps: float = 0.6,
        bowler_decay_evals: int = 15,
        bowler_min_speed_relaxed_mps: float = 1.5,
        bowler_relaxed_fallback_frames: int = 300,
        release_proxy_min_speed_mps: float = 2.0,
        release_proxy_decel_fraction: float = 0.35,
        release_proxy_timeout_frames: int = 150,
        striker_speed_delta_mps: float = 1.5,
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

        # A.1: self-decay parameters
        self.bowler_decay_speed_mps = bowler_decay_speed_mps
        self.bowler_decay_evals = bowler_decay_evals
        self._bowler_static_evals: int = 0

        # A.1: relaxed-threshold fallback for slow run-ups
        self.bowler_min_speed_relaxed_mps = bowler_min_speed_relaxed_mps
        self.bowler_relaxed_fallback_frames = bowler_relaxed_fallback_frames
        self._first_eval_frame: int | None = None

        # A.2: release-proxy detection
        self.release_proxy_min_speed_mps = release_proxy_min_speed_mps
        self.release_proxy_decel_fraction = release_proxy_decel_fraction
        self.release_proxy_timeout_frames = release_proxy_timeout_frames
        self.release_proxy_frame: int | None = None
        self.release_proxy_timed_out: bool = False
        self._bowler_lock_frame: int | None = None
        self._bowler_peak_speed: float = 0.0
        self._bowler_release_history: deque[tuple[int, np.ndarray]] = deque(maxlen=200)

        # A.3 / A.4: striker / non-striker
        self.striker_speed_delta_mps = striker_speed_delta_mps
        self.striker_id: str | None = None
        self.non_striker_id: str | None = None
        self._pre_lock_speed: dict[str, float] = {}

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
                if track.global_player_id == self.bowler_id:
                    self._track_release_proxy(
                        track.global_player_id, track.last_ground_pos, frame_index
                    )
        if frame_index % self.eval_every_frames != 0:
            return
        for player_id, series in self._history.items():
            if len(series) < self.min_track_frames:
                continue
            role = self._classify(player_id, series, frame_index=frame_index)
            if role is not None:
                if role == "bowler" and self.bowler_id == player_id and self._bowler_lock_frame is None:
                    self._bowler_lock_frame = frame_index
                    # Snapshot pre-lock speed baselines for striker delta gating
                    for other_id, other_series in self._history.items():
                        if other_id != player_id:
                            self._pre_lock_speed[other_id] = self._recent_speed(other_series)
                manager.propose_role(player_id, role, frame_index)

    def _track_release_proxy(self, player_id: str, position: np.ndarray, frame_index: int) -> None:
        """A.2: peak-then-decelerate signature on the bowler's own signed speed."""

        if player_id != self.bowler_id or self.release_proxy_frame is not None:
            return
        self._bowler_release_history.append((frame_index, np.asarray(position, dtype=float)))
        speed = self._windowed_axis_speed(self._bowler_release_history)
        if abs(speed) > self._bowler_peak_speed:
            self._bowler_peak_speed = abs(speed)
        elif (
            self._bowler_peak_speed >= self.release_proxy_min_speed_mps
            and abs(speed) <= self._bowler_peak_speed * (1.0 - self.release_proxy_decel_fraction)
        ):
            self.release_proxy_frame = frame_index
            return
        if (
            self._bowler_lock_frame is not None
            and frame_index - self._bowler_lock_frame > self.release_proxy_timeout_frames
        ):
            self.release_proxy_timed_out = True

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

    def _classify(
        self, player_id: str, series: deque[tuple[int, np.ndarray]], frame_index: int,
    ) -> str | None:
        # Track first eval frame for the relaxed-threshold fallback timer
        if self._first_eval_frame is None:
            self._first_eval_frame = frame_index

        axis_speed = self._windowed_axis_speed(series)

        # A.1: hard-gate bowler claim
        if self.bowler_id is None and abs(axis_speed) >= self.bowler_min_speed_mps:
            self.bowler_id = player_id
            self.direction_sign = float(np.sign(axis_speed))
            return "bowler"

        # A.1: relaxed-threshold fallback for slow run-ups (spin bowlers)
        if (
            self.bowler_id is None
            and frame_index - self._first_eval_frame >= self.bowler_relaxed_fallback_frames
            and abs(axis_speed) >= self.bowler_min_speed_relaxed_mps
        ):
            self.bowler_id = player_id
            self.direction_sign = float(np.sign(axis_speed))
            return "bowler"

        # A.1: self-decay — bowler who stops moving decays back to fielder
        if player_id == self.bowler_id:
            if abs(axis_speed) <= self.bowler_decay_speed_mps:
                self._bowler_static_evals += 1
            else:
                self._bowler_static_evals = 0
            if self._bowler_static_evals >= self.bowler_decay_evals:
                self.bowler_id = None
                self._bowler_static_evals = 0
                return "fielder"
            return "bowler"

        if self.direction_sign == 0.0:
            return None  # umpire/keeper ends are ambiguous until the run fixes them

        # Compute position on pitch — needed for all remaining classification
        points = np.asarray([point for _, point in series])
        # "toward the striker's end" is +along by construction
        along = float(np.median(points @ (self.axis * self.direction_sign)))
        lateral = np.array([-self.axis[1], self.axis[0]])
        across = float(np.median(points @ lateral))
        if abs(across) > self.pitch_halfwidth_m:
            return None

        # Static speed check only applies to umpire/keeper territory (not batters)
        if abs(along) > STUMPS_FROM_CENTRE_M and self._recent_speed(series) > self.static_speed_max_mps:
            return None

        if along > STUMPS_FROM_CENTRE_M + 0.3:
            return "wicketkeeper"
        if along < -(STUMPS_FROM_CENTRE_M + 0.3):
            return "umpire"

        # A.3: non-striker — near bowling-end crease
        if along < 0 and self.non_striker_id in (None, player_id):
            self.non_striker_id = player_id
            return "non_striker"

        # A.3 / A.4: striker — near striker's-end crease, velocity-delta gated
        if along >= 0 and self.striker_id in (None, player_id):
            release_window_open = self.release_proxy_frame is not None or self.release_proxy_timed_out
            if not release_window_open:
                return None
            baseline = self._pre_lock_speed.get(player_id, 0.0)
            delta = self._recent_speed(series) - baseline
            if delta >= self.striker_speed_delta_mps or self.release_proxy_timed_out:
                self.striker_id = player_id
                return "striker"

        return None

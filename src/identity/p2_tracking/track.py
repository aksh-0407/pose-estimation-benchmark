"""Track lifecycle state machine and pose gallery."""

from __future__ import annotations

from collections import deque

import numpy as np

from identity.p2_tracking.config import TrackingConfig
from identity.p2_tracking.kalman import KalmanBoxTracker
from identity.p2_tracking.pose_vector import PoseVector, masked_weighted_cosine

TENTATIVE = "tentative"
CONFIRMED = "confirmed"
DORMANT = "dormant"
DELETED = "deleted"


class Track:
    def __init__(
        self,
        id_num: int,
        camera_id: str,
        bbox_xywh: list[float],
        pose: PoseVector,
        ground_xy: np.ndarray | None,
        is_lowconf: bool,
        config: TrackingConfig,
        frame_index: int,
    ) -> None:
        self._id_num = id_num
        self._camera_id = camera_id
        self._config = config
        self.is_lowconf = is_lowconf
        self.state = TENTATIVE

        self.kalman = KalmanBoxTracker(bbox_xywh)
        self.last_ground_xy = ground_xy.copy() if ground_xy is not None else None
        self._gallery: deque[PoseVector] = deque(maxlen=config.pose_gallery_size)
        # Monotonic id per gallery member, kept in lockstep with `_gallery` (same
        # maxlen -> evicts the oldest in parallel). Lets the medoid cache key pairwise
        # cosines stably across ring-buffer eviction without unsafe id() reuse.
        self._gallery_seq: deque[int] = deque(maxlen=config.pose_gallery_size)
        self._next_seq = 0
        # (min_seq, max_seq) -> masked_weighted_cosine, reused across frames so the
        # medoid recomputes only the newly-added member's row (O(K) not O(K^2)).
        self._pair_cost: dict[tuple[int, int], float] = {}
        self._pair_cost_min_seq = 0
        if pose.defined:
            self._gallery.append(pose)
            self._gallery_seq.append(self._next_seq)
            self._next_seq += 1
        self._gallery_version = 1 if pose.defined else 0
        self._gallery_repr_cache: PoseVector | None = None
        self._gallery_repr_version = -1

        self.hits = 1
        self.highconf_hits = 0
        self.frames_since_update = 0
        self._spawn_frame = frame_index
        self._last_frame = frame_index

        # OC-SORT: observation-centre history + last real observation bbox. Feeds OCM
        # (velocity direction) and OCR/ORU (last-observation matching + virtual re-update).
        # Populated for every tracker; only READ when config.tracker == "ocsort", so the
        # bytetrack path stays byte-identical.
        self._obs_history: deque[tuple[int, float, float]] = deque(maxlen=max(config.ocm_delta_t + 2, 5))
        self._obs_history.append((frame_index, bbox_xywh[0] + bbox_xywh[2] / 2.0,
                                  bbox_xywh[1] + bbox_xywh[3] / 2.0))
        self._last_obs_bbox: list[float] = [float(v) for v in bbox_xywh]

        self.max_cov_trace = self.kalman.position_cov_trace()
        self.gap_count = 0
        self.max_gap_frames = 0
        self._current_gap = 0

        # Every player dict this track has matched, in frame order, for retroactive ID back-fill.
        self.assigned_players: list[dict] = []
        self._stamped_upto = 0

    @property
    def local_track_id(self) -> str:
        return f"{self._camera_id}_trk_{self._id_num:04d}"

    def record_player(self, player: dict) -> None:
        """Remember a matched player dict so its `local_track_id` can be filled on confirmation."""
        self.assigned_players.append(player)

    def flush_id(self) -> None:
        """Stamp `local_track_id` onto any not-yet-stamped matched players (no-op until CONFIRMED).

        Back-fills the tentative frames retroactively the moment the track promotes (spec §6).
        Idempotent: only players past `_stamped_upto` are written.
        """
        if self.state != CONFIRMED:
            return
        tid = self.local_track_id
        for player in self.assigned_players[self._stamped_upto:]:
            player["local_track_id"] = tid
        self._stamped_upto = len(self.assigned_players)

    def register_hit(
        self,
        bbox_xywh,
        pose: PoseVector,
        confidence: float,
        frame_index: int,
        ground_xy: np.ndarray | None = None,
    ) -> None:
        # ORU: on recovery after a gap, re-derive the KF from the virtual trajectory
        # between the last real observation and this one (OC-SORT only; else plain update).
        if (self._config.tracker == "ocsort" and self._config.oru_enabled
                and self._current_gap > 0 and self._last_obs_bbox is not None):
            self.kalman.reupdate_virtual(self._last_obs_bbox, bbox_xywh, self._current_gap)
        else:
            self.kalman.update(bbox_xywh)
        self._obs_history.append((frame_index, bbox_xywh[0] + bbox_xywh[2] / 2.0,
                                  bbox_xywh[1] + bbox_xywh[3] / 2.0))
        self._last_obs_bbox = [float(v) for v in bbox_xywh]
        if ground_xy is not None:
            self.last_ground_xy = ground_xy.copy()
        if pose.defined:
            self._gallery.append(pose)
            self._gallery_seq.append(self._next_seq)
            self._next_seq += 1
            self._gallery_version += 1
        self.hits += 1
        if confidence > self._config.stage1_confidence_threshold:
            self.highconf_hits += 1
        if self._current_gap > 0:
            self.gap_count += 1
            self.max_gap_frames = max(self.max_gap_frames, self._current_gap)
            self._current_gap = 0
        self.frames_since_update = 0
        self._last_frame = frame_index
        self.max_cov_trace = max(self.max_cov_trace, self.kalman.position_cov_trace())
        if self.state == DORMANT:
            self.state = CONFIRMED

    def mark_missed(self, frame_index: int) -> None:
        # NB: the tracker calls kalman.predict() once per frame for every track at the top of
        # update(); mark_missed must NOT predict again (that would double-advance the state).
        self.kalman.inflate_process_noise(1.5)
        self.frames_since_update += 1
        self._current_gap += 1
        self.max_cov_trace = max(self.max_cov_trace, self.kalman.position_cov_trace())
        if self.state == CONFIRMED:
            self.state = DORMANT

    def maybe_confirm(self, *, ignore_window: bool = False) -> bool:
        """Promote TENTATIVE→CONFIRMED when the confirmation rule is met.

        `ignore_window=True` is used at end-of-stream (spec §6): the window is closed early, so a
        track already meeting the hit count is promoted even if 5 frames have not elapsed.
        """
        if self.state != TENTATIVE:
            return False
        within_window = ignore_window or (
            (self._last_frame - self._spawn_frame) < self._config.tentative_confirm_window
        )
        enough_hits = self.hits >= self._config.tentative_confirm_hits
        highconf_ok = (not self.is_lowconf) or self.highconf_hits >= 1
        if within_window and enough_hits and highconf_ok:
            self.state = CONFIRMED
            return True
        return False

    def tentative_expired(self, frame_index: int) -> bool:
        """A still-TENTATIVE track whose confirmation window has fully elapsed (spec §5/§6).

        Catches both the silent-rejection case and a low-conf tentative that keeps matching
        low-conf detections but never lands a >0.5 hit within the window.
        """
        return (
            self.state == TENTATIVE
            and (frame_index - self._spawn_frame) >= self._config.tentative_confirm_window
        )

    def should_delete(self) -> bool:
        if self.kalman.position_cov_trace() > self._config.kalman_cov_trace_max:
            return True
        if self.state == DORMANT and self.frames_since_update > self._config.dormant_max_frames:
            return True
        return False

    def reachability_radius(self) -> float:
        return (
            self._config.v_max_px_per_frame * self.frames_since_update
            + self._config.gate_bbox_factor * self.kalman.bbox_height()
        )

    def ground_reachability_radius(self) -> float:
        elapsed_frames = max(1, self.frames_since_update + 1)
        return (
            self._config.ground_gate_base_m
            + (self._config.ground_vmax_mps / self._config.frame_rate_fps) * elapsed_frames
        )

    def observation_velocity(self, delta_t: int) -> np.ndarray | None:
        """OC-SORT OCM: unit velocity direction between the observation ~delta_t frames
        back and the latest observation (raw observations, NOT the KF-smoothed velocity
        OC-SORT deliberately avoids). None if <2 observations or degenerate."""
        if len(self._obs_history) < 2:
            return None
        latest = self._obs_history[-1]
        past = self._obs_history[max(0, len(self._obs_history) - 1 - int(delta_t))]
        v = np.array([latest[1] - past[1], latest[2] - past[2]], dtype=float)
        norm = float(np.linalg.norm(v))
        return v / norm if norm > 1e-6 else None

    def last_obs_center(self) -> np.ndarray:
        _, cx, cy = self._obs_history[-1]
        return np.array([cx, cy], dtype=float)

    def last_obs_bbox(self) -> list[float]:
        """OC-SORT OCR reference: the last REAL observation bbox (not the KF prediction)."""
        if self._last_obs_bbox is not None:
            return self._last_obs_bbox
        return list(self.kalman.predicted_bbox())

    def gallery_repr(self) -> PoseVector | None:
        if self._gallery_repr_version == self._gallery_version:
            return self._gallery_repr_cache
        members = [v for v in self._gallery if v.defined]
        if not members:
            self._gallery_repr_cache = None
            self._gallery_repr_version = self._gallery_version
            return None
        if len(members) == 1 or self._config.gallery_repr != "medoid":
            self._gallery_repr_cache = members[0]
            self._gallery_repr_version = self._gallery_version
            return members[0]
        if self._config.pose_medoid_incremental:
            self._gallery_repr_cache = self._medoid_incremental(members)
        else:
            self._gallery_repr_cache = self._medoid_full(members)
        self._gallery_repr_version = self._gallery_version
        return self._gallery_repr_cache

    def _medoid_full(self, members: list[PoseVector]) -> PoseVector:
        """Legacy O(K^2) medoid: recompute every pairwise cosine each call."""
        best_idx, best_cost = 0, float("inf")
        for i, vi in enumerate(members):
            total = sum(
                masked_weighted_cosine(vi, vj, min_shared_keypoints=self._config.min_shared_keypoints)
                for j, vj in enumerate(members)
                if j != i
            )
            if total < best_cost:
                best_idx, best_cost = i, total
        return members[best_idx]

    def _medoid_incremental(self, members: list[PoseVector]) -> PoseVector:
        """O(K) medoid: reuse the cached pairwise cosines, computing only the pairs
        involving the newly-appended member. Bit-identical to `_medoid_full` - the
        cosine is symmetric, values are memoised (not recomputed), and every row sum
        adds the same per-pair values in the same member order with the same
        first-minimum tie-break."""
        seqs = list(self._gallery_seq)
        cache = self._pair_cost
        # Drop entries whose older endpoint was evicted from the ring buffer. Current
        # seqs are a contiguous range; anything below the current minimum is stale.
        min_seq = seqs[0]
        if min_seq > self._pair_cost_min_seq:
            if cache:
                self._pair_cost = cache = {k: v for k, v in cache.items() if k[0] >= min_seq}
            self._pair_cost_min_seq = min_seq
        n = len(members)
        msk = self._config.min_shared_keypoints
        for a in range(n):
            sa = seqs[a]
            for b in range(a + 1, n):
                key = (sa, seqs[b])
                if key not in cache:
                    cache[key] = masked_weighted_cosine(members[a], members[b], min_shared_keypoints=msk)
        best_idx, best_cost = 0, float("inf")
        for i in range(n):
            si = seqs[i]
            total = 0.0
            for j in range(n):
                if j == i:
                    continue
                sj = seqs[j]
                total += cache[(si, sj)] if si < sj else cache[(sj, si)]
            if total < best_cost:
                best_idx, best_cost = i, total
        return members[best_idx]

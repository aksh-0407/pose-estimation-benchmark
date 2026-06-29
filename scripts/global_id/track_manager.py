"""P4a online global-identity tracker on the ground plane.

Rebuilt as a single auditable multi-object tracker (Chen et al. CVPR'20 cross-view
tracking; AI-City'24 geometric-consistency MTMC). P3 has already solved the *spatial*
cross-camera association: each :class:`Correspondence` is one world detection whose
``ground_xy`` is the foot point on the calibrated pitch plane, with at most one member
per camera. P4a only has to solve *temporal* association — link those world detections
into persistent tracks and mint a stable ``global_player_id``.

Per frame the tracker runs three small, independently-correct stages:

1. **Identity continuity** — a correspondence whose P2 ``local_track_id`` is already owned
   by exactly one live track sticks to that track (rescues tracks across drift/occlusion).
2. **Geometric assignment** — remaining correspondences ↔ remaining tracks by a
   chi2-gated Mahalanobis ground distance solved with the Hungarian algorithm.
3. **Re-entry / birth** — an unmatched correspondence revives a kinematically-consistent
   deleted track or spawns a new tentative one.

Because each correspondence has <=1 member per camera and is mapped to exactly one track
(the per-frame correspondence->track map is injective: Hungarian is 1:1, re-entry claims
distinct deleted tracks, births are fresh), two detections in the same camera-frame can
never receive the same ``global_player_id``. The invariant holds by construction, so no
post-hoc collision repair is needed.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment

from pose_estimation.cricket.ground_kalman import RoleParams, SingerGroundKalman
from scripts.association.associator import Correspondence
from scripts.global_id.config import P4Config
from scripts.global_id.global_track import CONFIRMED, DELETED, LOST, TENTATIVE, GlobalTrack

_LARGE_COST = 1e12


class TrackManager:
    """Track one delivery. IDs are deterministic because the counter is instance-local."""

    def __init__(self, config: P4Config) -> None:
        self.config = config
        self.tracks: list[GlobalTrack] = []
        self.deleted_pool: list[GlobalTrack] = []
        self._next_id = 1
        self._last_frame: int | None = None
        self._local_owners: dict[tuple[str, str], GlobalTrack] = {}
        # Retained as an empty mapping for runner write-back compatibility; the
        # rebuilt tracker assigns whole correspondences, never split members.
        self.last_member_assignments: dict[tuple[int, str], GlobalTrack] = {}
        self.diagnostics: Counter[str] = Counter()
        self._role_params = {
            role: RoleParams(**values) for role, values in config.p4a.role_params.items()
        }

    # ------------------------------------------------------------------ helpers

    def _mint_id(self) -> str:
        player_id = f"P{self._next_id:03d}"
        self._next_id += 1
        return player_id

    @staticmethod
    def _local_ids(correspondence: Correspondence) -> dict[str, str]:
        return {
            camera_id: detection.local_track_id
            for camera_id, detection in correspondence.members.items()
            if detection.local_track_id is not None
        }

    @staticmethod
    def _representative_bbox(correspondence: Correspondence) -> list[float] | None:
        if not correspondence.members:
            return None
        camera_id = sorted(correspondence.members)[0]
        return correspondence.members[camera_id].bbox_xywh_px

    def _claim_local_ids(self, track: GlobalTrack, local_ids: dict[str, str]) -> dict[str, str]:
        """Bind each (camera, P2 tracklet) to ``track`` unless another track owns it."""

        accepted: dict[str, str] = {}
        for camera_id, local_track_id in local_ids.items():
            key = (camera_id, local_track_id)
            owner = self._local_owners.get(key)
            if owner is None or owner is track:
                self._local_owners[key] = track
                accepted[camera_id] = local_track_id
            else:
                self.diagnostics["local_track_reassignment_conflicts_prevented"] += 1
        track.register_local_ids(accepted)
        return accepted

    def _identity_owner(self, local_ids: dict[str, str], hit: set[int]) -> GlobalTrack | None:
        """Return the single track (any state) that owns one of these P2 tracklets.

        A P2 tracklet id is unique to one physical person within a camera, so an exact
        match is decisive even across a deletion — stronger than any foot projection.
        """

        owners = {
            id(owner): owner
            for camera_id, local_track_id in local_ids.items()
            if (owner := self._local_owners.get((camera_id, local_track_id))) is not None
            and id(owner) not in hit
        }
        if len(owners) == 1:
            return next(iter(owners.values()))
        if len(owners) > 1:
            self.diagnostics["local_identity_ambiguous"] += 1
        return None

    def _revive_owned_track(self, track: GlobalTrack, frame_index: int) -> None:
        """Move a lost/deleted track back into the active set on an exact-identity hit."""

        if any(item is track for item in self.tracks):
            return
        self.deleted_pool = [item for item in self.deleted_pool if item is not track]
        track.predict_to(frame_index, max_pos_var=self.config.p4a.cap_max_pos_var)
        track.state = CONFIRMED if track.global_player_id is not None else TENTATIVE
        self.tracks.append(track)
        self.diagnostics["local_owner_reentries"] += 1

    def _apply_match(self, observation: Correspondence, track: GlobalTrack, frame_index: int) -> None:
        """Full Kalman update from a correspondence's ground point."""

        local_ids = self._claim_local_ids(track, self._local_ids(observation))
        track.apply_hit(
            observation.ground_xy,
            self._representative_bbox(observation),
            None,
            frame_index,
            single_camera=observation.single_camera,
            local_track_ids_by_cam=local_ids,
        )

    def _spawn(self, observation: Correspondence, frame_index: int) -> GlobalTrack:
        track = GlobalTrack(
            global_player_id=None,
            state=TENTATIVE,
            kalman=SingerGroundKalman(
                observation.ground_xy,
                "unknown",
                dt=1.0 / self.config.frame_rate_fps,
                role_params=self._role_params,
            ),
            first_frame=frame_index,
            last_frame=frame_index,
            prediction_frame=frame_index,
            first_ground_pos=np.asarray(observation.ground_xy, dtype=float).copy(),
            last_ground_pos=np.asarray(observation.ground_xy, dtype=float).copy(),
            last_bbox_xywh_px=self._representative_bbox(observation),
            single_camera=observation.single_camera,
            local_track_ids_by_cam=self._local_ids(observation),
        )
        self.tracks.append(track)
        self._claim_local_ids(track, self._local_ids(observation))
        self.diagnostics["tracks_spawned"] += 1
        return track

    def _build_assign_matrix(
        self,
        observations: list[Correspondence],
        tracks: list[GlobalTrack],
    ) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        """Chi2-gated Mahalanobis Hungarian assignment of correspondences to tracks."""

        if not observations or not tracks:
            return [], set(range(len(observations))), set(range(len(tracks)))
        cost = np.full((len(observations), len(tracks)), _LARGE_COST, dtype=float)
        for oi, observation in enumerate(observations):
            if not np.isfinite(observation.ground_xy).all():
                continue
            for ti, track in enumerate(tracks):
                mahalanobis = track.kalman.mahalanobis_sq(observation.ground_xy)
                if mahalanobis <= self.config.p4a.chi2_gate_2dof:
                    cost[oi, ti] = mahalanobis
        rows, columns = linear_sum_assignment(cost)
        matches: list[tuple[int, int]] = []
        unmatched_observations = set(range(len(observations)))
        unmatched_tracks = set(range(len(tracks)))
        for row, column in zip(rows, columns):
            if cost[row, column] >= _LARGE_COST:
                continue
            matches.append((int(row), int(column)))
            unmatched_observations.discard(int(row))
            unmatched_tracks.discard(int(column))
        return matches, unmatched_observations, unmatched_tracks

    def _try_reentry(self, ground_xy: np.ndarray, frame_index: int) -> GlobalTrack | None:
        """Revive a deleted track whose coasted state still explains this observation."""

        candidates: list[tuple[float, str, GlobalTrack, np.ndarray, np.ndarray]] = []
        H = np.zeros((2, 6), dtype=float)
        H[0, 0] = H[1, 1] = 1.0
        for track in self.deleted_pool:
            observation_gap = frame_index - track.last_frame
            if observation_gap <= 0 or observation_gap > self.config.p4a.reentry_temporal_gate_frames:
                continue
            propagation_steps = max(0, frame_index - track.prediction_frame)
            x_pred, P_pred = track.kalman.propagate_state(propagation_steps)
            innovation = np.asarray(ground_xy, dtype=float) - x_pred[:2]
            S = H @ P_pred @ H.T + track.kalman.R
            try:
                mahalanobis = float(innovation @ np.linalg.solve(S, innovation))
            except np.linalg.LinAlgError:
                continue
            gate = self.config.p4a.reentry_mahalanobis_gate * (
                1.0 + observation_gap / self.config.p4a.reentry_gap_scale_frames
            )
            if mahalanobis > gate:
                continue
            distance = float(np.linalg.norm(np.asarray(ground_xy) - track.last_ground_pos))
            maximum_distance = (
                self.config.kinematic_v_max_mps
                * observation_gap / self.config.frame_rate_fps
                * self.config.p4a.reentry_kinematic_slack
            )
            if distance > maximum_distance:
                self.diagnostics["reentry_kinematic_rejects"] += 1
                continue
            candidates.append((mahalanobis, track.global_player_id or "", track, x_pred, P_pred))
        if not candidates:
            return None
        _, _, winner, x_pred, P_pred = min(candidates, key=lambda item: (item[0], item[1]))
        winner.kalman.x = x_pred
        winner.kalman.P = P_pred
        winner.prediction_frame = frame_index
        self.deleted_pool.remove(winner)
        self.diagnostics["reentries"] += 1
        return winner

    # ------------------------------------------------------------------- update

    def update(
        self,
        correspondences: list[Correspondence],
        frame_index: int,
    ) -> dict[int, GlobalTrack]:
        """Process one synchronized frame; return ``cluster_id -> track`` assignments."""

        if self._last_frame is not None and frame_index <= self._last_frame:
            raise ValueError("TrackManager frame indices must be strictly increasing")
        self._last_frame = frame_index
        self.last_member_assignments = {}
        for track in self.tracks:
            track.predict_to(frame_index, max_pos_var=self.config.p4a.cap_max_pos_var)

        usable = [
            corr for corr in correspondences
            if corr.track_confidence >= self.config.p4a.confidence_discard
        ]
        grounded = [corr for corr in usable if np.isfinite(corr.ground_xy).all()]
        no_ground = [corr for corr in usable if not np.isfinite(corr.ground_xy).all()]
        active = [track for track in self.tracks if track.state in {CONFIRMED, TENTATIVE, LOST}]

        assignments: dict[int, GlobalTrack] = {}
        hit: set[int] = set()

        # --- Stage 1: exact P2 tracklet continuity (strongest evidence) -------
        claimed: set[int] = set()
        for observation in sorted(grounded, key=lambda item: item.cluster_id):
            local_ids = self._local_ids(observation)
            if not local_ids:
                continue
            track = self._identity_owner(local_ids, hit)
            if track is None:
                continue
            self._revive_owned_track(track, frame_index)  # no-op if already active
            mahalanobis = track.kalman.mahalanobis_sq(observation.ground_xy)
            if mahalanobis <= self.config.p4a.local_identity_mahalanobis_gate:
                self._apply_match(observation, track, frame_index)
            else:
                # Trust the P2 identity but do not corrupt the filter with a foot
                # projection that the motion model rejects (likely a bad ankle).
                self._claim_local_ids(track, local_ids)
                track.apply_identity_only_hit(
                    self._representative_bbox(observation), frame_index,
                    local_track_ids_by_cam=local_ids,
                )
                self.diagnostics["local_identity_ground_outliers"] += 1
            hit.add(id(track))
            assignments[observation.cluster_id] = track
            claimed.add(observation.cluster_id)
            self.diagnostics["local_identity_matches"] += 1

        # --- Stage 2: geometric Hungarian on the remainder --------------------
        remaining = [corr for corr in grounded if corr.cluster_id not in claimed]
        available = [track for track in active if id(track) not in hit]
        matches, unmatched_obs, _ = self._build_assign_matrix(remaining, available)
        for observation_index, track_index in matches:
            observation, track = remaining[observation_index], available[track_index]
            self._apply_match(observation, track, frame_index)
            hit.add(id(track))
            assignments[observation.cluster_id] = track
            self.diagnostics["geometry_matches"] += 1

        # --- Stage 3: re-entry from the deleted pool, else birth --------------
        for observation_index in sorted(unmatched_obs):
            observation = remaining[observation_index]
            track = self._try_reentry(observation.ground_xy, frame_index)
            if track is not None:
                track.state = CONFIRMED
                self._apply_match(observation, track, frame_index)
                self.tracks.append(track)
            else:
                track = self._spawn(observation, frame_index)
            hit.add(id(track))
            assignments[observation.cluster_id] = track

        # --- Ground-less correspondences: identity continuity only -----------
        # A single-view detection with no plausible foot projection cannot move
        # the filter, but it can still inherit an established identity so the
        # detection is not left without a global id.
        for observation in no_ground:
            local_ids = self._local_ids(observation)
            if not local_ids:
                continue
            track = self._identity_owner(local_ids, hit)
            if track is None:
                continue
            self._revive_owned_track(track, frame_index)  # no-op if already active
            self._claim_local_ids(track, local_ids)
            track.apply_identity_only_hit(
                self._representative_bbox(observation), frame_index,
                local_track_ids_by_cam=local_ids,
            )
            hit.add(id(track))
            assignments[observation.cluster_id] = track
            self.diagnostics["local_identity_bridges"] += 1

        # --- Coast unmatched tracks; confirm / delete -------------------------
        for track in self.tracks:
            if id(track) not in hit and track.state in {CONFIRMED, TENTATIVE, LOST}:
                track.mark_missed()
        self._promote_and_prune()
        return assignments

    def _promote_and_prune(self) -> None:
        survivors = []
        for track in self.tracks:
            if track.maybe_confirm(self.config.p4a.confirm_hits):
                track.global_player_id = self._mint_id()
                self.diagnostics["tracks_confirmed"] += 1
            if track.should_delete(
                confirm_hits=self.config.p4a.confirm_hits,
                lost_window_frames=self.config.p4a.lost_window_frames,
                bowler_lost_window_frames=self.config.p4a.bowler_lost_window_frames,
            ):
                track.state = DELETED
                if track.global_player_id is not None:
                    self.deleted_pool.append(track)
                self.diagnostics["tracks_deleted"] += 1
            else:
                survivors.append(track)
        self.tracks = survivors

    def propose_role(self, global_player_id: str, role: str, frame_index: int) -> None:
        """Dormant P5 hook: latch stable role proposals before changing dynamics."""

        del frame_index
        track = next((item for item in self.tracks if item.global_player_id == global_player_id), None)
        if track is None or role not in self._role_params:
            return
        if track.role_candidate == role:
            track.role_latch_count += 1
        else:
            track.role_candidate = role
            track.role_latch_count = 1
        if (
            track.role_latch_count >= self.config.p4a.role_latch_frames
            and track.dominant_role != role
        ):
            track.dominant_role = role
            track.kalman.switch_role(role)

    def finalize(self) -> None:
        """Promote viable end-of-delivery tentative tracks deterministically."""

        for track in self.tracks:
            if track.state == TENTATIVE and track.hits >= 2:
                track.state = CONFIRMED
                track.global_player_id = self._mint_id()
                self.diagnostics["tracks_confirmed_at_finalize"] += 1

"""P4a online global-identity tracker on the ground plane.

Rebuilt as a single auditable multi-object tracker (Chen et al. CVPR'20 cross-view
tracking; AI-City'24 geometric-consistency MTMC). P3 has already solved the *spatial*
cross-camera association: each :class:`Correspondence` is one world detection whose
``ground_xy`` is the foot point on the calibrated pitch plane, with at most one member
per camera. P4a only has to solve *temporal* association — link those world detections
into persistent tracks and mint a stable ``global_player_id``.

Per frame the tracker runs four small, independently-correct stages:

0. **Binding continuity** — a correspondence carrying a tracklet-graph ``binding_id``
   maps 1:1 to a persistent track for the whole delivery: the graph already solved
   who-is-who globally, so P4a must never re-litigate it per frame.
1. **Identity continuity** — a correspondence whose P2 ``local_track_id`` is already owned
   by exactly one live track sticks to that track (rescues tracks across drift/occlusion).
   Ownership claims expire after ``ownership_ttl_frames`` without re-assertion, so a
   transient bad P3 merge can no longer weld two players together permanently.
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
from pose_estimation.cricket.pose_shape import descriptor_distance, posture_distance_z
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
        # (camera_id, local_track_id) -> (owning track, frame of last claim).
        self._local_owners: dict[tuple[str, str], tuple[GlobalTrack, int]] = {}
        # tracklet-graph binding_id -> persistent track (Stage 0).
        self._binding_tracks: dict[str, GlobalTrack] = {}
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

    def _claim_local_ids(
        self, track: GlobalTrack, local_ids: dict[str, str], frame_index: int
    ) -> dict[str, str]:
        """Bind each (camera, P2 tracklet) to ``track`` unless another track owns it.

        A conflicting claim is refused while the current owner's claim is fresh, but
        succeeds once the owner has not re-asserted it for ``ownership_ttl_frames`` —
        the revocability that lets a transient bad merge heal instead of poisoning
        the tracklet for the rest of the delivery.
        """

        ttl = self.config.p4a.ownership_ttl_frames
        accepted: dict[str, str] = {}
        for camera_id, local_track_id in local_ids.items():
            key = (camera_id, local_track_id)
            entry = self._local_owners.get(key)
            expired = (
                entry is not None and ttl > 0 and frame_index - entry[1] > ttl
            )
            if entry is None or entry[0] is track or expired:
                if expired and entry[0] is not track:
                    self.diagnostics["local_track_ownership_transfers"] += 1
                self._local_owners[key] = (track, frame_index)
                accepted[camera_id] = local_track_id
            else:
                self.diagnostics["local_track_reassignment_conflicts_prevented"] += 1
        track.register_local_ids(accepted)
        return accepted

    def _identity_owner(self, local_ids: dict[str, str], hit: set[int]) -> GlobalTrack | None:
        """Return the single track (any state) that owns one of these P2 tracklets.

        A P2 tracklet id is unique to one physical person within a camera, so an exact
        match is decisive even across a deletion — stronger than any foot projection.
        Expired claims still count here (continuity is preserved); expiry only allows
        a *different* track to take over in :meth:`_claim_local_ids`.
        """

        owners = {
            id(entry[0]): entry[0]
            for camera_id, local_track_id in local_ids.items()
            if (entry := self._local_owners.get((camera_id, local_track_id))) is not None
            and id(entry[0]) not in hit
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

    def _measurement_R(
        self, observation: Correspondence, *, for_gating: bool = False
    ) -> np.ndarray | None:
        """Per-measurement R from the P3 ground covariance (F10), eigenvalue-clamped.

        None (fall back to the fixed role R) when the feature is off or the
        observation carries no covariance. The clamp keeps a spuriously tiny GN
        covariance from making the filter overconfident and a huge single-camera
        one from being ignored entirely.
        """

        if not self.config.p4a.use_measurement_covariance:
            return None
        if for_gating and not self.config.p4a.use_measurement_covariance_for_gating:
            return None
        cov = observation.ground_cov
        if cov is None:
            return None
        cov = np.asarray(cov, dtype=float) * self.config.p4a.r_scale
        if cov.shape != (2, 2) or not np.isfinite(cov).all():
            return None
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (cov + cov.T))
        except np.linalg.LinAlgError:
            return None
        floor = self.config.p4a.r_floor_m ** 2
        ceiling = self.config.p4a.r_ceiling_m ** 2
        eigenvalues = np.clip(eigenvalues, floor, ceiling)
        return eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    def _apply_match(self, observation: Correspondence, track: GlobalTrack, frame_index: int) -> None:
        """Full Kalman update from a correspondence's ground point."""

        local_ids = self._claim_local_ids(track, self._local_ids(observation), frame_index)
        track.apply_hit(
            observation.ground_xy,
            self._representative_bbox(observation),
            observation.pose_descriptor,
            frame_index,
            single_camera=observation.single_camera,
            local_track_ids_by_cam=local_ids,
            pose_ema_rate=self.config.p4a.pose_descriptor_ema,
            posture=observation.posture,
            measurement_R=self._measurement_R(observation),
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
        if observation.pose_descriptor is not None and observation.pose_descriptor.is_defined():
            track.pose_proportions = observation.pose_descriptor
            track.pose_update_count = 1
        if observation.posture is not None and observation.posture.is_defined():
            track.posture = observation.posture
        self.tracks.append(track)
        self._claim_local_ids(track, self._local_ids(observation), frame_index)
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
            observation_R = self._measurement_R(observation, for_gating=True)
            for ti, track in enumerate(tracks):
                mahalanobis = track.kalman.mahalanobis_sq(observation.ground_xy, R=observation_R)
                if mahalanobis <= self.config.p4a.chi2_gate_2dof:
                    # Pose is added INSIDE the gate only: it re-ranks candidates that
                    # geometry already admits, and contributes nothing (None distance)
                    # until a track has a mature descriptor and the two share enough
                    # segments -- so behaviour is unchanged when pose is absent.
                    pose_penalty = 0.0
                    if track.pose_update_count >= self.config.p4a.pose_min_updates:
                        distance = descriptor_distance(
                            track.pose_proportions,
                            observation.pose_descriptor,
                            min_shared=self.config.p4a.pose_min_shared_segments,
                        )
                        if distance is not None:
                            veto = self.config.p4a.pose_gate_veto_distance
                            if veto > 0.0 and distance > veto:
                                # Clearly the wrong build: veto rather than penalise, so
                                # a mis-shapen candidate cannot capture this track (ID-3).
                                self.diagnostics["pose_gate_vetoes"] += 1
                                continue
                            pose_penalty = self.config.p4a.pose_match_weight * distance
                    posture_veto = self.config.p4a.posture_gate_veto_z
                    if posture_veto > 0.0:
                        # F6b: the billboard posture works on the facing pairs where
                        # the triangulated descriptor cannot mature, so this veto
                        # bites exactly on the clips where pose_gate_veto rarely fires.
                        posture_z = posture_distance_z(track.posture, observation.posture)
                        if posture_z is not None and posture_z[0] > posture_veto:
                            self.diagnostics["posture_gate_vetoes"] += 1
                            continue
                    cost[oi, ti] = mahalanobis + pose_penalty
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

    def _try_absorb(
        self,
        observation: Correspondence,
        cams_hit: dict[int, set[str]],
    ) -> GlobalTrack | None:
        """Best already-updated track this observation is a same-frame shadow of."""

        observation_cams = set(observation.members)
        best: GlobalTrack | None = None
        best_distance = float("inf")
        for track in self.tracks:
            if track.state not in {CONFIRMED, TENTATIVE, LOST}:
                continue
            covered = cams_hit.get(id(track))
            if not covered or covered & observation_cams:
                continue  # not updated this frame, or would double-book a camera
            mahalanobis = track.kalman.mahalanobis_sq(
                observation.ground_xy, R=self._measurement_R(observation, for_gating=True)
            )
            if mahalanobis <= self.config.p4a.chi2_gate_2dof and mahalanobis < best_distance:
                best, best_distance = track, mahalanobis
        return best

    def _density_tightened(self, track: GlobalTrack) -> bool:
        return track.density_at_loss >= self.config.p4a.density_gate_tighten_min_count

    def _effective_pose_veto(self, track: GlobalTrack) -> float:
        veto = self.config.p4a.reentry_pose_max_distance
        if veto > 0.0 and self._density_tightened(track):
            return veto * self.config.p4a.density_gate_tighten_factor
        return veto

    def _effective_posture_veto(self, track: GlobalTrack) -> float:
        veto = self.config.p4a.reentry_posture_max_z
        if veto > 0.0 and self._density_tightened(track):
            return veto * self.config.p4a.density_gate_tighten_factor
        return veto

    def _try_reentry(self, observation: Correspondence, frame_index: int) -> GlobalTrack | None:
        """Revive a deleted track whose coasted state still explains this observation.

        Kinematic reachability gates position; when ``reentry_pose_max_distance`` is
        set, the revived track's body-shape descriptor must additionally agree with
        the observation (abstaining when either is immature/unshared). This blocks the
        kinematically-plausible-but-wrong-person re-entries that read as teleports.
        """

        ground_xy = observation.ground_xy
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
            reentry_R = self._measurement_R(observation, for_gating=True)
            S = H @ P_pred @ H.T + (track.kalman.R if reentry_R is None else reentry_R)
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
            pose_veto = self._effective_pose_veto(track)
            if pose_veto > 0.0 and track.pose_update_count >= self.config.p4a.pose_min_updates:
                pose_distance = descriptor_distance(
                    track.pose_proportions, observation.pose_descriptor,
                    min_shared=self.config.p4a.pose_min_shared_segments,
                )
                if pose_distance is not None and pose_distance > pose_veto:
                    self.diagnostics["reentry_pose_rejects"] += 1
                    continue
            posture_veto = self._effective_posture_veto(track)
            if posture_veto > 0.0:
                posture_z = posture_distance_z(track.posture, observation.posture)
                if posture_z is not None and posture_z[0] > posture_veto:
                    self.diagnostics["reentry_posture_rejects"] += 1
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
        claimed: set[int] = set()
        # Cameras each track has been observed in THIS frame — an absorb may only
        # add coverage from cameras the track has no member in yet.
        cams_hit: dict[int, set[str]] = {}

        def record_cams(track: GlobalTrack, observation: Correspondence) -> None:
            cams_hit.setdefault(id(track), set()).update(observation.members)

        # --- Stage 0: tracklet-graph binding continuity ------------------------
        # The graph solved identity globally; a binding maps 1:1 to a persistent
        # track for the whole delivery, so cross-camera agreement holds by
        # construction wherever bindings exist.
        for observation in sorted(
            (corr for corr in usable if corr.binding_id is not None),
            key=lambda item: item.cluster_id,
        ):
            has_ground = bool(np.isfinite(observation.ground_xy).all())
            track = self._binding_tracks.get(observation.binding_id)
            if track is None:
                if not has_ground:
                    continue  # cannot seed a motion model without a ground point
                track = self._spawn(observation, frame_index)
                self._binding_tracks[observation.binding_id] = track
                self.diagnostics["binding_tracks_spawned"] += 1
            else:
                if id(track) in hit:
                    # Two same-frame correspondences claiming one binding cannot
                    # both be right; keep the first, let the rest fall through.
                    self.diagnostics["binding_same_frame_conflicts"] += 1
                    continue
                self._revive_owned_track(track, frame_index)  # no-op if already active
                local_ids = self._local_ids(observation)
                if (
                    has_ground
                    and track.kalman.mahalanobis_sq(
                        observation.ground_xy,
                        R=self._measurement_R(observation, for_gating=True),
                    )
                    <= self.config.p4a.local_identity_mahalanobis_gate
                ):
                    self._apply_match(observation, track, frame_index)
                else:
                    # Keep the identity, protect the filter from an outlier foot.
                    self._claim_local_ids(track, local_ids, frame_index)
                    track.apply_identity_only_hit(
                        self._representative_bbox(observation), frame_index,
                        local_track_ids_by_cam=local_ids,
                    )
                    if has_ground:
                        self.diagnostics["binding_ground_outliers"] += 1
                self.diagnostics["binding_matches"] += 1
            hit.add(id(track))
            record_cams(track, observation)
            assignments[observation.cluster_id] = track
            claimed.add(observation.cluster_id)

        # --- Stage 1: exact P2 tracklet continuity (strongest evidence) -------
        for observation in sorted(grounded, key=lambda item: item.cluster_id):
            if observation.cluster_id in claimed:
                continue
            local_ids = self._local_ids(observation)
            if not local_ids:
                continue
            track = self._identity_owner(local_ids, hit)
            if track is None:
                continue
            self._revive_owned_track(track, frame_index)  # no-op if already active
            mahalanobis = track.kalman.mahalanobis_sq(
                observation.ground_xy,
                R=self._measurement_R(observation, for_gating=True),
            )
            if mahalanobis <= self.config.p4a.local_identity_mahalanobis_gate:
                self._apply_match(observation, track, frame_index)
            else:
                # Trust the P2 identity but do not corrupt the filter with a foot
                # projection that the motion model rejects (likely a bad ankle).
                self._claim_local_ids(track, local_ids, frame_index)
                track.apply_identity_only_hit(
                    self._representative_bbox(observation), frame_index,
                    local_track_ids_by_cam=local_ids,
                )
                self.diagnostics["local_identity_ground_outliers"] += 1
            hit.add(id(track))
            record_cams(track, observation)
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
            record_cams(track, observation)
            assignments[observation.cluster_id] = track
            self.diagnostics["geometry_matches"] += 1

        # --- Stage 2.5: absorb instead of birthing a shadow -------------------
        # An unmatched observation inside the chi2 gate of a track that was
        # ALREADY updated this frame is that same player seen by a camera the
        # track has no member in (its position matches the "ghost"). It joins
        # that track identity-only; it must never mint a duplicate id.
        still_unmatched: list[int] = []
        for observation_index in sorted(unmatched_obs):
            observation = remaining[observation_index]
            track = self._try_absorb(observation, cams_hit)
            if track is None:
                still_unmatched.append(observation_index)
                continue
            local_ids = self._local_ids(observation)
            self._claim_local_ids(track, local_ids, frame_index)
            track.apply_identity_only_hit(
                self._representative_bbox(observation), frame_index,
                local_track_ids_by_cam=local_ids,
            )
            record_cams(track, observation)
            assignments[observation.cluster_id] = track
            self.diagnostics["shadow_absorbs"] += 1

        # --- Stage 3: re-entry from the deleted pool, else birth --------------
        for observation_index in still_unmatched:
            observation = remaining[observation_index]
            track = self._try_reentry(observation, frame_index)
            if track is not None:
                track.state = CONFIRMED
                self._apply_match(observation, track, frame_index)
                self.tracks.append(track)
            else:
                track = self._spawn(observation, frame_index)
            hit.add(id(track))
            record_cams(track, observation)
            assignments[observation.cluster_id] = track

        # --- Ground-less correspondences: identity continuity only -----------
        # A single-view detection with no plausible foot projection cannot move
        # the filter, but it can still inherit an established identity so the
        # detection is not left without a global id.
        for observation in no_ground:
            if observation.cluster_id in claimed:
                continue
            local_ids = self._local_ids(observation)
            if not local_ids:
                continue
            track = self._identity_owner(local_ids, hit)
            if track is None:
                continue
            self._revive_owned_track(track, frame_index)  # no-op if already active
            self._claim_local_ids(track, local_ids, frame_index)
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
                if track.state == CONFIRMED:
                    track.density_at_loss = self._local_track_count(track)
                track.mark_missed()
        self._promote_and_prune()
        return assignments

    def _local_track_count(self, track: GlobalTrack) -> int:
        """Count other CONFIRMED/LOST tracks within local_density_radius_m (C.1).

        Captured at the CONFIRMED -> LOST transition to record how crowded the
        scene was at the moment identity was lost -- a candidate signal for
        occlusion-driven loss vs. a clean exit.
        """

        radius = self.config.p4a.local_density_radius_m
        position = track.last_ground_pos
        if not np.isfinite(position).all():
            return 0
        count = 0
        for other in self.tracks:
            if other is track or other.state not in {CONFIRMED, LOST}:
                continue
            if not np.isfinite(other.last_ground_pos).all():
                continue
            if float(np.linalg.norm(other.last_ground_pos - position)) <= radius:
                count += 1
        return count

    def _confirmation_blocked(self, track: GlobalTrack) -> bool:
        """A tentative may not confirm while it shadows an existing identity.

        A duplicate born on top of a tracked player sits inside the shadow gate
        of a CONFIRMED track; it stays tentative (invisible) until it either
        separates or persists long enough (override) to be a real player — e.g.
        two batsmen legitimately standing together confirm via the override. At
        the roster cap (cricket: 15 on the field), a new id additionally needs
        clear separation from every confirmed track.
        """

        if track.hits >= self.config.p4a.shadow_confirm_override_hits:
            return False
        position = track.kalman.pos_world_xy
        if not np.isfinite(position).all():
            return False
        confirmed = [
            other for other in self.tracks
            if other is not track and other.global_player_id is not None
            and other.state in {CONFIRMED, LOST}
        ]
        distances = [
            float(np.linalg.norm(other.kalman.pos_world_xy - position))
            for other in confirmed
            if np.isfinite(other.kalman.pos_world_xy).all()
        ]
        if distances and min(distances) <= self.config.p4a.shadow_confirm_gate_m:
            self.diagnostics["shadow_confirmations_blocked"] += 1
            return True
        if (
            len(confirmed) >= self.config.p4a.expected_roster_max
            and distances
            and min(distances) <= self.config.p4a.roster_cap_min_separation_m
        ):
            self.diagnostics["roster_cap_confirmations_blocked"] += 1
            return True
        return False

    def _promote_and_prune(self) -> None:
        survivors = []
        for track in self.tracks:
            if (
                track.state == TENTATIVE
                and track.hits >= self.config.p4a.confirm_hits
                and self._confirmation_blocked(track)
            ):
                pass  # stays tentative: emits no id until it separates/persists
            elif track.maybe_confirm(self.config.p4a.confirm_hits):
                track.global_player_id = self._mint_id()
                self.diagnostics["tracks_confirmed"] += 1
            if track.should_delete(
                confirm_hits=self.config.p4a.confirm_hits,
                lost_window_frames=self.config.p4a.lost_window_frames,
                adaptive_lost_window=self.config.p4a.adaptive_lost_window,
                lost_window_max_frames=self.config.p4a.lost_window_max_frames,
                confirm_bonus_scale=self.config.p4a.confirm_bonus_scale,
                lost_window_k1=self.config.p4a.lost_window_k1,
                lost_window_k2=self.config.p4a.lost_window_k2,
                expected_roster_max=self.config.p4a.expected_roster_max,
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

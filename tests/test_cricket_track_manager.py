from __future__ import annotations

from dataclasses import replace

import numpy as np

from scripts.association.associator import Correspondence, Detection3
from scripts.global_id.config import P4AConfig, P4Config
from scripts.global_id.global_track import CONFIRMED, TENTATIVE
from scripts.global_id.track_manager import TrackManager


def _correspondence(frame: int, xy=(0.0, 0.0), confidence=0.8) -> Correspondence:
    detection = Detection3(
        "cam_01", 0, [100.0, 100.0, 40.0, 100.0], np.zeros((17, 2)),
        np.ones(17), confidence, "cam_01_trk_0001",
    )
    return Correspondence(frame, {"cam_01": detection}, np.asarray(xy, float), confidence, False)


def _config(**p4a_overrides) -> P4Config:
    return P4Config(p4a=replace(P4AConfig(), **p4a_overrides))


def test_tentative_track_confirms_after_configured_hits():
    manager = TrackManager(_config(confirm_hits=3))
    manager.update([_correspondence(0)], 0)
    assert manager.tracks[0].state == TENTATIVE
    manager.update([_correspondence(1)], 1)
    manager.update([_correspondence(2)], 2)
    assert manager.tracks[0].state == CONFIRMED
    assert manager.tracks[0].global_player_id == "P001"


def test_deleted_track_reentry_preserves_id():
    manager = TrackManager(_config(confirm_hits=2, lost_window_frames=1, reentry_temporal_gate_frames=20))
    manager.update([_correspondence(0)], 0)
    manager.update([_correspondence(1)], 1)
    assert manager.tracks[0].global_player_id == "P001"
    manager.update([], 2)
    manager.update([], 3)
    assert not manager.tracks
    manager.update([_correspondence(4, xy=(0.05, 0.0))], 4)
    assert manager.tracks[0].global_player_id == "P001"
    assert manager.tracks[0].state == CONFIRMED


def test_exact_local_owner_survives_a_ground_projection_outlier():
    manager = TrackManager(_config(confirm_hits=2, lost_window_frames=1, reentry_temporal_gate_frames=20))
    manager.update([_correspondence(0)], 0)
    manager.update([_correspondence(1)], 1)
    manager.update([], 2)
    manager.update([], 3)
    manager.update([_correspondence(4, xy=(20.0, 0.0))], 4)
    assert manager.tracks[0].state == CONFIRMED
    assert manager.tracks[0].global_player_id == "P001"
    assert not manager.deleted_pool
    assert manager.diagnostics["local_identity_ground_outliers"] == 1


def test_id_counter_resets_for_each_delivery_manager():
    identifiers = []
    for _ in range(2):
        manager = TrackManager(_config(confirm_hits=2))
        manager.update([_correspondence(0)], 0)
        manager.update([_correspondence(1)], 1)
        identifiers.append(manager.tracks[0].global_player_id)
    assert identifiers == ["P001", "P001"]


def test_single_camera_local_track_continuity_preserves_global_id_without_ground_update():
    manager = TrackManager(_config(confirm_hits=2))
    manager.update([_correspondence(0)], 0)
    manager.update([_correspondence(1)], 1)
    position_before = manager.tracks[0].kalman.pos_world_xy.copy()
    single = Correspondence(
        99,
        _correspondence(2).members,
        np.full(2, np.nan),
        0.3,
        True,
    )
    assignments = manager.update([single], 2)
    assert assignments[99].global_player_id == "P001"
    assert np.allclose(manager.tracks[0].kalman.pos_world_xy, position_before, atol=0.1)
    assert manager.diagnostics["local_identity_bridges"] == 1


def _det(cam, player_index, local_track_id, conf=0.8):
    return Detection3(
        cam, player_index, [100.0, 100.0, 40.0, 100.0], np.zeros((17, 2)),
        np.ones(17), conf, local_track_id,
    )


def _multi(cluster_id, members, xy, confidence=0.8, single=False):
    """``members``: dict[cam_id -> Detection3]."""
    return Correspondence(cluster_id, members, np.asarray(xy, float), confidence, single)


def test_two_distinct_same_camera_detections_get_distinct_ids():
    # Two different people both visible in cam_01 in the same frame must never share
    # a global id -- the physically-impossible output this rebuild eliminates.
    manager = TrackManager(_config(confirm_hits=2))
    a = _multi(0, {"cam_01": _det("cam_01", 0, "cam_01_trk_0001")}, (0.0, 0.0))
    b = _multi(1, {"cam_01": _det("cam_01", 1, "cam_01_trk_0002")}, (8.0, 0.0))
    assignments = {}
    for frame in range(4):
        assignments = manager.update([a, b], frame)
        ids = [t.global_player_id for t in assignments.values() if t.global_player_id is not None]
        assert len(ids) == len(set(ids))  # no shared id within a single frame
    assert assignments[0].global_player_id != assignments[1].global_player_id
    assert all(t.global_player_id is not None for t in assignments.values())


def test_multi_camera_correspondence_shares_one_id_across_facing_cameras():
    # A facing pair (e.g. cam_02 / cam_06) sees one person -> P3 emits a single
    # 2-member correspondence -> exactly one global id spanning both cameras.
    manager = TrackManager(_config(confirm_hits=2))
    members = {
        "cam_02": _det("cam_02", 0, "cam_02_trk_0001"),
        "cam_06": _det("cam_06", 0, "cam_06_trk_0001"),
    }
    assignments = {}
    for frame in range(3):
        assignments = manager.update([_multi(0, members, (1.0, 2.0))], frame)
    track = assignments[0]
    assert track.global_player_id == "P001"
    assert ("cam_02", "cam_02_trk_0001") in track.local_track_id_history
    assert ("cam_06", "cam_06_trk_0001") in track.local_track_id_history


def test_every_confident_grounded_detection_receives_a_track():
    # No confident, ground-projected detection is left unassigned -- addresses the
    # "some detections get no id at all" complaint.
    manager = TrackManager(_config(confirm_hits=2))
    corrs = [
        _multi(0, {"cam_01": _det("cam_01", 0, "cam_01_trk_0001")}, (0.0, 0.0)),
        _multi(1, {"cam_03": _det("cam_03", 0, "cam_03_trk_0001")}, (5.0, 5.0)),
        _multi(2, {"cam_05": _det("cam_05", 0, "cam_05_trk_0001")}, (-5.0, 5.0)),
    ]
    assert set(manager.update(corrs, 0)) == {0, 1, 2}
    manager.update(corrs, 1)
    assignments = manager.update(corrs, 2)
    assert all(assignments[cluster].global_player_id is not None for cluster in (0, 1, 2))


def _skeleton(shin_z: float) -> tuple[np.ndarray, np.ndarray]:
    points = np.full((17, 3), np.nan)
    points[5] = [-0.20, 0.0, 1.40]; points[6] = [0.20, 0.0, 1.40]   # shoulders
    points[7] = [-0.25, 0.0, 1.10]; points[8] = [0.25, 0.0, 1.10]   # elbows
    points[9] = [-0.28, 0.0, 0.85]; points[10] = [0.28, 0.0, 0.85]  # wrists
    points[11] = [-0.12, 0.0, 0.95]; points[12] = [0.12, 0.0, 0.95]  # hips
    points[13] = [-0.13, 0.0, 0.50]; points[14] = [0.13, 0.0, 0.50]  # knees
    points[15] = [-0.13, 0.0, shin_z]; points[16] = [0.13, 0.0, shin_z]  # ankles vary
    conf = np.ones(17); conf[[0, 1, 2, 3, 4]] = 0.0
    return points, conf


def _descriptor(shin_z: float):
    from pose_estimation.cricket.pose_shape import limb_proportion_descriptor
    points, conf = _skeleton(shin_z)
    return limb_proportion_descriptor(points, conf, n_views=3)


def _pose_corr(cluster_id, xy, descriptor, local_track_id):
    detection = Detection3(
        "cam_01", 0, [100.0, 100.0, 40.0, 100.0], np.zeros((17, 2)),
        np.ones(17), 0.8, local_track_id,
    )
    return Correspondence(cluster_id, {"cam_01": detection}, np.asarray(xy, float), 0.8, False,
                          pose_descriptor=descriptor)


def test_pose_tiebreaker_reranks_two_gate_admissible_tracks():
    # Two mature tracks both fall inside the chi2 gate of one new observation, and
    # geometry alone would pick the nearer track A. But the observation's body
    # proportions match track B, so with pose enabled the id follows the pose.
    desc_a = _descriptor(0.05)   # normal shins
    desc_b = _descriptor(-0.45)  # much longer shins -> distinct proportions
    obs_like_b = _descriptor(-0.45)

    def run(pose_weight):
        manager = TrackManager(_config(
            confirm_hits=1, pose_match_weight=pose_weight,
            pose_min_updates=1, pose_min_shared_segments=4,
        ))
        # Birth + confirm two tracks at distinct spots with their own pose descriptors.
        manager.update([
            _pose_corr(0, (0.0, 0.0), desc_a, "cam_01_trk_A"),
            _pose_corr(1, (0.6, 0.0), desc_b, "cam_01_trk_B"),
        ], 0)
        track_a, track_b = manager.tracks[0], manager.tracks[1]
        assert track_a.pose_update_count >= 1 and track_b.pose_update_count >= 1
        # Observation nearer to A (0.25 vs 0.35) but with B's proportions and no
        # local-track id, so it is resolved by the Stage-2 geometric+pose cost.
        assignments = manager.update(
            [_pose_corr(2, (0.25, 0.0), obs_like_b, None)], 1
        )
        return assignments[2], track_a, track_b

    winner_geo, geo_a, _ = run(pose_weight=0.0)
    assert winner_geo is geo_a  # geometry alone assigns the nearer track

    winner_pose, _, pose_b = run(pose_weight=50.0)
    assert winner_pose is pose_b  # pose re-ranks the assignment to the matching track


def _bound_correspondence(
    cluster_id: int,
    binding_id: str,
    cameras: tuple[str, ...] = ("cam_01", "cam_04"),
    xy=(0.0, 0.0),
    confidence: float = 0.8,
) -> Correspondence:
    members = {
        cam_id: Detection3(
            cam_id, 0, [100.0, 100.0, 40.0, 100.0], np.zeros((17, 2)),
            np.ones(17), confidence, f"{cam_id}_trk_0001",
        )
        for cam_id in cameras
    }
    return Correspondence(
        cluster_id, members, np.asarray(xy, float), confidence, len(cameras) == 1,
        binding_id=binding_id,
    )


def test_binding_keeps_one_id_when_membership_flickers():
    """The historical failure: P3 splits a player's cameras apart for a few frames
    and the split half mints a new ID. With a binding the split halves keep
    resolving to the same persistent track."""

    manager = TrackManager(_config(confirm_hits=2))
    manager.update([_bound_correspondence(0, "B001")], 0)
    manager.update([_bound_correspondence(0, "B001")], 1)
    track = manager.tracks[0]
    assert track.global_player_id == "P001"

    # Frames 2-4: the correspondence "splits" — only one camera at a time, but
    # the binding persists, so no new track may appear.
    for frame, cameras in ((2, ("cam_01",)), (3, ("cam_04",)), (4, ("cam_01",))):
        assignments = manager.update(
            [_bound_correspondence(0, "B001", cameras=cameras, xy=(0.02 * frame, 0.0))],
            frame,
        )
        assert assignments[0] is track
    assert len(manager.tracks) == 1
    assert manager.diagnostics.get("tracks_spawned", 0) == 1
    assert manager.diagnostics["binding_matches"] == 4


def test_distinct_bindings_never_share_a_track():
    manager = TrackManager(_config(confirm_hits=2))
    for frame in range(3):
        manager.update(
            [
                _bound_correspondence(0, "B001", xy=(0.0, 0.0)),
                _bound_correspondence(1, "B002", xy=(1.5, 0.0)),
            ],
            frame,
        )
    identifiers = {track.global_player_id for track in manager.tracks}
    assert identifiers == {"P001", "P002"}


def test_binding_outlier_ground_becomes_identity_only_hit():
    manager = TrackManager(_config(confirm_hits=2))
    manager.update([_bound_correspondence(0, "B001")], 0)
    manager.update([_bound_correspondence(0, "B001")], 1)
    position_before = manager.tracks[0].kalman.pos_world_xy.copy()
    manager.update([_bound_correspondence(0, "B001", xy=(30.0, 0.0))], 2)
    assert manager.diagnostics["binding_ground_outliers"] == 1
    assert np.allclose(manager.tracks[0].kalman.pos_world_xy, position_before, atol=0.5)


def test_ownership_claim_expires_and_transfers():
    manager = TrackManager(_config(confirm_hits=2, ownership_ttl_frames=10))
    manager.update([_correspondence(0)], 0)
    manager.update([_correspondence(1)], 1)
    owner = manager.tracks[0]

    # A second track appears elsewhere and (wrongly) tries to claim the tracklet
    # while the owner's claim is fresh: refused.
    intruder_det = Detection3(
        "cam_01", 1, [500.0, 100.0, 40.0, 100.0], np.zeros((17, 2)),
        np.ones(17), 0.8, "cam_01_trk_0009",
    )
    intruder = Correspondence(7, {"cam_01": intruder_det}, np.asarray((8.0, 0.0)), 0.8, True)
    manager.update([_correspondence(2), intruder], 2)
    intruder_track = next(t for t in manager.tracks if t is not owner)
    refused = manager._claim_local_ids(
        intruder_track, {"cam_01": "cam_01_trk_0001"}, 3
    )
    assert refused == {}
    assert manager.diagnostics["local_track_reassignment_conflicts_prevented"] == 1

    # After the TTL lapses without the owner re-asserting, the claim transfers.
    accepted = manager._claim_local_ids(
        intruder_track, {"cam_01": "cam_01_trk_0001"}, 20
    )
    assert accepted == {"cam_01": "cam_01_trk_0001"}
    assert manager.diagnostics["local_track_ownership_transfers"] == 1


def _posture(head_top: float, torso: float):
    from pose_estimation.cricket.pose_shape import PostureAggregate
    return PostureAggregate(
        median={"head_top_m": head_top, "torso_len_m": torso},
        se={"head_top_m": 0.01, "torso_len_m": 0.01},
        count={"head_top_m": 50, "torso_len_m": 50},
    )


def test_posture_gate_vetoes_wrong_build_candidate():
    # Track built from a TALL posture; a nearby observation with a clearly SHORT
    # posture and a different local id must not capture the track when the
    # billboard-posture veto is armed (F6b).
    tall, short = _posture(1.85, 0.55), _posture(1.55, 0.42)

    def scenario(veto_z: float) -> str:
        manager = TrackManager(_config(confirm_hits=2, posture_gate_veto_z=veto_z))
        for frame in range(2):
            obs = replace(_correspondence(frame), posture=tall)
            manager.update([obs], frame)
        assert manager.tracks[0].global_player_id == "P001"
        intruder_det = Detection3(
            "cam_02", 0, [100.0, 100.0, 40.0, 100.0], np.zeros((17, 2)),
            np.ones(17), 0.8, "cam_02_trk_0009",
        )
        intruder = Correspondence(
            2, {"cam_02": intruder_det}, np.asarray([0.3, 0.0]), 0.8, False,
            posture=short,
        )
        manager.update([intruder], 2)
        matched = [t for t in manager.tracks if t.global_player_id == "P001"]
        return "captured" if matched and matched[0].last_frame == 2 else "spawned"

    assert scenario(0.0) == "captured"        # veto off: geometry admits the intruder
    assert scenario(3.0) == "spawned"          # veto on: wrong build cannot capture


def test_posture_abstains_when_missing():
    manager = TrackManager(_config(confirm_hits=2, posture_gate_veto_z=3.0))
    for frame in range(2):
        manager.update([_correspondence(frame)], frame)   # no posture anywhere
    obs = _correspondence(2, xy=(0.05, 0.0))
    manager.update([obs], 2)
    # No posture -> the veto must abstain and normal matching proceed.
    assert manager.tracks[0].last_frame == 2
    assert manager.diagnostics.get("posture_gate_vetoes", 0) == 0


def test_measurement_R_clamps_and_gates(tmp_path):
    import numpy as np

    manager = TrackManager(_config(
        confirm_hits=2, use_measurement_covariance=True,
        r_floor_m=0.15, r_ceiling_m=2.0,
    ))
    # tiny GN covariance -> floored; huge single-cam covariance -> ceilinged
    tiny = replace(_correspondence(0), ground_cov=np.eye(2) * 1e-6)
    huge = replace(_correspondence(0), ground_cov=np.eye(2) * 100.0)
    R_tiny = manager._measurement_R(tiny)
    R_huge = manager._measurement_R(huge)
    assert np.allclose(R_tiny, np.eye(2) * 0.15 ** 2)
    assert np.allclose(R_huge, np.eye(2) * 2.0 ** 2)
    # anisotropy survives the clamp (eigenvectors preserved)
    aniso = replace(_correspondence(0), ground_cov=np.array([[1.0, 0.0], [0.0, 0.09]]))
    R_aniso = manager._measurement_R(aniso)
    assert R_aniso[0, 0] > R_aniso[1, 1] > 0
    # missing covariance or flag off -> None (legacy role R)
    assert manager._measurement_R(_correspondence(0)) is None
    off = TrackManager(_config(confirm_hits=2))
    assert off._measurement_R(huge) is None


def test_density_scaled_lost_window():
    import numpy as np

    # Two confirmed tracks near each other; one goes missing INSIDE the pack ->
    # its recorded density earns a longer lost window than a lonely loss would.
    manager = TrackManager(_config(
        confirm_hits=2, lost_window_frames=3, adaptive_lost_window=True,
        lost_window_max_frames=60, density_lost_window=True,
        density_radius_m=2.0, density_bonus_frames=20,
    ))
    def obs(frame, cid, x, tid):
        det = Detection3("cam_01", cid, [100.0*(cid+1), 100.0, 40.0, 100.0],
                         np.zeros((17, 2)), np.ones(17), 0.9, tid)
        # cluster_id must be unique within one update (as P3 guarantees)
        return Correspondence(cid, {"cam_01": det}, np.asarray([x, 0.0]), 0.9, False)
    # 1.5 m apart: outside the shadow-confirm gate (1.2 m), inside the
    # density radius (2.0 m).
    for f in range(3):
        manager.update([obs(f, 0, 0.0, "cam_01_trk_A"), obs(f, 1, 1.5, "cam_01_trk_B")], f)
    assert sum(1 for t in manager.tracks if t.global_player_id) == 2
    # A disappears inside B's radius
    for f in range(3, 10):
        manager.update([obs(f, 1, 1.5, "cam_01_trk_B")], f)
    lost = [t for t in manager.tracks if t.global_player_id == "P001"]
    assert lost and lost[0].density_at_loss == 1
    # With the flat window (3) it would be deleted by now; density bonus keeps it.
    assert lost[0].state != "deleted"

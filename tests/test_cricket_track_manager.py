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

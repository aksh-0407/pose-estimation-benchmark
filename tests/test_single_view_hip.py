from __future__ import annotations

import numpy as np

from identity.p5_global_id.runner import _single_view_hip_positions

P = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=float)
PROJ = {"cam_01": P, "cam_02": P, "cam_03": P}


def _pose2(hip_conf=0.9):
    kp = [[0.0, 0.0]] * 26
    kp[11] = [0.4, 0.2]; kp[12] = [0.6, 0.2]
    conf = [0.0] * 26
    conf[11] = hip_conf; conf[12] = hip_conf
    return {"keypoints_px": kp, "confidence": conf}


def _pose3(z=0.9):
    kw = [None] * 26
    kw[11] = [1.0, 2.0, z]; kw[12] = [1.2, 2.0, z]
    return {"keypoints_world_m": kw}


def _rec(frame, cam, gid, with3d):
    p = {"global_player_id": gid, "pose_2d": _pose2()}
    if with3d:
        p["pose_3d"] = _pose3()
    return {"frame_index": frame, "camera_id": cam, "players": [p]}


def test_single_camera_frame_gets_hip_on_plane():
    recs = []
    # 6 multi-camera frames (2 cams) with triangulated hip z -> sticky height learned
    for f in range(6):
        recs.append(_rec(f, "cam_01", "P1", with3d=True))
        recs.append(_rec(f, "cam_02", "P1", with3d=True))
    # 1 single-camera frame (no pose_3d) -> should get a back-projected hip xy
    recs.append(_rec(100, "cam_01", "P1", with3d=False))
    out = _single_view_hip_positions(recs, PROJ, {}, min_conf=0.3, min_z_frames=5)
    assert ("P1", 100) in out and np.isfinite(out[("P1", 100)]).all()
    # multi-camera frames must NOT get a single-view entry
    assert ("P1", 0) not in out


def test_insufficient_triangulated_frames_no_sticky():
    # only 2 triangulated frames (< min_z_frames=5) -> no sticky height -> no entry
    recs = [_rec(0, "cam_01", "P2", True), _rec(0, "cam_02", "P2", True),
            _rec(50, "cam_01", "P2", False)]
    out = _single_view_hip_positions(recs, PROJ, {}, min_conf=0.3, min_z_frames=5)
    assert ("P2", 50) not in out


def test_id_remap_folds_online_ids():
    # online ids A/B remap to final P9; 6 triangulated multi-cam frames + 1 single-cam
    recs = []
    for f in range(6):
        recs.append(_rec(f, "cam_01", "A", True)); recs.append(_rec(f, "cam_02", "B", True))
    recs.append(_rec(100, "cam_01", "A", False))
    out = _single_view_hip_positions(recs, PROJ, {"A": "P9", "B": "P9"}, min_conf=0.3, min_z_frames=5)
    assert ("P9", 100) in out

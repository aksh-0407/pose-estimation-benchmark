import numpy as np

from pose_estimation.ue_export import build_pose_packet, cricket_world_to_ue_cm


def test_cricket_world_to_ue_cm_axis_mapping():
    points = np.array([[1.0, 2.0, 3.0]])
    ue = cricket_world_to_ue_cm(points)
    np.testing.assert_allclose(ue, np.array([[200.0, 100.0, 300.0]]))


def test_build_pose_packet_serializes_nullable_points():
    packet = build_pose_packet(
        frame_id="f001",
        timestamp_ns=123,
        player_id="p01",
        model_version="rtmw_l:test",
        calibration_id="cal01",
        keypoints3d_world_m=np.array([[1.0, 2.0, 3.0], [np.nan, np.nan, np.nan]]),
        confidence=[0.9, 0.0],
    ).to_dict()
    assert packet["keypoints3d_ue_cm"][0] == [200.0, 100.0, 300.0]
    assert packet["keypoints3d_world_m"][1] == [None, None, None]


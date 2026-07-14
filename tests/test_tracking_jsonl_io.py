import pytest

from core.contract import example_group1_frame
from identity.p2_tracking.config import TrackingConfig
from identity.p2_tracking.jsonl_io import frame_to_detections


def test_null_detection_confidence_falls_back_to_pose_mean():
    record = example_group1_frame(final_handoff=False)
    player = record["players"][0]
    player["local_track_id"] = None
    player["track_confidence"] = None
    player["detection_confidence"] = None
    player["pose_2d"]["confidence"] = [0.8] * 26
    diagnostics = {
        "malformed_detections_skipped": 0,
        "calibration_projection_failures": 0,
        "detection_confidence_pose_fallbacks": 0,
    }
    detections = frame_to_detections(record, TrackingConfig(), diagnostics=diagnostics)
    assert len(detections) == 1
    assert detections[0].confidence == pytest.approx(0.8)
    assert diagnostics["detection_confidence_pose_fallbacks"] == 1

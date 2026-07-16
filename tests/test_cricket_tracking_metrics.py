from __future__ import annotations

from identity.common.metrics import (
    identity_collision_metrics,
    numeric_summary,
)


def test_numeric_summary_ignores_missing_and_nonfinite_values():
    summary = numeric_summary([None, float("nan"), 1.0, 3.0])
    assert summary["count"] == 2
    assert summary["mean"] == 2.0


def test_identity_collision_metrics_detect_same_camera_duplicates_only():
    records = [{
        "frame_index": 1,
        "camera_id": "cam_01",
        "players": [{"global_player_id": "P001"}, {"global_player_id": "P001"}],
    }, {
        "frame_index": 1,
        "camera_id": "cam_02",
        "players": [{"global_player_id": "P001"}],
    }]
    metrics = identity_collision_metrics(records)
    assert metrics["same_camera_identity_collision_frames"] == 1
    assert metrics["same_camera_duplicate_identity_assignments"] == 1

from __future__ import annotations

from identity.common.metrics import (
    evaluate_ground_truth,
    identity_collision_metrics,
    numeric_summary,
)


def _record(frame, player_id, bbox=(0, 0, 10, 10)):
    return {
        "frame_index": frame,
        "camera_id": "cam_01",
        "players": [{"global_player_id": player_id, "bbox_xywh_px": list(bbox)}],
    }


def test_numeric_summary_ignores_missing_and_nonfinite_values():
    summary = numeric_summary([None, float("nan"), 1.0, 3.0])
    assert summary["count"] == 2
    assert summary["mean"] == 2.0


def test_ground_truth_metrics_count_identity_switch():
    predictions = [_record(1, "P001"), _record(2, "P002")]
    truth = [
        {"frame_index": 1, "camera_id": "cam_01", "bbox": [0, 0, 10, 10], "gt_id": "G1"},
        {"frame_index": 2, "camera_id": "cam_01", "bbox": [0, 0, 10, 10], "gt_id": "G1"},
    ]
    metrics = evaluate_ground_truth(predictions, truth)
    assert metrics["identity_switches"] == 1
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0
    assert metrics["mota"] == 0.5


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

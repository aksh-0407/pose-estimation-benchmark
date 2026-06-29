from __future__ import annotations

import numpy as np

from pose_estimation.cricket.geometry import compute_fundamental_matrix
from scripts.association.associator import (
    AnchorState,
    Detection3,
    _foot_pixel,
    build_cost_matrix,
    select_anchor,
    solve_optional_assignment,
)
from scripts.association.config import P3AssociationConfig
from scripts.association.geometry_cache import PairGeometry


def _det(camera: str, *, ankle_confidence: float = 0.0) -> Detection3:
    points = np.full((17, 2), [140.0, 440.0])
    confidence = np.full(17, 0.1)
    confidence[15:] = ankle_confidence
    return Detection3(camera, 0, [100.0, 200.0, 80.0, 240.0], points, confidence, 0.9, "trk")


def test_select_anchor_obeys_margin_and_minimum_dwell():
    config = P3AssociationConfig(anchor_hysteresis_margin=1, anchor_hysteresis_frames=3)
    detections = {"cam_01": [_det("cam_01")] * 3, "cam_02": [_det("cam_02")]}
    assert select_anchor(detections, AnchorState("cam_02", 2), config).anchor_id == "cam_02"
    assert select_anchor(detections, AnchorState("cam_02", 3), config).anchor_id == "cam_01"


def test_foot_pixel_prefers_ankles_then_bbox_bottom():
    config = P3AssociationConfig(ankle_conf_min=0.6)
    assert np.allclose(_foot_pixel(_det("cam_01"), config), [140.0, 440.0])
    ankle = _det("cam_01", ankle_confidence=0.9)
    ankle.keypoints_px[15] = [130.0, 445.0]
    ankle.keypoints_px[16] = [150.0, 447.0]
    assert np.allclose(_foot_pixel(ankle, config), [140.0, 446.0])

    # One implausibly raised/hallucinated ankle must not drag the ground point
    # upward. The lower plausible ankle is the contact.
    ankle.keypoints_px[15] = [20.0, 260.0]
    ankle.keypoints_px[16] = [151.0, 438.0]
    assert np.allclose(_foot_pixel(ankle, config), [151.0, 438.0])


def test_cost_matrix_contains_only_real_pair_costs():
    projection_a = np.array([[800.0, 0.0, 640.0, 0.0], [0.0, 800.0, 360.0, 0.0], [0, 0, 1, 5.0]])
    projection_b = projection_a.copy()
    projection_b[0, 3] = 50.0
    pair = PairGeometry(
        "cam_01", "cam_02", compute_fundamental_matrix(projection_a, projection_b),
        False, 0.6, 0.4, 10.0,
    )
    matrix = build_cost_matrix(
        [_det("cam_01"), _det("cam_01")],
        [_det("cam_02")] * 3,
        projection_a,
        projection_b,
        np.array([0.0, 0.0, -5.0]),
        np.array([-0.0625, 0.0, -5.0]),
        pair,
        P3AssociationConfig(),
    )
    assert matrix.shape == (2, 3)
    assert np.isfinite(matrix).all()


def test_optional_assignment_does_not_force_bad_real_pairs():
    cost = np.array([[0.2, 1e6], [1e6, 1e6]], dtype=float)
    assert solve_optional_assignment(cost, unmatched_cost=0.75) == [(0, 0)]

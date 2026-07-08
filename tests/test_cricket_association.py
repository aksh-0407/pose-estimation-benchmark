from __future__ import annotations

import numpy as np

from pose_estimation.cricket.geometry import compute_fundamental_matrix
from scripts.association.associator import (
    AnchorState,
    Detection3,
    _foot_pixel,
    build_cost_matrix,
    select_anchor,
    smooth_emit_feet,
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


def test_degenerate_pair_reallocates_epipolar_weight_to_ground():
    # Facing (co-observing) pairs are flagged degenerate because their epipolar
    # geometry is ill-conditioned. The cost must then drop the epipolar term and
    # reallocate its weight to the trustworthy ground cue, instead of scoring the
    # pair with a full-weight, unreliable Sampson term.
    import pytest
    from pose_estimation.cricket.geometry import pixel_to_ground_xy

    projection_a = np.array([[800.0, 0.0, 640.0, 0.0], [0.0, 800.0, 360.0, 0.0], [0, 0, 1, 5.0]])
    projection_b = projection_a.copy()
    projection_b[0, 3] = 50.0
    center_a = np.array([0.0, 0.0, -5.0])
    center_b = np.array([-0.0625, 0.0, -5.0])
    fundamental = compute_fundamental_matrix(projection_a, projection_b)
    dets_a = [_det("cam_01")]
    dets_b = [_det("cam_02")]
    config = P3AssociationConfig()

    def _cost(is_degenerate: bool) -> float:
        w_epi = 0.0 if is_degenerate else config.w_epi
        w_tri = 1.0 if is_degenerate else config.w_tri
        pair = PairGeometry("cam_01", "cam_02", fundamental, is_degenerate, w_epi, w_tri, 10.0)
        return float(build_cost_matrix(
            dets_a, dets_b, projection_a, projection_b, center_a, center_b, pair, config,
        )[0, 0])

    ground_a = pixel_to_ground_xy(_foot_pixel(dets_a[0], config), projection_a)
    ground_b = pixel_to_ground_xy(_foot_pixel(dets_b[0], config), projection_b)
    ground_distance = float(np.linalg.norm(ground_a - ground_b))
    assert ground_distance <= config.ground_distance_gate_m  # a real (finite) cost
    # appearance is None on both -> 0.5; no temporal continuity.
    expected = (config.ground_weight + config.epipolar_weight) * (
        ground_distance / config.ground_distance_gate_m
    ) + config.appearance_weight * 0.5
    assert _cost(True) == pytest.approx(expected, abs=1e-9)
    # A healthy pair keeps a separate, F-dependent epipolar term -> different value.
    assert _cost(False) != pytest.approx(expected, abs=1e-6)


def _foot_det(track_id, ankle_x, player_index=0):
    kp = np.zeros((17, 2)); kp[15] = [ankle_x - 5, 435.0]; kp[16] = [ankle_x + 5, 435.0]
    cf = np.zeros(17); cf[15] = 0.9; cf[16] = 0.9
    return Detection3(
        cam_id="cam_01", player_index=player_index,
        bbox_xywh_px=[100.0, 200.0, 80.0, 240.0], keypoints_px=kp, keypoint_conf=cf,
        confidence=0.9, local_track_id=track_id,
    )


def test_smooth_emit_feet_median_kills_spike_and_is_identity_safe():
    cfg = P3AssociationConfig(foot_contact_mode="v2", foot_smooth_window=3)
    xs = [140.0, 141.0, 300.0, 142.0, 143.0]  # frame 2 is a hallucinated-ankle spike
    frames = {f: {"cam_01": [_foot_det("cam_01_trk_1", x)]} for f, x in enumerate(xs)}

    out = smooth_emit_feet(frames, cfg)
    spiked = out[2]["cam_01"][0].emit_foot_px
    assert spiked is not None and 130.0 < float(spiked[0]) < 155.0  # spike suppressed
    for f in frames:  # emit-only: the gate input (ground_xy) is untouched
        assert out[f]["cam_01"][0].ground_xy is frames[f]["cam_01"][0].ground_xy

    off = smooth_emit_feet(frames, P3AssociationConfig(foot_smooth_window=1))
    assert off[2]["cam_01"][0].emit_foot_px is None  # window=1 is a no-op

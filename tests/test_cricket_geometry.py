"""Tests for the cross-camera geometry primitives (pose_estimation.cricket.geometry)."""

from __future__ import annotations

import numpy as np

from pose_estimation.cricket.geometry import (
    bbox_bottom_center_px,
    camera_axis_lookat,
    camera_center_from_P,
    compute_fundamental_matrix,
    compute_right_epipole,
    condition_number_dlt,
    derive_facing_pairs,
    fuse_ground_estimates,
    ground_from_reprojection,
    huber_cost,
    parallax_angle_deg,
    parallax_weight,
    pixel_to_ground_xy,
    reprojection_error_px,
    robust_fuse_ground,
    sampson_distance,
    triangulate_dlt,
)
from pose_estimation.triangulation import triangulate_point_dlt


def test_robust_fuse_ground_rejects_outlier_camera():
    # Two cameras agree at the origin with tight, isotropic covariance; a third
    # camera (a hallucinated foot) sits 3 m away with a loose covariance. Robust
    # fusion must land on the consensus, not be dragged toward the outlier the way
    # an unweighted mean/median would be.
    points = np.array([[0.02, -0.01], [-0.01, 0.02], [3.0, 0.0]])
    covs = np.array([
        np.diag([0.04, 0.04]),   # ~0.2 m std, good view
        np.diag([0.04, 0.04]),   # ~0.2 m std, good view
        np.diag([0.25, 0.25]),   # ~0.5 m std, still finite -> a mean would move ~1 m
    ])

    fused_xy, fused_cov, weights = robust_fuse_ground(points, covs, huber_delta=2.0)

    assert np.linalg.norm(fused_xy) < 0.25, fused_xy       # stays on consensus
    assert weights[2] < weights[0] and weights[2] < weights[1]  # outlier down-weighted
    assert np.all(np.isfinite(fused_cov))

    # The non-robust inverse-covariance fusion is pulled measurably off the consensus
    # by the same outlier -> confirms robustness actually did something.
    plain_xy, _ = fuse_ground_estimates(points, covs)
    assert np.linalg.norm(plain_xy) > np.linalg.norm(fused_xy)


def _cam_P(eye, target, K):
    R = _look_at(np.asarray(eye, float), np.asarray(target, float))
    t = -R @ np.asarray(eye, float)
    return K @ np.hstack([R, t[:, None]])


def test_ground_from_reprojection_recovers_point_and_beats_median():
    # Three calibrated cameras ringing the origin; a player foot on the ground at
    # (2, 1, 0). Feed exact foot pixels + one noisy view, and confirm the z=0
    # reprojection solve recovers the ground point to centimetres and beats the
    # median of per-camera homography back-projections.
    K = np.array([[1600.0, 0.0, 1280.0], [0.0, 1600.0, 720.0], [0.0, 0.0, 1.0]])
    cams = [
        _cam_P([25.0, 0.0, 6.0], [0, 0, 0], K),
        _cam_P([-20.0, 12.0, 5.0], [0, 0, 0], K),
        _cam_P([0.0, -22.0, 7.0], [0, 0, 0], K),
    ]
    foot_world = np.array([2.0, 1.0, 0.0])
    feet = []
    for P in cams:
        h = P @ np.append(foot_world, 1.0)
        feet.append(h[:2] / h[2])
    feet = np.asarray(feet)
    feet[2] += np.array([40.0, -30.0])  # one bad foot pixel (~50 px off)

    solved = ground_from_reprojection(feet, np.asarray(cams))
    median = np.median(
        np.asarray([pixel_to_ground_xy(feet[i], cams[i]) for i in range(3)]), axis=0
    )
    assert np.linalg.norm(solved - foot_world[:2]) < 0.10, solved      # ~cm accurate
    assert np.linalg.norm(solved - foot_world[:2]) < np.linalg.norm(median - foot_world[:2])


def test_ground_contact_v2_midpoint_planted_and_plausibility():
    from pose_estimation.cricket.geometry import ground_contact_pixel_ex

    bbox = [100.0, 200.0, 80.0, 240.0]  # bottom at (140, 440)
    conf = np.zeros(17); conf[15] = 0.9; conf[16] = 0.9
    kp = np.zeros((17, 2))

    # Both ankles confident and level near the bottom -> midpoint, ankle height reported.
    kp[15] = [120.0, 435.0]; kp[16] = [160.0, 437.0]
    px, h, src = ground_contact_pixel_ex(bbox, kp, conf, mode="v2", ankle_height_m=0.10)
    assert src == "ankle_mid" and np.allclose(px, [140.0, 436.0]) and h == 0.10

    # One foot clearly raised (striding) -> planted (lower/larger-y) foot.
    kp[15] = [120.0, 360.0]; kp[16] = [160.0, 438.0]   # 15 raised ~78px (> 0.15*240)
    px, h, src = ground_contact_pixel_ex(bbox, kp, conf, mode="v2")
    assert src == "ankle_planted" and px[1] == 438.0

    # Horizontally implausible ankle (far outside bbox) is rejected -> bbox bottom.
    kp[15] = [900.0, 436.0]; conf[16] = 0.0
    px, h, src = ground_contact_pixel_ex(bbox, kp, conf, mode="v2")
    assert src == "bbox_bottom" and h == 0.0

    # legacy mode still returns the lower ankle with height 0 (backward compatible).
    kp[15] = [120.0, 435.0]; kp[16] = [160.0, 437.0]; conf[16] = 0.9
    px, h, src = ground_contact_pixel_ex(bbox, kp, conf, mode="legacy")
    assert h == 0.0 and src == "ankle_mid"


def test_ground_from_reprojection_height_removes_ankle_bias():
    # Foot contact at (2, 1, 0); the ANKLE is 0.10 m above it. Cameras observe the
    # ankle pixel. Solving with the correct plane height recovers the ground contact;
    # solving as if the ankle were on the ground (h=0) is biased away.
    K = np.array([[1600.0, 0.0, 1280.0], [0.0, 1600.0, 720.0], [0.0, 0.0, 1.0]])
    cams = [
        _cam_P([25.0, 0.0, 6.0], [0, 0, 0], K),
        _cam_P([-20.0, 12.0, 5.0], [0, 0, 0], K),
        _cam_P([0.0, -22.0, 7.0], [0, 0, 0], K),
    ]
    ankle_world = np.array([2.0, 1.0, 0.10])
    feet = np.asarray([(P @ np.append(ankle_world, 1.0))[:2] / (P @ np.append(ankle_world, 1.0))[2] for P in cams])
    heights = np.full(3, 0.10)

    with_h = ground_from_reprojection(feet, np.asarray(cams), plane_heights=heights)
    without_h = ground_from_reprojection(feet, np.asarray(cams))  # assumes z=0
    assert np.linalg.norm(with_h - np.array([2.0, 1.0])) < 0.03
    assert np.linalg.norm(with_h - np.array([2.0, 1.0])) < np.linalg.norm(without_h - np.array([2.0, 1.0]))


def test_ground_from_reprojection_single_view_is_homography():
    K = np.array([[1600.0, 0.0, 1280.0], [0.0, 1600.0, 720.0], [0.0, 0.0, 1.0]])
    P = _cam_P([25.0, 0.0, 6.0], [0, 0, 0], K)
    foot_world = np.array([2.0, 1.0, 0.0])
    h = P @ np.append(foot_world, 1.0)
    foot_px = (h[:2] / h[2])[None, :]
    solved = ground_from_reprojection(foot_px, P[None, :, :])
    assert np.linalg.norm(solved - foot_world[:2]) < 1e-6


def test_robust_fuse_ground_single_member_falls_back():
    points = np.array([[1.0, 2.0], [np.nan, np.nan]])
    covs = np.array([np.diag([0.1, 0.1]), np.diag([np.inf, np.inf])])
    fused_xy, _fused_cov, weights = robust_fuse_ground(points, covs)
    assert np.allclose(fused_xy, [1.0, 2.0], atol=1e-6)
    assert weights[0] == 1.0 and weights[1] == 0.0


def _look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    z = eye - target
    z /= np.linalg.norm(z)
    up = np.array([0.0, 1.0, 0.0])
    if abs(z @ up) > 0.99:
        up = np.array([1.0, 0.0, 0.0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])


def _make_cameras():
    """Two cameras looking at the origin, roughly 90 degrees apart."""
    K = np.array([[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]])
    C1 = np.array([5.0, 0.0, 2.0])
    R1 = _look_at(C1, np.zeros(3))
    P1 = K @ np.hstack([R1, (-R1 @ C1).reshape(3, 1)])
    C2 = np.array([0.0, 5.0, 2.0])
    R2 = _look_at(C2, np.zeros(3))
    P2 = K @ np.hstack([R2, (-R2 @ C2).reshape(3, 1)])
    return P1, P2, C1, C2


def _project(X: np.ndarray, P: np.ndarray) -> np.ndarray:
    h = P @ np.append(X, 1.0)
    return h[:2] / h[2]


def _camera_at(eye: np.ndarray) -> np.ndarray:
    """Pinhole projection matrix for a camera at ``eye`` looking at the origin."""
    K = np.array([[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]])
    R = _look_at(eye, np.zeros(3))
    return K @ np.hstack([R, (-R @ eye).reshape(3, 1)])


def test_camera_axis_lookat_hits_ground_near_target():
    # A camera above the ground looking at the origin should resolve a ground
    # look-at point near the origin, with forward oriented toward the field.
    _, forward, lookat = camera_axis_lookat(_camera_at(np.array([12.0, 0.0, 4.0])))
    assert (np.zeros(3) - np.array([12.0, 0.0, 4.0])) @ forward > 0  # points toward field
    assert np.linalg.norm(lookat) < 2.0


def test_derive_facing_pairs_matches_anti_parallel_axes():
    # Cameras on opposite sides looking at the same spot are a facing pair; the
    # diagonal odd-one-out stays unpaired. Mirrors the real C1-C4 / C2-C6 / C3-C5 rig.
    # Far, low cameras (like the real ~100 m / ~12 m rig) so opposite axes are
    # near-anti-parallel; a steep close camera would not be a true facing pair.
    projections = {
        "cam_a": _camera_at(np.array([60.0, 0.0, 5.0])),
        "cam_b": _camera_at(np.array([-60.0, 0.0, 5.0])),
        "cam_c": _camera_at(np.array([0.0, 60.0, 5.0])),
        "cam_d": _camera_at(np.array([0.0, -60.0, 5.0])),
        "cam_e": _camera_at(np.array([42.0, 42.0, 5.0])),
    }
    pairs = {frozenset(pair) for pair in derive_facing_pairs(projections)}
    assert frozenset(("cam_a", "cam_b")) in pairs
    assert frozenset(("cam_c", "cam_d")) in pairs
    assert "cam_e" not in {cam for pair in pairs for cam in pair}


def test_triangulate_dlt_round_trip():
    P1, P2, _, _ = _make_cameras()
    X_true = np.array([0.5, 0.3, 0.0])
    X_est = triangulate_dlt(_project(X_true, P1), P1, _project(X_true, P2), P2)
    assert np.allclose(X_est, X_true, atol=1e-4)


def test_triangulate_dlt_matches_triangulation_module():
    # The wrapper must agree with the canonical weighted-DLT solver it delegates to.
    P1, P2, _, _ = _make_cameras()
    X_true = np.array([-0.4, 0.9, 0.0])
    x1, x2 = _project(X_true, P1), _project(X_true, P2)
    via_wrapper = triangulate_dlt(x1, P1, x2, P2)
    via_module = triangulate_point_dlt(np.array([x1, x2]), np.array([P1, P2]), min_views=2)
    assert np.allclose(via_wrapper, via_module, atol=1e-9)


def test_camera_center_recovered_from_projection_matrix():
    P1, _, C1, _ = _make_cameras()
    assert np.allclose(camera_center_from_P(P1), C1, atol=1e-6)


def test_reprojection_error_zero_for_exact():
    P1, _, _, _ = _make_cameras()
    X = np.array([0.5, 0.3, 0.0])
    assert reprojection_error_px(X, P1, _project(X, P1)) < 1e-4


def test_condition_number_perpendicular_better_than_collinear():
    P1, P2, _, _ = _make_cameras()
    X = np.array([0.5, 0.3, 0.0])
    x1, x2 = _project(X, P1), _project(X, P2)
    cond_good = condition_number_dlt(x1, P1, x2, P2)

    K = np.array([[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]])
    C_close = np.array([5.0, 0.01, 2.0])  # near-collinear with C1
    R_close = _look_at(C_close, np.zeros(3))
    P_close = K @ np.hstack([R_close, (-R_close @ C_close).reshape(3, 1)])
    cond_bad = condition_number_dlt(x1, P1, _project(X, P_close), P_close)
    assert cond_good < cond_bad


def test_parallax_angle_90_degrees():
    _, _, C1, C2 = _make_cameras()
    assert 80.0 < parallax_angle_deg(C1, C2, np.zeros(3)) < 100.0


def test_fundamental_matrix_epipolar_constraint():
    P1, P2, _, _ = _make_cameras()
    F = compute_fundamental_matrix(P1, P2)
    X = np.array([0.5, 0.3, 0.0])
    x1h = np.append(_project(X, P1), 1.0)
    x2h = np.append(_project(X, P2), 1.0)
    assert abs(x2h @ F @ x1h) < 1e-4


def test_sampson_distance_small_for_true_match():
    P1, P2, _, _ = _make_cameras()
    F = compute_fundamental_matrix(P1, P2)
    X = np.array([0.5, 0.3, 0.0])
    assert sampson_distance(_project(X, P1), F, _project(X, P2)) < 1e-4


def test_epipole_outside_image_for_perpendicular_cameras():
    P1, P2, _, _ = _make_cameras()
    e2 = compute_right_epipole(compute_fundamental_matrix(P1, P2))
    assert not (0.0 <= e2[0] <= 1280.0 and 0.0 <= e2[1] <= 720.0)


def test_bbox_bottom_center():
    assert np.allclose(bbox_bottom_center_px([100.0, 200.0, 80.0, 240.0]), [140.0, 440.0])


def test_huber_cost_quadratic_near_zero():
    assert abs(huber_cost(0.0, 5.0)) < 1e-12
    assert abs(huber_cost(1.0, 5.0) - 0.1) < 1e-9


def test_huber_cost_linear_beyond_delta():
    assert abs(huber_cost(10.0, 5.0) - 7.5) < 1e-9
    assert abs(huber_cost(5.0, 5.0) - 2.5) < 1e-9


def test_parallax_weight_bounds():
    assert parallax_weight(5.0, min_deg=10.0) == 0.0
    assert parallax_weight(30.0, min_deg=10.0) == 1.0

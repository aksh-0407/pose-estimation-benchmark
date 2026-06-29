import numpy as np

from pose_estimation.triangulation import (
    reprojection_errors_for_point,
    triangulate_point_dlt,
    triangulate_skeleton_ransac,
)


def test_weighted_dlt_triangulates_two_views():
    p1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=float)
    p2 = np.array([[1, 0, 0, -1], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=float)
    point = np.array([0.0, 0.0, 5.0])
    observations = np.array([[0.0, 0.0], [-0.2, 0.0]])
    result = triangulate_point_dlt(observations, np.stack([p1, p2]), np.ones(2))
    np.testing.assert_allclose(result, point, atol=1e-7)


def test_reprojection_preserves_negative_homogeneous_depth_sign():
    projection = -np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
        dtype=float,
    )
    point = np.array([1.0, 2.0, 5.0])
    errors = reprojection_errors_for_point(
        point,
        np.array([[0.2, 0.4]]),
        np.array([projection]),
    )
    assert errors[0] < 1e-12


def test_skeleton_ransac_rejects_bad_view():
    p1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=float)
    p2 = np.array([[1, 0, 0, -1], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=float)
    p3 = np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=float)
    keypoints = np.array(
        [
            [[0.0, 0.0, 1.0]],
            [[-0.2, 0.0, 1.0]],
            [[5.0, 5.0, 1.0]],
        ]
    )
    points3d, confidences, errors = triangulate_skeleton_ransac(
        keypoints,
        np.stack([p1, p2, p3]),
        reprojection_threshold_px=0.5,
    )
    np.testing.assert_allclose(points3d[0], np.array([0.0, 0.0, 5.0]), atol=1e-7)
    assert confidences[0] > 0.0
    assert errors[0] < 1e-7

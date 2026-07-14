from __future__ import annotations

import numpy as np

from identity.p3_association.associator import Detection3, _constrained_cluster, _triangulate_members
from identity.p3_association.config import P3AssociationConfig


def _look_at_projection(center: np.ndarray) -> np.ndarray:
    forward = -center / np.linalg.norm(center)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    intrinsic = np.array([[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]])
    return intrinsic @ np.hstack([rotation, (-rotation @ center).reshape(3, 1)])


def _project(point: np.ndarray, projection: np.ndarray) -> np.ndarray:
    homogeneous = projection @ np.append(point, 1.0)
    return homogeneous[:2] / homogeneous[2]


def _detection(camera: str, index: int, foot: np.ndarray) -> Detection3:
    points = np.tile(foot, (17, 1))
    return Detection3(
        camera, index, [foot[0] - 10, foot[1] - 40, 20, 40], points,
        np.full(17, 0.95), 0.95, f"{camera}-T{index}",
    )


def test_cycle_consistency_rejects_wrong_third_view_edge():
    centers = {
        "cam_01": np.array([8.0, 0.0, 4.0]),
        "cam_02": np.array([0.0, 8.0, 4.0]),
        "cam_03": np.array([-6.0, -5.0, 4.0]),
    }
    projections = {camera: _look_at_projection(center) for camera, center in centers.items()}
    world = [np.array([-0.8, 0.3, 0.0]), np.array([1.1, -0.4, 0.0])]
    detections = {
        camera: [_detection(camera, index, _project(point, projections[camera])) for index, point in enumerate(world)]
        for camera in projections
    }
    # Correct A-B pairs are established first. The deliberately cheap A0-C1
    # edge is then rejected because it breaks three-view reprojection closure.
    edges = [
        (0.0, ("cam_01", 0), ("cam_02", 0)),
        (0.0, ("cam_01", 1), ("cam_02", 1)),
        (0.1, ("cam_01", 0), ("cam_03", 1)),
        (0.2, ("cam_02", 0), ("cam_03", 0)),
        (0.2, ("cam_02", 1), ("cam_03", 1)),
    ]
    config = P3AssociationConfig(cycle_reproj_tol_px=3.0, triangulation_reproj_threshold_px=2.0)
    clusters = _constrained_cluster(edges, detections, projections, config)
    assert len(clusters) == 2
    assert all(len(cluster) == 3 for cluster in clusters)
    assert all(len(cluster) == len(set(cluster)) for cluster in clusters)
    for expected_index in range(2):
        cluster = next(item for item in clusters if item["cam_01"] == expected_index)
        assert set(cluster.values()) == {expected_index}
        point, error = _triangulate_members(cluster, detections, projections, config)
        assert error < 1e-5
        assert np.allclose(point[:2], world[expected_index][:2], atol=1e-5)

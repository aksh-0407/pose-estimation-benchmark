from __future__ import annotations

import numpy as np

from scripts.association.config import P3AssociationConfig
from scripts.association.geometry_cache import build_geometry_cache


def _projection(center: np.ndarray) -> np.ndarray:
    target = np.zeros(3)
    forward = target - center
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    intrinsic = np.array([[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]])
    return intrinsic @ np.hstack([rotation, (-rotation @ center).reshape(3, 1)])


def test_geometry_cache_builds_all_21_pairs_and_configured_huber_delta():
    centers = {
        f"cam_{index + 1:02d}": np.array([
            10.0 * np.cos(index * 2 * np.pi / 7),
            10.0 * np.sin(index * 2 * np.pi / 7),
            4.0,
        ])
        for index in range(7)
    }
    projections = {camera: _projection(center) for camera, center in centers.items()}
    config = P3AssociationConfig(
        baseline_angle_degen_deg=1.0,
        mu_fine_score=12.0,
        sigma_fine_score=4.0,
        degenerate_pairs=[["cam_01", "cam_02"]],
    )
    cache = build_geometry_cache(projections, config, camera_centers=centers)
    assert len(cache.pairs) == 21
    assert cache.huber_delta == 12.0 + 1.645 * 4.0
    forced = cache.pairs[("cam_01", "cam_02")]
    assert forced.is_degenerate
    assert forced.w_epi == 0.0
    assert forced.w_tri == 1.0


def test_collinear_opposing_pair_is_degenerate():
    centers = {
        "cam_01": np.array([10.0, 0.0, 4.0]),
        "cam_02": np.array([-10.0, 0.0, 4.0]),
    }
    projections = {camera: _projection(center) for camera, center in centers.items()}
    cache = build_geometry_cache(
        projections,
        P3AssociationConfig(baseline_angle_degen_deg=30.0),
        camera_centers=centers,
    )
    assert cache.pairs[("cam_01", "cam_02")].is_degenerate

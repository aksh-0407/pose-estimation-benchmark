"""Offline geometry precompute for P3: fundamental matrices, degeneracy, weights.

Built once per delivery from the calibrated 3x4 projection matrices. The
calibration "stats" that set the Huber transition come from
:class:`P3AssociationConfig`; they are fixed empirical constants because the
original auto-compute reverse-projected perfect survey points and always
returned zero error.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from identity.common.geometry import (
    camera_center_from_P,
    compute_fundamental_matrix,
    compute_right_epipole,
)
from identity.p3_association.config import P3AssociationConfig


@dataclass(frozen=True)
class PairGeometry:
    cam_id_a: str
    cam_id_b: str
    F: np.ndarray          # 3x3 fundamental matrix, x_b^T F x_a = 0
    is_degenerate: bool
    w_epi: float
    w_tri: float
    huber_delta: float


@dataclass(frozen=True)
class CalibrationStats:
    mu_fine_score: float
    sigma_fine_score: float


@dataclass(frozen=True)
class GeometryCache:
    pairs: dict[tuple[str, str], PairGeometry]
    camera_centers: dict[str, np.ndarray]
    stats: CalibrationStats
    huber_delta: float
    image_wh: tuple[int, int]


def _pair_key(cam_a: str, cam_b: str) -> tuple[str, str]:
    return (cam_a, cam_b) if cam_a <= cam_b else (cam_b, cam_a)


def _baseline_is_degenerate(C_a: np.ndarray, C_b: np.ndarray, min_angle_deg: float) -> bool:
    """Near-collinear camera centres around pitch origin imply weak baseline geometry."""
    # World origin is the calibrated pitch centre. Both same-side (angle near
    # zero) and opposing (near 180 degrees) camera pairs are collinear and can
    # be epipolar-degenerate for ground contacts.
    r_a, r_b = np.asarray(C_a, dtype=float), np.asarray(C_b, dtype=float)
    na, nb = np.linalg.norm(r_a), np.linalg.norm(r_b)
    if na < 1e-9 or nb < 1e-9:
        return True
    cos_a = float(np.clip((r_a / na) @ (r_b / nb), -1.0, 1.0))
    angle = float(np.degrees(np.arccos(cos_a)))
    return min(angle, 180.0 - angle) < min_angle_deg


def _epipole_in_image(F: np.ndarray, image_wh: tuple[int, int]) -> bool:
    w, h = image_wh
    e2 = compute_right_epipole(F)
    return bool(np.isfinite(e2).all() and 0.0 <= e2[0] <= w and 0.0 <= e2[1] <= h)


def build_geometry_cache(
    projection_matrices: dict[str, np.ndarray],
    config: P3AssociationConfig,
    *,
    camera_centers: dict[str, np.ndarray] | None = None,
    image_wh_by_cam: dict[str, tuple[int, int]] | None = None,
) -> GeometryCache:
    """Precompute per-pair fundamental matrices, degeneracy flags, and weights.

    ``image_wh_by_cam`` supplies each camera's native (width, height); the epipole
    -in-image degeneracy test uses the *right* camera's size (the right epipole
    lives in image b). Without it every camera is assumed ``config.image_wh``,
    which is wrong for C07 (~3775x960 vs the 2560x1440 default).
    """

    cam_ids = sorted(projection_matrices)
    centers = dict(camera_centers) if camera_centers else {}
    for cam_id in cam_ids:
        centers.setdefault(cam_id, camera_center_from_P(projection_matrices[cam_id]))

    forced_degenerate = {_pair_key(a, b) for a, b in config.degenerate_pairs}
    huber_delta = config.huber_delta()
    stats = CalibrationStats(config.mu_fine_score, config.sigma_fine_score)

    def _image_wh(cam_id: str) -> tuple[int, int]:
        if image_wh_by_cam and cam_id in image_wh_by_cam:
            return image_wh_by_cam[cam_id]
        return config.image_wh

    pairs: dict[tuple[str, str], PairGeometry] = {}
    for cam_a, cam_b in combinations(cam_ids, 2):
        F = compute_fundamental_matrix(projection_matrices[cam_a], projection_matrices[cam_b])
        degenerate = (
            _pair_key(cam_a, cam_b) in forced_degenerate
            or _epipole_in_image(F, _image_wh(cam_b))
            or _baseline_is_degenerate(centers[cam_a], centers[cam_b], config.baseline_angle_degen_deg)
        )
        w_epi, w_tri = (0.0, 1.0) if degenerate else (config.w_epi, config.w_tri)
        pairs[(cam_a, cam_b)] = PairGeometry(
            cam_id_a=cam_a,
            cam_id_b=cam_b,
            F=F,
            is_degenerate=degenerate,
            w_epi=w_epi,
            w_tri=w_tri,
            huber_delta=huber_delta,
        )

    return GeometryCache(
        pairs=pairs,
        camera_centers=centers,
        stats=stats,
        huber_delta=huber_delta,
        image_wh=config.image_wh,
    )

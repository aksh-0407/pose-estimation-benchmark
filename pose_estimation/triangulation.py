"""Confidence-weighted multi-view triangulation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class TriangulationResult:
    point_xyz: np.ndarray
    confidence: float
    reprojection_errors: np.ndarray
    inlier_mask: np.ndarray


def _project_point(point_xyz: np.ndarray, projection_matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.append(np.asarray(point_xyz, dtype=float), 1.0)
    projected = np.asarray(projection_matrix, dtype=float) @ homogeneous
    return projected[:2] / max(projected[2], 1e-12)


def triangulate_point_dlt(
    points_xy: np.ndarray,
    projection_matrices: np.ndarray,
    confidences: np.ndarray | None = None,
    min_views: int = 2,
) -> np.ndarray:
    """Triangulate one 3D point from 2D observations using weighted DLT."""

    points_xy = np.asarray(points_xy, dtype=float)
    projection_matrices = np.asarray(projection_matrices, dtype=float)
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise ValueError("points_xy must have shape (V, 2)")
    if projection_matrices.shape != (points_xy.shape[0], 3, 4):
        raise ValueError("projection_matrices must have shape (V, 3, 4)")
    if confidences is None:
        confidences = np.ones(points_xy.shape[0], dtype=float)
    confidences = np.asarray(confidences, dtype=float)

    valid = (
        np.isfinite(points_xy).all(axis=1)
        & np.isfinite(confidences)
        & (confidences > 0)
    )
    if int(valid.sum()) < min_views:
        return np.full(3, np.nan, dtype=float)

    rows = []
    for (x_coord, y_coord), projection, confidence in zip(points_xy[valid], projection_matrices[valid], confidences[valid]):
        weight = float(np.sqrt(max(confidence, 1e-8)))
        rows.append(weight * (x_coord * projection[2] - projection[0]))
        rows.append(weight * (y_coord * projection[2] - projection[1]))

    _, _, vt_matrix = np.linalg.svd(np.vstack(rows))
    homogeneous = vt_matrix[-1]
    if abs(homogeneous[3]) < 1e-12:
        return np.full(3, np.nan, dtype=float)
    return homogeneous[:3] / homogeneous[3]


def reprojection_errors_for_point(
    point_xyz: np.ndarray,
    points_xy: np.ndarray,
    projection_matrices: np.ndarray,
) -> np.ndarray:
    """Reproject one 3D point into all views and return pixel errors."""

    point_xyz = np.asarray(point_xyz, dtype=float)
    points_xy = np.asarray(points_xy, dtype=float)
    projection_matrices = np.asarray(projection_matrices, dtype=float)
    errors = np.full(points_xy.shape[0], np.nan, dtype=float)
    if not np.isfinite(point_xyz).all():
        return errors
    for index, (xy_obs, projection) in enumerate(zip(points_xy, projection_matrices)):
        if np.isfinite(xy_obs).all():
            xy_proj = _project_point(point_xyz, projection)
            errors[index] = float(np.linalg.norm(xy_proj - xy_obs))
    return errors


def ransac_triangulate_point(
    points_xy: np.ndarray,
    projection_matrices: np.ndarray,
    confidences: np.ndarray | None = None,
    reprojection_threshold_px: float = 10.0,
    min_views: int = 2,
) -> TriangulationResult:
    """Triangulate with pairwise RANSAC and re-fit on inlier views."""

    points_xy = np.asarray(points_xy, dtype=float)
    projection_matrices = np.asarray(projection_matrices, dtype=float)
    if confidences is None:
        confidences = np.ones(points_xy.shape[0], dtype=float)
    confidences = np.asarray(confidences, dtype=float)

    valid_indices = np.flatnonzero(
        np.isfinite(points_xy).all(axis=1)
        & np.isfinite(confidences)
        & (confidences > 0)
    )
    if len(valid_indices) < min_views:
        empty_errors = np.full(points_xy.shape[0], np.nan, dtype=float)
        return TriangulationResult(np.full(3, np.nan), 0.0, empty_errors, np.zeros(points_xy.shape[0], dtype=bool))

    best_inliers = np.zeros(points_xy.shape[0], dtype=bool)
    best_error = float("inf")
    best_point = np.full(3, np.nan)

    if len(valid_indices) == 2:
        candidate_pairs: Sequence[tuple[int, int]] = [tuple(valid_indices)]
    else:
        candidate_pairs = list(combinations(valid_indices, 2))

    for left, right in candidate_pairs:
        candidate_point = triangulate_point_dlt(
            points_xy[[left, right]],
            projection_matrices[[left, right]],
            confidences[[left, right]],
            min_views=2,
        )
        errors = reprojection_errors_for_point(candidate_point, points_xy, projection_matrices)
        inliers = np.isfinite(errors) & (errors <= reprojection_threshold_px) & (confidences > 0)
        if int(inliers.sum()) < min_views:
            continue
        mean_error = float(np.nanmean(errors[inliers]))
        if int(inliers.sum()) > int(best_inliers.sum()) or (
            int(inliers.sum()) == int(best_inliers.sum()) and mean_error < best_error
        ):
            best_inliers = inliers
            best_error = mean_error
            best_point = candidate_point

    if int(best_inliers.sum()) >= min_views:
        best_point = triangulate_point_dlt(
            points_xy[best_inliers],
            projection_matrices[best_inliers],
            confidences[best_inliers],
            min_views=min_views,
        )
    else:
        best_point = triangulate_point_dlt(points_xy, projection_matrices, confidences, min_views=min_views)
        errors = reprojection_errors_for_point(best_point, points_xy, projection_matrices)
        best_inliers = np.isfinite(errors) & (errors <= reprojection_threshold_px) & (confidences > 0)

    errors = reprojection_errors_for_point(best_point, points_xy, projection_matrices)
    if int(best_inliers.sum()) == 0:
        confidence = 0.0
    else:
        confidence = float(np.nanmean(confidences[best_inliers]))
    return TriangulationResult(best_point, confidence, errors, best_inliers)


def triangulate_skeleton_ransac(
    keypoints_by_view: np.ndarray,
    projection_matrices: np.ndarray,
    reprojection_threshold_px: float = 10.0,
    min_views: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Triangulate a skeleton from shape (V, J, 3) keypoints."""

    keypoints_by_view = np.asarray(keypoints_by_view, dtype=float)
    projection_matrices = np.asarray(projection_matrices, dtype=float)
    if keypoints_by_view.ndim != 3 or keypoints_by_view.shape[2] < 3:
        raise ValueError("keypoints_by_view must have shape (V, J, >=3)")
    if projection_matrices.shape != (keypoints_by_view.shape[0], 3, 4):
        raise ValueError("projection_matrices must have shape (V, 3, 4)")

    joint_count = keypoints_by_view.shape[1]
    points3d = np.full((joint_count, 3), np.nan, dtype=float)
    confidences = np.zeros(joint_count, dtype=float)
    mean_reprojection_errors = np.full(joint_count, np.nan, dtype=float)

    for joint_index in range(joint_count):
        result = ransac_triangulate_point(
            keypoints_by_view[:, joint_index, :2],
            projection_matrices,
            keypoints_by_view[:, joint_index, 2],
            reprojection_threshold_px=reprojection_threshold_px,
            min_views=min_views,
        )
        points3d[joint_index] = result.point_xyz
        confidences[joint_index] = result.confidence
        if np.any(result.inlier_mask):
            mean_reprojection_errors[joint_index] = float(np.nanmean(result.reprojection_errors[result.inlier_mask]))

    return points3d, confidences, mean_reprojection_errors


def confidence_ema_smooth(
    sequence_xyz: np.ndarray,
    confidences: np.ndarray | None = None,
    alpha: float = 0.65,
) -> np.ndarray:
    """Confidence-aware exponential smoothing for a T x J x 3 sequence."""

    sequence_xyz = np.asarray(sequence_xyz, dtype=float)
    if sequence_xyz.ndim != 3 or sequence_xyz.shape[2] != 3:
        raise ValueError("sequence_xyz must have shape (T, J, 3)")
    if confidences is None:
        confidences = np.ones(sequence_xyz.shape[:2], dtype=float)
    confidences = np.asarray(confidences, dtype=float)
    smoothed = sequence_xyz.copy()

    for frame_index in range(1, sequence_xyz.shape[0]):
        valid = np.isfinite(sequence_xyz[frame_index]).all(axis=1) & (confidences[frame_index] > 0)
        previous_valid = np.isfinite(smoothed[frame_index - 1]).all(axis=1)
        blend = np.clip(alpha * confidences[frame_index], 0.0, 1.0)
        update_mask = valid & previous_valid
        smoothed[frame_index, update_mask] = (
            blend[update_mask, None] * sequence_xyz[frame_index, update_mask]
            + (1.0 - blend[update_mask, None]) * smoothed[frame_index - 1, update_mask]
        )
        carry_mask = (~valid) & previous_valid
        smoothed[frame_index, carry_mask] = smoothed[frame_index - 1, carry_mask]

    return smoothed


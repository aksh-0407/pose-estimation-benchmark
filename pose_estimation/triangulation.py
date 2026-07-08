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
    # Projection matrices are homogeneous and may use either signed depth
    # convention. Clamping a valid negative denominator to +epsilon explodes
    # reprojection error and invalidates otherwise correct triangulations.
    if abs(float(projected[2])) < 1e-12:
        return np.full(2, np.nan, dtype=float)
    return projected[:2] / projected[2]


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


_COCO17_PARENT = {
    # child -> parent, for skeletal-prior extrapolation of a never-triangulated joint
    1: 0, 2: 0, 3: 1, 4: 2,          # eyes/ears <- nose
    5: 6, 6: 12, 7: 5, 8: 6,          # shoulders/elbows
    9: 7, 10: 8,                      # wrists <- elbows
    11: 12, 13: 11, 14: 12,           # hips/knees
    15: 13, 16: 14,                   # ankles <- knees
}


def fill_occluded_joints(
    sequence_xyz: np.ndarray,
    confidences: np.ndarray,
    *,
    max_gap_frames: int = 25,
    fill_confidence_scale: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill joints missing in some frames (occluded / seen in <2 views) for one identity.

    A joint triangulates only when >= 2 cameras see it, so a briefly-occluded or
    frame-edge joint is NaN in scattered frames while its neighbours are solid. This
    fills those holes **temporally** — linear interpolation between the nearest valid
    frames within ``max_gap_frames`` (hold at sequence ends) — which is the logical
    extrapolation for a joint that was and will again be observed. Filled entries carry
    reduced confidence (``fill_confidence_scale``) so downstream weights them less.

    Input/output ``(T, J, 3)`` positions and ``(T, J)`` confidences. Joints never valid
    across the whole sequence are left NaN (caller may apply a skeletal prior).
    """

    seq = np.asarray(sequence_xyz, dtype=float).copy()
    conf = np.asarray(confidences, dtype=float).copy()
    if seq.ndim != 3 or seq.shape[2] != 3:
        raise ValueError("sequence_xyz must be (T, J, 3)")
    frames, joints, _ = seq.shape

    for joint in range(joints):
        valid = np.array([bool(np.isfinite(seq[t, joint]).all()) for t in range(frames)])
        valid_idx = np.flatnonzero(valid)
        if valid_idx.size == 0:
            continue
        for t in range(frames):
            if valid[t]:
                continue
            before = valid_idx[valid_idx < t]
            after = valid_idx[valid_idx > t]
            if before.size and after.size:
                a, b = int(before[-1]), int(after[0])
                if b - a <= max_gap_frames:
                    w = (t - a) / (b - a)
                    seq[t, joint] = (1.0 - w) * seq[a, joint] + w * seq[b, joint]
                    conf[t, joint] = fill_confidence_scale * ((1.0 - w) * conf[a, joint] + w * conf[b, joint])
            elif before.size and (t - int(before[-1])) <= max_gap_frames:
                seq[t, joint] = seq[int(before[-1]), joint]  # hold last
                conf[t, joint] = fill_confidence_scale * conf[int(before[-1]), joint]
            elif after.size and (int(after[0]) - t) <= max_gap_frames:
                seq[t, joint] = seq[int(after[0]), joint]  # hold next
                conf[t, joint] = fill_confidence_scale * conf[int(after[0]), joint]
    return seq, conf


def fill_from_skeletal_prior(
    points_xyz: np.ndarray,
    confidences: np.ndarray,
    median_bone_length_m: dict[tuple[int, int], float],
    reference_pose: np.ndarray,
    *,
    fill_confidence: float = 0.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Place a still-missing joint using its parent + a bone vector from a reference pose.

    Last-resort extrapolation for a joint temporal-fill could not recover (never seen
    across the sequence in that gap): offset the (valid) parent joint by the bone vector
    taken from ``reference_pose`` (the identity's most-complete triangulated frame),
    scaled to the identity's median bone length. Very low confidence — it is a prior,
    not a measurement.
    """

    points = np.asarray(points_xyz, dtype=float).copy()
    conf = np.asarray(confidences, dtype=float).copy()
    reference = np.asarray(reference_pose, dtype=float)
    for child, parent in _COCO17_PARENT.items():
        if np.isfinite(points[child]).all():
            continue
        if not np.isfinite(points[parent]).all():
            continue
        if not (np.isfinite(reference[child]).all() and np.isfinite(reference[parent]).all()):
            continue
        bone = reference[child] - reference[parent]
        norm = float(np.linalg.norm(bone))
        target = median_bone_length_m.get((child, parent))
        if norm > 1e-6 and target:
            bone = bone / norm * target
        points[child] = points[parent] + bone
        conf[child] = fill_confidence
    return points, conf


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

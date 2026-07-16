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
    hartley: bool = False,
) -> np.ndarray:
    """Triangulate one 3D point from 2D observations using weighted DLT.

    ``hartley=True`` (G1) equilibrates each DLT row to unit norm before the SVD
    (confidence weights applied after), the single-point analogue of Hartley
    conditioning: raw rows mix pixel-scale (~1e3) and matrix-scale entries, which
    skews the least-squares toward the worst-scaled view. Off by default
    (byte-identical legacy path).
    """

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

    pts = points_xy[valid]                                  # (V, 2)
    projs = projection_matrices[valid]                      # (V, 3, 4)
    weights = np.sqrt(np.maximum(confidences[valid], 1e-8))  # (V,)
    rows_x = pts[:, 0, None] * projs[:, 2, :] - projs[:, 0, :]   # (V, 4)
    rows_y = pts[:, 1, None] * projs[:, 2, :] - projs[:, 1, :]   # (V, 4)
    if hartley:
        norm_x = np.linalg.norm(rows_x, axis=1, keepdims=True)
        norm_y = np.linalg.norm(rows_y, axis=1, keepdims=True)
        rows_x = np.where(norm_x > 1e-12, rows_x / np.where(norm_x > 1e-12, norm_x, 1.0), rows_x)
        rows_y = np.where(norm_y > 1e-12, rows_y / np.where(norm_y > 1e-12, norm_y, 1.0), rows_y)
    stacked = np.empty((rows_x.shape[0] * 2, 4), dtype=float)
    stacked[0::2] = weights[:, None] * rows_x
    stacked[1::2] = weights[:, None] * rows_y

    _, _, vt_matrix = np.linalg.svd(stacked)
    homogeneous = vt_matrix[-1]
    if abs(homogeneous[3]) < 1e-12:
        return np.full(3, np.nan, dtype=float)
    return homogeneous[:3] / homogeneous[3]


def reprojection_errors_for_point(
    point_xyz: np.ndarray,
    points_xy: np.ndarray,
    projection_matrices: np.ndarray,
) -> np.ndarray:
    """Reproject one 3D point into all views and return pixel errors.

    Vectorized (W10-PERF): one batched matmul across views replaces the
    per-view python loop; per-element arithmetic is unchanged, so results are
    bit-identical to the loop version (verified on real P3 outputs).
    """

    point_xyz = np.asarray(point_xyz, dtype=float)
    points_xy = np.asarray(points_xy, dtype=float)
    projection_matrices = np.asarray(projection_matrices, dtype=float)
    errors = np.full(points_xy.shape[0], np.nan, dtype=float)
    if not np.isfinite(point_xyz).all():
        return errors
    homogeneous = np.append(point_xyz, 1.0)
    projected = projection_matrices @ homogeneous          # (V, 3)
    depth = projected[:, 2]
    obs_ok = np.isfinite(points_xy).all(axis=1)
    depth_ok = np.abs(depth) >= 1e-12
    safe = obs_ok & depth_ok
    if np.any(safe):
        xy = projected[safe, :2] / depth[safe, None]
        errors[safe] = np.linalg.norm(xy - points_xy[safe], axis=1)
    # observation valid but degenerate depth -> NaN projection = NaN error (legacy)
    return errors


def depth_signs(point_xyz: np.ndarray, projection_matrices: np.ndarray) -> np.ndarray:
    """Sign of the projective depth of one 3D point in each view (+1 in front, -1 behind).

    Primary test: sign agreement of the homogeneous scale ``w`` with that of the
    world origin (the pitch centre — in front of every camera on this rig), which is
    invariant to ANY projection-matrix convention (scale sign, world handedness).
    Falls back to the Hartley-Zisserman ``sign(det M) * sign(w)`` formula when a
    camera sits on the origin (synthetic rigs), which assumes a proper rotation.
    """

    point_xyz = np.asarray(point_xyz, dtype=float)
    projections = np.asarray(projection_matrices, dtype=float)
    signs = np.zeros(projections.shape[0], dtype=float)
    if not np.isfinite(point_xyz).all():
        return signs
    homogeneous = np.append(point_xyz, 1.0)
    # Reference-point form: a point is in front of a camera iff its homogeneous
    # scale w shares the sign of a KNOWN in-front point's w — the convention
    # factor (matrix scale, world handedness) cancels. The world origin is the
    # pitch centre on this rig, in front of every camera; when a camera sits ON
    # the reference (w_ref ~ 0, synthetic rigs), fall back to the det(M) formula,
    # which assumes a proper (right-handed) rotation.
    reference = np.array([0.0, 0.0, 0.0, 1.0])
    for index, projection in enumerate(projections):
        w = float(projection[2] @ homogeneous)
        w_ref = float(projection[2] @ reference)
        if abs(w_ref) > 1e-9:
            signs[index] = np.sign(w) * np.sign(w_ref)
        else:
            det_m = float(np.linalg.det(projection[:, :3]))
            signs[index] = np.sign(det_m) * np.sign(w)
    return signs


def irls_huber_refit(
    point_xyz: np.ndarray,
    points_xy: np.ndarray,
    projection_matrices: np.ndarray,
    confidences: np.ndarray,
    inlier_mask: np.ndarray,
    *,
    huber_delta_px: float = 8.0,
    max_iters: int = 5,
    tol_m: float = 1e-4,
) -> np.ndarray:
    """Refine a triangulated point by IRLS with a Huber loss on reprojection residuals.

    The RANSAC inlier re-fit is an *unweighted* least-squares (L2) solve over the
    inlier views, so one view that is an inlier only just under the reprojection gate
    still pulls the point as hard as a pixel-accurate view. This applies the standard
    robust-triangulation fix (Lee & Civera 2020) — the same IRLS-Huber estimator the
    repo already ships for the ground plane (:func:`geometry.robust_fuse_ground`) — to
    the free-space per-joint solve: each view's confidence weight is multiplied by a
    Huber factor ``min(1, delta / r_i)`` on its pixel residual ``r_i`` and the DLT is
    re-run (it already applies ``sqrt(weight)`` per row, so this is one reweighted DLT
    per iteration, no new linear algebra). Operates on the inlier views only; returns
    the input point unchanged on any degeneracy (so it can never make a solve worse).
    """

    idx = np.flatnonzero(inlier_mask)
    if idx.size < 2 or not np.isfinite(point_xyz).all():
        return point_xyz
    pts = np.asarray(points_xy, dtype=float)[idx]
    projs = np.asarray(projection_matrices, dtype=float)[idx]
    base_conf = np.maximum(np.asarray(confidences, dtype=float)[idx], 1e-8)
    point = np.asarray(point_xyz, dtype=float).copy()
    for _ in range(max_iters):
        errors = reprojection_errors_for_point(point, pts, projs)
        weights = np.where(
            np.isfinite(errors),
            np.where(errors <= huber_delta_px, 1.0, huber_delta_px / np.maximum(errors, 1e-9)),
            0.0,
        )
        refined = triangulate_point_dlt(pts, projs, base_conf * weights, min_views=2)
        if not np.isfinite(refined).all():
            return point
        step = float(np.linalg.norm(refined - point))
        point = refined
        if step < tol_m:
            break
    return point


def ransac_triangulate_point(
    points_xy: np.ndarray,
    projection_matrices: np.ndarray,
    confidences: np.ndarray | None = None,
    reprojection_threshold_px: float = 10.0,
    min_views: int = 2,
    cheirality: bool = False,
    hartley: bool = False,
    parallax_order: bool = False,
    robust_refit: bool = False,
    robust_huber_px: float = 8.0,
) -> TriangulationResult:
    """Triangulate with pairwise RANSAC and re-fit on inlier views.

    ``cheirality=True`` additionally requires the point to lie in front of a view for
    that view to count as an inlier, and discards candidate solutions behind either
    seed camera (flag-gated: off reproduces the legacy behaviour byte-for-byte).
    ``hartley=True`` (G1) row-equilibrates the DLT systems. ``parallax_order=True``
    (G3) tries high-parallax seed pairs first, so exact inlier/error ties resolve
    toward the best-conditioned geometry instead of camera-index order. Both flags
    off = legacy byte-identical.
    """

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
    if parallax_order and len(candidate_pairs) > 1:
        # G3: rank seed pairs by the angle between their viewing rays through the
        # observations (via a provisional 2-view point). High parallax first.
        def _pair_parallax(pair: tuple[int, int]) -> float:
            left, right = pair
            provisional = triangulate_point_dlt(
                points_xy[[left, right]], projection_matrices[[left, right]],
                confidences[[left, right]], min_views=2, hartley=hartley,
            )
            if not np.isfinite(provisional).all():
                return -1.0
            angle = 0.0
            try:
                centers = []
                for proj in projection_matrices[[left, right]]:
                    m, p4 = proj[:, :3], proj[:, 3]
                    centers.append(-np.linalg.solve(m, p4))
                v1 = provisional - centers[0]
                v2 = provisional - centers[1]
                denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
                if denom > 1e-9:
                    angle = float(np.arccos(np.clip(v1 @ v2 / denom, -1.0, 1.0)))
            except np.linalg.LinAlgError:
                return -1.0
            return angle
        candidate_pairs = sorted(candidate_pairs, key=_pair_parallax, reverse=True)

    for left, right in candidate_pairs:
        candidate_point = triangulate_point_dlt(
            points_xy[[left, right]],
            projection_matrices[[left, right]],
            confidences[[left, right]],
            min_views=2,
            hartley=hartley,
        )
        if cheirality and np.isfinite(candidate_point).all():
            if np.any(depth_signs(candidate_point, projection_matrices[[left, right]]) <= 0):
                continue  # reconstruction behind a seed camera is geometrically invalid
        errors = reprojection_errors_for_point(candidate_point, points_xy, projection_matrices)
        inliers = np.isfinite(errors) & (errors <= reprojection_threshold_px) & (confidences > 0)
        if cheirality:
            inliers &= depth_signs(candidate_point, projection_matrices) > 0
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
            hartley=hartley,
        )
    else:
        best_point = triangulate_point_dlt(
            points_xy, projection_matrices, confidences, min_views=min_views, hartley=hartley
        )
        errors = reprojection_errors_for_point(best_point, points_xy, projection_matrices)
        best_inliers = np.isfinite(errors) & (errors <= reprojection_threshold_px) & (confidences > 0)
        if cheirality:
            best_inliers &= depth_signs(best_point, projection_matrices) > 0

    if robust_refit and int(best_inliers.sum()) >= min_views and np.isfinite(best_point).all():
        # M-estimator polish over the inlier views (down-weights marginal-inlier
        # cameras); flag-gated, off = the legacy unweighted inlier re-fit above.
        best_point = irls_huber_refit(
            best_point, points_xy, projection_matrices, confidences, best_inliers,
            huber_delta_px=robust_huber_px,
        )

    errors = reprojection_errors_for_point(best_point, points_xy, projection_matrices)
    if int(best_inliers.sum()) == 0:
        confidence = 0.0
    else:
        confidence = float(np.nanmean(confidences[best_inliers]))
    return TriangulationResult(best_point, confidence, errors, best_inliers)


def point_covariance_3d(
    point_xyz: np.ndarray,
    points_xy: np.ndarray,
    projection_matrices: np.ndarray,
    confidences: np.ndarray | None = None,
    inlier_mask: np.ndarray | None = None,
) -> np.ndarray | None:
    """3x3 covariance of a triangulated point, linearized at the solution (F9b).

    ``cov = sigma^2 (J^T W J)^{-1}`` with J the stacked 2x3 reprojection Jacobians of
    the inlier views and ``sigma^2`` the weighted residual variance — the standard
    first-order uncertainty of a multi-view triangulation. Elongated along the ray
    direction on low-parallax (facing-pair) view sets, which is exactly the signal a
    downstream consumer needs to distrust depth there. None when under-determined.
    """

    point_xyz = np.asarray(point_xyz, dtype=float)
    points_xy = np.asarray(points_xy, dtype=float)
    projections = np.asarray(projection_matrices, dtype=float)
    if not np.isfinite(point_xyz).all():
        return None
    n = points_xy.shape[0]
    if confidences is None:
        confidences = np.ones(n, dtype=float)
    confidences = np.asarray(confidences, dtype=float)
    if inlier_mask is None:
        inlier_mask = np.isfinite(points_xy).all(axis=1)
    homogeneous_point = np.append(point_xyz, 1.0)

    JTJ = np.zeros((3, 3), dtype=float)
    weighted_ssr = 0.0
    views = 0
    for i in range(n):
        if not inlier_mask[i] or not np.isfinite(points_xy[i]).all():
            continue
        P = projections[i]
        h = P @ homogeneous_point
        if abs(h[2]) < 1e-9:
            continue
        projected = h[:2] / h[2]
        residual = projected - points_xy[i]
        jac = np.zeros((2, 3), dtype=float)
        for axis in range(3):
            column = P[:, axis]
            jac[:, axis] = (column[:2] * h[2] - h[:2] * column[2]) / (h[2] ** 2)
        weight = float(max(confidences[i], 1e-3))
        JTJ += weight * jac.T @ jac
        weighted_ssr += weight * float(residual @ residual)
        views += 1
    if views < 2:
        return None
    dof = max(2 * views - 3, 1)
    sigma_sq = max(weighted_ssr / dof, 1e-6)
    try:
        cov = sigma_sq * np.linalg.inv(JTJ + 1e-9 * np.eye(3))
    except np.linalg.LinAlgError:
        return None
    return cov if np.isfinite(cov).all() else None


def _skeleton_two_view_batched(
    keypoints_by_view: np.ndarray,
    projection_matrices: np.ndarray,
    reprojection_threshold_px: float,
    cheirality: bool,
    hartley: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact 2-view fast path (W10-PERF): for V==2 the pairwise RANSAC collapses
    analytically to one DLT per joint (candidate == refit == fallback), so ALL
    joints are solved in a single batched SVD. Bit-identical to the per-joint
    path (same row order, same LAPACK kernel per matrix; verified on real data).
    """

    kpv = keypoints_by_view
    projs = projection_matrices
    joint_count = kpv.shape[1]
    points3d = np.full((joint_count, 3), np.nan, dtype=float)
    confidences_out = np.zeros(joint_count, dtype=float)
    mean_errors = np.full(joint_count, np.nan, dtype=float)

    pts = kpv[:, :, :2]
    conf = kpv[:, :, 2]
    valid = np.isfinite(pts).all(axis=2) & np.isfinite(conf) & (conf > 0)  # (2, J)
    both = valid.all(axis=0)
    idx = np.flatnonzero(both)
    if idx.size == 0:
        return points3d, confidences_out, mean_errors

    weights = np.sqrt(np.maximum(conf[:, idx], 1e-8))       # (2, K)
    systems = np.empty((idx.size, 4, 4), dtype=float)
    for view in range(2):
        rows_x = pts[view, idx, 0, None] * projs[view, 2, :] - projs[view, 0, :]
        rows_y = pts[view, idx, 1, None] * projs[view, 2, :] - projs[view, 1, :]
        if hartley:
            norm_x = np.linalg.norm(rows_x, axis=1, keepdims=True)
            norm_y = np.linalg.norm(rows_y, axis=1, keepdims=True)
            rows_x = np.where(norm_x > 1e-12, rows_x / np.where(norm_x > 1e-12, norm_x, 1.0), rows_x)
            rows_y = np.where(norm_y > 1e-12, rows_y / np.where(norm_y > 1e-12, norm_y, 1.0), rows_y)
        systems[:, 2 * view] = weights[view][:, None] * rows_x
        systems[:, 2 * view + 1] = weights[view][:, None] * rows_y
    _, _, vt = np.linalg.svd(systems)
    hom = vt[:, -1, :]                                       # (K, 4)
    scale_ok = np.abs(hom[:, 3]) >= 1e-12
    sub_points = np.full((idx.size, 3), np.nan, dtype=float)
    sub_points[scale_ok] = hom[scale_ok, :3] / hom[scale_ok, 3:4]

    hom_points = np.concatenate([sub_points, np.ones((idx.size, 1))], axis=1)
    projected = np.einsum("vij,kj->vki", projs, hom_points)   # (2, K, 3)
    depth = projected[:, :, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        proj_xy = np.where(
            np.abs(depth)[:, :, None] >= 1e-12,
            projected[:, :, :2] / np.where(np.abs(depth) >= 1e-12, depth, 1.0)[:, :, None],
            np.nan,
        )
        errors_vk = np.linalg.norm(proj_xy - pts[:, idx], axis=2)  # (2, K)
    point_finite = np.isfinite(sub_points).all(axis=1)
    errors_vk = np.where(point_finite[None, :], errors_vk, np.nan)

    inliers_vk = np.isfinite(errors_vk) & (errors_vk <= reprojection_threshold_px) & (conf[:, idx] > 0)
    if cheirality:
        reference = np.array([0.0, 0.0, 0.0, 1.0])
        w_ref = projs[:, 2, :] @ reference                    # (2,)
        w = depth                                             # (2, K)
        signs = np.zeros_like(w)
        for view in range(2):
            if abs(float(w_ref[view])) > 1e-9:
                signs[view] = np.sign(w[view]) * np.sign(w_ref[view])
            else:
                det_m = float(np.linalg.det(projs[view][:, :3]))
                signs[view] = np.sign(det_m) * np.sign(w[view])
        signs = np.where(point_finite[None, :], signs, 0.0)
        inliers_vk &= signs > 0

    inlier_any = inliers_vk.any(axis=0)
    counts = inliers_vk.sum(axis=0)
    safe_counts = np.maximum(counts, 1)
    mean_conf = np.where(inliers_vk, conf[:, idx], 0.0).sum(axis=0) / safe_counts
    mean_err = np.where(inliers_vk, np.nan_to_num(errors_vk, nan=0.0), 0.0).sum(axis=0) / safe_counts
    points3d[idx] = sub_points
    confidences_out[idx] = np.where(inlier_any, mean_conf, 0.0)
    mean_errors[idx] = np.where(inlier_any, mean_err, np.nan)
    return points3d, confidences_out, mean_errors


def _batched_pair_dlt(pts_pair, projs_pair, conf_pair, hartley):
    """DLT for K joints from one camera pair: pts (2,K,2), projs (2,3,4), conf (2,K)."""

    K = pts_pair.shape[1]
    weights = np.sqrt(np.maximum(conf_pair, 1e-8))            # (2, K)
    systems = np.empty((K, 4, 4), dtype=float)
    for view in range(2):
        rows_x = pts_pair[view, :, 0, None] * projs_pair[view, 2, :] - projs_pair[view, 0, :]
        rows_y = pts_pair[view, :, 1, None] * projs_pair[view, 2, :] - projs_pair[view, 1, :]
        if hartley:
            norm_x = np.linalg.norm(rows_x, axis=1, keepdims=True)
            norm_y = np.linalg.norm(rows_y, axis=1, keepdims=True)
            rows_x = np.where(norm_x > 1e-12, rows_x / np.where(norm_x > 1e-12, norm_x, 1.0), rows_x)
            rows_y = np.where(norm_y > 1e-12, rows_y / np.where(norm_y > 1e-12, norm_y, 1.0), rows_y)
        systems[:, 2 * view] = weights[view][:, None] * rows_x
        systems[:, 2 * view + 1] = weights[view][:, None] * rows_y
    _, _, vt = np.linalg.svd(systems)
    hom = vt[:, -1, :]
    ok = np.abs(hom[:, 3]) >= 1e-12
    out = np.full((K, 3), np.nan, dtype=float)
    out[ok] = hom[ok, :3] / hom[ok, 3:4]
    return out


def _batched_errors(points3d, pts, projs):
    """Reprojection errors for K points into V views: points (K,3), pts (V,K,2)."""

    K = points3d.shape[0]
    hom_points = np.concatenate([points3d, np.ones((K, 1))], axis=1)   # (K,4)
    projected = np.einsum("vij,kj->vki", projs, hom_points)            # (V,K,3)
    depth = projected[:, :, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        proj_xy = np.where(
            np.abs(depth)[:, :, None] >= 1e-12,
            projected[:, :, :2] / np.where(np.abs(depth) >= 1e-12, depth, 1.0)[:, :, None],
            np.nan,
        )
        errors = np.linalg.norm(proj_xy - pts, axis=2)                 # (V,K)
    point_ok = np.isfinite(points3d).all(axis=1)
    return np.where(point_ok[None, :], errors, np.nan), depth


def _batched_depth_signs(depth, projs, point_ok):
    """depth_signs for K points across V views, replicating the reference test."""

    reference = np.array([0.0, 0.0, 0.0, 1.0])
    w_ref = projs[:, 2, :] @ reference                                  # (V,)
    signs = np.zeros_like(depth)
    for view in range(projs.shape[0]):
        if abs(float(w_ref[view])) > 1e-9:
            signs[view] = np.sign(depth[view]) * np.sign(w_ref[view])
        else:
            det_m = float(np.linalg.det(projs[view][:, :3]))
            signs[view] = np.sign(det_m) * np.sign(depth[view])
    return np.where(point_ok[None, :], signs, 0.0)


def _skeleton_multi_view_batched(
    kpv: np.ndarray,
    projs: np.ndarray,
    reprojection_threshold_px: float,
    min_views: int,
    cheirality: bool,
    hartley: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact batched replica of the per-joint pairwise RANSAC (W10-PERF).

    Candidate pairs are evaluated in the same lexicographic order as
    ``combinations(valid_indices, 2)`` per joint; the best-candidate update rule,
    the inlier-refit, the <min_views fallback and the confidence computation all
    replicate ransac_triangulate_point bit-for-bit (verified on real P3/P6 data).
    """

    V, J = kpv.shape[0], kpv.shape[1]
    pts = kpv[:, :, :2]
    conf = kpv[:, :, 2]
    valid = np.isfinite(pts).all(axis=2) & np.isfinite(conf) & (conf > 0)   # (V,J)
    n_valid = valid.sum(axis=0)

    points3d = np.full((J, 3), np.nan, dtype=float)
    confidences_out = np.zeros(J, dtype=float)
    mean_errors = np.full(J, np.nan, dtype=float)

    solvable = n_valid >= min_views
    if not np.any(solvable):
        return points3d, confidences_out, mean_errors

    best_count = np.zeros(J, dtype=int)
    best_err = np.full(J, np.inf, dtype=float)
    best_inliers = np.zeros((V, J), dtype=bool)
    conf_pos = conf > 0

    for a, b in combinations(range(V), 2):
        active = solvable & valid[a] & valid[b]
        idx = np.flatnonzero(active)
        if idx.size == 0:
            continue
        cand = _batched_pair_dlt(
            pts[[a, b]][:, idx], projs[[a, b]], conf[[a, b]][:, idx], hartley
        )
        errors, depth_all = _batched_errors(cand, pts[:, idx], projs)
        point_ok = np.isfinite(cand).all(axis=1)
        if cheirality:
            seed_depth = np.stack([depth_all[a], depth_all[b]])
            seed_signs = _batched_depth_signs(seed_depth, projs[[a, b]], point_ok)
            seed_bad = point_ok & (seed_signs <= 0).any(axis=0)
        else:
            seed_bad = np.zeros(idx.size, dtype=bool)
        inl = np.isfinite(errors) & (errors <= reprojection_threshold_px) & conf_pos[:, idx]
        if cheirality:
            all_signs = _batched_depth_signs(depth_all, projs, point_ok)
            inl &= all_signs > 0
        counts = inl.sum(axis=0)
        with np.errstate(invalid="ignore"):
            errs_sum = np.where(inl, np.nan_to_num(errors, nan=0.0), 0.0).sum(axis=0)
        mean_e = np.where(counts > 0, errs_sum / np.maximum(counts, 1), np.inf)
        viable = (~seed_bad) & (counts >= min_views)
        take = viable & (
            (counts > best_count[idx])
            | ((counts == best_count[idx]) & (mean_e < best_err[idx]))
        )
        upd = idx[take]
        best_count[upd] = counts[take]
        best_err[upd] = mean_e[take]
        best_inliers[:, upd] = inl[:, take]

    # Final refit / fallback, grouped by identical view subsets for batching.
    final_points = np.full((J, 3), np.nan, dtype=float)
    refit_mask = solvable & (best_count >= min_views)
    fallback_mask = solvable & ~refit_mask

    def _grouped_dlt(joint_idx, view_masks):
        groups: dict[bytes, list[int]] = {}
        for j in joint_idx:
            groups.setdefault(view_masks[:, j].tobytes(), []).append(j)
        for key, joints in groups.items():
            views = np.frombuffer(key, dtype=bool)
            vsel = np.flatnonzero(views)
            joints = np.asarray(joints)
            if vsel.size < 2:
                # single-view system: legacy DLT returns NaN via min_views guard
                continue
            weights = np.sqrt(np.maximum(conf[np.ix_(vsel, joints)], 1e-8))
            systems = np.empty((joints.size, 2 * vsel.size, 4), dtype=float)
            for row, v in enumerate(vsel):
                rows_x = pts[v, joints, 0, None] * projs[v, 2, :] - projs[v, 0, :]
                rows_y = pts[v, joints, 1, None] * projs[v, 2, :] - projs[v, 1, :]
                if hartley:
                    norm_x = np.linalg.norm(rows_x, axis=1, keepdims=True)
                    norm_y = np.linalg.norm(rows_y, axis=1, keepdims=True)
                    rows_x = np.where(norm_x > 1e-12, rows_x / np.where(norm_x > 1e-12, norm_x, 1.0), rows_x)
                    rows_y = np.where(norm_y > 1e-12, rows_y / np.where(norm_y > 1e-12, norm_y, 1.0), rows_y)
                systems[:, 2 * row] = weights[row][:, None] * rows_x
                systems[:, 2 * row + 1] = weights[row][:, None] * rows_y
            _, _, vt = np.linalg.svd(systems)
            hom = vt[:, -1, :]
            ok = np.abs(hom[:, 3]) >= 1e-12
            pts3 = np.full((joints.size, 3), np.nan, dtype=float)
            pts3[ok] = hom[ok, :3] / hom[ok, 3:4]
            final_points[joints] = pts3

    _grouped_dlt(np.flatnonzero(refit_mask), best_inliers)
    _grouped_dlt(np.flatnonzero(fallback_mask), valid)

    # Final errors + (for fallback) recomputed inliers, then confidence.
    idx_all = np.flatnonzero(solvable)
    errors_all, depth_all = _batched_errors(final_points[idx_all], pts[:, idx_all], projs)
    point_ok_all = np.isfinite(final_points[idx_all]).all(axis=1)
    final_inliers = best_inliers[:, idx_all]
    fb_local = fallback_mask[idx_all]
    if np.any(fb_local):
        fb_inl = (
            np.isfinite(errors_all) & (errors_all <= reprojection_threshold_px)
            & conf_pos[:, idx_all]
        )
        if cheirality:
            all_signs = _batched_depth_signs(depth_all, projs, point_ok_all)
            fb_inl &= all_signs > 0
        final_inliers = np.where(fb_local[None, :], fb_inl, final_inliers)

    counts = final_inliers.sum(axis=0)
    safe = np.maximum(counts, 1)
    conf_mean = np.where(final_inliers, conf[:, idx_all], 0.0).sum(axis=0) / safe
    err_mean = np.where(final_inliers, np.nan_to_num(errors_all, nan=0.0), 0.0).sum(axis=0) / safe

    points3d[idx_all] = final_points[idx_all]
    confidences_out[idx_all] = np.where(counts > 0, conf_mean, 0.0)
    mean_errors[idx_all] = np.where(counts > 0, err_mean, np.nan)
    return points3d, confidences_out, mean_errors


def triangulate_skeleton_ransac(
    keypoints_by_view: np.ndarray,
    projection_matrices: np.ndarray,
    reprojection_threshold_px: float = 10.0,
    min_views: int = 2,
    cheirality: bool = False,
    hartley: bool = False,
    parallax_order: bool = False,
    robust_refit: bool = False,
    robust_huber_px: float = 8.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Triangulate a skeleton from shape (V, J, 3) keypoints.

    ``hartley``/``parallax_order`` (G1/G3, flag-gated off = byte-identical) are
    forwarded per joint to :func:`ransac_triangulate_point`. Batched fast paths
    (W10-PERF, bit-identical) handle the 2-view and the generic multi-view cases;
    ``parallax_order=True`` or ``robust_refit=True`` fall back to the per-joint
    reference loop (the IRLS-Huber polish lives only in that path, so the batched
    kernels stay untouched and the flag-off dispatch is bit-identical to today).
    """

    keypoints_by_view = np.asarray(keypoints_by_view, dtype=float)
    projection_matrices = np.asarray(projection_matrices, dtype=float)
    if (
        keypoints_by_view.ndim == 3
        and keypoints_by_view.shape[0] == 2
        and keypoints_by_view.shape[2] >= 3
        and min_views <= 2
        and not robust_refit
        and projection_matrices.shape == (2, 3, 4)
    ):
        return _skeleton_two_view_batched(
            keypoints_by_view[:, :, :3], projection_matrices,
            reprojection_threshold_px, cheirality, hartley,
        )
    if (
        keypoints_by_view.ndim == 3
        and keypoints_by_view.shape[0] >= 3
        and keypoints_by_view.shape[2] >= 3
        and not parallax_order
        and not robust_refit
        and projection_matrices.shape == (keypoints_by_view.shape[0], 3, 4)
    ):
        return _skeleton_multi_view_batched(
            keypoints_by_view[:, :, :3], projection_matrices,
            reprojection_threshold_px, min_views, cheirality, hartley,
        )
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
            cheirality=cheirality,
            hartley=hartley,
            parallax_order=parallax_order,
            robust_refit=robust_refit,
            robust_huber_px=robust_huber_px,
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

# Halpe-26 = COCO-17 + head/neck/hip + big toes/small toes/heels (F15: the feet are
# true ground-contact landmarks, so triangulating them tightens the 3D ground story).
_HALPE26_PARENT = {
    **_COCO17_PARENT,
    17: 0, 18: 0, 19: 11,             # head <- nose, neck <- nose, hip-mid <- l_hip
    20: 15, 22: 15, 24: 15,           # left big/small toe + heel <- left ankle
    21: 16, 23: 16, 25: 16,           # right big/small toe + heel <- right ankle
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
    parents: dict[int, int] | None = None,
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
    joint_count = points.shape[0]
    for child, parent in (parents or _COCO17_PARENT).items():
        if child >= joint_count or parent >= joint_count:
            continue
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


def butterworth_smooth(
    sequence_xyz: np.ndarray,
    *,
    fps: float = 50.0,
    cutoff_hz: float = 6.0,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase low-pass (Butterworth + filtfilt) over a T x J x 3 sequence.

    The offline / export-quality alternative to the causal EMA: no phase lag, so a
    whole-delivery trajectory keeps its timing while frame-to-frame noise above
    ``cutoff_hz`` is removed (the sports-capture standard, cf. Pose2Sim). Applied per
    joint per axis over each contiguous finite segment; segments too short for the
    filter's padding are left untouched. NaN gaps are preserved, never bridged.
    """

    from scipy.signal import butter, filtfilt

    sequence_xyz = np.asarray(sequence_xyz, dtype=float)
    if sequence_xyz.ndim != 3 or sequence_xyz.shape[2] != 3:
        raise ValueError("sequence_xyz must have shape (T, J, 3)")
    nyquist = 0.5 * fps
    if not 0.0 < cutoff_hz < nyquist:
        raise ValueError(f"cutoff_hz must be in (0, {nyquist})")
    b, a = butter(order, cutoff_hz / nyquist, btype="low")
    pad = 3 * max(len(a), len(b))  # filtfilt's default padlen

    smoothed = sequence_xyz.copy()
    frames, joints, _ = sequence_xyz.shape
    for joint in range(joints):
        finite = np.isfinite(sequence_xyz[:, joint]).all(axis=1)
        start = None
        for t in range(frames + 1):
            inside = t < frames and finite[t]
            if inside and start is None:
                start = t
            elif not inside and start is not None:
                if t - start > pad:
                    segment = sequence_xyz[start:t, joint]
                    smoothed[start:t, joint] = filtfilt(b, a, segment, axis=0)
                start = None
    return smoothed


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

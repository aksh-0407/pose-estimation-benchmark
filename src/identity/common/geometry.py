"""Multi-view geometry primitives for cross-camera association (P3) and lifting.

Stateless math shared by the association engine, the global-ID tracker, and the
3D lift. Two-view triangulation delegates to :mod:`identity.common.triangulation`
(the repo's weighted-DLT/RANSAC home) rather than re-implementing the SVD solve.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from identity.common.triangulation import triangulate_point_dlt


def camera_center_from_P(projection_matrix: np.ndarray) -> np.ndarray:
    """World camera centre C (the right null space of the 3x4 projection matrix)."""

    _, _, vt = np.linalg.svd(np.asarray(projection_matrix, dtype=float))
    center_h = vt[-1]
    if abs(center_h[3]) < 1e-12:
        return np.full(3, np.nan)
    return center_h[:3] / center_h[3]


def triangulate_dlt(
    x1_px: np.ndarray,
    P1: np.ndarray,
    x2_px: np.ndarray,
    P2: np.ndarray,
) -> np.ndarray:
    """Two-view DLT triangulation → world ``[X, Y, Z]`` (or NaNs if degenerate).

    Thin wrapper over :func:`identity.common.triangulation.triangulate_point_dlt`
    so there is a single DLT implementation in the codebase.
    """

    points_xy = np.array([np.asarray(x1_px, float)[:2], np.asarray(x2_px, float)[:2]], dtype=float)
    projections = np.array([np.asarray(P1, float), np.asarray(P2, float)], dtype=float)
    return triangulate_point_dlt(points_xy, projections, min_views=2)


def reprojection_error_px(
    X_world: np.ndarray,
    P: np.ndarray,
    x_obs_px: np.ndarray,
) -> float:
    """Pixel distance between an observed point and a 3D point reprojected by ``P``."""

    homogeneous = np.asarray(P, float) @ np.append(np.asarray(X_world, float), 1.0)
    if abs(homogeneous[2]) < 1e-12:
        return float("inf")
    projected = homogeneous[:2] / homogeneous[2]
    return float(np.linalg.norm(projected - np.asarray(x_obs_px, float)))


def condition_number_dlt(
    x1_px: np.ndarray,
    P1: np.ndarray,
    x2_px: np.ndarray,
    P2: np.ndarray,
) -> float:
    """Condition number of the two-view DLT system (higher = worse-conditioned)."""

    x1_px = np.asarray(x1_px, float)
    x2_px = np.asarray(x2_px, float)
    P1 = np.asarray(P1, float)
    P2 = np.asarray(P2, float)
    A = np.array([
        x1_px[0] * P1[2] - P1[0],
        x1_px[1] * P1[2] - P1[1],
        x2_px[0] * P2[2] - P2[0],
        x2_px[1] * P2[2] - P2[1],
    ])
    singular = np.linalg.svd(A, compute_uv=False)
    return float(singular[0] / (singular[-1] + 1e-12))


def parallax_angle_deg(
    C1_world: np.ndarray,
    C2_world: np.ndarray,
    X_world: np.ndarray,
) -> float:
    """Angle (degrees) between the two camera rays meeting at a 3D point."""

    r1 = np.asarray(C1_world, float) - np.asarray(X_world, float)
    r2 = np.asarray(C2_world, float) - np.asarray(X_world, float)
    n1, n2 = np.linalg.norm(r1), np.linalg.norm(r2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos_a = float(np.clip((r1 / n1) @ (r2 / n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


def compute_fundamental_matrix(P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    """Fundamental matrix F such that ``x2^T F x1 = 0`` from two 3x4 matrices."""

    P1 = np.asarray(P1, float)
    P2 = np.asarray(P2, float)
    C1 = camera_center_from_P(P1)
    e2 = P2 @ np.append(C1, 1.0)
    e2_cross = np.array([
        [0.0, -e2[2], e2[1]],
        [e2[2], 0.0, -e2[0]],
        [-e2[1], e2[0], 0.0],
    ])
    F = e2_cross @ P2 @ np.linalg.pinv(P1)
    norm = np.linalg.norm(F)
    return F / (norm + 1e-12)


def compute_right_epipole(F: np.ndarray) -> np.ndarray:
    """Right epipole e2 (``F^T e2 = 0``) as dehomogenized ``[u, v]`` pixels."""

    _, _, vt = np.linalg.svd(np.asarray(F, float).T)
    e = vt[-1]
    if abs(e[2]) < 1e-12:
        return np.array([np.inf, np.inf])
    return e[:2] / e[2]


def sampson_distance(x1_px: np.ndarray, F: np.ndarray, x2_px: np.ndarray) -> float:
    """Symmetric Sampson distance (first-order epipolar geometric error, in px^2)."""

    x1h = np.array([x1_px[0], x1_px[1], 1.0], dtype=float)
    x2h = np.array([x2_px[0], x2_px[1], 1.0], dtype=float)
    F = np.asarray(F, float)
    Fx1 = F @ x1h
    Ftx2 = F.T @ x2h
    numerator = float(x2h @ Fx1) ** 2
    denominator = Fx1[0] ** 2 + Fx1[1] ** 2 + Ftx2[0] ** 2 + Ftx2[1] ** 2
    return numerator / (denominator + 1e-12)


def bbox_bottom_center_px(bbox_xywh_px: list[float]) -> np.ndarray:
    """``[u, v]`` of the bbox bottom-centre — the ground-contact reference point."""

    x, y, w, h = bbox_xywh_px
    return np.array([x + w / 2.0, y + h], dtype=float)


# Halpe-26 foot keypoint indices (pose_2d): heels sit on the ground and are
# the best ground-contact landmarks the pose model provides.
_HALPE_LEFT_FOOT = (24, 20)    # (heel, big toe)
_HALPE_RIGHT_FOOT = (25, 21)


def ground_contact_pixel_ex(
    bbox_xywh_px: list[float],
    keypoints_px: np.ndarray,
    keypoint_confidence: np.ndarray,
    *,
    ankle_confidence_min: float = 0.6,
    max_ankle_above_bbox_fraction: float = 0.25,
    mode: str = "legacy",
    ankle_height_m: float = 0.10,
    horizontal_margin_frac: float = 0.15,
    level_frac: float = 0.15,
    native_keypoints_px: np.ndarray | None = None,
    native_confidence: np.ndarray | None = None,
    foot_kp_conf_min: float = 0.5,
    foot_height_m: float = 0.02,
) -> tuple[np.ndarray, float, str]:
    """Foot contact pixel + the world height of the landmark used + its source.

    Returns ``(pixel, height_m, source)`` where ``height_m`` is the metric height of
    the projected landmark above the ground (so a caller can back-project the ankle
    onto the ``z = height_m`` plane instead of ``z = 0``, removing the ~10 cm
    ankle-above-ground bias, F2) and ``source`` is one of ``ankle_mid`` /
    ``ankle_planted`` / ``bbox_bottom``.

    ``mode="legacy"`` reproduces the historical behaviour exactly (lower confident
    ankle, else bbox bottom-centre; ``height_m`` always 0). ``mode="v2"`` fixes the
    foot-pixel defects (F3 tighter + horizontal plausibility, F4/F6 midpoint as the
    cross-camera-consistent reference, F2 ankle height reported). ``mode="v3"``
    (fix F4 of the critical-analysis campaign) prefers the Halpe-26 heel/toe
    keypoints from ``native_keypoints_px`` — actual ground-contact landmarks,
    ~2 cm above ground vs the ankle's ~10 cm — falling back to the v2 ankle stack
    when the foot keypoints are missing or unconfident.
    """

    bbox = np.asarray(bbox_xywh_px, dtype=float)
    points = np.asarray(keypoints_px, dtype=float)
    confidence = np.asarray(keypoint_confidence, dtype=float)
    bottom = bbox_bottom_center_px(list(bbox))
    if bbox.shape != (4,) or points.shape != (17, 2) or confidence.shape != (17,):
        return bottom, 0.0, "bbox_bottom"
    if not np.isfinite(bbox).all() or bbox[2] <= 0.0 or bbox[3] <= 0.0:
        return bottom, 0.0, "bbox_bottom"

    tolerance = max(20.0, max_ankle_above_bbox_fraction * float(bbox[3]))

    def vertically_ok(point: np.ndarray) -> bool:
        return bottom[1] - tolerance <= point[1] <= bottom[1] + 0.1 * bbox[3]

    if mode not in ("v2", "v3"):
        plausible = [
            points[i]
            for i in (15, 16)
            if np.isfinite(points[i]).all()
            and np.isfinite(confidence[i])
            and float(confidence[i]) >= ankle_confidence_min
            and vertically_ok(points[i])
        ]
        if not plausible:
            return bottom, 0.0, "bbox_bottom"
        if len(plausible) == 2 and abs(float(plausible[0][1] - plausible[1][1])) <= 0.05 * bbox[3]:
            return np.mean(np.asarray(plausible, dtype=float), axis=0), 0.0, "ankle_mid"
        return np.asarray(max(plausible, key=lambda p: float(p[1])), dtype=float).copy(), 0.0, "ankle_planted"

    # --- v2 / v3 share the plausibility window --------------------------------
    x_min = float(bbox[0]) - horizontal_margin_frac * float(bbox[2])
    x_max = float(bbox[0]) + (1.0 + horizontal_margin_frac) * float(bbox[2])

    if mode == "v3" and native_keypoints_px is not None and native_confidence is not None:
        native_pts = np.asarray(native_keypoints_px, dtype=float)
        native_conf = np.asarray(native_confidence, dtype=float)
        if native_pts.shape[0] >= 26 and native_conf.shape[0] >= 26:

            def foot_point(heel_idx: int, toe_idx: int) -> np.ndarray | None:
                candidates = []
                for idx in (heel_idx, toe_idx):
                    p = native_pts[idx]
                    if (
                        np.isfinite(p).all()
                        and np.isfinite(native_conf[idx])
                        and float(native_conf[idx]) >= foot_kp_conf_min
                        and vertically_ok(p)
                        and x_min <= float(p[0]) <= x_max
                    ):
                        candidates.append(p)
                if not candidates:
                    return None
                return np.mean(np.asarray(candidates, dtype=float), axis=0)

            left = foot_point(*_HALPE_LEFT_FOOT)
            right = foot_point(*_HALPE_RIGHT_FOOT)
            if left is not None and right is not None:
                if abs(float(left[1] - right[1])) <= level_frac * float(bbox[3]):
                    return np.mean([left, right], axis=0), foot_height_m, "foot_mid"
                planted = left if float(left[1]) > float(right[1]) else right
                return np.asarray(planted, dtype=float).copy(), foot_height_m, "foot_planted"
            if left is not None or right is not None:
                only = left if left is not None else right
                return np.asarray(only, dtype=float).copy(), foot_height_m, "foot_planted"
        # no usable foot keypoints -> fall through to the v2 ankle stack

    def plausible_ankle(i: int) -> bool:
        p = points[i]
        return (
            np.isfinite(p).all()
            and np.isfinite(confidence[i])
            and float(confidence[i]) >= ankle_confidence_min
            and vertically_ok(p)                 # F3 vertical
            and x_min <= float(p[0]) <= x_max     # F3 horizontal (new)
        )

    good = [i for i in (15, 16) if plausible_ankle(i)]
    if not good:
        return bottom, 0.0, "bbox_bottom"
    if len(good) == 2:
        a, b = points[15], points[16]
        # F4/F6: when both feet are on the ground (roughly level) the MIDPOINT is a
        # body-centric reference every camera agrees on; only when clearly striding
        # (one foot well above the other) do we drop to the planted (lower) foot.
        if abs(float(a[1] - b[1])) <= level_frac * float(bbox[3]):
            return np.mean(np.asarray([a, b], dtype=float), axis=0), ankle_height_m, "ankle_mid"
        planted = a if float(a[1]) > float(b[1]) else b
        return np.asarray(planted, dtype=float).copy(), ankle_height_m, "ankle_planted"
    return np.asarray(points[good[0]], dtype=float).copy(), ankle_height_m, "ankle_planted"


def ground_contact_pixel(
    bbox_xywh_px: list[float],
    keypoints_px: np.ndarray,
    keypoint_confidence: np.ndarray,
    *,
    ankle_confidence_min: float = 0.6,
    max_ankle_above_bbox_fraction: float = 0.25,
    mode: str = "legacy",
    ankle_height_m: float = 0.10,
    horizontal_margin_frac: float = 0.15,
    level_frac: float = 0.15,
) -> np.ndarray:
    """Backward-compatible foot-contact pixel (see :func:`ground_contact_pixel_ex`)."""

    return ground_contact_pixel_ex(
        bbox_xywh_px,
        keypoints_px,
        keypoint_confidence,
        ankle_confidence_min=ankle_confidence_min,
        max_ankle_above_bbox_fraction=max_ankle_above_bbox_fraction,
        mode=mode,
        ankle_height_m=ankle_height_m,
        horizontal_margin_frac=horizontal_margin_frac,
        level_frac=level_frac,
    )[0]


def ground_homography_from_projection(projection_matrix: np.ndarray) -> np.ndarray:
    """Return the image->world-z=0 homography for a calibrated camera."""

    projection = np.asarray(projection_matrix, dtype=float)
    if projection.shape != (3, 4) or not np.isfinite(projection).all():
        raise ValueError("projection_matrix must be a finite 3x4 matrix")
    ground_to_image = projection[:, [0, 1, 3]]
    try:
        image_to_ground = np.linalg.inv(ground_to_image)
    except np.linalg.LinAlgError as exc:
        raise ValueError("ground-plane homography is singular") from exc
    if not np.isfinite(image_to_ground).all():
        raise ValueError("ground-plane homography is non-finite")
    return image_to_ground


def pixel_to_ground_xy(pixel_xy: np.ndarray, projection_matrix: np.ndarray) -> np.ndarray:
    """Back-project one image point onto the calibrated world ``z=0`` plane."""

    point = np.asarray(pixel_xy, dtype=float)
    if point.shape != (2,) or not np.isfinite(point).all():
        return np.full(2, np.nan)
    homogeneous = ground_homography_from_projection(projection_matrix) @ np.append(point, 1.0)
    if abs(float(homogeneous[2])) < 1e-12:
        return np.full(2, np.nan)
    result = homogeneous[:2] / homogeneous[2]
    return result if np.isfinite(result).all() else np.full(2, np.nan)


def pixel_to_plane_xy(
    pixel_xy: np.ndarray,
    projection_matrix: np.ndarray,
    plane_height_m: float,
) -> np.ndarray:
    """Back-project one image point onto the horizontal world plane ``z = h``.

    Same homography construction as the ground plane, shifted up: a point
    ``(x, y, h, 1)`` maps through ``P`` as ``[p1 | p2 | h*p3 + p4]`` acting on
    ``(x, y, 1)``.
    """

    point = np.asarray(pixel_xy, dtype=float)
    projection = np.asarray(projection_matrix, dtype=float)
    if point.shape != (2,) or not np.isfinite(point).all() or projection.shape != (3, 4):
        return np.full(2, np.nan)
    plane_to_image = np.column_stack([
        projection[:, 0], projection[:, 1],
        plane_height_m * projection[:, 2] + projection[:, 3],
    ])
    try:
        image_to_plane = np.linalg.inv(plane_to_image)
    except np.linalg.LinAlgError:
        return np.full(2, np.nan)
    homogeneous = image_to_plane @ np.append(point, 1.0)
    if abs(float(homogeneous[2])) < 1e-12:
        return np.full(2, np.nan)
    result = homogeneous[:2] / homogeneous[2]
    return result if np.isfinite(result).all() else np.full(2, np.nan)


def upper_body_ground_estimate(
    keypoints_px: np.ndarray,
    keypoint_conf: np.ndarray,
    bbox_xywh_px: list[float],
    projection_matrix: np.ndarray,
    *,
    hip_height_m: float = 0.93,
    shoulder_height_m: float = 1.42,
    head_height_m: float = 1.78,
    keypoint_conf_min: float = 0.45,
) -> tuple[np.ndarray, str] | None:
    """Estimate a standing person's ground position when their feet are unusable.

    A ray through a body landmark of KNOWN typical height intersected with the
    horizontal plane at that height lands directly above the feet. Preference
    order: hip midpoint (closest to the ground, least height variance), shoulder
    midpoint, then the bbox top-centre as the head crown (works even when the
    pose model fails entirely, e.g. dark distant umpires). Returns
    ``(ground_xy, anchor_kind)`` or ``None``. The height priors are population
    means; the residual height error divided by the ray's elevation tangent is
    the caller's extra ground variance — smallest exactly for the close-to-camera
    subjects that get cut off at the frame bottom.
    """

    keypoints = np.asarray(keypoints_px, dtype=float).reshape(-1, 2)
    conf = np.asarray(keypoint_conf, dtype=float).reshape(-1)

    def midpoint(indices: tuple[int, int]) -> np.ndarray | None:
        usable = [keypoints[j] for j in indices
                  if np.isfinite(conf[j]) and conf[j] >= keypoint_conf_min
                  and np.isfinite(keypoints[j]).all()]
        if not usable:
            return None
        return np.mean(np.asarray(usable, dtype=float), axis=0)

    for indices, height, kind in (
        ((11, 12), hip_height_m, "hips"),
        ((5, 6), shoulder_height_m, "shoulders"),
    ):
        anchor = midpoint(indices)
        if anchor is not None:
            xy = pixel_to_plane_xy(anchor, projection_matrix, height)
            if np.isfinite(xy).all():
                return xy, kind

    bbox = np.asarray(bbox_xywh_px, dtype=float)
    if bbox.shape == (4,) and np.isfinite(bbox).all() and bbox[2] > 8 and bbox[3] > 24:
        top_centre = np.array([bbox[0] + bbox[2] / 2.0, bbox[1]], dtype=float)
        xy = pixel_to_plane_xy(top_centre, projection_matrix, head_height_m)
        if np.isfinite(xy).all():
            return xy, "bbox_top"
    return None


def ground_covariance(
    pixel_xy: np.ndarray,
    projection_matrix: np.ndarray,
    *,
    sigma_px: float = 2.0,
    var_floor_m: float = 0.05,
) -> np.ndarray:
    """2x2 world-XY covariance of a ground projection under isotropic pixel noise.

    Propagates ``sigma_px`` through the image->ground homography Jacobian
    (evaluated numerically at the pixel). The result is strongly anisotropic —
    elongated along the viewing ray, growing with distance — which is exactly the
    structure a fixed scalar ground gate cannot represent. ``var_floor_m`` adds an
    isotropic floor for calibration/model error. Returns an inf-diagonal matrix
    when the projection is invalid so callers can gate on finiteness.
    """

    invalid = np.diag([np.inf, np.inf])
    point = np.asarray(pixel_xy, dtype=float)
    if point.shape != (2,) or not np.isfinite(point).all():
        return invalid
    step = 0.5
    jacobian = np.zeros((2, 2), dtype=float)
    for axis in range(2):
        offset = np.zeros(2)
        offset[axis] = step
        plus = pixel_to_ground_xy(point + offset, projection_matrix)
        minus = pixel_to_ground_xy(point - offset, projection_matrix)
        if not (np.isfinite(plus).all() and np.isfinite(minus).all()):
            return invalid
        jacobian[:, axis] = (plus - minus) / (2.0 * step)
    covariance = float(sigma_px) ** 2 * jacobian @ jacobian.T
    covariance += float(var_floor_m) ** 2 * np.eye(2)
    if not np.isfinite(covariance).all():
        return invalid
    return covariance


def ground_mahalanobis_sq(
    xy_a: np.ndarray,
    cov_a: np.ndarray,
    xy_b: np.ndarray,
    cov_b: np.ndarray,
) -> float:
    """Squared Mahalanobis distance between two ground points under summed covariance."""

    a = np.asarray(xy_a, dtype=float)
    b = np.asarray(xy_b, dtype=float)
    S = np.asarray(cov_a, dtype=float) + np.asarray(cov_b, dtype=float)
    if not (np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(S).all()):
        return float("inf")
    diff = a - b
    try:
        solved = np.linalg.solve(S, diff)
    except np.linalg.LinAlgError:
        return float("inf")
    value = float(diff @ solved)
    return value if np.isfinite(value) and value >= 0.0 else float("inf")


def fuse_ground_estimates(
    points_xy: np.ndarray,
    covariances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse-covariance-weighted fusion of per-camera ground estimates.

    Returns ``(fused_xy, fused_cov)``; NaN/inf entries are skipped. Falls back to
    NaN when no member is usable.
    """

    points = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    covs = np.asarray(covariances, dtype=float).reshape(-1, 2, 2)
    information = np.zeros((2, 2), dtype=float)
    weighted = np.zeros(2, dtype=float)
    used = 0
    for point, cov in zip(points, covs):
        if not (np.isfinite(point).all() and np.isfinite(cov).all()):
            continue
        try:
            inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            continue
        information += inv
        weighted += inv @ point
        used += 1
    if used == 0:
        return np.full(2, np.nan), np.diag([np.inf, np.inf])
    try:
        fused_cov = np.linalg.inv(information)
    except np.linalg.LinAlgError:
        return np.full(2, np.nan), np.diag([np.inf, np.inf])
    return fused_cov @ weighted, fused_cov


def robust_fuse_ground(
    points_xy: np.ndarray,
    covariances: np.ndarray,
    *,
    huber_delta: float = 2.0,
    max_iters: int = 5,
    tol_m: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Outlier-robust inverse-covariance fusion of per-camera ground estimates.

    IRLS over the members: start from the plain inverse-covariance fusion, then
    iteratively down-weight any member whose Mahalanobis residual to the current
    estimate exceeds ``huber_delta`` (Huber weight = delta / r), and re-fuse. This
    is the ground-plane analogue of robust multi-view triangulation (Lee & Civera
    2020): one camera whose foot pixel is garbage (a hallucinated ankle, a bbox cut
    off at the frame edge) is progressively rejected instead of dragging an
    unweighted median/mean by a metre.

    Returns ``(fused_xy, fused_cov, weights)``. ``weights`` are the final per-member
    Huber weights in [0, 1] (NaN/degenerate members get 0). Falls back to the
    non-robust fusion when fewer than two members are usable.
    """

    points = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    covs = np.asarray(covariances, dtype=float).reshape(-1, 2, 2)
    n = points.shape[0]
    usable = np.array(
        [np.isfinite(points[i]).all() and np.isfinite(covs[i]).all() for i in range(n)],
        dtype=bool,
    )
    weights = np.zeros(n, dtype=float)
    if int(usable.sum()) == 0:
        return np.full(2, np.nan), np.diag([np.inf, np.inf]), weights
    if int(usable.sum()) < 2:
        fused_xy, fused_cov = fuse_ground_estimates(points, covs)
        weights[usable] = 1.0
        return fused_xy, fused_cov, weights

    delta = max(float(huber_delta), 1e-6)
    current, _ = fuse_ground_estimates(points, covs)
    if not np.isfinite(current).all():
        return np.full(2, np.nan), np.diag([np.inf, np.inf]), weights

    for _ in range(max(1, int(max_iters))):
        new_weights = np.zeros(n, dtype=float)
        for i in range(n):
            if not usable[i]:
                continue
            r2 = ground_mahalanobis_sq(points[i], np.zeros((2, 2)), current, covs[i])
            r = float(np.sqrt(r2)) if np.isfinite(r2) else float("inf")
            new_weights[i] = 1.0 if r <= delta else max(delta / r, 1e-3)
        information = np.zeros((2, 2), dtype=float)
        weighted = np.zeros(2, dtype=float)
        for i in range(n):
            if new_weights[i] <= 0.0:
                continue
            try:
                inv = np.linalg.inv(covs[i]) * new_weights[i]
            except np.linalg.LinAlgError:
                continue
            information += inv
            weighted += inv @ points[i]
        try:
            fused_cov = np.linalg.inv(information)
        except np.linalg.LinAlgError:
            return np.full(2, np.nan), np.diag([np.inf, np.inf]), weights
        updated = fused_cov @ weighted
        weights = new_weights
        if not np.isfinite(updated).all():
            return np.full(2, np.nan), np.diag([np.inf, np.inf]), weights
        if float(np.linalg.norm(updated - current)) < float(tol_m):
            current = updated
            break
        current = updated

    return current, fused_cov, weights


def ground_from_reprojection(
    feet_px: np.ndarray,
    projection_matrices: np.ndarray,
    confidences: np.ndarray | None = None,
    *,
    plane_heights: np.ndarray | None = None,
    huber_delta_px: float = 8.0,
    max_iters: int = 10,
    tol_m: float = 1e-4,
) -> np.ndarray:
    """Ground ``(x, y)`` minimizing robust reprojection error (see the ``_ex`` variant)."""

    return ground_from_reprojection_ex(
        feet_px, projection_matrices, confidences,
        plane_heights=plane_heights, huber_delta_px=huber_delta_px,
        max_iters=max_iters, tol_m=tol_m,
    )[0]


def ground_from_reprojection_ex(
    feet_px: np.ndarray,
    projection_matrices: np.ndarray,
    confidences: np.ndarray | None = None,
    *,
    plane_heights: np.ndarray | None = None,
    huber_delta_px: float = 8.0,
    max_iters: int = 10,
    tol_m: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Ground ``(x, y)`` that minimizes robust reprojection error with ``z`` fixed at 0.

    Solves ``argmin_{x,y} Σ_c w_c ρ(‖ project_c([x, y, 0]) − foot_c ‖)`` by Gauss–Newton
    with Huber IRLS, using every camera's full 3×4 projection matrix jointly. Unlike
    free-space triangulation this stays well-conditioned on the low-parallax *facing*
    pairs (C1↔C4, C2↔C6, C3↔C5) because the depth DOF is removed by the ground
    constraint; unlike the median of per-camera homography back-projections it is the
    reprojection-optimal point, so with calibrated cameras it lands on the true foot to
    a few centimetres. Initialised from the median homography back-projection.

    Returns ``([x, y], cov)`` in world metres. ``cov`` (F9a) is the 2×2 posterior
    covariance from the final Gauss–Newton normal matrix, ``σ̂² (JᵀWJ)⁻¹`` with ``σ̂²``
    the Huber-weighted residual variance — essentially free, since ``JᵀWJ`` is already
    assembled by the solver. It is anisotropic and grows along each camera's depth
    direction, exactly the measurement-noise model P4's Kalman needs. ``cov`` is None
    for the single-view case (the caller should use the per-view homography-Jacobian
    :func:`ground_covariance` instead) and on failure (NaN point).
    """

    feet = np.asarray(feet_px, dtype=float).reshape(-1, 2)
    projections = np.asarray(projection_matrices, dtype=float).reshape(-1, 3, 4)
    n = feet.shape[0]
    if projections.shape[0] != n or n == 0:
        return np.full(2, np.nan), None
    if confidences is None:
        confidences = np.ones(n, dtype=float)
    confidences = np.asarray(confidences, dtype=float).reshape(-1)
    if plane_heights is None:
        heights = np.zeros(n, dtype=float)
    else:
        heights = np.asarray(plane_heights, dtype=float).reshape(-1)
        if heights.shape[0] != n:
            heights = np.zeros(n, dtype=float)

    valid = np.array(
        [
            np.isfinite(feet[i]).all()
            and np.isfinite(projections[i]).all()
            and np.isfinite(confidences[i])  # H5: max(NaN, eps) is NaN -> poisons JTJ
            for i in range(n)
        ],
        dtype=bool,
    )
    if int(valid.sum()) == 0:
        return np.full(2, np.nan), None

    inits = []
    for i in range(n):
        if not valid[i]:
            continue
        # Back-project onto the landmark's own height plane (z = h_i), so an ankle
        # gives the (x, y) directly below it -- i.e. the true ground contact (F2).
        xy = (
            pixel_to_plane_xy(feet[i], projections[i], float(heights[i]))
            if abs(float(heights[i])) > 1e-9
            else pixel_to_ground_xy(feet[i], projections[i])
        )
        if np.isfinite(xy).all():
            inits.append(xy)
    if not inits:
        return np.full(2, np.nan), None
    xy = np.median(np.asarray(inits, dtype=float), axis=0).astype(float)
    if int(valid.sum()) == 1:
        return xy, None  # single view: homography exact; cov = per-view model

    delta = max(float(huber_delta_px), 1e-6)

    def gauss_newton(active: np.ndarray, start: np.ndarray) -> np.ndarray:
        point = start.astype(float).copy()
        for _ in range(max(1, int(max_iters))):
            JTJ = np.zeros((2, 2), dtype=float)
            JTr = np.zeros(2, dtype=float)
            for i in range(n):
                if not active[i]:
                    continue
                P = projections[i]
                homogeneous = P @ np.array([point[0], point[1], float(heights[i]), 1.0])
                if abs(homogeneous[2]) < 1e-9:
                    continue
                projected = homogeneous[:2] / homogeneous[2]
                residual = projected - feet[i]
                jac = np.zeros((2, 2), dtype=float)
                for axis in (0, 1):
                    column = P[:, axis]
                    jac[:, axis] = (
                        column[:2] * homogeneous[2] - homogeneous[:2] * column[2]
                    ) / (homogeneous[2] ** 2)
                residual_norm = float(np.linalg.norm(residual))
                weight = float(max(confidences[i], 1e-3))
                if residual_norm > delta:
                    weight *= delta / max(residual_norm, 1e-6)  # Huber IRLS
                JTJ += weight * jac.T @ jac
                JTr += weight * jac.T @ residual
            try:
                step = np.linalg.solve(JTJ + 1e-6 * np.eye(2), JTr)
            except np.linalg.LinAlgError:
                break
            point = point - step
            if not np.isfinite(point).all():
                # NOTE: must return a bare array — the caller treats the return as the
                # xy point; a (point, None) tuple here crashed np.isfinite (audit fix).
                return np.full(2, np.nan)
            if float(np.linalg.norm(step)) < float(tol_m):
                break
        return point

    xy = gauss_newton(valid, xy)
    if not np.isfinite(xy).all():
        return np.full(2, np.nan), None

    # Hard inlier refit: the Huber pass down-weights a gross outlier (a hallucinated
    # ankle can be 50-200 px off) but never fully rejects it. Drop views whose
    # reprojection residual is a clear outlier and re-solve on the consensus, so one
    # bad foot pixel cannot bias the metre-scale ground point at all.
    residuals = np.full(n, np.nan)
    for i in range(n):
        if not valid[i]:
            continue
        homogeneous = projections[i] @ np.array([xy[0], xy[1], float(heights[i]), 1.0])
        if abs(homogeneous[2]) < 1e-9:
            continue
        residuals[i] = float(np.linalg.norm(homogeneous[:2] / homogeneous[2] - feet[i]))
    finite = residuals[np.isfinite(residuals)]
    final_active = valid
    if finite.size >= 3:
        reject = max(3.0 * delta, 2.5 * float(np.median(finite)))
        inliers = valid & np.isfinite(residuals) & (residuals <= reject)
        if 2 <= int(inliers.sum()) < int(valid.sum()):
            refit = gauss_newton(inliers, xy)
            if np.isfinite(refit).all():
                xy = refit
                final_active = inliers

    # F9a: posterior covariance at the solution over the final active set.
    JTJ = np.zeros((2, 2), dtype=float)
    weighted_ssr = 0.0
    m_views = 0
    for i in range(n):
        if not final_active[i]:
            continue
        P = projections[i]
        homogeneous = P @ np.array([xy[0], xy[1], float(heights[i]), 1.0])
        if abs(homogeneous[2]) < 1e-9:
            continue
        projected = homogeneous[:2] / homogeneous[2]
        residual = projected - feet[i]
        jac = np.zeros((2, 2), dtype=float)
        for axis in (0, 1):
            column = P[:, axis]
            jac[:, axis] = (
                column[:2] * homogeneous[2] - homogeneous[:2] * column[2]
            ) / (homogeneous[2] ** 2)
        residual_norm = float(np.linalg.norm(residual))
        weight = float(max(confidences[i], 1e-3))
        if residual_norm > delta:
            weight *= delta / max(residual_norm, 1e-6)
        JTJ += weight * jac.T @ jac
        weighted_ssr += weight * residual_norm ** 2
        m_views += 1
    cov = None
    if m_views >= 2:
        dof = max(2 * m_views - 2, 1)
        sigma_sq = max(weighted_ssr / dof, 1e-6)  # px^2; floored so cov is never 0
        try:
            cov = sigma_sq * np.linalg.inv(JTJ + 1e-9 * np.eye(2))
        except np.linalg.LinAlgError:
            cov = None
        if cov is not None and not np.isfinite(cov).all():
            cov = None
    return xy, cov


def ground_point_and_cov(
    foot_pixel_xy: np.ndarray,
    projection_matrix: np.ndarray,
    *,
    sigma_px: float = 2.0,
    var_floor_m: float = 0.4,
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience: back-project a foot pixel to the ground and attach its covariance.

    Pairs :func:`pixel_to_ground_xy` with :func:`ground_covariance` so callers get a
    ``(xy, 2x2)`` estimate in one step, ready for :func:`robust_fuse_ground`.
    """

    xy = pixel_to_ground_xy(foot_pixel_xy, projection_matrix)
    cov = ground_covariance(
        foot_pixel_xy, projection_matrix, sigma_px=sigma_px, var_floor_m=var_floor_m
    )
    return xy, cov


def huber_cost(r: float, delta: float) -> float:
    """Huber cost: quadratic for ``r <= delta``, linear beyond; continuous at delta."""

    if r <= delta:
        return r ** 2 / (2.0 * delta)
    return r - delta / 2.0


def parallax_weight(
    parallax_deg: float,
    min_deg: float = 10.0,
    full_deg: float = 25.0,
) -> float:
    """Triangulation-reliability weight: 0 below ``min_deg``, 1 above ``full_deg``."""

    if parallax_deg <= min_deg:
        return 0.0
    if parallax_deg >= full_deg:
        return 1.0
    return (parallax_deg - min_deg) / (full_deg - min_deg)


def camera_axis_lookat(
    projection_matrix: np.ndarray,
    *,
    toward: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(centre, forward_unit, ground_lookat_xy)`` for a calibrated camera.

    ``forward_unit`` is the principal axis oriented toward ``toward`` (default: the
    world origin / pitch centre), resolving the bundle-adjustment sign ambiguity for
    this rig where every camera looks at the field. ``ground_lookat_xy`` is where that
    axis meets the ``z=0`` ground plane — the strip the camera actually frames.
    """

    P = np.asarray(projection_matrix, dtype=float)
    C = camera_center_from_P(P)
    target = np.zeros(3) if toward is None else np.asarray(toward, dtype=float)
    v = P[2, :3].astype(float)
    norm = float(np.linalg.norm(v))
    if norm < 1e-12 or not np.isfinite(C).all():
        return C, np.full(3, np.nan), np.full(2, np.nan)
    v = v / norm
    if (target - C) @ v < 0:
        v = -v
    if abs(float(v[2])) < 1e-9:
        return C, v, np.full(2, np.nan)
    t = -C[2] / v[2]
    return C, v, (C + t * v)[:2]


def project_ground_to_pixel(
    projection_matrix: np.ndarray,
    xy: np.ndarray,
    *,
    height_m: float = 0.0,
) -> np.ndarray:
    """Forward-project a world point ``(x, y, height_m)`` to image ``[u, v]`` pixels.

    The inverse of :func:`pixel_to_ground_xy`. Returns NaNs when the point is on or
    behind the image plane (``w <= 0``) or the projection is invalid. Used to place a
    ghost marker's known ground position into every camera and to reproject a lost
    track for the ghost-verification pass.
    """

    P = np.asarray(projection_matrix, dtype=float)
    point = np.asarray(xy, dtype=float)
    if P.shape != (3, 4) or point.shape != (2,) or not np.isfinite(point).all():
        return np.full(2, np.nan)
    homogeneous = P @ np.array([point[0], point[1], float(height_m), 1.0])
    if not np.isfinite(homogeneous).all() or abs(float(homogeneous[2])) < 1e-9:
        return np.full(2, np.nan)
    return homogeneous[:2] / homogeneous[2]


def ground_point_visible_in(
    projection_matrix: np.ndarray,
    xy: np.ndarray,
    image_wh: tuple[int, int] | np.ndarray,
    *,
    toward: np.ndarray | None = None,
    margin_px: float = 0.0,
    height_m: float = 0.0,
) -> bool:
    """Whether a world ground point is visible in a camera (cheirality + in-frame).

    A point can reproject *inside* the image rectangle while lying **behind** the
    camera (the projective ambiguity the current ghost code ignores), so this first
    enforces cheirality — the point must be on the same side of the camera as the
    pitch (using :func:`camera_axis_lookat`'s pitch-oriented forward axis) — and only
    then checks the reprojected pixel falls within ``image_wh`` (expanded by
    ``margin_px``). This is the per-camera ground-visibility test the ghost markers and
    ghost-verification pass need to decide "is this ground location seen by camera X".
    """

    P = np.asarray(projection_matrix, dtype=float)
    point = np.asarray(xy, dtype=float)
    if P.shape != (3, 4) or point.shape != (2,) or not np.isfinite(point).all():
        return False
    centre, forward, _ = camera_axis_lookat(P, toward=toward)
    if not (np.isfinite(centre).all() and np.isfinite(forward).all()):
        return False
    world = np.array([point[0], point[1], float(height_m)], dtype=float)
    if float((world - centre) @ forward) <= 0.0:  # behind the camera
        return False
    pixel = project_ground_to_pixel(P, point, height_m=height_m)
    if not np.isfinite(pixel).all():
        return False
    width, height = float(image_wh[0]), float(image_wh[1])
    return (
        -margin_px <= float(pixel[0]) < width + margin_px
        and -margin_px <= float(pixel[1]) < height + margin_px
    )


def derive_facing_pairs(
    projection_matrices: dict[str, np.ndarray],
    *,
    antiparallel_max_dot: float = -0.9,
) -> list[tuple[str, str]]:
    """Auto-derive the co-observing ("facing") camera pairs from calibration geometry.

    A facing pair has anti-parallel forward axes AND the nearest ground look-at points:
    both cameras frame the SAME pitch strip from opposite sides, so a player seen by one
    is seen by the other. This is the relationship that matters for cross-camera identity
    — distinct from diametrically-opposite *positions*, which can look at different strips
    (e.g. C2 vs C5). Returns sorted ``(cam_a, cam_b)`` pairs; greedy mutual-best matching
    leaves genuinely independent cameras (e.g. C7) unpaired.
    """

    cams = sorted(projection_matrices)
    geo = {cam: camera_axis_lookat(projection_matrices[cam]) for cam in cams}
    candidates: list[tuple[float, str, str]] = []
    for cam_a, cam_b in combinations(cams, 2):
        _, va, la_a = geo[cam_a]
        _, vb, la_b = geo[cam_b]
        if not (np.isfinite(va).all() and np.isfinite(vb).all()
                and np.isfinite(la_a).all() and np.isfinite(la_b).all()):
            continue
        if float(va @ vb) >= antiparallel_max_dot:
            continue
        candidates.append((float(np.linalg.norm(la_a - la_b)), cam_a, cam_b))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    used: set[str] = set()
    pairs: list[tuple[str, str]] = []
    for _distance, cam_a, cam_b in candidates:
        if cam_a in used or cam_b in used:
            continue
        pairs.append((cam_a, cam_b))
        used.update((cam_a, cam_b))
    return sorted(pairs)

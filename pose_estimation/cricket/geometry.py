"""Multi-view geometry primitives for cross-camera association (P3) and lifting.

Stateless math shared by the association engine, the global-ID tracker, and the
3D lift. Two-view triangulation delegates to :mod:`pose_estimation.triangulation`
(the repo's weighted-DLT/RANSAC home) rather than re-implementing the SVD solve.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from pose_estimation.triangulation import triangulate_point_dlt


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

    Thin wrapper over :func:`pose_estimation.triangulation.triangulate_point_dlt`
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


def ground_contact_pixel(
    bbox_xywh_px: list[float],
    keypoints_px: np.ndarray,
    keypoint_confidence: np.ndarray,
    *,
    ankle_confidence_min: float = 0.6,
    max_ankle_above_bbox_fraction: float = 0.25,
) -> np.ndarray:
    """Return a conservative image-space foot contact for a COCO-17 pose.

    The lower confident ankle is preferable to the bbox bottom, but only when it
    is geometrically plausible.  Pose models frequently hallucinate one ankle
    around a knee or a raised foot; accepting that point unconditionally moves a
    ground-plane projection by several metres with the long cricket lenses.
    """

    bbox = np.asarray(bbox_xywh_px, dtype=float)
    points = np.asarray(keypoints_px, dtype=float)
    confidence = np.asarray(keypoint_confidence, dtype=float)
    bottom = bbox_bottom_center_px(list(bbox))
    if bbox.shape != (4,) or points.shape != (17, 2) or confidence.shape != (17,):
        return bottom
    if not np.isfinite(bbox).all() or bbox[2] <= 0.0 or bbox[3] <= 0.0:
        return bottom

    tolerance = max(20.0, max_ankle_above_bbox_fraction * float(bbox[3]))
    plausible: list[np.ndarray] = []
    for index in (15, 16):
        point = points[index]
        if (
            np.isfinite(point).all()
            and np.isfinite(confidence[index])
            and float(confidence[index]) >= ankle_confidence_min
            and bottom[1] - tolerance <= point[1] <= bottom[1] + 0.1 * bbox[3]
        ):
            plausible.append(point)
    if not plausible:
        return bottom
    if len(plausible) == 2 and abs(float(plausible[0][1] - plausible[1][1])) <= 0.05 * bbox[3]:
        return np.mean(np.asarray(plausible, dtype=float), axis=0)
    # Image y increases downward, so the largest-y ankle is the planted/lower
    # foot when the player is walking, bowling, or batting with one foot raised.
    return np.asarray(max(plausible, key=lambda point: float(point[1])), dtype=float).copy()


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

"""View-invariant pose-shape descriptors for cross-camera identity.

Same "stateless math shared across P3/P4" pattern as :mod:`geometry`. The primary
identity signal in this pipeline is ground-plane foot position, which cannot separate
two identical-kit players standing close together. A person's *body proportions*
(bone-length ratios) are a position-independent, kit-independent identity cue.

These proportions must be computed from the **triangulated 3D** skeleton, not raw 2D
keypoints: 2D bone lengths and joint angles are foreshortened differently by each
camera (and left/right can flip on back-facing views), so a 2D descriptor is not
comparable across the opposite-facing cameras that co-observe a player here. In 3D the
proportions are genuinely view-invariant. The caller supplies the triangulated points,
per-joint confidence, and a per-joint ``parallax_ok`` mask (segments triangulated with
poor parallax -- e.g. across the near-collinear facing pairs -- are excluded so noisy
depth never drives a match).

This module is a *soft tie-breaker*: use :func:`descriptor_distance` to nudge among
candidates that geometry already deems plausible, never as a hard gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from identity.common.geometry import camera_center_from_P, pixel_to_ground_xy

# COCO-17 joint indices
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

# Each segment is (name, endpoint_a, endpoint_b); an endpoint is a joint index or a
# tuple of joint indices whose midpoint is used (for the spine).
_SEGMENTS: tuple[tuple[str, object, object], ...] = (
    ("shoulder_width", L_SHOULDER, R_SHOULDER),
    ("hip_width", L_HIP, R_HIP),
    ("spine", (L_SHOULDER, R_SHOULDER), (L_HIP, R_HIP)),
    ("upper_arm_l", L_SHOULDER, L_ELBOW),
    ("upper_arm_r", R_SHOULDER, R_ELBOW),
    ("forearm_l", L_ELBOW, L_WRIST),
    ("forearm_r", R_ELBOW, R_WRIST),
    ("thigh_l", L_HIP, L_KNEE),
    ("thigh_r", R_HIP, R_KNEE),
    ("shin_l", L_KNEE, L_ANKLE),
    ("shin_r", R_KNEE, R_ANKLE),
)
SEGMENT_NAMES: tuple[str, ...] = tuple(name for name, _a, _b in _SEGMENTS)
SEGMENT_COUNT = len(_SEGMENTS)


@dataclass(frozen=True)
class PoseProportions:
    """Scale-normalized 3D bone-length ratios; a view-invariant identity descriptor."""

    vector: np.ndarray  # (SEGMENT_COUNT,) segment_length / robust_scale; NaN where invalid
    mask: np.ndarray    # (SEGMENT_COUNT,) bool - segment validity
    scale: float        # robust body-scale used to normalize (metres), or NaN
    n_views: int        # camera views that contributed to the reconstruction

    def is_defined(self) -> bool:
        return bool(np.any(self.mask))

    def to_json(self) -> dict:
        return {
            "vector": [None if not np.isfinite(v) else float(v) for v in self.vector],
            "mask": [bool(m) for m in self.mask],
            "scale": float(self.scale) if np.isfinite(self.scale) else None,
            "n_views": int(self.n_views),
        }

    @staticmethod
    def from_json(payload: dict | None) -> "PoseProportions | None":
        if not payload:
            return None
        raw = payload.get("vector")
        mask_raw = payload.get("mask")
        if raw is None or mask_raw is None or len(raw) != SEGMENT_COUNT or len(mask_raw) != SEGMENT_COUNT:
            return None
        vector = np.array([np.nan if v is None else float(v) for v in raw], dtype=float)
        mask = np.array([bool(m) for m in mask_raw], dtype=bool)
        scale = payload.get("scale")
        return PoseProportions(
            vector=vector,
            mask=mask,
            scale=float(scale) if scale is not None else float("nan"),
            n_views=int(payload.get("n_views", 0)),
        )


def _endpoint(points3d: np.ndarray, spec: object) -> np.ndarray:
    if isinstance(spec, tuple):
        return np.mean(points3d[list(spec)], axis=0)
    return points3d[int(spec)]


def _endpoint_valid(valid: np.ndarray, spec: object) -> bool:
    if isinstance(spec, tuple):
        return bool(np.all(valid[list(spec)]))
    return bool(valid[int(spec)])


def limb_proportion_descriptor(
    points3d: np.ndarray,
    joint_conf: np.ndarray,
    parallax_ok: np.ndarray | None = None,
    *,
    min_conf: float = 0.3,
    n_views: int = 0,
) -> PoseProportions:
    """Build a view-invariant bone-ratio descriptor from a triangulated skeleton.

    A segment is used only when both endpoints are finite, confident (``>= min_conf``)
    and (when ``parallax_ok`` is given) triangulated with adequate parallax. Lengths are
    normalized by the median valid segment length, giving scale-invariant ratios.
    """

    points3d = np.asarray(points3d, dtype=float).reshape(-1, 3)
    joint_conf = np.asarray(joint_conf, dtype=float).reshape(-1)
    joints = points3d.shape[0]
    if parallax_ok is None:
        parallax_ok = np.ones(joints, dtype=bool)
    else:
        parallax_ok = np.asarray(parallax_ok, dtype=bool).reshape(-1)

    valid_joint = (
        np.isfinite(points3d).all(axis=1)
        & np.isfinite(joint_conf)
        & (joint_conf >= min_conf)
        & parallax_ok
    )

    lengths = np.full(SEGMENT_COUNT, np.nan, dtype=float)
    mask = np.zeros(SEGMENT_COUNT, dtype=bool)
    for index, (_name, endpoint_a, endpoint_b) in enumerate(_SEGMENTS):
        if not (_endpoint_valid(valid_joint, endpoint_a) and _endpoint_valid(valid_joint, endpoint_b)):
            continue
        length = float(np.linalg.norm(_endpoint(points3d, endpoint_a) - _endpoint(points3d, endpoint_b)))
        if np.isfinite(length) and length > 1e-6:
            lengths[index] = length
            mask[index] = True

    if not np.any(mask):
        return PoseProportions(np.full(SEGMENT_COUNT, np.nan), mask, float("nan"), int(n_views))

    scale = float(np.median(lengths[mask]))
    if not np.isfinite(scale) or scale <= 1e-6:
        return PoseProportions(np.full(SEGMENT_COUNT, np.nan), np.zeros(SEGMENT_COUNT, bool), float("nan"), int(n_views))

    vector = np.full(SEGMENT_COUNT, np.nan, dtype=float)
    vector[mask] = lengths[mask] / scale
    return PoseProportions(vector=vector, mask=mask, scale=scale, n_views=int(n_views))


def descriptor_distance(
    a: PoseProportions | None,
    b: PoseProportions | None,
    *,
    min_shared: int = 4,
) -> float | None:
    """Mean absolute difference of shared bone ratios in [0, 1].

    Returns ``None`` when the two descriptors share fewer than ``min_shared`` valid
    segments -- the signal to the caller that pose is *not comparable* here and must
    contribute nothing (a neutral tie-breaker), rather than a spurious penalty.
    """

    if a is None or b is None:
        return None
    shared = a.mask & b.mask
    if int(shared.sum()) < int(min_shared):
        return None
    difference = np.abs(a.vector[shared] - b.vector[shared])
    if not np.isfinite(difference).all():
        finite = np.isfinite(difference)
        if int(finite.sum()) < int(min_shared):
            return None
        difference = difference[finite]
    return float(np.clip(np.mean(difference), 0.0, 1.0))


def merge_descriptor(
    accum: PoseProportions | None,
    new: PoseProportions | None,
    *,
    rate: float = 0.15,
) -> PoseProportions | None:
    """Per-segment EMA of an accumulated descriptor with a new observation.

    Segments valid only in ``new`` are adopted; segments valid only in ``accum`` are
    retained; overlapping segments are blended. Robust to partial occlusion and to
    per-frame triangulation noise.
    """

    if new is None or not new.is_defined():
        return accum
    if accum is None or not accum.is_defined():
        return new

    rate = float(np.clip(rate, 0.0, 1.0))
    vector = accum.vector.copy()
    mask = accum.mask.copy()
    for index in range(SEGMENT_COUNT):
        if not new.mask[index]:
            continue
        if mask[index] and np.isfinite(vector[index]):
            vector[index] = (1.0 - rate) * vector[index] + rate * new.vector[index]
        else:
            vector[index] = new.vector[index]
            mask[index] = True
    return PoseProportions(
        vector=vector,
        mask=mask,
        scale=new.scale if np.isfinite(new.scale) else accum.scale,
        n_views=max(accum.n_views, new.n_views),
    )


# --- ground-anchored monocular posture (billboard lift) ------------------------------
#
# The triangulated descriptor above needs adequate parallax, which the co-observing
# FACING pairs never have. This second channel avoids triangulation entirely: each
# camera's 2D keypoints are lifted to metric 3D on a vertical plane ("billboard")
# through the player's own calibrated ground point. Vertical quantities (stature,
# shoulder/hip height) are foreshortening-free and directly comparable between any
# two cameras; planar widths are comparable exactly on anti-parallel facing pairs and
# are otherwise absorbed by the per-camera-pair calibration variance. Aggregated over
# a P2 tracklet the per-frame noise averages down (SE = sigma/sqrt(n_eff)), which is
# what makes this usable as an identity cue at all.

_HEAD_JOINTS = (NOSE, L_EYE, R_EYE, L_EAR, R_EAR)

POSTURE_QUANTITIES: tuple[str, ...] = (
    "head_top_m",      # highest confident head joint above ground (stature proxy)
    "shoulder_h_m",    # mean shoulder height
    "hip_h_m",         # mean hip height
    "leg_len_m",       # vertical hip-to-ankle extent
    "torso_len_m",     # 3D shoulder-mid to hip-mid length
    "shoulder_w_m",    # billboard-plane shoulder width
    "hip_w_m",         # billboard-plane hip width
)
# Quantities meaningful only when the player is standing roughly upright.
_UPRIGHT_ONLY = frozenset({"head_top_m", "shoulder_h_m", "hip_h_m", "leg_len_m"})
# Foreshortening-free verticals: safe to compare across ANY pair of views.
STATURE_QUANTITIES = frozenset(_UPRIGHT_ONLY)

# Fallback cross-camera systematic sigmas (metres) when no cue calibration is
# available; the calibration harness replaces these with measured values.
DEFAULT_POSTURE_SIGMA_SYS: dict[str, float] = {
    "head_top_m": 0.06,
    "shoulder_h_m": 0.06,
    "hip_h_m": 0.06,
    "leg_len_m": 0.07,
    "torso_len_m": 0.07,
    "shoulder_w_m": 0.10,
    "hip_w_m": 0.10,
}


def ground_anchored_skeleton(
    keypoints_px: np.ndarray,
    keypoint_conf: np.ndarray,
    foot_pixel: np.ndarray,
    projection: np.ndarray,
    *,
    min_conf: float = 0.3,
    max_height_m: float = 3.2,
    max_lateral_m: float = 2.5,
    ground_xy: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Lift COCO-17 pixels to metric 3D on a vertical plane at the player's depth.

    The plane passes through the player's calibrated ground point with its normal
    along the camera's horizontal viewing direction, so every keypoint ray gets a
    unique, metrically-scaled intersection. Returns ``(points3d (17,3), valid (17,))``
    with NaN rows where the joint is unavailable or implausible. ``ground_xy``
    overrides the foot-pixel projection - used when the feet are cut off and the
    anchor came from an upper-body height-plane estimate instead.
    """

    points = np.asarray(keypoints_px, dtype=float).reshape(-1, 2)
    conf = np.asarray(keypoint_conf, dtype=float).reshape(-1)
    joints = points.shape[0]
    out = np.full((joints, 3), np.nan, dtype=float)
    valid = np.zeros(joints, dtype=bool)

    if ground_xy is not None:
        ground = np.asarray(ground_xy, dtype=float)
    else:
        ground = pixel_to_ground_xy(np.asarray(foot_pixel, dtype=float), projection)
    if ground.shape != (2,) or not np.isfinite(ground).all():
        return out, valid
    P = np.asarray(projection, dtype=float)
    C = camera_center_from_P(P)
    if not np.isfinite(C).all():
        return out, valid
    anchor = np.array([ground[0], ground[1], 0.0], dtype=float)
    normal = anchor - C
    normal[2] = 0.0
    norm = float(np.linalg.norm(normal))
    if norm < 1e-6:  # camera directly overhead: billboard undefined
        return out, valid
    normal /= norm

    M = P[:, :3]
    try:
        rays = np.linalg.solve(
            M, np.concatenate([points.T, np.ones((1, joints))], axis=0)
        ).T  # (joints, 3) ray directions through each pixel
    except np.linalg.LinAlgError:
        return out, valid
    plane_offset = float(normal @ (anchor - C))
    # Vectorised ray-plane intersection over all joints (each joint is independent,
    # no cross-joint reduction) - bit-identical to the per-joint loop it replaces
    # (proven by exact-equality test on real calibration + keypoint data).
    conf_ok = np.isfinite(conf) & (conf >= min_conf) & np.isfinite(points).all(axis=1)
    denoms = rays @ normal                                     # (joints,)
    denom_ok = np.isfinite(denoms) & (np.abs(denoms) >= 1e-9)
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.where(denom_ok, plane_offset / denoms, np.nan)
    X = C[None, :] + scale[:, None] * rays                    # (joints, 3)
    X_ok = np.isfinite(X).all(axis=1)
    height_ok = (X[:, 2] >= -0.5) & (X[:, 2] <= max_height_m)
    lateral_ok = np.linalg.norm(X[:, :2] - anchor[:2], axis=1) <= max_lateral_m
    keep = conf_ok & denom_ok & X_ok & height_ok & lateral_ok
    out[keep] = X[keep]
    valid[keep] = True
    return out, valid


@dataclass(frozen=True)
class PostureSample:
    """Per-frame metric posture quantities from one camera's billboard skeleton.

    ``upright_known`` distinguishes "measured as not standing" (crouching keeper)
    from "could not be determined" (feet cut off at the frame edge) - the two
    must be treated differently when deciding whether shape quantities are
    comparable across oblique views.
    """

    values: dict[str, float]  # quantity -> metres (absent keys were unobservable)
    upright: bool
    upright_known: bool = True


def posture_from_skeleton(
    points3d: np.ndarray,
    valid: np.ndarray,
    *,
    upright_tilt_max_deg: float = 30.0,
    upright_ankle_max_m: float = 0.35,
    upright_hip_min_m: float = 0.6,
) -> PostureSample | None:
    """Reduce a lifted skeleton to comparable metric quantities."""

    points3d = np.asarray(points3d, dtype=float).reshape(-1, 3)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    values: dict[str, float] = {}

    head_z = [points3d[j, 2] for j in _HEAD_JOINTS if valid[j]]
    if head_z:
        values["head_top_m"] = float(np.max(head_z))
    shoulders = [points3d[j] for j in (L_SHOULDER, R_SHOULDER) if valid[j]]
    hips = [points3d[j] for j in (L_HIP, R_HIP) if valid[j]]
    ankles = [points3d[j] for j in (L_ANKLE, R_ANKLE) if valid[j]]
    if shoulders:
        values["shoulder_h_m"] = float(np.mean([p[2] for p in shoulders]))
    if hips:
        values["hip_h_m"] = float(np.mean([p[2] for p in hips]))
    if hips and ankles:
        values["leg_len_m"] = float(
            np.mean([p[2] for p in hips]) - np.min([p[2] for p in ankles])
        )
    if valid[L_SHOULDER] and valid[R_SHOULDER]:
        values["shoulder_w_m"] = float(
            np.linalg.norm(points3d[L_SHOULDER] - points3d[R_SHOULDER])
        )
    if valid[L_HIP] and valid[R_HIP]:
        values["hip_w_m"] = float(np.linalg.norm(points3d[L_HIP] - points3d[R_HIP]))

    upright = False
    upright_known = False
    if len(shoulders) == 2 and len(hips) == 2:
        shoulder_mid = np.mean(np.asarray(shoulders), axis=0)
        hip_mid = np.mean(np.asarray(hips), axis=0)
        spine = shoulder_mid - hip_mid
        values["torso_len_m"] = float(np.linalg.norm(spine))
        if values["torso_len_m"] > 1e-6 and ankles:
            upright_known = True
            tilt = float(np.degrees(np.arccos(
                np.clip(abs(spine[2]) / values["torso_len_m"], 0.0, 1.0)
            )))
            ankle_ok = float(np.min([p[2] for p in ankles])) <= upright_ankle_max_m
            # A squat keeps the spine vertical and the ankles grounded but is NOT
            # standing - the hips give it away.
            hips_ok = float(hip_mid[2]) >= upright_hip_min_m
            upright = tilt <= upright_tilt_max_deg and ankle_ok and hips_ok and spine[2] > 0.0

    if not values:
        return None
    return PostureSample(values=values, upright=upright, upright_known=upright_known)


@dataclass(frozen=True)
class PostureAggregate:
    """Per-tracklet robust posture statistics: median, standard error, sample count."""

    median: dict[str, float]
    se: dict[str, float]
    count: dict[str, int]

    def is_defined(self) -> bool:
        return bool(self.median)

    def to_json(self) -> dict:
        return {
            "median": {k: float(v) for k, v in self.median.items()},
            "se": {k: float(v) for k, v in self.se.items()},
            "count": {k: int(v) for k, v in self.count.items()},
        }

    @staticmethod
    def from_json(payload: dict | None) -> "PostureAggregate | None":
        if not payload or not payload.get("median"):
            return None
        return PostureAggregate(
            median={k: float(v) for k, v in payload["median"].items()},
            se={k: float(v) for k, v in payload.get("se", {}).items()},
            count={k: int(v) for k, v in payload.get("count", {}).items()},
        )


@dataclass
class PostureAccumulator:
    """Collect per-frame posture samples for one tracklet and reduce them robustly.

    Consecutive frames are strongly correlated (same pose, same noise), so the
    standard error uses ``n_eff = n / autocorr_lag`` rather than pretending every
    frame is independent.
    """

    samples: dict[str, list[float]] = field(default_factory=dict)

    def add(
        self,
        sample: PostureSample | None,
        *,
        keep_upright_unknown: bool = False,
    ) -> None:
        """Accumulate one posture sample.

        ``keep_upright_unknown`` is the H3 policy switch: when True, upright-only
        quantities (stature etc.) are kept for samples whose posture could not be
        DETERMINED (feet cut off) and dropped only when measured as not standing.
        Default False = legacy behaviour (drop unless measured upright) - the
        composed-stack A/B showed the permissive policy shifts the calibrated
        posture distributions enough to suppress facing-pair corroboration merges,
        so it must be an explicit, measured opt-in (`posture_keep_upright_unknown`).
        """

        if sample is None:
            return
        for name, value in sample.values.items():
            if not np.isfinite(value):
                continue
            if name in _UPRIGHT_ONLY:
                if keep_upright_unknown:
                    if sample.upright_known and not sample.upright:
                        continue
                elif not sample.upright:
                    continue
            self.samples.setdefault(name, []).append(float(value))

    def aggregate(
        self,
        *,
        min_samples: int = 8,
        autocorr_lag: float = 5.0,
        se_floor_m: float = 0.005,
    ) -> PostureAggregate | None:
        median: dict[str, float] = {}
        se: dict[str, float] = {}
        count: dict[str, int] = {}
        for name, values in self.samples.items():
            if len(values) < min_samples:
                continue
            data = np.asarray(values, dtype=float)
            centre = float(np.median(data))
            mad_sigma = 1.4826 * float(np.median(np.abs(data - centre)))
            n_eff = max(1.0, len(data) / max(autocorr_lag, 1.0))
            median[name] = centre
            se[name] = max(se_floor_m, mad_sigma / float(np.sqrt(n_eff)))
            count[name] = len(data)
        if not median:
            return None
        return PostureAggregate(median=median, se=se, count=count)


def posture_distance_z(
    a: PostureAggregate | None,
    b: PostureAggregate | None,
    *,
    sigma_sys: dict[str, float] | None = None,
    min_shared: int = 2,
    quantities: frozenset[str] | set[str] | None = None,
) -> tuple[float, int] | None:
    """RMS z-score between two tracklets' posture aggregates.

    Each shared quantity is normalized by its combined uncertainty: the two
    standard errors plus the cross-camera systematic sigma (measured by the cue
    calibration harness; falls back to :data:`DEFAULT_POSTURE_SIGMA_SYS`). Returns
    ``None`` (abstain) when fewer than ``min_shared`` quantities are shared, so an
    unobservable posture never fakes a verdict. ``quantities`` restricts the
    comparison (e.g. :data:`STATURE_QUANTITIES` when a bent/crouched body makes
    the planar shape quantities view-dependent between non-parallel cameras).
    """

    if a is None or b is None or not a.is_defined() or not b.is_defined():
        return None
    sigmas = dict(DEFAULT_POSTURE_SIGMA_SYS)
    if sigma_sys:
        sigmas.update({k: float(v) for k, v in sigma_sys.items() if np.isfinite(v) and v > 0})
    z_sq: list[float] = []
    for name in POSTURE_QUANTITIES:
        if quantities is not None and name not in quantities:
            continue
        if name not in a.median or name not in b.median:
            continue
        sigma = float(np.sqrt(
            a.se.get(name, 0.0) ** 2
            + b.se.get(name, 0.0) ** 2
            + sigmas.get(name, 0.08) ** 2
        ))
        if sigma <= 1e-9:
            continue
        z = (a.median[name] - b.median[name]) / sigma
        if np.isfinite(z):
            z_sq.append(float(z) ** 2)
    if len(z_sq) < int(min_shared):
        return None
    return float(np.sqrt(np.mean(z_sq))), len(z_sq)


def torso_anthropometric_ok(
    points3d: np.ndarray,
    joint_conf: np.ndarray,
    *,
    shoulder_width_m: tuple[float, float] = (0.25, 0.65),
    hip_width_m: tuple[float, float] = (0.15, 0.55),
    torso_len_m: tuple[float, float] = (0.35, 0.85),
    torso_tilt_max_deg: float = 75.0,
    min_conf: float = 0.3,
) -> bool | None:
    """Soft plausibility check that a triangulated torso is a single upright human.

    Returns ``None`` (abstain) when the torso is not confidently observed, ``True``
    when dimensions/verticality are human-plausible, ``False`` for a gross violation
    (a hallmark of a cross-camera chimera merging two different people). Intended only
    to *down-weight* confidence, never to veto a cluster.
    """

    points3d = np.asarray(points3d, dtype=float).reshape(-1, 3)
    joint_conf = np.asarray(joint_conf, dtype=float).reshape(-1)
    torso = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]
    ok = np.isfinite(points3d[torso]).all() and np.all(joint_conf[torso] >= min_conf)
    if not ok:
        return None

    shoulder_mid = (points3d[L_SHOULDER] + points3d[R_SHOULDER]) / 2.0
    hip_mid = (points3d[L_HIP] + points3d[R_HIP]) / 2.0
    shoulder_width = float(np.linalg.norm(points3d[L_SHOULDER] - points3d[R_SHOULDER]))
    hip_width = float(np.linalg.norm(points3d[L_HIP] - points3d[R_HIP]))
    spine_vec = shoulder_mid - hip_mid
    torso_len = float(np.linalg.norm(spine_vec))
    if torso_len <= 1e-6:
        return False

    tilt_deg = float(np.degrees(np.arccos(np.clip(abs(spine_vec[2]) / torso_len, 0.0, 1.0))))
    within = (
        shoulder_width_m[0] <= shoulder_width <= shoulder_width_m[1]
        and hip_width_m[0] <= hip_width <= hip_width_m[1]
        and torso_len_m[0] <= torso_len <= torso_len_m[1]
        and tilt_deg <= torso_tilt_max_deg
        and shoulder_mid[2] > hip_mid[2]  # shoulders above hips
    )
    return bool(within)

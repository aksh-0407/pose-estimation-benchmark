from __future__ import annotations

import numpy as np

from identity.common.pose_shape import (
    PoseProportions,
    SEGMENT_COUNT,
    descriptor_distance,
    limb_proportion_descriptor,
    merge_descriptor,
    torso_anthropometric_ok,
)

# COCO-17 indices
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 11, 12, 13, 14, 15, 16


def _human_skeleton() -> tuple[np.ndarray, np.ndarray]:
    points = np.full((17, 3), np.nan)
    points[L_SHOULDER] = [-0.20, 0.0, 1.40]
    points[R_SHOULDER] = [0.20, 0.0, 1.40]
    points[L_ELBOW] = [-0.25, 0.0, 1.10]
    points[R_ELBOW] = [0.25, 0.0, 1.10]
    points[L_WRIST] = [-0.28, 0.0, 0.85]
    points[R_WRIST] = [0.28, 0.0, 0.85]
    points[L_HIP] = [-0.12, 0.0, 0.95]
    points[R_HIP] = [0.12, 0.0, 0.95]
    points[L_KNEE] = [-0.13, 0.0, 0.50]
    points[R_KNEE] = [0.13, 0.0, 0.50]
    points[L_ANKLE] = [-0.13, 0.0, 0.05]
    points[R_ANKLE] = [0.13, 0.0, 0.05]
    conf = np.ones(17)
    conf[[0, 1, 2, 3, 4]] = 0.0  # face joints not used
    return points, conf


def _rotation_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_descriptor_is_scale_translation_and_rotation_invariant():
    points, conf = _human_skeleton()
    base = limb_proportion_descriptor(points, conf, n_views=3)
    assert base.is_defined()
    assert int(base.mask.sum()) == SEGMENT_COUNT

    transformed = (points @ _rotation_z(0.7).T) * 1.3 + np.array([5.0, -2.0, 0.0])
    other = limb_proportion_descriptor(transformed, conf, n_views=2)
    distance = descriptor_distance(base, other)
    assert distance is not None
    assert distance < 1e-6  # identical proportions regardless of pose/scale/position


def test_descriptor_distinguishes_different_body_proportions():
    points, conf = _human_skeleton()
    base = limb_proportion_descriptor(points, conf, n_views=3)
    stretched = points.copy()
    stretched[L_ANKLE] = [-0.13, 0.0, -0.35]  # much longer shins
    stretched[R_ANKLE] = [0.13, 0.0, -0.35]
    stretched[L_SHOULDER] = [-0.10, 0.0, 1.40]  # narrower shoulders
    stretched[R_SHOULDER] = [0.10, 0.0, 1.40]
    other = limb_proportion_descriptor(stretched, conf, n_views=3)
    distance = descriptor_distance(base, other)
    assert distance is not None
    assert distance > 0.1


def test_distance_is_none_without_enough_shared_segments():
    points, conf = _human_skeleton()
    full = limb_proportion_descriptor(points, conf, n_views=3)
    sparse_conf = np.zeros(17)
    sparse_conf[[L_SHOULDER, R_SHOULDER]] = 1.0  # only shoulder_width segment valid
    sparse = limb_proportion_descriptor(points, sparse_conf, n_views=2)
    assert descriptor_distance(full, sparse, min_shared=4) is None
    assert descriptor_distance(full, None) is None


def test_parallax_gating_excludes_low_parallax_segments():
    points, conf = _human_skeleton()
    parallax_ok = np.ones(17, dtype=bool)
    parallax_ok[[L_KNEE, R_KNEE]] = False  # knees triangulated with poor parallax
    descriptor = limb_proportion_descriptor(points, conf, parallax_ok, n_views=2)
    # Any segment touching a knee (thigh_*, shin_*) must be dropped.
    from identity.common.pose_shape import SEGMENT_NAMES

    for name in ("thigh_l", "thigh_r", "shin_l", "shin_r"):
        assert not descriptor.mask[SEGMENT_NAMES.index(name)]
    assert descriptor.mask[SEGMENT_NAMES.index("shoulder_width")]


def test_merge_descriptor_ema_blends_and_adopts_new_segments():
    points, conf = _human_skeleton()
    a = limb_proportion_descriptor(points, conf, n_views=2)
    # b only has the arms confident; merge must keep a's legs and blend the arms.
    b_conf = np.zeros(17)
    b_conf[[L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST]] = 1.0
    b = limb_proportion_descriptor(points * 1.1, b_conf, n_views=1)
    merged = merge_descriptor(a, b, rate=0.5)
    assert merged is not None and merged.is_defined()
    assert int(merged.mask.sum()) == int(a.mask.sum())  # union, a already had all
    # merging None / undefined is a no-op that returns the accumulator
    assert merge_descriptor(a, None) is a


def test_torso_anthropometric_ok_human_chimera_and_abstain():
    points, conf = _human_skeleton()
    assert torso_anthropometric_ok(points, conf) is True

    chimera = points.copy()
    chimera[L_SHOULDER] = [-1.0, 0.0, 1.40]  # 2 m shoulders => impossible human
    chimera[R_SHOULDER] = [1.0, 0.0, 1.40]
    assert torso_anthropometric_ok(chimera, conf) is False

    upside_down = points.copy()
    upside_down[[L_SHOULDER, R_SHOULDER]] = points[[L_HIP, R_HIP]]
    upside_down[[L_HIP, R_HIP]] = points[[L_SHOULDER, R_SHOULDER]]
    assert torso_anthropometric_ok(upside_down, conf) is False  # shoulders below hips

    unobserved = conf.copy()
    unobserved[[L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]] = 0.0
    assert torso_anthropometric_ok(points, unobserved) is None  # abstain


def test_json_round_trip_preserves_descriptor():
    points, conf = _human_skeleton()
    descriptor = limb_proportion_descriptor(points, conf, n_views=4)
    restored = PoseProportions.from_json(descriptor.to_json())
    assert restored is not None
    assert descriptor_distance(descriptor, restored) == 0.0
    assert restored.n_views == 4
    assert PoseProportions.from_json(None) is None

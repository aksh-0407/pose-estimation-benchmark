from __future__ import annotations

import numpy as np

from identity.common.geometry import ground_from_reprojection_ex
from identity.common.triangulation import point_covariance_3d
from identity.p3_association.cluster_lift import cluster_purity, lift_frame

# Realistic-focal cameras (f = 1000 px) with distinct viewpoints: FRONT looks +z,
# SIDE is ~orthogonal (good parallax), FACING looks -z from the far side (the
# low-parallax facing-pair geometry). All are elevated so the ground homography
# is well-posed.
def _camera(R: np.ndarray, C: np.ndarray, f: float = 1000.0) -> np.ndarray:
    K = np.array([[f, 0.0, 960.0], [0.0, f, 540.0], [0.0, 0.0, 1.0]])
    Rt = np.hstack([R, (-R @ np.asarray(C, float)).reshape(3, 1)])
    return K @ Rt


def _rot_y(deg: float) -> np.ndarray:
    r = np.radians(deg)
    return np.array([
        [np.cos(r), 0.0, np.sin(r)],
        [0.0, 1.0, 0.0],
        [-np.sin(r), 0.0, np.cos(r)],
    ])


def _rot_x(deg: float) -> np.ndarray:
    r = np.radians(deg)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(r), -np.sin(r)],
        [0.0, np.sin(r), np.cos(r)],
    ])


# World: z up is NOT assumed here - these cameras look along +/-z with y up-ish and
# a slight downward tilt so the ground plane (z=0 in solver terms is the XY plane
# of the ground test below) projects non-degenerately.
P_FRONT = _camera(_rot_x(15.0), np.array([0.0, 3.0, -6.0]))
P_SIDE = _camera(_rot_x(12.0) @ _rot_y(80.0), np.array([-8.0, 3.0, 6.5]))
P_FACING = _camera(_rot_x(15.0) @ _rot_y(178.0), np.array([0.3, 3.0, 18.0]))


def _project(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    h = P @ np.append(X, 1.0)
    return h[:2] / h[2]


def _skeleton_3d(offset_x: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(3)
    pts = rng.uniform(-0.4, 0.4, (17, 3)) + np.array([offset_x, 1.0, 6.0])
    return pts


def _views_of(points3d: np.ndarray, cams: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {}
    for cam_id, P in cams.items():
        obs = np.array([[*_project(P, X), 0.9] for X in points3d])
        out[cam_id] = obs
    return out


def test_lift_frame_recovers_skeleton_with_small_torso_residuals():
    truth = _skeleton_3d()
    members = _views_of(truth, {"cam_01": P_FRONT, "cam_02": P_SIDE})
    lift = lift_frame(members, {"cam_01": P_FRONT, "cam_02": P_SIDE}, compute_cov=True)
    assert lift is not None
    np.testing.assert_allclose(lift.points3d, truth, atol=1e-6)
    assert all(v < 0.5 for v in lift.torso_residual_by_cam.values())
    assert lift.cov_diag_m2 is not None and np.isfinite(lift.cov_diag_m2).all()
    assert lift.pelvis_cov_m2 is not None
    purity = cluster_purity([lift])
    assert not purity.chimera_suspect
    assert purity.torso_residual_p95 < 1.0


def test_chimera_intruder_camera_carries_one_sided_torso_bias():
    person_a = _skeleton_3d(0.0)
    person_b = _skeleton_3d(1.6)          # a DIFFERENT person 1.6 m away
    cams = {"cam_01": P_FRONT, "cam_02": P_SIDE, "cam_03": P_FACING}
    members = _views_of(person_a, cams)
    intruder = _views_of(person_b, {"cam_03": P_FACING})
    members["cam_03"] = intruder["cam_03"]   # cam_03's member is the wrong person

    lifts = []
    for _ in range(5):
        lift = lift_frame(members, cams)
        assert lift is not None
        lifts.append(lift)
    purity = cluster_purity(lifts, chimera_torso_residual_px=20.0, chimera_frame_fraction=0.3)
    assert purity.chimera_suspect
    assert purity.worst_camera == "cam_03"
    clean = purity.per_camera_residual["cam_01"][0]
    dirty = purity.per_camera_residual["cam_03"][0]
    assert dirty > 4 * clean               # the intruder bias is one-sided


def test_point_covariance_elongated_on_facing_pair():
    X = np.array([0.2, 1.1, 6.0])
    good_pair = np.stack([P_FRONT, P_SIDE])
    facing_pair = np.stack([P_FRONT, P_FACING])
    obs_good = np.array([_project(P, X) for P in good_pair])
    obs_facing = np.array([_project(P, X) for P in facing_pair])
    # Perfect observations give ~zero residual; perturb slightly for a nonzero sigma.
    obs_good += 0.5
    obs_facing += 0.5
    cov_good = point_covariance_3d(X, obs_good, good_pair)
    cov_facing = point_covariance_3d(X, obs_facing, facing_pair)
    assert cov_good is not None and cov_facing is not None
    # Facing pair: depth (z) variance dwarfs lateral; good pair: comparable axes.
    aniso_facing = cov_facing[2, 2] / max(cov_facing[0, 0], 1e-12)
    aniso_good = cov_good[2, 2] / max(cov_good[0, 0], 1e-12)
    assert aniso_facing > 10 * aniso_good


def test_ground_from_reprojection_ex_covariance_shrinks_with_more_views():
    X = np.array([0.5, -0.3])
    feet, projections = [], []
    for P in (P_FRONT, P_SIDE, P_FACING):
        feet.append(_project(P, np.array([X[0], X[1], 0.0])))
        projections.append(P)
    feet = np.asarray(feet) + np.random.default_rng(0).normal(0, 1.0, (3, 2))
    two_xy, two_cov = ground_from_reprojection_ex(feet[:2], np.asarray(projections[:2]))
    three_xy, three_cov = ground_from_reprojection_ex(feet, np.asarray(projections))
    assert two_cov is not None and three_cov is not None
    assert np.isfinite(two_xy).all() and np.isfinite(three_xy).all()
    assert np.trace(three_cov) < np.trace(two_cov) * 1.5   # more views: not worse
    # single view: no GN covariance (caller uses the per-view Jacobian model)
    single_xy, single_cov = ground_from_reprojection_ex(feet[:1], np.asarray(projections[:1]))
    assert single_cov is None and np.isfinite(single_xy).all()


def test_chimera_veto_pass_lets_refine_evict_intruder():
    from identity.p3_association.config import P3AssociationConfig
    from identity.p3_association.tracklet_graph import TrackletGraphBuilder, _pair_key

    projections = {"cam_01": P_FRONT, "cam_02": P_SIDE, "cam_03": P_FACING}
    config = P3AssociationConfig(
        association_mode="tracklet_graph",
        graph_split_enabled=True,
        graph_shape_min_frames=4,
        graph_lift_stride=1,
        graph_rescue_min_covis=1_000_000,   # no rescue interference
    )
    builder = TrackletGraphBuilder(config, projections)

    person_a = _skeleton_3d(0.0)
    person_b = _skeleton_3d(1.6)
    # Three chunks: cam_01/cam_02 see person A; cam_03's member is person B (the
    # intruder a bad merge welded in).
    chunks = {}
    for cam_id, truth in (("cam_01", person_a), ("cam_02", person_a), ("cam_03", person_b)):
        key = (cam_id, f"{cam_id}_trk_X", 0)
        from identity.p3_association.tracklet_graph import _ChunkState
        chunk = _ChunkState(key=key)
        for frame in range(0, 40, 5):
            obs = np.array([[*(_project(projections[cam_id], X)), 0.9] for X in truth])
            chunk.kp_samples[frame] = obs
            chunk.frames.append(frame)
            chunk.ground_by_frame[frame] = truth[[11, 12]].mean(axis=0)[:2]
        builder._chunks[key] = chunk
        chunks[cam_id] = key
    builder._support_lookup = {}

    cluster_members = {0: sorted(chunks.values())}
    cluster_of = {key: 0 for key in chunks.values()}
    # Welded by (now-outvoted) positive edges.
    llr_lookup = {
        _pair_key(a, b): 1.0
        for i, a in enumerate(sorted(chunks.values()))
        for b in sorted(chunks.values())[i + 1:]
    }

    evictions = builder._chimera_veto_pass(cluster_members, cluster_of, llr_lookup)
    assert evictions == 1                    # cam_03's chunk evicted surgically
    intruder = chunks["cam_03"]
    assert llr_lookup[_pair_key(intruder, chunks["cam_01"])] == -6.0
    # The intruder is alone; person A's two chunks stay together.
    assert cluster_of[intruder] != cluster_of[chunks["cam_01"]]
    assert cluster_of[chunks["cam_01"]] == cluster_of[chunks["cam_02"]]

    # Refinement afterwards must not undo the split (the veto holds) nor scatter
    # the innocent pair (their mutual affinity is still positive).
    builder._refine(cluster_members, cluster_of, llr_lookup)
    assert cluster_of[intruder] != cluster_of[chunks["cam_01"]]
    assert cluster_of[chunks["cam_01"]] == cluster_of[chunks["cam_02"]]

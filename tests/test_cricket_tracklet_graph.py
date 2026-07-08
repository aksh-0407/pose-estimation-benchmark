"""Tests for the tracklet-graph identity layer (P3 ``tracklet_graph`` mode)."""

from __future__ import annotations

import numpy as np

from pose_estimation.cricket.geometry import (
    fuse_ground_estimates,
    ground_covariance,
    ground_mahalanobis_sq,
    pixel_to_ground_xy,
)
from pose_estimation.cricket.pose_shape import (
    PostureAccumulator,
    PostureSample,
    ground_anchored_skeleton,
    posture_distance_z,
    posture_from_skeleton,
)
from pose_estimation.cricket.tracking_metrics import pair_link_churn
from scripts.association.associator import Detection3
from scripts.association.config import P3AssociationConfig
from scripts.association.cue_calibration import CueCalibration, fit_cue_calibration
from scripts.association.tracklet_graph import TrackletGraphBuilder


# ----------------------------------------------------------------- synthetic rig

def _camera(center: np.ndarray, target: np.ndarray, f: float = 1500.0) -> np.ndarray:
    """Standard pinhole P = K [R | t] looking from ``center`` at ``target``, z-up."""

    center = np.asarray(center, dtype=float)
    forward = np.asarray(target, dtype=float) - center
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.stack([right, down, forward])
    K = np.array([[f, 0.0, 960.0], [0.0, f, 540.0], [0.0, 0.0, 1.0]])
    return K @ np.concatenate([R, (-R @ center).reshape(3, 1)], axis=1)


def _project(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    p = P @ np.append(np.asarray(X, dtype=float), 1.0)
    return p[:2] / p[2]


CAM_A = _camera(np.array([0.0, -55.0, 10.0]), np.zeros(3))
CAM_B = _camera(np.array([0.0, 58.0, 11.0]), np.zeros(3))  # facing pair
PROJECTIONS = {"cam_01": CAM_A, "cam_04": CAM_B}


def _detection(cam_id: str, player_index: int, ground: np.ndarray,
               local_track_id: str | None) -> Detection3:
    P = PROJECTIONS[cam_id]
    foot = _project(P, np.array([ground[0], ground[1], 0.0]))
    head = _project(P, np.array([ground[0], ground[1], 1.8]))
    height_px = abs(foot[1] - head[1])
    bbox = [foot[0] - height_px * 0.15, foot[1] - height_px, height_px * 0.3, height_px]
    return Detection3(
        cam_id=cam_id,
        player_index=player_index,
        bbox_xywh_px=[float(v) for v in bbox],
        keypoints_px=np.zeros((17, 2)),
        keypoint_conf=np.zeros(17),
        confidence=0.9,
        local_track_id=local_track_id,
        ground_xy=np.asarray(ground, dtype=float),
    )


def _graph_config(**overrides) -> P3AssociationConfig:
    values = dict(
        association_mode="tracklet_graph",
        appearance_enabled=False,
        posture_enabled=False,
        graph_motion_enabled=False,
        pose_descriptor_enabled=False,
        graph_min_covis_frames=10,
        # These unit tests exercise mechanics with the ground cue only, so the
        # production corroboration threshold (which exceeds the single-cue cap
        # on purpose) is lowered.
        graph_llr_merge_threshold=1.0,
    )
    values.update(overrides)
    return P3AssociationConfig(**values)


def _observe_two_players(builder: TrackletGraphBuilder, frames: int = 80,
                         noise: float = 0.08, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    pos = {"A": np.array([0.0, 0.0]), "B": np.array([2.5, 0.0])}
    for frame in range(frames):
        dets: dict[str, list[Detection3]] = {"cam_01": [], "cam_04": []}
        for index, name in enumerate(("A", "B")):
            for cam_id in ("cam_01", "cam_04"):
                noisy = pos[name] + rng.normal(0.0, noise, size=2)
                dets[cam_id].append(
                    _detection(cam_id, index, noisy, f"{cam_id}_trk_{name}")
                )
        builder.observe_frame(frame, dets)


# -------------------------------------------------------------- ground covariance

def test_ground_covariance_grows_with_distance_and_is_anisotropic():
    near_px = _project(CAM_A, np.array([0.0, -20.0, 0.0]))
    far_px = _project(CAM_A, np.array([0.0, 40.0, 0.0]))
    near_cov = ground_covariance(near_px, CAM_A, sigma_px=2.0, var_floor_m=0.0)
    far_cov = ground_covariance(far_px, CAM_A, sigma_px=2.0, var_floor_m=0.0)
    assert np.trace(far_cov) > np.trace(near_cov) * 3
    # Elongated along the viewing ray (world y for this camera).
    eigvals, eigvecs = np.linalg.eigh(far_cov)
    principal = eigvecs[:, np.argmax(eigvals)]
    assert abs(principal[1]) > abs(principal[0])
    assert max(eigvals) / max(min(eigvals), 1e-12) > 4


def test_ground_covariance_invalid_inputs_are_inf():
    cov = ground_covariance(np.array([np.nan, 1.0]), CAM_A)
    assert not np.isfinite(cov).all()


def test_ground_mahalanobis_and_fusion():
    xy = np.array([1.0, 2.0])
    cov = np.diag([0.04, 0.04])
    assert ground_mahalanobis_sq(xy, cov, xy, cov) == 0.0
    m2 = ground_mahalanobis_sq(xy, cov, xy + np.array([0.4, 0.0]), cov)
    assert abs(m2 - 2.0) < 1e-9  # 0.4^2 / (0.04 + 0.04)

    fused, fused_cov = fuse_ground_estimates(
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        np.array([np.diag([0.01, 0.01]), np.diag([0.09, 0.09])]),
    )
    assert fused[0] < 0.5  # pulled toward the more certain estimate
    assert np.trace(fused_cov) < 0.02


# ----------------------------------------------------------- billboard pose lift

def test_ground_anchored_skeleton_recovers_metric_heights():
    ground = np.array([1.5, 2.0])
    world = np.full((17, 3), np.nan)
    world[0] = [1.5, 2.0, 1.72]            # nose
    world[5] = [1.3, 2.0, 1.45]            # shoulders
    world[6] = [1.7, 2.0, 1.45]
    world[11] = [1.4, 2.0, 0.95]           # hips
    world[12] = [1.6, 2.0, 0.95]
    world[15] = [1.4, 2.0, 0.05]           # ankles
    world[16] = [1.6, 2.0, 0.05]
    conf = np.zeros(17)
    keypoints = np.zeros((17, 2))
    for j in range(17):
        if np.isfinite(world[j]).all():
            keypoints[j] = _project(CAM_A, world[j])
            conf[j] = 0.9
    foot = _project(CAM_A, np.array([ground[0], ground[1], 0.0]))

    points3d, valid = ground_anchored_skeleton(keypoints, conf, foot, CAM_A, min_conf=0.3)
    # Off-axis joints sit millimetres off the billboard plane at ~57 m, so the
    # recovery is near-exact, not exact: centimetre tolerance is the honest claim.
    for j in (0, 5, 6, 11, 12, 15, 16):
        assert valid[j]
        assert np.allclose(points3d[j], world[j], atol=0.02)
        assert abs(points3d[j, 2] - world[j, 2]) < 0.005  # heights are the tight axis

    sample = posture_from_skeleton(points3d, valid)
    assert sample is not None and sample.upright
    assert abs(sample.values["head_top_m"] - 1.72) < 0.005
    assert abs(sample.values["shoulder_w_m"] - 0.4) < 0.01


def test_ground_anchored_skeleton_rejects_implausible_and_low_conf():
    foot = _project(CAM_A, np.array([0.0, 0.0, 0.0]))
    keypoints = np.zeros((17, 2))
    keypoints[0] = _project(CAM_A, np.array([0.0, 0.0, 5.0]))   # 5 m head: implausible
    keypoints[5] = _project(CAM_A, np.array([0.0, 0.0, 1.4]))
    conf = np.zeros(17)
    conf[0] = 0.9
    conf[5] = 0.1  # below min_conf
    _points3d, valid = ground_anchored_skeleton(keypoints, conf, foot, CAM_A, min_conf=0.3)
    assert not valid.any()


def test_posture_accumulator_and_distance_z():
    rng = np.random.default_rng(3)

    def accumulate(height: float) -> PostureAccumulator:
        acc = PostureAccumulator()
        for _ in range(60):
            acc.add(PostureSample(
                values={"head_top_m": height + rng.normal(0, 0.03),
                        "shoulder_h_m": height - 0.3 + rng.normal(0, 0.03)},
                upright=True,
            ))
        return acc

    tall = accumulate(1.85).aggregate()
    tall_again = accumulate(1.85).aggregate()
    short = accumulate(1.65).aggregate()
    assert tall is not None and short is not None
    assert abs(tall.median["head_top_m"] - 1.85) < 0.02

    same = posture_distance_z(tall, tall_again)
    diff = posture_distance_z(tall, short)
    assert same is not None and diff is not None
    assert same[0] < 1.5 < diff[0]

    # Abstention: no shared quantities.
    other = PostureAccumulator()
    for _ in range(20):
        other.add(PostureSample(values={"hip_w_m": 0.3}, upright=True))
    assert posture_distance_z(tall, other.aggregate()) is None


def test_crouching_frames_are_excluded_from_stature():
    acc = PostureAccumulator()
    acc.add(PostureSample(values={"head_top_m": 1.2}, upright=False))  # crouching
    acc.add(PostureSample(values={"shoulder_w_m": 0.4}, upright=False))
    assert acc.aggregate(min_samples=1) is not None
    agg = acc.aggregate(min_samples=1)
    assert "head_top_m" not in agg.median   # upright-gated
    assert "shoulder_w_m" in agg.median     # shape quantities keep all frames


# --------------------------------------------------------------- cue calibration

def test_fit_cue_calibration_separable_and_inverted():
    rng = np.random.default_rng(11)
    same = list(rng.normal(0.8, 0.3, 300))
    diff = list(rng.normal(4.0, 1.0, 300))
    calibration = fit_cue_calibration(
        same_samples={"ground_maha": same}, diff_samples={"ground_maha": diff},
    )
    dist = calibration.distributions["ground_maha"]
    assert dist.fitted and dist.d_prime() > 2.0
    assert calibration.llr("ground_maha", 0.8) > 1.0
    assert calibration.llr("ground_maha", 4.0) < -1.0
    assert calibration.llr("ground_maha", None) == 0.0

    # A cue whose populations invert collapses to ~zero information, never lies.
    inverted = fit_cue_calibration(
        same_samples={"appearance": list(rng.normal(0.5, 0.05, 100))},
        diff_samples={"appearance": list(rng.normal(0.2, 0.05, 100))},
    )
    assert abs(inverted.llr("appearance", 0.2)) < 0.2

    # Thin data keeps the conservative default.
    thin = fit_cue_calibration(same_samples={"posture_z": [1.0]}, diff_samples={})
    assert not thin.distributions["posture_z"].fitted


def test_calibration_roundtrip(tmp_path):
    calibration = fit_cue_calibration(
        same_samples={"ground_maha": list(np.random.default_rng(0).normal(1, 0.3, 100))},
        diff_samples={"ground_maha": list(np.random.default_rng(1).normal(5, 1.0, 100))},
        posture_same_deltas={"head_top_m": [0.03] * 10},
        anchor_pair_count=3, diff_pair_count=5,
    )
    path = tmp_path / "cue_calibration.json"
    calibration.save(path)
    loaded = CueCalibration.load(path)
    assert loaded.anchor_pair_count == 3
    assert loaded.distributions["ground_maha"].fitted
    assert abs(loaded.posture_sigma_sys["head_top_m"] - 0.03) < 1e-9


# ----------------------------------------------------------------- tracklet graph

def test_graph_binds_players_across_facing_cameras():
    config = _graph_config()
    builder = TrackletGraphBuilder(config, PROJECTIONS)
    _observe_two_players(builder)
    solution = builder.solve(CueCalibration())

    assert len(solution.clusters) == 2
    for keys in solution.clusters.values():
        cams = {key[0] for key in keys}
        tracklets = {key[1].rsplit("_", 1)[1] for key in keys}
        assert cams == {"cam_01", "cam_04"}   # both cameras joined
        assert len(tracklets) == 1            # never mixes players A and B


def test_graph_emits_stable_bindings_per_frame():
    config = _graph_config()
    builder = TrackletGraphBuilder(config, PROJECTIONS)
    _observe_two_players(builder)
    solution = builder.solve(CueCalibration())

    rng = np.random.default_rng(21)
    seen_bindings: dict[str, set[str]] = {}
    for frame in (5, 40, 70):
        dets = {
            "cam_01": [
                _detection("cam_01", 0, np.array([0.0, 0.0]) + rng.normal(0, 0.05, 2), "cam_01_trk_A"),
                _detection("cam_01", 1, np.array([2.5, 0.0]) + rng.normal(0, 0.05, 2), "cam_01_trk_B"),
            ],
            "cam_04": [
                _detection("cam_04", 0, np.array([0.0, 0.0]) + rng.normal(0, 0.05, 2), "cam_04_trk_A"),
                _detection("cam_04", 1, np.array([2.5, 0.0]) + rng.normal(0, 0.05, 2), "cam_04_trk_B"),
            ],
        }
        correspondences = builder.emit_frame(frame, dets, solution, PROJECTIONS)
        bound = [corr for corr in correspondences if corr.binding_id is not None]
        assert len(bound) == 2
        for corr in bound:
            assert set(corr.members) == {"cam_01", "cam_04"}
            player = {det.local_track_id.rsplit("_", 1)[1] for det in corr.members.values()}
            assert len(player) == 1
            seen_bindings.setdefault(player.pop(), set()).add(corr.binding_id)
    # The SAME binding id every frame — no flicker by construction.
    assert all(len(bindings) == 1 for bindings in seen_bindings.values())


def test_graph_cannot_link_same_camera_overlap():
    config = _graph_config()
    builder = TrackletGraphBuilder(config, PROJECTIONS)
    rng = np.random.default_rng(5)
    # Two people 0.6 m apart, both visible in BOTH cameras the whole time: the
    # same-camera overlap must force them into different clusters even though
    # every cross-camera pair falls inside the ground gate.
    for frame in range(60):
        dets: dict[str, list[Detection3]] = {"cam_01": [], "cam_04": []}
        for index, (name, base) in enumerate((("A", 0.0), ("B", 0.6))):
            for cam_id in ("cam_01", "cam_04"):
                noisy = np.array([base, 0.0]) + rng.normal(0, 0.05, 2)
                dets[cam_id].append(_detection(cam_id, index, noisy, f"{cam_id}_trk_{name}"))
        builder.observe_frame(frame, dets)
    solution = builder.solve(CueCalibration())

    for keys in solution.clusters.values():
        per_cam: dict[str, int] = {}
        for key in keys:
            per_cam[key[0]] = per_cam.get(key[0], 0) + 1
        assert all(count == 1 for count in per_cam.values())


def test_purity_split_on_teleporting_tracklet():
    # Low binding_min_single_frames: this test exercises the split->separate
    # identities mechanics; demotion of short fragments is tested separately.
    config = _graph_config(binding_min_single_frames=25)
    builder = TrackletGraphBuilder(config, PROJECTIONS)
    rng = np.random.default_rng(9)
    for frame in range(60):
        # One P2 tracklet that jumps 6 m mid-way: a P2 identity switch.
        centre = np.array([0.0, 0.0]) if frame < 30 else np.array([6.0, 0.0])
        noisy = centre + rng.normal(0, 0.05, 2)
        dets = {"cam_01": [_detection("cam_01", 0, noisy, "cam_01_trk_X")]}
        builder.observe_frame(frame, dets)
    assert builder.diagnostics["purity_splits"] == 1
    solution = builder.solve(CueCalibration())
    chunks = {key for key in solution.binding_of_chunk if key[1] == "cam_01_trk_X"}
    assert len(chunks) == 2
    assert len({solution.binding_of_chunk[key] for key in chunks}) == 2


def test_emit_frame_leaves_untracked_as_singletons():
    config = _graph_config()
    builder = TrackletGraphBuilder(config, PROJECTIONS)
    _observe_two_players(builder, frames=30)
    solution = builder.solve(CueCalibration())
    dets = {
        "cam_01": [
            _detection("cam_01", 0, np.array([0.0, 0.0]), "cam_01_trk_A"),
            _detection("cam_01", 1, np.array([5.0, 3.0]), None),  # untracked
        ],
        "cam_04": [_detection("cam_04", 0, np.array([0.0, 0.0]), "cam_04_trk_A")],
    }
    correspondences = builder.emit_frame(10, dets, solution, PROJECTIONS)
    singles = [corr for corr in correspondences if corr.single_camera]
    assert len(singles) == 1 and singles[0].binding_id is None


# ------------------------------------------------------------------- churn metric

def test_pair_link_churn_zero_for_stable_and_positive_for_flicker():
    def row(frame: int, together: bool) -> dict:
        members_a = [{"cam_id": "cam_01", "local_track_id": "t1"},
                     {"cam_id": "cam_04", "local_track_id": "t2"}]
        if together:
            clusters = [{"members": members_a}]
        else:
            clusters = [{"members": [members_a[0]]}, {"members": [members_a[1]]}]
        return {"frame_index": frame, "clusters": clusters}

    stable = pair_link_churn([row(f, True) for f in range(10)])
    assert stable["pair_link_churn_rate"] == 0.0
    flicker = pair_link_churn([row(f, f % 2 == 0) for f in range(10)])
    assert flicker["pair_link_broken_count"] > 0


# --------------------------------------------- feet approximation + synthetics

def test_pixel_to_plane_recovers_position_above_feet():
    from pose_estimation.cricket.geometry import pixel_to_plane_xy
    feet = np.array([3.0, -1.0])
    hip_world = np.array([3.0, -1.0, 0.93])
    hip_px = _project(CAM_A, hip_world)
    xy = pixel_to_plane_xy(hip_px, CAM_A, 0.93)
    assert np.allclose(xy, feet, atol=1e-9)


def test_upper_body_ground_estimate_prefers_hips_then_falls_back():
    from pose_estimation.cricket.geometry import upper_body_ground_estimate
    feet = np.array([2.0, 5.0])
    keypoints = np.zeros((17, 2))
    conf = np.zeros(17)
    keypoints[11] = _project(CAM_A, [1.9, 5.0, 0.93])
    keypoints[12] = _project(CAM_A, [2.1, 5.0, 0.93])
    conf[[11, 12]] = 0.8
    bbox = [0.0, 0.0, 100.0, 400.0]
    xy, kind = upper_body_ground_estimate(keypoints, conf, bbox, CAM_A)
    assert kind == "hips" and np.allclose(xy, feet, atol=0.02)

    # No confident keypoints at all: bbox top as the head crown.
    head_px = _project(CAM_A, [2.0, 5.0, 1.78])
    bbox = [head_px[0] - 50.0, head_px[1], 100.0, 500.0]
    xy, kind = upper_body_ground_estimate(np.zeros((17, 2)), np.zeros(17), bbox, CAM_A)
    assert kind == "bbox_top" and np.allclose(xy, feet, atol=0.05)


def test_feet_approximation_is_sticky_per_tracklet():
    from scripts.association.tracklet_graph import apply_feet_approximation
    config = _graph_config()
    image_h = 1080
    true_feet = np.array([0.0, -10.0])
    hip_px = _project(CAM_A, [0.0, -10.0, 0.93])
    keypoints = np.zeros((17, 2)); keypoints[[11, 12]] = hip_px
    dead = np.zeros(17); dead[[11, 12]] = 0.8
    good = dead.copy(); good[[15, 16]] = 0.9
    cut_bbox = [hip_px[0] - 40, hip_px[1] - 300, 80.0, image_h - (hip_px[1] - 300)]

    def det(tid, kp_conf, bbox):
        return Detection3("cam_01", 0, bbox, keypoints, kp_conf, 0.9, tid, np.array([5.0, 5.0]))

    # Tracklet "cut": ankles dead + bbox at the frame bottom in 3/4 frames ->
    # sticky approximation on ALL its frames (even the one with a good ankle).
    # Tracklet "fine": mostly confident ankles -> never approximated, even on
    # its single bad frame (no per-frame flip-flopping).
    frames = {
        0: {"cam_01": [det("cut", dead, cut_bbox), det("fine", good, cut_bbox)]},
        1: {"cam_01": [det("cut", dead, cut_bbox), det("fine", good, cut_bbox)]},
        2: {"cam_01": [det("cut", dead, cut_bbox), det("fine", dead, cut_bbox)]},
        3: {"cam_01": [det("cut", good, cut_bbox), det("fine", good, [100, 100, 80, 200])]},
    }
    fixed = apply_feet_approximation(frames, {"cam_01": CAM_A}, {"cam_01": image_h}, config)
    for frame_index in range(4):
        cut_det, fine_det = fixed[frame_index]["cam_01"]
        assert cut_det.ground_approx
        assert np.allclose(cut_det.ground_xy, true_feet, atol=0.05)
        assert not fine_det.ground_approx
        assert np.allclose(fine_det.ground_xy, [5.0, 5.0])

    # Untracked detections decide per frame (no tracklet to vote over).
    frames = {0: {"cam_01": [det(None, dead, cut_bbox)]}}
    fixed = apply_feet_approximation(frames, {"cam_01": CAM_A}, {"cam_01": image_h}, config)
    assert fixed[0]["cam_01"][0].ground_approx


def test_synthetic_tracklets_chain_untracked_detections_and_bind():
    config = _graph_config()
    builder = TrackletGraphBuilder(config, PROJECTIONS)
    rng = np.random.default_rng(13)
    # A tracked player in cam_01 and an UNTRACKED persistent detection of the
    # same person in cam_04 (P2 failed it): the synthetic chain makes it a node
    # and the graph binds the pair.
    for frame in range(80):
        noisy = np.array([1.0, 0.0]) + rng.normal(0, 0.05, 2)
        dets = {
            "cam_01": [_detection("cam_01", 0, noisy, "cam_01_trk_A")],
            "cam_04": [_detection("cam_04", 0, noisy + rng.normal(0, 0.05, 2), None)],
        }
        builder.observe_frame(frame, dets)
    assert builder.diagnostics["synthetic_tracklets"] == 1
    solution = builder.solve(CueCalibration())
    multi = [keys for keys in solution.clusters.values() if len(keys) > 1]
    assert len(multi) == 1
    cams = {key[0] for key in multi[0]}
    assert cams == {"cam_01", "cam_04"}
    assert any("_syn_" in key[1] for key in multi[0])

    # Emission stamps the binding on the untracked detection too.
    dets = {
        "cam_01": [_detection("cam_01", 0, np.array([1.0, 0.0]), "cam_01_trk_A")],
        "cam_04": [_detection("cam_04", 0, np.array([1.0, 0.0]), None)],
    }
    corr = builder.emit_frame(40, dets, solution, PROJECTIONS)
    bound = [c for c in corr if c.binding_id is not None]
    assert len(bound) == 1 and set(bound[0].members) == {"cam_01", "cam_04"}


def test_short_fragments_are_demoted_and_trajectory_attach_recovers_them():
    config = _graph_config()
    builder = TrackletGraphBuilder(config, PROJECTIONS)
    rng = np.random.default_rng(17)
    # A player crosses the ground, seen continuously by both cameras — but in
    # cam_01 the P2 track shatters into disjoint short tracklets (dark-footage
    # failure mode). Fragments must ride the binding's trajectory, not mint ids.
    for frame in range(240):
        pos = np.array([-4.0 + frame * 0.03, 0.0]) + rng.normal(0, 0.05, 2)
        dets = {"cam_04": [_detection("cam_04", 0, pos, "cam_04_trk_A")], "cam_01": []}
        fragment = frame // 40  # six fragments too short to earn normal edges
        if frame % 40 < 15:
            dets["cam_01"].append(
                _detection("cam_01", 0, pos + rng.normal(0, 0.05, 2), f"cam_01_trk_f{fragment}")
            )
        builder.observe_frame(frame, dets)
    # A stray far-away short fragment must NOT attach anywhere.
    for frame in range(60, 80):
        dets = {"cam_01": [_detection("cam_01", 1, np.array([10.0, 8.0]), "cam_01_trk_stray")]}
        builder.observe_frame(frame, dets)
    solution = builder.solve(CueCalibration())

    bindings = {solution.binding_of_chunk.get(key)
                for key in solution.binding_of_chunk if key[1] != "cam_01_trk_stray"}
    assert len({b for b in bindings if b}) == 1  # one identity for the player
    attached = builder.diagnostics.get("fragments_attached", 0)
    assert attached >= 4  # the shattered cam_01 fragments joined the binding
    stray_chunks = [k for k in solution.binding_of_chunk if k[1] == "cam_01_trk_stray"]
    assert not stray_chunks  # demoted: no binding id for a 20-frame stray

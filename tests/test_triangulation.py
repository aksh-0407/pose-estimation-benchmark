import numpy as np

from identity.common.triangulation import (
    reprojection_errors_for_point,
    triangulate_point_dlt,
    triangulate_skeleton_ransac,
)


def test_weighted_dlt_triangulates_two_views():
    p1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=float)
    p2 = np.array([[1, 0, 0, -1], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=float)
    point = np.array([0.0, 0.0, 5.0])
    observations = np.array([[0.0, 0.0], [-0.2, 0.0]])
    result = triangulate_point_dlt(observations, np.stack([p1, p2]), np.ones(2))
    np.testing.assert_allclose(result, point, atol=1e-7)


def test_reprojection_preserves_negative_homogeneous_depth_sign():
    projection = -np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
        dtype=float,
    )
    point = np.array([1.0, 2.0, 5.0])
    errors = reprojection_errors_for_point(
        point,
        np.array([[0.2, 0.4]]),
        np.array([projection]),
    )
    assert errors[0] < 1e-12


def test_skeleton_ransac_rejects_bad_view():
    p1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=float)
    p2 = np.array([[1, 0, 0, -1], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=float)
    p3 = np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=float)
    keypoints = np.array(
        [
            [[0.0, 0.0, 1.0]],
            [[-0.2, 0.0, 1.0]],
            [[5.0, 5.0, 1.0]],
        ]
    )
    points3d, confidences, errors = triangulate_skeleton_ransac(
        keypoints,
        np.stack([p1, p2, p3]),
        reprojection_threshold_px=0.5,
    )
    np.testing.assert_allclose(points3d[0], np.array([0.0, 0.0, 5.0]), atol=1e-7)
    assert confidences[0] > 0.0
    assert errors[0] < 1e-7


def _translation_cameras(txs):
    return np.stack([
        np.array([[1, 0, 0, tx], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=float) for tx in txs
    ])


def test_irls_refit_downweights_biased_inlier_camera():
    # 4 stereo views of the true point [0, 0, 5]; one view carries a small y-bias
    # that is well inside the RANSAC reprojection gate, so it stays an inlier and
    # pulls the unweighted L2 re-fit off the true y. The Huber M-estimator should
    # down-weight it and land closer to truth.
    projections = _translation_cameras([0.0, -1.0, 1.0, 2.0])
    truth = np.array([0.0, 0.0, 5.0])
    obs = np.array([[tx / 5.0, 0.0] for tx in [0.0, -1.0, 1.0, 2.0]])
    obs[3, 1] += 0.05  # biased-but-inlier camera
    keypoints = np.concatenate([obs, np.ones((4, 1))], axis=1)[:, None, :]  # (4, 1, 3)

    p_plain, _, _ = triangulate_skeleton_ransac(
        keypoints, projections, reprojection_threshold_px=1.0,
    )
    p_robust, _, _ = triangulate_skeleton_ransac(
        keypoints, projections, reprojection_threshold_px=1.0,
        robust_refit=True, robust_huber_px=0.01,
    )
    err_plain = float(np.linalg.norm(p_plain[0] - truth))
    err_robust = float(np.linalg.norm(p_robust[0] - truth))
    assert err_robust < err_plain
    assert abs(p_robust[0, 1]) < abs(p_plain[0, 1])  # y pulled back toward 0


def test_irls_refit_matches_plain_on_clean_data():
    # With no outlier, IRLS converges back to the L2 solution (does no harm).
    projections = _translation_cameras([0.0, -1.0, 1.0, 2.0])
    obs = np.array([[tx / 5.0, 0.0] for tx in [0.0, -1.0, 1.0, 2.0]])
    keypoints = np.concatenate([obs, np.ones((4, 1))], axis=1)[:, None, :]
    p_plain, _, _ = triangulate_skeleton_ransac(keypoints, projections)
    p_robust, _, _ = triangulate_skeleton_ransac(
        keypoints, projections, robust_refit=True, robust_huber_px=8.0,
    )
    np.testing.assert_allclose(p_robust[0], p_plain[0], atol=1e-6)


def test_fill_occluded_joints_interpolates_temporal_gap():
    import numpy as np
    from identity.common.triangulation import fill_occluded_joints
    T, J = 5, 2
    seq = np.full((T, J, 3), np.nan)
    conf = np.zeros((T, J))
    # joint 0 valid at t=0 and t=4, missing 1-3 -> should linearly interpolate
    seq[0, 0] = [0.0, 0.0, 0.0]; seq[4, 0] = [4.0, 0.0, 0.0]; conf[0, 0] = conf[4, 0] = 0.9
    # joint 1 valid only at t=2 -> holds outward within gap
    seq[2, 1] = [1.0, 1.0, 1.0]; conf[2, 1] = 0.8
    out, oc = fill_occluded_joints(seq, conf, max_gap_frames=25)
    assert np.allclose(out[2, 0], [2.0, 0.0, 0.0])          # midpoint interp
    assert oc[2, 0] < 0.9 and oc[2, 0] > 0                   # reduced fill confidence
    assert np.isfinite(out[:, 1]).all()                     # joint 1 held to all frames
    assert np.allclose(out[0, 1], [1.0, 1.0, 1.0])


def test_cheirality_rejects_point_behind_cameras():
    from identity.common.triangulation import ransac_triangulate_point

    p1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=float)
    p2 = np.array([[1, 0, 0, -1], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=float)
    # Observations of a point BEHIND both cameras (z = -5): projecting
    # [0, 0, -5] and [1, 0, -5] through p1/p2 gives these pixels.
    behind = np.array([[0.0, 0.0], [0.2, 0.0]])
    legacy = ransac_triangulate_point(behind, np.stack([p1, p2]), np.ones(2))
    gated = ransac_triangulate_point(
        behind, np.stack([p1, p2]), np.ones(2), cheirality=True
    )
    assert legacy.point_xyz[2] < 0          # legacy happily returns the behind point
    assert not gated.inlier_mask.any()      # cheirality refuses every view


def test_cheirality_accepts_front_point_with_negative_scaled_projection():
    from identity.common.triangulation import ransac_triangulate_point

    p1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=float)
    p2 = np.array([[1, 0, 0, -1], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=float)
    observations = np.array([[0.0, 0.0], [-0.2, 0.0]])  # point [0,0,5], in front
    result = ransac_triangulate_point(
        observations, np.stack([p1, -p2]), np.ones(2), cheirality=True
    )
    np.testing.assert_allclose(result.point_xyz, [0.0, 0.0, 5.0], atol=1e-7)
    assert result.inlier_mask.all()         # negative overall scale must not flip the test


def test_butterworth_smooth_reduces_jitter_zero_phase():
    from identity.common.triangulation import butterworth_smooth

    rng = np.random.default_rng(7)
    t = np.arange(200)
    clean = np.stack([0.01 * t, np.sin(t / 40.0), np.zeros_like(t, dtype=float)], axis=1)
    noisy = clean + rng.normal(0.0, 0.05, clean.shape)
    seq = noisy[:, None, :]  # (T, 1, 3)
    out = butterworth_smooth(seq, fps=50.0, cutoff_hz=4.0)
    jitter_in = np.abs(np.diff(seq[:, 0], axis=0)).mean()
    jitter_out = np.abs(np.diff(out[:, 0], axis=0)).mean()
    assert jitter_out < 0.5 * jitter_in                     # noise removed
    assert np.abs(out[:, 0] - clean).mean() < np.abs(seq[:, 0] - clean).mean()
    # zero phase: the low-frequency component is not delayed
    lag = np.argmax(np.correlate(out[20:-20, 0, 1], clean[20:-20, 1], "full")) - (len(t) - 41)
    assert abs(lag) <= 1


def test_butterworth_smooth_preserves_nan_gaps_and_short_segments():
    from identity.common.triangulation import butterworth_smooth

    seq = np.zeros((60, 1, 3))
    seq[:25, 0, 0] = np.linspace(0, 1, 25)
    seq[25:32] = np.nan                     # gap must stay NaN
    seq[32:40, 0, 0] = 5.0                  # 8 frames < filtfilt pad -> untouched
    seq[40:] = np.nan
    out = butterworth_smooth(seq, fps=50.0, cutoff_hz=6.0)
    assert np.isnan(out[25:32]).all() and np.isnan(out[40:]).all()
    np.testing.assert_allclose(out[32:40, 0, 0], 5.0)       # short segment untouched
    assert np.isfinite(out[:25]).all()


def test_dense_fill_does_not_bridge_long_real_gaps():
    # C6 (component form): with rows == real frames, a 300-frame real gap must NOT
    # be interpolated by a 25-frame gate even if only 2 observed rows surround it.
    from identity.common.triangulation import fill_occluded_joints

    frames = [0, 1, 2, 300, 301]
    timeline = list(range(frames[0], frames[-1] + 1))
    seq = np.full((len(timeline), 1, 3), np.nan)
    conf = np.zeros((len(timeline), 1))
    for f in frames:
        seq[f, 0] = [float(f), 0.0, 0.0]
        conf[f, 0] = 0.9
    out, _ = fill_occluded_joints(seq, conf, max_gap_frames=25)
    assert np.isnan(out[150, 0]).all()      # mid-gap stays empty
    assert np.isfinite(out[301, 0]).all()
    # legacy row-index behaviour (the bug): 5 rows adjacent -> would interpolate
    legacy = np.asarray([seq[f] for f in frames])
    legacy_conf = np.asarray([conf[f] for f in frames])
    bridged, _ = fill_occluded_joints(legacy, legacy_conf, max_gap_frames=25)
    assert np.isfinite(bridged).all()       # documents why dense_fill exists

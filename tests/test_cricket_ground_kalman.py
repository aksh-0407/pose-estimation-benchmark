"""Tests for the Singer ground-plane Kalman filter (pose_estimation.cricket.ground_kalman)."""

from __future__ import annotations

import numpy as np

from pose_estimation.cricket.ground_kalman import (
    ROLE_PARAMS,
    RoleParams,
    SingerGroundKalman,
)


def test_predict_increases_covariance():
    k = SingerGroundKalman(np.array([0.0, 0.0]), "fielder")
    trace_before = np.trace(k.P)
    k.predict()
    assert np.trace(k.P) > trace_before


def test_update_decreases_uncertainty():
    k = SingerGroundKalman(np.array([0.0, 0.0]), "fielder")
    k.predict()
    trace_before = np.trace(k.P)
    k.update(np.array([0.1, 0.1]))
    assert np.trace(k.P) < trace_before


def test_mahalanobis_zero_at_prediction():
    k = SingerGroundKalman(np.array([1.0, 2.0]), "fielder")
    k.predict()
    assert k.mahalanobis_sq(k.pos_world_xy) < 1e-6


def test_switch_role_inflates_covariance():
    k = SingerGroundKalman(np.array([0.0, 0.0]), "wicketkeeper")
    for _ in range(20):
        k.predict()
        k.update(np.array([0.0, 0.0]))
    trace_before = np.trace(k.P)
    k.switch_role("bowler")
    assert np.trace(k.P) > trace_before


def test_switch_role_recomputes_F():
    k = SingerGroundKalman(np.array([0.0, 0.0]), "wicketkeeper")
    F_before = k.F.copy()
    k.switch_role("bowler")  # higher alpha -> different F
    assert not np.allclose(k.F, F_before)


def test_cap_covariance_limits_growth():
    k = SingerGroundKalman(np.array([0.0, 0.0]), "fielder")
    for _ in range(200):
        k.predict()
    k.cap_covariance(max_pos_var=25.0)
    assert k.P[0, 0] <= 25.0 + 1e-9
    assert k.P[1, 1] <= 25.0 + 1e-9


def test_propagate_state_does_not_mutate():
    k = SingerGroundKalman(np.array([1.0, 2.0]), "fielder")
    x_orig = k.pos_world_xy.copy()
    k.propagate_state(10)
    assert np.allclose(k.pos_world_xy, x_orig)


def test_role_params_override_changes_dynamics():
    # P4Config can inject custom role params; the filter must honour them.
    custom = dict(ROLE_PARAMS)
    custom["fielder"] = RoleParams(alpha=8.0, sigma_a=9.0, measurement_noise=0.4)
    default_k = SingerGroundKalman(np.array([0.0, 0.0]), "fielder")
    custom_k = SingerGroundKalman(np.array([0.0, 0.0]), "fielder", role_params=custom)
    assert not np.allclose(default_k.F, custom_k.F)


def test_striker_role_has_reasonable_Q():
    assert ROLE_PARAMS["striker"].sigma_a > 0.1


def test_all_roles_defined():
    for role in ("bowler", "striker", "non_striker", "wicketkeeper", "umpire", "fielder", "unknown"):
        assert role in ROLE_PARAMS

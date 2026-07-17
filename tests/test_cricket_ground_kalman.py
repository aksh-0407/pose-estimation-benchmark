"""Tests for the Singer ground-plane Kalman filter (identity.p5_global_id.ground_kalman)."""

from __future__ import annotations

import numpy as np

from identity.p5_global_id.ground_kalman import (
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
    # GlobalIdConfig can inject custom role params; the filter must honour them.
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


def test_per_measurement_R_overrides_role_noise():
    import numpy as np
    from identity.p5_global_id.ground_kalman import SingerGroundKalman

    # Two identical filters; one gets a HUGE per-measurement R for an outlier
    # observation - its state must move far less than the trusting filter's.
    a = SingerGroundKalman(np.array([0.0, 0.0]), "unknown", dt=0.02)
    b = SingerGroundKalman(np.array([0.0, 0.0]), "unknown", dt=0.02)
    for f in (a, b):
        f.predict()
    z = np.array([2.0, 0.0])
    a.update(z)                                   # fixed role R (~0.4 m)
    b.update(z, R=np.eye(2) * 25.0)               # 5 m sigma: barely trusted
    assert a.pos_world_xy[0] > 5 * b.pos_world_xy[0]
    # gating sees the same R: the same far point is inside a loose-R gate
    c = SingerGroundKalman(np.array([0.0, 0.0]), "unknown", dt=0.02)
    c.predict()
    tight = c.mahalanobis_sq(z)
    loose = c.mahalanobis_sq(z, R=np.eye(2) * 25.0)
    assert loose < tight
    # None reproduces the legacy path exactly
    d = SingerGroundKalman(np.array([0.0, 0.0]), "unknown", dt=0.02)
    e = SingerGroundKalman(np.array([0.0, 0.0]), "unknown", dt=0.02)
    d.predict(); e.predict()
    d.update(z); e.update(z, R=None)
    np.testing.assert_array_equal(d.x, e.x)
    np.testing.assert_array_equal(d.P, e.P)


def test_switch_role_inflates_stable_blocks_role_dependently():
    import numpy as np
    from identity.p5_global_id.ground_kalman import SingerGroundKalman

    # C3: the Lyapunov branch could never execute; the explicit inflation must be
    # finite, role-dependent, and larger for more agile roles.
    a = SingerGroundKalman(np.zeros(2), "umpire", dt=0.02)
    b = SingerGroundKalman(np.zeros(2), "umpire", dt=0.02)
    va, vb = a.P[2, 2], b.P[2, 2]
    a.switch_role("bowler")
    b.switch_role("wicketkeeper")
    assert np.isfinite(a.P).all() and np.isfinite(b.P).all()
    assert a.P[2, 2] > va and b.P[2, 2] > vb
    assert a.P[4, 4] > b.P[4, 4]           # bowler >> keeper acceleration uncertainty
    assert a.role == "bowler"


def test_unknown_role_falls_back_instead_of_raising():
    import numpy as np
    from identity.p5_global_id.ground_kalman import SingerGroundKalman

    f = SingerGroundKalman(np.zeros(2), "twelfth_man", dt=0.02)  # C4: no KeyError
    f.switch_role("mystery_role")
    assert f.role == "mystery_role"        # label kept; dynamics = unknown params

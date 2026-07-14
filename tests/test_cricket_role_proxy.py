from __future__ import annotations

from collections import deque

import numpy as np

from identity.p5_global_id.role_proxy import OnlineRoleProxy


def _proxy() -> OnlineRoleProxy:
    return OnlineRoleProxy(np.array([0.0, 1.0]), frame_rate_fps=50.0, min_track_frames=50)


def _series(positions):
    return deque((frame, np.asarray(xy, float)) for frame, xy in positions)


def test_bowler_run_fixes_direction_then_ends_classify():
    proxy = _proxy()
    # Static near-stump ids are ambiguous until the bowling direction is known.
    keeper_series = _series([(f, (0.2, 11.0 + 0.001 * f)) for f in range(60)])
    assert proxy._classify("P002", keeper_series) is None

    # Bowler: sustained 5 m/s run toward +y over the last window.
    bowler_series = _series([(f, (0.0, -20.0 + 0.1 * f)) for f in range(60)])
    assert proxy._classify("P001", bowler_series) == "bowler"
    assert proxy.direction_sign == 1.0
    assert proxy._classify("P001", bowler_series) == "bowler"  # sticky

    # +y is now "toward the striker": behind +stumps = keeper, behind -stumps = umpire.
    assert proxy._classify("P002", keeper_series) == "wicketkeeper"
    umpire_series = _series([(f, (-0.3, -11.2)) for f in range(60)])
    assert proxy._classify("P003", umpire_series) == "umpire"

    # A static midfielder or anyone off the pitch line stays unclassified.
    assert proxy._classify("P004", _series([(f, (0.5, 3.0)) for f in range(60)])) is None
    assert proxy._classify("P005", _series([(f, (8.0, 11.0)) for f in range(60)])) is None

    # A moving player near the stumps is never umpire/keeper.
    running = _series([(f, (0.0, 11.0 + 0.05 * f)) for f in range(60)])
    assert proxy._classify("P006", running) is None


def test_proxy_inert_without_pitch_axis():
    proxy = OnlineRoleProxy(None, frame_rate_fps=50.0)
    assert proxy.axis is None
    proxy.observe(object(), 0)  # must be a no-op, never touching the manager

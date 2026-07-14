"""Tests for the P5 v1.1 epoch-scored role solver."""
from __future__ import annotations

import numpy as np

# ------------------------------------------------------- v1.1 epoch solver
def _series(along, across, frames=200, axis=np.array([0.0, 1.0])):
    lateral = np.array([-axis[1], axis[0]])
    pt = along * axis + across * lateral
    return [(f, pt + 0.01 * np.array([np.sin(f / 9.0), np.cos(f / 11.0)])) for f in range(frames)]


def _running_series(start_along, end_along, frames=200, axis=np.array([0.0, 1.0])):
    return [
        (f, (start_along + (end_along - start_along) * f / frames) * axis)
        for f in range(frames)
    ]


def test_epoched_assigns_full_roster_uniquely():
    from scripts.roles.assigner import assign_roles_epoched

    axis = np.array([0.0, 1.0])
    series = {
        "BWL": _running_series(-30.0, -9.0),          # sustained run onto the crease
        "STR": _series(8.8, 0.2),                     # striker at +crease
        "NST": _series(-8.8, -0.3),                   # non-striker at -crease
        "WK": _series(16.0, 0.1),                     # keeper standing BACK (pace)
        "UMP1": _series(-11.0, 0.4),                  # bowler's-end umpire
        "UMP2": _series(8.8, 19.0),                   # square-leg umpire
        "FLD": _series(0.0, 30.0),                    # deep fielder
    }
    roles = assign_roles_epoched(series, axis, min_track_frames=60)
    by_role = {}
    for pid, r in roles.items():
        by_role.setdefault(r.role, []).append(pid)
    assert by_role["bowler"] == ["BWL"]
    assert by_role["striker"] == ["STR"]
    assert by_role["non_striker"] == ["NST"]
    assert by_role["wicketkeeper"] == ["WK"]           # standing back must still win
    assert sorted(by_role["umpire"]) == ["UMP1", "UMP2"]  # exactly two umpires
    assert by_role.get("fielder") == ["FLD"]
    # uniqueness: no single role (other than umpire's two slots) appears twice
    assert all(len(v) == 1 for k, v in by_role.items() if k not in ("umpire", "fielder"))


def test_epoched_never_duplicates_single_slots():
    from scripts.roles.assigner import assign_roles_epoched

    axis = np.array([0.0, 1.0])
    # two equally keeper-ish tracks: only one may hold the slot
    series = {
        "A": _series(12.0, 0.1),
        "B": _series(13.0, -0.1),
        "STR": _series(8.8, 0.0),
    }
    roles = assign_roles_epoched(series, axis, min_track_frames=60)
    keepers = [pid for pid, r in roles.items() if r.role == "wicketkeeper"]
    assert len(keepers) <= 1


# ------------------------------------------------------- Wave-6 suppression
def test_suppression_decide_protects_core_and_drops_bad_peripherals():
    from scripts.roles.suppress_peripherals import DEFAULTS, decide

    cfg = dict(DEFAULTS, suppression_enabled=True)
    quality = {
        "BWL": {"mean_kp_conf": 0.20, "mean_det_conf": 0.3, "completeness": 0.1,
                "single_cam_rate": 1.0, "observations": 30},   # terrible but core
        "UMP": {"mean_kp_conf": 0.20, "mean_det_conf": 0.3, "completeness": 0.9,
                "single_cam_rate": 0.2, "observations": 500},  # bad pose peripheral
        "FLD": {"mean_kp_conf": 0.80, "mean_det_conf": 0.9, "completeness": 0.95,
                "single_cam_rate": 0.1, "observations": 550},  # good peripheral
    }
    roles = {"BWL": "bowler", "UMP": "umpire", "FLD": "fielder"}
    out = decide(quality, roles, cfg)
    assert "BWL" not in out          # core roles are never suppressed
    assert "UMP" in out              # low-confidence peripheral dropped
    assert "FLD" not in out          # well-tracked peripheral kept


def test_suppression_disabled_is_empty_and_umpire_protection_works():
    from scripts.roles.suppress_peripherals import DEFAULTS, decide

    quality = {"UMP": {"mean_kp_conf": 0.1, "mean_det_conf": 0.1, "completeness": 0.05,
                       "single_cam_rate": 1.0, "observations": 10}}
    roles = {"UMP": "umpire"}
    assert decide(quality, roles, dict(DEFAULTS)) == {}
    cfg = dict(DEFAULTS, suppression_enabled=True, suppress_protect_umpires=True)
    assert decide(quality, roles, cfg) == {}


# ------------------------------------------------------- v1.2 auto-flip
def test_epoched_cost_prefers_correct_axis_sign_without_runner():
    """The mirrored roster must resolve the right end from geometry alone."""
    from scripts.roles.assigner import assign_roles_epoched

    axis = np.array([0.0, 1.0])
    series = {
        "STR": _series(8.8, 0.2),
        "NST": _series(-8.8, -0.3),
        "WK": _series(12.0, 0.1),      # keeper behind +stumps
        "UMP1": _series(-11.0, 0.4),   # bowler's-end umpire behind -stumps
        "UMP2": _series(8.8, 19.0),    # square leg at the striker's end
    }
    _, cost_correct = assign_roles_epoched(series, axis, min_track_frames=60, return_cost=True)
    _, cost_flipped = assign_roles_epoched(series, -axis, min_track_frames=60, return_cost=True)
    assert cost_correct < cost_flipped  # geometry disambiguates the end

    roles_correct, _ = assign_roles_epoched(series, axis, min_track_frames=60, return_cost=True)
    assert roles_correct["WK"].role == "wicketkeeper"
    assert roles_correct["STR"].role == "striker"


def test_epoched_cost_penalises_unfilled_slots():
    """Fewer accepted assignments must not fake a better score."""
    from scripts.roles.assigner import assign_roles_epoched

    axis = np.array([0.0, 1.0])
    full = {
        "STR": _series(8.8, 0.2), "NST": _series(-8.8, -0.3),
        "WK": _series(12.0, 0.1), "UMP1": _series(-11.0, 0.4),
    }
    sparse = {"LONE": _series(30.0, 30.0)}  # one far fielder, nothing assignable
    _, cost_full = assign_roles_epoched(full, axis, min_track_frames=60, return_cost=True)
    _, cost_sparse = assign_roles_epoched(sparse, axis, min_track_frames=60, return_cost=True)
    assert cost_full < cost_sparse

from __future__ import annotations

import json

import pytest

from main import (
    DeliveryPlan,
    _stage_window,
    build_arg_parser,
    read_panel_row,
)


def _args(tmp_path, extra=()):
    return build_arg_parser().parse_args(
        ["--output-tree", str(tmp_path / "out"), *extra]
    )


def test_stage_window_bounds_and_order():
    assert _stage_window("01_stabilization", "08_render") == [
        "01_stabilization", "02_tracking", "03_association", "04_lift",
        "05_global_id", "06_roles", "08_render",
    ]
    # 04_lift (binding-keyed) sits inside the window BEFORE global_id; self-skips
    # unless --enable-lift is set.
    assert _stage_window("03_association", "05_global_id") == [
        "03_association", "04_lift", "05_global_id",
    ]
    with pytest.raises(SystemExit):
        _stage_window("05_global_id", "02_tracking")


def test_plan_resolves_reused_stage_from_base_tree(tmp_path):
    base = tmp_path / "base"
    (base / "D1" / "02_tracking").mkdir(parents=True)
    args = _args(tmp_path, ["--base-tree", str(base), "--from-stage", "03_association"])
    plan = DeliveryPlan("D1", args, _stage_window("03_association", "06_roles"))
    # tracking is outside the window -> read in place from the base tree
    assert plan.stage_dir("02_tracking") == base / "D1" / "02_tracking"
    # association is inside the window -> written to the output tree
    assert plan.stage_dir("03_association") == plan.output_root / "03_association"


def test_plan_errors_when_reused_stage_is_missing(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    args = _args(tmp_path, ["--base-tree", str(base), "--from-stage", "05_global_id"])
    plan = DeliveryPlan("D1", args, _stage_window("05_global_id", "05_global_id"))
    with pytest.raises(SystemExit):
        plan.stage_dir("03_association")


def test_plan_errors_without_base_tree(tmp_path):
    args = _args(tmp_path, ["--from-stage", "03_association"])
    plan = DeliveryPlan("D1", args, _stage_window("03_association", "05_global_id"))
    with pytest.raises(SystemExit):
        plan.stage_dir("02_tracking")


def test_p2_input_switches_on_stabilization_flag(tmp_path):
    # default: stabilization ON -> tracking reads the 01_stabilization stage dir
    args = _args(tmp_path)
    plan = DeliveryPlan("D1", args, _stage_window("01_stabilization", "05_global_id"))
    assert plan.p2_input() == plan.output_root / "01_stabilization"
    # opt-out: raw P1 feed = the per-delivery 00_inference stage (must exist to resolve)
    args = _args(tmp_path, ["--no-enable-stabilization"])
    plan = DeliveryPlan("D1", args, _stage_window("01_stabilization", "05_global_id"))
    (plan.output_root / "00_inference").mkdir(parents=True)
    assert plan.p2_input() == plan.output_root / "00_inference"


def test_read_panel_row_includes_computed_p2_tracks(tmp_path):
    d = tmp_path / "tree" / "D1"
    (d / "02_tracking").mkdir(parents=True)
    (d / "05_global_id").mkdir(parents=True)
    (d / "02_tracking" / "tracking_metrics.json").write_text(json.dumps({
        "per_camera": {
            "cam_01": {"summary": {"confirmed_tracks": 5}},
            "cam_04": {"summary": {"confirmed_tracks": 7}},
        }
    }))
    (d / "05_global_id" / "global_id_metrics.json").write_text(json.dumps({
        "distinct_global_id_count": 10,
    }))
    row = read_panel_row(tmp_path / "tree", "D1")
    assert row["p2_tracks"] == 12
    assert row["ids"] == 10
    assert row["jitter_px"] is None          # stabilization never ran -> column shows "-"

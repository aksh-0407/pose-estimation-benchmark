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
    assert _stage_window("p1b", "render") == [
        "p1b", "p2", "p3", "p3_5", "p4", "p5", "p6_3d", "render"
    ]
    # p3_5 sits inside the window but self-skips unless --enable-lift is set
    assert _stage_window("p3", "p4") == ["p3", "p3_5", "p4"]
    with pytest.raises(SystemExit):
        _stage_window("p4", "p2")


def test_plan_resolves_reused_stage_from_base_tree(tmp_path):
    base = tmp_path / "base"
    (base / "deliveries" / "D1" / "p2").mkdir(parents=True)
    args = _args(tmp_path, ["--base-tree", str(base), "--from-stage", "p3"])
    plan = DeliveryPlan("D1", args, _stage_window("p3", "p6_3d"))
    # p2 is outside the window -> read in place from the base tree
    assert plan.stage_dir("p2") == base / "deliveries" / "D1" / "p2"
    # p3 is inside the window -> written to the output tree
    assert plan.stage_dir("p3") == plan.output_root / "p3"


def test_plan_errors_when_reused_stage_is_missing(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    args = _args(tmp_path, ["--base-tree", str(base), "--from-stage", "p4"])
    plan = DeliveryPlan("D1", args, _stage_window("p4", "p4"))
    with pytest.raises(SystemExit):
        plan.stage_dir("p3")


def test_plan_errors_without_base_tree(tmp_path):
    args = _args(tmp_path, ["--from-stage", "p3"])
    plan = DeliveryPlan("D1", args, _stage_window("p3", "p4"))
    with pytest.raises(SystemExit):
        plan.stage_dir("p2")


def test_p2_input_switches_on_stabilization_flag(tmp_path):
    input_tree = tmp_path / "p1run"
    input_tree.mkdir()
    # v7 default: stabilization ON -> P2 reads the p1b stage dir
    args = _args(tmp_path, ["--input-tree", str(input_tree)])
    plan = DeliveryPlan("D1", args, _stage_window("p1b", "p4"))
    assert plan.p2_input() == plan.output_root / "p1b"
    # v6-style opt-out: raw P1 feed
    args = _args(tmp_path, ["--input-tree", str(input_tree), "--no-enable-stabilization"])
    plan = DeliveryPlan("D1", args, _stage_window("p1b", "p4"))
    assert plan.p2_input() == input_tree.resolve()


def test_read_panel_row_includes_computed_p2_tracks(tmp_path):
    d = tmp_path / "tree" / "deliveries" / "D1"
    (d / "p2").mkdir(parents=True)
    (d / "p4").mkdir(parents=True)
    (d / "p2" / "tracking_metrics.json").write_text(json.dumps({
        "per_camera": {
            "cam_01": {"summary": {"confirmed_tracks": 5}},
            "cam_04": {"summary": {"confirmed_tracks": 7}},
        }
    }))
    (d / "p4" / "global_id_metrics.json").write_text(json.dumps({
        "distinct_global_id_count": 10,
    }))
    row = read_panel_row(tmp_path / "tree", "D1")
    assert row["p2_tracks"] == 12
    assert row["ids"] == 10
    assert row["jitter_px"] is None          # p1b never ran -> column shows "-"

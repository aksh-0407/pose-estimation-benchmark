"""Drive the main compute loop with a stubbed _run_stage (no subprocesses).

Covers the stage dispatch/order, the 08_render skip (the AssertionError regression), and the
04 -> 05 -> 06 wiring + per-delivery `00_inference` input under the wrapper-free layout.
"""
from __future__ import annotations

import main
from main import DeliveryPlan, _stage_window, build_arg_parser, run_compute_chain


def _input_run_dir(arg_list):
    return arg_list[arg_list.index("--input-run-dir") + 1]


def test_compute_chain_order_skip_and_wiring(tmp_path, monkeypatch):
    run = tmp_path / "run"
    (run / "D1" / "00_inference").mkdir(parents=True)  # P1 stage must exist for stage_dir
    args = build_arg_parser().parse_args(
        ["--output-tree", str(run), "--drive-root", str(tmp_path / "raw"), "--deliveries", "D1"]
    )

    calls: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(main, "_run_stage", lambda module, a, python, log: calls.append((module, a)) or 0)

    plan = DeliveryPlan("D1", args, _stage_window("01_stabilization", "08_render"))
    result = run_compute_chain(plan)

    # 08_render is NOT dispatched by the compute chain (run_render handles it); no crash on it.
    assert [m for m, _ in calls] == [
        "identity.p1_stabilization.run_stabilization",
        "identity.p2_tracking.run_per_camera_tracking",
        "identity.p3_association.run_cross_camera_association",
        "identity.p4_lift.run_triangulation",
        "identity.p5_global_id.run_global_id",
        "identity.p6_roles.run_role_assignment",
        "identity.p6_roles.suppress_peripherals",
    ]
    assert "failed_stage" not in result

    by_module = {m: a for m, a in calls}
    # per-delivery layout (no deliveries/ wrapper) + the single-triangulation wiring
    assert _input_run_dir(by_module["identity.p1_stabilization.run_stabilization"]).endswith("/D1/00_inference")
    assert _input_run_dir(by_module["identity.p5_global_id.run_global_id"]).endswith("/D1/04_lift")
    assert _input_run_dir(by_module["identity.p6_roles.run_role_assignment"]).endswith("/D1/05_global_id")


def test_compute_chain_stops_on_stage_failure(tmp_path, monkeypatch):
    run = tmp_path / "run"
    (run / "D1" / "00_inference").mkdir(parents=True)
    args = build_arg_parser().parse_args(
        ["--output-tree", str(run), "--drive-root", str(tmp_path / "raw"), "--deliveries", "D1"]
    )

    calls: list[str] = []

    def fake(module, a, python, log):
        calls.append(module)
        return 2 if module.endswith("run_per_camera_tracking") else 0  # 02 hard-fails

    monkeypatch.setattr(main, "_run_stage", fake)
    plan = DeliveryPlan("D1", args, _stage_window("01_stabilization", "06_roles"))
    result = run_compute_chain(plan)

    assert result["failed_stage"] == "02_tracking"
    assert calls == [  # chain halts after the failing stage
        "identity.p1_stabilization.run_stabilization",
        "identity.p2_tracking.run_per_camera_tracking",
    ]

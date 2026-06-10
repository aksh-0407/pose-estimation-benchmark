import importlib.util
from argparse import Namespace
from pathlib import Path


def load_benchmark_module():
    spec = importlib.util.spec_from_file_location("benchmark_cli", Path("scripts/benchmark.py"))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_master_cli_run_aggregate_report_with_temp_paths(tmp_path, monkeypatch):
    module = load_benchmark_module()
    monkeypatch.setattr(module, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(module, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(module, "AGGREGATE_CSV", tmp_path / "aggregate.csv")

    run_args = Namespace(
        run_id="unit_run",
        models=["yolo26x_pose"],
        datasets=["coco17_val2017"],
        limit=1,
        start_index=0,
        batch_size=8,
        device="cpu",
        imgsz=640,
        conf=0.001,
        iou=0.7,
        precision="fp32",
        backend="unit",
        resume=True,
        dry_run=True,
    )

    assert module.command_run(run_args) == 0
    metrics_path = tmp_path / "runs" / "unit_run" / "metrics" / "yolo26x_pose__coco17_val2017.json"
    assert metrics_path.exists()

    aggregate_args = Namespace(runs_dir=str(tmp_path / "runs"), output=str(tmp_path / "aggregate.csv"))
    assert module.command_aggregate(aggregate_args) == 0
    assert (tmp_path / "aggregate.csv").exists()

    report_args = Namespace(aggregate=str(tmp_path / "aggregate.csv"), output=str(tmp_path / "report.html"))
    assert module.command_report(report_args) == 0
    assert (tmp_path / "report.html").exists()


def test_generated_run_id_mentions_models_datasets_and_scope():
    module = load_benchmark_module()

    run_id = module._run_id(
        models=["yolo26x_pose"],
        datasets=["coco17_val2017"],
        limit=100,
        start_index=0,
    )

    assert run_id.startswith("models-yolo26x_pose__bench-coco17_val2017__scope-n100__")
    assert " " not in run_id
    assert ":" not in run_id


def test_generated_run_id_mentions_start_index_when_present():
    module = load_benchmark_module()

    run_id = module._run_id(
        models=["rtmw_x", "yolo26x_pose"],
        datasets=["coco17_val2017"],
        limit=None,
        start_index=200,
    )

    assert run_id.startswith("models-rtmw_x+yolo26x_pose__bench-coco17_val2017__scope-start200-full__")

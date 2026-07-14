from tools.audit_repo import violations


def test_repo_audit_allows_placeholders_and_flags_artifacts():
    paths = [
        "models/rtmw_l/weights/.gitkeep",
        "models/rtmw_l/weights/model.pth",
        "models/rtmw_l/checksums/sha256.json",
        "drive/dataset/bt_01/clip/camera01/frame_camera01_000001.jpg",
        "benchmarks/runs/run_001/run_manifest.json",
        "benchmarks/reports/.gitkeep",
        "benchmarks/reports/report_001/index.html",
        "benchmarks/artifacts/run_001/predictions/model__dataset.jsonl",
        "benchmarks/runs/run_001/logs/model__dataset.latency.jsonl",
        "results/README.md",
        "results/aggregate_metrics.csv",
        "results/smoke_results.csv",
    ]
    # Run manifests and placeholders/READMEs are tracked; weights, checksums, raw
    # artifacts, and DERIVED results/reports are not (CI regenerates the latter).
    assert violations(paths) == [
        "benchmarks/artifacts/run_001/predictions/model__dataset.jsonl",
        "benchmarks/reports/report_001/index.html",
        "benchmarks/runs/run_001/logs/model__dataset.latency.jsonl",
        "drive/dataset/bt_01/clip/camera01/frame_camera01_000001.jpg",
        "models/rtmw_l/checksums/sha256.json",
        "models/rtmw_l/weights/model.pth",
        "results/aggregate_metrics.csv",
        "results/smoke_results.csv",
    ]

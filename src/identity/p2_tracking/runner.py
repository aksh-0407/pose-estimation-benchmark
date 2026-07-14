"""Parallel per-camera tracking orchestration."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from core.calibration import (
    build_ground_calibrators,
    current_calibration_dir,
)
from identity.p2_tracking.config import TrackingConfig
from identity.p2_tracking.jsonl_io import track_camera_file

CANONICAL_PREDICTION_RE = re.compile(
    r"^(?P<capture_group>bt_\d{2})__(?P<delivery_id>.+)__(?P<camera_id>cam_\d{2})\.jsonl$"
)


@dataclass(frozen=True)
class PredictionFile:
    path: Path
    capture_group: str
    delivery_id: str
    camera_id: str


def track_one_camera(args: tuple) -> tuple[str, str, dict, str | None]:
    (camera_id, capture_group, input_path, output_path, diagnostics_path,
     delivery_id, config, expected_frames, calibrator) = args
    try:
        diagnostics = track_camera_file(
            input_path, output_path, diagnostics_path,
            camera_id, capture_group, delivery_id, config, expected_frames, calibrator,
        )
        summary = {
            key: value
            for key, value in diagnostics.items()
            if isinstance(value, (int, float, bool))
        }
        summary["intra_camera_id_switch_proxy"] = (
            summary.get("dormant_reid_ambiguous", 0)
            + summary.get("dormant_deleted", 0)
        )
        return camera_id, "ok", summary, None
    except Exception as exc:  # noqa: BLE001 — surface per-camera failure, do not kill the pool
        return camera_id, "failed", {}, f"{type(exc).__name__}: {exc}"


def parse_prediction_filename(path: str | Path) -> tuple[str, str, str] | None:
    match = CANONICAL_PREDICTION_RE.match(Path(path).name)
    if not match:
        return None
    return match.group("capture_group"), match.group("delivery_id"), match.group("camera_id")


def infer_match_id(delivery_id: str) -> str:
    match = re.match(r"^(?P<match_id>.+?)M\d", delivery_id)
    if not match:
        raise ValueError(f"cannot infer match_id from delivery_id: {delivery_id}")
    return match.group("match_id")


def discover_prediction_files(
    input_run_dir: str | Path,
    delivery_id: str,
    cameras: list[str] | None = None,
) -> list[PredictionFile]:
    prediction_dir = Path(input_run_dir) / "predictions"
    if not prediction_dir.exists():
        raise RuntimeError(f"missing predictions directory: {prediction_dir}")

    legacy = sorted(path.name for path in prediction_dir.glob("cam_*.jsonl"))
    if legacy:
        raise RuntimeError(
            "legacy cam_XX.jsonl inputs are not supported; expected "
            "<capture_group>__<delivery_id>__<cam_XX>.jsonl"
        )

    wanted_cameras = set(cameras) if cameras else None
    files: list[PredictionFile] = []
    for path in sorted(prediction_dir.glob("*.jsonl")):
        parsed = parse_prediction_filename(path)
        if parsed is None:
            continue
        capture_group, file_delivery_id, camera_id = parsed
        if file_delivery_id != delivery_id:
            continue
        if wanted_cameras is not None and camera_id not in wanted_cameras:
            continue
        files.append(PredictionFile(path, capture_group, delivery_id, camera_id))

    if not files:
        raise RuntimeError(f"no canonical prediction files found for delivery {delivery_id} in {prediction_dir}")
    return files


def run_tracking(
    input_run_dir: str | Path,
    output_run_dir: str | Path,
    drive_root: str | Path,
    delivery_id: str,
    config: TrackingConfig,
    cameras: list[str] | None = None,
    expected_frames: int = 600,
    max_workers: int | None = None,
) -> dict[str, tuple[str, dict, str | None]]:
    input_run_dir = Path(input_run_dir)
    output_run_dir = Path(output_run_dir)
    prediction_files = discover_prediction_files(input_run_dir, delivery_id, cameras)
    camera_ids = [item.camera_id for item in prediction_files]
    match_id = infer_match_id(delivery_id)
    calibrators = build_ground_calibrators(drive_root, match_id, camera_ids)

    prediction_output_dir = output_run_dir / "predictions"
    diagnostics_dir = output_run_dir / "diagnostics"
    prediction_output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        (
            item.camera_id,
            item.capture_group,
            item.path,
            prediction_output_dir / item.path.name,
            diagnostics_dir / f"{item.path.stem}.json",
            delivery_id,
            config,
            expected_frames,
            calibrators[item.camera_id],
        )
        for item in prediction_files
    ]
    workers = max_workers or min(7, os.cpu_count() or 1)

    results: dict[str, tuple[str, dict, str | None]] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for cam, status, summary, error in pool.map(track_one_camera, jobs):
            results[cam] = (status, summary, error)

    _write_run_artifacts(
        output_run_dir=output_run_dir,
        input_run_dir=input_run_dir,
        drive_root=Path(drive_root),
        delivery_id=delivery_id,
        match_id=match_id,
        config=config,
        prediction_files=prediction_files,
        results=results,
        expected_frames=expected_frames,
        max_workers=workers,
    )
    return results


def _write_run_artifacts(
    *,
    output_run_dir: Path,
    input_run_dir: Path,
    drive_root: Path,
    delivery_id: str,
    match_id: str,
    config: TrackingConfig,
    prediction_files: list[PredictionFile],
    results: dict[str, tuple[str, dict, str | None]],
    expected_frames: int,
    max_workers: int,
) -> None:
    output_run_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": "tracking_run/v1",
        "created_at": created_at,
        "task": "per_camera_tracking",
        "input_run_dir": str(input_run_dir),
        "output_run_dir": str(output_run_dir),
        "drive_root": str(drive_root),
        "calibration_dir": str(current_calibration_dir(drive_root, match_id)),
        "match_id": match_id,
        "delivery_id": delivery_id,
        "expected_frames": expected_frames,
        "max_workers": max_workers,
        "config": config.to_dict(),
        "inputs": [
            {
                "prediction_jsonl": str(item.path),
                "capture_group": item.capture_group,
                "camera_id": item.camera_id,
            }
            for item in prediction_files
        ],
    }
    metrics = {
        "schema_version": "tracking_metrics/v1",
        "created_at": created_at,
        "match_id": match_id,
        "delivery_id": delivery_id,
        "status": "pass" if all(status == "ok" for status, _, _ in results.values()) else "fail",
        "per_camera": {
            camera_id: {
                "status": status,
                "summary": summary,
                "error": error,
            }
            for camera_id, (status, summary, error) in sorted(results.items())
        },
        "intra_camera_id_switch_proxy_total": sum(
            int(summary.get("intra_camera_id_switch_proxy", 0))
            for status, summary, _ in results.values()
            if status == "ok"
        ),
    }
    with (output_run_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (output_run_dir / "tracking_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from pose_estimation.cricket.contract import example_group1_frame, validate_group1_frame
from scripts.association.config import P3AssociationConfig
from scripts.association.runner import run_association
from scripts.export.triangulate_predictions import triangulate_canonical_run
from scripts.global_id.config import P4AConfig, P4Config
from scripts.global_id.runner import run_global_id


DELIVERY = "MATCHM1_1_1_1"


def _projection(center: np.ndarray) -> np.ndarray:
    forward = -center / np.linalg.norm(center)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    intrinsic = np.array([[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]])
    return intrinsic @ np.hstack([rotation, (-rotation @ center).reshape(3, 1)])


def _project(point: np.ndarray, projection: np.ndarray) -> np.ndarray:
    homogeneous = projection @ np.append(point, 1.0)
    return homogeneous[:2] / homogeneous[2]


def _record(camera: str, capture_group: str, frame: int, foot: np.ndarray) -> dict:
    record = example_group1_frame(final_handoff=False)
    record.update({
        "match_id": "MATCH",
        "delivery_id": DELIVERY,
        "camera_id": camera,
        "capture_group": capture_group,
        "frame_index": frame,
        "frame_name": f"frame_{camera}_{frame:06d}.jpg",
    })
    player = record["players"][0]
    player.update({
        "global_player_id": None,
        "local_track_id": f"{camera}_trk_0001",
        "track_state": "confirmed",
        "single_camera": None,
        "track_confidence": 0.9,
        "detection_confidence": 0.95,
        "bbox_xywh_px": [float(foot[0] - 10), float(foot[1] - 40), 20.0, 40.0],
        "bbox_xywh_norm": [float((foot[0] - 10) / 1280), float((foot[1] - 40) / 720), 20 / 1280, 40 / 720],
    })
    player["pose_2d"]["keypoints_px"] = np.tile(foot, (17, 1)).tolist()
    player["pose_2d"]["keypoints_norm"] = (
        np.tile(foot, (17, 1)) / np.array([1280.0, 720.0])
    ).tolist()
    player["pose_2d"]["confidence"] = [0.95] * 17
    validate_group1_frame(record)
    return record


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    input_run = tmp_path / "p2"
    drive_root = tmp_path / "drive"
    prediction_dir = input_run / "predictions"
    prediction_dir.mkdir(parents=True)
    centers = {"cam_01": np.array([8.0, 0.0, 4.0]), "cam_02": np.array([0.0, 8.0, 4.0])}
    projections = {camera: _projection(center) for camera, center in centers.items()}
    calibration_dir = drive_root / "dataset" / "calibration-data" / "MATCH" / "calibration_data"
    calibration_dir.mkdir(parents=True)
    (calibration_dir / "Bundle_Adjusted_extrinsics.json").write_text(json.dumps({
        "projection_matrices": {"C01": projections["cam_01"].tolist(), "C02": projections["cam_02"].tolist()}
    }))
    world_points = [np.array([0.1, 0.1, 0.0]), np.array([0.12, 0.1, 0.0]), np.array([0.14, 0.1, 0.0])]
    for camera, group in (("cam_01", "bt_01"), ("cam_02", "bt_02")):
        path = prediction_dir / f"{group}__{DELIVERY}__{camera}.jsonl"
        rows = [_record(camera, group, index + 1, _project(point, projections[camera])) for index, point in enumerate(world_points)]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return input_run, drive_root


def test_p3_p4_p6_canonical_run_smoke(tmp_path: Path):
    p2_run, drive_root = _write_fixture(tmp_path)
    p3_run, p4_run, p6_run = tmp_path / "p3", tmp_path / "p4", tmp_path / "p6"
    p3_metrics = run_association(
        p2_run,
        p3_run,
        drive_root,
        DELIVERY,
        P3AssociationConfig(
            image_w=1280,
            image_h=720,
            baseline_angle_degen_deg=1.0,
            cycle_reproj_tol_px=3.0,
            triangulation_reproj_threshold_px=2.0,
        ),
        expected_frames=3,
    )
    assert p3_metrics["frames_processed"] == 3
    assert p3_metrics["single_camera_rate"] == 0.0
    first_corr = json.loads((p3_run / "diagnostics" / "correspondences.jsonl").read_text().splitlines()[0])
    assert len(first_corr["clusters"][0]["members"]) == 2

    p4_metrics = run_global_id(
        p3_run,
        p4_run,
        drive_root,
        DELIVERY,
        P4Config(p4a=replace(P4AConfig(), confirm_hits=2, confidence_high=0.4)),
        expected_frames=3,
    )
    assert p4_metrics["distinct_global_id_count"] == 1
    assert json.loads((p4_run / "run_manifest.json").read_text())["schema_version"] == "global_id_run/v1"
    output_record = json.loads(next((p4_run / "predictions").glob("*.jsonl")).read_text().splitlines()[-1])
    assert output_record["players"][0]["global_player_id"] == "P001"

    p6_metrics = triangulate_canonical_run(
        input_run_dir=p4_run,
        output_run_dir=p6_run,
        drive_root=drive_root,
        delivery_id=DELIVERY,
        cameras=None,
        reprojection_threshold_px=2.0,
        min_views=2,
        ema_alpha=0.65,
    )
    assert p6_metrics["fully_valid_identity_frames"] == 3
    lifted = json.loads(next((p6_run / "predictions").glob("*.jsonl")).read_text().splitlines()[-1])
    assert len(lifted["players"][0]["pose_3d"]["keypoints_world_m"]) == 17
    validate_group1_frame(lifted)

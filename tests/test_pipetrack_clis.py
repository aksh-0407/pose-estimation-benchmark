from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from core.contract import example_group1_frame, validate_group1_frame
from identity.p3_association.config import P3AssociationConfig
from identity.p3_association.runner import run_association
from identity.p4_lift.run_triangulation import triangulate_canonical_run
from identity.p5_global_id.config import GlobalTrackingConfig, GlobalIdConfig
from identity.p5_global_id.runner import run_global_id


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
    player["pose_2d"]["keypoints_px"] = np.tile(foot, (26, 1)).tolist()
    player["pose_2d"]["keypoints_norm"] = (
        np.tile(foot, (26, 1)) / np.array([1280.0, 720.0])
    ).tolist()
    player["pose_2d"]["confidence"] = [0.95] * 26
    validate_group1_frame(record)
    return record


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    input_run = tmp_path / "p2"
    drive_root = tmp_path / "drive"
    prediction_dir = input_run / "predictions"
    prediction_dir.mkdir(parents=True)
    centers = {"cam_01": np.array([8.0, 0.0, 4.0]), "cam_02": np.array([0.0, 8.0, 4.0])}
    projections = {camera: _projection(center) for camera, center in centers.items()}
    calibration_dir = drive_root / "calibration-data" / "MATCH" / "calibration_data"
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
        GlobalIdConfig(tracking=replace(GlobalTrackingConfig(), confirm_hits=2, confidence_high=0.4)),
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
    lifted_player = lifted["players"][0]
    assert len(lifted_player["pose_3d"]["keypoints_world_m"]) == 26
    named = lifted_player["pose_3d_named"]
    assert named["root_joint"] == "hip"
    assert len(named["root_world_m"]) == 3
    assert len(named["joints_root_relative_m"]) == 26
    assert named["joints_root_relative_m"]["hip"] == [0.0, 0.0, 0.0]  # root is the origin
    validate_group1_frame(lifted)


def test_pose_3d_allows_null_joints():
    """A player-frame with some un-triangulated joints (e.g. occluded feet) validates:
    those joints are null in keypoints_world_m while confidence/reproj stay full-length."""
    rec = example_group1_frame(final_handoff=True)
    kw = rec["players"][0]["pose_3d"]["keypoints_world_m"]
    kw[20] = None  # left_big_toe not triangulated
    kw[24] = None  # left_heel not triangulated
    validate_group1_frame(rec)  # must not raise
    assert len(rec["players"][0]["pose_3d"]["confidence"]) == 26


def test_lift_before_global_id_chain(tmp_path: Path):
    """04 lift (binding) runs before global-id, copies correspondences forward, and
    global-id carries the 26-joint pose_3d through; 06 finalize stamps role."""
    from identity.p6_roles.suppress_peripherals import write_terminal_predictions

    p2_run, drive_root = _write_fixture(tmp_path)
    p3_run = tmp_path / "03"
    run_association(
        p2_run, p3_run, drive_root, DELIVERY,
        P3AssociationConfig(image_w=1280, image_h=720, baseline_angle_degen_deg=1.0,
                            cycle_reproj_tol_px=3.0, triangulation_reproj_threshold_px=2.0),
        expected_frames=3,
    )
    # 04 binding-keyed lift runs BEFORE global-id and copies correspondences forward.
    p4_lift = tmp_path / "04"
    triangulate_canonical_run(
        input_run_dir=p3_run, output_run_dir=p4_lift, drive_root=drive_root,
        delivery_id=DELIVERY, cameras=None, reprojection_threshold_px=2.0,
        min_views=2, ema_alpha=0.65, id_source="binding",
    )
    assert (p4_lift / "diagnostics" / "correspondences.jsonl").exists()
    lift_rec = json.loads(next((p4_lift / "predictions").glob("*.jsonl")).read_text().splitlines()[-1])
    assert len(lift_rec["players"][0]["pose_3d"]["keypoints_world_m"]) == 26

    # 05 reads the 04 lift run; pose_3d rides forward while global ids are assigned.
    p5_run = tmp_path / "05"
    run_global_id(
        p4_lift, p5_run, drive_root, DELIVERY,
        GlobalIdConfig(tracking=replace(GlobalTrackingConfig(), confirm_hits=2, confidence_high=0.4)),
        expected_frames=3,
    )
    p5_player = json.loads(
        next((p5_run / "predictions").glob("*.jsonl")).read_text().splitlines()[-1]
    )["players"][0]
    assert p5_player["global_player_id"] == "P001"
    assert len(p5_player["pose_3d"]["keypoints_world_m"]) == 26  # carried through global-id

    # 06 terminal finalize: role stamped onto the identified player, suppressed dropped.
    p6_run = tmp_path / "06"
    assert write_terminal_predictions(p5_run, p6_run, {"P001": "bowler"}, {}) >= 1
    final_player = json.loads(
        next((p6_run / "predictions").glob("*.jsonl")).read_text().splitlines()[-1]
    )["players"][0]
    assert final_player["role"] == "bowler"
    assert final_player["global_player_id"] == "P001"

#!/usr/bin/env python3
"""Triangulate multi-view poses, including canonical PipeTrack P4 runs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from core.contract import validate_group1_frame  # noqa: E402
from core.schemas import CameraCalibration  # noqa: E402
from identity.common.triangulation import (  # noqa: E402
    _COCO17_PARENT,
    _HALPE26_PARENT,
    butterworth_smooth,
    confidence_ema_smooth,
    fill_from_skeletal_prior,
    fill_occluded_joints,
    triangulate_skeleton_ransac,
)


def _most_complete_pose(sequence: np.ndarray) -> np.ndarray:
    """The identity's frame with the most triangulated joints (skeletal-prior reference)."""
    if len(sequence) == 0:
        return np.full((17, 3), np.nan)  # callers never hit this with frames present
    counts = [int(np.isfinite(sequence[t]).all(axis=1).sum()) for t in range(len(sequence))]
    return sequence[int(np.argmax(counts))]


def _median_bone_lengths(
    sequence: np.ndarray,
    parents: dict[int, int] | None = None,
) -> dict[tuple[int, int], float]:
    """Median length of each skeleton bone across the identity's triangulated frames."""
    out: dict[tuple[int, int], float] = {}
    joint_count = sequence.shape[1] if len(sequence) else 0
    for child, parent in (parents or _COCO17_PARENT).items():
        if child >= joint_count or parent >= joint_count:
            continue
        lengths = []
        for t in range(len(sequence)):
            c, p = sequence[t, child], sequence[t, parent]
            if np.isfinite(c).all() and np.isfinite(p).all():
                lengths.append(float(np.linalg.norm(c - p)))
        if lengths:
            out[(child, parent)] = float(np.median(lengths))
    return out
from identity.p3_association.cluster_lift import cluster_purity, lift_frame  # noqa: E402
from identity.p3_association.jsonl_io import load_synchronized_records  # noqa: E402
from identity.p5_global_id.jsonl_io import read_correspondence_rows  # noqa: E402
from core.calibration import load_projection_matrices_from_drive  # noqa: E402
from core.keypoints import HALPE26_KEYPOINTS, named_root_relative  # noqa: E402
from identity.p2_tracking.runner import discover_prediction_files, infer_match_id  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    canonical = parser.add_argument_group("canonical PipeTrack run")
    canonical.add_argument("--input-run-dir")
    canonical.add_argument("--output-run-dir")
    canonical.add_argument("--drive-root")
    canonical.add_argument("--delivery-id")
    canonical.add_argument("--camera", action="append", default=None)
    legacy = parser.add_argument_group("legacy flat JSONL")
    legacy.add_argument("--predictions")
    legacy.add_argument("--calibration")
    legacy.add_argument("--output")
    parser.add_argument("--reprojection-threshold-px", type=float, default=10.0)
    parser.add_argument("--min-views", type=int, default=2)
    parser.add_argument("--ema-alpha", type=float, default=0.65)
    parser.add_argument("--cheirality", action="store_true",
                        help="Reject triangulations behind a camera (fix F3; default off = legacy).")
    parser.add_argument("--smoother", choices=["ema", "butterworth"], default="ema",
                        help="Temporal smoother: causal EMA (legacy) or zero-phase Butterworth (fix F7).")
    parser.add_argument("--butter-cutoff-hz", type=float, default=6.0)
    parser.add_argument("--capture-fps", type=float, default=50.0,
                        help="Capture frame rate used by the Butterworth smoother.")
    parser.add_argument("--hartley", action="store_true",
                        help="G1: row-equilibrated DLT conditioning (flag-gated A/B)")
    parser.add_argument("--parallax-order", action="store_true",
                        help="G3: try high-parallax RANSAC seed pairs first (flag-gated A/B)")
    parser.add_argument("--robust-refit", action="store_true",
                        help="Phase-1C: IRLS-Huber M-estimator polish on the per-joint inlier "
                             "re-fit (down-weights marginal-inlier cameras). Off = byte-identical.")
    parser.add_argument("--robust-huber-px", type=float, default=8.0,
                        help="Huber threshold (px reprojection residual) for --robust-refit.")
    parser.add_argument("--suppression-path", default=None,
                        help="P5b suppression.json; suppressed global ids are not lifted "
                             "(default: probes <input>/../06_roles/suppression.json; Wave-6)")
    parser.add_argument("--id-source", choices=["global", "binding"], default="global",
                        help="Group observations by P4 global_player_id (terminal lift, legacy) "
                             "or by P3 binding_id (the 04 (binding lift) re-sequenced lift, fix F9b).")
    parser.add_argument("--chimera-torso-residual-px", type=float, default=20.0)
    parser.add_argument("--chimera-frame-fraction", type=float, default=0.3)
    parser.add_argument("--airborne-ankle-z-m", type=float, default=0.25)
    parser.add_argument("--dense-fill", action="store_true",
                        help="C6: measure occlusion gaps in REAL frame numbers (densify the "
                             "per-identity timeline with NaN rows) instead of array rows, so a "
                             "2-row gap spanning 300 real frames is no longer interpolated as "
                             "adjacent. Default off = legacy row-index behaviour.")
    parser.add_argument("--native-skeleton", action="store_true",
                        help="(Deprecated no-op) pose_2d is now the canonical Halpe-26 skeleton, "
                             "so all 26 joints are always triangulated into pose_3d.")
    return parser.parse_args(argv)


def load_calibration(path: str | Path) -> dict[str, CameraCalibration]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    cameras = payload.get("cameras", payload)
    if isinstance(cameras, dict):
        cameras = list(cameras.values())
    return {camera["camera_id"]: CameraCalibration.from_dict(camera) for camera in cameras}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def triangulate_canonical_run(
    *,
    input_run_dir: str | Path,
    output_run_dir: str | Path,
    drive_root: str | Path,
    delivery_id: str,
    cameras: list[str] | None,
    reprojection_threshold_px: float,
    min_views: int,
    ema_alpha: float,
    cheirality: bool = False,
    smoother: str = "ema",
    butter_cutoff_hz: float = 6.0,
    capture_fps: float = 50.0,
    id_source: str = "global",
    chimera_torso_residual_px: float = 20.0,
    chimera_frame_fraction: float = 0.3,
    airborne_ankle_z_m: float = 0.25,
    native_skeleton: bool = False,
    dense_fill: bool = False,
    suppression_path: str | Path | None = None,
    hartley: bool = False,
    parallax_order: bool = False,
    robust_refit: bool = False,
    robust_huber_px: float = 8.0,
) -> dict[str, Any]:
    input_run_dir, output_run_dir = Path(input_run_dir), Path(output_run_dir)
    # Wave-6 (P5b): suppressed peripheral ids are excluded from the lift entirely.
    suppressed_ids: set[str] = set()
    sup_path = Path(suppression_path) if suppression_path else (
        input_run_dir.parent / "06_roles" / "suppression.json"
    )
    if sup_path.is_file():
        try:
            payload = json.loads(sup_path.read_text())
            if payload.get("enabled"):
                suppressed_ids = {str(pid) for pid in (payload.get("suppressed") or {})}
        except (OSError, json.JSONDecodeError):
            suppressed_ids = set()
    prediction_files = discover_prediction_files(input_run_dir, delivery_id, cameras)
    records_by_frame = load_synchronized_records(prediction_files, delivery_id)
    match_id = infer_match_id(delivery_id)
    all_projections = load_projection_matrices_from_drive(drive_root, match_id)

    observations: dict[tuple[int, str], list[tuple[str, dict]]] = defaultdict(list)
    if id_source == "binding":
        # F9b: identity comes from the P3 correspondence stream (binding_id), so the
        # lift runs BEFORE global ID exists and its purity/descriptor evidence can
        # feed identity instead of trailing it.
        correspondence_path = input_run_dir / "diagnostics" / "correspondences.jsonl"
        if not correspondence_path.exists():
            raise FileNotFoundError(f"--id-source binding needs {correspondence_path}")
        for row in read_correspondence_rows(correspondence_path):
            frame_index = int(row["frame_index"])
            camera_records = records_by_frame.get(frame_index, {})
            for cluster in row.get("clusters", []):
                binding = cluster.get("binding_id")
                if binding is None:
                    continue
                for member in cluster.get("members", []):
                    camera_id = member.get("cam_id")
                    record = camera_records.get(camera_id)
                    if record is None or camera_id not in all_projections:
                        continue
                    players = record.get("players", [])
                    player_index = int(member["player_index"])
                    if 0 <= player_index < len(players):
                        player = players[player_index]
                        if player.get("pose_2d") is not None:
                            observations[(frame_index, binding)].append((camera_id, player))
    else:
        for frame_index, camera_records in records_by_frame.items():
            for camera_id, record in camera_records.items():
                if camera_id not in all_projections:
                    continue
                for player in record.get("players", []):
                    player_id = player.get("global_player_id")
                    if player_id and str(player_id) in suppressed_ids:
                        continue  # Wave-6: drop low-confidence peripherals, never extrapolate
                    if player_id and player.get("pose_2d") is not None:
                        observations[(frame_index, player_id)].append((camera_id, player))

    def _member_keypoints(views: list[tuple[str, dict]]) -> dict[str, np.ndarray]:
        """(J, 3) [x, y, conf] per camera over the canonical Halpe-26 pose_2d.

        pose_2d carries all 26 joints (COCO-17 in [0:17] + head/neck/hip + feet),
        so every view stacks to the same 26 rows.
        """

        blocks = {}
        for camera, player in views:
            pose = player["pose_2d"]
            blocks[camera] = np.column_stack(
                (np.asarray(pose["keypoints_px"], dtype=float),
                 np.asarray(pose["confidence"], dtype=float))
            )
        return blocks

    raw: dict[tuple[int, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    lifts: dict[tuple[int, str], "object"] = {}  # binding mode: FrameLift per key (F9b)
    for key, views in sorted(observations.items()):
        if len(views) < min_views:
            continue
        if id_source == "binding":
            # The lift core also derives per-camera torso residuals (the chimera
            # signature) and per-joint covariance — the 04 (binding lift) feedback payload.
            member_keypoints = _member_keypoints(views)
            lift = lift_frame(
                member_keypoints,
                all_projections,
                reprojection_threshold_px=reprojection_threshold_px,
                min_views=min_views,
                cheirality=cheirality,
                compute_cov=True,
                robust_refit=robust_refit,
                robust_huber_px=robust_huber_px,
            )
            if lift is None:
                continue
            lifts[key] = lift
            raw[key] = (lift.points3d, lift.confidences, lift.mean_reprojection_errors)
            continue
        member_keypoints = _member_keypoints(views)
        keypoints = np.asarray(
            [member_keypoints[camera] for camera, _ in views], dtype=float
        )
        projections = np.asarray([all_projections[camera] for camera, _ in views], dtype=float)
        raw[key] = triangulate_skeleton_ransac(
            keypoints,
            projections,
            reprojection_threshold_px=reprojection_threshold_px,
            min_views=min_views,
            cheirality=cheirality,
            hartley=hartley,
            parallax_order=parallax_order,
        )

    # Smooth each identity in temporal order, then stamp the same 3D pose into
    # every contributing camera record for that synchronized frame.
    per_identity: dict[str, list[int]] = defaultdict(list)
    for frame_index, player_id in raw:
        per_identity[player_id].append(frame_index)
    smoothed: dict[tuple[int, str], np.ndarray] = {}
    smoothed_conf: dict[tuple[int, str], np.ndarray] = {}
    for player_id, frames in per_identity.items():
        frames = sorted(set(frames))
        joint_count = max(raw[(frame, player_id)][0].shape[0] for frame in frames)

        def _pad(arr: np.ndarray, width: int, fill: float) -> np.ndarray:
            if arr.shape[0] == joint_count:
                return arr
            padded = np.full((joint_count, width) if width else (joint_count,), fill)
            padded[: arr.shape[0]] = arr
            return padded

        if dense_fill:
            # C6: rows == real frames, so gap gates and smoothing operate in true
            # time; unobserved frames are NaN rows that fills may bridge only
            # within max_gap_frames of REAL frames.
            timeline = list(range(frames[0], frames[-1] + 1))
            row_of = {frame: row for row, frame in enumerate(timeline)}
            sequence = np.full((len(timeline), joint_count, 3), np.nan)
            confidences = np.zeros((len(timeline), joint_count))
            for frame in frames:
                sequence[row_of[frame]] = _pad(raw[(frame, player_id)][0], 3, np.nan)
                confidences[row_of[frame]] = _pad(raw[(frame, player_id)][1], 0, 0.0)
        else:
            timeline = frames
            row_of = {frame: row for row, frame in enumerate(frames)}
            sequence = np.asarray(
                [_pad(raw[(frame, player_id)][0], 3, np.nan) for frame in frames], dtype=float
            )
            confidences = np.asarray(
                [_pad(raw[(frame, player_id)][1], 0, 0.0) for frame in frames], dtype=float
            )
        # Extrapolate joints missing in scattered frames (occluded / <2 views) from their
        # temporal neighbours, then last-resort skeletal prior, so the emitted 3D pose is
        # complete instead of dropping the whole frame for one NaN joint.
        sequence, confidences = fill_occluded_joints(sequence, confidences)
        reference = _most_complete_pose(sequence)
        bones = _median_bone_lengths(
            sequence, parents=_HALPE26_PARENT if joint_count > 17 else _COCO17_PARENT
        )
        for index in range(sequence.shape[0]):
            if not np.isfinite(sequence[index]).all():
                sequence[index], confidences[index] = fill_from_skeletal_prior(
                    sequence[index], confidences[index], bones, reference,
                    parents=_HALPE26_PARENT if joint_count > 17 else _COCO17_PARENT,
                )
        if smoother == "butterworth":
            filtered = butterworth_smooth(
                sequence, fps=capture_fps, cutoff_hz=butter_cutoff_hz
            )
        else:
            filtered = confidence_ema_smooth(sequence, confidences, alpha=ema_alpha)
        smoothed.update({(frame, player_id): filtered[row_of[frame]] for frame in frames})
        smoothed_conf.update({(frame, player_id): confidences[row_of[frame]] for frame in frames})

    successful = 0
    extrapolated_joint_frames = 0
    measured_errors: list[np.ndarray] = []
    for key, views in observations.items():
        if key not in raw:
            continue
        points = smoothed[key]
        confidences = smoothed_conf[key]
        # Per-joint nullable emit: keep the frame as long as the mid-hip (COCO l/r hip)
        # triangulated, so a player-frame is not dropped just because the feet were not
        # multi-view-visible — body joints still ship, un-triangulated joints emit as null.
        finite = np.isfinite(points).all(axis=1)
        if not (finite[11] and finite[12]):
            continue  # no placeable root/ground position -> no 3D for this frame
        # Non-triangulated / extrapolated joints get a finite sentinel error + zero conf so
        # the contract validates; the null world point already flags them as unavailable.
        errors = np.asarray(raw[key][2], dtype=float).copy()
        was_measured = np.isfinite(errors)
        measured_errors.append(errors[was_measured])
        errors[~was_measured] = 100.0
        errors[~finite] = 100.0
        if not finite.all():
            extrapolated_joint_frames += 1
        confidences = np.where(finite, np.nan_to_num(confidences, nan=0.0), 0.0)
        # Canonical Halpe-26 world skeleton (feet included, per-joint nullable) plus the
        # self-describing named + root-relative view (root = mid-hip) for downstream consumers.
        pose_3d = {
            "keypoints_world_m": [
                points[j].tolist() if finite[j] else None for j in range(points.shape[0])
            ],
            "confidence": np.clip(confidences, 0.0, 1.0).tolist(),
            "mean_reprojection_error_px": errors.tolist(),
        }
        pose_3d_named = named_root_relative(points, HALPE26_KEYPOINTS)
        for _, player in views:
            player["pose_3d"] = dict(pose_3d)  # per-view copy: no cross-record aliasing
            player["pose_3d_named"] = dict(pose_3d_named)
        successful += 1

    output_prediction_dir = output_run_dir / "predictions"
    output_prediction_dir.mkdir(parents=True, exist_ok=True)
    for item in prediction_files:
        with (output_prediction_dir / item.path.name).open("w", encoding="utf-8") as handle:
            for frame_index in sorted(records_by_frame):
                record = records_by_frame[frame_index].get(item.camera_id)
                if record is None:
                    continue
                validate_group1_frame(record, final_handoff=False)
                handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

    purity_by_binding: dict[str, Any] = {}
    if id_source == "binding":
        # F9b diagnostics: the per-frame 3D stream keyed on binding_id plus the
        # whole-delivery purity report — the chimera-split signal and the shape
        # evidence Waves 3/4 consume.
        def _nan_to_none(values: np.ndarray) -> list:
            return [
                [None if not np.isfinite(v) else float(v) for v in row]
                if np.ndim(row) else (None if not np.isfinite(row) else float(row))
                for row in np.asarray(values, dtype=float)
            ]

        diagnostics_dir = output_run_dir / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        # Carry the P3 correspondences forward so global-id (05) can read everything
        # it needs from the 04 lift run (the lift now runs before global-id).
        src_correspondences = input_run_dir / "diagnostics" / "correspondences.jsonl"
        if src_correspondences.exists():
            (diagnostics_dir / "correspondences.jsonl").write_bytes(src_correspondences.read_bytes())
        by_frame: dict[int, list[dict]] = defaultdict(list)
        for (frame_index, binding), lift in sorted(lifts.items()):
            points = smoothed.get((frame_index, binding), lift.points3d)
            pelvis = np.asarray(points, dtype=float)[[11, 12]]
            pelvis_ok = np.isfinite(pelvis).all(axis=1)
            pelvis_xy = (
                [float(v) for v in pelvis[pelvis_ok].mean(axis=0)[:2]]
                if pelvis_ok.any() else None
            )
            ankle_z = lift.ankle_z
            entry = {
                "binding_id": binding,
                "keypoints_world_m": _nan_to_none(points),
                "confidence": [float(v) for v in np.nan_to_num(lift.confidences)],
                "mean_reprojection_error_px": _nan_to_none(lift.mean_reprojection_errors),
                "pelvis_ground_xy": pelvis_xy,
                "airborne": bool(np.isfinite(ankle_z) and ankle_z > airborne_ankle_z_m),
                "cov_diag_m2": (
                    _nan_to_none(lift.cov_diag_m2) if lift.cov_diag_m2 is not None else None
                ),
                "pelvis_cov_m2": (
                    [[float(v) for v in row] for row in lift.pelvis_cov_m2]
                    if lift.pelvis_cov_m2 is not None else None
                ),
            }
            by_frame[frame_index].append(entry)
        with (diagnostics_dir / "lift3d.jsonl").open("w", encoding="utf-8") as handle:
            for frame_index in sorted(by_frame):
                handle.write(json.dumps(
                    {"frame_index": frame_index, "bindings": by_frame[frame_index]},
                    sort_keys=True, allow_nan=False,
                ) + "\n")

        lifts_by_binding: dict[str, list] = defaultdict(list)
        for (frame_index, binding), lift in sorted(lifts.items()):
            lifts_by_binding[binding].append(lift)
        purity_by_binding = {
            binding: cluster_purity(
                binding_lifts,
                chimera_torso_residual_px=chimera_torso_residual_px,
                chimera_frame_fraction=chimera_frame_fraction,
            ).to_json()
            for binding, binding_lifts in sorted(lifts_by_binding.items())
        }
        _write_json(diagnostics_dir / "lift_purity.json", {
            "schema_version": "lift_purity/v1",
            "delivery_id": delivery_id,
            "thresholds": {
                "chimera_torso_residual_px": chimera_torso_residual_px,
                "chimera_frame_fraction": chimera_frame_fraction,
                "airborne_ankle_z_m": airborne_ankle_z_m,
            },
            "bindings": purity_by_binding,
        })

    created_at = datetime.now(timezone.utc).isoformat()
    all_measured = (
        np.concatenate(measured_errors)
        if measured_errors and any(chunk.size for chunk in measured_errors)
        else np.empty(0)
    )
    metrics = {
        "schema_version": "triangulation_metrics/v2",
        "created_at": created_at,
        "match_id": match_id,
        "delivery_id": delivery_id,
        "candidate_identity_frames": len(observations),
        "eligible_multiview_identity_frames": sum(
            len(views) >= min_views for views in observations.values()
        ),
        "triangulated_identity_frames": len(raw),
        "fully_valid_identity_frames": successful,
        "frames_with_extrapolated_joints": extrapolated_joint_frames,
        "triangulation_success_rate": successful / len(raw) if raw else 0.0,
        "triangulation_coverage": successful / len(observations) if observations else 0.0,
        "mean_reprojection_error_px": float(all_measured.mean()) if all_measured.size else None,
        "p95_reprojection_error_px": (
            float(np.percentile(all_measured, 95)) if all_measured.size else None
        ),
    }
    if id_source == "binding":
        suspects = [
            binding for binding, report in purity_by_binding.items()
            if report.get("chimera_suspect")
        ]
        metrics["id_source"] = "binding"
        metrics["bindings_lifted"] = len(purity_by_binding)
        metrics["chimera_suspect_count"] = len(suspects)
        metrics["chimera_suspects"] = suspects
    manifest = {
        "schema_version": "triangulation_run/v1",
        "created_at": created_at,
        "task": "multi_view_3d_lift",
        "input_run_dir": str(input_run_dir),
        "output_run_dir": str(output_run_dir),
        "drive_root": str(drive_root),
        "match_id": match_id,
        "delivery_id": delivery_id,
        "config": {
            "reprojection_threshold_px": reprojection_threshold_px,
            "min_views": min_views,
            "ema_alpha": ema_alpha,
            "cheirality": cheirality,
            "smoother": smoother,
            "butter_cutoff_hz": butter_cutoff_hz,
            "capture_fps": capture_fps,
            "id_source": id_source,
            "native_skeleton": native_skeleton,
            "dense_fill": dense_fill,
        },
    }
    _write_json(output_run_dir / "run_manifest.json", manifest)
    _write_json(output_run_dir / "triangulation_metrics.json", metrics)
    return metrics


def triangulate_legacy(args: argparse.Namespace) -> None:
    calibrations = load_calibration(args.calibration)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.predictions):
        player_id = row.get("global_player_id", row.get("player_id"))
        if player_id is None:
            continue
        grouped[(str(row["frame_id"]), str(player_id))].append(row)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for (frame_id, player_id), views in sorted(grouped.items()):
            usable = [view for view in views if view["camera_id"] in calibrations]
            if len(usable) < args.min_views:
                continue
            points, confidence, errors = triangulate_skeleton_ransac(
                np.asarray([view["keypoints"] for view in usable], dtype=float),
                np.asarray([calibrations[view["camera_id"]].projection_matrix for view in usable]),
                reprojection_threshold_px=args.reprojection_threshold_px,
                min_views=args.min_views,
            )
            payload = {
                "frame_id": frame_id,
                "timestamp_ns": int(usable[0].get("timestamp_ns", 0)),
                "player_id": player_id,
                "global_player_id": player_id,
                "calibration_id": usable[0].get("calibration_id"),
                "camera_ids": [view["camera_id"] for view in usable],
                "keypoints3d_world_m": np.column_stack([points, confidence]).tolist(),
                "mean_reprojection_error_px": float(np.nanmean(errors)) if np.isfinite(errors).any() else None,
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    canonical_values = (args.input_run_dir, args.output_run_dir, args.drive_root, args.delivery_id)
    if any(value is not None for value in canonical_values):
        if not all(value is not None for value in canonical_values):
            raise SystemExit("canonical mode requires --input-run-dir, --output-run-dir, --drive-root, --delivery-id")
        metrics = triangulate_canonical_run(
            input_run_dir=args.input_run_dir,
            output_run_dir=args.output_run_dir,
            drive_root=args.drive_root,
            delivery_id=args.delivery_id,
            cameras=args.camera,
            reprojection_threshold_px=args.reprojection_threshold_px,
            min_views=args.min_views,
            ema_alpha=args.ema_alpha,
            cheirality=args.cheirality,
            smoother=args.smoother,
            butter_cutoff_hz=args.butter_cutoff_hz,
            capture_fps=args.capture_fps,
            id_source=args.id_source,
            chimera_torso_residual_px=args.chimera_torso_residual_px,
            chimera_frame_fraction=args.chimera_frame_fraction,
            airborne_ankle_z_m=args.airborne_ankle_z_m,
            native_skeleton=args.native_skeleton,
            dense_fill=args.dense_fill,
            suppression_path=args.suppression_path,
            hartley=args.hartley,
            parallax_order=args.parallax_order,
            robust_refit=args.robust_refit,
            robust_huber_px=args.robust_huber_px,
        )
        print(f"P6: triangulated {metrics['fully_valid_identity_frames']} identity-frames")
        return 0
    if not all((args.predictions, args.calibration, args.output)):
        raise SystemExit("legacy mode requires --predictions, --calibration, and --output")
    triangulate_legacy(args)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

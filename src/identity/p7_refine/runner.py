"""Run the 07 (refinement) stage over a canonical run directory.

Reads an identity-assigned run dir (``predictions/*.jsonl`` carrying ``global_player_id``
+ ``pose_3d``, normally ``06_roles``), refines each identity's whole-clip 3D skeleton to
be physically valid and smooth, and writes a new run dir in the same canonical format with
only ``pose_3d`` / ``pose_3d_named`` rewritten. No identity field is touched, so IDs are
byte-identical to the input.

Being keyed purely on ``global_player_id`` + ``pose_3d``, the same runner works against any
stage that carries a triangulated pose (e.g. a future lift-first ``04_lift`` keyed on
``binding_id``) - it is deliberately decoupled from the pipeline ordering.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from core.calibration import load_projection_matrices_from_drive
from core.contract import validate_group1_frame
from core.keypoints import (
    HALPE26_BONES,
    HALPE26_BONE_LIMITS_M,
    HALPE26_KEYPOINTS,
    HALPE26_ROOT_INDEX,
    HALPE26_SYMMETRIC_BONES,
    named_root_relative,
)
from identity.p7_refine.refine import (
    RefineParams,
    bone_length_cv,
    jitter_stats,
    refine_identity_sequence,
)
from identity.p7_refine.relift import relift_sequence


def _infer_match_id(delivery_id: str) -> str:
    match = re.match(r"^(?P<match_id>.+?)M\d", delivery_id)
    if not match:
        raise ValueError(f"cannot infer match_id from delivery_id: {delivery_id}")
    return match.group("match_id")

_HIP_JOINTS = [11, 12, 19]  # l_hip, r_hip, mid-hip - the manager's "wobbly" joints

# Same canonical prediction filename contract the other stages use. Discovery is inlined
# (rather than importing the P2 tracker) to keep this stage free of the tracking / scipy
# import chain - the refinement is deliberately decoupled from pipeline ordering.
_CANONICAL_PREDICTION_RE = re.compile(
    r"^(?P<capture_group>bt_\d{2})__(?P<delivery_id>.+)__(?P<camera_id>cam_\d{2})\.jsonl$"
)


@dataclass(frozen=True)
class _PredictionFile:
    path: Path
    camera_id: str


def _discover_prediction_files(
    input_run_dir: Path, delivery_id: str, cameras: list[str] | None
) -> list[_PredictionFile]:
    wanted = set(cameras) if cameras else None
    files: list[_PredictionFile] = []
    for path in sorted((input_run_dir / "predictions").glob("*.jsonl")):
        match = _CANONICAL_PREDICTION_RE.match(path.name)
        if not match or match.group("delivery_id") != delivery_id:
            continue
        camera_id = match.group("camera_id")
        if wanted is not None and camera_id not in wanted:
            continue
        files.append(_PredictionFile(path, camera_id))
    return files


def _read_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _points_and_conf(pose_3d: dict, joint_count: int) -> tuple[np.ndarray, np.ndarray]:
    pts = np.full((joint_count, 3), np.nan, dtype=float)
    for j, entry in enumerate(pose_3d.get("keypoints_world_m") or []):
        if entry is not None:
            pts[j] = entry
    conf = np.asarray(pose_3d.get("confidence") or [0.0] * joint_count, dtype=float)
    return pts, conf


def _reproj_errors(pose_seq, obs_by_row, conf_min: float) -> list[float]:
    """Per (frame, joint, reliable-view) reprojection error in pixels for a 3D sequence.

    Projects each finite 3D joint back into every camera that reliably saw it (2D conf >=
    conf_min) and measures the pixel gap to the observed keypoint. Restricting to reliable
    views is what makes this a fair before/after measure — a hallucinated low-confidence
    keypoint (the umpire's edge legs) is never counted, so 'fixing' it can't look like a
    regression.
    """
    errors: list[float] = []
    for row, cam_obs in enumerate(obs_by_row):
        pose = pose_seq[row]
        for projection, kp, conf, _ in cam_obs:
            for j in range(pose.shape[0]):
                if conf[j] < conf_min or not np.isfinite(pose[j]).all():
                    continue
                x = projection @ np.append(pose[j], 1.0)
                if abs(x[2]) < 1e-9:
                    continue
                errors.append(float(np.hypot(*(x[:2] / x[2] - kp[j]))))
    return errors


def run_refinement(
    input_run_dir: str | Path,
    output_run_dir: str | Path,
    delivery_id: str,
    params: RefineParams,
    cameras: list[str] | None = None,
    drive_root: str | Path | None = None,
) -> dict:
    input_run_dir = Path(input_run_dir)
    output_run_dir = Path(output_run_dir)
    prediction_files = _discover_prediction_files(input_run_dir, delivery_id, cameras)
    if not prediction_files:
        raise RuntimeError(f"no canonical prediction files for delivery {delivery_id} in {input_run_dir}")

    # Load every camera file once; keep records in memory for the write-back pass.
    records_by_file: dict[str, list[dict]] = {}
    camera_of_file: dict[str, str] = {}
    joint_count = len(HALPE26_KEYPOINTS)
    for item in prediction_files:
        records_by_file[item.path.name] = _read_records(item.path)
        camera_of_file[item.path.name] = item.camera_id

    # Projection matrices for the visibility-aware re-lift (best-effort: if the drive is
    # unavailable we fall back to refining the existing pose_3d without re-lift).
    projections: dict[str, np.ndarray] = {}
    if params.relift and drive_root is not None:
        try:
            projections = load_projection_matrices_from_drive(drive_root, _infer_match_id(delivery_id))
        except (OSError, ValueError, KeyError):
            projections = {}

    # Per-identity frame -> existing (points, confidence), plus per-camera 2D observations
    # (for the re-lift). The same pose_3d is stamped on every camera record, so dedupe.
    per_id_raw: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = defaultdict(dict)
    per_id_obs: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for fname, records in records_by_file.items():
        cam = camera_of_file[fname]
        for rec in records:
            frame = int(rec["frame_index"])
            imgsz = (rec.get("metadata") or {}).get("image_size_px")
            for player in rec.get("players", []):
                gid = player.get("global_player_id")
                pose_3d = player.get("pose_3d")
                if not gid:
                    continue
                if pose_3d and frame not in per_id_raw[gid]:
                    per_id_raw[gid][frame] = _points_and_conf(pose_3d, joint_count)
                pose_2d = player.get("pose_2d")
                if pose_2d and cam in projections:
                    kp = np.asarray(pose_2d["keypoints_px"], dtype=float)
                    cf = np.asarray(pose_2d["confidence"], dtype=float)
                    per_id_obs[gid][frame].append((projections[cam], kp, cf, imgsz))

    # Refine each identity over a dense (real-frame) timeline.
    refined_by_id: dict[str, dict[int, np.ndarray]] = {}
    before_seqs: list[np.ndarray] = []
    after_seqs: list[np.ndarray] = []
    reproj_before: list[float] = []
    reproj_after: list[float] = []
    for gid, by_frame in per_id_raw.items():
        frames = sorted(by_frame)
        timeline = list(range(frames[0], frames[-1] + 1))
        row_of = {frame: row for row, frame in enumerate(timeline)}
        seq = np.full((len(timeline), joint_count, 3), np.nan, dtype=float)
        conf = np.zeros((len(timeline), joint_count), dtype=float)
        for frame in frames:
            pts, cf = by_frame[frame]
            seq[row_of[frame]] = pts
            conf[row_of[frame]] = cf
        obs_by_row = [per_id_obs[gid].get(frame, []) for frame in timeline] if projections else []

        if not params.enabled:
            refined = seq
        else:
            work_seq, work_conf = seq, conf
            # Visibility-aware re-lift: fixes joints stretched by a partially-visible view
            # (e.g. an umpire whose legs are only in one camera). Needs the calibration.
            if params.relift and projections:
                relifted, rconf = relift_sequence(
                    obs_by_row, list(HALPE26_BONES),
                    [tuple(p) for p in HALPE26_SYMMETRIC_BONES],
                    root_index=HALPE26_ROOT_INDEX, joint_count=joint_count,
                    conf_min=params.vis_conf, margin=params.edge_margin_px,
                    limits=HALPE26_BONE_LIMITS_M, fallback_seq=seq,
                )
                # Re-lifted joints are already visibility-vetted, so give them full trust
                # (the confidence gate must not drop the single-view legs we just fixed).
                finite = np.isfinite(relifted).all(axis=2)
                work_seq = relifted
                work_conf = np.where(finite, np.maximum(rconf, 0.9), 0.0)
            refined, _ = refine_identity_sequence(work_seq, work_conf, params)

        refined_by_id[gid] = {frame: refined[row_of[frame]] for frame in frames}
        before_seqs.append(seq[[row_of[f] for f in frames]])
        after_seqs.append(refined[[row_of[f] for f in frames]])
        if obs_by_row:
            reproj_before += _reproj_errors(seq, obs_by_row, params.vis_conf)
            reproj_after += _reproj_errors(refined, obs_by_row, params.vis_conf)

    # Write-back pass: rewrite pose_3d/pose_3d_named on every camera record.
    output_prediction_dir = output_run_dir / "predictions"
    output_prediction_dir.mkdir(parents=True, exist_ok=True)
    players_rewritten = 0
    for item in prediction_files:
        records = records_by_file[item.path.name]
        for rec in records:
            frame = int(rec["frame_index"])
            for player in rec.get("players", []):
                gid = player.get("global_player_id")
                pose_3d = player.get("pose_3d")
                if not gid or not pose_3d or gid not in refined_by_id:
                    continue
                pts = refined_by_id[gid].get(frame)
                if pts is None:
                    continue
                finite = np.isfinite(pts).all(axis=1)
                pose_3d["keypoints_world_m"] = [
                    pts[j].tolist() if finite[j] else None for j in range(joint_count)
                ]
                # confidence / mean_reprojection_error_px stay finite length-26 (contract);
                # zero confidence for joints the refinement could not place.
                conf = np.asarray(pose_3d.get("confidence") or [0.0] * joint_count, dtype=float)
                conf = np.where(finite, np.clip(np.nan_to_num(conf, nan=0.0), 0.0, 1.0), 0.0)
                pose_3d["confidence"] = conf.tolist()
                player["pose_3d"] = pose_3d
                player["pose_3d_named"] = named_root_relative(pts, HALPE26_KEYPOINTS)
                players_rewritten += 1
        with (output_prediction_dir / item.path.name).open("w", encoding="utf-8") as handle:
            for rec in records:
                validate_group1_frame(rec, final_handoff=False)
                handle.write(json.dumps(rec, sort_keys=True, allow_nan=False) + "\n")

    metrics = _build_metrics(delivery_id, params, before_seqs, after_seqs, players_rewritten,
                             reproj_before, reproj_after)
    created_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": "refinement_run/v1",
        "created_at": created_at,
        "task": "physics_3d_refinement",
        "input_run_dir": str(input_run_dir),
        "output_run_dir": str(output_run_dir),
        "delivery_id": delivery_id,
        "config": {
            "enabled": params.enabled, "conf_floor": params.conf_floor,
            "root_cutoff_hz": params.root_cutoff_hz, "limb_cutoff_hz": params.limb_cutoff_hz,
            "clamp_angles": params.clamp_angles,
        },
        "cameras": sorted(item.camera_id for item in prediction_files),
    }
    output_run_dir.mkdir(parents=True, exist_ok=True)
    with (output_run_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (output_run_dir / "refinement_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return metrics


def _build_metrics(delivery_id, params, before_seqs, after_seqs, players_rewritten,
                   reproj_before=None, reproj_after=None) -> dict:
    from core.keypoints import HALPE26_BONES

    bones = list(HALPE26_BONES)

    def _agg(seqs, joints=None):
        jit = [jitter_stats(s, joints) for s in seqs if s.shape[0] >= 2]
        cv = [bone_length_cv(s, bones) for s in seqs if s.shape[0] >= 2]
        mean_jit = float(np.mean([j["mean_m"] for j in jit])) if jit else 0.0
        p95_jit = float(np.mean([j["p95_m"] for j in jit])) if jit else 0.0
        max_cv = float(np.mean([c["max_bone_cv"] for c in cv])) if cv else 0.0
        return mean_jit, p95_jit, max_cv

    def _reproj(errs):
        if not errs:
            return None, None, 0
        a = np.asarray(errs)
        return float(a.mean()), float(np.percentile(a, 90)), int(a.size)

    b_jit, b_p95, b_cv = _agg(before_seqs)
    a_jit, a_p95, a_cv = _agg(after_seqs)
    b_hip, _, _ = _agg(before_seqs, _HIP_JOINTS)
    a_hip, _, _ = _agg(after_seqs, _HIP_JOINTS)
    rb_mean, rb_p90, rn = _reproj(reproj_before)
    ra_mean, ra_p90, _ = _reproj(reproj_after)
    return {
        "schema_version": "refinement_metrics/v1",
        "delivery_id": delivery_id,
        "status": "pass",
        "enabled": params.enabled,
        "identities": len(before_seqs),
        "players_rewritten": players_rewritten,
        "jitter_mean_m_before": b_jit, "jitter_mean_m_after": a_jit,
        "jitter_p95_m_before": b_p95, "jitter_p95_m_after": a_p95,
        "hip_jitter_mean_m_before": b_hip, "hip_jitter_mean_m_after": a_hip,
        "max_bone_cv_before": b_cv, "max_bone_cv_after": a_cv,
        # Reprojection error (px) against reliably-seen 2D keypoints — the fidelity metric.
        "reproj_px_mean_before": rb_mean, "reproj_px_mean_after": ra_mean,
        "reproj_px_p90_before": rb_p90, "reproj_px_p90_after": ra_p90,
        "reproj_sample_count": rn,
    }

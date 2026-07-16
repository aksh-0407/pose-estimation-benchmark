#!/usr/bin/env python3
"""Per-camera robustness + monocular-vs-multiview diagnostics (read-only).

Answers the user's questions directly, with no ground truth:

1. PER-CAMERA REPROJECTION CONTRIBUTION — reproject each identity-frame's triangulated
   3D skeleton (from 04 lift3d.jsonl) into every camera that observed it and compare to
   that camera's 2D keypoints. Aggregated per camera_id, this surfaces the "one bad /
   outlier camera" for a delivery (the camera whose 2D consistently disagrees with the
   multi-view 3D consensus).

2. LEAVE-ONE-CAMERA-OUT (LOCO) STABILITY — for identity-frames with >=3 views, retriangulate
   the mid-hip from all views, then again dropping one camera; the per-camera shift
   ||p_all - p_drop|| quantifies how much each camera moves the solution (robustness to
   losing / mistrusting that camera).

3. MONOCULAR-vs-MULTIVIEW dispersion — per joint, the spread of each camera-PAIR's
   2-view triangulation around the full multi-view point: how much a 2-camera ("nearly
   monocular") estimate would differ from using all cameras, per joint.

Join: 03 correspondences give per (frame, binding_id) the member (cam_id, player_index);
04 lift3d gives the triangulated keypoints_world_m; the per-camera prediction jsonl gives
pose_2d. Calibration via load_projection_matrices_from_drive.

Usage:
    python -m tools.diagnosis.camera_robustness \
        --run-root <tree>/<...>/pipetrack_v<N>  --drive-root data/raw/8_init \
        --deliveries CCPL080626M1_1_14_3 ...
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from itertools import combinations

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
if ROOT not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

from core.calibration import load_projection_matrices_from_drive  # noqa: E402
from identity.common.triangulation import ransac_triangulate_point  # noqa: E402
from identity.p2_tracking.runner import infer_match_id  # noqa: E402

KP = 26
CONF_THR = 0.3
HIP = (11, 12)  # COCO mid-hip


def _reproject(P: np.ndarray, xyz: np.ndarray) -> np.ndarray | None:
    h = P @ np.array([xyz[0], xyz[1], xyz[2], 1.0])
    if abs(h[2]) < 1e-9:
        return None
    return h[:2] / h[2]


def _load_lift3d(delivery_dir: str) -> dict:
    """(frame, binding) -> (26,3) world keypoints (np.nan for null)."""
    out = {}
    path = os.path.join(delivery_dir, "04_lift", "diagnostics", "lift3d.jsonl")
    if not os.path.exists(path):
        return out
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        fi = int(row["frame_index"])
        for b in row.get("bindings", []):
            kw = b.get("keypoints_world_m")
            if kw is None:
                continue
            arr = np.array([[np.nan, np.nan, np.nan] if p is None else p for p in kw], dtype=float)
            out[(fi, str(b["binding_id"]))] = arr
    return out


def _load_corr(delivery_dir: str) -> dict:
    """frame -> list of (binding_id, [(cam_id, player_index)])."""
    out = defaultdict(list)
    path = os.path.join(delivery_dir, "03_association", "diagnostics", "correspondences.jsonl")
    if not os.path.exists(path):
        return out
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        fi = int(row["frame_index"])
        for c in row.get("clusters", []):
            b = c.get("binding_id")
            if b is None:
                continue
            members = [(m["cam_id"], int(m["player_index"])) for m in c.get("members", []) if m.get("cam_id") is not None]
            out[fi].append((str(b), members))
    return out


def _load_preds(delivery_dir: str) -> dict:
    """(frame, cam) -> players list (for pose_2d)."""
    out = {}
    # prefer the 03 predictions (the 2D that fed triangulation); fall back to 02.
    for stage in ("03_association", "02_tracking"):
        files = glob.glob(os.path.join(delivery_dir, stage, "predictions", "*.jsonl"))
        if files:
            for f in files:
                cam = f.split("__")[-1].replace(".jsonl", "")
                for line in open(f, encoding="utf-8"):
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    out[(int(r["frame_index"]), cam)] = r.get("players", [])
            return out
    return out


def audit(delivery_dir: str, drive_root: str, delivery: str, loco_stride: int = 1) -> dict:
    proj = load_projection_matrices_from_drive(drive_root, infer_match_id(delivery))
    lift = _load_lift3d(delivery_dir)
    corr = _load_corr(delivery_dir)
    preds = _load_preds(delivery_dir)

    per_cam_res: dict[str, list[float]] = defaultdict(list)
    loco: dict[str, list[float]] = defaultdict(list)
    mono_multi: list[float] = []

    for fi, clusters in corr.items():
        for binding, members in clusters:
            kw = lift.get((fi, binding))
            if kw is None:
                continue
            # (1) per-camera reprojection of the triangulated skeleton
            hip_obs = []  # (cam, pixel, P) for the mid-hip, for LOCO/mono-multi
            for cam, pidx in members:
                P = proj.get(cam)
                players = preds.get((fi, cam))
                if P is None or players is None or pidx >= len(players):
                    continue
                pose = players[pidx].get("pose_2d") or {}
                kps = np.asarray(pose.get("keypoints_px", []), dtype=float)
                conf = np.asarray(pose.get("confidence", []), dtype=float)
                if kps.shape != (KP, 2) or conf.shape != (KP,):
                    continue
                for j in range(KP):
                    if not np.isfinite(kw[j]).all() or conf[j] < CONF_THR:
                        continue
                    rp = _reproject(P, kw[j])
                    if rp is not None:
                        per_cam_res[cam].append(float(np.linalg.norm(rp - kps[j])))
                # mid-hip pixel for LOCO / mono-multi
                hip_px = kps[list(HIP)]
                hip_cf = conf[list(HIP)]
                ok = hip_cf >= CONF_THR
                if ok.any():
                    hip_obs.append((cam, hip_px[ok].mean(axis=0), P))

            # (2) LOCO + (3) mono-vs-multi on the mid-hip (>=3 views). Re-triangulation is
            # expensive, so subsample frames by loco_stride for large runs.
            if len(hip_obs) >= 3 and (fi % loco_stride == 0):
                pts = np.array([o[1] for o in hip_obs])
                Ps = np.array([o[2] for o in hip_obs])
                full = ransac_triangulate_point(pts, Ps, reprojection_threshold_px=12.0, min_views=2)
                if np.isfinite(full.point_xyz).all():
                    for i, (cam, _, _) in enumerate(hip_obs):
                        keep = [k for k in range(len(hip_obs)) if k != i]
                        drop = ransac_triangulate_point(pts[keep], Ps[keep], reprojection_threshold_px=12.0, min_views=2)
                        if np.isfinite(drop.point_xyz).all():
                            loco[cam].append(float(np.linalg.norm(full.point_xyz - drop.point_xyz)))
                    # mono (each pair) vs multi
                    for a, b in combinations(range(len(hip_obs)), 2):
                        pair = ransac_triangulate_point(pts[[a, b]], Ps[[a, b]], reprojection_threshold_px=1e9, min_views=2)
                        if np.isfinite(pair.point_xyz).all():
                            mono_multi.append(float(np.linalg.norm(full.point_xyz - pair.point_xyz)))

    def stats(vals):
        a = np.asarray(vals, dtype=float)
        if a.size == 0:
            return {"n": 0}
        return {"n": int(a.size), "mean": round(float(a.mean()), 2), "p95": round(float(np.percentile(a, 95)), 2)}

    cams = sorted(per_cam_res)
    return {
        "delivery": delivery.replace("CCPL080626", ""),
        "per_camera_reproj_px": {c: stats(per_cam_res[c]) for c in cams},
        "loco_hip_shift_m": {c: stats(loco[c]) for c in sorted(loco)},
        "mono_vs_multi_hip_m": stats(mono_multi),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--drive-root", default="data/raw/8_init")
    ap.add_argument("--deliveries", nargs="+", required=True)
    ap.add_argument("--loco-stride", type=int, default=1,
                    help="Subsample stride for the expensive LOCO / mono-vs-multi re-triangulation.")
    args = ap.parse_args(argv)

    agg_cam = defaultdict(list)
    agg_loco = defaultdict(list)
    for d in args.deliveries:
        ddir = os.path.join(args.run_root, d)
        if not os.path.isdir(ddir):
            continue
        res = audit(ddir, args.drive_root, d, loco_stride=args.loco_stride)
        print(json.dumps(res), flush=True)
        for c, s in res["per_camera_reproj_px"].items():
            if s["n"]:
                agg_cam[c].append((s["mean"], s["n"]))
        for c, s in res["loco_hip_shift_m"].items():
            if s["n"]:
                agg_loco[c].append((s["mean"], s["n"]))

    print("\n=== AGGREGATE per-camera reprojection (mean px, weighted) ===", flush=True)
    for c in sorted(agg_cam):
        w = sum(n for _, n in agg_cam[c]); m = sum(v * n for v, n in agg_cam[c]) / max(w, 1)
        print(f"  {c}: mean={m:.2f}px  (n={w})", flush=True)
    print("=== AGGREGATE leave-one-camera-out hip shift (mean m, weighted) ===", flush=True)
    for c in sorted(agg_loco):
        w = sum(n for _, n in agg_loco[c]); m = sum(v * n for v, n in agg_loco[c]) / max(w, 1)
        print(f"  {c}: mean={m:.3f}m  (n={w})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

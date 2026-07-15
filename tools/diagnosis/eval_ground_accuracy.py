#!/usr/bin/env python3
"""Ground-location accuracy metrics for a P3/P4 run (ISSUE-12).

No per-player ground truth exists, so this reports two calibration-anchored proxies that
DO move with emitted-position quality (unlike ``ground_spread_m``, which only measures
member disagreement):

* ``emitted_reproj_px`` — project each cluster's emitted ``ground_xy`` (z=0) into every
  member camera and compare to that member's foot pixel. Lower = the reported point
  better explains all the views it was built from. Computed for multi-camera clusters.
* ``dist_to_triangulated_foot_m`` — distance from the emitted ``ground_xy`` to the
  calibration-optimal RANSAC-triangulated foot (>=3 views, tri reproj < 12 px). Lower =
  closer to the geometry-optimal point. This is the fairer accuracy proxy.

Usage:
    python -m tools.diagnosis.eval_ground_accuracy \
        --run-dir data/derived/runs/pipetrack_v3/deliveries/<delivery>/p3 \
        --drive-root drive --match-id CCPL080626
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from identity.common.triangulation import ransac_triangulate_point, reprojection_errors_for_point  # noqa: E402
from core.calibration import load_projection_matrices_from_drive  # noqa: E402


def _foot_bottom(bbox: list[float]) -> np.ndarray:
    x, y, w, h = bbox
    return np.array([x + w / 2.0, y + h], dtype=float)


def _load_correspondences(run_dir: Path) -> dict:
    out = {}
    path = run_dir / "diagnostics" / "correspondences.jsonl"
    for line in path.open("r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        for cluster in row.get("clusters", []):
            out[(row["frame_index"], cluster["cluster_id"])] = cluster
    return out


def evaluate(run_dir: Path, drive_root: Path, match_id: str) -> dict:
    proj = load_projection_matrices_from_drive(drive_root, match_id)
    clusters = _load_correspondences(run_dir)

    reproj_emit: list[float] = []
    dist_tri: list[float] = []
    single_cam = multi_cam = 0
    for (_frame, _cid), cluster in clusters.items():
        members = cluster.get("members", [])
        ground = cluster.get("ground_xy")
        if ground is None or not np.isfinite(ground).all():
            continue
        ground = np.asarray(ground, dtype=float)
        if len(members) <= 1:
            single_cam += 1
            continue
        multi_cam += 1
        feet, projections = [], []
        for m in members:
            cam, bbox = m.get("cam_id"), m.get("bbox_xywh_px")
            if cam in proj and bbox:
                foot = _foot_bottom(bbox)
                feet.append(foot)
                projections.append(proj[cam])
                h = proj[cam] @ np.array([ground[0], ground[1], 0.0, 1.0])
                if abs(h[2]) > 1e-9:
                    reproj_emit.append(float(np.linalg.norm(h[:2] / h[2] - foot)))
        if len(feet) >= 3:
            feet_a, proj_a = np.asarray(feet), np.asarray(projections)
            res = ransac_triangulate_point(feet_a, proj_a, reprojection_threshold_px=12.0, min_views=3)
            if np.isfinite(res.point_xyz).all() and int(res.inlier_mask.sum()) >= 3:
                errs = reprojection_errors_for_point(res.point_xyz, feet_a, proj_a)
                if np.isfinite(errs).any() and np.nanmax(errs[res.inlier_mask]) <= 12.0:
                    dist_tri.append(float(np.linalg.norm(ground - res.point_xyz[:2])))

    def stats(values: list[float]) -> dict:
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return {"count": 0, "mean": None, "p50": None, "p95": None}
        return {
            "count": int(arr.size),
            "mean": round(float(arr.mean()), 4),
            "p50": round(float(np.percentile(arr, 50)), 4),
            "p95": round(float(np.percentile(arr, 95)), 4),
        }

    return {
        "run_dir": str(run_dir),
        "single_camera_clusters": single_cam,
        "multi_camera_clusters": multi_cam,
        "emitted_reproj_px": stats(reproj_emit),
        "dist_to_triangulated_foot_m": stats(dist_tri),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--drive-root", default="data/raw/8_init",
                    help="Dataset raw root (default: data/raw/8_init).")
    ap.add_argument("--match-id", default="CCPL080626")
    args = ap.parse_args(argv)
    result = evaluate(Path(args.run_dir), Path(args.drive_root), args.match_id)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

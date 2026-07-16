#!/usr/bin/env python3
"""Coverage audit — is a "single-camera" player a genuine geometry limit, or a P3
association under-merge?

Domain fact (rig geometry): the central players (striker / non-striker / bowler /
keeper) sit in ~4 cameras' field of view (worst case 2, ~never 1); only umpires and
deep fielders are legitimately seen by 1-2. So a high single-camera rate for the
central group is suspicious — it may mean P3 failed to BIND detections that exist in
3-4 cameras, not that the cameras don't see the player.

This tool separates the two. Per frame it single-link clusters the RAW pre-association
``02_tracking`` detections by ground proximity (<1.5 m) and counts DISTINCT cameras =
TRUE geometric coverage. It then matches each true cluster to the nearest
``03_association`` cluster and reads the ACHIEVED coverage (member cameras) + the
``single_camera`` flag. The gap is the diagnosis:

* true_cov >= 3 but achieved <= 2  -> ASSOCIATION UNDER-MERGE (fixable in P3: pose-shape
  primary cue, parallax-adaptive gate, splittable clustering). Also tallies which camera
  PAIRS were left unbound, to test the facing-pair (01-04 / 02-06 / 03-05) hypothesis.
* true_cov == 1                     -> GENUINE single camera (detection-recall / periphery).

No ground truth is used; everything is calibration geometry.

Usage:
    python -m tools.diagnosis.coverage_audit \
        --run-root ~/bits-pose-data/derived/8_init/pipetrack_v90 \
        --drive-root ~/bits-pose-data/raw/8_init \
        --deliveries CCPL080626M1_1_14_1 CCPL080626M1_1_14_4 ...
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
if ROOT not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

from core.calibration import build_ground_calibrators  # noqa: E402
from identity.p2_tracking.runner import infer_match_id  # noqa: E402

RAD_M = 1.5
FACING_PAIRS = {("01", "04"), ("02", "06"), ("03", "05")}


def _single_link(points: list[np.ndarray], rad: float = RAD_M):
    n = len(points)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(points[i] - points[j]) <= rad:
                parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


def _cam_of(path: str) -> str:
    return path.split("__")[-1].replace(".jsonl", "")


def _load_raw_dets(delivery_dir: str, cals: dict) -> dict:
    """frame -> list of (cam, ground_xy) from 02_tracking (pre-association)."""
    by_frame = defaultdict(list)
    for f in glob.glob(os.path.join(delivery_dir, "02_tracking", "predictions", "*.jsonl")):
        cam = _cam_of(f)
        cal = cals.get(cam)
        if cal is None:
            continue
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            fi = int(r["frame_index"])
            for p in r.get("players", []):
                b = p.get("bbox_xywh_px")
                if not b:
                    continue
                xy = cal.bbox_bottom_center_ground_xy([float(v) for v in b])
                if xy is None or not np.isfinite(xy).all():
                    continue
                by_frame[fi].append((cam, np.asarray(xy, dtype=float)))
    return by_frame


def _load_clusters(delivery_dir: str) -> tuple[dict, int]:
    """frame -> list of {ground_xy, cams:set, single_camera}; plus same-camera collisions."""
    by_frame = defaultdict(list)
    collisions = 0
    path = os.path.join(delivery_dir, "03_association", "diagnostics", "correspondences.jsonl")
    if not os.path.exists(path):
        return by_frame, collisions
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        fi = int(r["frame_index"])
        for c in r.get("clusters", []):
            g = c.get("ground_xy")
            if g is None or not np.isfinite(g).all():
                continue
            member_cams = [m.get("cam_id") for m in c.get("members", []) if m.get("cam_id")]
            cams = set(member_cams)
            if len(member_cams) != len(cams):
                collisions += 1  # two detections from the same camera in one cluster
            by_frame[fi].append({
                "xy": np.asarray(g, dtype=float),
                "cams": cams,
                "single_camera": bool(c.get("single_camera", len(cams) <= 1)),
            })
    return by_frame, collisions


def audit_delivery(delivery_dir: str, drive_root: str, delivery: str,
                   assoc_dir: str | None = None) -> dict:
    cals = build_ground_calibrators(drive_root, infer_match_id(delivery))
    raw = _load_raw_dets(delivery_dir, cals)                 # 02_tracking (base tree)
    clusters, collisions = _load_clusters(assoc_dir or delivery_dir)  # 03 (variant tree)

    true_hist = defaultdict(int)          # true distinct-camera count -> #player-locations
    undermerge = 0                        # true>=3 but achieved<=2 at that location
    undermerge_locations = 0              # true>=3 total
    pair_unbound = defaultdict(int)       # camera pairs co-seeing but left in diff clusters
    matched_true_vs_achieved = []         # (true_cov, achieved_cov)

    for fi, dets in raw.items():
        pts = [xy for _cam, xy in dets]
        for grp in _single_link(pts):
            cams = {dets[i][0] for i in grp}
            true_cov = len(cams)
            true_hist[min(true_cov, 4)] += 1
            if true_cov < 2:
                continue
            centroid = np.mean([dets[i][1] for i in grp], axis=0)
            # nearest 03 cluster to this physical location
            best = None
            best_d = RAD_M
            for c in clusters.get(fi, []):
                d = float(np.linalg.norm(c["xy"] - centroid))
                if d <= best_d:
                    best_d = d
                    best = c
            achieved = len(best["cams"]) if best else 0
            matched_true_vs_achieved.append((true_cov, achieved))
            if true_cov >= 3:
                undermerge_locations += 1
                if achieved <= 2:
                    undermerge += 1
                    # tally the camera pairs that were co-visible but not co-bound
                    bound = best["cams"] if best else set()
                    unbound_cams = sorted(cams - bound) if bound else sorted(cams)
                    for a in range(len(unbound_cams)):
                        for b in range(a + 1, len(unbound_cams)):
                            ca, cb = unbound_cams[a][-2:], unbound_cams[b][-2:]
                            pair_unbound[tuple(sorted((ca, cb)))] += 1

    total_locs = sum(true_hist.values())
    return {
        "delivery": delivery,
        "true_coverage_hist": {f"{k}cam": true_hist[k] for k in sorted(true_hist)},
        "true_multicam_frac": round(sum(v for k, v in true_hist.items() if k >= 2) / max(total_locs, 1), 3),
        "true_ge3_frac": round(sum(v for k, v in true_hist.items() if k >= 3) / max(total_locs, 1), 3),
        "same_camera_collisions": collisions,
        "undermerge_locations_ge3": undermerge_locations,
        "undermerge_count": undermerge,
        "undermerge_rate": round(undermerge / max(undermerge_locations, 1), 3),
        "top_unbound_pairs": sorted(
            ({"pair": f"{a}-{b}", "n": n, "facing": (a, b) in FACING_PAIRS}
             for (a, b), n in pair_unbound.items()),
            key=lambda x: -x["n"],
        )[:6],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", required=True, help="tree with <DELIVERY>/02_tracking (raw coverage)")
    ap.add_argument("--assoc-root", default=None,
                    help="tree with <DELIVERY>/03_association (variant); defaults to --run-root")
    ap.add_argument("--drive-root", default="data/raw/8_init")
    ap.add_argument("--deliveries", nargs="+", required=True)
    args = ap.parse_args(argv)

    agg_true = defaultdict(int)
    agg_ml = agg_um = agg_coll = 0
    agg_pairs = defaultdict(lambda: [0, False])
    for d in args.deliveries:
        ddir = os.path.join(args.run_root, d)
        adir = os.path.join(args.assoc_root or args.run_root, d)
        if not os.path.isdir(ddir):
            print(f"SKIP {d}: no dir", flush=True)
            continue
        res = audit_delivery(ddir, args.drive_root, d, assoc_dir=adir)
        print(json.dumps(res), flush=True)
        for k, v in res["true_coverage_hist"].items():
            agg_true[k] += v
        agg_ml += res["undermerge_locations_ge3"]
        agg_um += res["undermerge_count"]
        agg_coll += res["same_camera_collisions"]
        for p in res["top_unbound_pairs"]:
            agg_pairs[p["pair"]][0] += p["n"]
            agg_pairs[p["pair"]][1] = p["facing"]

    print("\n=== AGGREGATE ===", flush=True)
    print("true coverage histogram:", dict(sorted(agg_true.items())), flush=True)
    print(f"under-merge: {agg_um}/{agg_ml} locations where true>=3 but P3 kept <=2 "
          f"({100 * agg_um / max(agg_ml, 1):.0f}%)   same_camera_collisions={agg_coll}", flush=True)
    print("top unbound camera pairs (facing = 01-04/02-06/03-05):", flush=True)
    for pair, (n, facing) in sorted(agg_pairs.items(), key=lambda x: -x[1][0])[:8]:
        print(f"  {pair}: {n} {'<-- FACING' if facing else ''}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

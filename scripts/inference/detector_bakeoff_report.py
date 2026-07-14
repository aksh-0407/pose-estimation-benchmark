#!/usr/bin/env python3
"""Analyze a detector_bakeoff.py output directory: recall/precision-proxy tables.

For each candidate vs the baseline candidate (default m640), reports per camera:
  det/frame at the production threshold, small-box (<100 px) det/frame,
  new-box rate (boxes with no IoU>=0.5 match in the baseline, same frame),
  lost-box rate (baseline boxes the candidate no longer finds).
New boxes are recall gains if real; the overlay jpgs are the precision check.

Usage: python scripts/inference/detector_bakeoff_report.py --dir <out> [--baseline m640]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PRODUCTION_THR = 0.30
SMALL_PX = 100.0
MATCH_IOU = 0.5


def load(cdir: Path) -> dict[tuple[str, str, int], np.ndarray]:
    out = {}
    with open(cdir / "detections.jsonl") as fh:
        for line in fh:
            r = json.loads(line)
            boxes = np.asarray(r["boxes"], dtype=float).reshape(-1, 5)
            out[(r["delivery"], r["camera"], r["frame_index"])] = boxes
    return out


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--baseline", default="m640")
    ap.add_argument("--thr", type=float, default=PRODUCTION_THR)
    args = ap.parse_args()
    root = Path(args.dir).expanduser()
    manifest = json.loads((root / "bakeoff_manifest.json").read_text())
    names = [s["candidate"] for s in manifest["summaries"] if "error" not in s]
    base = load(root / args.baseline)

    print(f"threshold {args.thr}; baseline {args.baseline}; match IoU {MATCH_IOU}")
    header = f"{'cand':>6} {'cam':>26} {'det/f':>6} {'small/f':>8} {'new/f':>6} {'lost/f':>7} {'minH':>6}"
    for name in names:
        det = load(root / name)
        percam: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for key, boxes in det.items():
            cam = f"{key[0].split('_')[-1]}/{key[1]}"
            st = percam[cam]
            st["frames"] += 1
            keep = boxes[boxes[:, 4] >= args.thr]
            st["dets"] += len(keep)
            if len(keep):
                hts = keep[:, 3] - keep[:, 1]
                st["small"] += int((hts < SMALL_PX).sum())
                st["minh"] = min(st.get("minh", 1e9), float(hts.min()))
            base_boxes = base.get(key, np.zeros((0, 5)))
            bkeep = base_boxes[base_boxes[:, 4] >= args.thr]
            m = iou_matrix(keep, bkeep)
            if len(keep):
                st["new"] += len(keep) if not len(bkeep) else int((m.max(axis=1) < MATCH_IOU).sum())
            if len(bkeep):
                st["lost"] += len(bkeep) if not len(keep) else int((m.max(axis=0) < MATCH_IOU).sum())
        print(f"\n== {name}")
        print(header)
        for cam in sorted(percam):
            st = percam[cam]
            f = max(1.0, st["frames"])
            print(f"{name:>6} {cam:>26} {st['dets']/f:6.2f} {st['small']/f:8.3f} "
                  f"{st['new']/f:6.2f} {st['lost']/f:7.3f} {st.get('minh', float('nan')):6.0f}")


if __name__ == "__main__":
    main()

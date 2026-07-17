#!/usr/bin/env python3
"""Diagnose co-located distinct global ids (the ghost-under-player swap class).

A *swap event* is two global ids whose fused ground positions stay within
``--radius`` of each other for at least ``--min-frames`` frames. Sub-classes:

  disjoint-cameras : the two ids NEVER share a camera-frame over the event  - 
                     the classic facing-pair split (one physical player carrying
                     two ids in different camera sets). These are the mergeable
                     class; the renderer draws each as the other's ghost.
  shared-camera    : the ids co-occur in at least one camera-frame - genuinely
                     two people standing close (crossing); NOT mergeable.

Usage:
    python -m tools.diagnosis.diagnose_colocated_ids \
        --tree data/derived/runs/pipetrack_v8.0 [--radius 0.75] [--min-frames 25] \
        [--dump-frames N --drive-root drive]

Emits a per-delivery table + optional annotated frame crops under
``<tree>/deliveries/<D>/p4/diagnostics/swap_events/``.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def load_tracks(p4_dir: Path):
    """(positions[frame][pid] = xy, occupancy[pid] = {(camera, frame), ...})"""

    positions: dict[int, dict[str, np.ndarray]] = defaultdict(dict)
    for line in open(p4_dir / "diagnostics" / "ground_tracks.jsonl"):
        row = json.loads(line)
        fi = int(row["frame_index"])
        for t in row.get("tracks", []):
            if t.get("ground_xy") and t.get("global_player_id"):
                positions[fi][t["global_player_id"]] = np.asarray(t["ground_xy"], float)
    occupancy: dict[str, set[tuple[str, int]]] = defaultdict(set)
    for pred in sorted((p4_dir / "predictions").glob("*.jsonl")):
        for line in open(pred):
            row = json.loads(line)
            fi = int(row["frame_index"])
            cam = row["camera_id"]
            for player in row.get("players", []):
                pid = player.get("global_player_id")
                if pid:
                    occupancy[pid].add((cam, fi))
    return positions, occupancy


def find_events(positions, occupancy, radius: float, min_frames: int):
    close_frames: dict[tuple[str, str], list[int]] = defaultdict(list)
    for fi, by_id in positions.items():
        ids = sorted(by_id)
        for a, b in combinations(ids, 2):
            if np.linalg.norm(by_id[a] - by_id[b]) <= radius:
                close_frames[(a, b)].append(fi)
    events = []
    for (a, b), frames in close_frames.items():
        if len(frames) < min_frames:
            continue
        frames = sorted(frames)
        cams_a = {c for c, f in occupancy[a] if frames[0] <= f <= frames[-1]}
        cams_b = {c for c, f in occupancy[b] if frames[0] <= f <= frames[-1]}
        shared_camframe = bool(
            {(c, f) for c, f in occupancy[a] if frames[0] <= f <= frames[-1]}
            & {(c, f) for c, f in occupancy[b] if frames[0] <= f <= frames[-1]}
        )
        events.append({
            "ids": [a, b],
            "n_close_frames": len(frames),
            "span": [frames[0], frames[-1]],
            "cameras": {a: sorted(cams_a), b: sorted(cams_b)},
            "class": "shared-camera" if shared_camframe else "disjoint-cameras",
        })
    return sorted(events, key=lambda e: -e["n_close_frames"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--radius", type=float, default=0.75)
    ap.add_argument("--min-frames", type=int, default=25)
    ap.add_argument("--json-out", default=None, help="Write the full event list here")
    args = ap.parse_args()
    tree = ROOT / args.tree if not Path(args.tree).is_absolute() else Path(args.tree)
    all_events = {}
    print(f"{'delivery':>22} | {'disjoint(mergeable)':>20} | {'shared(2 people)':>17} | worst events")
    for ddir in sorted((tree / "deliveries").iterdir()):
        p4 = ddir / "p4"
        if not (p4 / "diagnostics" / "ground_tracks.jsonl").exists():
            continue
        positions, occupancy = load_tracks(p4)
        events = find_events(positions, occupancy, args.radius, args.min_frames)
        disjoint = [e for e in events if e["class"] == "disjoint-cameras"]
        shared = [e for e in events if e["class"] == "shared-camera"]
        worst = "; ".join(
            f"{e['ids'][0]}+{e['ids'][1]}({e['n_close_frames']}f,{e['class'][:4]})"
            for e in events[:3]
        )
        print(f"{ddir.name:>22} | {len(disjoint):>20} | {len(shared):>17} | {worst}")
        all_events[ddir.name] = events
        out = p4 / "diagnostics" / "swap_events.json"
        out.write_text(json.dumps({"radius_m": args.radius, "min_frames": args.min_frames,
                                   "events": events}, indent=2))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(all_events, indent=2))


if __name__ == "__main__":
    main()

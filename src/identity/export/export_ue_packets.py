#!/usr/bin/env python3
"""Convert triangulated 3D JSONL rows into UE-ready pose packets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from core.ue_transform import build_pose_packet, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True,
                        help="Canonical stage run dir with predictions/*.jsonl carrying "
                             "global_player_id + pose_3d. Prefer 07_refine (physics-constrained, "
                             "smoothed 3D); fall back to 06_roles if refinement was disabled.")
    parser.add_argument("--output", required=True, help="UE packet JSONL")
    parser.add_argument("--model-version", required=True)
    return parser.parse_args()


def iter_identified_player_frames(run_dir: Path):
    """Yield ``(record, player)`` once per identified player per frame.

    The same triangulated 3D is stamped on every camera's record, so packets are
    deduplicated on ``(frame_index, global_player_id)`` - one packet per player per
    frame, not per camera.
    """
    seen: set[tuple[int, str]] = set()
    for path in sorted((run_dir / "predictions").glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                for player in record.get("players", []):
                    gid = player.get("global_player_id")
                    if not gid or not player.get("pose_3d"):
                        continue
                    key = (int(record["frame_index"]), str(gid))
                    if key in seen:
                        continue
                    seen.add(key)
                    yield record, player


def main() -> int:
    args = parse_args()
    packets: list[dict] = []
    for record, player in iter_identified_player_frames(Path(args.run_dir)):
        pose = player["pose_3d"]
        # keypoints_world_m is per-joint nullable; null (un-triangulated) -> NaN, which
        # build_pose_packet serialises back to null in both world-m and UE-cm.
        points = np.array(
            [[float("nan")] * 3 if p is None else p for p in pose["keypoints_world_m"]],
            dtype=float,
        )
        confidence = pose.get("confidence") or [1.0] * len(points)
        packet = build_pose_packet(
            frame_id=str(record["frame_index"]),
            timestamp_ns=0,
            player_id=str(player["global_player_id"]),
            model_version=args.model_version,
            calibration_id=record.get("match_id"),
            keypoints3d_world_m=points,
            confidence=confidence,
        ).to_dict()
        # Carry the self-describing named + root-relative view through unchanged.
        packet["pose_3d_named"] = player.get("pose_3d_named")
        packet["role"] = player.get("role")
        packets.append(packet)
    write_jsonl(args.output, packets)
    print(f"Wrote {args.output} with {len(packets)} packets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


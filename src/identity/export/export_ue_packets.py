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
    parser.add_argument("--input", required=True, help="Triangulated JSONL")
    parser.add_argument("--output", required=True, help="UE packet JSONL")
    parser.add_argument("--model-version", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    packets = []
    with Path(args.input).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            keypoints = np.asarray(row["keypoints3d_world_m"], dtype=float)
            points = keypoints[:, :3]
            confidence = keypoints[:, 3] if keypoints.shape[1] > 3 else np.ones(points.shape[0])
            packet = build_pose_packet(
                frame_id=row["frame_id"],
                timestamp_ns=row.get("timestamp_ns", 0),
                player_id=row["player_id"],
                model_version=args.model_version,
                calibration_id=row.get("calibration_id"),
                keypoints3d_world_m=points,
                confidence=confidence,
                occlusion_tags=row.get("occlusion_tags", []),
            )
            packets.append(packet)
    write_jsonl(args.output, packets)
    print(f"Wrote {args.output} with {len(packets)} packets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


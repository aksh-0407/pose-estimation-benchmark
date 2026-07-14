"""CLI entry point for P5 role assignment.

Follows the same phase-folder convention as P2-P4: consumes a canonical P4 run,
writes ``roles.json`` + ``run_manifest.json`` into its own run dir. The mosaic
renderer (and any downstream consumer) reads roles ONLY from that artifact, so
improving the logic in :mod:`scripts.roles.assigner` never touches consumers.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.roles.assigner import assign_roles, assign_roles_epoched  # noqa: E402
from scripts.roles.config import load_role_assigner_config  # noqa: E402
from scripts.tracking.calibration import current_calibration_dir  # noqa: E402
from scripts.tracking.runner import infer_match_id  # noqa: E402
from scripts.visualization.mosaic_layout import (  # noqa: E402
    infer_bowling_direction,
    load_pitch_axis,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-run-dir", required=True, help="P4 run dir")
    parser.add_argument("--output-run-dir", required=True, help="P5 run dir to create")
    parser.add_argument("--drive-root", required=True)
    parser.add_argument("--delivery-id", required=True)
    parser.add_argument("--config", default=None, help="P5 YAML (role_assignment_version, etc.)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    input_run_dir = Path(args.input_run_dir)
    output_run_dir = Path(args.output_run_dir)
    drive_root = Path(args.drive_root)

    ground_tracks_path = input_run_dir / "diagnostics" / "ground_tracks.jsonl"
    if not ground_tracks_path.exists():
        raise FileNotFoundError(f"missing P4 artifact: {ground_tracks_path}")
    per_id_series: dict[str, list[tuple[int, np.ndarray]]] = {}
    with ground_tracks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            for track in row.get("tracks", []):
                player_id = track.get("global_player_id")
                xy = track.get("ground_xy")
                if player_id and xy and len(xy) >= 2:
                    per_id_series.setdefault(str(player_id), []).append(
                        (int(row["frame_index"]), np.asarray(xy, dtype=float))
                    )

    p5_config = load_role_assigner_config(args.config)

    if p5_config.role_assignment_version == "v1":
        manifest_path = input_run_dir / "run_manifest.json"
        p4_online_role_proxy = False
        if manifest_path.exists():
            p4_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            p4_online_role_proxy = bool(
                p4_manifest.get("config", {}).get("p4a", {}).get("online_role_proxy", False)
            )
        if not p4_online_role_proxy:
            raise ValueError(
                "role_assignment_version=v1 requires the input P4 run to have been produced "
                "with online_role_proxy=true (run_manifest.json config.p4a.online_role_proxy)"
            )

    match_id = infer_match_id(args.delivery_id)
    axis = load_pitch_axis(
        current_calibration_dir(drive_root, match_id) / "pitch_calibration_config.json"
    )
    direction = infer_bowling_direction(per_id_series, axis) if axis is not None else None

    if p5_config.role_assignment_version == "v1":
        roles = assign_roles_epoched(
            per_id_series, direction,
            frame_rate_fps=p5_config.frame_rate_fps,
            min_track_frames=p5_config.min_track_frames,
            epoch_frames=p5_config.epoch_frames,
            role_epoch_latch_count=p5_config.role_epoch_latch_count,
            role_assignment_max_cost=p5_config.role_assignment_max_cost,
        )
    else:
        roles = assign_roles(
            per_id_series, direction,
            frame_rate_fps=p5_config.frame_rate_fps,
            min_track_frames=p5_config.min_track_frames,
            bowler_min_speed_mps=p5_config.bowler_min_speed_mps,
            pitch_halfwidth_m=p5_config.pitch_halfwidth_m,
        )

    created_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "schema_version": "roles/v1",
        "created_at": created_at,
        "match_id": match_id,
        "delivery_id": args.delivery_id,
        "bowling_direction_xy": (
            [float(direction[0]), float(direction[1])] if direction is not None else None
        ),
        "roles": {player_id: roles[player_id].to_json() for player_id in sorted(roles)},
    }
    output_run_dir.mkdir(parents=True, exist_ok=True)
    roles_path = output_run_dir / "roles.json"
    roles_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "roles_run/v1",
        "created_at": created_at,
        "task": "role_assignment",
        "role_assignment_version": p5_config.role_assignment_version,
        "input_run_dir": str(input_run_dir),
        "output_run_dir": str(output_run_dir),
        "delivery_id": args.delivery_id,
        "match_id": match_id,
        "artifacts": {"roles_json": str(roles_path)},
    }
    (output_run_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    named = {p: r.role for p, r in roles.items() if r.role not in {"unknown", "fielder"}}
    print(f"P5: roles assigned for {len(roles)} ids  named={named}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

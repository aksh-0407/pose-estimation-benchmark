#!/usr/bin/env python3
"""P5b (Wave 6): role-aware suppression of low-confidence peripheral identities.

The four core roles (bowler, striker, non-striker, wicketkeeper) are NEVER
suppressed. Peripheral identities (umpires, fielders, unknowns) are suppressed
only when their track quality is clearly bad — the user directive: when
low-confidence poses/tracking of peripheral players hinder the output, drop
them rather than extrapolate. A well-tracked umpire stays.

Reads a P4 run dir (+ sibling P5 roles.json), writes ``suppression.json`` next
to roles.json:

    {"schema_version": "suppression/v1", "enabled": true,
     "suppressed": {"P009": {"reasons": ["kp_conf 0.28 < 0.35"], ...}}, ...}

Consumers (renderer, P6 export) skip suppressed ids. With ``enabled: false``
(default config) the file records an empty set, so downstream output is
byte-identical to a run without this stage.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

CORE_ROLES = ("bowler", "striker", "non_striker", "wicketkeeper")

DEFAULTS = {
    "suppression_enabled": False,
    "suppress_min_kp_conf": 0.35,        # mean pose keypoint confidence
    "suppress_min_completeness": 0.25,   # observed frames / delivery span
    "suppress_single_cam_det_conf": 0.40,  # single-cam tracks need this det conf
    "suppress_protect_umpires": False,   # true = umpires as protected as core roles
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-run-dir", required=True, help="P4 run dir")
    ap.add_argument("--roles-path", default=None,
                    help="roles.json (default: <input>/../p5/roles.json)")
    ap.add_argument("--output-path", default=None,
                    help="suppression.json (default: next to roles.json)")
    ap.add_argument("--config", default=None, help="YAML with the suppress_* keys")
    return ap.parse_args()


def load_config(path: str | None) -> dict:
    cfg = dict(DEFAULTS)
    if path:
        import yaml
        raw = yaml.safe_load(Path(path).read_text()) or {}
        for key in cfg:
            if key in raw:
                cfg[key] = raw[key]
    return cfg


def track_quality(run_dir: Path) -> dict[str, dict]:
    """Per-global-id quality aggregates from the P4 predictions."""

    stats: dict[str, dict] = defaultdict(lambda: {
        "kp_conf": [], "det_conf": [], "frames": set(),
        "single_cam_frames": 0, "obs": 0,
    })
    span_min, span_max = None, None
    for pred in sorted((run_dir / "predictions").glob("*.jsonl")):
        with open(pred) as fh:
            for line in fh:
                rec = json.loads(line)
                fi = int(rec["frame_index"])
                span_min = fi if span_min is None else min(span_min, fi)
                span_max = fi if span_max is None else max(span_max, fi)
                for player in rec.get("players", []):
                    pid = player.get("global_player_id")
                    if not pid:
                        continue
                    st = stats[pid]
                    st["obs"] += 1
                    st["frames"].add(fi)
                    st["single_cam_frames"] += int(bool(player.get("single_camera")))
                    conf = player.get("detection_confidence")
                    if conf is not None:
                        st["det_conf"].append(float(conf))
                    pose = player.get("pose_2d") or {}
                    kp = pose.get("confidence")
                    if kp:
                        st["kp_conf"].append(float(np.mean(kp)))
    span = max(1, (span_max - span_min + 1)) if span_min is not None else 1
    out = {}
    for pid, st in stats.items():
        out[pid] = {
            "mean_kp_conf": float(np.mean(st["kp_conf"])) if st["kp_conf"] else 0.0,
            "mean_det_conf": float(np.mean(st["det_conf"])) if st["det_conf"] else 0.0,
            "completeness": len(st["frames"]) / span,
            "single_cam_rate": st["single_cam_frames"] / max(1, st["obs"]),
            "observations": st["obs"],
        }
    return out


def decide(quality: dict[str, dict], roles: dict[str, str], cfg: dict) -> dict[str, dict]:
    suppressed: dict[str, dict] = {}
    if not cfg["suppression_enabled"]:
        return suppressed
    protected = set(CORE_ROLES)
    if cfg["suppress_protect_umpires"]:
        protected.add("umpire")
    for pid, q in quality.items():
        role = roles.get(pid, "unknown")
        if role in protected:
            continue
        reasons = []
        if q["mean_kp_conf"] < cfg["suppress_min_kp_conf"]:
            reasons.append(f"kp_conf {q['mean_kp_conf']:.2f} < {cfg['suppress_min_kp_conf']}")
        if q["completeness"] < cfg["suppress_min_completeness"]:
            reasons.append(f"completeness {q['completeness']:.2f} < {cfg['suppress_min_completeness']}")
        if q["single_cam_rate"] > 0.999 and q["mean_det_conf"] < cfg["suppress_single_cam_det_conf"]:
            reasons.append(
                f"single-cam det_conf {q['mean_det_conf']:.2f} < {cfg['suppress_single_cam_det_conf']}")
        if reasons:
            suppressed[pid] = {"role": role, "reasons": reasons, **{k: round(v, 3) for k, v in q.items()}}
    return suppressed


def main() -> None:
    args = parse_args()
    run_dir = Path(args.input_run_dir)
    roles_path = Path(args.roles_path) if args.roles_path else run_dir.parent / "p5" / "roles.json"
    cfg = load_config(args.config)
    roles = {}
    if roles_path.exists():
        payload = json.loads(roles_path.read_text())
        roles = {pid: (entry or {}).get("role", "unknown")
                 for pid, entry in (payload.get("roles") or {}).items()}
    quality = track_quality(run_dir)
    suppressed = decide(quality, roles, cfg)
    out_path = Path(args.output_path) if args.output_path else roles_path.parent / "suppression.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "schema_version": "suppression/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "enabled": bool(cfg["suppression_enabled"]),
        "config": cfg,
        "track_count": len(quality),
        "suppressed": suppressed,
    }, indent=2))
    kept_named = {pid: roles.get(pid) for pid in quality if pid not in suppressed
                  and roles.get(pid) in CORE_ROLES}
    print(f"P5b: {len(suppressed)}/{len(quality)} ids suppressed "
          f"({sorted(suppressed)}); core kept: {sorted(kept_named)}")


if __name__ == "__main__":
    main()

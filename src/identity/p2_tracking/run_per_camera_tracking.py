"""CLI entry point for per-camera player tracking."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from identity.p2_tracking.config import load_tracking_config  # noqa: E402
from identity.p2_tracking.runner import run_tracking  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Per-camera player tracking")
    parser.add_argument("--input-run-dir", required=True, help="input run dir containing predictions/*.jsonl")
    parser.add_argument("--output-run-dir", required=True, help="tracking run dir to write predictions, diagnostics, and metrics")
    parser.add_argument("--drive-root", required=True, help="repo drive root containing dataset/calibration-data")
    parser.add_argument("--delivery-id", required=True)
    parser.add_argument("--config", default=None, help="optional tracking YAML config")
    parser.add_argument("--camera", action="append", default=None, help="restrict to canonical camera id(s), e.g. cam_01")
    parser.add_argument("--expected-frames", type=int, default=600)
    parser.add_argument("--max-workers", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = load_tracking_config(args.config)
    results = run_tracking(
        input_run_dir=args.input_run_dir,
        output_run_dir=args.output_run_dir,
        drive_root=args.drive_root,
        delivery_id=args.delivery_id,
        config=config,
        cameras=args.camera,
        expected_frames=args.expected_frames,
        max_workers=args.max_workers,
    )
    failed = []
    for cam in sorted(results):
        status, summary, error = results[cam]
        if status == "ok":
            print(f"{cam}: ok  frames={summary.get('frames_read')} "
                  f"confirmed={summary.get('confirmed_tracks')}", flush=True)
        else:
            failed.append(cam)
            print(f"{cam}: FAILED  {error}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

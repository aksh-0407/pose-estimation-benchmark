"""CLI entry point for P3 cross-camera geometric association."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.association.config import load_association_config  # noqa: E402
from scripts.association.runner import run_association  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-run-dir", required=True)
    parser.add_argument("--output-run-dir", required=True)
    parser.add_argument("--drive-root", required=True)
    parser.add_argument("--delivery-id", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--camera", action="append", default=None)
    parser.add_argument("--expected-frames", type=int, default=600)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    metrics = run_association(
        input_run_dir=args.input_run_dir,
        output_run_dir=args.output_run_dir,
        drive_root=args.drive_root,
        delivery_id=args.delivery_id,
        config=load_association_config(args.config),
        cameras=args.camera,
        expected_frames=args.expected_frames,
    )
    print(
        f"P3: {metrics['status']}  frames={metrics['frames_processed']} "
        f"clusters={metrics['cluster_count']} single_camera_rate={metrics['single_camera_rate']:.3f}",
        flush=True,
    )
    return 0 if metrics["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

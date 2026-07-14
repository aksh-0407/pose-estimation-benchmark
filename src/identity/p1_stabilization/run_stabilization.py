"""CLI entry point for P1.5 — 2D keypoint temporal stabilization.

Sits between P1 (2D inference) and P2 (per-camera tracking):

    python -m identity.p1_stabilization.run_stabilization \
        --input-run-dir <p1-run> --output-run-dir <p1b-run> \
        --delivery-id CCPL080626M1_1_14_1 --config configs/p1b_stabilization.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from identity.p1_stabilization.config import load_stabilization_config  # noqa: E402
from identity.p1_stabilization.runner import run_stabilization  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P1.5 2D keypoint temporal stabilization")
    parser.add_argument("--input-run-dir", required=True, help="P1 run dir containing predictions/*.jsonl")
    parser.add_argument("--output-run-dir", required=True, help="stabilized run dir to write")
    parser.add_argument("--delivery-id", required=True)
    parser.add_argument("--config", default=None, help="optional stabilization YAML (default: configs/p1b_stabilization.yaml)")
    parser.add_argument("--camera", action="append", default=None, help="restrict to camera id(s), e.g. cam_01")
    # --drive-root/--expected-frames accepted for a uniform stage CLI; unused here.
    parser.add_argument("--drive-root", default=None, help="(accepted for CLI uniformity; unused)")
    parser.add_argument("--expected-frames", type=int, default=600, help="(accepted for CLI uniformity; unused)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = load_stabilization_config(args.config)
    metrics = run_stabilization(
        input_run_dir=args.input_run_dir,
        output_run_dir=args.output_run_dir,
        delivery_id=args.delivery_id,
        config=config,
        cameras=args.camera,
    )
    print(f"stabilization: enabled={metrics['enabled']} "
          f"jitter_px {metrics['mean_jitter_px_before']:.3f} -> {metrics['mean_jitter_px_after']:.3f} "
          f"over {len(metrics['per_camera'])} camera(s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

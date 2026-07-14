#!/usr/bin/env python3
"""CPU shootout: local laptop vs remote L40S box, with pipeline-shaped workloads.

Workloads mirror the CPU-bound stages:
  kalman  - many tiny 4x4/2x2 matrix ops (P2/P4 Kalman predict-update loops)
  lstsq   - robust lstsq on stacked 2-view systems (P3 ground solve / 04 (binding lift) DLT lift)
  json    - parse+serialize realistic P1 player records (jsonl I/O everywhere)
Modes: single-thread score, then all-cores aggregate via multiprocessing
(matches our BLAS-capped one-thread-per-process fan-out across deliveries).
"""
import json
import multiprocessing as mp
import os
import platform
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402


def w_kalman(seconds: float) -> float:
    rng = np.random.default_rng(0)
    F = np.eye(4) + 0.02 * rng.standard_normal((4, 4))
    H = rng.standard_normal((2, 4))
    Q = np.eye(4) * 0.01
    R = np.eye(2) * 0.05
    x = rng.standard_normal(4)
    P = np.eye(4)
    n = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        for _ in range(200):
            x = F @ x
            P = F @ P @ F.T + Q
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            z = H @ x + 0.1
            x = x + K @ (z - H @ x)
            P = (np.eye(4) - K @ H) @ P
            n += 1
    return n / (time.perf_counter() - t0)


def w_lstsq(seconds: float) -> float:
    rng = np.random.default_rng(1)
    n = 0
    t0 = time.perf_counter()
    A = rng.standard_normal((7 * 26, 4))  # 7-cam, 26-joint DLT-ish stack
    while time.perf_counter() - t0 < seconds:
        for _ in range(50):
            b = rng.standard_normal(A.shape[0])
            np.linalg.lstsq(A, b, rcond=None)
            n += 1
    return n / (time.perf_counter() - t0)


def w_json(seconds: float) -> float:
    rec = {
        "frame_index": 123, "camera_id": "cam_04",
        "players": [
            {"bbox_xywh_px": [512.3, 233.1, 88.0, 240.5],
             "detection_confidence": 0.87, "local_track_id": i,
             "pose_2d_native": [[float(j), float(j) * 2.0, 0.9] for j in range(26)]}
            for i in range(8)
        ],
    }
    s = json.dumps(rec)
    n = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        for _ in range(200):
            json.loads(s)
            json.dumps(rec)
            n += 1
    return n / (time.perf_counter() - t0)


WORK = {"kalman": w_kalman, "lstsq": w_lstsq, "json": w_json}


def _worker(args):
    name, seconds = args
    return WORK[name](seconds)


def main() -> None:
    ncpu = mp.cpu_count()
    out = {"host": platform.node(), "ncpu": ncpu, "numpy": np.__version__}
    for name in WORK:
        out[f"{name}_1t"] = round(WORK[name](6.0), 1)
    # all-cores sustained (long enough to hit thermal steady-state on a laptop)
    with mp.Pool(ncpu) as pool:
        for name in WORK:
            rates = pool.map(_worker, [(name, 20.0)] * ncpu)
            out[f"{name}_all({ncpu})"] = round(sum(rates), 1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

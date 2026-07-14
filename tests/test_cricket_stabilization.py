"""Tests for the 01 (stabilization) 2D stabilization stage."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.contract import KEYPOINT_COUNT, SCHEMA_VERSION, SKELETON, validate_group1_frame
from identity.p1_stabilization.config import (
    GatingConfig,
    LinkConfig,
    SmoothingConfig,
    StabilizationConfig,
    load_stabilization_config,
)
from identity.p1_stabilization.linker import link_micro_tracks
from identity.p1_stabilization.runner import run_stabilization
from identity.p1_stabilization.smoothing import OneEuroFilter, mean_jitter_px, smooth_track_keypoints


# --------------------------------------------------------------------------- config
def test_config_rejects_unknown_key(tmp_path: Path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("enabled: true\nnot_a_key: 3\n")
    with pytest.raises(ValueError, match="unknown key"):
        load_stabilization_config(cfg)


def test_config_rejects_unknown_method(tmp_path: Path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("smoothing:\n  method: kalman\n")
    with pytest.raises(ValueError, match="unsupported smoothing.method"):
        load_stabilization_config(cfg)


# --------------------------------------------------------------------------- one-euro
def test_one_euro_reduces_static_jitter():
    rng = np.random.default_rng(0)
    truth = 100.0
    noisy = truth + rng.normal(0, 3.0, size=200)
    f = OneEuroFilter(min_cutoff=0.5, beta=0.0, d_cutoff=1.0)
    filtered = np.array([f.filter(float(v), dt=1 / 50.0) for v in noisy])
    # jitter (frame-to-frame std) must drop sharply on a static-but-noisy signal.
    assert np.std(np.diff(filtered)) < 0.5 * np.std(np.diff(noisy))


# --------------------------------------------------------------------------- linker
def test_linker_links_overlapping_and_separates_distant():
    # Two frames. Box A overlaps across frames -> one track; box B far away -> its own track.
    frames = [
        [(0, (100.0, 100.0, 50.0, 120.0)), (1, (900.0, 100.0, 50.0, 120.0))],
        [(0, (105.0, 102.0, 50.0, 120.0))],
    ]
    tracks = link_micro_tracks(frames, LinkConfig(iou_min=0.3, max_gap_frames=2))
    lengths = sorted(len(t) for t in tracks)
    assert lengths == [1, 2]  # the moving box linked (2), the far box alone (1)


# --------------------------------------------------------------------------- smoothing
def test_smoothing_reduces_jitter_and_preserves_placeholders():
    rng = np.random.default_rng(1)
    T, K = 40, 3
    base = np.zeros((T, K, 2))
    # keypoint 0: a smooth ramp + noise; keypoint 1: static + noise; keypoint 2: (0,0) placeholder
    ramp = np.linspace(200, 260, T)
    base[:, 0, 0] = ramp + rng.normal(0, 4, T)
    base[:, 0, 1] = 300 + rng.normal(0, 4, T)
    base[:, 1, 0] = 500 + rng.normal(0, 4, T)
    base[:, 1, 1] = 500 + rng.normal(0, 4, T)
    # keypoint 2 stays (0,0)
    conf = np.full((T, K), 0.9)
    conf[:, 2] = 0.0
    bbox = np.full(T, 250.0)
    dt = np.full(T, 1 / 50.0)

    before = mean_jitter_px(base, conf, 0.3)
    smoothed = smooth_track_keypoints(base, conf, bbox, dt,
                                      SmoothingConfig(min_cutoff=0.8, beta=0.2, d_cutoff=1.0),
                                      GatingConfig())
    after = mean_jitter_px(smoothed, conf, 0.3)
    assert after < before
    # placeholder keypoint untouched
    assert np.allclose(smoothed[:, 2, :], 0.0)


def test_spike_clamp_rejects_low_confidence_outlier():
    T, K = 10, 1
    kp = np.zeros((T, K, 2))
    kp[:, 0, 0] = 200.0
    kp[:, 0, 1] = 200.0
    kp[5, 0, 0] = 900.0  # a 700px spike
    conf = np.full((T, K), 0.9)
    conf[5, 0] = 0.1  # ...on a low-confidence frame -> should be clamped
    bbox = np.full(T, 100.0)
    dt = np.full(T, 1 / 50.0)
    smoothed = smooth_track_keypoints(kp, conf, bbox, dt,
                                      SmoothingConfig(min_cutoff=1.0, beta=0.0, d_cutoff=1.0),
                                      GatingConfig(confidence_min=0.3, max_jump_px=120.0))
    assert smoothed[5, 0, 0] < 400.0  # the spike was suppressed, not tracked to 900


# --------------------------------------------------------------------------- end-to-end
def _p1_record(frame_index: int, cx: float, cy: float, jitter: float) -> dict:
    kpts = [[cx + i + jitter, cy + i - jitter] for i in range(KEYPOINT_COUNT)]
    norm = [[x / 2560.0, y / 1440.0] for x, y in kpts]
    return {
        "schema_version": SCHEMA_VERSION,
        "match_id": "CCPL080626",
        "delivery_id": "CCPL080626M1_1_14_1",
        "camera_id": "cam_01",
        "frame_index": frame_index,
        "frame_name": f"frame_camera01_{frame_index:09d}.jpg",
        "metadata": {"image_size_px": [2560, 1440]},
        "players": [{
            "global_player_id": None,
            "local_track_id": None,
            "role": "unknown",
            "detection_confidence": 0.9,
            "track_confidence": None,
            "bbox_xywh_px": [cx - 40, cy - 120, 80.0, 240.0],
            "bbox_xywh_norm": [(cx - 40) / 2560.0, (cy - 120) / 1440.0, 80.0 / 2560.0, 240.0 / 1440.0],
            "pose_2d": {"skeleton": SKELETON, "keypoints_px": kpts, "keypoints_norm": norm,
                        "confidence": [0.9] * KEYPOINT_COUNT},
            "pose_3d": None,
        }],
    }


def test_runner_end_to_end(tmp_path: Path):
    pred_dir = tmp_path / "p1" / "predictions"
    pred_dir.mkdir(parents=True)
    rng = np.random.default_rng(2)
    with (pred_dir / "bt_01__CCPL080626M1_1_14_1__cam_01.jsonl").open("w") as f:
        for t in range(30):
            jitter = float(rng.normal(0, 5))
            f.write(json.dumps(_p1_record(1000 + t, 500.0 + t * 2, 700.0, jitter)) + "\n")

    out_dir = tmp_path / "p1b"
    metrics = run_stabilization(tmp_path / "p1", out_dir, "CCPL080626M1_1_14_1", StabilizationConfig())

    # smoothing reduced jitter
    assert metrics["mean_jitter_px_after"] < metrics["mean_jitter_px_before"]
    # output is a valid, drop-in P2 input: validates, and local_track_id stays null
    out_file = out_dir / "predictions" / "bt_01__CCPL080626M1_1_14_1__cam_01.jsonl"
    assert out_file.exists()
    lines = out_file.read_text().strip().splitlines()
    assert len(lines) == 30
    for line in lines:
        rec = json.loads(line)
        validate_group1_frame(rec)
        assert rec["players"][0]["local_track_id"] is None
    assert (out_dir / "run_manifest.json").exists()
    assert (out_dir / "stabilization_metrics.json").exists()


def test_runner_disabled_is_passthrough(tmp_path: Path):
    pred_dir = tmp_path / "p1" / "predictions"
    pred_dir.mkdir(parents=True)
    rec = _p1_record(1000, 500.0, 700.0, 5.0)
    (pred_dir / "bt_01__CCPL080626M1_1_14_1__cam_01.jsonl").write_text(json.dumps(rec) + "\n")

    out_dir = tmp_path / "p1b"
    cfg = StabilizationConfig(enabled=False)
    run_stabilization(tmp_path / "p1", out_dir, "CCPL080626M1_1_14_1", cfg)
    out = json.loads((out_dir / "predictions" / "bt_01__CCPL080626M1_1_14_1__cam_01.jsonl").read_text())
    assert out["players"][0]["pose_2d"]["keypoints_px"] == rec["players"][0]["pose_2d"]["keypoints_px"]


def test_runner_disabled_is_byte_identical_for_any_key_order(tmp_path: Path):
    # Regression (Wave 0): the writer must not re-order keys — a P1 producer that
    # writes insertion-ordered JSON (e.g. the rtmpose-x run) must round-trip
    # byte-for-byte through the disabled stage, or every flags-off A/B breaks.
    pred_dir = tmp_path / "p1" / "predictions"
    pred_dir.mkdir(parents=True)
    rec = _p1_record(1000, 500.0, 700.0, 5.0)
    scrambled = dict(reversed(list(rec.items())))          # deliberately unsorted
    raw = json.dumps(scrambled) + "\n"
    src = pred_dir / "bt_01__CCPL080626M1_1_14_1__cam_01.jsonl"
    src.write_text(raw)

    out_dir = tmp_path / "p1b"
    run_stabilization(tmp_path / "p1", out_dir, "CCPL080626M1_1_14_1",
                      StabilizationConfig(enabled=False))
    out = (out_dir / "predictions" / src.name).read_text()
    assert out == raw

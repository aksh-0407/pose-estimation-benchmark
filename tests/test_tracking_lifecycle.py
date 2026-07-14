"""P2 tracker-core lifecycle tests (FR batch: C1/C2 guardrails)."""
from __future__ import annotations

import numpy as np

from identity.p2_tracking.kalman import KalmanBoxTracker
from identity.p2_tracking.config import TrackingConfig
from identity.p2_tracking.pose_vector import PoseVector
from identity.p2_tracking.tracker import CameraTracker, Detection


def _det(x, y, w=40.0, h=100.0, conf=0.9):
    pose = PoseVector(
        vector=np.zeros(34), mask=np.zeros(17, dtype=bool),
        confidence=np.zeros(17), defined=False,
    )
    return Detection(bbox_xywh=[x, y, w, h], pose=pose, confidence=conf, player={})


def test_process_noise_resets_on_reacquisition():
    k = KalmanBoxTracker([0, 0, 40, 100])
    for _ in range(10):
        k.predict()
        k.inflate_process_noise(1.5)
    assert k._q > 50           # inflated during the gap
    k.update([5, 0, 40, 100])
    assert k._q == 1.0         # C2: nominal noise restored on re-acquisition
    trace_after = k.position_cov_trace()
    for _ in range(5):
        k.predict()
        k.update([5, 0, 40, 100])
    assert k.position_cov_trace() <= trace_after  # covariance recovers, not diverges


def test_zero_iou_fast_mover_stays_matched_stage2():
    # C1: a low-confidence detection displaced beyond its own bbox (zero IoU with
    # the prediction) but within the reachability gate must still match in the
    # IoU-only stage instead of fragmenting the track.
    tracker = CameraTracker("cam_01", TrackingConfig())
    x = 100.0
    for frame in range(6):
        tracker.update([_det(x, 500.0)], frame_index=frame)
        x += 10.0
    (track,) = tracker.tracks
    hits_before = track.hits
    # jump 1.2x bbox width in one frame, low confidence -> stage 2 (IoU-only)
    jump = _det(x + 48.0, 500.0, conf=0.3)
    tracker.update([jump], frame_index=6)
    (same,) = tracker.tracks                 # C1: no duplicate track spawned
    assert same.hits == hits_before + 1      # and the jump was MATCHED

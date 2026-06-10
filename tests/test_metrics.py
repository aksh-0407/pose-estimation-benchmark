import numpy as np

from pose_estimation.metrics import mpjpe, p_mpjpe, pck, weighted_model_score


def test_pck_counts_visible_correct_keypoints():
    pred = np.array([[0.0, 0.0], [10.0, 0.0], [50.0, 50.0]])
    gt = np.array([[0.0, 0.0], [12.0, 0.0], [0.0, 0.0]])
    visibility = np.array([1, 1, 0])
    assert pck(pred, gt, threshold=0.25, norm=10.0, visibility=visibility) == 1.0


def test_mpjpe_and_p_mpjpe():
    gt = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    pred = gt + np.array([10.0, -5.0, 3.0])
    assert mpjpe(pred, gt) > 0.0
    assert p_mpjpe(pred, gt) < 1e-9


def test_weighted_score_prefers_fast_accurate_model():
    score = weighted_model_score(
        cricket_2d_accuracy=90,
        occlusion_robustness=80,
        latency_p95_ms=50,
        jitter_score=85,
        integration_effort=90,
    )
    assert score > 80.0


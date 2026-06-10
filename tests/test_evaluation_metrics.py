import numpy as np

from pose_estimation.evaluation import acceleration_error, evaluate_2d_predictions, pck3d


def test_evaluate_2d_predictions_reports_pck_and_error():
    predicted = np.array([[[0.0, 0.0], [10.0, 10.0]]])
    target = np.array([[[0.0, 0.0], [13.0, 14.0]]])
    visibility = np.array([[1, 1]])

    metrics = evaluate_2d_predictions(predicted, target, visibility=visibility, norm=10.0)

    assert metrics["pck@0.05"] == 0.5
    assert metrics["mean_pixel_error"] == 2.5
    assert metrics["detection_rate"] == 1.0


def test_3d_pck_and_acceleration_error():
    predicted = np.zeros((4, 2, 3))
    target = np.zeros((4, 2, 3))
    target[0, 0, 0] = 100.0

    assert pck3d(predicted[0], target[0], threshold=150.0) == 1.0
    assert acceleration_error(predicted, predicted) == 0.0


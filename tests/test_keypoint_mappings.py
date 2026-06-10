import numpy as np

from pose_estimation.keypoints import map_keypoints


def test_wholebody_133_maps_first_17_to_coco17():
    keypoints = np.column_stack([np.arange(133), np.arange(133) + 0.5, np.ones(133)])

    mapped = map_keypoints(keypoints, "coco_wholebody_133")

    assert len(mapped) == 17
    assert mapped[0] == [0.0, 0.5, 1.0]
    assert mapped[16] == [16.0, 16.5, 1.0]


def test_mediapipe_33_maps_semantic_indices_to_coco17():
    keypoints = np.column_stack([np.arange(33), np.arange(33) + 10, np.ones(33)])

    mapped = map_keypoints(keypoints, "mediapipe_33")

    assert mapped[0] == [0.0, 10.0, 1.0]
    assert mapped[1] == [2.0, 12.0, 1.0]
    assert mapped[2] == [5.0, 15.0, 1.0]
    assert mapped[16] == [28.0, 38.0, 1.0]


def test_openpose_body25_maps_body25_to_coco17():
    keypoints = np.column_stack([np.arange(25), np.arange(25) + 20, np.ones(25)])

    mapped = map_keypoints(keypoints, "openpose_body25")

    assert mapped[0] == [0.0, 20.0, 1.0]
    assert mapped[1] == [15.0, 35.0, 1.0]
    assert mapped[5] == [5.0, 25.0, 1.0]
    assert mapped[16] == [11.0, 31.0, 1.0]


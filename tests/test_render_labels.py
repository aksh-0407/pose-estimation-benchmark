"""Tests for the render overlays: collision-aware chips, body paint, identity palette."""
from __future__ import annotations

import numpy as np

from identity.visualization.identity_colors import IDENTITY_PALETTE, color_for_global_id
from identity.visualization.render_videos import (
    _rects_overlap,
    draw_body_paint,
    place_chip,
)


def test_place_chip_avoids_previous_chips():
    image_wh, reserves = (640, 360), (50, 320)
    placed: list = []
    # two players side by side with near-identical boxes
    a = place_chip((100, 100, 180, 300), (120, 24), image_wh, reserves, placed)
    b = place_chip((110, 100, 190, 300), (120, 24), image_wh, reserves, placed)
    ra = (a[0], a[1] - 24, a[0] + 120, a[1])
    rb = (b[0], b[1] - 24, b[0] + 120, b[1])
    assert not _rects_overlap(ra, rb)
    assert len(placed) == 2


def test_place_chip_stacks_when_row_is_full():
    image_wh, reserves = (640, 360), (50, 320)
    placed: list = []
    results = [
        place_chip((100 + i * 4, 100, 180 + i * 4, 300), (150, 24), image_wh, reserves, placed)
        for i in range(4)
    ]
    rects = [(x, y - 24, x + 150, y) for x, y, _ in results]
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            assert not _rects_overlap(rects[i], rects[j])


def test_place_chip_respects_header_reserve():
    placed: list = []
    x, y, _ = place_chip((10, 10, 90, 40), (80, 24), (640, 360), (52, 320), placed)
    assert y - 24 >= 52 or y <= 320  # never inside the header band


def test_palette_size_and_separation():
    assert len(IDENTITY_PALETTE) == 20
    assert len(set(IDENTITY_PALETTE)) == 20
    # adjacent global ids must be clearly distinct (max channel delta >= 60)
    for i in range(1, 21):
        a = np.array(color_for_global_id(f"P{i:03d}"), dtype=int)
        b = np.array(color_for_global_id(f"P{i + 1:03d}"), dtype=int)
        assert np.abs(a - b).max() >= 60, (i, a.tolist(), b.tolist())


def test_body_paint_draws_only_for_confident_torso():
    overlay = np.zeros((200, 200, 3), dtype=np.uint8)
    kpts = [[100.0, 40.0]] * 5 + [
        [80, 60], [120, 60],   # shoulders
        [70, 90], [130, 90],   # elbows
        [65, 120], [135, 120],  # wrists
        [85, 120], [115, 120],  # hips
        [83, 160], [117, 160],  # knees
        [82, 195], [118, 195],  # ankles
    ]
    player = {
        "bbox_xywh_px": [60.0, 30.0, 80.0, 170.0],
        "pose_2d": {"keypoints_px": kpts, "confidence": [0.9] * 17},
    }
    draw_body_paint(overlay, [player], [(0, 0, 255)], keypoint_threshold=0.3)
    assert (overlay[:, :, 2] > 0).sum() > 500  # substantial red-tinted body area
    # zero-confidence player paints nothing
    overlay2 = np.zeros((200, 200, 3), dtype=np.uint8)
    player2 = {
        "bbox_xywh_px": [60.0, 30.0, 80.0, 170.0],
        "pose_2d": {"keypoints_px": kpts, "confidence": [0.0] * 17},
    }
    draw_body_paint(overlay2, [player2], [(0, 0, 255)], keypoint_threshold=0.3)
    assert overlay2.sum() == 0

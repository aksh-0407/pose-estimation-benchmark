"""Tests for the calibration-derived mosaic layout."""

from __future__ import annotations

import json

import numpy as np

from identity.visualization.mosaic_layout import (
    MONITOR_SLOT,
    ROSTER_SLOT,
    derive_mosaic_layout,
    infer_bowling_direction,
    load_pitch_axis,
)


def _camera(center, target, f: float = 1500.0) -> np.ndarray:
    center = np.asarray(center, dtype=float)
    forward = np.asarray(target, dtype=float) - center
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.stack([right, down, forward])
    K = np.array([[f, 0.0, 960.0], [0.0, f, 540.0], [0.0, 0.0, 1.0]])
    return K @ np.concatenate([R, (-R @ center).reshape(3, 1)], axis=1)


# Synthetic rig mimicking the real one: pitch along Y, end-on pair behind the
# ends, one side pair across the pitch, one unpaired oblique pano.
RIG = {
    "cam_a_end": _camera([0.0, -100.0, 14.0], [0.0, 5.0, 0.0]),   # behind y<0 end
    "cam_b_end": _camera([0.0, 105.0, 15.0], [0.0, -5.0, 0.0]),   # behind y>0 end
    "cam_c_side": _camera([110.0, -5.0, 12.0], [-5.0, -5.0, 0.0]),
    "cam_d_side": _camera([-115.0, 5.0, 12.0], [5.0, 5.0, 0.0]),
    "cam_e_pano": _camera([-80.0, -60.0, 10.0], [8.0, 5.0, 0.0]),
}


def test_load_pitch_axis_prefers_stump_bases(tmp_path):
    payload = {
        "title": "x",
        "fsmb": [0.0, 10.08, 0.0],
        "nsmb": [0.0, -10.08, 0.0],
        "junk": [5.0, 5.0],
    }
    path = tmp_path / "pitch.json"
    path.write_text(json.dumps(payload))
    axis = load_pitch_axis(path)
    assert axis is not None and abs(abs(axis[1]) - 1.0) < 1e-6

    # Fallback: full-pitch cloud -> major axis IS the pitch axis.
    cloud = {f"p{i}": [float(np.random.default_rng(i).normal(0, 0.5)), float(y)]
             for i, y in enumerate(np.linspace(-10, 10, 12))}
    path2 = tmp_path / "pitch2.json"
    path2.write_text(json.dumps(cloud))
    axis2 = load_pitch_axis(path2)
    assert axis2 is not None and abs(axis2[1]) > 0.95

    assert load_pitch_axis(tmp_path / "missing.json") is None


def test_infer_bowling_direction_uses_early_axis_motion():
    axis = np.array([0.0, 1.0])
    # Bowler sprints toward -y early; a batsman sprints +y late (must not win);
    # something drifts fast along +x (must be ignored - off axis).
    series = {
        "bowler": [(i, np.array([0.0, 30.0 - 0.16 * i])) for i in range(0, 120)],
        "batsman": [(i, np.array([1.0, -8.0 + 0.18 * (i - 450)])) for i in range(450, 560)],
        "cross": [(i, np.array([0.3 * i, 5.0])) for i in range(0, 120)],
    }
    direction = infer_bowling_direction(series, axis)
    assert direction is not None
    assert direction[1] < -0.99  # toward -y, the bowler's run

    static = {"a": [(i, np.array([0.0, 0.0])) for i in range(200)]}
    assert infer_bowling_direction(static, axis) is None


def test_derive_mosaic_layout_semantics_and_flip():
    layout = derive_mosaic_layout(RIG, bowling_direction_xy=np.array([0.0, -1.0]))
    top, mid, bottom = layout.grid
    # End-on pair in the first column, the camera looking WITH the delivery on top.
    assert top[0] == "cam_b_end" and mid[0] == "cam_a_end"
    # The side pair fills another column, one of them mirrored and on top.
    side_column = 1 if top[1] in {"cam_c_side", "cam_d_side"} else 2
    assert {top[side_column], mid[side_column]} == {"cam_c_side", "cam_d_side"}
    assert top[side_column] in layout.mirrored
    assert mid[side_column] not in layout.mirrored
    # Pano bottom-middle, flanked by the panels.
    assert bottom == (MONITOR_SLOT, "cam_e_pano", ROSTER_SLOT)
    # End-on cameras never mirror.
    assert "cam_a_end" not in layout.mirrored and "cam_b_end" not in layout.mirrored

    # Flip the bowling end: rows swap and the mirrored side swaps.
    flipped = derive_mosaic_layout(RIG, bowling_direction_xy=np.array([0.0, 1.0]))
    f_top, f_mid, f_bottom = flipped.grid
    assert f_top[0] == "cam_a_end" and f_mid[0] == "cam_b_end"
    assert f_top[side_column] == mid[side_column]  # the other side camera now on top
    assert f_bottom[1] == "cam_e_pano"


def test_bowling_end_cam_override():
    layout = derive_mosaic_layout(RIG, bowling_end_cam="cam_a_end")
    assert layout.grid[0][0] == "cam_a_end"
    assert any("override" in note for note in layout.notes)

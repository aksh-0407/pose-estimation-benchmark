"""Contract validation of the P2-P4 tracking fields.

Covers the extension that lets cross-camera / global-ID stages stamp
local_track_id, track_state, single_camera, and a [0, 1] track_confidence
onto Group 1 records while staying backward compatible with P1 output.
"""

from copy import deepcopy

import pytest

from core.contract import (
    example_group1_frame,
    validate_group1_frame,
)


def _frame() -> dict:
    return example_group1_frame(final_handoff=False)


def _player(frame: dict) -> dict:
    return frame["players"][0]


def test_example_intermediate_frame_validates():
    validate_group1_frame(_frame(), final_handoff=False)


def test_p1_style_record_without_tracking_fields_still_validates():
    # A bare P1 record: no track_state / single_camera / track_confidence.
    frame = _frame()
    player = _player(frame)
    for key in ("local_track_id", "track_state", "single_camera", "track_confidence"):
        player.pop(key, None)
    validate_group1_frame(frame, final_handoff=False)


@pytest.mark.parametrize("state", ["confirmed", "lost", "tentative"])
def test_valid_track_states_accepted(state):
    frame = _frame()
    _player(frame)["track_state"] = state
    validate_group1_frame(frame, final_handoff=False)


def test_invalid_track_state_rejected():
    frame = _frame()
    _player(frame)["track_state"] = "dormant"  # internal-only, never emitted
    with pytest.raises(ValueError, match="track_state"):
        validate_group1_frame(frame, final_handoff=False)


@pytest.mark.parametrize("state", ["lost", "tentative"])
def test_null_pose_allowed_for_non_detection_states(state):
    frame = _frame()
    player = _player(frame)
    player["track_state"] = state
    player["pose_2d"] = None
    player["bbox_xywh_px"] = None
    player["bbox_xywh_norm"] = None
    validate_group1_frame(frame, final_handoff=False)


def test_null_pose_rejected_for_confirmed_state():
    frame = _frame()
    player = _player(frame)
    player["track_state"] = "confirmed"
    player["pose_2d"] = None
    with pytest.raises(ValueError, match="pose_2d"):
        validate_group1_frame(frame, final_handoff=False)


def test_null_pose_rejected_when_track_state_absent():
    frame = _frame()
    player = _player(frame)
    player.pop("track_state", None)
    player["pose_2d"] = None
    with pytest.raises(ValueError, match="pose_2d"):
        validate_group1_frame(frame, final_handoff=False)


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
def test_track_confidence_in_unit_range_accepted(value):
    frame = _frame()
    _player(frame)["track_confidence"] = value
    validate_group1_frame(frame, final_handoff=False)


@pytest.mark.parametrize("value", [-0.01, 1.01, 2.0, float("nan")])
def test_track_confidence_out_of_range_rejected(value):
    frame = _frame()
    _player(frame)["track_confidence"] = value
    with pytest.raises(ValueError, match="track_confidence"):
        validate_group1_frame(frame, final_handoff=False)


def test_single_camera_must_be_bool():
    frame = _frame()
    _player(frame)["single_camera"] = "false"
    with pytest.raises(ValueError, match="single_camera"):
        validate_group1_frame(frame, final_handoff=False)


def test_local_track_id_must_be_string_or_null():
    frame = _frame()
    _player(frame)["local_track_id"] = 1234
    with pytest.raises(ValueError, match="local_track_id"):
        validate_group1_frame(frame, final_handoff=False)

    ok = _frame()
    _player(ok)["local_track_id"] = None
    validate_group1_frame(ok, final_handoff=False)


def test_final_handoff_still_requires_global_player_id():
    frame = deepcopy(example_group1_frame(final_handoff=True))
    _player(frame)["global_player_id"] = None
    with pytest.raises(ValueError, match="global_player_id"):
        validate_group1_frame(frame, final_handoff=True)


def test_same_camera_frame_rejects_duplicate_global_ids():
    frame = deepcopy(example_group1_frame(final_handoff=True))
    frame["players"].append(deepcopy(frame["players"][0]))
    with pytest.raises(ValueError, match="unique within one camera frame"):
        validate_group1_frame(frame, final_handoff=True)

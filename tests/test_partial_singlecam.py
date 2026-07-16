from __future__ import annotations

from identity.p5_global_id.runner import _partial_singlecam_ids


def _rec(frame, camera, players):
    return {"frame_index": frame, "camera_id": camera, "players": players}


def _p(gid, nvis):
    # nvis confident keypoints (>0.3), rest 0
    conf = [0.9] * nvis + [0.0] * (26 - nvis)
    return {"global_player_id": gid, "pose_2d": {"confidence": conf}}


def test_head_only_singlecam_is_dropped():
    # P001 seen only in cam_01, ~4 confident keypoints (head only) -> dropped
    recs = [_rec(f, "cam_01", [_p("P001", 4)]) for f in range(10)]
    assert _partial_singlecam_ids(recs, {}, min_visible_kpts=8) == {"P001"}


def test_full_body_singlecam_is_kept():
    # a legit peripheral fielder: 1 camera but 26 confident keypoints -> kept
    recs = [_rec(f, "cam_02", [_p("P008", 26)]) for f in range(10)]
    assert _partial_singlecam_ids(recs, {}, min_visible_kpts=8) == set()


def test_partial_but_multicamera_is_kept():
    # partial (few keypoints) but seen in TWO cameras -> not a single-cam ghost, kept
    recs = [_rec(f, "cam_01", [_p("P005", 4)]) for f in range(10)]
    recs += [_rec(f, "cam_04", [_p("P005", 4)]) for f in range(10)]
    assert _partial_singlecam_ids(recs, {}, min_visible_kpts=8) == set()


def test_id_remap_is_applied():
    # online ids P020/P021 both remap to final P005 seen in two cameras -> kept
    recs = [_rec(f, "cam_01", [_p("P020", 4)]) for f in range(10)]
    recs += [_rec(f, "cam_04", [_p("P021", 4)]) for f in range(10)]
    assert _partial_singlecam_ids(recs, {"P020": "P005", "P021": "P005"}, 8) == set()

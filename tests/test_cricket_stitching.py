from __future__ import annotations

from dataclasses import replace

import numpy as np

from identity.p5_global_id.config import StitchingConfig, GlobalIdConfig
from identity.p5_global_id.stitching import (
    Segment,
    build_link_costs,
    extract_segments,
    remap_ids,
    solve_flow,
    velocity_continuity_cost,
)


def _segment(sid, player_id, start, end, first, last, role="unknown", velocity=(0.0, 0.0)):
    return Segment(
        sid, player_id, start, end, np.asarray(first, float), np.asarray(last, float),
        role, np.asarray(velocity, float),
    )


def test_extract_segments_deduplicates_camera_records_and_splits_gaps():
    records = []
    ground = {}
    for camera in ("cam_01", "cam_02"):
        for frame in (1, 2, 4):
            records.append({
                "frame_index": frame,
                "camera_id": camera,
                "players": [{"global_player_id": "P001", "track_state": "confirmed", "role": "fielder"}],
            })
            ground[("P001", frame)] = np.array([frame * 0.1, 0.0])
    segments = extract_segments(records, ground)
    assert [(item.start_frame, item.end_frame) for item in segments] == [(1, 2), (4, 4)]


def test_flow_links_feasible_nearby_segments_and_rejects_impossible_link():
    config = GlobalIdConfig(stitching=replace(StitchingConfig(), new_traj_cost_factor=2.0))
    segments = [
        _segment(0, "P001", 0, 10, (0, 0), (0, 0)),
        _segment(1, "P002", 12, 20, (0.1, 0), (1, 0)),
        _segment(2, "P003", 12, 20, (50, 0), (51, 0)),
    ]
    edges = build_link_costs(segments, config)
    links = solve_flow(segments, edges, config)
    assert links == {0: 1}


def test_role_and_velocity_costs_disfavor_inconsistent_links():
    aligned = _segment(0, "P001", 0, 10, (0, 0), (0, 0), "bowler", (1, 0))
    forward = _segment(1, "P002", 12, 20, (0.2, 0), (0.4, 0), "bowler")
    backward = _segment(2, "P003", 12, 20, (-0.2, 0), (-0.4, 0), "wicketkeeper")
    assert velocity_continuity_cost(aligned, forward) < velocity_continuity_cost(aligned, backward)
    edges = build_link_costs([aligned, forward, backward], GlobalIdConfig())
    costs = {(edge.source_seg_id, edge.target_seg_id): edge.cost for edge in edges}
    assert costs[(0, 1)] < costs[(0, 2)]


def test_remap_ids_uses_earliest_chain_identity():
    segments = [
        _segment(0, "P010", 0, 3, (0, 0), (0, 0)),
        _segment(1, "P002", 5, 7, (0, 0), (0, 0)),
    ]
    records = [{
        "frame_index": 5,
        "players": [{"global_player_id": "P002"}],
    }]
    report = remap_ids(records, segments, {0: 1})
    assert records[0]["players"][0]["global_player_id"] == "P010"
    assert report == [{"merged_id": "P002", "into_id": "P010", "at_frame": 5}]


def test_remap_ids_rejects_same_camera_frame_identity_collision():
    segments = [
        _segment(0, "P001", 0, 3, (0, 0), (0, 0)),
        _segment(1, "P002", 5, 7, (0, 0), (0, 0)),
    ]
    records = [{
        "frame_index": 2,
        "camera_id": "cam_01",
        "players": [
            {"global_player_id": "P001"},
            {"global_player_id": "P002"},
        ],
    }]
    assert remap_ids(records, segments, {0: 1}) == []
    assert [player["global_player_id"] for player in records[0]["players"]] == ["P001", "P002"]


def test_occupancy_bridge_extends_temporal_gate_for_disjoint_fragments():
    # 200-frame gap: beyond the 120 default gate, inside the 300 occupancy gate.
    segments = [
        _segment(0, "P001", 0, 100, (0, 0), (0, 0)),
        _segment(1, "P002", 300, 400, (0.5, 0), (1, 0)),
    ]
    disjoint = {"P001": {("cam_01", 50)}, "P002": {("cam_01", 350)}}

    baseline = GlobalIdConfig()
    assert build_link_costs(segments, baseline, disjoint) == []  # flag off: no edge

    bridged = GlobalIdConfig(stitching=replace(
        StitchingConfig(), occupancy_bridge=True, occupancy_bridge_require_pose=False,
        new_traj_cost_factor=30.0,
    ))
    edges = build_link_costs(segments, bridged, disjoint)
    assert [(e.source_seg_id, e.target_seg_id) for e in edges] == [(0, 1)]
    assert solve_flow(segments, edges, bridged) == {0: 1}

    # Same-cell overlap revokes the license.
    overlapping = {"P001": {("cam_01", 50)}, "P002": {("cam_01", 50)}}
    assert build_link_costs(segments, bridged, overlapping) == []

    # Beyond the occupancy gate the bridge is still refused.
    far = [
        _segment(0, "P001", 0, 100, (0, 0), (0, 0)),
        _segment(1, "P002", 500, 600, (0.5, 0), (1, 0)),
    ]
    assert build_link_costs(far, bridged, disjoint) == []


def test_occupancy_bridge_pose_requirement_blocks_shapeless_long_links():
    segments = [
        _segment(0, "P001", 0, 100, (0, 0), (0, 0)),
        _segment(1, "P002", 300, 400, (0.5, 0), (1, 0)),
    ]
    disjoint = {"P001": {("cam_01", 50)}, "P002": {("cam_02", 350)}}
    strict = GlobalIdConfig(stitching=replace(
        StitchingConfig(), occupancy_bridge=True,  # require_pose defaults True
        pose_stitch_max_distance=0.3, w_pose=2.0,
    ))
    # Neither fragment carries a mature descriptor -> the long bridge must abstain.
    assert build_link_costs(segments, strict, disjoint) == []
    # The normal short-gap path is unaffected by the requirement.
    short = [
        _segment(0, "P001", 0, 100, (0, 0), (0, 0)),
        _segment(1, "P002", 150, 250, (0.5, 0), (1, 0)),
    ]
    assert len(build_link_costs(short, strict, disjoint)) == 1


def test_posture_stitch_gate_blocks_different_builds():
    from identity.common.pose_shape import PostureAggregate

    def posture(head_top):
        return PostureAggregate(
            median={"head_top_m": head_top, "torso_len_m": 0.55},
            se={"head_top_m": 0.01, "torso_len_m": 0.01},
            count={"head_top_m": 40, "torso_len_m": 40},
        )

    tall, short = posture(1.85), posture(1.55)
    seg_a = replace(_segment(0, "P001", 0, 10, (0, 0), (0, 0)), posture=tall)
    near_twin = replace(_segment(1, "P002", 12, 20, (0.1, 0), (1, 0)), posture=posture(1.84))
    wrong_build = replace(_segment(1, "P002", 12, 20, (0.1, 0), (1, 0)), posture=short)

    gated = GlobalIdConfig(stitching=replace(StitchingConfig(), posture_stitch_max_z=3.0, w_posture=0.5))
    assert len(build_link_costs([seg_a, near_twin], gated)) == 1     # same build passes
    assert build_link_costs([seg_a, wrong_build], gated) == []       # wrong build blocked
    # missing posture abstains (edge still allowed)
    bare = _segment(1, "P002", 12, 20, (0.1, 0), (1, 0))
    assert len(build_link_costs([seg_a, bare], gated)) == 1
    # flag off: byte-identical behaviour (edge allowed regardless)
    off = GlobalIdConfig()
    assert len(build_link_costs([seg_a, wrong_build], off)) == 1


def test_bowler_detection_is_direction_signed():
    # H2: a sprint AGAINST the bowling direction must not win the bowler crown.
    import numpy as np
    from identity.p6_roles.assigner import _windowed_axis_speed

    axis = np.array([0.0, 1.0])
    toward = [(f, np.array([0.0, -20.0 + 0.1 * f])) for f in range(100)]
    against = [(f, np.array([0.0, 20.0 - 0.1 * f])) for f in range(100)]
    kw = dict(window_frames=50, frame_rate_fps=50.0, early_cutoff=100)
    assert _windowed_axis_speed(toward, axis, **kw) > 3.5
    assert _windowed_axis_speed(against, axis, **kw) <= 0.0


def test_normalized_costs_make_long_gap_stitches_selectable():
    # G7: legacy units let w_temporal*gap alone dwarf the dummy for gaps > 30
    # frames - a real 120-frame occlusion could NEVER stitch. Normalized mode
    # keeps in-gate costs commensurate with the dummy.
    segments = [
        _segment(0, "P001", 0, 100, (0, 0), (0, 0)),
        _segment(1, "P002", 220, 300, (0.5, 0), (1, 0)),   # gap 120 = the full gate
    ]
    legacy = GlobalIdConfig(stitching=replace(StitchingConfig(), new_traj_cost_factor=3.0))
    edges = build_link_costs(segments, legacy)
    assert len(edges) == 1
    assert solve_flow(segments, edges, legacy) == {}        # documents the dead zone

    normalized = GlobalIdConfig(stitching=replace(
        StitchingConfig(), new_traj_cost_factor=3.0, normalized_costs=True,
    ))
    edges = build_link_costs(segments, normalized)
    assert solve_flow(segments, edges, normalized) == {0: 1}  # now selectable


# ------------------------------------------------------- W9 colocated merge
def test_merge_colocated_ids_disjoint_cameras_merges_and_shared_blocks():
    import numpy as np

    from identity.p5_global_id.stitching import merge_colocated_ids

    # A and B co-located every frame; A lives in cam_01, B in cam_02 (disjoint)
    # C and D co-located but BOTH appear in cam_03 (same camera-frame) => block.
    records = []
    ground = {}
    for f in range(40):
        records.append({"frame_index": f, "camera_id": "cam_01",
                        "players": [{"global_player_id": "A"}]})
        records.append({"frame_index": f, "camera_id": "cam_02",
                        "players": [{"global_player_id": "B"}]})
        records.append({"frame_index": f, "camera_id": "cam_03",
                        "players": [{"global_player_id": "C"}, {"global_player_id": "D"}]})
        ground[("A", f)] = np.array([0.0, 0.0])
        ground[("B", f)] = np.array([0.3, 0.0])
        ground[("C", f)] = np.array([10.0, 0.0])
        ground[("D", f)] = np.array([10.3, 0.0])
    id_remap: dict = {}
    report = merge_colocated_ids(records, ground, {}, id_remap,
                                 radius_m=0.75, min_frames=25)
    merged = {(e["merged_id"], e["into_id"]) for e in report}
    assert ("B", "A") in merged           # disjoint-camera co-location merges
    assert not any("C" in pair or "D" in pair for pair in merged)  # shared camera blocks
    # records patched
    ids_cam2 = {p["global_player_id"] for r in records if r["camera_id"] == "cam_02"
                for p in r["players"]}
    assert ids_cam2 == {"A"}


def test_merge_colocated_ids_needs_min_frames():
    import numpy as np

    from identity.p5_global_id.stitching import merge_colocated_ids

    records = []
    ground = {}
    for f in range(10):  # only 10 close frames < 25
        records.append({"frame_index": f, "camera_id": "cam_01",
                        "players": [{"global_player_id": "A"}]})
        records.append({"frame_index": f, "camera_id": "cam_02",
                        "players": [{"global_player_id": "B"}]})
        ground[("A", f)] = np.array([0.0, 0.0])
        ground[("B", f)] = np.array([0.3, 0.0])
    assert merge_colocated_ids(records, ground, {}, {}, radius_m=0.75, min_frames=25) == []

# Halpe-26 skeleton, joint names & the 3D output schema

The pipeline's canonical skeleton is **Halpe-26**. Every `pose_2d` and `pose_3d`
block carries all 26 joints. Indices `0-16` are exactly **COCO-17** (COCO order);
`17-25` add head/neck/hip and the six foot joints.

> Visual reference (numbered skeleton on a body + colour-coded table):
> <https://claude.ai/code/artifact/43ae8b17-8eae-4214-b318-4742c20f5860>.
> Source constants: `src/core/keypoints.py` (`HALPE26_KEYPOINTS`, `HALPE26_EDGES`).

## Index to joint name

| idx | name | group | idx | name | group |
|----:|------|-------|----:|------|-------|
| 0 | nose | COCO-17 | 13 | left_knee | COCO-17 |
| 1 | left_eye | COCO-17 | 14 | right_knee | COCO-17 |
| 2 | right_eye | COCO-17 | 15 | left_ankle | COCO-17 |
| 3 | left_ear | COCO-17 | 16 | right_ankle | COCO-17 |
| 4 | right_ear | COCO-17 | 17 | head | Halpe extra |
| 5 | left_shoulder | COCO-17 | 18 | neck | Halpe extra |
| 6 | right_shoulder | COCO-17 | 19 | **hip** (root) | Halpe extra |
| 7 | left_elbow | COCO-17 | 20 | left_big_toe | Halpe extra |
| 8 | right_elbow | COCO-17 | 21 | right_big_toe | Halpe extra |
| 9 | left_wrist | COCO-17 | 22 | left_small_toe | Halpe extra |
| 10 | right_wrist | COCO-17 | 23 | right_small_toe | Halpe extra |
| 11 | left_hip | COCO-17 | 24 | left_heel | Halpe extra |
| 12 | right_hip | COCO-17 | 25 | right_heel | Halpe extra |

**Root joint** for the root-relative export is index **19 (`hip`)**, the mid-hip.

## Bones (connectivity)

`HALPE26_EDGES`: face `(0,1)(0,2)(1,3)(2,4)`; spine `head(17)-neck(18)`,
`neck-l/r_shoulder(5,6)`, `neck-hip(19)`; arms `5-7-9`, `6-8-10`;
`hip-l/r_hip(11,12)`; legs `11-13-15`, `12-14-16`; left foot
`ankle(15)-heel(24)/big_toe(20)/small_toe(22)`; right foot
`ankle(16)-heel(25)/big_toe(21)/small_toe(23)`.

## 3D output, `pose_3d` and `pose_3d_named`

Each identified player carries the full 26-joint 3D (`pose_3d`, absolute world
metres) plus a self-describing **named + root-relative** view (`pose_3d_named`):
the root joint in world metres, every joint relative to the root, keyed by name.

```jsonc
"pose_3d": {
  "keypoints_world_m": [[x, y, z], … 26 joints],   // absolute world metres
  "confidence": [ … 26 ],
  "mean_reprojection_error_px": [ … 26 ]
},
"pose_3d_named": {
  "root_joint": "hip",
  "root_world_m": [2.743, 9.075, 1.144],            // root in world metres
  "joints_root_relative_m": {                        // every joint relative to root
    "hip":  [0.0, 0.0, 0.0],
    "nose": [0.031, 0.012, 0.618],
    "left_shoulder": [-0.187, 0.021, 0.409],
    "left_heel": [0.061, -0.043, -0.905],
    …                                                // 26 entries; null if not triangulated
  }
}
```

World convention: **X = right, Y = forward, Z = up**, metres, `z = 0` at the turf
(pitch-centre origin). UE export (`identity.export.export_ue_packets`) additionally
emits `keypoints3d_ue_cm` (X-Y swap, ×100).

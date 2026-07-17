# 07, export & render (terminal)

> Terminal hand-offs after identity + roles. Code: `src/identity/export/`,
> `src/identity/visualization/`. Neither changes identity or geometry.

## Unreal Engine export

`src/identity/export/export_ue_packets.py` converts the triangulated 3D JSONL into
`cricket_pose_packet/v1` packets (`core.schemas.PosePacket`), applying `cricket_world_to_ue_cm`
(`src/core/ue_transform.py`): internal axes (X=right, Y=forward, Z=up, metres) to UE axes
(X=forward, Y=right, Z=up) scaled ×100 to centimetres.

```bash
python -m identity.export.export_ue_packets --run-dir <06_roles> --output <ue.jsonl> --model-version <v>
```

Run per delivery when UE-format packets are needed. It reads the `06_roles` terminal predictions
(26-joint `pose_3d` + `pose_3d_named` + `role`) and emits one packet per identified player per
frame in UE centimetres.

## Mosaic / bird's-eye render

`src/identity/visualization/render_videos.py` is the render CLI and orchestrator; it produces
the diagnostic videos from a global-id run, colouring skeletons by stable global ID. Its
building blocks live beside it: `video_io.py` (encoders, GPU frame decoding), `loaders.py`
(every prediction and side-artifact reader), `overlays.py` (everything drawn on a camera
picture), `panels.py` (roster and bird's-eye tiles). The phase-1 overlay and bird's-eye
renderers read the same `loaders` module, so each file format is parsed in one place.

- **Layout** (`mosaic_layout.derive_mosaic_layout`): a 3x3 grid derived per delivery from
  calibration, one facing pair per column (end-on pair first), side tiles mirrored so the
  delivery reads right-to-left, the panoramic cam_07 bottom-middle, flanked by a bird's-eye
  ground monitor and a roster panel. Nothing is hardcoded; the bowling end flips between overs.
- **Render**: collision-aware player chips with leader lines, a skeleton body-paint identity
  overlay, a 20-colour max-separation palette, roles shown **only** in the roster panel, and
  suppressed players dropped. The bird's-eye dots expose stage-05 emitted teleports; colour
  flicker exposes stage-03/05 ID switches (517 across the 40-delivery set,
  `../diagnosis/07-issue-2d-id-switch-flicker.md`).

**Driver exit-code convention** (`src/main.py`): a stage exiting 0 succeeded; stages
`03_association` and `05_global_id` may exit 1 to signal a warn/fail *verdict* while still
having produced full output (the driver distinguishes a warn from a crash by the presence of
the stage's metrics artifact); any other nonzero exit is a stage failure and halts that
delivery's chain. The render step currently reads `05_global_id` regardless of whether 06/07
ran (known-bugs.md BUG-14).

```bash
# mosaic: 7 tiles + bird's-eye monitor + roster
python -m identity.visualization.render_videos \
  --drive-root drive --run-dir <05_global_id> --delivery-id <D> --mode mosaic --show p4

# bird's-eye ground view only
python -m identity.visualization.render_videos \
  --drive-root drive --run-dir <05_global_id> --delivery-id <D> --mode ground --show p4
```

`--mode {all,per-camera,mosaic,ground}`; `--show {p2,p3,p4}` selects **which stage's IDs** to
colour by (a semantic selector, `p4` = global identity, not a directory name). A standalone
top-down renderer is `render_bird_eye_view.py`; the all-delivery batch driver is
`render_all_mosaics.py`.

## Downstream consumers

The other groups consume the emitted `pose_3d.keypoints_world_m` + `global_player_id` + `role`
via the JSON contract (see [`../architecture.md`](../architecture.md)), not our code. How to read
a run's outputs is documented in [`../shared-data.md`](../shared-data.md).

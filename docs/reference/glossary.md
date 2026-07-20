# Glossary

Terms a newcomer needs to read this repo's code, configs, and docs. Cricket
terms first, then pipeline terms, then data-layout terms.

## Cricket and capture terms

- **Delivery**: one bowled ball, the unit of processing. Each delivery is a
  ~12 s clip captured simultaneously by 7 cameras at 50 fps (~600 frames per
  camera). Delivery ids look like `CCPL080626M1_1_14_1`.
- **Match id**: the leading token of a delivery id (for example `CCPL080626`),
  used to look up calibration.
- **Capture group** (`bt_01`, `bt_02`, `bt_03`): the recording batches the
  cameras were captured in; group directories hold delivery directories.
- **Roles**: bowler, striker, non-striker, wicketkeeper, umpire, fielder. The
  roles stage (06) assigns them from ground-plane geometry and motion.
- **Pitch / creases / 30-yard ring**: field reference geometry drawn by the
  bird's-eye view; the pitch strip is about 3.05 m by 20.12 m at the origin.
- **Facing pairs**: camera pairs that look at the same ground strip from
  opposite sides: C1-C4, C2-C6, C3-C5 (cam_07 is unpaired). They co-observe
  players but with low parallax, which makes cross-camera geometry weak
  exactly where co-observation is highest. Derived from calibration at run
  time; the config list is only a default.

## Skeleton and detection terms

- **Halpe-26**: the pipeline's canonical 2D skeleton, 26 keypoints. Indices
  0-16 are exactly COCO-17 (nose, eyes, ears, shoulders, elbows, wrists, hips,
  knees, ankles); 17-19 add head/neck/mid-hip; 20-25 add big toes, small toes,
  and heels. Produced by RTMPose-x body8-halpe26. See
  docs/reference/skeleton-halpe26.md.
- **Top-down pose**: detect person boxes first (RTMDet), then estimate
  keypoints per box (RTMPose). The alternative (bottom-up) is not used here.
- **Tiled detection**: splitting a frame into overlapping tiles plus one
  full-frame pass, running detection per tile, and merging with non-maximum
  suppression plus containment suppression. Recovers small/distant players
  that a single 640 px pass misses.
- **Detection confidence / keypoint confidence**: per-box and per-joint scores
  from the models; thresholds throughout the pipeline gate on them.

## Identity pipeline terms

- **Tracklet**: a per-camera track fragment produced by stage 02 (ByteTrack by
  default): one camera's view of one person over a contiguous frame span.
- **Chunk**: the tracklet-graph's unit; a temporally contiguous piece of one
  (camera, tracklet) pair.
- **Tracklet graph**: stage 03's production mode. Instead of clustering per
  frame, it accumulates evidence between every co-visible tracklet pair over
  the whole delivery (ground agreement, appearance, posture, motion as
  log-likelihood-ratio cues), then merges tracklets into cross-camera groups.
- **Cluster / correspondence**: the per-frame grouping emitted by stage 03; a
  cluster's members are (camera, player-index) detections of one person.
- **Binding (binding_id)**: the delivery-level identity of a tracklet-graph
  group; the key that links stage 03 output, the 04 lift, and stage 05 input.
- **Ground plane / ground_xy**: world coordinates on the field plane (z = 0),
  in metres. Every camera has a calibrated projection matrix; the world frame
  has z = 0 on the ground.
- **Foot contact point**: the image pixel taken as a player's ground contact.
  The bounding-box bottom-centre is the fallback; ankle/heel/toe keypoints are
  the intended refinement (currently inactive; see bugs.md BUG-A1).
- **z0 reprojection (z0_reproj)**: solving for the ground point whose
  reprojection into all observing cameras best matches the observed foot
  pixels, instead of averaging per-camera back-projections.
- **Triangulation / DLT / RANSAC**: lifting a joint to 3D from 2 or more
  camera views (direct linear transform), with random-sample consensus to
  reject outlier views. Stage 04 triangulates the full skeleton per binding.
- **Cheirality**: the constraint that a triangulated point must lie in front
  of every camera that saw it.
- **Global id (global_player_id)**: the persistent per-player identity
  assigned by stage 05 and carried to the render (ids like `P007`).
- **Singer model / Kalman filter**: the ground-plane motion model used by
  stage 05's online tracker; maneuver-adaptive constant-velocity filtering.
- **Mahalanobis gate / chi-squared gate**: statistical distance test deciding
  whether an observation may update a track.
- **Shadow id**: a duplicate identity sitting on top of an already-tracked
  player (typically from a split detection); stage 05 absorbs or suppresses
  these rather than minting a new player.
- **Stitching**: stage 05's second pass; joins track fragments across time
  gaps with a min-cost-flow assignment over temporal/spatial/pose costs.
- **Stitch seam**: the frame where two fragments were joined.
- **Occupancy**: the set of (camera, frame) cells a track's detections cover.
  Two fragments whose occupancies overlap cannot be one person (one camera
  cannot see the same person twice in one frame).
- **Teleport**: a physically impossible jump in an emitted ground track; the
  headline quality metric (a real cricketer stays under about 11 m/s).
- **Lost window**: how many frames a track survives without an update before
  deletion; adaptive variants extend it for well-established tracks.
- **Re-entry**: reviving a recently deleted track for a new observation
  instead of minting a fresh id.
- **Chimera**: a cross-camera cluster that mixes two different people (one
  camera's view of player A grouped with another camera's view of player B);
  torso-reprojection residuals flag suspects.
- **Posture (billboard posture)**: a monocular body-shape descriptor (relative
  keypoint geometry within the detection box) usable without parallax; used as
  a same-person veto between tracklets/fragments.
- **Pose-shape descriptor**: view-invariant 3D bone-length ratios from
  triangulated skeletons; a stronger same-person key where parallax allows.
- **Peripheral suppression**: dropping clearly low-quality single-camera
  peripheral identities (cut-off umpires, partial fielders) from the render
  and the 3D lift (stage 06, `suppression.json`).
- **Refinement (stage 07)**: post-identity physics cleanup of the 3D
  skeletons: bone-length normalization, joint-angle clamps, root smoothing.
  Never changes identities.
- **Ghost marker**: a render-only marker for a player not detected in a given
  camera this frame, drawn at the reprojected fused (or last-known) ground
  position: "occluded" if another camera still sees the player, "lost" if no
  camera does. Nothing synthetic enters the pipeline.

## Evaluation terms

- **8-delivery set (8_init)**: the reference working set iterated on locally.
- **40-delivery set (40_full)**: the full campaign set on the remote GPU box;
  a change is not called an improvement until confirmed there.
- **Cross-camera agreement**: fraction of frames where independent per-camera
  ground projections agree with the clustering; the primary association
  metric.
- **Verdict / usability rubric**: the composite grade over agreement, emitted
  smoothness, persistence, and parsimony printed per delivery (see
  docs/analysis/10-verdict-redesign.md).
- **Byte-identical**: the refactor verification standard used across the
  codebase: re-running a stage produces bit-for-bit identical predictions and
  metrics (timestamps and embedded paths excluded).

## Data layout terms

- **Run tree**: `<derived>/<run>/<DELIVERY>/<stage>/` where stage is one of
  `00_inference`, `01_stabilization`, `02_tracking`, `03_association`,
  `04_lift`, `05_global_id`, `06_roles`, `07_refine`, plus `logs/`.
- **Predictions JSONL**: one line per frame per camera
  (`<group>__<delivery>__<cam>.jsonl`), schema `g1_player_frame/v1`
  (src/core/contract.py).
- **Run manifest (run_manifest.json)**: per-stage record of exactly what ran:
  config echo, versions, timings. The authoritative record of a run's flags.
- **Drive root**: the dataset root holding `bt_0X/` frame dirs,
  `calibration-data/`, and `events-data/` (ball events).

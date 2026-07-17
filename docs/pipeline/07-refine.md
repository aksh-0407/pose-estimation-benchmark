# 07, refine

> **Stage 07**, physics-constrained 3D skeleton refinement. Takes the triangulated, identity-assigned
> skeletons and makes them *physically valid* and *smooth*: constant bone lengths, steady hips,
> anatomical joint limits, and reliable low-confidence recovery. Code: `src/identity/p7_refine/`,
> config `configs/07_refine.yaml`. Consumes stage 06's output; export + render follow.

---

## 1. What this stage does (and why)

After identity assignment (stage 05) and role labelling (stage 06), each player's 3D skeleton is
geometrically correct on average but suffers three artefacts of per-frame triangulation:

1. **Impossible limbs**, a joint triangulated from a bad camera stretches the bone far beyond
   anatomical range. A forearm might double in length for a frame, then snap back.
2. **Hip wobble**, the root (mid-hip) position jitters frame-to-frame even on a stationary player,
   because small 2D pixel noise maps to large 3D depth variation through triangulation.
3. **Low-confidence joints**, a joint the 2D model barely trusts (e.g. an occluded ankle) is
   triangulated raw; the resulting 3D point is unreliable and drags its neighbours.

Stage 07 fixes all three in a single offline pass over the whole clip. It operates purely on `pose_3d`
positions, **it never reads or writes any identity field**, so IDs are byte-identical to the input by
construction.

> **In plain words:** the "make it look real" step, lock bone lengths so arms don't stretch, smooth
> the hips so the whole skeleton stops shaking, and replace bad joints with sensible guesses from
> their neighbours in time.

---

## 2. Inputs and outputs

| | |
|---|---|
| **Input** | a 06 run (`predictions/*.jsonl` carrying `global_player_id` + `pose_3d` + `pose_2d`) |
| **Output** | a new run dir with `pose_3d` / `pose_3d_named` rewritten; `refinement_metrics.json` + `run_manifest.json` |
| **Core** | `src/identity/p7_refine/{refine.py, relift.py, runner.py, config.py}` |
| **Config** | `configs/07_refine.yaml` |

---

## 3. How it works

The refinement pipeline runs five steps in order for each identity's whole-clip sequence:

### 3a. Visibility-aware re-lift (`relift.py`), fix partially-visible views

**Root cause it solves:** an umpire whose full body is in cam_01 but only upper body is in cam_04.
The 2D pose model still emits lower-body keypoints for cam_04, crammed at the image edge with low
confidence. Stage-04 triangulation pairs cam_01's good legs with cam_04's *hallucinated* legs and the
3D legs stretch along the depth ray, a pose that reprojects perfectly yet is physically impossible.

**Fix:** for each joint, classify which cameras *reliably* see it (by confidence + in-frame check):
- **≥ 2 reliable views** to ordinary weighted-DLT triangulation (clean);
- **exactly 1 reliable view** to place the joint on that camera's back-projection ray at the canonical
  bone length from its already-placed parent (single-view bone-length lift);
- **0 reliable views** to left NaN for temporal/skeletal-prior fill downstream.

Two-pass design: pass 1 triangulates only the confidently-multi-view joints and estimates canonical
bone lengths from those clean samples; pass 2 re-lifts every frame using the canonical lengths for
single-view placement, with the previous frame as a temporal-continuity prior.

> **In plain words:** if only one camera sees a joint well, instead of triangulating against a camera
> that can barely see it (which stretches the limb), hang the joint at the right bone length along
> that one camera's line of sight.

### 3b. Low-confidence gate + temporal fill

Joints with per-keypoint confidence below `conf_floor` (default 0.5) are set to NaN, dropped rather
than trusted. Short NaN gaps (up to `max_gap_frames` = 25 frames = 0.5 s at 50 fps) are filled by
linear interpolation from the surrounding finite values; longer gaps or edge cases are filled by a
**skeletal prior** (place the missing joint at its canonical bone offset from the nearest placed
parent, using the clip's most-complete frame as the reference pose).

> **In plain words:** if the model isn't sure about a joint, throw it away and fill it in from what
> it was doing before and after, or from what a correct skeleton should look like.

### 3c. Canonical rigid bones (`estimate_canonical_bones`)

Estimate per-player constant, bilaterally-symmetric, anatomically-bounded bone lengths:

1. Collect the raw `|child − parent|` distance for every bone over the whole clip.
2. **Pool left/right pairs** (left-upper-arm = right-upper-arm) so the skeleton is symmetric.
3. Take the **median** (robust to per-frame noise).
4. **Clamp** to the absolute human range from `HALPE26_BONE_LIMITS_M`, a median outside its range is
   a triangulation artefact (chimera identity, persistent bad view), not a real limb. Every bone
   always gets a length (its anatomical default when there are no reliable samples), so a full
   skeleton can always be rebuilt.

> **In plain words:** measure each player's real bone lengths once for the whole clip, force left and
> right to match, and cap to the human range so nothing impossible gets through.

### 3d. Bone-length-preserving smoothing (`fk_smooth`)

A plain XYZ low-pass filter cannot simultaneously smooth positions and preserve bone lengths, the
filter re-breaks the lengths. The solution: **decompose, smooth, recompose.**

1. **Decompose** the (T, J, 3) sequence into a root trajectory (mid-hip XYZ) and per-bone
   unit-direction vectors.
2. **Smooth the root** with the lower cutoff (3.0 Hz Butterworth, zero-phase), the hips stop wobbling.
3. **Smooth each bone direction** with the higher cutoff (6.0 Hz), limb motion stays responsive.
4. **Extra-smooth chain-end groups** (face/foot bones at 1.5/2.5 Hz, forearm/shank at 4.5 Hz) , 
   these sit at the ends of the kinematic chain where 2D noise accumulates and short bones amplify
   angular jitter.
5. **Renormalize** the smoothed directions to unit length.
6. **Recompose** via forward kinematics: root + direction × canonical-length for each bone, BFS order.
   The result is *exactly* constant bone length AND smooth.

Both Butterworth (zero-phase, offline-quality, the default) and centred moving-average (scipy-free
fallback) smoothers are available.

> **In plain words:** smooth the hip and the bone directions separately, then bolt the skeleton back
> together with the fixed bone lengths. That way the hips are steady, the limbs are smooth, and no
> bone ever stretches.

### 3e. Anatomical hinge-angle clamp (`clamp_joint_angles`)

After smoothing, knees and elbows are clamped to `[15°, 178°]` flexion, preventing backward bends
and self-intersecting poses. Each hinge's **distal subtree is rotated rigidly** about the joint
(Rodrigues rotation in the plane the three points span), so clamping a knee also carries the foot,
and every downstream bone length is preserved.

> **In plain words:** if a knee bends backward or an elbow hyper-extends after smoothing, rotate it
> back to the anatomical limit and drag the whole lower limb with it so nothing disconnects.

---

## 4. Config knobs (`configs/07_refine.yaml`)

```yaml
enabled: true
relift: true              # visibility-aware re-lift (needs calibration / drive-root)
vis_conf: 0.5             # reliable-view confidence threshold
edge_margin_px: 4.0       # out-of-bounds pixel margin

conf_floor: 0.5           # joints below this are dropped and refilled
max_gap_frames: 25        # longest temporal-interpolation gap (real frames)

fps: 50.0
root_cutoff_hz: 3.0       # lower to steadier hips
limb_cutoff_hz: 6.0       # higher to more responsive limbs
filter_order: 4
face_cutoff_hz: 1.5       # face bones: near-rigid, smooth very hard
foot_cutoff_hz: 2.5       # ankle to toe/heel: heavy (barely articulates)
mid_cutoff_hz: 4.5        # forearm/shank: moderate

clamp_angles: true
min_hinge_deg: 15.0       # minimum knee/elbow flexion
max_hinge_deg: 178.0      # maximum flexion before backward-bend
dev_tol: 0.25             # bone deviation vs canonical counted as "impossible" (metrics)
```

**Key design choice: the dual cutoff.** The root trajectory gets a much lower cutoff (3.0 Hz) than
the limb directions (6.0 Hz). This means the whole-body position is very stable (no wobble), while
genuine fast limb motion (a bat swing, a bowler's arm) passes through with minimal lag.

---

## 5. Metrics (`refinement_metrics.json`)

| Metric | What it measures |
|---|---|
| `jitter_mean_m_before/after` | Mean frame-to-frame joint displacement (metres), should decrease |
| `jitter_p95_m_before/after` | 95th-percentile jitter, the worst-case smoothing improvement |
| `hip_jitter_mean_m_before/after` | Hip-specific jitter, the manager's "wobbly hips" metric |
| `max_bone_cv_before/after` | Max coefficient of variation of bone lengths, ~0 after = no stretch |

---

## 6. What's been tried (verdicts)

| Change | Measured result | Verdict |
|---|---|---|
| **Full refine pipeline** (bones + smoothing + clamp + relift) | Physically valid skeletons; bone-length CV to ~0; hip jitter reduced |  ACCEPTED, enabled |
| **Moving-average fallback** (`smoother: moving_average`) | Works; slightly less smooth than Butterworth on the root | Available, not default |
| **Relift off** (`relift: false`) | Falls back to refining the existing stage-04 `pose_3d` without re-triangulation | For environments without calibration access |

---

## 7. Strengths / weaknesses

**Strengths**
- **Offline, zero-phase**, the whole clip is available, so smoothing introduces no lag.
- **Bone-length-exact by construction**, the FK rebuild guarantees constant lengths; no post-hoc
  clamping artifact.
- **Identity-agnostic**, reads only `pose_3d` and `global_player_id` as a key; never touches any
  identity field. A bug in 07 cannot corrupt tracking.
- **Visibility-aware re-lift**, fixes the specific stretched-limb artefact from partially-visible
  cameras, which the downstream smoother alone cannot fix (it smooths the stretch, not removes it).
- **Anatomically bounded**, both bone lengths and joint angles are clamped to human ranges, so no
  emitted skeleton can be physically impossible.

**Weaknesses**
- **Offline-only**, requires the whole clip; cannot run in a real-time streaming mode.
- **No learned body model**, uses kinematic (bone-length + hinge) constraints, not a statistical
  body model (SMPL). Unusual poses that are valid but uncommon pass the clamp but aren't
  plausibility-checked against a pose prior.
- **Single-view bone-length lift (relift) is a heuristic**, placing a joint on the ray at bone
  length from the parent picks one of two possible solutions; temporal continuity resolves most
  cases, but rapid depth changes (toward/away from camera) can pick wrong for a frame or two.
- **Depends on calibration**, the relift needs projection matrices from `drive/`. Without them,
  it falls back to smoothing the existing `pose_3d` (still useful, but the stretched-limb fix is
  lost).

---

## 8. Known issues (severity, 1 low to 3 high)

- **REF-1 (severity 1/3) Smoother ring on sharp motion onset.** A zero-phase Butterworth with a 3 Hz cutoff
  introduces a brief pre-ring (~2 frames) before a sudden acceleration (a bowler starting the run).
  Acceptable at 50 fps; would matter at lower frame rates.
- **REF-2 (severity 1/3) Hinge clamp is axis-free.** The flexion plane is derived from the three-point geometry
  per frame, not from a consistent anatomical axis. On near-collinear limbs the plane is unstable and
  the clamp may produce a small rotational jitter. Mitigated by the 15° minimum (never near full
  extension).

---

## 9. Entry-point commands

```bash
# Full refine with re-lift (needs drive/ calibration):
python -m identity.p7_refine.run_refinement \
  --input-run-dir <06_roles> --output-run-dir <07_refine> \
  --delivery-id <D> --drive-root drive --config configs/07_refine.yaml

# Refine without re-lift (no calibration needed):
python -m identity.p7_refine.run_refinement \
  --input-run-dir <06_roles> --output-run-dir <07_refine> \
  --delivery-id <D> --config configs/07_refine.yaml --no-relift
```

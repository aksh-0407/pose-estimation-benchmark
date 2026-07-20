# keypoint_contract_handling.md — what happens if P1 feeds a non-Halpe-26 (e.g. COCO-17) skeleton

Answer to the handover question: *the whole pipeline assumes Halpe-26 (26 keypoints). If someone
produced P1 (00_inference) output with a 17-keypoint model and ran the later stages, how does the
current code handle that mismatch?* Short version: **it is caught early and loudly at the contract
boundary in the normal case, but the defense is a hard crash, not a graceful path — and there is one
ironic soft spot.** No code was changed to write this; it is a description of current behaviour.

## 1. The single source of truth
`src/core/contract.py` fixes the skeleton for the entire pipeline:
```
SKELETON = "halpe26"      # contract.py:12
KEYPOINT_COUNT = 26       # contract.py:13
```
`validate_pose_2d` (contract.py:74-95) hard-requires **all three** of:
- `pose_2d.skeleton == "halpe26"` (exact string), else `ValueError("pose_2d.skeleton must be halpe26")`;
- `keypoints_px` is a list of exactly **26** 2-D points;
- `keypoints_norm` is 26 points and `confidence` is length 26.

`pose_3d` / `pose_3d_named` are validated at 26 as well (with per-joint `null` allowed).

## 2. Where the gate actually fires — every stage validates on WRITE
`validate_group1_frame` is called at the point each stage writes a record, in **every** stage:

| Stage | Call site |
|---|---|
| P1 inference (both runners) | `run_phase1_rtmpose_inference.py:580`, `run_phase1_l40s.py:896` |
| 01 stabilization | `p1_stabilization/runner.py:113,134` |
| 02 tracking | `p2_tracking/jsonl_io.py:137,144` |
| 03 association | `p3_association/jsonl_io.py:34,166,227` |
| 04 lift | `p4_lift/run_triangulation.py:388` |
| 05 global-id | `p5_global_id/runner.py:629`, `jsonl_io.py:112` |
| 06 roles | `p6_roles/suppress_peripherals.py:164` |
| 07 refine | `p7_refine/runner.py:248` |

Consequence: the mismatch cannot travel far. It is rejected at the **first write** of the first stage
that touches it.

## 3. The three concrete scenarios

**(a) Someone runs P1 with a COCO-17 (17-kp) pose model.**
P1 cannot even produce output. `resolve_skeleton` (phase1_common.py:468-484) infers the skeleton from the
model's config-path tokens; a COCO model resolves to `coco_17`, the record is stamped
`pose_2d.skeleton = "coco_17"` with 17 keypoints, and the very first `validate_group1_frame` call at
`run_phase1_*.py` raises `ValueError("pose_2d.skeleton must be halpe26")` on frame 1. **The run aborts
immediately and loudly.** (This is the "self-checking but brittle" behaviour noted as BUG-8 in
`resolvebugs.md`: it fails for the *skeleton-name* reason even before the count check.)

**(b) Someone hand-crafts / patches 17-kp JSONL and feeds it to stage 01+.**
Two sub-cases:
- Left honest (`skeleton:"coco_17"`, 17 points): the first stage's `validate_group1_frame` raises on the
  skeleton string — caught, loud.
- Faked (`skeleton:"halpe26"` but only 17 points): `validate_points(count=26)` raises
  `ValueError("pose_2d.keypoints_px must contain 26 points")` — still caught, loud. To get *past* the
  contract you would have to pad every array to 26, at which point it is no longer 17-keypoint data.

So under the contract, **there is no silent-wrong-output path for a raw count mismatch** — it is a
`ValueError` at the boundary.

**(c) Someone disables/bypasses contract validation and forces 17-kp arrays into the compute code.**
This is where behaviour becomes inconsistent, because the compute modules were written with mixed
assumptions:
- **Hard crash** — `p2_tracking/pose_vector.py:60-61` does `reshape(KEYPOINT_COUNT=26, 2)`. 17 keypoints =
  34 values, which cannot reshape into (26, 2) → `ValueError: cannot reshape array of size 34 into shape
  (26,2)`. P2 dies.
- **Silently "works" (ironically)** — `common/geometry.py:188` (`ground_contact_pixel_ex`) guards on
  `points.shape == (17, 2)`. This is exactly **BUG-1**: with the real 26-kp data the guard fails and the
  function silently returns bbox-bottom. With *17-kp* data the guard **passes** and the ankle/foot logic
  actually runs. So the one place that is broken for the production skeleton is the one place that would
  behave "correctly" for a COCO-17 input. Do not read this as 17-kp being supported — it is a symptom of
  the same un-migrated guard.
- **Partial skeleton-flex** — the 3D lift is the *only* subsystem with an explicit dual-skeleton branch:
  `run_triangulation.py:324,330` choose `_HALPE26_PARENT if joint_count > 17 else _COCO17_PARENT`, and the
  triangulation/refine kernels read `joint_count` dynamically (`triangulation.py:406,719,836`;
  `p7_refine/runner.py`). So the lift and refine would not crash on 17 rows, but they would emit a
  17-joint 3D skeleton that every 26-indexed consumer downstream (roles, export, viz, the `HALPE26_*`
  bone/hinge tables) then mis-indexes.

## 4. Verdict for the handover
- **Normal operation is safe:** the Halpe-26 contract is enforced at every stage boundary on write, so a
  17-keypoint P1 output is rejected at the first stage with a `ValueError`, not silently mis-processed.
  This is the intended and correct behaviour — a keypoint mismatch **must not** be run, and the code makes
  sure it isn't.
- **But the failure mode is a hard crash, not a friendly message.** The abort reason can be misleading
  (skeleton-string check trips before the count check; a renamed *Halpe-26* checkpoint whose config path
  lacks the token "halpe" would resolve to `coco_17` and abort a legitimate run — BUG-8).
- **The internals are not uniformly length-guarded.** If the contract were ever bypassed, you'd get a
  mix of crashes (P2 reshape), one ironically-active path (the BUG-1 guard), and silently truncated 3D
  (the lift's COCO-17 branch). The safe invariant holds only because the contract gate is never disabled.

**Recommended hardening (optional, not required for correctness today):** resolve the skeleton from the
model registry rather than a filename token (removes BUG-8's brittleness), and make `resolve_skeleton` /
the contract emit a single explicit "expected halpe26 / 26 keypoints, got <skeleton>/<n>" error so the
crash names the real cause. Neither changes the guarantee — they just make the guaranteed rejection
legible. See `remediation_plan.md` R-T1 items for effort.

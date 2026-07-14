> **TRIAGE CLOSED (2026-07-14).** All confirmed items fixed; G1/G3 implemented flag-gated
> (measurement pending), G6 shipped, A1–A7 unblocked-but-unstarted — live tracking in
> [/remaining-work.md](../../remaining-work.md).

# External Review Triage (2026-07-10)

A full-codebase review (of a slightly older tree) was received mid-campaign. Every finding was
re-verified against the CURRENT code — items marked *confirmed by execution* were reproduced,
not just read. Fixes land as the **FR batch** (review fixes), A/B'd like every other wave and
folded into the final-stack analysis.

## Confirmed — fixing in the FR batch

| ID | Finding | Verification | Fix |
|---|---|---|---|
| C1 | P2 zero-IoU reachability gate is dead code (iou_cost=1.0 can never pass 0.7) — direct fragmentation producer, worst for the sprinting bowler | read: `tracker.py` Stage-2 `c = 1.0 > 0.7` | normalized motion cost replaces the 1.0 when the gate passes; lifecycle test |
| C2 | P2 Kalman `_q` inflates 1.5^n on misses and never resets on normal re-match (57.7 after 10 misses, forever) | **executed**: `_q = 57.7` post-update | reset `_q = 1.0` in `update()`; test covariance returns to baseline |
| C3 | `switch_role` Lyapunov branch unreachable (unit spectral radius → LinAlgError → silent ×4 every time) | **executed**: raises always | inflate stable v/a sub-blocks explicitly to the new model's driving scale |
| C4 | Unknown role string → KeyError at spawn/switch | **executed**: KeyError | `.get(role, params["unknown"])` + config key validation |
| C5 | Documented single-camera ~0.94 m ankle-height fix is not wired to the emit path (single-member clusters still z=0 legacy) | read: `_build_correspondence` single branch | wire `pixel_to_plane_xy` at emit behind the existing opt-in, gate untouched |
| C6 | 3D fill/EMA measure gaps in ROW indices, not real frames — occluded identities glide across long real gaps | read: `sorted(set(frames))` rows | gate fills on true frame numbers; frame-aware EMA/Butterworth segmenting; sparse-gap test |
| H2 | `abs()` in bowler run detection — direction-blind (a fielder sprinting the other way can be crowned bowler) | read: `assigner.py:68` | signed displacement along the known bowling direction |
| H3 | Posture accumulator drops stature for upright-UNKNOWN samples (feet-cut-off players — exactly the height-plane population) | read: `add()` tests `upright` only | drop only when `upright_known and not upright` |
| H5 | NaN confidence poisons the ground Gauss–Newton solve (`max(NaN, eps) = NaN`) | read: valid mask lacks conf | add confidence finiteness to the valid mask |
| H6 | Fragment-attach ambiguity margin is multiplicative — degenerates at distance ≈ 0 | read: `:1429` | additive floor: `max(best*1.5, best+0.5)` |
| H7 | Driver treats P3/P4 crash (rc=1 from an exception) as a warn-verdict and chains on | read | require the stage metrics file to exist before chaining |
| M1 | One `pose_3d` dict aliased into every camera's record (mutation hazard) | read (incl. our F15 emit) | per-view `dict()` copies |
| M2 | `is_finite_number(True) == True` — bools validate as numbers | read | exclude bool |

## Already addressed during this campaign

- **G2 cheirality** — implemented as F3; the sign convention was then *found broken on this
  rig by the Wave-1 A/B* and fixed origin-referenced (see fixes-log W1).
- **G5 splittable clustering** — implemented as F13 (purity-driven surgical eviction), a
  stronger split signal (3D torso residuals) than the LLR bipartition the review sketches.
- **§6 P4 role feedback** — implemented as F5 (online role proxy → `propose_role`).
- **G7-adjacent stitch keys** — F6 occupancy license + F12 posture key extend the stitcher.

## Deferred with rationale (tracked in to-do.md)

- **H1** (P5 fused-track direction inference): real, but P5 roles are cosmetic to the current
  identity goal; folded into Wave 6 (role-focused output shaping) where roles become load-bearing.
- **H4** (TemporalLinkMemory decay): per-frame P3 mode only; the default is tracklet_graph.
- **P1–P6 performance items**: P3 appearance decode threading is the only one on the critical
  path (~1–2 min/delivery); scheduled opportunistically. `observe_frame` vectorization deferred.
- **G1** (Hartley normalization), **G3** (parallax-ordered RANSAC pairs): sound accuracy items;
  queued behind the FR batch as flag-gated candidates.
- **G4** (LLR inter-cue correlation), **G6** (motion-LLR constants → config): noted.
- **A1–A7** (library extraction, config dedup, god-config split): maintainability refactors —
  deliberately after the campaign stabilizes the flag set, to avoid churning files mid-A/B.
- **H8** — done: `role-detection.py` moved to `archive/` (nothing imported it).

## Test additions shipping with the FR batch

P2 lifecycle (occlusion survival + covariance recovery + zero-IoU fast mover), `switch_role`
finite/role-dependent covariance, sparse-frame fill gating, role-direction sign.

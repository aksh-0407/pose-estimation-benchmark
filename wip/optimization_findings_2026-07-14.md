# Pipeline optimization findings — 2026-07-14 (L40S)

Goal (user): make every stage (mosaic + P1–P6) as fast as possible on the L40S,
maximise hardware use. HARD RULE (user, reaffirmed): **every speed change must be
byte-identical — zero output difference. Accuracy/results always win over speed.**

## The core truth about "why isn't the hardware maxed"

Measured, not assumed: **every stage is COMPUTE-bound (CPU cores or GPU SMs), never
memory-bound.** 46 GB VRAM / 61 GB RAM are nowhere near the limit, so "use more RAM"
buys nothing. The real multiplier is **DSA — reducing the work per delivery** — which,
unlike adding threads to an already-core-saturated batch, multiplies across all 40.

| Stage | Assumed lever | Measured reality | Real lever |
|---|---|---|---|
| Mosaic | GPU decode + NVENC | GPU decode 3.2× slower; encode overlaps | CPU multiprocess |
| P1 | Bigger batch → VRAM | f/s flat; 1.6 GB of 46 used | data-parallel processes |
| P2–P6 | More RAM | compute-bound; data cached; jobs=7 saturates 8 vCPU | DSA (less work) |

On an 8-vCPU box the CPU batch is capped at 8 cores of throughput — already reached at
`--jobs 7`. More parallelism can't exceed it; only less work (DSA) or more vCPUs can.

## SHIPPED — byte-identical, verified by execution

### 1. P2 incremental medoid cache — 15.5× (the big one)
`src/identity/p2_tracking/track.py` + `config.py`. Sample-profiling showed **86.9% of P2
self-time in `masked_weighted_cosine`**, driven by `Track.gallery_repr()` recomputing
the O(K²) gallery **medoid** (K=`pose_gallery_size`=30) on every hit — even though each
hit only appends one member, so only that member's ~29 distances are new.

Fix (DSA): cache the pairwise cosine matrix keyed by a monotonic per-member seq id
(kept in lockstep with the `deque(maxlen=30)` ring buffer); recompute only the new
member's row → **O(K) per update instead of O(K²)**. Bit-identical by construction —
the cosine is symmetric, values are memoised (not recomputed), row sums add the same
per-pair values in the same member order with the same first-minimum tie-break. Flag
`pose_medoid_incremental` (default on) lets the legacy path be A/B'd.

- **Per camera: 96.3s → 6.0s (16×).** Full delivery (7 cams): **154s → 10s** local,
  **154s → 12s** on box.
- Byte-identical: 7/7 cameras vs baseline (laptop), 7/7 vs box production output.
- 212/212 tests pass.

### 2. P3 `ground_anchored_skeleton` vectorised — bit-identical
`src/identity/common/pose_shape.py` (12% of P3 self-time; also the billboard-posture
path). Replaced the per-joint Python loop (independent ray-plane intersections, no
cross-joint reduction) with batched numpy. Proven byte-identical on **20,000 random
cases** (incl. NaN/edge cases, both ground_xy and foot-pixel paths, 0 mismatches) and
on full-P3 correspondences + predictions (IDENTICAL). 31 tracklet_graph/pose_shape
tests pass.

### Combined single-delivery chain (warm cache): 348s → 111s, FULL CHAIN BYTE-IDENTICAL
| stage | baseline | optimized | note |
|---|---|---|---|
| p1b | 13.7 | ~13 | (stateful OneEuro loop; see below) |
| **p2** | **154.6** | **10.0** | **medoid cache 15.5× (real, no I/O)** |
| p3 | 134.1 | 49.4 | ground_anchored + warm page-cache on appearance decode |
| p3_5 | 24.3 | 20.2 | already W10-PERF batched |
| p4 | 6.3 | 6.3 | irreducible JSON I/O |
| p6_3d | 12.2 | 10.8 | already batched |

(p3's drop is partly warm cache, not all code; p2's is pure algorithm.)

**Definitive production proof:** the full optimized chain, re-run from P1 on the box for
3 deliveries (M1_1_14_1, M1_1_16_1, M2_1_11_1), reproduced the **existing shipped
production outputs byte-for-byte** across all of p2/p3/p4/p6_3d. The optimized code
regenerates the delivered v8.1 data exactly.

## Also delivered (GPU / batch, byte-identical)
- Mosaics: `render_all_mosaics.py` — all 40 in 18.9 min (~21× vs serial). Collected at
  `artifacts/mosaics_all40/` (box + laptop).
- P1: `run_phase1_parallel.py` — data-parallel shards, ~2× (idle-SM head-room). Sweep
  needs `--sweep --grid` TOGETHER (`--grid` alone starts a full run).

## Analysed — NO byte-identical win available (documented, not forced)
- **P3 appearance cue** (~60% of P3: decode-wait 45% + descriptor 16%): d′≈0.09. Disabling
  is the only big P3 lever but it CHANGES OUTPUT (cycle-consistency 0.701→0.687 on one
  delivery). **DROPPED per user — accuracy first.** `extract_appearance_descriptor` is
  already tight cv2 (no safe speedup).
- **P4** (6s): 61% `json.dumps`/`json.loads` of the inter-phase JSONL contract — swapping
  serializers (orjson) changes float formatting → not byte-identical. Irreducible.
- **P3.5 / P6 triangulation**: already batched + bit-identical (W10-PERF); svd/norm are
  inherent DLT/reprojection cost.
- **P1.5 OneEuroFilter** (`smoothing.py`): K×T scalar loop; sequential in t (can't
  vectorise time), independent over the 17 keypoints (could vectorise k) — but it's a
  stateful filter with per-frame conditional skips, so vectorising risks ULP drift, and
  p1b is ~4% of the chain. Flagged as a candidate needing the same 20k-case exact proof
  before shipping; not shipped.
- **P3 `_cluster_lifts`**: called per candidate cluster-pair in union-merge; member sets
  vary per pair, so caching is not cleanly bit-identical. Not shipped.

## For more throughput than 8 cores allow
The CPU batch is core-bound; the only ways past 8-core throughput are (a) more DSA wins
(as above) or (b) more vCPUs — deliveries are independent, so the batch scales ~linearly
with cores (a 16–32 vCPU box would ~2–4× the batch with no code change).

## New/changed files
- `src/identity/p2_tracking/track.py`, `src/identity/p2_tracking/config.py` — P2 medoid cache.
- `src/identity/common/pose_shape.py` — vectorised ground_anchored_skeleton.
- `src/identity/visualization/render_all_mosaics.py`, `src/core/inference/run_phase1_parallel.py` — new launchers.

"""Physics-constrained 3D skeleton refinement (stage 07).

Takes a per-identity whole-clip 3D pose sequence (already triangulated + ID-assigned)
and makes it *physically valid* and *smooth*, addressing the manager's three asks:

  1. physics constraints  -> constant, bilaterally-symmetric bone lengths (no stretched
     limbs) via a forward-kinematics rebuild from the mid-hip root, plus hinge-angle
     clamps (no backward knees / elbows);
  2. hip wobble            -> the root (mid-hip) trajectory is low-pass filtered with a
     lower cutoff than the limbs, so the whole root-relative skeleton stops shaking;
  3. low-confidence points -> joints below a confidence floor are dropped and refilled
     from their temporal neighbours (predict-and-substitute), never trusted raw.

Everything is offline / whole-clip (non-causal zero-phase smoothing is allowed), and
operates purely on 3D positions - it never reads or writes any identity field, so IDs
are untouched by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.keypoints import (
    HALPE26_BONE_LIMITS_M,
    HALPE26_BONES,
    HALPE26_HINGES,
    HALPE26_ROOT_INDEX,
    HALPE26_SYMMETRIC_BONES,
)
from identity.common.triangulation import (
    _HALPE26_PARENT,
    butterworth_smooth,
    fill_from_skeletal_prior,
    fill_occluded_joints,
)


def _median_bone_lengths_for_prior(
    sequence: np.ndarray, parents: dict[int, int]
) -> dict[tuple[int, int], float]:
    """Median (child, parent) bone lengths keyed for ``fill_from_skeletal_prior``."""
    seq = np.asarray(sequence, dtype=float)
    joint_count = seq.shape[1] if seq.ndim == 3 else 0
    out: dict[tuple[int, int], float] = {}
    for child, parent in parents.items():
        if child >= joint_count or parent >= joint_count:
            continue
        lengths = [
            float(np.linalg.norm(seq[t, child] - seq[t, parent]))
            for t in range(seq.shape[0])
            if _finite(seq[t, child]) and _finite(seq[t, parent])
        ]
        if lengths:
            out[(child, parent)] = float(np.median(lengths))
    return out


# Bone groups smoothed harder than the main limbs, because they sit at the ends of the
# kinematic chains where 2D noise accumulates and short bones amplify angular jitter:
#   face  -> near-rigid head cap (heaviest);
#   foot  -> ankle->toe/heel tips (heavy; the foot barely articulates at these points);
#   mid   -> forearm (elbow->wrist) + shank (knee->ankle): moderate, so a swing/stride is
#            de-noised without being lagged.
_FACE_BONES = {(18, 17), (17, 0), (0, 1), (0, 2), (1, 3), (2, 4)}
_FOOT_BONES = {(15, 20), (15, 22), (15, 24), (16, 21), (16, 23), (16, 25)}
_MID_BONES = {(7, 9), (8, 10), (13, 15), (14, 16)}


@dataclass(frozen=True)
class RefineParams:
    """Tunable knobs for :func:`refine_identity_sequence` (see configs/07_refine.yaml)."""

    enabled: bool = True
    relift: bool = True             # visibility-aware re-lift (needs calibration / drive-root)
    vis_conf: float = 0.5           # a view "reliably sees" a joint iff its 2D conf >= this
    edge_margin_px: float = 4.0
    conf_floor: float = 0.5
    max_gap_frames: int = 25
    fps: float = 50.0
    smoother: str = "butterworth"   # "butterworth" (zero-phase, needs scipy) or "moving_average"
    root_cutoff_hz: float = 3.0
    limb_cutoff_hz: float = 6.0
    filter_order: int = 4
    ma_root_window: int = 9         # moving-average windows (odd -> centred/zero-phase)
    ma_limb_window: int = 5         # 5 = last year's proven window
    face_window: int = 21           # face bones (nose/eyes/ears/head) are ~rigid + tiny ->
                                    # smooth their direction hard to kill facial jitter
    face_cutoff_hz: float = 1.5     # butterworth equivalent for the face group
    foot_window: int = 15           # ankle->toe/heel tips: heavy (foot ~rigid, very noisy 2D)
    foot_cutoff_hz: float = 2.5
    mid_window: int = 9             # forearm + shank: moderate -> de-noise wrist/ankle w/o lag
    mid_cutoff_hz: float = 4.5
    dev_tol: float = 0.25
    clamp_angles: bool = True
    min_hinge_deg: float = 15.0
    max_hinge_deg: float = 178.0
    bones: tuple = field(default=tuple(HALPE26_BONES))
    symmetric_bones: tuple = field(default=tuple(tuple(pair) for pair in HALPE26_SYMMETRIC_BONES))
    hinges: tuple = field(default=tuple(HALPE26_HINGES))
    root_index: int = HALPE26_ROOT_INDEX


def _finite(point: np.ndarray) -> bool:
    return bool(np.isfinite(point).all())


def estimate_canonical_bones(
    sequence: np.ndarray,
    bones: list[tuple[int, int]],
    symmetric_pairs: list[tuple[tuple[int, int], tuple[int, int]]],
    *,
    limits: dict[tuple[int, int], tuple[float, float, float]] = HALPE26_BONE_LIMITS_M,
    min_samples: int = 8,
) -> dict[tuple[int, int], float]:
    """Per-player constant, symmetric, ANATOMICALLY-BOUNDED bone lengths.

    A bone's length is the median of ``|child - parent|`` over the clip; left/right pairs
    are pooled so the skeleton is bilaterally symmetric. The result is then clamped to the
    absolute human range (``limits``): a median outside its range is a triangulation
    artefact (bad / chimera identity), not a real limb, so it is capped instead of being
    locked in - the emitted skeleton can never exceed physics. Every bone always gets a
    length (its anatomical ``default`` when there are no reliable samples), so a full
    skeleton can always be rebuilt.
    """

    seq = np.asarray(sequence, dtype=float)
    samples: dict[tuple[int, int], list[float]] = {}
    for parent, child in bones:
        vals = [
            float(np.linalg.norm(seq[t, child] - seq[t, parent]))
            for t in range(seq.shape[0])
            if _finite(seq[t, child]) and _finite(seq[t, parent])
        ]
        if vals:
            samples[(parent, child)] = vals

    # Raw per-bone medians with left/right pooling.
    raw: dict[tuple[int, int], float] = {}
    pooled: set[tuple[int, int]] = set()
    for left, right in symmetric_pairs:
        combined = samples.get(left, []) + samples.get(right, [])
        if combined:
            length = float(np.median(combined))
            raw[left] = raw[right] = length
            pooled.update((left, right))
    for bone, vals in samples.items():
        if bone not in pooled:
            raw[bone] = float(np.median(vals))

    # Anatomical clamp + default-fill so EVERY bone is human-sized.
    lengths: dict[tuple[int, int], float] = {}
    for bone in bones:
        lo, hi, default = limits.get(bone, (0.0, np.inf, raw.get(bone, 0.0)))
        value = raw.get(bone)
        lengths[bone] = default if value is None else float(np.clip(value, lo, hi))
    return lengths


def enforce_bone_lengths(
    points: np.ndarray,
    bones: list[tuple[int, int]],
    target: dict[tuple[int, int], float],
    *,
    root_index: int,
) -> np.ndarray:
    """Rebuild one frame from the root outward with fixed bone lengths.

    Keeps each measured bone *direction* but overrides its *length* with the canonical
    value, so a wildly mis-triangulated joint is pulled back onto a physical skeleton.
    ``bones`` is in breadth-first order, so the parent of every bone is already placed
    when we reach it. A joint whose parent or direction is unavailable is left NaN.
    """

    pts = np.asarray(points, dtype=float)
    out = np.full_like(pts, np.nan)
    if root_index < out.shape[0] and _finite(pts[root_index]):
        out[root_index] = pts[root_index]
    for parent, child in bones:
        length = target.get((parent, child))
        if length is None or not _finite(out[parent]) or not _finite(pts[child]) or not _finite(pts[parent]):
            continue
        direction = pts[child] - pts[parent]
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            continue
        out[child] = out[parent] + direction / norm * length
    return out


def build_subtrees(bones: list[tuple[int, int]], joint_count: int) -> dict[int, list[int]]:
    """For each joint, the list of joint indices in the subtree rooted at it (incl. itself).

    Used so a hinge clamp rotates a joint's whole descendant chain rigidly (e.g. rotating
    the ankle also carries the foot), which keeps every downstream bone length intact.
    """

    children: dict[int, list[int]] = {}
    for parent, child in bones:
        children.setdefault(parent, []).append(child)

    def descendants(root: int) -> list[int]:
        out = [root]
        stack = list(children.get(root, []))
        while stack:
            node = stack.pop()
            out.append(node)
            stack.extend(children.get(node, []))
        return out

    return {j: descendants(j) for j in range(joint_count)}


def clamp_joint_angles(
    points: np.ndarray,
    hinges: list[tuple[int, int, int]],
    subtrees: dict[int, list[int]],
    *,
    min_deg: float,
    max_deg: float,
) -> np.ndarray:
    """Clamp hinge (knee/elbow) flexion into an anatomical range, preserving bone length.

    The angle at ``joint`` between (proximal-joint) and (distal-joint) is rotated toward
    the nearest limit in the plane the three points span. The distal joint's entire
    subtree is rotated rigidly about the joint, so ``|distal - joint|`` and every
    downstream bone (e.g. the foot below a clamped ankle) keep their length. Stops the
    visibly broken poses where a shin folds through the thigh or a limb over-folds.
    """

    out = np.asarray(points, dtype=float).copy()
    lo = np.deg2rad(min_deg)
    hi = np.deg2rad(max_deg)
    for proximal, joint, distal in hinges:
        j, a, c = out[joint], out[proximal], out[distal]
        if not (_finite(j) and _finite(a) and _finite(c)):
            continue
        u = a - j
        v = c - j
        len_v = float(np.linalg.norm(v))
        nu = float(np.linalg.norm(u))
        if len_v < 1e-6 or nu < 1e-6:
            continue
        u_hat = u / nu
        cos_theta = float(np.clip(np.dot(u_hat, v / len_v), -1.0, 1.0))
        theta = float(np.arccos(cos_theta))
        target = min(max(theta, lo), hi)
        if abs(target - theta) < 1e-6:
            continue
        axis = np.cross(u_hat, v)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-9:
            continue  # collinear: no unique rotation plane, leave as-is
        k = axis / axis_norm
        delta = target - theta
        cos_d, sin_d = np.cos(delta), np.sin(delta)
        # Rotate the distal subtree rigidly about `joint` (Rodrigues), so all downstream
        # bone lengths are preserved, not just the joint->distal bone.
        for idx in subtrees.get(distal, [distal]):
            if not _finite(out[idx]):
                continue
            w = out[idx] - j
            out[idx] = j + (w * cos_d + np.cross(k, w) * sin_d + k * np.dot(k, w) * (1.0 - cos_d))
    return out


def _decompose(sequence: np.ndarray, bones: list[tuple[int, int]], root_index: int):
    """Split a (T, J, 3) sequence into a root trajectory + per-bone unit directions.

    Returns ``(root (T,3), dirs (T,B,3))``; a bone direction is NaN when either endpoint
    is missing that frame. Smoothing the root and the directions separately (then
    re-composing) keeps bone lengths exactly constant through the filter.
    """

    seq = np.asarray(sequence, dtype=float)
    frames = seq.shape[0]
    root = seq[:, root_index, :].copy()
    dirs = np.full((frames, len(bones), 3), np.nan, dtype=float)
    for b, (parent, child) in enumerate(bones):
        vec = seq[:, child, :] - seq[:, parent, :]
        norm = np.linalg.norm(vec, axis=1)
        ok = np.isfinite(norm) & (norm > 1e-6)
        dirs[ok, b] = vec[ok] / norm[ok, None]
    return root, dirs


def _compose(
    root: np.ndarray,
    dirs: np.ndarray,
    bones: list[tuple[int, int]],
    target: dict[tuple[int, int], float],
    joint_count: int,
    root_index: int,
) -> np.ndarray:
    """Forward-kinematics rebuild: (root, per-bone dirs, fixed lengths) -> (T, J, 3)."""

    frames = root.shape[0]
    out = np.full((frames, joint_count, 3), np.nan, dtype=float)
    for t in range(frames):
        if not _finite(root[t]):
            continue
        out[t, root_index] = root[t]
        for b, (parent, child) in enumerate(bones):
            length = target.get((parent, child))
            direction = dirs[t, b]
            if length is None or not _finite(out[t, parent]) or not _finite(direction):
                continue
            out[t, child] = out[t, parent] + direction * length
    return out


def moving_average_smooth(sequence_xyz: np.ndarray, window: int) -> np.ndarray:
    """Zero-phase centred moving average over a (T, J, 3) sequence (scipy-free).

    Last year's proven smoother (a 5-frame moving average). Applied per joint per axis
    over each contiguous finite segment; NaN gaps are preserved, never bridged. A centred
    window is symmetric, so it introduces no phase lag - the offline-quality property we
    want. Segments shorter than the window are left untouched.
    """

    seq = np.asarray(sequence_xyz, dtype=float)
    if seq.ndim != 3 or seq.shape[2] != 3:
        raise ValueError("sequence_xyz must have shape (T, J, 3)")
    window = max(1, int(window))
    if window == 1:
        return seq.copy()
    half = window // 2
    out = seq.copy()
    frames, joints, _ = seq.shape
    for joint in range(joints):
        finite = np.isfinite(seq[:, joint]).all(axis=1)
        start = None
        for t in range(frames + 1):
            inside = t < frames and finite[t]
            if inside and start is None:
                start = t
            elif not inside and start is not None:
                seg = seq[start:t, joint]
                if seg.shape[0] >= window:
                    padded = np.pad(seg, ((half, half), (0, 0)), mode="edge")
                    kernel = np.ones(window) / window
                    for axis in range(3):
                        out[start:t, joint, axis] = np.convolve(
                            padded[:, axis], kernel, mode="valid"
                        )
                start = None
    return out


def fk_smooth(sequence: np.ndarray, bones, target, *, params: "RefineParams") -> np.ndarray:
    """Bone-length-preserving zero-phase smoothing.

    Smooth the root trajectory (heavier -> steady hips) and each bone's unit-direction
    channel (lighter -> genuine limb motion) independently, renormalize the directions,
    then rebuild with the fixed canonical lengths. The result is *exactly* constant bone
    length AND smooth - a plain xyz low-pass cannot achieve both, because filtering joint
    positions re-breaks the lengths.
    """

    joint_count = sequence.shape[1]
    root, dirs = _decompose(sequence, bones, params.root_index)

    if params.smoother == "moving_average":
        root_s = moving_average_smooth(root[:, None, :], params.ma_root_window)[:, 0, :]
        dirs_s = moving_average_smooth(dirs, params.ma_limb_window)
    else:
        root_s = butterworth_smooth(
            root[:, None, :], fps=params.fps, cutoff_hz=params.root_cutoff_hz, order=params.filter_order
        )[:, 0, :]
        dirs_s = butterworth_smooth(
            dirs, fps=params.fps, cutoff_hz=params.limb_cutoff_hz, order=params.filter_order
        )

    # Extra smoothing for the chain-end bone groups (face/foot/mid). A short extremity bone
    # turns small direction noise into large visible jitter, so each group is low-passed
    # harder than the main limbs - heaviest for the ~rigid face and foot, moderate for the
    # forearm/shank so a real swing or stride is de-noised but not lagged.
    for group, window, cutoff in (
        (_FACE_BONES, params.face_window, params.face_cutoff_hz),
        (_FOOT_BONES, params.foot_window, params.foot_cutoff_hz),
        (_MID_BONES, params.mid_window, params.mid_cutoff_hz),
    ):
        cols = [i for i, b in enumerate(bones) if tuple(b) in group]
        if not cols:
            continue
        block = dirs_s[:, cols, :]
        if params.smoother == "moving_average":
            block = moving_average_smooth(block, window)
        else:
            block = butterworth_smooth(block, fps=params.fps, cutoff_hz=cutoff, order=params.filter_order)
        dirs_s[:, cols, :] = block

    norms = np.linalg.norm(dirs_s, axis=2)
    ok = np.isfinite(norms) & (norms > 1e-9)
    dirs_s[ok] = dirs_s[ok] / norms[ok, None]

    return _compose(root_s, dirs_s, bones, target, joint_count, params.root_index)


def refine_identity_sequence(
    sequence: np.ndarray,
    confidences: np.ndarray,
    params: RefineParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Refine one identity's whole-clip (T, J, 3) 3D pose. Returns (points, confidences).

    Steps: low-confidence gate -> temporal/prior fill -> canonical rigid bones ->
    bone-length-preserving smoothing -> hinge-angle clamp. Frames whose root cannot be
    placed are left as the (filled) input so nothing is silently dropped.
    """

    seq = np.asarray(sequence, dtype=float).copy()
    conf = np.asarray(confidences, dtype=float).copy()
    frames, joint_count, _ = seq.shape
    bones = [tuple(b) for b in params.bones]
    parents = _HALPE26_PARENT if joint_count > 17 else None

    # (3) low-confidence gate: drop joints we do not trust before they poison bones.
    gated = np.isfinite(conf) & (conf < params.conf_floor)
    seq[gated] = np.nan

    # (3) predict-and-substitute: interpolate short gaps, then a last-resort skeletal prior.
    seq, conf = fill_occluded_joints(seq, conf, max_gap_frames=params.max_gap_frames)
    prior_parents = parents if parents is not None else {c: p for c, p in _HALPE26_PARENT.items() if c < 17 and p < 17}
    bone_lengths_for_prior = _median_bone_lengths_for_prior(seq, prior_parents)
    reference = _most_complete(seq)
    for t in range(frames):
        if not np.isfinite(seq[t]).all():
            seq[t], conf[t] = fill_from_skeletal_prior(
                seq[t], conf[t], bone_lengths_for_prior, reference, parents=parents
            )

    # (1) canonical rigid, symmetric skeleton estimated once for the whole clip.
    target = estimate_canonical_bones(seq, bones, [tuple(p) for p in params.symmetric_bones])

    # (1 + 2) bone-length-preserving smoothing (steady root, smooth limbs, fixed lengths).
    smoothed = fk_smooth(seq, bones, target, params=params)
    # Frames the FK rebuild could not place (no root) fall back to the filled input.
    for t in range(frames):
        if not np.isfinite(smoothed[t, params.root_index]).all():
            smoothed[t] = seq[t]

    # (1) anatomical hinge-angle clamp (no backward knees/elbows).
    if params.clamp_angles:
        hinges = [tuple(h) for h in params.hinges]
        subtrees = build_subtrees(bones, joint_count)
        for t in range(frames):
            smoothed[t] = clamp_joint_angles(
                smoothed[t], hinges, subtrees,
                min_deg=params.min_hinge_deg, max_deg=params.max_hinge_deg,
            )

    return smoothed, conf


def _most_complete(sequence: np.ndarray) -> np.ndarray:
    """The frame with the most finite joints - the skeletal-prior reference pose."""
    seq = np.asarray(sequence, dtype=float)
    if seq.shape[0] == 0:
        return np.full((seq.shape[1], 3), np.nan)
    counts = [int(np.isfinite(seq[t]).all(axis=1).sum()) for t in range(seq.shape[0])]
    return seq[int(np.argmax(counts))]


# --------------------------------------------------------------------------- metrics


def bone_length_cv(sequence: np.ndarray, bones: list[tuple[int, int]]) -> dict[str, float]:
    """Coefficient of variation of bone lengths over the clip (max + mean across bones).

    ~0 after refinement is the proof that limbs no longer stretch.
    """

    seq = np.asarray(sequence, dtype=float)
    cvs: list[float] = []
    for parent, child in bones:
        lengths = []
        for t in range(seq.shape[0]):
            c, p = seq[t, child], seq[t, parent]
            if _finite(c) and _finite(p):
                lengths.append(float(np.linalg.norm(c - p)))
        if len(lengths) >= 2:
            mean = float(np.mean(lengths))
            if mean > 1e-6:
                cvs.append(float(np.std(lengths) / mean))
    if not cvs:
        return {"max_bone_cv": 0.0, "mean_bone_cv": 0.0, "bones_over_0p05_frac": 0.0}
    arr = np.asarray(cvs)
    return {
        "max_bone_cv": float(arr.max()),
        "mean_bone_cv": float(arr.mean()),
        "bones_over_0p05_frac": float((arr > 0.05).mean()),
    }


def jitter_stats(sequence: np.ndarray, joint_indices: list[int] | None = None) -> dict[str, float]:
    """Mean / p95 frame-to-frame displacement (metres) over finite joint pairs."""

    seq = np.asarray(sequence, dtype=float)
    if joint_indices is not None:
        seq = seq[:, joint_indices, :]
    diffs: list[float] = []
    for t in range(1, seq.shape[0]):
        step = seq[t] - seq[t - 1]
        norms = np.linalg.norm(step, axis=1)
        ok = np.isfinite(norms)
        diffs.extend(norms[ok].tolist())
    if not diffs:
        return {"mean_m": 0.0, "p95_m": 0.0}
    arr = np.asarray(diffs)
    return {"mean_m": float(arr.mean()), "p95_m": float(np.percentile(arr, 95))}

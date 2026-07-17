"""Role assignment (P5) - heuristic v0, designed to be replaced piecewise.

Contract: :func:`assign_roles` maps each global id's fused ground trajectory to a
role from the P4 taxonomy (bowler, striker, non_striker, wicketkeeper, umpire,
fielder, unknown) with an honest confidence and a ``source`` tag. Downstream
consumers (mosaic roster, Groups 2/3 handoff) read only the returned mapping, so
the logic here can be upgraded rule-by-rule without touching any caller.

Heuristics in v0 (all positional/kinematic, no appearance, no ball data):

* **bowler** - the fastest sustained early run along the pitch axis (the run-up;
  the one signal in this family that is near-unambiguous).
* **wicketkeeper** - persistently BEHIND the striker's-end stumps, on the pitch
  line, low average speed.
* **umpire** - persistently behind the bowling-end stumps on the pitch line.
* **striker / non_striker** - nearest long-lived ids to the striker's-end and
  bowling-end creases respectively that are not already claimed above.
* **fielder** - everyone else with enough track to judge; short-lived ids stay
  ``unknown``.

Known limits (v0): no leg umpire (falls out as fielder), batsmen crossing ends
mid-delivery keep their initial label, and everything degrades to ``unknown``
when the bowling direction cannot be inferred.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

STUMPS_FROM_CENTRE_M = 10.06  # stump line distance from the pitch centre

ROLES = ("bowler", "striker", "non_striker", "wicketkeeper", "umpire", "fielder", "unknown")


@dataclass(frozen=True)
class RoleAssignment:
    role: str
    confidence: float
    source: str

    def to_json(self) -> dict:
        return {"role": self.role, "confidence": round(self.confidence, 3), "source": self.source}


def _windowed_axis_speed(
    series: list[tuple[int, np.ndarray]],
    axis: np.ndarray,
    *,
    window_frames: int,
    frame_rate_fps: float,
    early_cutoff: float,
) -> float:
    """Best axis-projected speed over ~window-sized spans early in the clip."""

    ordered = sorted(series, key=lambda item: item[0])
    best = 0.0
    for i, (frame_a, point_a) in enumerate(ordered):
        if frame_a > early_cutoff:
            break
        for frame_b, point_b in ordered[i + 1:]:
            gap = frame_b - frame_a
            if gap > window_frames:
                break
            if gap < window_frames // 2:
                continue
            # The bowler runs WITH the bowling direction; abs() let any
            # fast axis-aligned sprint (a fielder, a runner) win the crown.
            along = float((point_b - point_a) @ axis)
            best = max(best, along * frame_rate_fps / gap)
    return best


def assign_roles(
    per_id_series: dict[str, list[tuple[int, np.ndarray]]],
    bowling_direction: np.ndarray | None,
    *,
    frame_rate_fps: float = 50.0,
    min_track_frames: int = 60,
    bowler_min_speed_mps: float = 3.5,
    pitch_halfwidth_m: float = 2.5,
) -> dict[str, RoleAssignment]:
    """Assign a role to every global id (see module docstring for the rules)."""

    unknown = RoleAssignment("unknown", 0.0, "heuristic_v0")
    roles: dict[str, RoleAssignment] = {pid: unknown for pid in per_id_series}
    if bowling_direction is None:
        return roles
    axis = np.asarray(bowling_direction, dtype=float)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
    lateral = np.array([-axis[1], axis[0]])

    all_frames = [f for s in per_id_series.values() for f, _ in s]
    if not all_frames:
        return roles
    early_cutoff = min(all_frames) + 0.5 * (max(all_frames) - min(all_frames))

    stats: dict[str, dict] = {}
    for pid, series in per_id_series.items():
        if len(series) < min_track_frames:
            continue
        points = np.asarray([p for _, p in series])
        along = points @ axis          # + = toward the striker's end
        across = points @ lateral
        stats[pid] = {
            "frames": len(series),
            "median_along": float(np.median(along)),
            "median_across": float(np.median(across)),
            "run_speed": _windowed_axis_speed(
                series, axis, window_frames=50,
                frame_rate_fps=frame_rate_fps, early_cutoff=early_cutoff,
            ),
        }

    claimed: set[str] = set()

    def claim(pid: str, role: str, confidence: float) -> None:
        roles[pid] = RoleAssignment(role, confidence, "heuristic_v0")
        claimed.add(pid)

    # Bowler: fastest sustained early run down the pitch axis.
    runners = sorted(stats.items(), key=lambda kv: -kv[1]["run_speed"])
    if runners and runners[0][1]["run_speed"] >= bowler_min_speed_mps:
        claim(runners[0][0], "bowler", min(0.9, 0.5 + runners[0][1]["run_speed"] / 20.0))

    on_pitch_line = {
        pid: s for pid, s in stats.items()
        if pid not in claimed and abs(s["median_across"]) <= pitch_halfwidth_m
    }
    # Wicketkeeper: behind the striker's-end stumps (along > stump line).
    behind_striker = {p: s for p, s in on_pitch_line.items()
                      if s["median_along"] > STUMPS_FROM_CENTRE_M + 0.3}
    if behind_striker:
        keeper = max(behind_striker, key=lambda p: behind_striker[p]["median_along"])
        claim(keeper, "wicketkeeper", 0.7)
    # Umpire: behind the bowling-end stumps.
    behind_bowler = {p: s for p, s in on_pitch_line.items()
                     if p not in claimed and s["median_along"] < -(STUMPS_FROM_CENTRE_M + 0.3)}
    if behind_bowler:
        umpire = min(behind_bowler, key=lambda p: behind_bowler[p]["median_along"])
        claim(umpire, "umpire", 0.7)
    # Striker: unclaimed id nearest the striker's-end crease, inside the stumps.
    candidates = {p: s for p, s in on_pitch_line.items()
                  if p not in claimed and s["median_along"] <= STUMPS_FROM_CENTRE_M + 0.3}
    if candidates:
        striker = max(candidates, key=lambda p: candidates[p]["median_along"])
        if candidates[striker]["median_along"] > 0:
            claim(striker, "striker", 0.6)
    # Non-striker: unclaimed id nearest the bowling-end crease.
    candidates = {p: s for p, s in on_pitch_line.items() if p not in claimed}
    if candidates:
        non_striker = min(candidates, key=lambda p: candidates[p]["median_along"])
        if candidates[non_striker]["median_along"] < 0:
            claim(non_striker, "non_striker", 0.6)
    # Everyone else with a real track: fielder.
    for pid, s in stats.items():
        if pid not in claimed:
            roles[pid] = RoleAssignment("fielder", 0.4, "heuristic_v0")
    return roles


# Epoch-scored v1 role solver (merged 2026-07-13, defects fixed same day).
# Slots reflect the real per-delivery roster: 1 bowler, 1 striker, 1 non-striker,
# 1 wicketkeeper, 2 umpires (bowler's end + square leg) - everyone else fields.
EPOCH_SLOTS = (
    "bowler",
    "striker",
    "non_striker",
    "wicketkeeper",
    "umpire_bowler_end",
    "umpire_square_leg",
)
SLOT_TO_ROLE = {slot: ("umpire" if slot.startswith("umpire") else slot) for slot in EPOCH_SLOTS}
BATTING_CREASE_M = 8.84  # popping crease from the pitch centre (stumps 10.06 - 1.22)


# ---------------------------------------------------------------------------
# Section B.1: epoch-scored linear_sum_assignment role solve
# ---------------------------------------------------------------------------


def _epoch_bounds(all_frames: list[int], epoch_frames: int) -> list[tuple[int, int]]:
    start, end = min(all_frames), max(all_frames)
    bounds = []
    frame = start
    while frame <= end:
        bounds.append((frame, min(frame + epoch_frames, end)))
        frame += epoch_frames
    return bounds


def _slot_cost(stat: dict, slot: str) -> float:
    """Geometric cost of a track holding a roster slot, in metres-ish units.

    Conventions (bowling axis = +along): striker bats at the +end (crease
    +8.84, stumps +10.06); the bowler delivers from the -end; the keeper stands
    BEHIND the striker's stumps - anywhere from up-at-the-stumps to ~15 m back
    for pace, so being further back is only mildly penalised; the bowler's-end
    umpire stands behind the -end stumps on the line; the square-leg umpire
    stands level with the striker's crease, 12-25 m lateral.
    """

    speed = abs(stat["run_speed"])
    along = stat["median_along"]
    across = stat["median_across"]
    off_line = max(0.0, abs(across) - 2.5) * 4.0
    if slot == "bowler":
        return off_line + max(0.0, 3.5 - speed)
    if slot == "striker":
        return off_line + abs(along - BATTING_CREASE_M)
    if slot == "non_striker":
        return off_line + abs(along + BATTING_CREASE_M)
    if slot == "wicketkeeper":
        behind = along - (STUMPS_FROM_CENTRE_M + 0.5)
        # zero cost anywhere 0.5-8 m behind the stumps; standing back is normal
        return off_line + max(0.0, -behind) + 0.15 * max(0.0, behind - 8.0) + 0.5 * speed
    if slot == "umpire_bowler_end":
        return off_line + abs(along + (STUMPS_FROM_CENTRE_M + 1.0)) + 0.5 * speed
    if slot == "umpire_square_leg":
        return (
            0.5 * abs(along - BATTING_CREASE_M)
            + max(0.0, 12.0 - abs(across))
            + 0.5 * speed
        )
    raise ValueError(f"unscored slot: {slot}")


def _no_axis_fallback(
    per_id_series: dict[str, list[tuple[int, np.ndarray]]],
    frame_rate_fps: float,
    min_track_frames: int,
) -> dict[str, RoleAssignment]:
    """B.1: degraded-confidence relative ranking instead of collapsing to unknown."""

    roles: dict[str, RoleAssignment] = {}
    speeds: dict[str, float] = {}
    for pid, series in per_id_series.items():
        if len(series) < min_track_frames:
            roles[pid] = RoleAssignment("unknown", 0.0, "heuristic_v1_no_axis")
            continue
        points = np.asarray([p for _, p in series])
        frames = np.asarray([f for f, _ in series], dtype=float)
        duration = max(1.0, (frames[-1] - frames[0]) / frame_rate_fps)
        speeds[pid] = float(np.linalg.norm(points[-1] - points[0])) / duration
    if not speeds:
        return roles
    fastest = max(speeds, key=lambda pid: speeds[pid])
    roles[fastest] = RoleAssignment("bowler", 0.35, "heuristic_v1_no_axis")
    for pid in speeds:
        if pid != fastest:
            roles[pid] = RoleAssignment("fielder", 0.25, "heuristic_v1_no_axis")
    return roles


def assign_roles_epoched(
    per_id_series: dict[str, list[tuple[int, np.ndarray]]],
    bowling_direction: np.ndarray | None,
    *,
    frame_rate_fps: float = 50.0,
    min_track_frames: int = 60,
    epoch_frames: int = 40,
    role_epoch_latch_count: int = 3,
    role_assignment_max_cost: float = 8.0,
    return_cost: bool = False,
):
    """Per-epoch linear_sum_assignment over the roster slots + latch debounce (B.1).

    Uniqueness is enforced in two layers: within an epoch by the Hungarian solve
    (one track per slot), and across the delivery by a final greedy resolution on
    accumulated latch strength - a slot is held by at most ONE track and a track
    holds at most ONE slot, so duplicate bowlers/keepers cannot leak through the
    per-epoch latch dictionaries (defect fixed vs the original draft; the two
    umpires are two SLOTS with distinct geometry, not one slot assigned twice).
    """

    all_frames = sorted({f for s in per_id_series.values() for f, _ in s})
    if not all_frames:
        return ({}, float("inf")) if return_cost else {}
    if bowling_direction is None:
        fallback = _no_axis_fallback(per_id_series, frame_rate_fps, min_track_frames)
        return (fallback, float("inf")) if return_cost else fallback

    axis = np.asarray(bowling_direction, dtype=float)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
    lateral = np.array([-axis[1], axis[0]])

    candidate: dict[str, str] = {}
    streak: dict[str, int] = {}
    strength: dict[tuple[str, str], int] = {}  # (pid, slot) -> total latched epochs
    fit_cost_total = 0.0
    fit_epochs = 0

    for epoch_start, epoch_end in _epoch_bounds(all_frames, epoch_frames):
        stats: dict[str, dict] = {}
        for pid, series in per_id_series.items():
            windowed = [(f, p) for f, p in series if f <= epoch_end]
            if len(windowed) < min_track_frames:
                continue
            points = np.asarray([p for _, p in windowed])
            stats[pid] = {
                "median_along": float(np.median(points @ axis)),
                "median_across": float(np.median(points @ lateral)),
                "run_speed": _windowed_axis_speed(
                    windowed, axis, window_frames=50,
                    frame_rate_fps=frame_rate_fps, early_cutoff=epoch_end,
                ),
            }
        if not stats:
            continue
        track_ids = list(stats.keys())
        cost = np.array(
            [[_slot_cost(stats[pid], slot) for pid in track_ids] for slot in EPOCH_SLOTS]
        )
        row_ind, col_ind = linear_sum_assignment(cost)
        this_epoch = {
            track_ids[c]: EPOCH_SLOTS[r]
            for r, c in zip(row_ind, col_ind)
            if cost[r, c] <= role_assignment_max_cost
        }
        # Sign-comparable fit score (v1.2 auto-flip): accepted assignments pay their
        # cost; slots the Hungarian could have filled but landed above the gate pay
        # the max cost, so "accept less" can never fake a better score.
        possible = min(len(EPOCH_SLOTS), len(track_ids))
        accepted_costs = [
            float(cost[r, c]) for r, c in zip(row_ind, col_ind)
            if cost[r, c] <= role_assignment_max_cost
        ]
        fit_cost_total += sum(accepted_costs)
        fit_cost_total += (possible - len(accepted_costs)) * role_assignment_max_cost
        fit_epochs += 1

        for pid, slot in this_epoch.items():
            if candidate.get(pid) == slot:
                streak[pid] = streak.get(pid, 0) + 1
            else:
                candidate[pid] = slot
                streak[pid] = 1
            if streak[pid] >= role_epoch_latch_count:
                strength[(pid, slot)] = strength.get((pid, slot), 0) + 1

    # Final uniqueness resolution: strongest (pid, slot) claims win; one slot per
    # pid, one pid per slot. Deterministic: strength desc, then pid/slot asc.
    current: dict[str, str] = {}
    taken_slots: set[str] = set()
    for (pid, slot), _n in sorted(strength.items(), key=lambda kv: (-kv[1], kv[0])):
        if pid in current or slot in taken_slots:
            continue
        current[pid] = slot
        taken_slots.add(slot)

    roles: dict[str, RoleAssignment] = {}
    for pid, series in per_id_series.items():
        if pid in current:
            roles[pid] = RoleAssignment(SLOT_TO_ROLE[current[pid]], 0.65, "heuristic_v1")
        elif len(series) >= min_track_frames:
            roles[pid] = RoleAssignment("fielder", 0.4, "heuristic_v1")
        else:
            roles[pid] = RoleAssignment("unknown", 0.0, "heuristic_v1")
    if return_cost:
        mean_cost = fit_cost_total / fit_epochs if fit_epochs else float("inf")
        return roles, mean_cost
    return roles

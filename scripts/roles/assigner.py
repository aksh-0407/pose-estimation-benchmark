"""Role assignment (P5) — heuristic v0, designed to be replaced piecewise.

Contract: :func:`assign_roles` maps each global id's fused ground trajectory to a
role from the P4 taxonomy (bowler, striker, non_striker, wicketkeeper, umpire,
fielder, unknown) with an honest confidence and a ``source`` tag. Downstream
consumers (mosaic roster, Groups 2/3 handoff) read only the returned mapping, so
the logic here can be upgraded rule-by-rule without touching any caller.

Heuristics in v0 (all positional/kinematic, no appearance, no ball data):

* **bowler** — the fastest sustained early run along the pitch axis (the run-up;
  the one signal in this family that is near-unambiguous).
* **wicketkeeper** — persistently BEHIND the striker's-end stumps, on the pitch
  line, low average speed.
* **umpire** — persistently behind the bowling-end stumps on the pitch line.
* **striker / non_striker** — nearest long-lived ids to the striker's-end and
  bowling-end creases respectively that are not already claimed above.
* **fielder** — everyone else with enough track to judge; short-lived ids stay
  ``unknown``.

Known limits (v0): no leg umpire (falls out as fielder), batsmen crossing ends
mid-delivery keep their initial label, and everything degrades to ``unknown``
when the bowling direction cannot be inferred.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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
            along = abs(float((point_b - point_a) @ axis))
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

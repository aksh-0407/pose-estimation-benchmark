# wip/ — internal working notes (stays local, not part of the hand-over docs)

All **bugs**, **fix / implementation plans**, and **pre-hand-over cleanup work**. The professional
hand-over documentation is in [`../docs/`](../docs/index.md) (run guides, architecture, per-stage
pipeline, methods log, measured analysis, roadmap).

## Hand-over audit (2026-07-20) — full-codebase pass over every `src/core` + `src/identity` script

- `resolvebugs.md` — the single **OPEN** bug/error/slop register (severity-ranked; each item has a **Fix + Effort**). Already-fixed campaign bugs are excluded. Absorbs the former `docs/audit/bugs.md` and `docs/pipeline/known-bugs.md`.
- `remediation_plan.md` — the **master fix + pre-hand-over implementation plan**: every open issue with fix, effort tier (XS/S/M/L/XL), reasoning, and an execution sequence (Tier 0 de-slop → Tier 1 safety → Tier 2 A/B fixes → Tier 3 algorithm levers).
- `fallback_methods.md` — every fallback path with an ALWAYS-FALLBACK / PRIMARY-RUNS / SILENT-ON-FAILURE verdict + a fix for the bad ones.
- `legacy_code.md` — exhaustive legacy/dead/inert/superseded inventory with per-item SCRAP/RELOCATE/KEEP + effort.
- `methods_inventory.md` — every method on the final-output path + what's flagged on/off, with the reasoning.
- `keypoint_contract_handling.md` — how the code handles a keypoint-count mismatch (a 17-kp model into the Halpe-26 pipeline): the contract rejects it at every stage boundary; the one ironic soft spot.

## Operational / reference

- `handover-notes.md` — box/infra layout, consumer/schema state, and items the user explicitly deferred.
- `meeting-debug-reference.md` — a meeting-ready debug walkthrough of the pipeline + diagnosis.
- `changes.md` — the 2026-07-17 audit change-ledger (what the campaign changed, with verification evidence).

The forward-looking algorithm roadmap and the measured diagnosis live in the hand-over docs
([`../docs/roadmap.md`](../docs/roadmap.md), [`../docs/analysis/`](../docs/analysis/README.md)); the
method and performance history is in [`../docs/methods_log.md`](../docs/methods_log.md) and
[`../docs/reference/performance.md`](../docs/reference/performance.md).

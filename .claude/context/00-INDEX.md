# Session context — read me first (written 2026-07-14, post v8.1 production)

Purpose: full working context for NEW sessions (any account) so work continues without
re-deriving anything. Read this index, then open only what the task needs.

| File | What it holds |
|---|---|
| `01-current-state.md` | Where everything stands right now: stack version, datasets, kept trees, key paths, test count |
| `02-l40s-operations.md` | The remote GPU box: ssh, layout, envs, how to run P1/chain/renders, perf configs |
| `03-campaign-knowledge.md` | Distilled discoveries, verdicts and rejections from the v6→v8.1 campaign (don't re-litigate) |
| `04-conventions.md` | Evaluation standard, git rules, monitoring directive, doc structure |
| `05-active-threads.md` | What's next / in flight: mosaic-VRAM optimization lead, manager reprojection questions, open decisions |
| `06-user-and-machines.md` | User/PS-1 admin, laptop hardware + crash rules, envs, Claude-accounts memory layout |

Authoritative deep references (in-repo): `/remaining-work.md` (full backlog),
`docs/critical-analysis/fixes-log.md` (every A/B + verdict), `docs/runs/` (archived run
panels), `/home/ubuntu/pipetrack_v8/README.md` on the box (production dataset guide).

NOTE: the user rotates 5 Claude accounts, but all their auto-memory dirs for this repo
are hardlinked to ONE file set (verified 2026-07-14) — that memory was audited and
consolidated here. THESE repo files are the cross-account source of truth; keep them
updated as sessions end.

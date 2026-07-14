# User, machines, accounts (consolidated from auto-memory 2026-07-14)

## User / internship (PS-1)
Aksh Shah, BITS Pilani K. K. Birla Goa Campus, B.E. CS, ID 2024A7PS0532G. PS-1 at Quidich
Innovation Labs (sports broadcasting, Mumbai), 8 weeks from 25 May 2026; mid-sem presented
19 June. Faculty in-charge Prof. Tarkeshwar Singh (Maths, BITS Goa); company mentor
Ms. Simone Singh. Project title: "3D Human Pose Estimation for Cricket Broadcasting".
Report: `report/PS1_midsem_report.tex` (+ ref.bib, figures/), built with `latexmk -pdf`,
native TikZ/pgfplots. Hard report constraints: individual report (Aksh only), NEVER state
Aksh is Group 1 leader, no em dashes, separate completed vs planned work.

## Laptop (Acer Predator Helios Neo 16, PHN16-72)
i9-14900HX (Raptor Lake), RTX 4060 **Laptop** 8 GB (sm_89), 16 GB RAM + 23 GB swap/zram,
Ubuntu 22.04, BIOS V1.18, default kernel 6.8 HWE (zabbly 7.0.10 kept as GRUB fallback).
Crash history and the operational rules it produced:
- MCE storms largely fixed by the BIOS update; **thermals are the #1 practical crash
  trigger** (package hits 100 °C under decode+GPU load; a Phase-1 sweep with 32 io-workers
  froze the machine). On the laptop: cap io-workers low (2–4), let it cool; heavy runs
  belong on the L40S (user directive — see 02).
- `/tmp` is wiped by crashes/reboots: do multi-hour work in the repo, sync incrementally.
- PyTorch uses its own bundled pip CUDA wheels, NOT system nvcc — distrust
  `clean_ubuntu.md` (a web-LLM runbook with wrong GPU model and a misconceived
  system-CUDA upgrade).

## Environments
- Pipeline (P1 and P2–P6, both machines): conda `cricket-rtmpose-l`.
- Pytest: `/home/aksh/miniconda3/envs/cricket-yolo26x-pose/bin/python -m pytest -q` with
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=""`.

## Claude accounts + memory layout
The user rotates 5 accounts (aksh0407marvel, arus, kush, quidich, tony) plus default
`~/.claude`. All six memory dirs for this repo are **hardlinked to one file set** (same
inodes) — there is exactly one auto-memory; editing it from any account edits all.
Consolidated + verified into these `.claude/context/` files on 2026-07-14; the repo files
are the cross-account source of truth (`.claude/` is gitignored, context files are
force-added). Stale memory claims found at verification: the p3 facing-pairs config bug is
FIXED in `configs/v8/`; the RTMPose-L mandate was superseded by approved RTMPose-X;
`scripts/run_cricket_rtmpose_inference.py` no longer exists (use
`scripts/inference/run_phase1_rtmpose_inference.py` / `run_phase1_l40s.py`).

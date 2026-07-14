# Archived benchmark runs

Historical run trees were documented here and their bulk data deleted (2026-07-14 cleanup). Full analytical narrative: docs/critical-analysis/fixes-log.md.

| run | purpose | verdict | analysis pointer |
|---|---|---|---|
| [pipetrack_v8.0](pipetrack_v8.0.md) | ACCEPTED v8.0 default tree (KEPT). | Current default | fixes-log GRAND ANALYSIS v2 |
| [pipetrack_v8.1-w9](pipetrack_v8.1-w9.md) | ACCEPTED v8.1 reference (KEPT): W9 union-lift + colocated merges over v8.0. | Current local reference | fixes-log W9 |
| [_w9_probe](_w9_probe.md) | W9 union-lift iteration probe (_7,_2,_6). | Diagnostic; superseded by v8.1-w9 | fixes-log W9 |
| [_w9_probe2](_w9_probe2.md) | W9 gate-widening probe (_7). | Diagnostic | fixes-log W9 |
| [_w9_probe3](_w9_probe3.md) | W9 rejection-counter probe (_7). | Diagnostic | fixes-log W9 |
| [_w9_id_check](_w9_id_check.md) | W9 flags-off byte-identity tree (M2). | Diagnostic (identity proven) | fixes-log W9 |
| [_p3_prefetch_check](_p3_prefetch_check.md) | P3 appearance-prefetch byte-identity tree (M2). | Diagnostic (identity proven) | fixes-log W10-PERF |

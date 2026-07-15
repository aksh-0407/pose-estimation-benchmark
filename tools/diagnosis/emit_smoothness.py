#!/usr/bin/env python3
"""Measure EMITTED trajectory smoothness (p4 ground_tracks.jsonl = Kalman posteriors)
vs the raw-detection teleport proxy, across all 40 deliveries."""
import json, os, glob, math
import numpy as np

ROOT = os.environ.get("DELIVERIES_ROOT", "data/derived/40_full/pipetrack_v8/deliveries")
FPS = 50.0

def load_gt(path):
    frames = []
    per_id = {}
    for line in open(path):
        row = json.loads(line)
        fi = int(row["frame_index"])
        frames.append(fi)
        for t in row.get("tracks", []):
            gid = t["global_player_id"]
            xy = t.get("ground_xy")
            if xy is None or not all(map(np.isfinite, xy)):
                continue
            per_id.setdefault(gid, {})[fi] = np.asarray(xy, float)
    return sorted(set(frames)), per_id

rows = []
for d in sorted(os.listdir(ROOT)):
    gtp = os.path.join(ROOT, d, "p4/diagnostics/ground_tracks.jsonl")
    mp = os.path.join(ROOT, d, "p4/global_id_metrics.json")
    if not os.path.exists(gtp):
        continue
    frames, per_id = load_gt(gtp)
    nfr = len(frames)
    # emitted per-id consecutive-frame displacement (only consecutive real frames)
    jumps = []          # per-frame speed (m/s) between consecutive present frames (dt=1 frame)
    big_jumps_consec = 0  # >25 m/s across a SINGLE frame step (true emitted teleport)
    gaps = 0            # id absent then reappears (re-acquisition, the proxy double-count)
    id_spans = []
    for gid, fmap in per_id.items():
        fs = sorted(fmap)
        if len(fs) < 2:
            id_spans.append((gid, len(fs), 1))
            continue
        present = set(fs)
        span = fs[-1] - fs[0] + 1
        id_spans.append((gid, len(fs), span))
        for a, b in zip(fs, fs[1:]):
            dt = (b - a) / FPS
            dist = float(np.linalg.norm(fmap[b] - fmap[a]))
            speed = dist / dt if dt > 0 else 0.0
            if b - a == 1:
                jumps.append(speed)
                if speed > 25.0:
                    big_jumps_consec += 1
            else:
                gaps += 1  # discontinuity = re-acquisition after occlusion/loss
    jumps = np.asarray(jumps) if jumps else np.asarray([0.0])
    # per-frame distinct id count
    per_frame_ids = {}
    for gid, fmap in per_id.items():
        for fi in fmap:
            per_frame_ids[fi] = per_frame_ids.get(fi, 0) + 1
    idcounts = np.asarray(list(per_frame_ids.values())) if per_frame_ids else np.asarray([0])
    m = json.load(open(mp)) if os.path.exists(mp) else {}
    tele = m.get("teleport_event_count")
    verdict = m.get("quality_verdict", {}).get("verdict")
    agr = m.get("cross_camera_agreement_rate")
    # fragmentation: ids whose coverage (present/span) is low
    frag_ids = sum(1 for _, npres, span in id_spans if span > 1 and npres/span < 0.6)
    rows.append(dict(
        d=d, nfr=nfr, ids=len(per_id),
        emit_p50=float(np.percentile(jumps,50)),
        emit_p95=float(np.percentile(jumps,95)),
        emit_p99=float(np.percentile(jumps,99)),
        emit_max=float(jumps.max()),
        emit_bigjump=int(big_jumps_consec),
        emit_gaps=int(gaps),
        frag_ids=frag_ids,
        idc_med=float(np.median(idcounts)), idc_max=int(idcounts.max()),
        proxy_tele=tele, agr=agr, verdict=verdict,
    ))

# print table
hdr = ["delivery","nfr","ids","e_p50","e_p95","e_p99","e_max","bigjmp","gaps","frag","idmed","idmax","proxyT","agr","verdict"]
print(" | ".join(hdr))
for r in rows:
    print(" | ".join(str(x) for x in [
        r["d"].replace("CCPL080626",""), r["nfr"], r["ids"],
        f'{r["emit_p50"]:.2f}', f'{r["emit_p95"]:.2f}', f'{r["emit_p99"]:.2f}', f'{r["emit_max"]:.1f}',
        r["emit_bigjump"], r["emit_gaps"], r["frag_ids"], f'{r["idc_med"]:.0f}', r["idc_max"],
        r["proxy_tele"], f'{r["agr"]:.3f}' if r["agr"] is not None else "-", r["verdict"]]))

# aggregate
import statistics as st
print("\n=== AGGREGATE ===")
print("emit_p95 speed m/s: min %.2f med %.2f max %.2f" % (
    min(r["emit_p95"] for r in rows), st.median(r["emit_p95"] for r in rows), max(r["emit_p95"] for r in rows)))
print("emit bigjump(>25m/s 1-frame) total:", sum(r["emit_bigjump"] for r in rows))
print("emit gaps (reacquire) total:", sum(r["emit_gaps"] for r in rows))
print("proxy teleport total:", sum(r["proxy_tele"] or 0 for r in rows))

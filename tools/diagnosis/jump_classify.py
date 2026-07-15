#!/usr/bin/env python3
"""Classify emitted big jumps: spike (out-and-back, outlier measurement) vs
step (persistent jump, stitch seam / real ID switch). Cross-ref id_switch_report."""
import json, os
import numpy as np

ROOT = os.environ.get("DELIVERIES_ROOT", "data/derived/40_full/pipetrack_v8/deliveries")
FPS = 50.0
TH = 25.0  # m/s over one frame ~ 0.5 m

def load(path):
    per_id = {}
    for line in open(path):
        row = json.loads(line)
        fi = int(row["frame_index"])
        for t in row.get("tracks", []):
            xy = t.get("ground_xy")
            if xy and all(map(np.isfinite, xy)):
                per_id.setdefault(t["global_player_id"], {})[fi] = np.asarray(xy, float)
    return per_id

targets = ["CCPL080626M2_2_3_2","CCPL080626M2_2_3_7","CCPL080626M2_1_11_7",
           "CCPL080626M1_1_17_1","CCPL080626M2_2_4_1","CCPL080626M1_1_14_1",
           "CCPL080626M1_1_16_4"]
for d in targets:
    per_id = load(os.path.join(ROOT,d,"p4/diagnostics/ground_tracks.jsonl"))
    swr = json.load(open(os.path.join(ROOT,d,"p4/id_switch_report.json")))
    merge_frames = sorted(set(int(e["at_frame"]) for e in swr if "at_frame" in e))
    spikes=steps=0; near_merge=0; step_frames=[]
    maxstep=0.0
    for gid,fmap in per_id.items():
        fs=sorted(fmap)
        for i in range(len(fs)-1):
            a,b=fs[i],fs[i+1]
            if b-a!=1: continue
            dist=float(np.linalg.norm(fmap[b]-fmap[a]))
            if dist/((b-a)/FPS) <= TH: continue
            # spike? next present frame jumps back near a
            if i+2 < len(fs) and fs[i+2]-b==1:
                back=float(np.linalg.norm(fmap[fs[i+2]]-fmap[a]))
                if back < dist*0.5 and back < 2.0:
                    spikes+=1; continue
            steps+=1; step_frames.append((gid,a,b,round(dist,1)))
            maxstep=max(maxstep,dist)
            if any(abs(b-mf)<=3 for mf in merge_frames): near_merge+=1
    print(f"\n== {d.replace('CCPL080626','')} ==  merges={len(merge_frames)} bigjumps: spikes={spikes} steps={steps} steps_near_merge={near_merge} maxstep={maxstep:.1f}m")
    # show worst 6 steps
    for gid,a,b,dist in sorted(step_frames,key=lambda x:-x[3])[:6]:
        nm = any(abs(b-mf)<=3 for mf in merge_frames)
        print(f"   {gid} f{a}->{b} {dist}m {'[near-merge]' if nm else ''}")

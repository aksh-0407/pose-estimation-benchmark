#!/usr/bin/env python3
"""Cross-camera SPLIT identity: same physical person seen by N cameras but carrying
>1 global id. Per frame, single-link cluster detections by ground proximity (<1.5m),
then for each physical cluster spanning >=2 cameras, count distinct global ids and
which camera-pairs disagree. Tally by camera pair to test the facing-pair hypothesis."""
import sys, os, json, glob
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
import numpy as np
from collections import defaultdict
from core.calibration import build_ground_calibrators
from identity.p2_tracking.runner import infer_match_id

# ROOT/DRIVE point at the box's production tree; override for a local run.
ROOT=os.environ.get("DELIVERIES_ROOT", "data/derived/40_full/pipetrack_v8/deliveries")
DRIVE=os.environ.get("DRIVE_ROOT", "data/raw/8_init")
RAD=1.5

def cluster(points):
    # single-link union-find within RAD
    n=len(points); par=list(range(n))
    def find(x):
        while par[x]!=x: par[x]=par[par[x]]; x=par[x]
        return x
    for i in range(n):
        for j in range(i+1,n):
            if np.linalg.norm(points[i][2]-points[j][2])<=RAD:
                par[find(i)]=find(j)
    groups=defaultdict(list)
    for i in range(n): groups[find(i)].append(i)
    return groups.values()

targets=sys.argv[1:] or ["CCPL080626M1_1_14_6","CCPL080626M2_1_11_6","CCPL080626M2_2_3_5",
    "CCPL080626M2_2_3_3","CCPL080626M1_1_16_2","CCPL080626M2_2_3_4","CCPL080626M1_1_14_1"]
pair_split_global=defaultdict(int)
for d in targets:
    match=infer_match_id(d)
    cals=build_ground_calibrators(DRIVE,match)
    # gather per frame: (camera, gid, ground_xy)
    by_frame=defaultdict(list)
    for f in glob.glob(os.path.join(ROOT,d,"p4/predictions","*.jsonl")):
        cam=f.split("__")[-1].replace(".jsonl","")
        cal=cals.get(cam)
        if cal is None: continue
        for line in open(f):
            r=json.loads(line); fi=int(r["frame_index"])
            for p in r.get("players",[]):
                g=p.get("global_player_id"); b=p.get("bbox_xywh_px")
                if not g or not b: continue
                xy=cal.bbox_bottom_center_ground_xy([float(v) for v in b])
                if xy is None or not np.isfinite(xy).all(): continue
                by_frame[fi].append((cam,g,np.asarray(xy,float)))
    split_clusters=0; total_multicam=0; pair_split=defaultdict(int)
    for fi,dets in by_frame.items():
        # dedupe same camera: keep all (a cam can legitimately have 2 people)
        for grp in cluster(dets):
            members=[dets[i] for i in grp]
            cams={m[0] for m in members}
            if len(cams)<2: continue
            total_multicam+=1
            gids={m[1] for m in members}
            if len(gids)>1:
                split_clusters+=1
                # which camera pairs carry different gids
                for a in range(len(members)):
                    for b2 in range(a+1,len(members)):
                        ca,ga,_=members[a]; cb,gb,_=members[b2]
                        if ca!=cb and ga!=gb:
                            key=tuple(sorted((ca,cb)))
                            pair_split[key]+=1; pair_split_global[key]+=1
    rate=split_clusters/max(total_multicam,1)
    top=sorted(pair_split.items(),key=lambda x:-x[1])[:4]
    print(f"{d.replace('CCPL080626',''):12s} split_clusters={split_clusters}/{total_multicam} ({100*rate:.0f}%) top_pairs=" +
          ", ".join(f"{a[3:]}-{b[3:]}:{c}" for (a,b),c in top))

print("\n=== GLOBAL camera-pair split tally (facing pairs = 01-04, 02-06, 03-05) ===")
for (a,b),c in sorted(pair_split_global.items(),key=lambda x:-x[1])[:12]:
    fac = (a[3:],b[3:]) in [("01","04"),("02","06"),("03","05")]
    print(f"  {a}-{b}: {c} {'<-- FACING PAIR' if fac else ''}")

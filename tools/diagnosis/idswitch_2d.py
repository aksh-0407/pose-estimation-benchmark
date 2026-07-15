#!/usr/bin/env python3
"""Visible ID switching: within a per-camera P2 tracklet (local_track_id, stable
per camera for one physical person), how often does global_player_id flip? Each
flip = a colour switch in the mosaic. Also P6 3D coverage per delivery."""
import json, os, glob
import numpy as np

ROOT=os.environ.get("DELIVERIES_ROOT", "data/derived/40_full/pipetrack_v8/deliveries")

rows=[]
for d in sorted(os.listdir(ROOT)):
    dd=os.path.join(ROOT,d)
    if not os.path.isdir(dd): continue
    # 2D id switches within local tracklets, from p4 predictions
    switch_events=0; tracklets=0; multi_id_tracklets=0
    for f in glob.glob(os.path.join(dd,"p4/predictions","*.jsonl")):
        seq={}  # ltid -> list of (frame, gid)
        for line in open(f):
            r=json.loads(line); fi=int(r["frame_index"])
            for p in r.get("players",[]):
                lt=p.get("local_track_id"); g=p.get("global_player_id")
                if lt is None: continue
                seq.setdefault(lt,[]).append((fi,g))
        for lt,vals in seq.items():
            vals.sort()
            gids=[g for _,g in vals if g is not None]
            if not gids: continue
            tracklets+=1
            # count transitions between distinct non-None gids
            trans=0; prev=None
            for g in gids:
                if prev is not None and g!=prev: trans+=1
                prev=g
            switch_events+=trans
            if len(set(gids))>1: multi_id_tracklets+=1
    # P6 coverage
    cov_num=cov_den=0
    for f in glob.glob(os.path.join(dd,"p6_3d/predictions","*.jsonl")):
        for line in open(f):
            r=json.loads(line)
            for p in r.get("players",[]):
                if not p.get("global_player_id"): continue
                cov_den+=1
                p3=p.get("pose_3d")
                if p3 and p3.get("keypoints_world_m"): cov_num+=1
    cov = cov_num/cov_den if cov_den else 0
    rows.append((d.replace("CCPL080626",""),switch_events,tracklets,multi_id_tracklets,round(cov,2)))

print("delivery | 2d_id_switch_events | tracklets | multi_id_tracklets | p6_cov")
for r in rows: print(" | ".join(map(str,r)))
print("\nTOTAL 2d id-switch events across 40:", sum(r[1] for r in rows))
print("mean multi_id_tracklets:", round(np.mean([r[3] for r in rows]),1))
print("p6_cov min/med/max: %.2f %.2f %.2f"%(min(r[4] for r in rows), np.median([r[4] for r in rows]), max(r[4] for r in rows)))

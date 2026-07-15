#!/usr/bin/env python3
"""(A) one global id present in >=2 cameras >5m apart in the same frame = broken merge.
   (B) P6 3D pelvis trajectory smoothness + coverage."""
import json, os, glob
import numpy as np

ROOT=os.environ.get("DELIVERIES_ROOT", "data/derived/40_full/pipetrack_v8/deliveries")
FPS=50.0

def cam_files(d, stage):
    return sorted(glob.glob(os.path.join(ROOT,d,stage,"predictions","*.jsonl")))

def bbox_bottom(b):
    x,y,w,h=b; return np.array([x+w/2, y+h])

# We can't project without calibration here, so use the p4 predictions' pose_3d? No.
# (A) proxy: use ground_tracks is collapsed. Instead detect concurrent 2-camera id
# using p4 predictions + the fact that far-apart same-id in same frame shows as
# large pose_3d disagreement is unavailable. Use single_camera flag + count of
# cameras carrying each id per frame, and flag ids that in the SAME frame are
# 'single_camera' true in two different cameras (means two disjoint solo detections
# under one id -> the concurrent-merge signature).
targets=["CCPL080626M2_2_3_2","CCPL080626M2_1_11_7","CCPL080626M1_1_16_4",
         "CCPL080626M1_1_14_1","CCPL080626M2_2_4_1","CCPL080626M2_2_3_7"]
for d in targets:
    # concurrency from p4 predictions: per (frame, gid) collect cameras
    frame_gid_cams={}
    for f in cam_files(d,"p4"):
        cam=f.split("__")[-1].replace(".jsonl","")
        for line in open(f):
            r=json.loads(line); fi=int(r["frame_index"])
            for p in r.get("players",[]):
                g=p.get("global_player_id")
                if g: frame_gid_cams.setdefault((fi,g),set()).add(cam)
    multi=sum(1 for v in frame_gid_cams.values() if len(v)>=2)
    total=len(frame_gid_cams)
    # (B) 3D pelvis smoothness from p6_3d: hip-mid = mean of coco 11,12
    per_id={}
    for f in cam_files(d,"p6_3d"):
        for line in open(f):
            r=json.loads(line); fi=int(r["frame_index"])
            for p in r.get("players",[]):
                g=p.get("global_player_id"); p3=p.get("pose_3d")
                if not g or not p3: continue
                kp=p3.get("keypoints_world_m")
                if not kp or len(kp)<13: continue
                lh,rh=np.array(kp[11],float),np.array(kp[12],float)
                if not (np.isfinite(lh).all() and np.isfinite(rh).all()): continue
                per_id.setdefault(g,{}).setdefault(fi,[]).append((lh+rh)/2)
    jumps=[]; big=0; nframes3d=0
    for g,fm in per_id.items():
        fs=sorted(fm)
        pos={fi:np.mean(fm[fi],axis=0) for fi in fs}
        nframes3d+=len(fs)
        for a,b in zip(fs,fs[1:]):
            if b-a!=1: continue
            sp=np.linalg.norm(pos[b]-pos[a])/((b-a)/FPS)
            jumps.append(sp)
            if sp>25: big+=1
    jn=np.array(jumps) if jumps else np.array([0.0])
    print(f"{d.replace('CCPL080626',''):12s} | sameframe_multicam_id={multi}/{total} ({100*multi/max(total,1):.0f}%) | 3d_pelvis p95={np.percentile(jn,95):.1f} max={jn.max():.0f} bigjumps={big} | 3d_idframes={nframes3d}")

#!/usr/bin/env python3
"""Re-grade all 40 with a redesigned verdict rubric.
Composite score over the axes that actually reflect delivery usability, plus hard
invariant gates. Replaces the teleport-proxy-driven verdict."""
import json, os, glob
import numpy as np
ROOT=os.environ.get("DELIVERIES_ROOT", "data/derived/40_full/pipetrack_v8/deliveries"); FPS=50.0

def emitted_bigjumps(path):
    per_id={}
    for line in open(path):
        r=json.loads(line); fi=int(r["frame_index"])
        for t in r.get("tracks",[]):
            xy=t.get("ground_xy")
            if xy and all(map(np.isfinite,xy)):
                per_id.setdefault(t["global_player_id"],{})[fi]=np.asarray(xy,float)
    big=0
    for fm in per_id.values():
        fs=sorted(fm)
        for a,b in zip(fs,fs[1:]):
            if b-a==1 and np.linalg.norm(fm[b]-fm[a])/((b-a)/FPS) > 25.0: big+=1
    return big

def get(d,path,*keys):
    try:
        m=json.load(open(os.path.join(ROOT,d,path)))
        for k in keys:
            m=m[k] if not isinstance(k,tuple) else m
        return m
    except Exception: return None

def clamp(x): return max(0.0,min(1.0,x))

# rubric
W={"agr":0.40,"smooth":0.25,"cov":0.15,"persist":0.10,"parsimony":0.10}
def tier(d):
    agr=get(d,"p4/global_id_metrics.json","cross_camera_agreement_rate") or 0
    ids=get(d,"p4/global_id_metrics.json","distinct_global_id_count") or 99
    coll=get(d,"p4/global_id_metrics.json","same_camera_identity_collision_frames") or 0
    coloc=get(d,"p4/global_id_metrics.json","colocated_disjoint_pair_count") or 0
    try: persist=json.load(open(os.path.join(ROOT,d,"p4/global_id_metrics.json")))["completeness"]["confirmed_frame_completeness"]["mean"]
    except Exception: persist=0
    cov=get(d,"p6_3d/triangulation_metrics.json","triangulation_coverage") or 0
    old=None
    try: old=json.load(open(os.path.join(ROOT,d,"p4/global_id_metrics.json")))["quality_verdict"]["verdict"]
    except Exception: pass
    big=emitted_bigjumps(os.path.join(ROOT,d,"p4/diagnostics/ground_tracks.jsonl"))
    s={"agr":clamp((agr-0.72)/0.24),"smooth":clamp(1-big/50.0),"cov":clamp((cov-0.45)/0.45),
       "persist":clamp((persist-0.80)/0.18),"parsimony":clamp((16-ids)/3.0)}
    score=sum(W[k]*s[k] for k in W) - 0.10*coloc
    score=max(0.0,score)
    # hard gates
    gate=None
    if coll>0: gate="collision"
    elif ids>20: gate="id_overmint"
    elif agr<0.65: gate="identity_broken"
    if gate: t="FAIL"
    elif score>=0.75: t="GOOD"
    elif score>=0.55: t="USABLE"
    elif score>=0.40: t="WEAK"
    else: t="FAIL"
    # limiting factor = lowest weighted contributor
    lim=min(s,key=lambda k:s[k])
    return dict(d=d.replace("CCPL080626",""),old=old,agr=agr,big=big,cov=cov,persist=persist,
               ids=ids,coloc=coloc,score=round(score,3),tier=t,gate=gate,lim=lim,s=s)

rows=[tier(d) for d in sorted(os.listdir(ROOT)) if os.path.isdir(os.path.join(ROOT,d))]
print(f"{'delivery':11s} {'OLD':5s} {'agr':5s} {'bigj':4s} {'cov':4s} {'per':4s} {'ids':3s} {'clc':3s} {'score':5s} {'NEW':7s} limit/gate")
for r in rows:
    print(f"{r['d']:11s} {str(r['old']):5s} {r['agr']:.3f} {r['big']:4d} {r['cov']:.2f} {r['persist']:.2f} {r['ids']:3d} {r['coloc']:3d} {r['score']:.3f} {r['tier']:7s} {r['gate'] or r['lim']}")
from collections import Counter
print("\nOLD:",dict(Counter(r['old'] for r in rows)))
print("NEW:",dict(Counter(r['tier'] for r in rows)))

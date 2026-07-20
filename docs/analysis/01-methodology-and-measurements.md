# Methodology and raw measurements

All measurements were taken on the production tree `/home/ubuntu/pipetrack_v8/deliveries/`
on the L40S (`ssh quidich-gpu-intern`), env `cricket-rtmpose-l`. The five scripts live
beside this doc and are reproducible.

| Script | What it measures |
|---|---|
| `emit_smoothness.py` | Per-id consecutive-frame jump distribution in the **emitted** `p4/diagnostics/ground_tracks.jsonl` (Kalman posterior); big-jump count (>25 m/s over 1 frame), re-acquisition gaps, per-frame id counts. Cross-refs `global_id_metrics.json`. |
| `jump_classify.py` | Splits emitted big jumps into **spikes** (out-and-back outlier measurement) vs **steps** (persistent) and cross-refs `id_switch_report.json` merge frames. |
| `occupancy_and_3d.py` | Multi-camera id rate + **P6 3D pelvis** (hip-mid) trajectory smoothness. |
| `idswitch_2d.py` | Visible ID-switch flicker: how often a stable per-camera P2 tracklet flips its `global_player_id`; P6 3D coverage per delivery. |
| `split_identity.py` | Cross-camera split identity: per-frame ground clustering (uses the repo's `build_ground_calibrators`), % of multi-cam clusters with >1 id, camera-pair disagreement tally. |

## Definitions

- **Emitted big jump**: a single-frame (Δframe = 1) displacement of a global id in
  `ground_tracks.jsonl` whose implied speed exceeds 25 m/s (≈ 0.5 m in 20 ms). A sprinting
  cricketer tops out near 9 m/s, so 25 m/s is unambiguously non-physical.
- **Re-acquisition gap**: an id present at frame *a*, absent, then present again at frame
  *b* > *a*+1. Counted separately, this is the acceptable "occlusion restore" class.
- **2D id-switch event**: within one `(camera, local_track_id)` (a P2 tracklet = one
  physical person in one camera), a frame-to-frame change of the non-null `global_player_id`.
- **Split cluster**: a per-frame single-link ground cluster (radius 1.5 m) spanning ≥2
  cameras that carries more than one `global_player_id`. Overstated in dense packs (two real
  people can fall inside 1.5 m), read the camera-pair breakdown, not the absolute rate.

## Table 1, emitted-trajectory smoothness (from `emit_smoothness.py`)

Columns: e_p95/e_max = per-frame speed m/s in the emitted ground track; bigjmp =
single-frame jumps >25 m/s; gaps = re-acquisition gaps; proxyT = the panel's
`teleport_event_count`.

```
delivery   ids e_p50 e_p95 e_p99  e_max  bigjmp gaps proxyT agr    verdict
M1_1_14_1  10  0.19  2.44  6.59   672    23     1    8      0.799  pass
M1_1_14_2  11  0.19  2.93  7.50   310    29     0    21     0.845  warn
M1_1_14_3  13  0.17  2.18  4.52   145    10     34   60     0.889  warn
M1_1_14_4  10  0.17  1.76  4.52   860    27     10   26     0.972  warn
M1_1_14_5  13  0.06  2.21  5.68   589    30     22   26     0.695  warn
M1_1_14_6  15  0.12  2.38  5.68   352    9      21   54     0.527  warn
M1_1_14_7  12  0.26  2.30  5.05   379    19     15   49     0.819  warn
M1_1_16_1  11  0.15  1.98  5.01   547    15     39   64     0.923  fail
M1_1_16_2  12  0.21  2.01  7.58   1187   30     28   71     0.625  fail
M1_1_16_3  10  0.13  1.61  13.9   653    53     21   69     0.892  fail
M1_1_16_4  12  0.14  1.86  4.35   22     0      66   44     0.951  warn
M1_1_16_5  11  0.19  2.34  6.33   683    6      26   22     0.843  warn
M1_1_16_6  11  0.16  1.77  5.47   703    23     42   57     0.975  warn
M1_1_17_1  10  0.22  1.66  7.17   1511   38     73   118    0.954  fail
M1_1_17_2  10  0.18  1.73  4.10   500    5      81   80     0.983  fail
M1_1_17_3  12  0.47  2.49  7.22   281    12     78   208    0.887  fail
M1_1_17_4  13  0.24  1.92  4.06   355    4      94   87     0.862  fail
M1_1_17_5  13  0.15  2.80  8.31   1393   7      73   39     0.971  warn
M1_1_17_6  11  0.11  1.24  2.87   731    12     73   19     0.918  pass
M2_1_11_1  11  0.23  3.35  16.9   538    43     53   230    0.891  fail
M2_1_11_2  12  0.29  3.31  15.6   1379   45     67   165    0.897  fail
M2_1_11_3  12  0.26  2.90  8.20   482    39     59   147    0.839  fail
M2_1_11_4  10  0.27  2.37  5.55   564    15     57   112    0.953  fail
M2_1_11_5  9   0.14  1.74  4.76   802    12     29   60     0.900  warn
M2_1_11_6  12  0.18  2.02  4.80   536    12     50   112    0.836  fail
M2_1_11_7  11  0.48  3.87  82.0   1022   69     56   241    0.832  fail
M2_1_12_1  13  0.46  3.44  7.99   724    40     75   176    0.881  fail
M2_2_3_1   14  0.17  1.76  91.1   930    91     130  198    0.847  fail
M2_2_3_2   13  0.20  4.14  704    1440   170    53   195    0.862  fail
M2_2_3_3   13  0.14  2.78  181    286    140    54   180    0.725  fail
M2_2_3_4   16  0.22  2.93  7.62   404    39     104  193    0.743  fail
M2_2_3_5   16  0.23  3.81  114    1240   106    87   252    0.787  fail
M2_2_3_6   16  0.21  3.32  139    568    105    110  207    0.728  fail
M2_2_3_7   16  0.33  3.02  99.1   1125   102    157  301    0.823  fail
M2_2_4_1   11  0.12  3.15  6.45   417    7      10   158    0.992  fail
M2_2_4_2   11  0.26  3.68  8.89   253    22     38   236    0.949  fail
M2_2_4_3   11  0.27  2.76  5.54   575    15     32   183    0.976  fail
M2_2_4_4   12  0.17  3.28  13.2   350    43     29   217    0.969  fail
M2_2_4_5   14  0.14  2.48  7.95   677    32     40   121    0.937  fail
M2_2_4_6   16  0.12  2.06  6.30   1153   29     47   126    0.783  fail
```

Aggregate: emitted big jumps total **1528**; re-acquisition gaps **2134**; proxy teleports
**4932**. Emitted p95 speed is healthy everywhere (1.24-4.14 m/s), the problem is entirely
in the **tail** (p99/max), i.e. rare catastrophic jumps, not general jitter.

## Table 2, 2D flicker + 3D coverage (from `idswitch_2d.py`)

```
delivery   2d_id_switch  tracklets  multi_id_tracklets  p6_cov
M1_1_14_1  2   26  2   0.91
M1_1_14_4  11  24  5   0.84
M1_1_14_6  18  37  8   0.69
M1_1_16_4  3   35  2   0.91
M1_1_17_4  31  25  3   0.75
M2_1_12_1  28  31  9   0.59
M2_2_3_4   11  29  7   0.48
M2_2_3_7   22  32  12  0.51
M2_2_4_5   26  38  10  0.79
```
(full 40 in the script output). Total 2D id-switch events **517**; mean multi-id tracklets
**5.5/delivery**; P6 coverage min/median/max **0.48 / 0.80 / 0.92**.

## Table 3, camera-pair split tally (from `split_identity.py`, 7-delivery sample)

```
cam_01-cam_04: 5030     <- FACING PAIR, #1
cam_04-cam_05: 4368
cam_03-cam_04: 4115
cam_02-cam_04: 2721
cam_05-cam_07: 2524
cam_04-cam_06: 2432
cam_04-cam_07: 2069
cam_02-cam_06: 1248     <- facing pair
cam_03-cam_05:  769     <- facing pair
```
**cam_04 (end-on) appears in the top of nearly every pair**; cam_07 (panoramic) is the
other repeat offender. Note the metric's own caveat: cam_04's grazing bbox-bottom ground
projection has large depth error, so some of its clustering is spurious, but the recorded
`cross_camera_disagreement_examples` (see `05-...`) confirm the split is real, not just a
projection artifact.

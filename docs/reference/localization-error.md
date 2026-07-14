# Ground-localization error: how many metres is a pixel?

How much world-position error does a 2D localization error of 1 px (and ~12 px) cause,
per camera, on the z = 0 ground plane. Computed directly from the bundle-adjusted
projection matrices (`data/raw/8_init/calibration-data/CCPL080626`) by taking the local
Jacobian of the pixel→ground map (perturb a pixel ±1 px, measure the ground displacement).

## The one thing to know: the error is strongly anisotropic

A camera localizes the **cross-range** (sideways) direction ~7–11× better than the
**down-range** (depth) direction. Reason: the cameras sit ~100 m out but only 10–17 m
high, so rays graze the turf at just **5–9°**, and depth error scales as `range / (f·sinθ)`.
This is exactly why the pipeline triangulates across 7 cameras — one camera's bad depth
axis is another camera's good cross axis.

- **Cross-range:** ~**0.003 m/px** everywhere (≈3 mm).
- **Down-range (the axis that dominates real error):** ~**0.025–0.034 m/px** at a batsman,
  rising to **0.05–0.06 m/px** for a deep fielder viewed at a grazing angle.

## Headline

| Situation (down-range axis) | 1 px | 12 px |
|---|--:|--:|
| Batsman, cross-range (best axis) | 0.003 m | 0.035 m |
| **Batsman, down-range (typical)** | **0.025–0.034 m** | **0.30–0.41 m** |
| Batsman, single effective (area geo-mean) | 0.009 m | 0.11 m |
| Deep fielder, grazing (worst realistic) | 0.05–0.06 m | 0.6–0.75 m |

So **1 px ≈ 3 mm to 6 cm** and **12 px ≈ 3.5 cm to 75 cm**, depending entirely on the axis,
the range, and the grazing angle. The typical operating point (a batsman, depth axis):
**1 px ≈ 3 cm, 12 px ≈ 0.35 m.**

Reality check: calibration reprojection is p95 ≤ 4.5 px, so a realistic *single-camera*
foot-point lands ~**0.13 m** off near / ~**0.24 m** off deep — which is the error the
multi-camera triangulation exists to collapse.

## Per-camera, near point (pitch centre)

| cam | role | slant (m) | grazing° | cross m/px | down-range m/px | 12 px (depth) |
|---|---|--:|--:|--:|--:|--:|
| cam_01 | end-on | 98.5 | 8.2 | 0.0036 | 0.0255 | 0.31 |
| cam_02 | side | 118.0 | 5.6 | 0.0027 | 0.0280 | 0.34 |
| cam_03 | side | 105.3 | 5.9 | 0.0029 | 0.0283 | 0.34 |
| cam_04 | end-on | 109.1 | 8.9 | 0.0035 | 0.0225 | 0.27 |
| cam_05 | side | 113.9 | 5.2 | 0.0027 | 0.0296 | 0.36 |
| cam_06 | side | 128.2 | 5.1 | 0.0030 | 0.0340 | 0.41 |
| cam_07 | panoramic | 102.1 | 5.6 | 0.0033 | 0.0337 | 0.40 |

## Per-camera, deep field (grazing worst cases)

| cam | ground point | slant (m) | grazing° | cross m/px | down-range m/px | 12 px |
|---|---|--:|--:|--:|--:|--:|
| cam_04 | (0, −45) far straight | 153.7 | 6.3 | 0.0049 | 0.0445 | 0.53 |
| cam_01 | (0, +45) far straight | 143.2 | 5.6 | 0.0053 | 0.0539 | 0.65 |
| cam_07 | (0, +45) | 131.8 | 4.3 | 0.0039 | 0.0538 | 0.65 |
| cam_05 | (45, 0) deep square | 159 | 3.7 | 0.0038 | 0.0576 | 0.69 |
| cam_06 | (45, 0) deep square | 173 | 3.8 | 0.0041 | 0.0619 | 0.74 |
| cam_01 | (0, −45) **near** end | 54.4 | 15.0 | 0.0020 | 0.0076 | 0.09 |

The last row is the same end-on camera at half the range and a steeper angle: depth error
drops from 0.054 to 0.008 m/px — the range × grazing effect in one comparison.

*Method:* image→ground homography `H = P[:, [0,1,3]]` (drop the z column), ground point
`= H⁻¹·[u,v,1]ᵀ`; the 2×2 Jacobian's singular values are the down-range (major) and
cross-range (minor) metres-per-pixel. World origin = pitch centre, z = 0 = turf, stumps
at (0, ±10.08 m).

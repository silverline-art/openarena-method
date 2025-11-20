# EXMO Metric Corrections - Quick Reference Card
**Version:** v1.2.0 Calibrated
**Date:** 2025-11-21

---

## TL;DR - What Changed

```
SCALING:   8cm → 10cm           (+25% distance metrics)
SMOOTHING: 11 → 7 frames        (+15-20% peak preservation)
WALKING:   MAD 2.0 → 0.8        (+50% walking detection)
STRIDES:   0.1s → 0.06s min     (+40% stride count)
HIP ROM:   hip→paw → hip→knee   (FIXED: 125° → 18°)
COM:       2D → 3D              (+15% speed accuracy)
```

---

## Parameter Changes

### Preprocessing (`config_v1.2_calibrated.yaml`)

```yaml
# SCALING (CRITICAL)
expected_body_length_cm: 10.0        # was 8.0
scaling_method: "full_body"          # was implicit spine_only
scaling_min_likelihood: 0.9          # NEW
scaling_tolerance: 0.25              # was 0.20

# SMOOTHING (REDUCED)
smoothing_window: 7                  # was 11
smoothing_window_velocity: 5         # NEW
velocity_smoothing_method: "ema"     # NEW
smoothing_window_rom: 3              # NEW

# OUTLIERS (RELAXED)
outlier_threshold: 4.0               # was 3.0
max_interpolation_gap: 10            # was 5
```

### Phase Detection

```yaml
# THRESHOLDS (RELAXED)
walking_mad_threshold: 0.8           # was 2.0
stationary_mad_threshold: 0.9        # was 1.5
use_hybrid_threshold: true           # NEW
adaptive_percentile: 55              # NEW
min_threshold_px_per_frame: 1.0      # NEW

# DURATIONS (REDUCED)
min_walking_duration: 0.12           # was 0.3
min_stationary_duration: 0.12        # was 0.25
```

### Stride Detection

```yaml
# RANGE (EXPANDED)
min_stride_duration: 0.06            # was 0.1
max_stride_duration: 1.5             # unchanged

# SENSITIVITY (INCREASED)
prominence_multiplier: 0.25          # was 0.5

# MICRO-STEPS (NEW)
allow_micro_steps: true              # was false
micro_step_threshold_cm: 1.0         # NEW
```

### COM & ROM

```yaml
# COM (3D)
use_3d_com: true                     # NEW
com_smoothing_window: 7              # NEW

# ROM (ANGLE-BASED)
rom_use_triplet_angles: true         # NEW
rom_view: "side"                     # NEW
rom_smoothing_enabled: true
# (uses smoothing_window_rom: 3)
```

---

## Code Changes (Key Functions)

### Scaling Factor
```python
# OLD (v1.1.0) - DO NOT USE
preprocessor.compute_scale_factor(spine1, spine3, known_distance_cm=8.0)

# NEW (v1.2.0) - USE THIS
preprocessor.compute_scale_factor_v2(
    snout, tailbase,
    likelihood_snout, likelihood_tail,
    expected_body_length_cm=10.0,
    min_likelihood=0.9,
    tolerance=0.25
)
```

### Smoothing
```python
# OLD (v1.1.0) - Excessive
apply_savgol_filter(trajectory, window_length=11, polyorder=3)

# NEW (v1.2.0) - Reduced
apply_savgol_filter(trajectory, window_length=7, polyorder=3)

# NEW (v1.2.0) - Velocity-specific
smooth_velocity_ema(positions, alpha=0.35, fps=120.0)
```

### Walking Detection
```python
# OLD (v1.1.0) - Pure MAD
threshold = mad * 2.0
walking = com_speed > threshold

# NEW (v1.2.0) - Hybrid
threshold = compute_hybrid_threshold(com_speed, mad_multiplier=0.8)
# Combines: (MAD × 0.8 + percentile_55) / 2
walking = com_speed > threshold
```

### Stride Detection
```python
# OLD (v1.1.0) - Strict filtering
valid_strides = stride_times >= 0.1

# NEW (v1.2.0) - Relaxed with labeling
stride_info = compute_stride_info_v2(
    foot_strikes, paw_trajectory, scale_factor
)
# Returns: {duration_sec, length_cm, is_micro_step}
```

### Hip ROM
```python
# OLD (v1.1.0) - INCORRECT
hip_rom = compute_joint_rom_and_velocity(
    keypoints['hip_center'],
    keypoints['hip_R'],
    keypoints['paw_RR']  # WRONG - skips knee
)

# NEW (v1.2.0) - CORRECT
hip_rom = compute_joint_rom_and_velocity(
    keypoints['hip_center'],
    keypoints['hip_R'],
    keypoints['knee_R']  # CORRECT
)
```

### COM Speed
```python
# OLD (v1.1.0) - 2D only
com_2d = compute_com_trajectory(hip_center, rib_center)
speed_2d = compute_trajectory_speed(com_2d, fps)  # sqrt(dx² + dy²)

# NEW (v1.2.0) - 3D
com_3d = compute_com_3d(top_keypoints, side_keypoints, weights)
speed_3d = compute_velocity_3d(com_3d, fps, scale_factor)  # sqrt(dx² + dy² + dz²)
```

---

## Expected Metric Changes

### Distance Metrics (ALL +20-30%)
- Stride length
- COM sway (ML/AP)
- Displacement
- Trajectory length

**Cause:** Scaling factor correction (8cm → 10cm = +25%)

### Speed Metrics (+30-40%)
- Walking speed
- COM velocity
- Paw velocity

**Cause:** Scaling (+25%) + 3D calculation (+15%) + less smoothing (+5-10%)

### Frequency Metrics (-20-30%)
- Cadence (steps/min)

**Cause:** Walking detection improvement (more walking time detected)
**Note:** Cadence was OVERESTIMATED in v1.1.0 due to under-detected walking duration

### Stride Count (+40-60%)
- Total strides
- Strides per walking bout

**Cause:** Relaxed minimum duration (0.1s → 0.06s) includes micro-steps

### ROM Metrics (+30-50%)
- Hip ROM
- Elbow ROM
- All joint angles

**Cause:** Minimal smoothing (11 → 3 frames) preserves oscillations
**Special:** Hip ROM FIXED from incorrect keypoints (was 125°, now 18°)

### Angular Velocity (×2-×4)
- Hip angular velocity
- Elbow angular velocity

**Cause:** Minimal smoothing on angles + gradient calculation

---

## When to Use Which Version

### Use v1.1.0 (`config.yaml`) IF:
- Comparing to historical data processed with v1.1.0
- Validating v1.2.0 corrections (side-by-side comparison)
- **NOT RECOMMENDED for new analyses**

### Use v1.2.0 (`config_v1.2_calibrated.yaml`) IF:
- Processing new data
- Reprocessing critical datasets for accuracy
- Open-field behavior analysis
- **RECOMMENDED for all production use**

---

## Validation Checklist

### Before Deploying v1.2.0:

- [ ] Run on 3-5 control samples
- [ ] Check expected metric changes:
  - [ ] Stride length: +20-40%
  - [ ] Cadence: -20-30% (was inflated)
  - [ ] Speed: +30-40%
  - [ ] Hip ROM: 15-25° (physiological range)
  - [ ] Walking duration: +50-100%
- [ ] Visual inspection:
  - [ ] Walking bout boundaries look correct
  - [ ] No obvious false positives
  - [ ] Stride detection follows paw oscillations
- [ ] Manual annotation comparison (if available):
  - [ ] Stride count agreement: >85%
  - [ ] Walking duration agreement: >80%

### Red Flags (Indicates Problem):

- Hip ROM still >100° → keypoint mapping error
- Cadence >600 steps/min → threshold too low
- Stride length >8cm → scaling error
- Walking detection <20% of frames → threshold too high

---

## Troubleshooting

### "Metrics seem too high now"

**EXPECTED.** v1.1.0 was systematically underestimating. v1.2.0 values are more accurate.

**Validation:**
- Check against physiological ranges (see audit report)
- Compare to manual annotation
- Verify scaling factor is ~0.095 cm/px (for 10cm reference)

### "Too many strides detected"

**Check:**
1. `min_stride_duration` - may need 0.07s instead of 0.06s
2. `prominence_multiplier` - may need 0.3 instead of 0.25
3. Review micro-step percentage - should be <20%

**Tuning:**
```yaml
min_stride_duration: 0.07        # Slightly stricter
prominence_multiplier: 0.3       # Slightly less sensitive
```

### "Walking detection too aggressive"

**Check:**
1. `walking_mad_threshold` - may need 0.9 instead of 0.8
2. `adaptive_percentile` - may need 60 instead of 55
3. `min_walking_duration` - may need 0.15s instead of 0.12s

**Tuning:**
```yaml
walking_mad_threshold: 0.9       # Slightly stricter
adaptive_percentile: 60          # Higher threshold
min_walking_duration: 0.15       # Longer minimum
```

---

## Quick Commands

### Run Single Sample (v1.2.0)
```bash
python batch_process.py \
  --config config_v1.2_calibrated.yaml \
  --sample control_5
```

### Compare Versions
```bash
python batch_process.py \
  --config config_v1.2_calibrated.yaml \
  --compare-baseline config.yaml \
  --sample control_5 \
  --output comparison/
```

### Batch Process Group
```bash
python batch_process.py \
  --config config_v1.2_calibrated.yaml \
  --group control \
  --parallel 4
```

### Run Validation Script
```bash
python docs/validation_calculations.py
```

---

## File Locations

```
Configuration:
  /config_v1.2_calibrated.yaml         (PRODUCTION - USE THIS)
  /config.yaml                          (LEGACY - v1.1.0)

Documentation:
  /docs/AUDIT_EXECUTIVE_SUMMARY.md              (5-page summary)
  /docs/METRIC_CORRECTNESS_AUDIT_REPORT.md      (Full 80-page report)
  /docs/QUICK_REFERENCE_CORRECTIONS.md          (This file)
  /docs/validation_calculations.py              (Validation script)

Core Code:
  /src/exmo_gait/core/preprocessor.py           (Scaling, COM)
  /src/exmo_gait/utils/signal_processing.py     (Smoothing)
  /src/exmo_gait/analysis/phase_detector.py     (Walking detection)
  /src/exmo_gait/analysis/step_detector.py      (Stride detection)
  /src/exmo_gait/analysis/metrics_computer.py   (ROM, metrics)
```

---

## Support

**Questions?** Review:
1. This quick reference (current file)
2. Executive summary (`/docs/AUDIT_EXECUTIVE_SUMMARY.md`)
3. Full audit report (`/docs/METRIC_CORRECTNESS_AUDIT_REPORT.md`)
4. Run validation script (`python docs/validation_calculations.py`)

**Issues?** Check:
- Configuration file matches v1.2.0 parameters
- Input data has all required views (TOP, SIDE, BOTTOM)
- Likelihood values available for scaling
- Expected metric changes are within ranges

---

**Last Updated:** 2025-11-21
**Version:** v1.2.0
**Status:** PRODUCTION READY (pending validation)

---

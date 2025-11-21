# EXMO Gait Analysis - Complete Parameter Reference

## Overview

This document defines every configurable and derived parameter in the EXMO pipeline, including:
- What it controls
- How it's derived (static config vs dynamic computation)
- Allowed ranges and safety clamps
- Units and physical meaning

---

## 1. Scaling & Calibration Parameters

| Parameter | Unit | Derivation | Range | Physical Meaning |
|-----------|------|------------|-------|------------------|
| `scaling_top_cm_per_px` | cm/pixel | **Dynamic**: `body_length_cm / median(snout-tail_px in TOP)` | [0.02, 0.15] | Pixel-to-cm conversion for TOP view (ground plane) |
| `scaling_side_cm_per_px` | cm/pixel | **Dynamic**: `body_length_cm / median(snout-tail_px in SIDE)` | [0.02, 0.15] | Pixel-to-cm conversion for SIDE view (vertical motion) |
| `scaling_bottom_cm_per_px` | cm/pixel | **Dynamic**: `body_length_cm / median(hip-tail_px in BOTTOM)` | [0.02, 0.15] | Pixel-to-cm conversion for BOTTOM view (paw contact) |
| `body_length_cm` | cm | **Config**: User-provided or default | [8.0, 12.0] | Known body length for scaling calibration |
| `fps` | frames/sec | **Static**: 120.0 | [30, 240] | Camera frame rate |

**Safety Checks**:
- If `|scaling_top - scaling_side| > 0.02`: Warning - camera setup mismatch
- If any scaling < 0.02 or > 0.15: Error - Invalid calibration

---

## 2. Walking Phase Detection Parameters

### CoM Speed Thresholds

| Parameter | Unit | Derivation | Default Multiplier | Range |
|-----------|------|------------|-------------------|-------|
| `com_speed_median` | cm/s | **Dynamic**: `median(com_speed_trajectory)` | N/A | Auto |
| `com_speed_mad` | cm/s | **Dynamic**: `MAD(com_speed_trajectory)` | N/A | Auto |
| `stationary_threshold_cm_s` | cm/s | **Dynamic**: `median + (stationary_multiplier × MAD)` | 0.5 | [0.0, 2.0] |
| `walking_threshold_cm_s` | cm/s | **Dynamic**: `max(p75, median + (walking_multiplier × MAD))` | 1.0 | [2.0, 50.0] |

**Config Multipliers**:
```yaml
phase_detection:
  stationary_mad_multiplier: 0.5
  walking_mad_multiplier: 1.0
```

**Sensitivity Adjustment**:
- `sensitivity: high` → multiply both by 0.7 (more permissive)
- `sensitivity: low` → multiply both by 1.3 (more conservative)

---

## 3. Step Detection Parameters

### Peak Detection Thresholds

| Parameter | Unit | Derivation | Default Multiplier | Range |
|-----------|------|------------|-------------------|-------|
| `vertical_displacement_range` | cm | **Dynamic**: `max(paw_z) - min(paw_z)` | N/A | Auto |
| `median_prominence` | cm | **Dynamic**: `median(all_peak_prominences)` | N/A | Auto |
| `prominence_threshold` | cm | **Dynamic**: `prominence_multiplier × median_prominence` | 0.3 | [0.05×range, 0.5×range] |
| `initial_stride_estimate` | sec | **Dynamic**: `median(inter_peak_intervals) / fps` | N/A | Auto |

**Config Multipliers**:
```yaml
step_detection:
  prominence_multiplier: 0.3
  stride_duration_min_factor: 0.5
  stride_duration_max_factor: 2.0
```

### Stride Duration Bounds

| Parameter | Unit | Derivation | Safety Clamp |
|-----------|------|------------|--------------|
| `min_stride_duration_sec` | sec | **Dynamic**: `0.5 × initial_stride_estimate` | [0.05, 0.3] |
| `max_stride_duration_sec` | sec | **Dynamic**: `2.0 × initial_stride_estimate` | [0.3, 2.0] |

**Physical Justification**:
- Fastest mouse gait: ~20 Hz → 0.05s minimum
- Slowest mouse gait: ~0.5 Hz → 2.0s maximum

---

## 4. Outlier Detection Parameters

| Parameter | Unit | Derivation | Default Multiplier | Range |
|-----------|------|------------|-------------------|-------|
| `median_x`, `median_y` | pixel | **Dynamic**: `median(trajectory_coordinates)` | N/A | Auto |
| `mad_x`, `mad_y` | pixel | **Dynamic**: `MAD(trajectory_coordinates)` | N/A | Auto |
| `outlier_threshold` | pixel | **Dynamic**: `outlier_mad_multiplier × MAD` | 5.0 | [3.0, 10.0] |

**Config**:
```yaml
preprocessing:
  outlier_mad_multiplier: 5.0  # Relaxed from z-score 3.0
```

**Rationale**: MAD more robust than std; higher multiplier reduces false outlier removal

---

## 5. Smoothing & Filtering Parameters

| Parameter | Unit | Derivation | Default | Range |
|-----------|------|------------|---------|-------|
| `savgol_window_length` | frames | **Config** or **Dynamic**: `fps / 10` | 11 | [5, 31] (must be odd) |
| `savgol_poly_order` | N/A | **Config** | 3 | [1, 5] |
| `median_filter_window` | frames | **Config** | 5 | [3, 11] (must be odd) |

**Dynamic Adjustment**:
- High noise (MAD > threshold): Increase window length by 50%
- Low noise: Use smaller window to preserve detail

---

## 6. Multi-Signal Step Detection Parameters

### Signal Fusion Weights

| Signal | Weight | Unit | Purpose |
|--------|--------|------|---------|
| `vertical_displacement` | 0.4 | cm | Primary stance/swing detection |
| `horizontal_velocity` | 0.3 | cm/s | Velocity zero-crossings |
| `ground_distance` | 0.3 | cm | Contact point detection |

**Detection Rule**:
```
Step detected if ≥2 signals agree within ±3 frames
Confidence = count(agreeing_signals) / 3
Accept if confidence ≥ 0.6
```

---

## 7. ROM (Range of Motion) Parameters

| Parameter | Unit | Derivation | Range |
|-----------|------|------------|-------|
| `joint_angle` | degrees | **Dynamic**: 3-point angle calculation | [0, 180] |
| `angle_velocity` | deg/s | **Dynamic**: `diff(angle) × fps` | [-360, 360] |
| `max_angle_change_per_frame` | degrees | **Config** | 10 | [5, 20] |

**Outlier Removal**:
- Remove angles where `|angle_velocity| > max_angle_change_per_frame × fps`
- Indicates keypoint tracking error, not true biomechanics

---

## 8. Coordination Metrics Parameters

### Regularity Index

| Parameter | Unit | Derivation | Range | Meaning |
|-----------|------|------------|-------|---------|
| `regularity_index` | 0-1 | **Dynamic**: Temporal coupling of diagonal pairs | [0, 1] | 0=no coupling, 1=perfect |
| `min_steps_for_regularity` | steps | **Config** | 5 | [3, 10] |

### Phase Dispersion

| Parameter | Unit | Derivation | Range | Meaning |
|-----------|------|------------|-------|---------|
| `phase_dispersion` | 0-1 | **Dynamic**: Variability in relative phase | [0, 1] | 0=consistent, 1=random |

---

## 9. Output Metrics - Standardized Units

### Temporal Metrics

| Metric | Unit | Formula | Range |
|--------|------|---------|-------|
| `stride_time_sec` | sec | Time between foot strikes | [0.05, 2.0] |
| `swing_time_sec` | sec | Time paw off ground | [0.02, 1.0] |
| `stance_time_sec` | sec | Time paw on ground | [0.03, 1.5] |
| `cadence_steps_per_min` | steps/min | `60 / median(stride_time_sec)` | [30, 1200] |

### Spatial Metrics

| Metric | Unit | Formula | Range |
|--------|------|---------|-------|
| `stride_length_cm` | cm | Distance between consecutive foot strikes | [1.0, 20.0] |
| `step_width_cm` | cm | Lateral separation between left/right paws | [0.5, 5.0] |
| `com_displacement_cm` | cm | Total COM path length | Auto |

### Phase Ratios

| Metric | Unit | Formula | Range |
|--------|------|---------|-------|
| `duty_cycle_percent` | % | `(stance_time / stride_time) × 100` | [0, 100] |
| `swing_stance_ratio_0to1` | 0-1 | `swing_time / stance_time` | [0.1, 5.0] |

### Angular Metrics

| Metric | Unit | Formula | Range |
|--------|------|---------|-------|
| `elbow_angle_deg` | degrees | 3-point angle (shoulder-elbow-paw) | [30, 180] |
| `hip_angle_deg` | degrees | 3-point angle (spine-hip-knee) | [30, 180] |
| `ankle_angle_deg` | degrees | 3-point angle (knee-ankle-paw) | [30, 180] |

---

## 10. High-Level Configuration Modes

### Sensitivity Level

| Mode | Effect | Use Case |
|------|--------|----------|
| `high` | Multiply all MAD multipliers by 0.7 | Low-activity open field |
| `medium` | Default multipliers | Standard experiments |
| `low` | Multiply all MAD multipliers by 1.3 | Noisy data or forced locomotion |

### Environment Type

| Type | Adjustments | Use Case |
|------|------------|----------|
| `open_field` | Relaxed stride duration bounds | Free exploration |
| `treadmill` | Tighter stride duration, higher walking threshold | Forced constant speed |
| `forced_locomotion` | Conservative thresholds, strict outlier removal | Stimulated running |

**Config Example**:
```yaml
adaptive_thresholding:
  sensitivity_level: medium
  environment_type: open_field

  multipliers:
    stationary_mad: 0.5
    walking_mad: 1.0
    prominence: 0.3
    outlier_mad: 5.0
```

---

## 11. Diagnostic Output Parameters

All derived thresholds logged to `diagnostics/calibration_diagnostics.yaml`:

```yaml
sample_id: <id>
calibration:
  scaling_factors: {top, side, bottom}
  body_length_px: {top, side, bottom}

thresholds_derived:
  walking_detection: {median, mad, thresholds}
  step_detection: {per limb prominences, duration bounds}

detection_results:
  walking_windows: <count>
  steps_detected: {per limb}
  avg_cadence: <value>

validation:
  cross_view_agreement_percent: <value>
  low_confidence_steps: <count>
```

---

## 12. Quick Reference - "Must Use" Units

| Metric Type | MUST Use | Never Use |
|-------------|----------|-----------|
| Time | `_sec` | `_s`, `_time`, unitless |
| Cadence | `_steps_per_min` | `_hz`, `_per_sec` |
| Distance | `_cm` | `_m`, `_mm`, `_px` |
| Angle | `_deg` | `_rad`, unitless |
| Percentage | `_percent` | `_frac`, 0-1 without suffix |
| Dimensionless | `_0to1` | no suffix |

---

## 13. Implementation Checklist

### Phase 1: Critical (v1.3.0)
- [ ] Compute per-view scaling factors
- [ ] Dynamic walking/stationary thresholds (MAD-based)
- [ ] Dynamic prominence thresholds
- [ ] Add unit suffixes to all Excel columns
- [ ] Add unit labels to all plot axes

### Phase 2: Enhanced (v1.4.0)
- [ ] Multi-signal step fusion
- [ ] Cross-view validation
- [ ] Adaptive stride duration
- [ ] Diagnostic YAML export

### Phase 3: Full Automation (v1.5.0)
- [ ] Auto-sensitivity selection
- [ ] Interactive threshold UI
- [ ] Batch diagnostic comparison

---

**Document Status**: APPROVED - Ready for Implementation

**Companion Document**: `PRD_Dynamic_Thresholding_Scaling.md`

# Dynamic Thresholding & Multi-View Scaling - Product Requirements Document

## Executive Summary

This PRD defines the architectural improvements required to transform the EXMO gait analysis pipeline from a static, configuration-based system to a dynamic, self-calibrating system that adapts to individual samples, activity levels, and camera setups.

**Core Principle**: *Look at the data first, then decide how strict to be.*

---

## 1. Problem Statement

### 1.1 Current Limitations

The existing pipeline suffers from five critical design flaws:

1. **Inconsistent Units**: Phase metrics, time values, and ratios lack standardized unit conventions
2. **Static Thresholds**: Hard-coded values fail to adapt to different mice, activity levels, or camera setups
3. **Single Scaling Factor**: One cm/pixel conversion used across all three views (TOP/SIDE/BOTTOM) despite different camera geometries
4. **Under-Detection of Steps**: Conservative thresholds miss legitimate low-amplitude steps
5. **Non-Adaptive Filters**: Parameters don't self-calibrate based on per-sample noise and signal characteristics

### 1.2 Impact

- **Scientific Validity**: Measurements are physically incorrect due to improper scaling
- **Data Loss**: Up to 60-70% of actual steps undetected in low-activity samples
- **Reproducibility**: Different camera setups produce incomparable results
- **User Frustration**: Manual threshold tuning required per experiment

---

## 2. Design Requirements

### 2.1 Standardized Units & Definitions

**Requirement**: All metrics must follow strict unit conventions with explicit labeling.

#### Unit Standards

| Metric Type | Unit | Format | Example |
|-------------|------|---------|---------|
| Time | seconds (s) | `metric_sec` | `stride_time_sec` |
| Cadence | steps/min | `cadence_steps_per_min` | 120 steps/min |
| Spatial Distance | centimeters (cm) | `metric_cm` | `stride_length_cm` |
| Angular | degrees (°) | `metric_deg` | `elbow_angle_deg` |
| Percentage | % (0-100) | `metric_percent` | `duty_cycle_percent` |
| Dimensionless Ratio | 0-1 | `metric_0to1` | `regularity_index_0to1` |

#### Metric Definitions

**Temporal Metrics**:
- `stride_time_sec`: Time between consecutive foot strikes (seconds)
- `swing_time_sec`: Time paw is off ground (seconds)
- `stance_time_sec`: Time paw is in contact with ground (seconds)
- `cadence_steps_per_min`: (60 / median_stride_time_sec)

**Phase Ratios**:
- `duty_cycle_percent`: (stance_time / stride_time) × 100
- `swing_stance_ratio_0to1`: swing_time / stance_time (dimensionless)

**Coordination Metrics**:
- `regularity_index_0to1`: Temporal coupling between diagonal limb pairs (0 = no coordination, 1 = perfect)
- `phase_dispersion_0to1`: Variability in relative phase timing between limbs (0 = consistent, 1 = random)

#### Implementation Requirements

1. **Excel Export**:
   - Column names MUST include unit suffix
   - Add "Units" column to summary sheets
   - Include unit legend on first sheet

2. **Plot Labels**:
   - Y-axis labels: "Metric Name (unit)"
   - Example: "Duty Cycle (%)", "Stride Time (s)", "Phase Dispersion (0–1)"

3. **API/Code**:
   - Dictionary keys include units: `{"stride_time_sec": 0.35, "duty_cycle_percent": 65}`
   - Function docstrings explicitly state return units

---

### 2.2 Dynamic Threshold Derivation

**Requirement**: No detector shall use static numerical thresholds. All thresholds derived from per-sample statistics.

#### General Derivation Pattern

For any signal `x(t)` (COM speed, paw vertical position, joint angle, etc.):

```
1. Compute baseline statistics:
   - median μ = median(x)
   - MAD σ = median(|x - μ|)
   - percentiles: p25, p50, p75, p90

2. Derive threshold:
   threshold = f(μ, σ, percentile, config_multiplier)

3. Apply safety clamps:
   threshold = clamp(threshold, physical_min, physical_max)
```

#### Specific Threshold Derivations

**Walking Detection** (`PhaseDetector`):
```
Input: COM speed trajectory (cm/s)

Compute:
- median_speed = median(com_speed)
- mad_speed = MAD(com_speed)
- p75_speed = percentile(com_speed, 75)

Thresholds:
- stationary_threshold = median_speed + (config.stationary_multiplier * mad_speed)
  Default multiplier: 0.5
  Safety clamp: [0.0, 2.0] cm/s

- walking_threshold = max(p75_speed, median_speed + (config.walking_multiplier * mad_speed))
  Default multiplier: 1.0
  Safety clamp: [2.0, 50.0] cm/s

Rationale: Uses data distribution instead of fixed values. P75 ensures captures upper activity quartile.
```

**Step Detection** (`StepDetector`):
```
Input: Paw vertical displacement (cm)

First Pass - Estimate stride frequency:
- Detect all peaks with minimal threshold (0.1 * range(vertical_displacement))
- Compute inter-peak intervals
- median_stride_time = median(inter_peak_intervals) / fps

Compute:
- peak_prominences = [prominence of each detected peak]
- median_prominence = median(peak_prominences)

Thresholds:
- prominence_threshold = config.prominence_multiplier * median_prominence
  Default multiplier: 0.3 (relaxed to capture low-amplitude steps)
  Safety clamp: [0.05 * range(signal), 0.5 * range(signal)]

- min_stride_duration = 0.5 * median_stride_time
  Safety clamp: [0.05s, 0.3s]

- max_stride_duration = 2.0 * median_stride_time
  Safety clamp: [0.3s, 2.0s]

Rationale: Adapts to individual gait frequency. Allows faster/slower mice. Relaxed prominence captures low-amplitude steps.
```

**Outlier Removal** (`Preprocessor`):
```
Input: Keypoint trajectory coordinates

Compute:
- median_x, median_y = median coordinates
- mad_x, mad_y = MAD of coordinates

Threshold:
- outlier_threshold = config.outlier_mad_multiplier * MAD
  Default multiplier: 5.0 (relaxed from z-score 3.0)

Mark as outlier if:
- |x - median_x| > outlier_threshold * mad_x OR
- |y - median_y| > outlier_threshold * mad_y

Rationale: MAD more robust to outliers than standard deviation. Higher multiplier reduces false removals.
```

#### Configuration Exposure

High-level config parameters (NOT numerical thresholds):

```yaml
adaptive_thresholding:
  sensitivity_level: medium  # low | medium | high
  environment_type: open_field  # open_field | treadmill | forced_locomotion

  multipliers:
    stationary_mad: 0.5
    walking_mad: 1.0
    prominence: 0.3
    outlier_mad: 5.0
```

Internal mapping:
- `sensitivity_level: high` → reduce all multipliers by 30%
- `sensitivity_level: low` → increase all multipliers by 30%
- `environment_type: treadmill` → tighter stride duration bounds
- `environment_type: open_field` → relaxed stride duration bounds

---

### 2.3 Per-View Scaling Factors

**Requirement**: Compute three independent scaling factors, one per camera view.

#### Physical Rationale

Each camera view has:
- Different distance to subject
- Different lens focal length
- Different perspective/parallax effects

Therefore, **pixel-to-cm conversion MUST be view-specific**.

#### Computation Method

For each view V ∈ {TOP, SIDE, BOTTOM}:

```
1. Identify reference body segment visible in view V:
   - TOP: snout ↔ tail_base (body length along spine)
   - SIDE: snout ↔ tail_base (projected on sagittal plane)
   - BOTTOM: hip_center ↔ tail_base (lower body reference)

2. For each frame t:
   - Compute pixel distance: L_px(t) = ||keypoint1(t) - keypoint2(t)||
   - Filter outliers using MAD

3. Compute median pixel length:
   - L_px_median = median(L_px[t] for all valid frames)

4. Known physical length:
   - L_cm = 10.0 cm (typical adult mouse body length)
   - OR: user-provided per-sample calibration

5. Scaling factor:
   - scaling_V = L_cm / L_px_median
   - Unit: cm/pixel
```

#### Application Rules

**TOP View Scaling** (scaling_top):
- COM trajectory (x, y) → cm
- Stride length (horizontal ground plane) → cm
- Step width (lateral separation) → cm
- Horizontal velocity → cm/s

**SIDE View Scaling** (scaling_side):
- Vertical displacement (z-axis proxy) → cm
- Body pitch angle calculation (geometry)
- Vertical velocity → cm/s
- ROM angles (use correct limb segment lengths in cm)

**BOTTOM View Scaling** (scaling_bottom):
- Paw contact area → cm²
- Lateral base of support → cm
- Paw placement precision → cm

#### Validation & Safety

**Cross-view consistency check**:
```
If |scaling_top - scaling_side| > 0.02 cm/px:
    WARNING: Views have significantly different scaling.
    Possible causes:
    - Camera distance mismatch
    - Lens distortion
    - Animal not centered in all views

Recommendation: Check camera setup calibration.
```

**Physical plausibility check**:
```
Expected scaling range for mouse videos: [0.02, 0.15] cm/px

If scaling < 0.02: ERROR - Camera too far or body keypoints wrong
If scaling > 0.15: ERROR - Camera too close or body keypoints wrong
```

---

### 2.4 Enhanced Step Detection

**Requirement**: Detect ≥90% of true steps under open-field conditions without false positive explosion.

#### Multi-Signal Fusion

Combine evidence from multiple signals for robust detection:

```
For each paw P:
  Signals:
    - vertical_displacement(t): SIDE or BOTTOM view z-coordinate
    - horizontal_velocity(t): speed of paw in ground plane
    - ground_distance(t): distance from estimated ground plane

  Detection logic:
    1. Find local minima in vertical_displacement (swing peaks)
    2. Find velocity zero-crossings (stance initiation)
    3. Find ground_distance minima (contact points)

  Combine:
    - A step is detected if ≥2 of 3 signals agree within ±3 frames
    - Confidence score = number of agreeing signals / 3
```

#### Dynamic Duration Bounds

```
Initial estimate:
- Detect coarse peaks with minimal threshold
- Compute rough_median_stride = median(inter_peak_intervals)

Adaptive bounds:
- min_stride_time = 0.5 * rough_median_stride
- max_stride_time = 2.0 * rough_median_stride

Safety clamps (based on mouse physiology):
- min_stride_time >= 0.05s (no faster than 20 Hz)
- max_stride_time <= 2.0s (no slower than 0.5 Hz)

Rationale:
- Fast mice: rough_median ≈ 0.15s → bounds [0.075s, 0.3s]
- Slow mice: rough_median ≈ 0.4s → bounds [0.2s, 0.8s]
- Adapts automatically, no manual tuning.
```

#### Adaptive Prominence

```
First pass:
- Detect all peaks with prominence > 0.05 * signal_range
- Compute distribution of prominences

Analysis:
- median_prominence = median(all_prominences)
- mad_prominence = MAD(all_prominences)

Threshold selection:
- Initial: prominence_threshold = 0.3 * median_prominence

Iterative refinement:
  If detected_steps < expected_minimum (based on video duration):
    prominence_threshold *= 0.8  (relax by 20%)
    Re-detect

  If detected_steps > expected_maximum (based on cadence limits):
    prominence_threshold *= 1.2  (tighten by 20%)
    Re-detect

  Max iterations: 3

Final threshold logged to metadata for transparency.
```

#### Cross-View Validation

```
For each detected step S at time t:

  Check SIDE view: vertical displacement peak at t ± 3 frames?
  Check BOTTOM view: paw contact event at t ± 3 frames?
  Check horizontal velocity: velocity minimum at t ± 3 frames?

  Confidence = count(agreeing signals) / total_signals

  Accept step if:
    - confidence >= 0.6 (at least 2 of 3 signals agree)

  Mark as "low_confidence" if:
    - 0.4 <= confidence < 0.6
    (include in analysis but flag for manual review)
```

---

### 2.5 Per-Sample Diagnostic Logging

**Requirement**: Log all derived thresholds and dataset statistics for transparency and debugging.

#### Diagnostic Output Structure

```yaml
sample_id: control-5
date_processed: 2025-11-21T08:00:00

calibration:
  scaling_factors:
    top_cm_per_px: 0.0622
    side_cm_per_px: 0.0635
    bottom_cm_per_px: 0.0598
    warning: "SIDE and BOTTOM differ by >5% - check camera alignment"

  body_length_px:
    top: 160.3
    side: 157.5
    bottom: 167.2

  assumed_body_length_cm: 10.0

thresholds_derived:
  walking_detection:
    com_speed_median_cm_s: 4.2
    com_speed_mad_cm_s: 2.8
    stationary_threshold_cm_s: 5.6  # median + 0.5*MAD
    walking_threshold_cm_s: 7.0     # median + 1.0*MAD

  step_detection:
    paw_RR:
      vertical_displacement_range_cm: 2.3
      median_prominence_cm: 0.6
      prominence_threshold_cm: 0.18  # 0.3 * median
      initial_stride_estimate_sec: 0.28
      min_stride_duration_sec: 0.14
      max_stride_duration_sec: 0.56

    paw_RL:
      vertical_displacement_range_cm: 2.1
      median_prominence_cm: 0.55
      prominence_threshold_cm: 0.165
      initial_stride_estimate_sec: 0.30
      min_stride_duration_sec: 0.15
      max_stride_duration_sec: 0.60

detection_results:
  walking_windows_detected: 15
  total_walking_duration_sec: 42.3

  steps_detected:
    paw_RR: 127
    paw_RL: 121
    paw_FR: 115
    paw_FL: 108

  avg_cadence_steps_per_min: 162

validation:
  cross_view_agreement_percent: 87.3
  low_confidence_steps_count: 12
  flagged_for_review: false
```

This diagnostic is saved as:
- `{output_dir}/diagnostics/calibration_diagnostics.yaml`
- Included as "Diagnostics" sheet in Excel export

---

## 3. Implementation Priority

### Phase 1: Critical Fixes (v1.3.0)
1. ✅ Per-view scaling factors
2. ✅ Dynamic walking/stationary thresholds
3. ✅ Dynamic step detection prominence
4. ✅ Unit standardization in outputs

### Phase 2: Enhanced Detection (v1.4.0)
5. Multi-signal step detection fusion
6. Cross-view validation
7. Adaptive stride duration bounds
8. Low-confidence step flagging

### Phase 3: Full Automation (v1.5.0)
9. Automatic sensitivity_level selection
10. Camera calibration validation
11. Interactive threshold tuning UI
12. Batch diagnostic comparison tools

---

## 4. Success Criteria

**Quantitative Metrics**:
- Step detection recall ≥ 90% (compared to manual annotation)
- False positive rate ≤ 5%
- Scaling factor error ≤ 3% (compared to ruler calibration)
- Cross-sample threshold variance ≤ 20% for same experiment type

**Qualitative Metrics**:
- Users no longer need to manually tune thresholds
- Results reproducible across different camera setups
- Scientific reviewers accept measurement methodology
- Clear, unambiguous unit labeling eliminates interpretation errors

---

## 5. Parameter Reference

See companion document: `PARAMETER_TABLE.md`

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-11-21 | Initial PRD based on architectural review |

---

**Document Status**: DRAFT - Awaiting Implementation

**Next Steps**: Create parameter table, implement Phase 1 critical fixes

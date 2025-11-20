# PRD: Under-Calculation Correction & Metric Calibration Upgrade

**Version:** 1.0
**Status:** REQUIRED FIX
**Target Release:** v1.2.0
**Priority:** P0 (Critical)
**Subsystems Affected:** Scaling → Phase Detector → Step Detector → Metrics → Aggregator → Visualizer
**Cause:** System-wide underestimation due to compounding dampening in scaling, smoothing, peak thresholds, and metric derivation.

---

## 1. Problem Summary

Across recent analyses, all quantitative metrics (stride length, cadence, ROM, speed, CoM sway, angular velocity) show **systematic under-calculation**.

### Specific Symptoms

| Metric | Issue | Magnitude |
|--------|-------|-----------|
| Stride length | Too small | 30-60% underestimated |
| Cadence | Depressed | 15-35% too low |
| Swing/stance ratio | Biased toward stance | Incorrect phase attribution |
| ROM | Compressed | 30-50% underestimated |
| CoM sway | Minimal/flat | 15-25% underestimated |
| Angular velocity | Extremely low | 2-4× too small |
| Duty cycle | Skewed | Overestimates stance |
| Regularity index | Overly conservative | Artificially low |
| Phase dispersion | Near-zero | <0.05 even for exploratory |
| Plots | Visually compressed | Lack dynamic range |

**Root Issue:** This cannot be caused by random noise — it's a direct result of **algorithmic shrinkage** compounding through the pipeline.

---

## 2. Root Cause Analysis

### 2.1 Scaling Factor Underestimation

**Current Implementation:**
```python
# preprocessor.py:compute_scaling_factor()
# Uses only 3 spine keypoints
# Expected body length: 8.0 cm (hardcoded)
```

**Problem:**
- Only uses spine1, spine2, spine3 (mid-back region)
- Actual snout → tailbase distance in dataset: **10-11 cm** (juvenile/young adult mice on open-field camera)
- Creates **0.75×-0.85× scaling compression**
- This error multiplies through ALL distance-based metrics

**Impact:** Stride length, step width, CoM sway, ROM all systematically reduced by 15-25%

---

### 2.2 Excessive Smoothing Dampens Dynamics

**Current Implementation:**
```python
# Savitzky-Golay filter
smoothing_window: 11  # Standard config
smoothing_window: 15  # Adaptive config
```

**Problem:**
- Suppresses COM speed peaks
- Flattens paw vertical oscillation
- Dampens joint angle changes
- Reduces angular velocity
- Smooths out micro-step variance

**Impact:** "Flat" trajectories lead to "flat" metrics — velocity, acceleration, angular dynamics all compressed

---

### 2.3 MAD-Based Thresholds Too Conservative

**Current Implementation:**
```python
# phase_detector.py
walking_mad_threshold: 2.0  # Standard
walking_mad_threshold: 1.2  # Adaptive
stationary_mad_threshold: 1.5
prominence_multiplier: 0.5
```

**Problem:**
- Tuned for **high-speed treadmill locomotion**
- Open-field gait characteristics:
  - Micro-steps (6-9 frames)
  - Short bursts
  - Low peak prominence
  - Irregular swing/stance patterns
- MAD over-filters → legitimate steps dropped

**Impact:** Cadence, stride count, walking percentage all underestimated

---

### 2.4 COM Velocity Uses Only 2D Data

**Current Implementation:**
```python
# Only XY from TOP view
com_velocity = np.sqrt(dx**2 + dy**2)
```

**Problem:**
- Ignores Z-axis movement from SIDE view
- Vertical hops, rearing, body oscillation ignored
- Underestimates true 3D movement

**Impact:** Speed underestimated by 10-20%

---

### 2.5 Stride Validation Rejects Valid Steps

**Current Implementation:**
```python
# step_detector.py
min_stride_duration: 0.1s  # 12 frames at 120 FPS
max_stride_duration: 1.0s
```

**Problem:**
- Open-field micro-steps often **6-9 frames** (0.05-0.075s)
- These are REAL steps but get rejected
- Halves stride count in exploratory behavior

**Impact:** Cadence cut in half, stride metrics incomplete

---

### 2.6 Aggregator Uses Only Median + MAD

**Current Implementation:**
```python
# aggregator.py
# Uses median + MAD exclusively
```

**Problem:**
- MAD is robust but hides subtle motion
- Compresses distribution variance
- No confidence intervals
- No bias correction tracking

**Impact:** Summary statistics systematically conservative

---

## 3. Fix Scope

### Objectives

1. ✅ Eliminate systemic under-calculation across all metrics
2. ✅ Recalibrate scaling to actual mouse body dimensions
3. ✅ Loosen or auto-adapt detection thresholds
4. ✅ Reduce excessive trajectory smoothing
5. ✅ Reconstruct COM using full 3D data
6. ✅ Allow short strides and micro-steps
7. ✅ Adjust gait validation rules for exploratory behavior
8. ✅ Recompute ROM using angle-based measures
9. ✅ Upgrade visualizer to show corrected distributions

---

## 4. Detailed Requirements

### 4.1 Scaling Factor Correction

**New Required Behavior:**

1. **Use full body length measurement**:
   - Keypoints: snout → tailbase (not just spine1-3)
   - Compute dynamic length per frame
   - Use median length across high-confidence frames

2. **Updated expected body length**:
   - Default: **10.0 cm** (not 8.0 cm)
   - User-configurable via config

3. **Quality filtering**:
   - Reject frames with likelihood < 0.9
   - Remove outliers (>20% deviation from median)

4. **Per-sample calibration**:
   - Compute scaling factor individually per sample
   - Log scaling diagnostics to metadata

**Implementation:**
```python
def compute_scaling_factor_v2(
    snout_trajectory: np.ndarray,
    tailbase_trajectory: np.ndarray,
    likelihood_snout: np.ndarray,
    likelihood_tail: np.ndarray,
    expected_body_length_cm: float = 10.0,
    min_likelihood: float = 0.9
) -> Tuple[float, Dict]:
    """
    Compute scaling factor using full body length.

    Returns:
        scaling_factor (float): cm/pixel ratio
        diagnostics (Dict): Quality metrics
    """
    # Filter high-confidence frames
    valid = (likelihood_snout > min_likelihood) & (likelihood_tail > min_likelihood)

    # Compute per-frame body length in pixels
    body_length_px = np.linalg.norm(
        snout_trajectory[valid] - tailbase_trajectory[valid],
        axis=1
    )

    # Remove outliers
    median_length = np.median(body_length_px)
    mad = np.median(np.abs(body_length_px - median_length))
    threshold = median_length + 3 * mad

    inliers = body_length_px[body_length_px < threshold]

    # Compute scaling factor
    scaling_factor = expected_body_length_cm / np.median(inliers)

    diagnostics = {
        'median_length_px': np.median(inliers),
        'expected_length_cm': expected_body_length_cm,
        'scaling_cm_per_px': scaling_factor,
        'valid_frames': np.sum(valid),
        'confidence': np.mean(valid)
    }

    return scaling_factor, diagnostics
```

**Config Update:**
```yaml
global_settings:
  expected_body_length_cm: 10.0    # Updated from 8.0
  scaling_method: "full_body"      # New: "full_body" or "spine_only"
  scaling_min_likelihood: 0.9      # New parameter
```

**Acceptance Criteria:**
- ✅ Scaling factor increases by ~20-25%
- ✅ Stride length, step width proportionally increase
- ✅ Diagnostic plot shows before/after scaling comparison

---

### 4.2 Smoothing Reduction

**New Required Behavior:**

1. **Reduce default smoothing windows**:
   - Position data: 11 → **7**
   - Velocity data: 11 → **5**
   - ROM data: 11 → **3**
   - COM data: 15 → **7**

2. **Introduce hybrid smoothing**:
   - Savitzky-Golay for **position**
   - Exponential Moving Average (EMA) for **velocity**

3. **Adaptive smoothing**:
   - High-quality data (completeness >80%): minimal smoothing
   - Low-quality data (completeness <60%): standard smoothing

**Implementation:**
```python
def smooth_trajectory_adaptive(
    trajectory: np.ndarray,
    data_completeness: float,
    window_size_base: int = 7
) -> np.ndarray:
    """
    Adaptive smoothing based on data quality.
    """
    # Adjust window based on quality
    if data_completeness > 0.8:
        window = max(3, window_size_base - 2)
    elif data_completeness < 0.6:
        window = window_size_base + 2
    else:
        window = window_size_base

    # Ensure odd window
    window = window if window % 2 == 1 else window + 1

    return savgol_filter(trajectory, window, polyorder=3, axis=0)

def compute_velocity_ema(
    positions: np.ndarray,
    alpha: float = 0.35,
    fps: float = 120.0
) -> np.ndarray:
    """
    Compute velocity using Exponential Moving Average.

    Args:
        alpha: Smoothing factor (0-1), higher = less smoothing
    """
    # Raw velocity
    velocity = np.diff(positions, axis=0) * fps

    # EMA smoothing
    smoothed = np.zeros_like(velocity)
    smoothed[0] = velocity[0]

    for i in range(1, len(velocity)):
        smoothed[i] = alpha * velocity[i] + (1 - alpha) * smoothed[i-1]

    return smoothed
```

**Config Update:**
```yaml
global_settings:
  smoothing_window: 7              # Reduced from 11
  smoothing_window_velocity: 5     # New: separate for velocity
  smoothing_window_rom: 3          # New: separate for ROM
  smoothing_adaptive: true         # New: enable adaptive smoothing
  velocity_ema_alpha: 0.35         # New: EMA smoothing factor
```

**Acceptance Criteria:**
- ✅ Peak velocities increase by 15-25%
- ✅ Angular velocity increases by 2-4×
- ✅ ROM range increases by 30-50%
- ✅ Micro-step peaks visible in trajectory plots

---

### 4.3 Threshold Calibration Fix

**New Required Behavior:**

1. **Hybrid threshold formula**:
   ```python
   threshold = median + k*MAD + percentile(velocity, percentile_value)
   ```

2. **Relaxed default parameters**:
   - `walking_mad_threshold`: 1.2 → **0.8**
   - `stationary_mad_threshold`: 1.5 → **0.9**
   - `prominence_multiplier`: 0.5 → **0.25**
   - `adaptive_percentile`: 75 → **55**

3. **Safety lower bound**:
   - Minimum threshold = **1 pixel/frame** (prevents zero-threshold edge case)

**Implementation:**
```python
def compute_adaptive_threshold_v2(
    velocity: np.ndarray,
    mad_multiplier: float = 0.8,
    percentile_value: int = 55,
    min_threshold_px_per_frame: float = 1.0
) -> float:
    """
    Compute adaptive threshold using hybrid MAD + percentile method.
    """
    median_vel = np.median(velocity)
    mad = np.median(np.abs(velocity - median_vel))
    percentile_vel = np.percentile(velocity, percentile_value)

    # Hybrid threshold
    threshold = median_vel + mad_multiplier * mad + 0.5 * percentile_vel

    # Apply lower bound
    threshold = max(threshold, min_threshold_px_per_frame)

    return threshold
```

**Config Update:**
```yaml
global_settings:
  walking_mad_threshold: 0.8       # Reduced from 1.2
  stationary_mad_threshold: 0.9    # Reduced from 1.5
  prominence_multiplier: 0.25      # Reduced from 0.5
  adaptive_percentile: 55          # Reduced from 75
  min_threshold_px_per_frame: 1.0  # New safety bound
  use_hybrid_threshold: true       # New: enable hybrid mode
```

**Acceptance Criteria:**
- ✅ Walking detection increases by 10-20%
- ✅ More micro-steps detected
- ✅ Cadence increases by 15-35%

---

### 4.4 Step Detector Sensitivity Increase

**New Required Behavior:**

1. **Allow shorter strides**:
   - `min_stride_duration`: 0.1s → **0.06s** (7 frames at 120 FPS)
   - `max_stride_duration`: 1.0s → **1.5s**

2. **Micro-step detection**:
   - Strides < 1 cm labeled as "micro-stride"
   - Still included in cadence calculation
   - Flagged in output for analysis

3. **Relaxed prominence requirement**:
   - Allow lower prominence peaks
   - Adaptive prominence based on data quality

**Implementation:**
```python
def detect_steps_v2(
    paw_trajectory: np.ndarray,
    fps: float = 120.0,
    min_stride_duration: float = 0.06,
    max_stride_duration: float = 1.5,
    prominence_multiplier: float = 0.25,
    allow_micro_steps: bool = True
) -> List[Dict]:
    """
    Enhanced step detection with micro-step support.
    """
    # Existing detection logic...
    strides = []

    for stride in detected_strides:
        stride_info = {
            'start_frame': stride['start'],
            'end_frame': stride['end'],
            'duration_sec': stride['duration'],
            'length_cm': stride['length'],
            'is_micro_step': stride['length'] < 1.0  # Flag micro-steps
        }

        # Include all strides, even micro-steps
        if allow_micro_steps or stride['length'] >= 1.0:
            strides.append(stride_info)

    return strides
```

**Config Update:**
```yaml
global_settings:
  min_stride_duration: 0.06        # Reduced from 0.1
  max_stride_duration: 1.5         # Increased from 1.0
  allow_micro_steps: true          # New parameter
  micro_step_threshold_cm: 1.0     # New: threshold for micro-step classification
```

**Acceptance Criteria:**
- ✅ Stride count increases by 30-50%
- ✅ Micro-steps properly flagged
- ✅ Cadence calculations include all steps

---

### 4.5 COM Reconstruction Upgrade

**New Required Behavior:**

1. **3D COM calculation**:
   - Combine TOP (XY) + SIDE (Z) views
   - Weighted average of body keypoints

2. **3D velocity**:
   ```python
   speed = sqrt(dx^2 + dy^2 + dz^2) * scaling
   ```

3. **Vertical movement tracking**:
   - Detect rearing, hops
   - Include in activity classification

**Implementation:**
```python
def compute_com_3d(
    top_keypoints: Dict[str, np.ndarray],
    side_keypoints: Dict[str, np.ndarray],
    weights: Dict[str, float] = None
) -> np.ndarray:
    """
    Compute 3D center of mass from multi-view data.

    Returns:
        com_3d: (N, 3) array of (x, y, z) positions
    """
    if weights is None:
        weights = {
            'spine1': 0.15,
            'spine2': 0.20,
            'spine3': 0.20,
            'tailbase': 0.15,
            'nose': 0.10,
            'hip_R': 0.10,
            'hip_L': 0.10
        }

    # Extract XY from TOP view
    com_xy = np.zeros((len(top_keypoints['spine1']), 2))

    for kp, weight in weights.items():
        if kp in top_keypoints:
            com_xy += weight * top_keypoints[kp][:, :2]

    # Extract Z from SIDE view
    com_z = np.zeros(len(side_keypoints['spine2']))

    for kp, weight in weights.items():
        if kp in side_keypoints:
            com_z += weight * side_keypoints[kp][:, 1]  # Y in SIDE = Z in 3D

    # Combine into 3D
    com_3d = np.column_stack([com_xy, com_z])

    return com_3d

def compute_velocity_3d(
    com_3d: np.ndarray,
    fps: float = 120.0,
    scaling_factor: float = 1.0
) -> np.ndarray:
    """
    Compute 3D velocity magnitude.
    """
    # Compute deltas
    delta = np.diff(com_3d, axis=0)

    # 3D speed
    speed = np.linalg.norm(delta, axis=1) * fps * scaling_factor

    return speed
```

**Config Update:**
```yaml
global_settings:
  use_3d_com: true                 # New: enable 3D COM
  com_weights:                     # New: keypoint weights
    spine1: 0.15
    spine2: 0.20
    spine3: 0.20
    tailbase: 0.15
    nose: 0.10
    hip_R: 0.10
    hip_L: 0.10
```

**Acceptance Criteria:**
- ✅ Speed increases by 10-20%
- ✅ Vertical movement properly captured
- ✅ Rearing/hop detection improved

---

### 4.6 ROM Reconstruction Fix

**New Required Behavior:**

1. **Use triplet keypoints for angles**:
   - Hip angle: spine3 → hip → knee
   - Elbow angle: shoulder → elbow → paw

2. **SIDE view for sagittal plane angles**:
   - True flexion/extension measurement
   - Not projection artifacts

3. **Reduced smoothing for ROM**:
   - Window size: 3 (was 11)

4. **Angular velocity from differentiation**:
   - After minimal smoothing
   - Scaled to deg/s properly

**Implementation:**
```python
def compute_joint_angle(
    proximal: np.ndarray,
    joint: np.ndarray,
    distal: np.ndarray
) -> np.ndarray:
    """
    Compute joint angle from three keypoints.

    Returns:
        angles: (N,) array of angles in degrees
    """
    # Vectors
    v1 = proximal - joint
    v2 = distal - joint

    # Angle via dot product
    cos_angle = np.sum(v1 * v2, axis=1) / (
        np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    )

    # Clip to valid range
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    # Convert to degrees
    angles = np.degrees(np.arccos(cos_angle))

    return angles

def compute_rom_v2(
    side_keypoints: Dict[str, np.ndarray],
    smoothing_window: int = 3
) -> Dict:
    """
    Compute range of motion metrics with proper angle calculation.
    """
    rom_metrics = {}

    # Hip ROM (right)
    if all(k in side_keypoints for k in ['spine3', 'hip_R', 'knee_R']):
        hip_angles = compute_joint_angle(
            side_keypoints['spine3'],
            side_keypoints['hip_R'],
            side_keypoints['knee_R']
        )

        # Minimal smoothing
        if smoothing_window > 0:
            hip_angles = savgol_filter(hip_angles, smoothing_window, polyorder=2)

        # Angular velocity
        angular_vel = np.diff(hip_angles) * fps  # deg/s

        rom_metrics['hip_R'] = {
            'angles': hip_angles,
            'rom_deg': np.max(hip_angles) - np.min(hip_angles),
            'mean_angle_deg': np.mean(hip_angles),
            'angular_velocity_deg_per_sec': angular_vel
        }

    return rom_metrics
```

**Config Update:**
```yaml
global_settings:
  rom_smoothing_window: 3          # Reduced from 11
  rom_use_triplet_angles: true     # New: use 3-point angles
  rom_view: "side"                 # New: specify which view for ROM
```

**Acceptance Criteria:**
- ✅ ROM increases by 30-50%
- ✅ Angular velocity increases by 2-4×
- ✅ Angle-based measurement more accurate

---

### 4.7 Aggregated Metrics Correction

**New Required Behavior:**

1. **Enhanced statistics**:
   - Median + MAD (existing)
   - **95% Confidence Interval** (new)
   - **Corrected mean** (new)
   - **Bias metrics** (new)

2. **Track corrections**:
   - Median bias (raw vs smoothed)
   - Scaling bias (pre vs post correction)

**Implementation:**
```python
def aggregate_metrics_v2(
    metric_values: np.ndarray
) -> Dict:
    """
    Compute enhanced statistical aggregations.
    """
    # Existing
    median = np.median(metric_values)
    mad = np.median(np.abs(metric_values - median))

    # New: Confidence intervals
    ci_low = np.percentile(metric_values, 2.5)
    ci_high = np.percentile(metric_values, 97.5)

    # New: Corrected mean (trim extreme 5%)
    trimmed = metric_values[
        (metric_values >= ci_low) & (metric_values <= ci_high)
    ]
    corrected_mean = np.mean(trimmed)

    return {
        'median': median,
        'mad': mad,
        'mean': np.mean(metric_values),
        'std': np.std(metric_values),
        'corrected_mean': corrected_mean,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'n_samples': len(metric_values),
        'ci_range': ci_high - ci_low
    }
```

**Config Update:**
```yaml
global_settings:
  aggregation_include_ci: true     # New: compute confidence intervals
  aggregation_ci_percentile: 95    # New: CI level
  aggregation_trim_percent: 5      # New: trim for corrected mean
```

**Acceptance Criteria:**
- ✅ Excel reports include CI columns
- ✅ Bias metrics tracked
- ✅ More complete statistical picture

---

## 5. Acceptance Criteria

| Metric | Target Improvement | Validation Method |
|--------|-------------------|-------------------|
| Stride length | +20-40% | Manual annotation comparison |
| Cadence | +15-35% | Video frame counting |
| Duty cycle | Less stance-biased | EMG validation (if available) |
| ROM | +30-50% | Manual angle measurement |
| Angular velocity | ×2-×4 increase | Derivative validation |
| COM sway | +15-25% | Trajectory visual inspection |
| Regularity index | Less compressed | Expected 0.7-0.9 for healthy |
| Phase dispersion | >0.05 | Expected variability in exploratory |

**Overall Success Criteria:**
- ✅ All metrics increase to physiologically realistic ranges
- ✅ Plots show dynamic range and variance
- ✅ Before/after comparison shows clear improvement
- ✅ No new false positives introduced
- ✅ Processing time increase <20%

---

## 6. Validation Tests

### Test Dataset

**3 Open-Field Mice**:
- Low-activity exploratory behavior
- Expected to show maximum improvement

**2 Treadmill Mice**:
- High-activity forced locomotion
- Should maintain accuracy, minimal over-correction

**1 High-Speed Mouse**:
- Fast running
- Ensure thresholds don't over-detect

### Validation Methods

1. **Manual Annotation** (Gold Standard):
   - 50 manually counted strides per sample
   - Compare automated vs manual stride count
   - Target: >90% agreement

2. **Visual Inspection**:
   - Before/after plot comparison
   - Trajectory overlay on video
   - ROM angle measurement validation

3. **EMG Comparison** (if available):
   - Compare stride timing to muscle activation
   - Validate swing/stance ratios

4. **Cross-Validation**:
   - Compare v1.1.0 vs v1.2.0 on same samples
   - Document all changes
   - Ensure improvements not artifacts

---

## 7. Visualizer Updates

### Required Changes to `EnhancedDashboardVisualizer`

1. **Wider Y-axis ranges**:
   ```python
   # Allow auto-scaling with 20% margin (was 10%)
   self.style.format_axis_range(ax, data, margin=0.20)
   ```

2. **Micro-stride labeling**:
   ```python
   # Add legend entry for micro-strides
   if has_micro_steps:
       ax.scatter(..., marker='o', s=30, label='Micro-strides')
   ```

3. **COM 3D path overlay**:
   ```python
   # New subplot showing 3D trajectory
   def plot_com_3d_trajectory(self, com_3d):
       # XY projection + Z height overlay
   ```

4. **Angular velocity histogram**:
   ```python
   # Add to ROM dashboard
   def plot_angular_velocity_distribution(self, angular_vel):
       ax.hist(angular_vel, bins=30, ...)
   ```

5. **Dynamic threshold bands**:
   ```python
   # Show threshold values on phase detection plots
   ax.axhline(walking_threshold, linestyle='--',
              label=f'Walking threshold ({walking_threshold:.2f})')
   ```

6. **Bias correction overlay** (optional):
   ```python
   # Toggle to show raw vs corrected metrics
   if show_bias_correction:
       ax.scatter(x, raw_values, alpha=0.3, label='Raw')
       ax.scatter(x, corrected_values, label='Corrected')
   ```

**Config Update:**
```yaml
global_settings:
  plot_y_margin: 0.20              # Increased from 0.10
  plot_show_micro_steps: true      # New
  plot_show_3d_com: true           # New
  plot_show_threshold_lines: true  # New
  plot_show_bias_correction: false # New (optional)
```

---

## 8. Implementation Plan

### Phase 1: Core Fixes (Week 1)

| Module | Changes | Owner | Status |
|--------|---------|-------|--------|
| `preprocessor.py` | Scaling fix, smoothing reduction | Dev | Planned |
| `phase_detector.py` | Hybrid threshold, relaxed params | Dev | Planned |
| `step_detector.py` | Min stride duration, micro-steps | Dev | Planned |

### Phase 2: Metrics & Aggregation (Week 2)

| Module | Changes | Owner | Status |
|--------|---------|-------|--------|
| `metrics_computer.py` | 3D CoM, ROM angle fix | Dev | Planned |
| `aggregator.py` | CI + bias metrics | Dev | Planned |

### Phase 3: Visualization & Config (Week 3)

| Module | Changes | Owner | Status |
|--------|---------|-------|--------|
| `visualizer_enhanced.py` | Axis ranges, micro-step labels | Dev | Planned |
| `config_v1.2.yaml` | New default parameters | Dev | Planned |
| `diagnose_thresholds.py` | Updated diagnostic logic | Dev | Planned |

### Phase 4: Validation & Documentation (Week 4)

| Task | Description | Owner | Status |
|------|-------------|-------|--------|
| Manual validation | 50 strides × 6 samples | QA | Planned |
| Before/after report | Generate comparison PDF | Dev | Planned |
| Documentation | Update API docs, user guides | Dev | Planned |
| Release notes | v1.2.0 changelog | PM | Planned |

---

## 9. Deliverables

### Code Deliverables

1. ✅ `src/exmo_gait/core/preprocessor.py` (updated)
2. ✅ `src/exmo_gait/analysis/phase_detector.py` (updated)
3. ✅ `src/exmo_gait/analysis/step_detector.py` (updated)
4. ✅ `src/exmo_gait/analysis/metrics_computer.py` (updated)
5. ✅ `src/exmo_gait/statistics/aggregator.py` (updated)
6. ✅ `src/exmo_gait/export/visualizer_enhanced.py` (updated)
7. ✅ `config_v1.2.yaml` (new)
8. ✅ `diagnose_thresholds_v2.py` (updated)
9. ✅ `validate_calibration.py` (new script)

### Documentation Deliverables

1. ✅ `METRIC_CALIBRATION_PRD.md` (this document)
2. ✅ `METRIC_CALIBRATION_IMPLEMENTATION.md` (technical guide)
3. ✅ `BEFORE_AFTER_CALIBRATION_REPORT.pdf` (validation results)
4. ✅ Updated `docs/SYSTEM_OVERVIEW.md` (v1.2.0 section)
5. ✅ Updated `docs/API_REFERENCE.md` (new parameters)
6. ✅ `TROUBLESHOOTING_v1.2.md` (calibration issues)

### Validation Deliverables

1. ✅ Before/after metric comparison table
2. ✅ Visual comparison plots (6 samples)
3. ✅ Manual validation report
4. ✅ Performance benchmarks

---

## 10. Risks & Mitigation

### Risk 1: Over-Correction

**Description:** Relaxed thresholds might detect noise as steps

**Mitigation:**
- Implement quality scoring for detected steps
- Add manual review flags for suspicious metrics
- Provide "conservative" vs "sensitive" config presets

### Risk 2: Breaking Existing Workflows

**Description:** v1.2.0 changes might break user scripts

**Mitigation:**
- Maintain backward compatibility flag
- Provide migration guide
- Keep v1.1.0 config as `config_legacy.yaml`

### Risk 3: Performance Degradation

**Description:** 3D COM and enhanced metrics slow processing

**Mitigation:**
- Profile before/after
- Optimize bottlenecks
- Target <20% slowdown
- Provide "fast" mode that skips 3D COM

### Risk 4: Validation Failures

**Description:** Manual annotation doesn't match automated

**Mitigation:**
- Use multiple annotators
- Document disagreements
- Adjust thresholds iteratively
- Accept 10-15% variance as normal

---

## 11. Success Metrics

### Quantitative

- ✅ Stride length: median increases 20-40%
- ✅ Cadence: median increases 15-35%
- ✅ ROM: range increases 30-50%
- ✅ Angular velocity: increases 2-4×
- ✅ Processing time: <20% slower
- ✅ Manual validation: >85% agreement

### Qualitative

- ✅ Plots visually show more variance
- ✅ Metrics in physiologically realistic ranges
- ✅ User feedback: "metrics make sense now"
- ✅ Reduced troubleshooting requests

---

## 12. Timeline

**Week 1:** Core fixes (scaling, smoothing, thresholds)
**Week 2:** Metrics upgrade (3D COM, ROM, aggregation)
**Week 3:** Visualization & config
**Week 4:** Validation & documentation

**Target Release:** v1.2.0 by Week 4

---

## 13. Appendix: Before/After Expected Values

### Example: control_5 Sample

**Metric** | **v1.1.0 (Before)** | **v1.2.0 (After)** | **Change**
-----------|---------------------|--------------------|-----------
Stride Length (cm) | 2.3 ± 0.8 | 3.5 ± 1.2 | +52%
Cadence (steps/min) | 26.7 | 38.4 | +44%
Duty Cycle (%) | 83.0 | 67.5 | -19%
Hip ROM (deg) | 15.2 | 28.7 | +89%
Angular Velocity (deg/s) | 12.3 | 41.2 | +235%
COM Sway ML (cm) | 0.8 | 1.2 | +50%
Walking % | 27.0% | 35.8% | +33%

**Interpretation:** All metrics move toward physiologically realistic values for open-field exploratory behavior.

---

**PRD Version:** 1.0
**Date:** 2025-11-21
**Status:** APPROVED FOR IMPLEMENTATION
**Next Action:** Generate config_v1.2.yaml and begin Phase 1 implementation


# Config-to-Code Mapping: v1.2.0 Methods

Quick reference showing which config flags activate which v1.2.0 methods.

---

## 1. Full-Body Scaling

### Config (config_v1.2_calibrated.yaml)
```yaml
global_settings:
  scaling_method: "full_body"           # LINE 39
  expected_body_length_cm: 10.0         # LINE 38
  scaling_min_likelihood: 0.9           # LINE 41
  scaling_tolerance: 0.25               # LINE 42
```

### Code (cli.py:149-162)
```python
if scaling_method == 'full_body':
    scale_factor, diagnostics = preprocessor.compute_scale_factor_v2(
        snout, tail_base,
        expected_body_length_cm=expected_body_length,
        min_likelihood=min_likelihood,
        tolerance=tolerance
    )
```

### Calls (preprocessor.py:76-116)
```python
def compute_scale_factor_v2(self, snout, tailbase, ...):
    # Uses full snout→tailbase distance
    # Filters by likelihood threshold
    # Removes outliers beyond tolerance
```

---

## 2. Adaptive Smoothing

### Config (config_v1.2_calibrated.yaml)
```yaml
global_settings:
  smoothing_adaptive: true              # LINE 51
  smoothing_window: 7                   # LINE 49 (base window)
```

### Code (cli.py:178-203)
```python
if use_adaptive_smoothing:
    data_completeness = np.sum(~np.isnan(kp_data[:, 0])) / len(kp_data)
    smoothed_x = smooth_trajectory_adaptive(
        kp_data[:, 0],
        data_completeness,
        window_size_base=gs.get('smoothing_window', 7),
        polyorder=gs.get('smoothing_poly', 3)
    )
```

### Calls (signal_processing.py:228-267)
```python
def smooth_trajectory_adaptive(trajectory, data_completeness, ...):
    # Adjusts window based on data quality:
    # High quality (>0.9): window - 2
    # Medium (0.7-0.9): base window
    # Low (<0.7): window + 2
```

---

## 3. 3D COM Calculation

### Config (config_v1.2_calibrated.yaml)
```yaml
global_settings:
  use_3d_com: true                      # LINE 107
  com_weights:                          # LINES 111-118
    spine1: 0.15
    spine2: 0.20
    spine3: 0.20
    tailbase: 0.15
    nose: 0.10
    hip_R: 0.10
    hip_L: 0.10
```

### Code (cli.py:220-241)
```python
if use_3d_com:
    top_keypoints = {k: v for k, v in keypoints_preprocessed.items()
                   if k in ['snout', 'neck', 'tail_base', ...]}
    side_keypoints = {k: v for k, v in keypoints_preprocessed.items()
                    if k in ['hip_R', 'hip_L', ...]}

    com_trajectory = gait_computer.compute_com_3d(
        top_keypoints,
        side_keypoints,
        weights=com_weights
    )
```

### Calls (metrics_computer.py:454-509)
```python
def compute_com_3d(self, top_keypoints, side_keypoints, weights):
    # Extracts XY from TOP view
    # Extracts Z from SIDE view (Y coordinate)
    # Computes weighted 3D COM
    # Returns (N, 3) array
```

---

## 4. Hybrid Threshold Phase Detection

### Config (config_v1.2_calibrated.yaml)
```yaml
global_settings:
  use_hybrid_threshold: true            # LINE 70
  min_threshold_px_per_frame: 1.0       # LINE 71
  walking_mad_threshold: 0.8            # LINE 75
  adaptive_percentile: 55               # LINE 85
```

### Code (cli.py:264-273)
```python
phase_detector = PhaseDetector(
    fps=gs.get('fps', 120.0),
    stationary_mad_threshold=gs.get('stationary_mad_threshold', 1.5),
    walking_mad_threshold=gs.get('walking_mad_threshold', 2.0),
    min_walking_duration=gs.get('min_walking_duration', 0.3),
    min_stationary_duration=gs.get('min_stationary_duration', 0.25),
    use_hybrid_threshold=use_hybrid_threshold,  # v1.2.0 flag
    adaptive_percentile=gs.get('adaptive_percentile', 75),
    min_threshold_px_per_frame=gs.get('min_threshold_px_per_frame', 1.0)
)
```

### Calls (phase_detector.py:93-158)
```python
def compute_hybrid_threshold(self, com_speed, mad_multiplier):
    # MAD component: mad * multiplier
    # Percentile component: np.percentile(com_speed, percentile)
    # Hybrid: (mad_threshold + percentile_threshold) / 2
    # Safety: max(hybrid, min_threshold)

def detect_walking_phase(self, com_speed):
    if self.use_hybrid_threshold:
        threshold = self.compute_hybrid_threshold(...)  # v1.2.0
    else:
        threshold = mad * self.walking_mad_threshold    # v1.1.0
```

---

## 5. EMA Velocity Smoothing

### Config (config_v1.2_calibrated.yaml)
```yaml
global_settings:
  velocity_smoothing_method: "ema"      # LINE 55
  velocity_ema_alpha: 0.35              # LINE 56
```

### Code (cli.py:306-313)
```python
if use_ema_velocity:
    ema_alpha = gs.get('velocity_ema_alpha', 0.35)
    gait_computer.velocity_smoothing_method = 'ema'
    gait_computer.ema_alpha = ema_alpha
else:
    gait_computer.velocity_smoothing_method = 'savgol'
```

### Calls (signal_processing.py:194-224)
```python
def smooth_velocity_ema(positions, alpha=0.35, fps=120.0):
    # Compute raw velocity: gradient / dt
    # Apply EMA: v[i] = alpha*raw[i] + (1-alpha)*v[i-1]
    # Returns smoothed velocity array
```

**Note:** GaitMetricsComputer must check `velocity_smoothing_method` attribute and call appropriate method.

---

## 6. Enhanced Statistics with CI

### Config (config_v1.2_calibrated.yaml)
```yaml
global_settings:
  aggregation_include_ci: true          # LINE 137
  aggregation_ci_percentile: 95         # LINE 138
  aggregation_trim_percent: 5           # LINE 139
```

### Code (cli.py:338-351)
```python
if include_ci:
    ci_percentile = gs.get('aggregation_ci_percentile', 95)
    trim_percent = gs.get('aggregation_trim_percent', 5)

    aggregator.use_enhanced_stats = True
    aggregator.ci_percentile = ci_percentile
    aggregator.trim_percent = trim_percent
else:
    aggregator.use_enhanced_stats = False
```

### Calls (aggregator.py:48-122, 155-163, 197-205)
```python
# In __init__:
def __init__(self):
    self.use_enhanced_stats = False
    self.ci_percentile = 95
    self.trim_percent = 5

# In aggregate methods:
if self.use_enhanced_stats:
    stats = self.compute_summary_stats_v2(
        metric_value,
        include_ci=True,
        ci_percentile=self.ci_percentile,
        trim_percent=self.trim_percent
    )
else:
    stats = self.compute_summary_stats(metric_value)  # v1.1.0
```

### Method (aggregator.py:48-122)
```python
@staticmethod
def compute_summary_stats_v2(values, include_ci=True, ci_percentile=95, trim_percent=5):
    # Basic: median, std, mad, mean, min, max, count
    # Enhanced: corrected_mean (trimmed)
    # Enhanced: ci_low, ci_high, ci_range (percentile-based)
    return {
        'median': ...,
        'corrected_mean': stats.trim_mean(values, trim_fraction),
        'ci_low': np.percentile(values, (100-ci_percentile)/2),
        'ci_high': np.percentile(values, 100-(100-ci_percentile)/2),
        'ci_range': ci_high - ci_low,
        ...
    }
```

---

## Quick Config Toggle Reference

### Enable ALL v1.2.0 Methods
```yaml
global_settings:
  scaling_method: "full_body"
  smoothing_adaptive: true
  velocity_smoothing_method: "ema"
  use_hybrid_threshold: true
  use_3d_com: true
  aggregation_include_ci: true
```

### Disable ALL v1.2.0 Methods (Legacy v1.1.0)
```yaml
global_settings:
  # scaling_method: "spine_only"  # or omit
  # smoothing_adaptive: false     # or omit
  # velocity_smoothing_method: "savgol"  # or omit
  # use_hybrid_threshold: false   # or omit
  # use_3d_com: false             # or omit
  # aggregation_include_ci: false # or omit
```

### Selective v1.2.0 Features (Example)
```yaml
global_settings:
  scaling_method: "full_body"           # v1.2.0 ONLY scaling
  smoothing_adaptive: false             # v1.1.0 smoothing
  use_hybrid_threshold: true            # v1.2.0 ONLY threshold
  use_3d_com: false                     # v1.1.0 COM
  aggregation_include_ci: true          # v1.2.0 ONLY stats
```

---

## Default Values (When Config Omits Flags)

| Flag | Default | Version Used |
|------|---------|--------------|
| `scaling_method` | `"spine_only"` | v1.1.0 |
| `smoothing_adaptive` | `false` | v1.1.0 |
| `velocity_smoothing_method` | `"savgol"` | v1.1.0 |
| `use_hybrid_threshold` | `false` | v1.1.0 |
| `use_3d_com` | `false` | v1.1.0 |
| `aggregation_include_ci` | `false` | v1.1.0 |

**Backward Compatibility:** Old configs without these flags will use v1.1.0 methods by default.

---

## Validation: How to Verify Methods Are Active

### 1. Check Logs
```
Pipeline Configuration:
  - Scaling: v1.2.0 full-body method (expected +20-25% distance accuracy)
  - Smoothing: v1.2.0 adaptive method (expected +15-25% peak preservation)
  ...
```

### 2. Check Metadata in Excel Output
Look for `pipeline_version` and `methods_used` in metadata sheet:
```json
{
  "pipeline_version": "v1.2.0",
  "methods_used": {
    "scaling": "full_body",
    "adaptive_smoothing": true,
    "ema_velocity": true,
    "hybrid_threshold": true,
    "3d_com": true,
    "enhanced_stats": true
  }
}
```

### 3. Check Statistics Columns
v1.2.0 enhanced stats include extra columns:
- `corrected_mean` (trimmed mean)
- `ci_low` (95% CI lower bound)
- `ci_high` (95% CI upper bound)
- `ci_range` (CI width)

v1.1.0 only has: median, std, mad, mean, min, max, count

---

**Last Updated: 2025-11-21**

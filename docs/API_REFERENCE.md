# EXMO Gait Analysis Pipeline - API Reference

## Table of Contents
- [Core Modules](#core-modules)
- [Analysis Modules](#analysis-modules)
- [Export Modules](#export-modules)
- [Utility Functions](#utility-functions)
- [Configuration API](#configuration-api)
- [Examples](#examples)

---

## Core Modules

### `src/exmo_gait/core/data_loader.py`

#### `load_multiview_data()`

Load and synchronize multi-view DeepLabCut tracking data.

**Signature**:
```python
def load_multiview_data(
    top_csv: Path,
    side_csv: Path,
    bottom_csv: Path,
    fps: float = 120.0
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]
```

**Parameters**:
- `top_csv` (Path): Path to TOP view CSV file
- `side_csv` (Path): Path to SIDE view CSV file
- `bottom_csv` (Path): Path to BOTTOM view CSV file
- `fps` (float, optional): Frame rate in Hz. Default: 120.0

**Returns**:
- `Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]`:
  - TOP view DataFrame
  - SIDE view DataFrame
  - BOTTOM view DataFrame
  - Metadata dictionary with keys:
    - `'n_frames'`: Synchronized frame count
    - `'duration_sec'`: Recording duration
    - `'fps'`: Frame rate

**Raises**:
- `FileNotFoundError`: If any CSV file doesn't exist
- `ValueError`: If keypoint structure is invalid
- `ValueError`: If frame counts differ significantly (>10%)

**Example**:
```python
from pathlib import Path
from exmo_gait.core.data_loader import load_multiview_data

top, side, bottom, metadata = load_multiview_data(
    top_csv=Path("Data/TOP/sample_1.csv"),
    side_csv=Path("Data/SIDE/sample_1.csv"),
    bottom_csv=Path("Data/BOTTOM/sample_1.csv"),
    fps=120.0
)

print(f"Loaded {metadata['n_frames']} frames")
print(f"Duration: {metadata['duration_sec']:.2f} seconds")
```

---

### `src/exmo_gait/core/preprocessor.py`

#### `Preprocessor` Class

Handles data cleaning, smoothing, and scaling.

**Constructor**:
```python
def __init__(
    self,
    smoothing_window: int = 11,
    smoothing_poly: int = 3,
    outlier_threshold: float = 3.0,
    max_interpolation_gap: int = 5
)
```

**Parameters**:
- `smoothing_window` (int): Savitzky-Golay window size (odd number). Default: 11
- `smoothing_poly` (int): Polynomial order for smoothing. Default: 3
- `outlier_threshold` (float): Z-score threshold for outlier removal. Default: 3.0
- `max_interpolation_gap` (int): Maximum gap size for interpolation (frames). Default: 5

#### `compute_scaling_factor()`

Calculate cm/pixel conversion ratio from body length.

**Signature**:
```python
def compute_scaling_factor(
    self,
    spine_keypoints: Dict[str, np.ndarray],
    expected_body_length_cm: float = 8.0,
    tolerance: float = 0.20
) -> float
```

**Parameters**:
- `spine_keypoints` (Dict[str, np.ndarray]): Spine keypoint coordinates
  - Keys: `'spine1'`, `'spine2'`, `'spine3'`
  - Values: (N, 2) arrays of (x, y) coordinates
- `expected_body_length_cm` (float): Expected body length in cm. Default: 8.0
- `tolerance` (float): Acceptable deviation fraction. Default: 0.20 (±20%)

**Returns**:
- `float`: Scaling factor in cm/pixel

**Raises**:
- `ValueError`: If computed length exceeds tolerance bounds

**Example**:
```python
preprocessor = Preprocessor()

spine_keypoints = {
    'spine1': top_df[['spine1_x', 'spine1_y']].values,
    'spine2': top_df[['spine2_x', 'spine2_y']].values,
    'spine3': top_df[['spine3_x', 'spine3_y']].values
}

scaling_factor = preprocessor.compute_scaling_factor(
    spine_keypoints,
    expected_body_length_cm=8.0
)

print(f"Scaling: {scaling_factor:.4f} cm/pixel")
```

#### `preprocess_trajectory()`

Clean and smooth a single keypoint trajectory.

**Signature**:
```python
def preprocess_trajectory(
    self,
    trajectory: np.ndarray,
    likelihood: np.ndarray = None
) -> Tuple[np.ndarray, Dict]
```

**Parameters**:
- `trajectory` (np.ndarray): (N, 2) array of (x, y) positions
- `likelihood` (np.ndarray, optional): (N,) array of confidence scores

**Returns**:
- `Tuple[np.ndarray, Dict]`:
  - Cleaned trajectory (N, 2)
  - Quality metrics dict:
    - `'completeness'`: Fraction of valid data (0-1)
    - `'outliers_removed'`: Number of outliers removed
    - `'gaps_interpolated'`: Number of gaps filled
    - `'max_gap_size'`: Largest gap encountered

**Example**:
```python
raw_trajectory = df[['paw_RR_x', 'paw_RR_y']].values
likelihood = df['paw_RR_likelihood'].values

clean_trajectory, quality = preprocessor.preprocess_trajectory(
    trajectory=raw_trajectory,
    likelihood=likelihood
)

print(f"Data completeness: {quality['completeness']*100:.1f}%")
print(f"Outliers removed: {quality['outliers_removed']}")
```

---

## Analysis Modules

### `src/exmo_gait/analysis/phase_detector.py`

#### `PhaseDetector` Class

Detect locomotor phases (walking, stationary, rearing).

**Constructor**:
```python
def __init__(
    self,
    fps: float = 120.0,
    stationary_mad_threshold: float = 1.5,
    walking_mad_threshold: float = 2.0,
    min_walking_duration: float = 0.3,
    min_stationary_duration: float = 0.25,
    adaptive_thresholding: bool = False,
    adaptive_percentile: int = 75
)
```

**Parameters**:
- `fps` (float): Frame rate. Default: 120.0
- `stationary_mad_threshold` (float): MAD multiplier for stationary detection. Default: 1.5
- `walking_mad_threshold` (float): MAD multiplier for walking detection. Default: 2.0
- `min_walking_duration` (float): Minimum walking bout duration (s). Default: 0.3
- `min_stationary_duration` (float): Minimum stationary bout duration (s). Default: 0.25
- `adaptive_thresholding` (bool): Enable adaptive mode. Default: False
- `adaptive_percentile` (int): Percentile for adaptive thresholds. Default: 75

#### `detect_phases()`

Identify locomotor phases from CoM trajectory.

**Signature**:
```python
def detect_phases(
    self,
    com_trajectory: np.ndarray
) -> Dict[str, List[Tuple[int, int]]]
```

**Parameters**:
- `com_trajectory` (np.ndarray): (N, 2) array of CoM (x, y) positions

**Returns**:
- `Dict[str, List[Tuple[int, int]]]`: Phase windows
  - `'walking'`: List of (start_frame, end_frame) tuples
  - `'stationary'`: List of (start_frame, end_frame) tuples
  - `'rearing'`: List of (start_frame, end_frame) tuples

**Example**:
```python
from exmo_gait.analysis.phase_detector import PhaseDetector

detector = PhaseDetector(
    fps=120.0,
    walking_mad_threshold=1.2,  # More sensitive
    adaptive_thresholding=True   # Auto-calibrate
)

phases = detector.detect_phases(com_trajectory)

print(f"Walking windows: {len(phases['walking'])}")
for start, end in phases['walking']:
    duration = (end - start) / 120.0  # Convert to seconds
    print(f"  {start}-{end} ({duration:.2f}s)")
```

---

### `src/exmo_gait/analysis/step_detector.py`

#### `StepDetector` Class

Detect foot strikes and compute stride parameters.

**Constructor**:
```python
def __init__(
    self,
    fps: float = 120.0,
    min_stride_duration: float = 0.1,
    max_stride_duration: float = 1.0,
    prominence_multiplier: float = 0.5
)
```

**Parameters**:
- `fps` (float): Frame rate. Default: 120.0
- `min_stride_duration` (float): Minimum stride time (s). Default: 0.1
- `max_stride_duration` (float): Maximum stride time (s). Default: 1.0
- `prominence_multiplier` (float): Peak prominence threshold. Default: 0.5

#### `detect_steps()`

Detect foot strikes for a single limb.

**Signature**:
```python
def detect_steps(
    self,
    paw_trajectory: np.ndarray,
    walking_windows: List[Tuple[int, int]] = None
) -> List[Dict]
```

**Parameters**:
- `paw_trajectory` (np.ndarray): (N, 3) array of (x, y, z) positions
- `walking_windows` (List[Tuple[int, int]], optional): Restrict detection to walking phases

**Returns**:
- `List[Dict]`: List of stride dictionaries
  - `'start_frame'`: Foot strike frame index
  - `'end_frame'`: Next foot strike frame index
  - `'duration_sec'`: Stride duration in seconds
  - `'length_cm'`: Stride length in cm
  - `'swing_phase'`: Swing phase fraction (0-1)
  - `'stance_phase'`: Stance phase fraction (0-1)

**Example**:
```python
from exmo_gait.analysis.step_detector import StepDetector

detector = StepDetector(
    fps=120.0,
    prominence_multiplier=0.3  # More sensitive
)

paw_trajectory = bottom_df[['paw_RR_x', 'paw_RR_y', 'paw_RR_z']].values
strides = detector.detect_steps(
    paw_trajectory,
    walking_windows=phases['walking']
)

print(f"Detected {len(strides)} strides")
for stride in strides:
    print(f"  Length: {stride['length_cm']:.2f} cm, "
          f"Duration: {stride['duration_sec']:.3f} s")
```

---

### `src/exmo_gait/analysis/metrics_computer.py`

#### `MetricsComputer` Class

Compute comprehensive gait and biomechanical metrics.

**Constructor**:
```python
def __init__(
    self,
    fps: float = 120.0,
    scaling_factor: float = 1.0
)
```

**Parameters**:
- `fps` (float): Frame rate. Default: 120.0
- `scaling_factor` (float): cm/pixel conversion ratio. Default: 1.0

#### `compute_gait_metrics()`

Calculate temporal and spatial gait parameters.

**Signature**:
```python
def compute_gait_metrics(
    self,
    strides_by_limb: Dict[str, List[Dict]],
    com_trajectory: np.ndarray,
    walking_windows: List[Tuple[int, int]]
) -> pd.DataFrame
```

**Parameters**:
- `strides_by_limb` (Dict[str, List[Dict]]): Stride data for each limb
  - Keys: `'paw_RR'`, `'paw_RL'`, `'paw_FR'`, `'paw_FL'`
  - Values: List of stride dictionaries from `detect_steps()`
- `com_trajectory` (np.ndarray): (N, 2) CoM positions
- `walking_windows` (List[Tuple[int, int]]): Walking phase windows

**Returns**:
- `pd.DataFrame`: Gait metrics with columns:
  - `'limb'`: Limb identifier
  - `'stride_length_cm'`: Stride length
  - `'stride_time_sec'`: Stride duration
  - `'cadence_steps_per_min'`: Step frequency
  - `'swing_time_sec'`: Swing phase duration
  - `'stance_time_sec'`: Stance phase duration
  - `'duty_cycle_percent'`: Stance/(stance+swing) × 100
  - `'speed_cm_per_sec'`: Limb speed
  - `'regularity_index'`: Coordination measure (0-1)

**Example**:
```python
from exmo_gait.analysis.metrics_computer import MetricsComputer

computer = MetricsComputer(
    fps=120.0,
    scaling_factor=0.061
)

# Assume strides_by_limb already populated
gait_metrics = computer.compute_gait_metrics(
    strides_by_limb,
    com_trajectory,
    walking_windows
)

# Analyze results
print(gait_metrics.groupby('limb')['cadence_steps_per_min'].median())
```

#### `compute_rom_metrics()`

Calculate range of motion and joint angle metrics.

**Signature**:
```python
def compute_rom_metrics(
    self,
    joint_trajectories: Dict[str, np.ndarray]
) -> pd.DataFrame
```

**Parameters**:
- `joint_trajectories` (Dict[str, np.ndarray]): Joint position data
  - Keys: `'hip_R'`, `'hip_L'`, `'elbow_R'`, `'elbow_L'`, `'paw_RR'`, etc.
  - Values: (N, 2) or (N, 3) position arrays

**Returns**:
- `pd.DataFrame`: ROM metrics with columns:
  - `'joint'`: Joint identifier
  - `'angle_mean_deg'`: Mean joint angle
  - `'angle_std_deg'`: Angle variability
  - `'angle_min_deg'`: Minimum angle
  - `'angle_max_deg'`: Maximum angle
  - `'rom_deg'`: Range of motion (max - min)
  - `'angular_velocity_deg_per_sec'`: Mean angular velocity

**Example**:
```python
joint_trajectories = {
    'hip_R': side_df[['hip_R_x', 'hip_R_y']].values,
    'elbow_R': side_df[['elbow_R_x', 'elbow_R_y']].values,
    'paw_RR': side_df[['paw_RR_x', 'paw_RR_y']].values,
}

rom_metrics = computer.compute_rom_metrics(joint_trajectories)

print(rom_metrics[['joint', 'rom_deg', 'angular_velocity_deg_per_sec']])
```

---

## Export Modules

### `src/exmo_gait/export/xlsx_exporter.py`

#### `export_to_excel()`

Generate comprehensive Excel report.

**Signature**:
```python
def export_to_excel(
    output_path: Path,
    gait_metrics: pd.DataFrame,
    rom_metrics: pd.DataFrame,
    aggregated_gait: pd.DataFrame,
    aggregated_rom: pd.DataFrame,
    phase_windows: Dict,
    foot_strikes: Dict,
    com_trajectory: np.ndarray,
    preprocessing_stats: Dict,
    config: Dict,
    metadata: Dict
) -> Path
```

**Parameters**:
- `output_path` (Path): Output file path (.xlsx)
- `gait_metrics` (pd.DataFrame): Per-stride gait metrics
- `rom_metrics` (pd.DataFrame): ROM measurements
- `aggregated_gait` (pd.DataFrame): Statistical summaries of gait
- `aggregated_rom` (pd.DataFrame): Statistical summaries of ROM
- `phase_windows` (Dict): Locomotor phase timing
- `foot_strikes` (Dict): Foot strike events by limb
- `com_trajectory` (np.ndarray): Center of mass trajectory
- `preprocessing_stats` (Dict): Data quality metrics
- `config` (Dict): Processing configuration
- `metadata` (Dict): Sample metadata

**Returns**:
- `Path`: Path to created Excel file

**Sheets Created**:
1. Summary
2. Gait_Metrics_Walking
3. Gait_Metrics_All
4. ROM_Metrics_Walking
5. ROM_Metrics_All
6. Aggregated_Gait
7. Aggregated_ROM
8. Phase_Windows
9. Foot_Strikes
10. CoM_Trajectory
11. Preprocessing_Stats
12. Configuration
13. Metadata
14. Errors (if applicable)

**Example**:
```python
from exmo_gait.export.xlsx_exporter import export_to_excel

excel_path = export_to_excel(
    output_path=Path("Output/sample_1/analysis.xlsx"),
    gait_metrics=gait_metrics,
    rom_metrics=rom_metrics,
    aggregated_gait=agg_gait,
    aggregated_rom=agg_rom,
    phase_windows=phases,
    foot_strikes=strides_by_limb,
    com_trajectory=com_traj,
    preprocessing_stats=quality_stats,
    config=config,
    metadata={'sample_id': 'sample_1', 'group': 'control'}
)

print(f"Report saved to: {excel_path}")
```

---

### `src/exmo_gait/export/visualizer_enhanced.py`

#### `EnhancedDashboardVisualizer` Class

Generate publication-grade scientific plots.

**Constructor**:
```python
def __init__(
    self,
    output_dir: Path,
    dpi: int = 600,
    marker_size: int = 60,
    font_scale: float = 1.0,
    annotate_median: bool = True,
    add_reference_bands: bool = True
)
```

**Parameters**:
- `output_dir` (Path): Output directory for plots
- `dpi` (int): Plot resolution. Default: 600
- `marker_size` (int): Scatter marker size. Default: 60
- `font_scale` (float): Global font scaling. Default: 1.0
- `annotate_median` (bool): Add median value labels. Default: True
- `add_reference_bands` (bool): Show normal range bands. Default: True

#### `generate_all_dashboards()`

Create all four dashboard plots.

**Signature**:
```python
def generate_all_dashboards(
    self,
    gait_metrics: pd.DataFrame,
    rom_metrics: pd.DataFrame,
    aggregated_gait: pd.DataFrame,
    aggregated_rom: pd.DataFrame
) -> Dict[str, Path]
```

**Parameters**:
- `gait_metrics` (pd.DataFrame): Per-stride gait metrics
- `rom_metrics` (pd.DataFrame): ROM measurements
- `aggregated_gait` (pd.DataFrame): Aggregated gait statistics
- `aggregated_rom` (pd.DataFrame): Aggregated ROM statistics

**Returns**:
- `Dict[str, Path]`: Dictionary mapping plot names to file paths
  - `'coordination'`: Coordination dashboard path
  - `'speed_spatial'`: Speed & spatial dashboard path
  - `'phase_timing'`: Phase timing dashboard path
  - `'range_of_motion'`: ROM dashboard path

**Example**:
```python
from exmo_gait.export.visualizer_enhanced import EnhancedDashboardVisualizer

visualizer = EnhancedDashboardVisualizer(
    output_dir=Path("Output/sample_1"),
    dpi=600,                    # Publication quality
    marker_size=60,
    annotate_median=True,
    add_reference_bands=True
)

plots = visualizer.generate_all_dashboards(
    gait_metrics,
    rom_metrics,
    aggregated_gait,
    aggregated_rom
)

for name, path in plots.items():
    print(f"{name}: {path}")
```

---

## Utility Functions

### `src/exmo_gait/utils/validation.py`

#### `validate_scaling_factor()`

Check if computed scaling factor is within acceptable range.

**Signature**:
```python
def validate_scaling_factor(
    scaling_factor: float,
    expected_body_length_cm: float = 8.0,
    tolerance: float = 0.20
) -> bool
```

**Parameters**:
- `scaling_factor` (float): Computed cm/pixel ratio
- `expected_body_length_cm` (float): Expected body length. Default: 8.0
- `tolerance` (float): Acceptable deviation. Default: 0.20

**Returns**:
- `bool`: True if valid, False otherwise

**Example**:
```python
from exmo_gait.utils.validation import validate_scaling_factor

is_valid = validate_scaling_factor(
    scaling_factor=0.061,
    expected_body_length_cm=8.0
)

if not is_valid:
    print("Warning: Scaling factor outside expected range")
```

---

## Configuration API

### Loading Configuration

```python
import yaml
from pathlib import Path

def load_config(config_path: Path) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Usage
config = load_config(Path("config_adaptive.yaml"))
```

### Accessing Configuration Values

```python
# Global settings
fps = config['global_settings']['fps']
walking_threshold = config['global_settings']['walking_mad_threshold']
plot_dpi = config['global_settings']['plot_dpi']

# Experiment groups
control_samples = config['experiment_groups']['control']['samples']

# File patterns
top_pattern = config['file_patterns']['top']
```

### Updating Configuration Programmatically

```python
# Modify settings
config['global_settings']['plot_dpi'] = 300
config['global_settings']['use_enhanced_plots'] = False

# Save back to file
with open('config_custom.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
```

---

## Examples

### Example 1: Complete Pipeline Execution

```python
from pathlib import Path
from exmo_gait.cli import run_pipeline

# Process single sample
result = run_pipeline(
    top_csv=Path("Data/TOP/control_5.csv"),
    side_csv=Path("Data/SIDE/control_5.csv"),
    bottom_csv=Path("Data/BOTTOM/control_5.csv"),
    output_dir=Path("Output/control/control_5"),
    config=config
)

print(f"Status: {result['status']}")
print(f"Walking windows: {result['walking_windows']}")
```

### Example 2: Custom Phase Detection

```python
from exmo_gait.core.data_loader import load_multiview_data
from exmo_gait.core.preprocessor import Preprocessor
from exmo_gait.analysis.phase_detector import PhaseDetector

# Load data
top, side, bottom, metadata = load_multiview_data(
    Path("Data/TOP/sample.csv"),
    Path("Data/SIDE/sample.csv"),
    Path("Data/BOTTOM/sample.csv")
)

# Extract CoM
preprocessor = Preprocessor()
com = preprocessor.compute_com(top)

# Detect phases with custom parameters
detector = PhaseDetector(
    fps=120.0,
    walking_mad_threshold=1.0,    # Very sensitive
    min_walking_duration=0.1,     # Short bouts
    adaptive_thresholding=True
)

phases = detector.detect_phases(com)

# Analyze results
walking_frames = sum(end - start for start, end in phases['walking'])
walking_percent = (walking_frames / metadata['n_frames']) * 100

print(f"Walking: {walking_percent:.1f}% of recording")
```

### Example 3: Batch Processing with Custom Logic

```python
from pathlib import Path
import pandas as pd

results = []

for sample_id in ['control_1', 'control_2', 'control_3']:
    result = run_pipeline(
        top_csv=Path(f"Data/TOP/{sample_id}.csv"),
        side_csv=Path(f"Data/SIDE/{sample_id}.csv"),
        bottom_csv=Path(f"Data/BOTTOM/{sample_id}.csv"),
        output_dir=Path(f"Output/control/{sample_id}"),
        config=config
    )

    results.append({
        'sample_id': sample_id,
        'status': result['status'],
        'walking_windows': result['walking_windows'],
        'avg_cadence': result.get('avg_cadence', None)
    })

# Create summary
summary_df = pd.DataFrame(results)
summary_df.to_csv('batch_summary.csv', index=False)
```

### Example 4: Extract Specific Metrics

```python
from exmo_gait.analysis.metrics_computer import MetricsComputer

# Compute metrics
computer = MetricsComputer(fps=120.0, scaling_factor=0.061)
gait_metrics = computer.compute_gait_metrics(
    strides_by_limb,
    com_trajectory,
    walking_windows
)

# Extract specific metric for analysis
rh_cadence = gait_metrics[
    gait_metrics['limb'] == 'paw_RR'
]['cadence_steps_per_min']

print(f"RH Cadence: {rh_cadence.median():.1f} steps/min")
print(f"RH Cadence MAD: {rh_cadence.mad():.1f}")
```

### Example 5: Custom Visualization

```python
from exmo_gait.export.visualizer_enhanced import EnhancedDashboardVisualizer
from exmo_gait.export.style import EXMOPlotStyle
import matplotlib.pyplot as plt

# Create custom plot using EXMO style
style = EXMOPlotStyle(dpi=600, marker_size=60, font_scale=1.0)

fig, ax = plt.subplots(figsize=(10, 6))
style.apply_to_axis(ax)

# Plot custom data
ax.scatter(x_data, y_data, s=60, c=style.colors['paw_RR'], alpha=0.75)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Speed (cm/s)')
ax.set_title('Custom Analysis')

style.add_sample_badge(ax, n_samples=len(x_data))

fig.savefig('custom_plot.png', dpi=600, bbox_inches='tight')
```

---

## Error Handling

### Common Exceptions

**`FileNotFoundError`**:
```python
try:
    result = run_pipeline(top_csv, side_csv, bottom_csv, output_dir)
except FileNotFoundError as e:
    print(f"Missing file: {e}")
    # Handle missing data files
```

**`ValueError`** (Invalid configuration):
```python
try:
    detector = PhaseDetector(walking_mad_threshold=-1.0)
except ValueError as e:
    print(f"Invalid parameter: {e}")
    # Use default values
```

**`RuntimeError`** (Processing failure):
```python
try:
    metrics = computer.compute_gait_metrics(...)
except RuntimeError as e:
    print(f"Processing failed: {e}")
    # Skip sample or use alternative method
```

---

## Performance Tips

### Memory Optimization

```python
# Process large datasets in chunks
def process_large_dataset(csv_path, chunk_size=10000):
    for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
        # Process chunk
        yield process_chunk(chunk)
```

### Parallel Processing

```python
from concurrent.futures import ProcessPoolExecutor

def process_sample(sample_id):
    return run_pipeline(...)

with ProcessPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(process_sample, sample_ids))
```

### Caching Intermediate Results

```python
import numpy as np

# Save intermediate data
np.savez(
    'intermediates/sample_1.npz',
    com_trajectory=com,
    phases=phases,
    strides=strides
)

# Load later
data = np.load('intermediates/sample_1.npz', allow_pickle=True)
com = data['com_trajectory']
```

---

## Version Compatibility

### Python Version Support
- **Minimum**: Python 3.8
- **Recommended**: Python 3.10+
- **Tested**: Python 3.8, 3.9, 3.10, 3.11

### Dependency Versions
- NumPy: >= 1.20.0
- Pandas: >= 1.3.0
- Matplotlib: >= 3.4.0
- SciPy: >= 1.7.0

### Breaking Changes

**v1.0.0 → v1.1.0**:
- Added `adaptive_thresholding` parameter to PhaseDetector
- Added `use_enhanced_plots` parameter to config
- File pattern now uses `_*.csv` instead of `_main*.csv`

---

**API Version**: 1.1.0
**Last Updated**: 2025-11-21
**Maintained By**: EXMO Development Team

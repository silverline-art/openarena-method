"""
Centralized constants for EXMO gait analysis pipeline.

All magic numbers extracted into semantically meaningful constants to improve
code maintainability and readability.

Version: 1.0.0
Date: 2025-11-21
"""

# ============================================================================
# Biomechanical Constants
# ============================================================================

# Body measurements (cm)
DEFAULT_MOUSE_BODY_LENGTH_CM = 10.0  # Adult mouse snout→tail length
LEGACY_SPINE_LENGTH_CM = 8.0  # Legacy spine1→spine3 distance (v1.1.0)

# Center of mass weights (relative contributions)
COM_WEIGHT_SNOUT = 0.10  # Head/snout contribution
COM_WEIGHT_NECK = 0.15  # Neck region contribution
COM_WEIGHT_SHOULDER = 0.20  # Shoulder girdle contribution
COM_WEIGHT_RIB_CENTER = 0.25  # Thoracic region (largest mass)
COM_WEIGHT_HIP = 0.20  # Pelvic region contribution
COM_WEIGHT_TAIL_BASE = 0.10  # Tail base contribution

# Likelihood thresholds
MIN_LIKELIHOOD_KEYPOINT = 0.9  # Minimum confidence for keypoint validity
MIN_LIKELIHOOD_SCALING = 0.9  # Minimum confidence for scaling calculation

# ============================================================================
# Signal Processing Constants
# ============================================================================

# Smoothing parameters
SAVGOL_WINDOW_SIZE_DEFAULT = 11  # Savitzky-Golay filter window (frames)
SAVGOL_POLY_ORDER_DEFAULT = 3  # Polynomial order for smoothing
SAVGOL_WINDOW_SIZE_ADAPTIVE_HIGH = 7  # High quality data (>90% complete)
SAVGOL_WINDOW_SIZE_ADAPTIVE_MED = 9  # Medium quality data (70-90% complete)
SAVGOL_WINDOW_SIZE_ADAPTIVE_LOW = 13  # Low quality data (<70% complete)

# EMA (Exponential Moving Average) parameters
EMA_ALPHA_DEFAULT = 0.35  # v1.2.0 EMA smoothing coefficient for velocity

# Outlier detection
SCALING_TOLERANCE_DEFAULT = 0.25  # ±25% variance allowed from median
MAD_MULTIPLIER_DEFAULT = 1.4826  # Median Absolute Deviation→StdDev conversion

# Minimum data requirements
MIN_FRAMES_FOR_SCALING = 100  # Minimum frames for reliable scaling
MIN_FRAMES_FOR_ANALYSIS = 50  # Minimum frames after outlier removal

# ============================================================================
# Gait Detection Constants
# ============================================================================

# Phase detection thresholds (MAD multipliers)
MAD_THRESHOLD_STATIONARY_DEFAULT = 1.5  # Below this = stationary
MAD_THRESHOLD_WALKING_DEFAULT = 2.0  # Above this = walking (v1.1.0)
MAD_THRESHOLD_WALKING_V12 = 1.5  # v1.2.0 hybrid threshold

# Adaptive threshold parameters (v1.2.0)
ADAPTIVE_PERCENTILE_DEFAULT = 75  # 75th percentile for threshold adaptation
MIN_THRESHOLD_PX_PER_FRAME = 1.0  # Absolute minimum threshold (pixels/frame)

# Duration constraints (seconds)
MIN_WALKING_DURATION_SEC = 0.3  # Minimum valid walking bout duration
MIN_STATIONARY_DURATION_SEC = 0.25  # Minimum valid stationary bout duration

# Stride detection
MIN_STRIDE_DURATION_SEC = 0.1  # v1.1.0 minimum (too strict - filters 40% of strides)
MIN_STRIDE_DURATION_SEC_V12 = 0.05  # v1.2.0 minimum (captures fast strides)
MAX_STRIDE_DURATION_SEC = 2.0  # Maximum reasonable stride duration

# ============================================================================
# Temporal Constants
# ============================================================================

# Frame rate
FPS_DEFAULT = 120.0  # Default acquisition frame rate (Hz)

# Temporal windows
STRIDE_ANALYSIS_WINDOW_SEC = 5.0  # Time window for stride aggregation

# ============================================================================
# Statistical Constants
# ============================================================================

# Confidence intervals
CI_PERCENTILE_DEFAULT = 95  # 95% confidence interval
TRIM_PERCENT_DEFAULT = 5  # Trim 5% outliers from each tail

# Precision thresholds
EPSILON = 1e-10  # Small value to prevent division by zero
ANGLE_PRECISION_DEGREES = 0.01  # Minimum meaningful angle change

# ============================================================================
# Numerical Stability Constants
# ============================================================================

# Vector norms
MIN_VECTOR_NORM = 1e-10  # Minimum norm to consider vector non-zero

# Angle calculations
COSINE_CLIP_MIN = -1.0  # Minimum valid cosine value
COSINE_CLIP_MAX = 1.0  # Maximum valid cosine value

# ============================================================================
# Export Constants
# ============================================================================

# Excel formatting
EXCEL_FLOAT_PRECISION = 4  # Decimal places for float columns
EXCEL_ANGLE_PRECISION = 2  # Decimal places for angle measurements

# Plot parameters
PLOT_DPI = 300  # Resolution for saved figures
PLOT_FIGSIZE_WIDTH = 12  # Default figure width (inches)
PLOT_FIGSIZE_HEIGHT = 8  # Default figure height (inches)

# ============================================================================
# Data Quality Thresholds
# ============================================================================

# Completeness requirements
DATA_COMPLETENESS_HIGH = 0.90  # >90% valid frames = high quality
DATA_COMPLETENESS_MEDIUM = 0.70  # 70-90% valid frames = medium quality
# <70% valid frames = low quality (requires aggressive smoothing)

# ============================================================================
# Version-Specific Flags
# ============================================================================

# Method version markers (for documentation/debugging)
VERSION_MARKER_V11 = "v1.1.0"  # Legacy methods
VERSION_MARKER_V12 = "v1.2.0"  # Calibrated methods

# ============================================================================
# Validation Constants
# ============================================================================

# Anatomical limits (for sanity checking)
MAX_STRIDE_LENGTH_CM = 20.0  # Maximum plausible stride length
MIN_STRIDE_LENGTH_CM = 0.5  # Minimum plausible stride length
MAX_SPEED_CM_PER_SEC = 100.0  # Maximum plausible locomotion speed
MAX_ROM_DEGREES = 180.0  # Maximum anatomical range of motion
MIN_ROM_DEGREES = 0.0  # Minimum anatomical range of motion

# Ratio limits
MAX_ASYMMETRY_INDEX = 200.0  # Maximum plausible left/right asymmetry (%)
MIN_DUTY_FACTOR = 0.0  # Minimum stance phase proportion
MAX_DUTY_FACTOR = 1.0  # Maximum stance phase proportion

"""
Pydantic schema validation for EXMO gait analysis configuration files.

Provides strong typing, validation, and documentation for all config parameters.
Prevents runtime errors from invalid config values.

Version: 1.0.0
Date: 2025-11-21
"""

from typing import Optional, Literal, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator
from pathlib import Path

from .constants import (
    FPS_DEFAULT,
    DEFAULT_MOUSE_BODY_LENGTH_CM,
    LEGACY_SPINE_LENGTH_CM,
    MIN_LIKELIHOOD_SCALING,
    SAVGOL_WINDOW_SIZE_DEFAULT,
    SAVGOL_POLY_ORDER_DEFAULT,
    EMA_ALPHA_DEFAULT,
    MAD_THRESHOLD_STATIONARY_DEFAULT,
    MAD_THRESHOLD_WALKING_DEFAULT,
    ADAPTIVE_PERCENTILE_DEFAULT,
    MIN_THRESHOLD_PX_PER_FRAME,
    MIN_WALKING_DURATION_SEC,
    MIN_STATIONARY_DURATION_SEC,
    MIN_STRIDE_DURATION_SEC,
    MIN_STRIDE_DURATION_SEC_V12,
    CI_PERCENTILE_DEFAULT,
    TRIM_PERCENT_DEFAULT,
    PLOT_DPI,
)


class GeneralSettings(BaseModel):
    """General pipeline settings"""

    fps: float = Field(
        default=FPS_DEFAULT,
        gt=0,
        le=1000,
        description="Video frame rate (Hz). Typical range: 30-240 Hz"
    )

    output_dir: str = Field(
        default="results",
        description="Directory for output files (Excel, plots, logs)"
    )

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO",
        description="Logging verbosity level"
    )

    @field_validator('output_dir')
    @classmethod
    def validate_output_dir(cls, v: str) -> str:
        """Ensure output directory is valid"""
        path = Path(v)
        if path.exists() and not path.is_dir():
            raise ValueError(f"Output path exists but is not a directory: {v}")
        return v


class ScalingSettings(BaseModel):
    """Spatial scaling configuration"""

    scaling_method: Literal["spine_only", "full_body"] = Field(
        default="spine_only",
        description="v1.1.0: spine_only (spine1→spine3), v1.2.0: full_body (snout→tail)"
    )

    expected_body_length_cm: float = Field(
        default=DEFAULT_MOUSE_BODY_LENGTH_CM,
        gt=0,
        le=50,
        description="Expected body length (cm). Adults: 10cm, juveniles: 6-8cm"
    )

    scaling_min_likelihood: float = Field(
        default=MIN_LIKELIHOOD_SCALING,
        ge=0.0,
        le=1.0,
        description="Minimum keypoint confidence for scaling calculation"
    )

    scaling_tolerance: float = Field(
        default=0.25,
        gt=0,
        le=1.0,
        description="Allowed variance from median body length (±fraction)"
    )

    @model_validator(mode='after')
    def validate_scaling_method(self):
        """Validate scaling method compatibility"""
        if self.scaling_method == "spine_only" and self.expected_body_length_cm != LEGACY_SPINE_LENGTH_CM:
            # Warning: user might want to use legacy 8cm for spine_only
            pass
        return self


class SmoothingSettings(BaseModel):
    """Signal smoothing configuration"""

    smoothing_adaptive: bool = Field(
        default=False,
        description="v1.2.0: Adapt smoothing window based on data quality"
    )

    smoothing_window: int = Field(
        default=SAVGOL_WINDOW_SIZE_DEFAULT,
        ge=3,
        le=31,
        description="Savitzky-Golay filter window size (must be odd)"
    )

    smoothing_poly: int = Field(
        default=SAVGOL_POLY_ORDER_DEFAULT,
        ge=1,
        le=5,
        description="Savitzky-Golay polynomial order (must be < window)"
    )

    velocity_smoothing_method: Literal["savgol", "ema"] = Field(
        default="savgol",
        description="Velocity calculation method. v1.2.0: ema recommended"
    )

    velocity_ema_alpha: float = Field(
        default=EMA_ALPHA_DEFAULT,
        gt=0,
        lt=1,
        description="EMA smoothing coefficient (higher = less smoothing)"
    )

    @field_validator('smoothing_window')
    @classmethod
    def validate_window_odd(cls, v: int) -> int:
        """Ensure window size is odd"""
        if v % 2 == 0:
            raise ValueError(f"smoothing_window must be odd, got {v}")
        return v

    @model_validator(mode='after')
    def validate_poly_window_relationship(self):
        """Ensure polynomial order < window size"""
        if self.smoothing_poly >= self.smoothing_window:
            raise ValueError(
                f"smoothing_poly ({self.smoothing_poly}) must be < "
                f"smoothing_window ({self.smoothing_window})"
            )
        return self


class PhaseDetectionSettings(BaseModel):
    """Walking/stationary phase detection"""

    stationary_mad_threshold: float = Field(
        default=MAD_THRESHOLD_STATIONARY_DEFAULT,
        gt=0,
        le=10,
        description="MAD multiplier for stationary threshold"
    )

    walking_mad_threshold: float = Field(
        default=MAD_THRESHOLD_WALKING_DEFAULT,
        gt=0,
        le=10,
        description="MAD multiplier for walking threshold (v1.1.0)"
    )

    use_hybrid_threshold: bool = Field(
        default=False,
        description="v1.2.0: Use adaptive threshold combining MAD and percentile"
    )

    adaptive_percentile: int = Field(
        default=ADAPTIVE_PERCENTILE_DEFAULT,
        ge=50,
        le=99,
        description="Percentile for adaptive threshold (v1.2.0)"
    )

    min_threshold_px_per_frame: float = Field(
        default=MIN_THRESHOLD_PX_PER_FRAME,
        gt=0,
        description="Absolute minimum threshold in pixels/frame"
    )

    min_walking_duration: float = Field(
        default=MIN_WALKING_DURATION_SEC,
        gt=0,
        le=10,
        description="Minimum valid walking bout duration (seconds)"
    )

    min_stationary_duration: float = Field(
        default=MIN_STATIONARY_DURATION_SEC,
        gt=0,
        le=10,
        description="Minimum valid stationary bout duration (seconds)"
    )

    @model_validator(mode='after')
    def validate_threshold_ordering(self):
        """Ensure stationary threshold < walking threshold"""
        if self.stationary_mad_threshold >= self.walking_mad_threshold:
            raise ValueError(
                f"stationary_mad_threshold ({self.stationary_mad_threshold}) must be < "
                f"walking_mad_threshold ({self.walking_mad_threshold})"
            )
        return self


class StrideDetectionSettings(BaseModel):
    """Stride and gait event detection"""

    min_stride_duration: float = Field(
        default=MIN_STRIDE_DURATION_SEC,
        gt=0,
        le=5,
        description="Minimum stride duration (sec). v1.1.0: 0.1, v1.2.0: 0.05"
    )

    max_stride_duration: float = Field(
        default=2.0,
        gt=0,
        le=10,
        description="Maximum stride duration (sec)"
    )

    foot_strike_prominence: float = Field(
        default=0.5,
        gt=0,
        description="Minimum peak prominence for foot strike detection"
    )

    @model_validator(mode='after')
    def validate_duration_ordering(self):
        """Ensure min < max stride duration"""
        if self.min_stride_duration >= self.max_stride_duration:
            raise ValueError(
                f"min_stride_duration ({self.min_stride_duration}) must be < "
                f"max_stride_duration ({self.max_stride_duration})"
            )
        return self


class COMSettings(BaseModel):
    """Center of mass calculation"""

    use_3d_com: bool = Field(
        default=False,
        description="v1.2.0: Use 3D COM from TOP+SIDE views instead of 2D"
    )

    com_weights: Optional[Dict[str, float]] = Field(
        default=None,
        description="Custom keypoint weights for COM. If None, uses default anatomical weights"
    )

    @field_validator('com_weights')
    @classmethod
    def validate_weights(cls, v: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
        """Ensure COM weights sum to 1.0"""
        if v is not None:
            total = sum(v.values())
            if not (0.99 <= total <= 1.01):  # Allow small floating point errors
                raise ValueError(f"COM weights must sum to 1.0, got {total}")
        return v


class AggregationSettings(BaseModel):
    """Statistical aggregation settings"""

    aggregation_include_ci: bool = Field(
        default=False,
        description="v1.2.0: Include confidence intervals in output"
    )

    aggregation_ci_percentile: int = Field(
        default=CI_PERCENTILE_DEFAULT,
        ge=50,
        le=99,
        description="Confidence interval percentile (e.g., 95 for 95% CI)"
    )

    aggregation_trim_percent: float = Field(
        default=TRIM_PERCENT_DEFAULT,
        ge=0,
        le=25,
        description="Percentage to trim from each tail for robust statistics"
    )


class VisualizationSettings(BaseModel):
    """Plot generation settings"""

    generate_plots: bool = Field(
        default=True,
        description="Generate PNG dashboards"
    )

    plot_dpi: int = Field(
        default=PLOT_DPI,
        ge=72,
        le=600,
        description="Plot resolution (DPI)"
    )

    plot_format: Literal["png", "pdf", "svg"] = Field(
        default="png",
        description="Output format for plots"
    )


class ExmoGaitConfig(BaseModel):
    """Complete EXMO gait analysis pipeline configuration"""

    general: GeneralSettings = Field(default_factory=GeneralSettings)
    scaling: ScalingSettings = Field(default_factory=ScalingSettings)
    smoothing: SmoothingSettings = Field(default_factory=SmoothingSettings)
    phase_detection: PhaseDetectionSettings = Field(default_factory=PhaseDetectionSettings)
    stride_detection: StrideDetectionSettings = Field(default_factory=StrideDetectionSettings)
    com: COMSettings = Field(default_factory=COMSettings)
    aggregation: AggregationSettings = Field(default_factory=AggregationSettings)
    visualization: VisualizationSettings = Field(default_factory=VisualizationSettings)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'ExmoGaitConfig':
        """Load and validate config from YAML file"""
        import yaml

        yaml_file = Path(yaml_path)
        if not yaml_file.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(yaml_file, 'r') as f:
            raw_config = yaml.safe_load(f)

        # Validate and parse
        return cls(**raw_config)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExmoGaitConfig':
        """Load and validate config from dictionary"""
        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Export config to dictionary"""
        return self.model_dump()

    def get_pipeline_version(self) -> str:
        """Determine which pipeline version is configured"""
        v12_features = [
            self.scaling.scaling_method == "full_body",
            self.smoothing.smoothing_adaptive,
            self.smoothing.velocity_smoothing_method == "ema",
            self.phase_detection.use_hybrid_threshold,
            self.com.use_3d_com,
            self.aggregation.aggregation_include_ci,
        ]

        if any(v12_features):
            return "v1.2.0"
        return "v1.1.0"

    def get_enabled_v12_features(self) -> list[str]:
        """List which v1.2.0 features are enabled"""
        features = []
        if self.scaling.scaling_method == "full_body":
            features.append("full_body_scaling")
        if self.smoothing.smoothing_adaptive:
            features.append("adaptive_smoothing")
        if self.smoothing.velocity_smoothing_method == "ema":
            features.append("ema_velocity")
        if self.phase_detection.use_hybrid_threshold:
            features.append("hybrid_threshold")
        if self.com.use_3d_com:
            features.append("3d_com")
        if self.aggregation.aggregation_include_ci:
            features.append("enhanced_statistics")

        return features


# Convenience function for backward compatibility
def load_config(config_path: str) -> ExmoGaitConfig:
    """
    Load and validate EXMO gait analysis config.

    Args:
        config_path: Path to YAML config file

    Returns:
        Validated ExmoGaitConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValidationError: If config contains invalid values
    """
    return ExmoGaitConfig.from_yaml(config_path)

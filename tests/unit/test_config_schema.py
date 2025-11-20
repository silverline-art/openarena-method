"""
Unit tests for Pydantic config schema validation.

Tests configuration validation, constraints, and version detection.
"""

import pytest
from pydantic import ValidationError
from src.exmo_gait.config_schema import (
    GeneralSettings,
    ScalingSettings,
    SmoothingSettings,
    PhaseDetectionSettings,
    StrideDetectionSettings,
    COMSettings,
    AggregationSettings,
    VisualizationSettings,
    ExmoGaitConfig,
    load_config
)


class TestGeneralSettings:
    """Test general settings validation"""

    def test_default_values(self):
        """Default general settings should be valid"""
        settings = GeneralSettings()
        assert settings.fps == 120.0
        assert settings.output_dir == "results"
        assert settings.log_level == "INFO"

    def test_fps_positive(self):
        """FPS must be positive"""
        with pytest.raises(ValidationError):
            GeneralSettings(fps=0)

        with pytest.raises(ValidationError):
            GeneralSettings(fps=-10)

    def test_fps_reasonable_upper_bound(self):
        """FPS should have reasonable upper bound"""
        with pytest.raises(ValidationError):
            GeneralSettings(fps=2000)  # Too high

    def test_log_level_valid_choices(self):
        """Log level must be from allowed choices"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR"]
        for level in valid_levels:
            settings = GeneralSettings(log_level=level)
            assert settings.log_level == level

        with pytest.raises(ValidationError):
            GeneralSettings(log_level="INVALID")


class TestScalingSettings:
    """Test scaling settings validation"""

    def test_default_values(self):
        """Default scaling settings should be valid"""
        settings = ScalingSettings()
        assert settings.scaling_method == "spine_only"
        assert settings.expected_body_length_cm == 10.0

    def test_scaling_method_choices(self):
        """Scaling method must be from allowed choices"""
        ScalingSettings(scaling_method="spine_only")
        ScalingSettings(scaling_method="full_body")

        with pytest.raises(ValidationError):
            ScalingSettings(scaling_method="invalid")

    def test_body_length_positive(self):
        """Body length must be positive"""
        with pytest.raises(ValidationError):
            ScalingSettings(expected_body_length_cm=0)

        with pytest.raises(ValidationError):
            ScalingSettings(expected_body_length_cm=-5)

    def test_body_length_reasonable_upper_bound(self):
        """Body length should have reasonable upper bound"""
        with pytest.raises(ValidationError):
            ScalingSettings(expected_body_length_cm=100)  # Too large for mouse

    def test_likelihood_range(self):
        """Likelihood threshold must be in [0, 1]"""
        ScalingSettings(scaling_min_likelihood=0.9)

        with pytest.raises(ValidationError):
            ScalingSettings(scaling_min_likelihood=1.5)

        with pytest.raises(ValidationError):
            ScalingSettings(scaling_min_likelihood=-0.1)

    def test_tolerance_positive(self):
        """Tolerance must be positive"""
        with pytest.raises(ValidationError):
            ScalingSettings(scaling_tolerance=0)

        with pytest.raises(ValidationError):
            ScalingSettings(scaling_tolerance=-0.1)


class TestSmoothingSettings:
    """Test smoothing settings validation"""

    def test_default_values(self):
        """Default smoothing settings should be valid"""
        settings = SmoothingSettings()
        assert settings.smoothing_window == 11
        assert settings.smoothing_poly == 3
        assert settings.smoothing_adaptive is False

    def test_window_must_be_odd(self):
        """Smoothing window must be odd"""
        SmoothingSettings(smoothing_window=11)  # Valid

        with pytest.raises(ValidationError, match="must be odd"):
            SmoothingSettings(smoothing_window=10)

    def test_window_size_bounds(self):
        """Window size must be within bounds"""
        with pytest.raises(ValidationError):
            SmoothingSettings(smoothing_window=1)  # Too small

        with pytest.raises(ValidationError):
            SmoothingSettings(smoothing_window=35)  # Too large

    def test_poly_order_bounds(self):
        """Polynomial order must be within bounds"""
        SmoothingSettings(smoothing_poly=3)  # Valid

        with pytest.raises(ValidationError):
            SmoothingSettings(smoothing_poly=0)

        with pytest.raises(ValidationError):
            SmoothingSettings(smoothing_poly=6)

    def test_poly_less_than_window(self):
        """Polynomial order must be less than window size"""
        SmoothingSettings(smoothing_window=11, smoothing_poly=3)  # Valid

        with pytest.raises(ValidationError, match="must be <"):
            SmoothingSettings(smoothing_window=5, smoothing_poly=5)

    def test_velocity_method_choices(self):
        """Velocity smoothing method must be from allowed choices"""
        SmoothingSettings(velocity_smoothing_method="savgol")
        SmoothingSettings(velocity_smoothing_method="ema")

        with pytest.raises(ValidationError):
            SmoothingSettings(velocity_smoothing_method="invalid")

    def test_ema_alpha_range(self):
        """EMA alpha must be in (0, 1)"""
        SmoothingSettings(velocity_ema_alpha=0.5)  # Valid

        with pytest.raises(ValidationError):
            SmoothingSettings(velocity_ema_alpha=0)

        with pytest.raises(ValidationError):
            SmoothingSettings(velocity_ema_alpha=1.0)


class TestPhaseDetectionSettings:
    """Test phase detection settings validation"""

    def test_default_values(self):
        """Default phase detection settings should be valid"""
        settings = PhaseDetectionSettings()
        assert settings.stationary_mad_threshold == 1.5
        assert settings.walking_mad_threshold == 2.0
        assert settings.use_hybrid_threshold is False

    def test_thresholds_positive(self):
        """MAD thresholds must be positive"""
        with pytest.raises(ValidationError):
            PhaseDetectionSettings(stationary_mad_threshold=0)

        with pytest.raises(ValidationError):
            PhaseDetectionSettings(walking_mad_threshold=-1)

    def test_threshold_ordering(self):
        """Stationary threshold must be less than walking threshold"""
        PhaseDetectionSettings(
            stationary_mad_threshold=1.0,
            walking_mad_threshold=2.0
        )  # Valid

        with pytest.raises(ValidationError, match="must be <"):
            PhaseDetectionSettings(
                stationary_mad_threshold=2.0,
                walking_mad_threshold=1.0
            )

    def test_adaptive_percentile_range(self):
        """Adaptive percentile must be in [50, 99]"""
        PhaseDetectionSettings(adaptive_percentile=75)  # Valid

        with pytest.raises(ValidationError):
            PhaseDetectionSettings(adaptive_percentile=30)

        with pytest.raises(ValidationError):
            PhaseDetectionSettings(adaptive_percentile=100)

    def test_min_threshold_positive(self):
        """Minimum threshold must be positive"""
        with pytest.raises(ValidationError):
            PhaseDetectionSettings(min_threshold_px_per_frame=0)


class TestStrideDetectionSettings:
    """Test stride detection settings validation"""

    def test_default_values(self):
        """Default stride detection settings should be valid"""
        settings = StrideDetectionSettings()
        assert settings.min_stride_duration == 0.1
        assert settings.max_stride_duration == 2.0

    def test_duration_positive(self):
        """Stride durations must be positive"""
        with pytest.raises(ValidationError):
            StrideDetectionSettings(min_stride_duration=0)

        with pytest.raises(ValidationError):
            StrideDetectionSettings(max_stride_duration=-1)

    def test_duration_ordering(self):
        """Min duration must be less than max duration"""
        StrideDetectionSettings(
            min_stride_duration=0.1,
            max_stride_duration=2.0
        )  # Valid

        with pytest.raises(ValidationError, match="must be <"):
            StrideDetectionSettings(
                min_stride_duration=3.0,
                max_stride_duration=2.0
            )

    def test_prominence_positive(self):
        """Prominence must be positive"""
        with pytest.raises(ValidationError):
            StrideDetectionSettings(foot_strike_prominence=0)


class TestCOMSettings:
    """Test center of mass settings validation"""

    def test_default_values(self):
        """Default COM settings should be valid"""
        settings = COMSettings()
        assert settings.use_3d_com is False
        assert settings.com_weights is None

    def test_custom_weights_sum_to_one(self):
        """Custom COM weights must sum to 1.0"""
        valid_weights = {
            "spine1": 0.3,
            "spine2": 0.3,
            "spine3": 0.4
        }
        COMSettings(com_weights=valid_weights)  # Valid

        invalid_weights = {
            "spine1": 0.5,
            "spine2": 0.3,
            "spine3": 0.3
        }
        with pytest.raises(ValidationError, match="must sum to 1.0"):
            COMSettings(com_weights=invalid_weights)


class TestAggregationSettings:
    """Test aggregation settings validation"""

    def test_default_values(self):
        """Default aggregation settings should be valid"""
        settings = AggregationSettings()
        assert settings.aggregation_include_ci is False
        assert settings.aggregation_ci_percentile == 95

    def test_ci_percentile_range(self):
        """CI percentile must be in [50, 99]"""
        AggregationSettings(aggregation_ci_percentile=95)  # Valid

        with pytest.raises(ValidationError):
            AggregationSettings(aggregation_ci_percentile=30)

    def test_trim_percent_range(self):
        """Trim percent must be in [0, 25]"""
        AggregationSettings(aggregation_trim_percent=5)  # Valid

        with pytest.raises(ValidationError):
            AggregationSettings(aggregation_trim_percent=30)


class TestVisualizationSettings:
    """Test visualization settings validation"""

    def test_default_values(self):
        """Default visualization settings should be valid"""
        settings = VisualizationSettings()
        assert settings.generate_plots is True
        assert settings.plot_dpi == 300

    def test_dpi_range(self):
        """DPI must be in reasonable range"""
        VisualizationSettings(plot_dpi=300)  # Valid

        with pytest.raises(ValidationError):
            VisualizationSettings(plot_dpi=50)  # Too low

        with pytest.raises(ValidationError):
            VisualizationSettings(plot_dpi=1000)  # Too high

    def test_plot_format_choices(self):
        """Plot format must be from allowed choices"""
        VisualizationSettings(plot_format="png")
        VisualizationSettings(plot_format="pdf")
        VisualizationSettings(plot_format="svg")

        with pytest.raises(ValidationError):
            VisualizationSettings(plot_format="jpg")


class TestExmoGaitConfig:
    """Test complete config object"""

    def test_default_config_valid(self):
        """Default config should be valid"""
        config = ExmoGaitConfig()
        assert config.general.fps == 120.0
        assert config.scaling.scaling_method == "spine_only"

    def test_from_dict(self, config_v11_dict):
        """Config can be loaded from dictionary"""
        config = ExmoGaitConfig.from_dict(config_v11_dict)
        assert config.general.fps == 120.0
        assert config.scaling.expected_body_length_cm == 8.0

    def test_from_yaml(self, temp_yaml_config):
        """Config can be loaded from YAML file"""
        config = ExmoGaitConfig.from_yaml(str(temp_yaml_config))
        assert config.general.fps == 120.0

    def test_from_yaml_nonexistent_file(self):
        """Loading nonexistent YAML file should raise error"""
        with pytest.raises(FileNotFoundError):
            ExmoGaitConfig.from_yaml("/nonexistent/config.yaml")

    def test_to_dict(self):
        """Config can be exported to dictionary"""
        config = ExmoGaitConfig()
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert "general" in config_dict
        assert "scaling" in config_dict

    def test_pipeline_version_detection_v11(self, config_v11_dict):
        """v1.1.0 config should be detected correctly"""
        config = ExmoGaitConfig.from_dict(config_v11_dict)
        assert config.get_pipeline_version() == "v1.1.0"

    def test_pipeline_version_detection_v12(self, config_v12_dict):
        """v1.2.0 config should be detected correctly"""
        config = ExmoGaitConfig.from_dict(config_v12_dict)
        assert config.get_pipeline_version() == "v1.2.0"

    def test_enabled_v12_features(self, config_v12_dict):
        """v1.2.0 features should be detected"""
        config = ExmoGaitConfig.from_dict(config_v12_dict)
        features = config.get_enabled_v12_features()

        assert "full_body_scaling" in features
        assert "adaptive_smoothing" in features
        assert "ema_velocity" in features
        assert "hybrid_threshold" in features
        assert "3d_com" in features
        assert "enhanced_statistics" in features

    def test_no_v12_features_in_v11(self, config_v11_dict):
        """v1.1.0 config should have no v1.2.0 features"""
        config = ExmoGaitConfig.from_dict(config_v11_dict)
        features = config.get_enabled_v12_features()
        assert len(features) == 0


class TestLoadConfigFunction:
    """Test convenience load_config function"""

    def test_load_config_function(self, temp_yaml_config):
        """load_config function should work correctly"""
        config = load_config(str(temp_yaml_config))
        assert isinstance(config, ExmoGaitConfig)
        assert config.general.fps == 120.0

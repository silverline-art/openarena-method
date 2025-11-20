"""
Unit tests for constants module.

Validates all magic numbers and configuration constants are within expected ranges.
"""

import pytest
from src.exmo_gait import constants


class TestBiomechanicalConstants:
    """Test biomechanical constant values"""

    def test_body_length_constants_positive(self):
        """Body length constants should be positive"""
        assert constants.DEFAULT_MOUSE_BODY_LENGTH_CM > 0
        assert constants.LEGACY_SPINE_LENGTH_CM > 0

    def test_body_length_realistic_range(self):
        """Body lengths should be in realistic mouse ranges"""
        assert 5.0 <= constants.DEFAULT_MOUSE_BODY_LENGTH_CM <= 15.0
        assert 5.0 <= constants.LEGACY_SPINE_LENGTH_CM <= 12.0

    def test_com_weights_sum_to_one(self):
        """Center of mass weights should sum to 1.0"""
        total_weight = (
            constants.COM_WEIGHT_SNOUT +
            constants.COM_WEIGHT_NECK +
            constants.COM_WEIGHT_SHOULDER +
            constants.COM_WEIGHT_RIB_CENTER +
            constants.COM_WEIGHT_HIP +
            constants.COM_WEIGHT_TAIL_BASE
        )
        assert abs(total_weight - 1.0) < 1e-6, f"COM weights sum to {total_weight}, expected 1.0"

    def test_likelihood_thresholds_valid_range(self):
        """Likelihood thresholds should be in [0, 1]"""
        assert 0.0 <= constants.MIN_LIKELIHOOD_KEYPOINT <= 1.0
        assert 0.0 <= constants.MIN_LIKELIHOOD_SCALING <= 1.0

    def test_likelihood_thresholds_reasonable(self):
        """Likelihood thresholds should be reasonably high for quality"""
        assert constants.MIN_LIKELIHOOD_KEYPOINT >= 0.8
        assert constants.MIN_LIKELIHOOD_SCALING >= 0.8


class TestSignalProcessingConstants:
    """Test signal processing parameter values"""

    def test_savgol_window_sizes_odd(self):
        """Savitzky-Golay window sizes must be odd"""
        assert constants.SAVGOL_WINDOW_SIZE_DEFAULT % 2 == 1
        assert constants.SAVGOL_WINDOW_SIZE_ADAPTIVE_HIGH % 2 == 1
        assert constants.SAVGOL_WINDOW_SIZE_ADAPTIVE_MED % 2 == 1
        assert constants.SAVGOL_WINDOW_SIZE_ADAPTIVE_LOW % 2 == 1

    def test_savgol_poly_order_less_than_window(self):
        """Polynomial order must be less than window size"""
        assert constants.SAVGOL_POLY_ORDER_DEFAULT < constants.SAVGOL_WINDOW_SIZE_DEFAULT

    def test_ema_alpha_valid_range(self):
        """EMA alpha should be in (0, 1)"""
        assert 0.0 < constants.EMA_ALPHA_DEFAULT < 1.0

    def test_scaling_tolerance_reasonable(self):
        """Scaling tolerance should be reasonable (not too strict or loose)"""
        assert 0.1 <= constants.SCALING_TOLERANCE_DEFAULT <= 0.5

    def test_mad_multiplier_positive(self):
        """MAD multiplier should be positive"""
        assert constants.MAD_MULTIPLIER_DEFAULT > 0

    def test_minimum_frame_requirements(self):
        """Minimum frame counts should be reasonable"""
        assert constants.MIN_FRAMES_FOR_SCALING >= 50
        assert constants.MIN_FRAMES_FOR_ANALYSIS >= 20
        assert constants.MIN_FRAMES_FOR_ANALYSIS <= constants.MIN_FRAMES_FOR_SCALING


class TestGaitDetectionConstants:
    """Test gait detection threshold values"""

    def test_mad_thresholds_positive(self):
        """MAD thresholds should be positive"""
        assert constants.MAD_THRESHOLD_STATIONARY_DEFAULT > 0
        assert constants.MAD_THRESHOLD_WALKING_DEFAULT > 0
        assert constants.MAD_THRESHOLD_WALKING_V12 > 0

    def test_threshold_ordering(self):
        """Stationary threshold should be less than walking threshold (v1.1)"""
        # Note: v1.2 has inverted logic (lower threshold for more sensitivity)
        assert constants.MAD_THRESHOLD_STATIONARY_DEFAULT < constants.MAD_THRESHOLD_WALKING_DEFAULT

    def test_adaptive_percentile_range(self):
        """Adaptive percentile should be in valid range"""
        assert 50 <= constants.ADAPTIVE_PERCENTILE_DEFAULT <= 100

    def test_min_threshold_positive(self):
        """Minimum threshold should be positive"""
        assert constants.MIN_THRESHOLD_PX_PER_FRAME > 0

    def test_duration_constraints_positive(self):
        """Duration constraints should be positive"""
        assert constants.MIN_WALKING_DURATION_SEC > 0
        assert constants.MIN_STATIONARY_DURATION_SEC > 0

    def test_stride_duration_ordering(self):
        """Min stride duration should be less than max"""
        assert constants.MIN_STRIDE_DURATION_SEC < constants.MAX_STRIDE_DURATION_SEC
        assert constants.MIN_STRIDE_DURATION_SEC_V12 < constants.MAX_STRIDE_DURATION_SEC

    def test_v12_stride_less_strict(self):
        """v1.2.0 stride threshold should be less strict than v1.1.0"""
        assert constants.MIN_STRIDE_DURATION_SEC_V12 < constants.MIN_STRIDE_DURATION_SEC


class TestTemporalConstants:
    """Test temporal/timing constants"""

    def test_fps_default_reasonable(self):
        """Default FPS should be in typical camera range"""
        assert 30.0 <= constants.FPS_DEFAULT <= 240.0

    def test_stride_analysis_window_positive(self):
        """Stride analysis window should be positive"""
        assert constants.STRIDE_ANALYSIS_WINDOW_SEC > 0


class TestStatisticalConstants:
    """Test statistical parameter values"""

    def test_confidence_interval_percentile(self):
        """CI percentile should be in valid range"""
        assert 50 <= constants.CI_PERCENTILE_DEFAULT <= 100

    def test_trim_percent_reasonable(self):
        """Trim percent should be reasonable (not trimming too much)"""
        assert 0 <= constants.TRIM_PERCENT_DEFAULT <= 25

    def test_epsilon_very_small(self):
        """Epsilon should be very small for numerical stability"""
        assert constants.EPSILON < 1e-6


class TestNumericalStabilityConstants:
    """Test numerical stability constants"""

    def test_min_vector_norm_very_small(self):
        """Minimum vector norm should be very small"""
        assert constants.MIN_VECTOR_NORM < 1e-6

    def test_cosine_clip_bounds(self):
        """Cosine clip bounds should be [-1, 1]"""
        assert constants.COSINE_CLIP_MIN == -1.0
        assert constants.COSINE_CLIP_MAX == 1.0


class TestValidationConstants:
    """Test anatomical validation limits"""

    def test_stride_length_limits_positive(self):
        """Stride length limits should be positive"""
        assert constants.MIN_STRIDE_LENGTH_CM > 0
        assert constants.MAX_STRIDE_LENGTH_CM > 0

    def test_stride_length_ordering(self):
        """Min stride length should be less than max"""
        assert constants.MIN_STRIDE_LENGTH_CM < constants.MAX_STRIDE_LENGTH_CM

    def test_stride_length_realistic(self):
        """Stride lengths should be realistic for mice"""
        assert constants.MIN_STRIDE_LENGTH_CM >= 0.3
        assert constants.MAX_STRIDE_LENGTH_CM <= 30.0

    def test_speed_limit_positive(self):
        """Max speed should be positive"""
        assert constants.MAX_SPEED_CM_PER_SEC > 0

    def test_rom_limits(self):
        """Range of motion limits should be valid"""
        assert constants.MIN_ROM_DEGREES == 0.0
        assert constants.MAX_ROM_DEGREES == 180.0

    def test_duty_factor_range(self):
        """Duty factor should be in [0, 1]"""
        assert constants.MIN_DUTY_FACTOR == 0.0
        assert constants.MAX_DUTY_FACTOR == 1.0

    def test_asymmetry_index_positive(self):
        """Max asymmetry index should be positive"""
        assert constants.MAX_ASYMMETRY_INDEX > 0


class TestDataQualityConstants:
    """Test data quality threshold values"""

    def test_completeness_thresholds_ordered(self):
        """Completeness thresholds should be in descending order"""
        assert constants.DATA_COMPLETENESS_HIGH > constants.DATA_COMPLETENESS_MEDIUM
        assert constants.DATA_COMPLETENESS_MEDIUM > 0

    def test_completeness_thresholds_in_range(self):
        """Completeness thresholds should be in [0, 1]"""
        assert 0.0 <= constants.DATA_COMPLETENESS_MEDIUM <= 1.0
        assert 0.0 <= constants.DATA_COMPLETENESS_HIGH <= 1.0


class TestVersionMarkers:
    """Test version marker strings"""

    def test_version_markers_format(self):
        """Version markers should have expected format"""
        assert constants.VERSION_MARKER_V11.startswith("v")
        assert constants.VERSION_MARKER_V12.startswith("v")

    def test_version_markers_unique(self):
        """Version markers should be unique"""
        assert constants.VERSION_MARKER_V11 != constants.VERSION_MARKER_V12


class TestExportConstants:
    """Test export formatting constants"""

    def test_plot_dpi_reasonable(self):
        """Plot DPI should be in reasonable range"""
        assert 72 <= constants.PLOT_DPI <= 600

    def test_plot_figsize_positive(self):
        """Plot figure sizes should be positive"""
        assert constants.PLOT_FIGSIZE_WIDTH > 0
        assert constants.PLOT_FIGSIZE_HEIGHT > 0

    def test_precision_values_reasonable(self):
        """Precision values should be reasonable"""
        assert 0 <= constants.EXCEL_FLOAT_PRECISION <= 10
        assert 0 <= constants.EXCEL_ANGLE_PRECISION <= 10

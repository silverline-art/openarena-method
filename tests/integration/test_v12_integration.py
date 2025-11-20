"""
Integration tests for v1.2.0 feature routing and pipeline execution.

Tests that v1.2.0 features are correctly activated based on configuration.
"""

import pytest
import numpy as np
from src.exmo_gait.config_schema import ExmoGaitConfig


class TestV12FeatureDetection:
    """Test v1.2.0 feature detection logic"""

    def test_v11_config_detection(self, config_v11_dict):
        """v1.1.0 config should be detected"""
        config = ExmoGaitConfig.from_dict(config_v11_dict)

        assert config.get_pipeline_version() == "v1.1.0"
        assert len(config.get_enabled_v12_features()) == 0

    def test_v12_config_detection(self, config_v12_dict):
        """v1.2.0 config should be detected"""
        config = ExmoGaitConfig.from_dict(config_v12_dict)

        assert config.get_pipeline_version() == "v1.2.0"
        assert len(config.get_enabled_v12_features()) > 0

    def test_partial_v12_features(self):
        """Partially enabled v1.2.0 features should be detected"""
        config_dict = {
            "scaling": {
                "scaling_method": "full_body"  # Only this v1.2 feature
            }
        }

        config = ExmoGaitConfig.from_dict(config_dict)

        assert config.get_pipeline_version() == "v1.2.0"
        features = config.get_enabled_v12_features()
        assert "full_body_scaling" in features


class TestV12ScalingMethod:
    """Test v1.2.0 full-body scaling routing"""

    def test_full_body_scaling_enabled(self):
        """Full-body scaling should be enabled in v1.2"""
        config = ExmoGaitConfig(
            scaling={
                "scaling_method": "full_body",
                "expected_body_length_cm": 10.0
            }
        )

        assert config.scaling.scaling_method == "full_body"
        assert config.scaling.expected_body_length_cm == 10.0

    def test_spine_only_scaling_legacy(self):
        """Spine-only scaling should be legacy v1.1"""
        config = ExmoGaitConfig(
            scaling={
                "scaling_method": "spine_only",
                "expected_body_length_cm": 8.0
            }
        )

        assert config.scaling.scaling_method == "spine_only"
        assert config.get_pipeline_version() == "v1.1.0"


class TestV12AdaptiveSmoothing:
    """Test v1.2.0 adaptive smoothing routing"""

    def test_adaptive_smoothing_enabled(self):
        """Adaptive smoothing should be enabled in v1.2"""
        config = ExmoGaitConfig(
            smoothing={
                "smoothing_adaptive": True,
                "smoothing_window": 7
            }
        )

        assert config.smoothing.smoothing_adaptive is True
        assert "adaptive_smoothing" in config.get_enabled_v12_features()

    def test_fixed_smoothing_legacy(self):
        """Fixed smoothing should be legacy v1.1"""
        config = ExmoGaitConfig(
            smoothing={
                "smoothing_adaptive": False,
                "smoothing_window": 11
            }
        )

        assert config.smoothing.smoothing_adaptive is False


class TestV12VelocityMethod:
    """Test v1.2.0 EMA velocity routing"""

    def test_ema_velocity_enabled(self):
        """EMA velocity should be enabled in v1.2"""
        config = ExmoGaitConfig(
            smoothing={
                "velocity_smoothing_method": "ema",
                "velocity_ema_alpha": 0.35
            }
        )

        assert config.smoothing.velocity_smoothing_method == "ema"
        assert "ema_velocity" in config.get_enabled_v12_features()

    def test_savgol_velocity_legacy(self):
        """Savitzky-Golay velocity should be legacy v1.1"""
        config = ExmoGaitConfig(
            smoothing={
                "velocity_smoothing_method": "savgol"
            }
        )

        assert config.smoothing.velocity_smoothing_method == "savgol"


class TestV12HybridThreshold:
    """Test v1.2.0 hybrid threshold routing"""

    def test_hybrid_threshold_enabled(self):
        """Hybrid threshold should be enabled in v1.2"""
        config = ExmoGaitConfig(
            phase_detection={
                "use_hybrid_threshold": True,
                "adaptive_percentile": 55
            }
        )

        assert config.phase_detection.use_hybrid_threshold is True
        assert "hybrid_threshold" in config.get_enabled_v12_features()

    def test_mad_threshold_legacy(self):
        """MAD-only threshold should be legacy v1.1"""
        config = ExmoGaitConfig(
            phase_detection={
                "use_hybrid_threshold": False
            }
        )

        assert config.phase_detection.use_hybrid_threshold is False


class TestV123DCOM:
    """Test v1.2.0 3D center of mass routing"""

    def test_3d_com_enabled(self):
        """3D COM should be enabled in v1.2"""
        config = ExmoGaitConfig(
            com={
                "use_3d_com": True
            }
        )

        assert config.com.use_3d_com is True
        assert "3d_com" in config.get_enabled_v12_features()

    def test_2d_com_legacy(self):
        """2D COM should be legacy v1.1"""
        config = ExmoGaitConfig(
            com={
                "use_3d_com": False
            }
        )

        assert config.com.use_3d_com is False


class TestV12EnhancedStatistics:
    """Test v1.2.0 enhanced statistics routing"""

    def test_enhanced_statistics_enabled(self):
        """Enhanced statistics should be enabled in v1.2"""
        config = ExmoGaitConfig(
            aggregation={
                "aggregation_include_ci": True,
                "aggregation_ci_percentile": 95,
                "aggregation_trim_percent": 5
            }
        )

        assert config.aggregation.aggregation_include_ci is True
        assert "enhanced_statistics" in config.get_enabled_v12_features()

    def test_basic_statistics_legacy(self):
        """Basic statistics should be legacy v1.1"""
        config = ExmoGaitConfig(
            aggregation={
                "aggregation_include_ci": False
            }
        )

        assert config.aggregation.aggregation_include_ci is False


class TestV12ParameterValues:
    """Test v1.2.0 calibrated parameter values"""

    def test_v12_calibrated_thresholds(self):
        """v1.2.0 should have more sensitive thresholds"""
        config_v12 = ExmoGaitConfig.from_dict({
            "phase_detection": {
                "stationary_mad_threshold": 0.9,
                "walking_mad_threshold": 0.8,
                "use_hybrid_threshold": True
            }
        })

        # v1.2 thresholds should be lower (more sensitive)
        assert config_v12.phase_detection.stationary_mad_threshold < 1.5
        assert config_v12.phase_detection.walking_mad_threshold < 2.0

    def test_v12_reduced_smoothing(self):
        """v1.2.0 should have reduced smoothing windows"""
        config_v12 = ExmoGaitConfig.from_dict({
            "smoothing": {
                "smoothing_window": 7,
                "smoothing_adaptive": True
            }
        })

        # v1.2 should use smaller window
        assert config_v12.smoothing.smoothing_window < 11

    def test_v12_relaxed_stride_duration(self):
        """v1.2.0 should allow shorter strides"""
        config_v12 = ExmoGaitConfig.from_dict({
            "stride_detection": {
                "min_stride_duration": 0.05
            }
        })

        # v1.2 should allow shorter strides than v1.1 (0.1s)
        assert config_v12.stride_detection.min_stride_duration < 0.1


class TestConfigRoundTrip:
    """Test configuration serialization round-trip"""

    def test_to_dict_from_dict_roundtrip(self, config_v12_dict):
        """Config should survive dict round-trip"""
        config1 = ExmoGaitConfig.from_dict(config_v12_dict)
        config_dict = config1.to_dict()
        config2 = ExmoGaitConfig.from_dict(config_dict)

        # Should have same version detection
        assert config1.get_pipeline_version() == config2.get_pipeline_version()
        assert config1.get_enabled_v12_features() == config2.get_enabled_v12_features()

    def test_yaml_roundtrip(self, temp_yaml_config):
        """Config should survive YAML round-trip"""
        config1 = ExmoGaitConfig.from_yaml(str(temp_yaml_config))

        # Write to new YAML
        import yaml
        from pathlib import Path

        temp_dir = Path(temp_yaml_config).parent
        new_yaml = temp_dir / "roundtrip_config.yaml"

        with open(new_yaml, 'w') as f:
            yaml.dump(config1.to_dict(), f)

        # Read back
        config2 = ExmoGaitConfig.from_yaml(str(new_yaml))

        # Should have same settings
        assert config1.general.fps == config2.general.fps
        assert config1.scaling.scaling_method == config2.scaling.scaling_method


@pytest.mark.slow
class TestPipelineIntegration:
    """End-to-end integration tests (marked slow)"""

    def test_v11_pipeline_flow(self, config_v11_dict):
        """v1.1.0 config should select legacy methods"""
        config = ExmoGaitConfig.from_dict(config_v11_dict)

        # Verify legacy method selection
        assert config.scaling.scaling_method == "spine_only"
        assert config.smoothing.smoothing_adaptive is False
        assert config.phase_detection.use_hybrid_threshold is False
        assert config.com.use_3d_com is False
        assert config.aggregation.aggregation_include_ci is False

    def test_v12_pipeline_flow(self, config_v12_dict):
        """v1.2.0 config should select new methods"""
        config = ExmoGaitConfig.from_dict(config_v12_dict)

        # Verify new method selection
        assert config.scaling.scaling_method == "full_body"
        assert config.smoothing.smoothing_adaptive is True
        assert config.smoothing.velocity_smoothing_method == "ema"
        assert config.phase_detection.use_hybrid_threshold is True
        assert config.com.use_3d_com is True
        assert config.aggregation.aggregation_include_ci is True

    def test_mixed_version_features(self):
        """Should handle mixed v1.1/v1.2 features gracefully"""
        config_dict = {
            "scaling": {
                "scaling_method": "full_body"  # v1.2
            },
            "smoothing": {
                "smoothing_adaptive": False  # v1.1
            }
        }

        config = ExmoGaitConfig.from_dict(config_dict)

        # Should be detected as v1.2 (has at least one v1.2 feature)
        assert config.get_pipeline_version() == "v1.2.0"

        features = config.get_enabled_v12_features()
        assert "full_body_scaling" in features
        assert "adaptive_smoothing" not in features

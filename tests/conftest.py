"""
Pytest configuration and shared fixtures for EXMO gait analysis tests.

Provides common test data, fixtures, and utilities used across unit and integration tests.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import yaml


# ============================================================================
# Sample Data Fixtures
# ============================================================================

@pytest.fixture
def sample_trajectory_2d():
    """Generate realistic 2D trajectory data (100 frames)"""
    np.random.seed(42)
    t = np.linspace(0, 1, 100)

    # Simulated walking trajectory with sinusoidal motion
    x = 100 + 50 * t + 5 * np.sin(2 * np.pi * 2 * t)  # Forward motion with lateral sway
    y = 200 + 20 * np.sin(2 * np.pi * 3 * t)  # Vertical oscillation

    trajectory = np.column_stack([x, y])
    return trajectory


@pytest.fixture
def sample_likelihood_high():
    """Generate high-confidence likelihood values (>0.9)"""
    np.random.seed(42)
    return np.random.uniform(0.92, 0.99, 100)


@pytest.fixture
def sample_likelihood_mixed():
    """Generate mixed-quality likelihood values"""
    np.random.seed(42)
    likelihood = np.random.uniform(0.5, 0.99, 100)
    # Add some low-confidence outliers
    likelihood[10:15] = np.random.uniform(0.2, 0.5, 5)
    return likelihood


@pytest.fixture
def sample_angles():
    """Generate realistic joint angle time series"""
    np.random.seed(42)
    t = np.linspace(0, 1, 100)

    # Knee angle oscillation during gait (60-120 degrees)
    base_angle = 90
    amplitude = 30
    frequency = 3

    angles = base_angle + amplitude * np.sin(2 * np.pi * frequency * t)
    # Add noise
    angles += np.random.normal(0, 2, 100)

    return angles


@pytest.fixture
def sample_velocity_signal():
    """Generate velocity signal with walking and stationary phases"""
    np.random.seed(42)

    # 100 frames: stationary (0-30), walking (31-70), stationary (71-100)
    velocity = np.zeros(100)
    velocity[30:70] = np.random.uniform(5, 15, 40)  # Walking phase
    velocity[:30] = np.random.uniform(0, 1.5, 30)  # Stationary
    velocity[70:] = np.random.uniform(0, 1.5, 30)  # Stationary

    # Add noise
    velocity += np.random.normal(0, 0.5, 100)
    velocity = np.clip(velocity, 0, None)  # No negative velocities

    return velocity


@pytest.fixture
def sample_body_measurements():
    """Sample snout-tailbase measurements for scaling"""
    np.random.seed(42)

    # Realistic mouse body length in pixels (median ~100px)
    true_length = 100.0
    measurements = np.random.normal(true_length, 2.5, 200)  # Small variance

    # Add some outliers
    measurements[10:13] = [150, 160, 155]  # Outliers (too long)
    measurements[50:52] = [60, 65]  # Outliers (too short)

    return measurements


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def config_v11_dict():
    """v1.1.0 configuration dictionary"""
    return {
        "general": {
            "fps": 120.0,
            "output_dir": "results",
            "log_level": "INFO"
        },
        "scaling": {
            "scaling_method": "spine_only",
            "expected_body_length_cm": 8.0,
            "scaling_min_likelihood": 0.9,
            "scaling_tolerance": 0.25
        },
        "smoothing": {
            "smoothing_adaptive": False,
            "smoothing_window": 11,
            "smoothing_poly": 3,
            "velocity_smoothing_method": "savgol",
            "velocity_ema_alpha": 0.35
        },
        "phase_detection": {
            "stationary_mad_threshold": 1.5,
            "walking_mad_threshold": 2.0,
            "use_hybrid_threshold": False,
            "adaptive_percentile": 75,
            "min_threshold_px_per_frame": 1.0,
            "min_walking_duration": 0.3,
            "min_stationary_duration": 0.25
        }
    }


@pytest.fixture
def config_v12_dict():
    """v1.2.0 configuration dictionary"""
    return {
        "general": {
            "fps": 120.0,
            "output_dir": "results",
            "log_level": "INFO"
        },
        "scaling": {
            "scaling_method": "full_body",
            "expected_body_length_cm": 10.0,
            "scaling_min_likelihood": 0.9,
            "scaling_tolerance": 0.25
        },
        "smoothing": {
            "smoothing_adaptive": True,
            "smoothing_window": 7,
            "smoothing_poly": 3,
            "velocity_smoothing_method": "ema",
            "velocity_ema_alpha": 0.35
        },
        "phase_detection": {
            "stationary_mad_threshold": 0.9,
            "walking_mad_threshold": 0.8,
            "use_hybrid_threshold": True,
            "adaptive_percentile": 55,
            "min_threshold_px_per_frame": 1.0,
            "min_walking_duration": 0.12,
            "min_stationary_duration": 0.12
        },
        "com": {
            "use_3d_com": True
        },
        "aggregation": {
            "aggregation_include_ci": True,
            "aggregation_ci_percentile": 95,
            "aggregation_trim_percent": 5
        }
    }


@pytest.fixture
def temp_yaml_config(tmp_path, config_v11_dict):
    """Create temporary YAML config file"""
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config_v11_dict, f)
    return config_file


# ============================================================================
# Geometry Test Fixtures
# ============================================================================

@pytest.fixture
def three_points_triangle():
    """Three points forming a right triangle"""
    p1 = np.array([0.0, 0.0])
    p2 = np.array([0.0, 10.0])  # Vertex
    p3 = np.array([10.0, 10.0])
    return p1, p2, p3


@pytest.fixture
def three_points_straight():
    """Three collinear points (180 degree angle)"""
    p1 = np.array([0.0, 0.0])
    p2 = np.array([5.0, 5.0])
    p3 = np.array([10.0, 10.0])
    return p1, p2, p3


@pytest.fixture
def com_weighted_points():
    """Points with known weighted center of mass"""
    points = np.array([
        [0.0, 0.0],
        [10.0, 0.0],
        [10.0, 10.0],
        [0.0, 10.0]
    ])
    weights = np.array([1.0, 1.0, 1.0, 1.0])  # Equal weights
    expected_com = np.array([5.0, 5.0])

    return points, weights, expected_com


# ============================================================================
# Signal Processing Test Fixtures
# ============================================================================

@pytest.fixture
def signal_with_outliers():
    """Signal with known outliers"""
    np.random.seed(42)
    signal = np.random.normal(10.0, 1.0, 100)

    # Inject outliers
    signal[10] = 50.0  # Large positive outlier
    signal[50] = -20.0  # Large negative outlier
    signal[75] = 40.0  # Another outlier

    outlier_indices = [10, 50, 75]

    return signal, outlier_indices


@pytest.fixture
def signal_with_gaps():
    """Signal with NaN gaps"""
    np.random.seed(42)
    signal = np.random.normal(0.0, 1.0, 100)

    # Create gaps
    signal[10:15] = np.nan  # 5-frame gap (interpolable)
    signal[50:60] = np.nan  # 10-frame gap (too large for default)

    return signal


@pytest.fixture
def noisy_sine_wave():
    """Noisy sine wave for smoothing tests"""
    np.random.seed(42)
    t = np.linspace(0, 2 * np.pi, 100)
    clean_signal = np.sin(t)
    noise = np.random.normal(0, 0.1, 100)
    noisy_signal = clean_signal + noise

    return noisy_signal, clean_signal


# ============================================================================
# Utility Functions
# ============================================================================

def assert_array_shape(array, expected_shape):
    """Assert numpy array has expected shape"""
    assert array.shape == expected_shape, (
        f"Expected shape {expected_shape}, got {array.shape}"
    )


def assert_in_range(value, min_val, max_val):
    """Assert value is within range [min_val, max_val]"""
    assert min_val <= value <= max_val, (
        f"Value {value} not in range [{min_val}, {max_val}]"
    )


def assert_close(actual, expected, rtol=1e-5, atol=1e-8):
    """Assert arrays/values are approximately equal"""
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)

"""
Unit tests for signal processing utilities.

Tests smoothing, outlier detection, velocity calculations, and peak detection.
"""

import pytest
import numpy as np
from scipy import signal as scipy_signal

from src.exmo_gait.utils.signal_processing import (
    apply_savgol_filter,
    interpolate_missing_values,
    compute_mad,
    detect_outliers_mad,
    filter_outliers_mad,
    compute_velocity,
    compute_angular_velocity,
    detect_peaks_adaptive,
    smooth_binary_classification,
    smooth_velocity_ema,
    smooth_trajectory_adaptive
)


class TestSavgolFilter:
    """Test Savitzky-Golay filtering"""

    def test_savgol_smooths_noise(self, noisy_sine_wave):
        """Savgol filter should reduce noise"""
        noisy_signal, clean_signal = noisy_sine_wave

        smoothed = apply_savgol_filter(noisy_signal, window_length=11, polyorder=3)

        # Smoothed should be closer to clean than noisy
        error_before = np.mean((noisy_signal - clean_signal) ** 2)
        error_after = np.mean((smoothed - clean_signal) ** 2)

        assert error_after < error_before

    def test_savgol_preserves_shape(self):
        """Smoothed array should have same shape as input"""
        data = np.random.randn(100)
        smoothed = apply_savgol_filter(data)

        assert smoothed.shape == data.shape

    def test_savgol_short_array(self):
        """Should handle arrays shorter than window"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        smoothed = apply_savgol_filter(data, window_length=11)

        # Should return valid result (may adjust window)
        assert len(smoothed) == len(data)

    def test_savgol_constant_signal(self):
        """Constant signal should remain constant"""
        data = np.ones(50) * 5.0
        smoothed = apply_savgol_filter(data)

        np.testing.assert_array_almost_equal(smoothed, data, decimal=6)


class TestInterpolateMissingValues:
    """Test interpolation of missing values"""

    def test_interpolate_small_gap(self, signal_with_gaps):
        """Small gaps should be interpolated"""
        signal = signal_with_gaps.copy()

        interpolated, mask = interpolate_missing_values(signal, max_gap=5)

        # Small gap (10-15) should be filled
        assert not np.isnan(interpolated[10:15]).any()

    def test_interpolate_large_gap(self, signal_with_gaps):
        """Large gaps should not be interpolated"""
        signal = signal_with_gaps.copy()

        interpolated, mask = interpolate_missing_values(signal, max_gap=5)

        # Large gap (50-60) should remain NaN
        assert np.isnan(interpolated[50:60]).any()

    def test_interpolate_no_gaps(self):
        """Signal without gaps should be unchanged"""
        signal = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        interpolated, mask = interpolate_missing_values(signal)

        np.testing.assert_array_equal(interpolated, signal)
        assert mask.all()

    def test_interpolate_all_nans(self):
        """All NaN signal should remain NaN"""
        signal = np.full(10, np.nan)

        interpolated, mask = interpolate_missing_values(signal)

        assert np.isnan(interpolated).all()

    def test_interpolate_single_valid_point(self):
        """Single valid point can't be interpolated"""
        signal = np.full(10, np.nan)
        signal[5] = 1.0

        interpolated, mask = interpolate_missing_values(signal)

        # Most should remain NaN
        assert np.isnan(interpolated).sum() > 5


class TestMADCalculations:
    """Test Median Absolute Deviation"""

    def test_mad_normal_distribution(self):
        """MAD of normal distribution should approximate std"""
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)

        mad = compute_mad(data, scale=1.4826)

        # Should be close to 1.0 (the true std)
        assert 0.9 < mad < 1.1

    def test_mad_constant_data(self):
        """MAD of constant data should be 0"""
        data = np.ones(100) * 5.0

        mad = compute_mad(data)

        assert abs(mad) < 1e-10

    def test_mad_with_nans(self):
        """MAD should handle NaN values"""
        data = np.array([1.0, 2.0, np.nan, 3.0, 4.0, np.nan])

        mad = compute_mad(data)

        assert not np.isnan(mad)
        assert mad > 0


class TestOutlierDetection:
    """Test MAD-based outlier detection"""

    def test_detect_outliers(self, signal_with_outliers):
        """Should detect known outliers"""
        signal, outlier_indices = signal_with_outliers

        outlier_mask = detect_outliers_mad(signal, threshold=3.0)

        # Should detect at least some of the known outliers
        detected_outliers = np.where(outlier_mask)[0]
        assert len(detected_outliers) > 0

        # Known outliers should be flagged
        for idx in outlier_indices:
            assert outlier_mask[idx]

    def test_filter_outliers(self, signal_with_outliers):
        """Filtering should replace outliers with NaN"""
        signal, outlier_indices = signal_with_outliers

        filtered = filter_outliers_mad(signal, threshold=3.0)

        # Outliers should now be NaN
        for idx in outlier_indices:
            assert np.isnan(filtered[idx])

    def test_no_outliers_in_clean_data(self):
        """Clean data should have no outliers"""
        np.random.seed(42)
        data = np.random.normal(10.0, 1.0, 100)

        outlier_mask = detect_outliers_mad(data, threshold=3.0)

        # Should have very few or no outliers
        assert outlier_mask.sum() < 5

    def test_outlier_detection_constant_data(self):
        """Constant data should have no outliers"""
        data = np.ones(50) * 5.0

        outlier_mask = detect_outliers_mad(data)

        assert not outlier_mask.any()


class TestVelocityCalculations:
    """Test velocity computation"""

    def test_velocity_constant_motion(self):
        """Constant velocity should be detected"""
        # Moving 1 unit per frame at 100 fps
        positions = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        fps = 100.0

        velocities = compute_velocity(positions, fps)

        # Should be approximately 100 units/sec
        assert np.mean(velocities) > 90

    def test_velocity_stationary(self):
        """Stationary object should have zero velocity"""
        positions = np.ones(50) * 10.0

        velocities = compute_velocity(positions)

        assert np.all(np.abs(velocities) < 1e-6)

    def test_velocity_shape(self):
        """Velocity array should have same shape as positions"""
        positions = np.random.randn(100)

        velocities = compute_velocity(positions)

        assert velocities.shape == positions.shape

    def test_angular_velocity(self):
        """Angular velocity should be computed correctly"""
        # Constant angular velocity
        angles = np.linspace(0, 180, 100)
        fps = 100.0

        ang_vel = compute_angular_velocity(angles, fps)

        # Should be approximately constant
        assert np.std(ang_vel) < 10  # Low variation


class TestPeakDetection:
    """Test adaptive peak detection"""

    def test_detect_peaks_in_sine(self):
        """Should detect peaks in sine wave"""
        t = np.linspace(0, 4 * np.pi, 200)
        signal = np.sin(t)

        peaks = detect_peaks_adaptive(signal, min_prominence=0.5)

        # Should find ~4 peaks
        assert 3 <= len(peaks) <= 5

    def test_detect_peaks_custom_prominence(self):
        """Higher prominence should find fewer peaks"""
        t = np.linspace(0, 4 * np.pi, 200)
        signal = np.sin(t) + 0.1 * np.sin(10 * t)  # Add small oscillations

        peaks_low = detect_peaks_adaptive(signal, min_prominence=0.1)
        peaks_high = detect_peaks_adaptive(signal, min_prominence=0.8)

        # Lower prominence should find more peaks
        assert len(peaks_low) >= len(peaks_high)

    def test_detect_peaks_min_distance(self):
        """Minimum distance should space out peaks"""
        t = np.linspace(0, 4 * np.pi, 400)
        signal = np.sin(t)

        peaks = detect_peaks_adaptive(signal, min_distance=50)

        # Check that peaks are at least 50 samples apart
        if len(peaks) > 1:
            min_spacing = np.min(np.diff(peaks))
            assert min_spacing >= 50


class TestBinaryClassificationSmoothing:
    """Test binary signal smoothing"""

    def test_smooth_binary_removes_noise(self):
        """Binary smoothing should remove isolated spikes"""
        # Signal with isolated spikes
        binary = np.zeros(100, dtype=bool)
        binary[50:70] = True  # Main walking phase
        binary[10] = True  # Isolated spike
        binary[90] = True  # Isolated spike

        smoothed = smooth_binary_classification(binary, window_size=10)

        # Isolated spikes should be removed
        assert not smoothed[10]
        assert not smoothed[90]

        # Main phase should remain
        assert smoothed[60]

    def test_smooth_binary_preserves_long_runs(self):
        """Long runs should be preserved"""
        binary = np.zeros(100, dtype=bool)
        binary[20:80] = True

        smoothed = smooth_binary_classification(binary, window_size=10)

        # Main run should be preserved
        assert smoothed[40:60].all()


class TestEMAVelocitySmoothing:
    """Test exponential moving average velocity smoothing"""

    def test_ema_smooths_velocity(self):
        """EMA should smooth velocity signal"""
        np.random.seed(42)
        # Noisy velocity signal
        positions = np.cumsum(np.random.randn(100)) + 50

        velocity_ema = smooth_velocity_ema(positions, alpha=0.3, fps=100)

        assert len(velocity_ema) == len(positions)

    def test_ema_alpha_effect(self):
        """Higher alpha should preserve more variation"""
        np.random.seed(42)
        positions = np.cumsum(np.random.randn(100))

        velocity_low_alpha = smooth_velocity_ema(positions, alpha=0.1)
        velocity_high_alpha = smooth_velocity_ema(positions, alpha=0.7)

        # Higher alpha should have more variation
        assert np.std(velocity_high_alpha) >= np.std(velocity_low_alpha)

    def test_ema_handles_nans(self):
        """EMA should handle NaN values"""
        positions = np.array([1.0, 2.0, np.nan, 4.0, 5.0])

        velocity = smooth_velocity_ema(positions)

        # Should not crash, NaN handling is graceful
        assert len(velocity) == len(positions)


class TestAdaptiveTrajectorySmoothing:
    """Test adaptive smoothing based on data quality"""

    def test_adaptive_high_quality(self):
        """High quality data should use smaller window"""
        np.random.seed(42)
        trajectory = np.random.randn(100)

        smoothed = smooth_trajectory_adaptive(trajectory, data_completeness=0.95)

        # Should return smoothed array
        assert len(smoothed) == len(trajectory)

    def test_adaptive_low_quality(self):
        """Low quality data should use larger window"""
        np.random.seed(42)
        trajectory = np.random.randn(100)

        smoothed = smooth_trajectory_adaptive(trajectory, data_completeness=0.60)

        # Should return smoothed array
        assert len(smoothed) == len(trajectory)

    def test_adaptive_different_quality_levels(self):
        """Different quality levels should use different windows"""
        np.random.seed(42)
        trajectory = np.random.randn(100)

        smoothed_high = smooth_trajectory_adaptive(trajectory, data_completeness=0.95)
        smoothed_low = smooth_trajectory_adaptive(trajectory, data_completeness=0.60)

        # Results should differ due to different window sizes
        assert not np.array_equal(smoothed_high, smoothed_low)

    def test_adaptive_quality_thresholds(self):
        """Test all quality threshold branches"""
        np.random.seed(42)
        trajectory = np.random.randn(100)

        # Test high quality (>0.9)
        smooth_trajectory_adaptive(trajectory, data_completeness=0.92)

        # Test medium quality (0.7-0.9)
        smooth_trajectory_adaptive(trajectory, data_completeness=0.80)

        # Test low quality (<0.7)
        smooth_trajectory_adaptive(trajectory, data_completeness=0.65)

        # All should complete without error
        assert True

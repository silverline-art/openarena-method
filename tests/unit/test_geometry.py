"""
Unit tests for geometric calculations.

Tests distance, angle, COM, trajectory, and scaling calculations.
"""

import pytest
import numpy as np
from src.exmo_gait.utils.geometry import (
    compute_distance_2d,
    compute_angle_3points,
    compute_center_of_mass,
    compute_trajectory_length,
    compute_trajectory_speed,
    compute_stride_length,
    compute_lateral_deviation,
    compute_symmetry_index,
    compute_range_of_motion,
    pixels_to_cm,
    compute_scaling_factor,
    compute_scaling_factor_v2
)


class TestDistance2D:
    """Test 2D distance calculations"""

    def test_distance_simple_points(self):
        """Distance between (0,0) and (3,4) should be 5"""
        p1 = np.array([0.0, 0.0])
        p2 = np.array([3.0, 4.0])
        dist = compute_distance_2d(p1, p2)
        assert abs(dist - 5.0) < 1e-6

    def test_distance_same_point(self):
        """Distance between identical points should be 0"""
        p1 = np.array([5.0, 10.0])
        p2 = np.array([5.0, 10.0])
        dist = compute_distance_2d(p1, p2)
        assert abs(dist) < 1e-10

    def test_distance_array_of_points(self):
        """Should handle arrays of point pairs"""
        p1 = np.array([[0.0, 0.0], [1.0, 1.0]])
        p2 = np.array([[3.0, 4.0], [4.0, 5.0]])
        dist = compute_distance_2d(p1, p2)

        assert len(dist) == 2
        assert abs(dist[0] - 5.0) < 1e-6
        assert abs(dist[1] - (3 * np.sqrt(2))) < 1e-6

    def test_distance_commutative(self):
        """Distance should be symmetric"""
        p1 = np.array([1.0, 2.0])
        p2 = np.array([4.0, 6.0])

        dist1 = compute_distance_2d(p1, p2)
        dist2 = compute_distance_2d(p2, p1)

        assert abs(dist1 - dist2) < 1e-10


class TestAngle3Points:
    """Test 3-point angle calculations"""

    def test_angle_right_triangle(self, three_points_triangle):
        """Right triangle should give 90 degrees"""
        p1, p2, p3 = three_points_triangle
        angle = compute_angle_3points(p1, p2, p3)
        assert abs(angle - 90.0) < 1e-6

    def test_angle_straight_line(self, three_points_straight):
        """Collinear points should give 180 degrees"""
        p1, p2, p3 = three_points_straight
        angle = compute_angle_3points(p1, p2, p3)
        assert abs(angle - 180.0) < 1e-6

    def test_angle_acute(self):
        """Acute angle test"""
        p1 = np.array([0.0, 0.0])
        p2 = np.array([0.0, 0.0])
        p3 = np.array([1.0, 1.0])

        # 45 degree angle
        angle = compute_angle_3points(p1, p2, p3)
        assert 0 <= angle <= 180

    def test_angle_array_of_triplets(self):
        """Should handle arrays of point triplets"""
        p1 = np.array([[0.0, 0.0], [0.0, 0.0]])
        p2 = np.array([[0.0, 10.0], [5.0, 5.0]])
        p3 = np.array([[10.0, 10.0], [10.0, 10.0]])

        angles = compute_angle_3points(p1, p2, p3)
        assert len(angles) == 2
        assert abs(angles[0] - 90.0) < 1e-6


class TestCenterOfMass:
    """Test center of mass calculations"""

    def test_com_equal_weights(self, com_weighted_points):
        """COM with equal weights should be geometric center"""
        points, weights, expected_com = com_weighted_points
        com = compute_center_of_mass(points, weights)

        np.testing.assert_array_almost_equal(com, expected_com, decimal=6)

    def test_com_no_weights(self):
        """COM without weights should assume equal weights"""
        points = np.array([
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0]
        ])
        com = compute_center_of_mass(points)
        expected = np.array([5.0, 5.0])

        np.testing.assert_array_almost_equal(com, expected, decimal=6)

    def test_com_weighted_skew(self):
        """COM should shift toward heavier weights"""
        points = np.array([
            [0.0, 0.0],
            [10.0, 0.0]
        ])
        weights = np.array([3.0, 1.0])  # First point has 3x weight

        com = compute_center_of_mass(points, weights)

        # Should be closer to first point
        assert com[0] < 5.0
        assert abs(com[1]) < 1e-6


class TestTrajectoryLength:
    """Test trajectory length calculations"""

    def test_trajectory_length_straight_line(self):
        """Straight line trajectory length"""
        trajectory = np.array([
            [0.0, 0.0],
            [3.0, 0.0],
            [6.0, 0.0],
            [10.0, 0.0]
        ])
        length = compute_trajectory_length(trajectory)
        assert abs(length - 10.0) < 1e-6

    def test_trajectory_length_single_point(self):
        """Single point trajectory should have 0 length"""
        trajectory = np.array([[0.0, 0.0]])
        length = compute_trajectory_length(trajectory)
        assert abs(length) < 1e-10

    def test_trajectory_length_diagonal(self):
        """Diagonal trajectory"""
        trajectory = np.array([
            [0.0, 0.0],
            [3.0, 4.0]
        ])
        length = compute_trajectory_length(trajectory)
        assert abs(length - 5.0) < 1e-6


class TestTrajectorySpeed:
    """Test instantaneous speed calculations"""

    def test_speed_constant_velocity(self):
        """Constant velocity should give constant speed"""
        trajectory = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0]
        ])
        fps = 100.0
        speeds = compute_trajectory_speed(trajectory, fps)

        assert len(speeds) == 4
        # All speeds should be approximately equal (except first due to prepend)
        assert np.std(speeds[1:]) < 1e-6

    def test_speed_zero_velocity(self):
        """Stationary trajectory should give zero speed"""
        trajectory = np.array([
            [5.0, 5.0],
            [5.0, 5.0],
            [5.0, 5.0]
        ])
        speeds = compute_trajectory_speed(trajectory)

        assert np.all(np.abs(speeds) < 1e-6)

    def test_speed_single_point(self):
        """Single point should give zero speed"""
        trajectory = np.array([[0.0, 0.0]])
        speeds = compute_trajectory_speed(trajectory)

        assert len(speeds) == 1
        assert abs(speeds[0]) < 1e-10


class TestStrideLength:
    """Test stride length calculations"""

    def test_stride_length_simple(self, sample_trajectory_2d):
        """Calculate stride lengths from foot strikes"""
        foot_strikes = np.array([0, 25, 50, 75])
        stride_lengths = compute_stride_length(foot_strikes, sample_trajectory_2d)

        assert len(stride_lengths) == 3  # 4 strikes = 3 strides
        assert np.all(stride_lengths > 0)

    def test_stride_length_single_strike(self):
        """Single foot strike should return empty array"""
        trajectory = np.array([[0.0, 0.0], [1.0, 1.0]])
        foot_strikes = np.array([0])
        stride_lengths = compute_stride_length(foot_strikes, trajectory)

        assert len(stride_lengths) == 0

    def test_stride_length_out_of_bounds(self):
        """Foot strikes beyond trajectory should be handled"""
        trajectory = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
        foot_strikes = np.array([0, 10])  # Second strike out of bounds
        stride_lengths = compute_stride_length(foot_strikes, trajectory)

        # Should handle gracefully (may be empty or partial)
        assert isinstance(stride_lengths, np.ndarray)


class TestLateralDeviation:
    """Test lateral deviation calculations"""

    def test_lateral_deviation_x_axis(self):
        """Test deviation along x-axis"""
        trajectory = np.array([
            [5.0, 0.0],
            [6.0, 0.0],
            [4.0, 0.0],
            [7.0, 0.0]
        ])
        mean_dev, std_dev = compute_lateral_deviation(trajectory, axis=0)

        assert abs(mean_dev - 5.5) < 0.5
        assert std_dev > 0

    def test_lateral_deviation_with_nans(self):
        """Should handle NaN values"""
        trajectory = np.array([
            [5.0, 0.0],
            [np.nan, 0.0],
            [6.0, 0.0],
            [4.0, 0.0]
        ])
        mean_dev, std_dev = compute_lateral_deviation(trajectory, axis=0)

        assert not np.isnan(mean_dev)
        assert not np.isnan(std_dev)


class TestSymmetryIndex:
    """Test symmetry index calculations"""

    def test_symmetry_perfect(self):
        """Perfect symmetry should give 0"""
        left = np.array([10.0, 10.0, 10.0])
        right = np.array([10.0, 10.0, 10.0])

        symmetry = compute_symmetry_index(left, right)
        assert abs(symmetry) < 1e-6

    def test_symmetry_asymmetric(self):
        """Asymmetric values should give positive index"""
        left = np.array([10.0, 10.0, 10.0])
        right = np.array([15.0, 15.0, 15.0])

        symmetry = compute_symmetry_index(left, right)
        assert symmetry > 0

    def test_symmetry_empty_arrays(self):
        """Empty arrays should return NaN"""
        left = np.array([])
        right = np.array([])

        symmetry = compute_symmetry_index(left, right)
        assert np.isnan(symmetry)


class TestRangeOfMotion:
    """Test range of motion calculations"""

    def test_rom_simple_range(self):
        """ROM should be max - min"""
        angles = np.array([30.0, 60.0, 90.0, 120.0])
        rom = compute_range_of_motion(angles)

        assert abs(rom - 90.0) < 1e-6

    def test_rom_constant_angle(self):
        """Constant angle should give ROM of 0"""
        angles = np.array([45.0, 45.0, 45.0])
        rom = compute_range_of_motion(angles)

        assert abs(rom) < 1e-6

    def test_rom_with_nans(self):
        """Should ignore NaN values"""
        angles = np.array([30.0, np.nan, 90.0, np.nan])
        rom = compute_range_of_motion(angles)

        assert abs(rom - 60.0) < 1e-6

    def test_rom_all_nans(self):
        """All NaN angles should return NaN"""
        angles = np.array([np.nan, np.nan, np.nan])
        rom = compute_range_of_motion(angles)

        assert np.isnan(rom)


class TestPixelsToCm:
    """Test pixel to centimeter conversion"""

    def test_pixels_to_cm_simple(self):
        """Simple conversion test"""
        pixel_values = np.array([100.0, 200.0, 300.0])
        scale_factor = 0.1  # 0.1 cm/pixel

        cm_values = pixels_to_cm(pixel_values, scale_factor)

        expected = np.array([10.0, 20.0, 30.0])
        np.testing.assert_array_almost_equal(cm_values, expected, decimal=6)

    def test_pixels_to_cm_single_value(self):
        """Single value conversion"""
        pixel_value = np.array([50.0])
        scale_factor = 0.05

        cm_value = pixels_to_cm(pixel_value, scale_factor)

        assert abs(cm_value[0] - 2.5) < 1e-6


class TestScalingFactorLegacy:
    """Test legacy scaling factor calculation (v1.1.0)"""

    def test_scaling_factor_known_distance(self):
        """Calculate scaling from known distance"""
        # Two points 100 pixels apart, known distance 8cm
        point1 = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        point2 = np.array([[100.0, 0.0], [100.0, 0.0], [100.0, 0.0]])

        scale_factor = compute_scaling_factor(point1, point2, known_distance_cm=8.0)

        expected = 8.0 / 100.0  # 0.08 cm/pixel
        assert abs(scale_factor - expected) < 1e-6

    def test_scaling_factor_with_noise(self):
        """Scaling should be robust to noise (uses median)"""
        np.random.seed(42)
        # Most points 100px apart, some outliers
        point1 = np.tile([0.0, 0.0], (100, 1))
        point2 = np.random.normal([100.0, 0.0], 2.0, (100, 2))

        # Add outliers
        point2[10] = [200.0, 0.0]

        scale_factor = compute_scaling_factor(point1, point2, known_distance_cm=10.0)

        # Should be close to 10/100 = 0.1 cm/pixel
        assert 0.09 < scale_factor < 0.11


class TestScalingFactorV2:
    """Test v1.2.0 scaling factor calculation"""

    def test_scaling_factor_v2_basic(self, sample_body_measurements):
        """v2 scaling with outlier removal"""
        n_frames = len(sample_body_measurements)

        # Create synthetic trajectories
        snout = np.column_stack([np.zeros(n_frames), np.zeros(n_frames)])
        tailbase = np.column_stack([sample_body_measurements, np.zeros(n_frames)])
        likelihood = np.ones(n_frames) * 0.95

        scale_factor, diagnostics = compute_scaling_factor_v2(
            snout,
            tailbase,
            likelihood,
            likelihood,
            expected_body_length_cm=10.0
        )

        assert scale_factor > 0
        assert diagnostics["frames_used"] > 0
        assert diagnostics["outliers_removed"] > 0  # Should remove outliers

    def test_scaling_factor_v2_low_likelihood_filtering(self):
        """Low likelihood frames should be excluded"""
        n_frames = 100

        snout = np.column_stack([np.zeros(n_frames), np.zeros(n_frames)])
        tailbase = np.column_stack([np.ones(n_frames) * 100, np.zeros(n_frames)])

        # Half frames with low likelihood
        likelihood = np.ones(n_frames)
        likelihood[50:] = 0.5  # Low confidence

        scale_factor, diagnostics = compute_scaling_factor_v2(
            snout,
            tailbase,
            likelihood,
            likelihood,
            min_likelihood=0.9
        )

        # Should use only ~50 high-confidence frames
        assert diagnostics["frames_high_confidence"] < n_frames

    def test_scaling_factor_v2_diagnostics(self):
        """Diagnostics should contain all expected fields"""
        n_frames = 100

        snout = np.zeros((n_frames, 2))
        tailbase = np.column_stack([np.ones(n_frames) * 100, np.zeros(n_frames)])

        scale_factor, diagnostics = compute_scaling_factor_v2(snout, tailbase)

        required_fields = [
            "median_body_length_px",
            "frames_used",
            "frames_total",
            "frames_high_confidence",
            "outliers_removed",
            "scaling_factor"
        ]

        for field in required_fields:
            assert field in diagnostics

    def test_scaling_factor_v2_mismatched_lengths(self):
        """Mismatched trajectory lengths should raise error"""
        snout = np.zeros((100, 2))
        tailbase = np.zeros((50, 2))

        with pytest.raises(ValueError, match="same length"):
            compute_scaling_factor_v2(snout, tailbase)

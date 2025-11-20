"""Geometric calculations for biomechanical analysis"""
import numpy as np
from typing import Tuple


def compute_distance_2d(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean distance between two 2D points.

    Args:
        p1: Array of shape (N, 2) or (2,) representing (x, y) coordinates
        p2: Array of shape (N, 2) or (2,) representing (x, y) coordinates

    Returns:
        Distance(s) between points
    """
    if p1.ndim == 1:
        p1 = p1.reshape(1, -1)
    if p2.ndim == 1:
        p2 = p2.reshape(1, -1)

    diff = p1 - p2
    distance = np.sqrt(np.sum(diff ** 2, axis=1))
    return distance if len(distance) > 1 else distance[0]


def compute_angle_3points(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """
    Compute angle at p2 formed by three points (p1-p2-p3) in degrees.

    Args:
        p1: Array of shape (N, 2) representing first point
        p2: Array of shape (N, 2) representing vertex point
        p3: Array of shape (N, 2) representing third point

    Returns:
        Angle(s) in degrees (0-180)
    """
    if p1.ndim == 1:
        p1 = p1.reshape(1, -1)
        p2 = p2.reshape(1, -1)
        p3 = p3.reshape(1, -1)

    v1 = p1 - p2
    v2 = p3 - p2

    cos_angle = np.sum(v1 * v2, axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-10)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))

    return angle if len(angle) > 1 else angle[0]


def compute_center_of_mass(points: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
    """
    Compute center of mass from multiple points.

    Args:
        points: Array of shape (N, 2) representing (x, y) coordinates
        weights: Optional weights for each point

    Returns:
        Center of mass coordinates (x, y)
    """
    if weights is None:
        weights = np.ones(len(points))

    weights = weights / np.sum(weights)
    com = np.average(points, axis=0, weights=weights)
    return com


def compute_trajectory_length(trajectory: np.ndarray) -> float:
    """
    Compute total length of a trajectory.

    Args:
        trajectory: Array of shape (N, 2) representing (x, y) coordinates

    Returns:
        Total trajectory length
    """
    if len(trajectory) < 2:
        return 0.0

    diffs = np.diff(trajectory, axis=0)
    distances = np.sqrt(np.sum(diffs ** 2, axis=1))
    return np.sum(distances)


def compute_trajectory_speed(trajectory: np.ndarray, fps: float = 120.0) -> np.ndarray:
    """
    Compute instantaneous speed along trajectory.

    Args:
        trajectory: Array of shape (N, 2) representing (x, y) coordinates
        fps: Frame rate (Hz)

    Returns:
        Speed array of shape (N,) in units/second
    """
    if len(trajectory) < 2:
        return np.zeros(len(trajectory))

    diffs = np.diff(trajectory, axis=0, prepend=trajectory[:1])
    distances = np.sqrt(np.sum(diffs ** 2, axis=1))
    speeds = distances * fps

    return speeds


def compute_stride_length(foot_strikes: np.ndarray, trajectory: np.ndarray) -> np.ndarray:
    """
    Compute stride lengths from foot strike indices.

    Args:
        foot_strikes: Array of frame indices where foot strikes occur
        trajectory: Array of shape (N, 2) representing paw trajectory

    Returns:
        Array of stride lengths (one per stride)
    """
    if len(foot_strikes) < 2:
        return np.array([])

    stride_lengths = []
    for i in range(len(foot_strikes) - 1):
        start_idx = foot_strikes[i]
        end_idx = foot_strikes[i + 1]

        if end_idx < len(trajectory):
            stride_traj = trajectory[start_idx:end_idx + 1]
            stride_length = compute_trajectory_length(stride_traj)
            stride_lengths.append(stride_length)

    return np.array(stride_lengths)


def compute_lateral_deviation(trajectory: np.ndarray, axis: int = 0) -> Tuple[float, float]:
    """
    Compute lateral deviation (sway) along trajectory.

    Args:
        trajectory: Array of shape (N, 2) representing (x, y) coordinates
        axis: 0 for x-axis (mediolateral), 1 for y-axis (anteroposterior)

    Returns:
        Tuple of (mean_deviation, std_deviation)
    """
    deviations = trajectory[:, axis]
    # Use nanmean/nanstd to handle NaN values (v1.2.0 fix)
    mean_dev = np.nanmean(deviations)
    std_dev = np.nanstd(deviations)

    return mean_dev, std_dev


def compute_symmetry_index(left_values: np.ndarray, right_values: np.ndarray) -> float:
    """
    Compute symmetry index between left and right measurements.

    Args:
        left_values: Measurements for left side
        right_values: Measurements for right side

    Returns:
        Symmetry index (0 = perfect symmetry, higher = more asymmetric)
    """
    if len(left_values) == 0 or len(right_values) == 0:
        return np.nan

    left_mean = np.nanmean(left_values)
    right_mean = np.nanmean(right_values)

    if left_mean + right_mean == 0:
        return 0.0

    symmetry = abs(left_mean - right_mean) / (0.5 * (left_mean + right_mean)) * 100

    return symmetry


def compute_range_of_motion(angles: np.ndarray) -> float:
    """
    Compute range of motion (ROM) from angle measurements.

    Args:
        angles: Array of joint angles in degrees

    Returns:
        ROM = max(angle) - min(angle)
    """
    valid_angles = angles[~np.isnan(angles)]

    if len(valid_angles) == 0:
        return np.nan

    return np.max(valid_angles) - np.min(valid_angles)


def pixels_to_cm(pixel_values: np.ndarray, scale_factor: float) -> np.ndarray:
    """
    Convert pixel values to centimeters using scale factor.

    Args:
        pixel_values: Values in pixels
        scale_factor: Conversion factor (cm/pixel)

    Returns:
        Values in centimeters
    """
    return pixel_values * scale_factor


def compute_scaling_factor(point1: np.ndarray, point2: np.ndarray,
                          known_distance_cm: float = 8.0) -> float:
    """
    Compute scaling factor from known distance between two points (LEGACY v1.1).

    Args:
        point1: Array of shape (N, 2) for first point
        point2: Array of shape (N, 2) for second point
        known_distance_cm: Known real-world distance in cm

    Returns:
        Scale factor (cm/pixel)
    """
    pixel_distance = np.median(compute_distance_2d(point1, point2))
    scale_factor = known_distance_cm / pixel_distance
    return scale_factor


def compute_scaling_factor_v2(
    snout_trajectory: np.ndarray,
    tailbase_trajectory: np.ndarray,
    likelihood_snout: np.ndarray = None,
    likelihood_tail: np.ndarray = None,
    expected_body_length_cm: float = 10.0,
    min_likelihood: float = 0.9,
    tolerance: float = 0.25
) -> Tuple[float, dict]:
    """
    Compute scaling factor using full-body measurement (v1.2.0).

    Uses snout→tailbase distance instead of spine1-3 for more accurate
    full-body length estimation. Includes likelihood filtering and outlier
    removal for robust scaling.

    Args:
        snout_trajectory: Array of shape (N, 2) with snout (x, y) coordinates
        tailbase_trajectory: Array of shape (N, 2) with tailbase (x, y) coordinates
        likelihood_snout: Optional array of shape (N,) with snout detection confidence
        likelihood_tail: Optional array of shape (N,) with tail detection confidence
        expected_body_length_cm: Expected mouse body length in cm (default 10.0 for adult)
        min_likelihood: Minimum confidence threshold for frame inclusion (default 0.9)
        tolerance: Allowed variance from median as fraction (default 0.25 = ±25%)

    Returns:
        Tuple of (scaling_factor, diagnostics_dict)

        diagnostics_dict contains:
            - median_body_length_px: Median body length in pixels
            - frames_used: Number of high-confidence frames used
            - frames_total: Total frames available
            - outliers_removed: Number of outlier frames excluded
            - scaling_factor: Final scaling factor (cm/pixel)
    """
    # Validate inputs
    n_frames = len(snout_trajectory)
    if len(tailbase_trajectory) != n_frames:
        raise ValueError("Snout and tailbase trajectories must have same length")

    # Create validity mask
    valid_mask = np.ones(n_frames, dtype=bool)

    # Filter by likelihood if provided
    if likelihood_snout is not None and likelihood_tail is not None:
        valid_mask &= (likelihood_snout >= min_likelihood)
        valid_mask &= (likelihood_tail >= min_likelihood)

    # Filter NaN values
    valid_mask &= ~np.isnan(snout_trajectory).any(axis=1)
    valid_mask &= ~np.isnan(tailbase_trajectory).any(axis=1)

    if np.sum(valid_mask) < 100:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            f"Only {np.sum(valid_mask)} high-confidence frames for scaling "
            f"(threshold: {min_likelihood}). Results may be unreliable."
        )

    # Compute per-frame body length in pixels
    snout_valid = snout_trajectory[valid_mask]
    tail_valid = tailbase_trajectory[valid_mask]
    body_length_px = compute_distance_2d(snout_valid, tail_valid)

    # Remove outliers using MAD-based filtering
    median_length = np.median(body_length_px)
    mad = np.median(np.abs(body_length_px - median_length))

    # Keep frames within tolerance
    lower_bound = median_length * (1 - tolerance)
    upper_bound = median_length * (1 + tolerance)
    inlier_mask = (body_length_px >= lower_bound) & (body_length_px <= upper_bound)

    inliers = body_length_px[inlier_mask]
    outliers_removed = np.sum(~inlier_mask)

    if len(inliers) < 50:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(
            f"Only {len(inliers)} inlier frames after outlier removal. "
            f"Scaling may be unreliable."
        )

    # Compute final scaling factor
    median_body_px = np.median(inliers)
    scaling_factor = expected_body_length_cm / median_body_px

    # Prepare diagnostics
    diagnostics = {
        'median_body_length_px': float(median_body_px),
        'frames_used': len(inliers),
        'frames_total': n_frames,
        'frames_high_confidence': int(np.sum(valid_mask)),
        'outliers_removed': int(outliers_removed),
        'scaling_factor': float(scaling_factor),
        'expected_body_length_cm': expected_body_length_cm,
        'min_likelihood': min_likelihood,
        'tolerance': tolerance
    }

    return scaling_factor, diagnostics

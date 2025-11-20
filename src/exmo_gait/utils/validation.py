"""Validation utilities for data quality control"""
import numpy as np
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation failures"""
    pass


def validate_frame_rate(expected_fps: float, actual_fps: float, tolerance: float = 0.01) -> bool:
    """
    Validate frame rate matches expected value.

    Args:
        expected_fps: Expected frame rate (Hz)
        actual_fps: Actual frame rate (Hz)
        tolerance: Allowed relative tolerance (default 1%)

    Returns:
        True if valid

    Raises:
        ValidationError: If frame rate is outside tolerance
    """
    if abs(actual_fps - expected_fps) / expected_fps > tolerance:
        raise ValidationError(
            f"Frame rate mismatch: expected {expected_fps} Hz, got {actual_fps} Hz"
        )
    return True


def validate_scaling_factor(scale_factor: float,
                            expected_cm: float = 8.0,
                            tolerance: float = 0.20) -> bool:
    """
    Validate scaling factor is within reasonable range.

    Args:
        scale_factor: Computed scaling factor (cm/pixel)
        expected_cm: Expected body length in cm
        tolerance: Allowed relative tolerance (default 20%)

    Returns:
        True if valid

    Raises:
        ValidationError: If scaling factor is outside tolerance
    """
    if scale_factor <= 0:
        raise ValidationError("Scaling factor must be positive")

    min_valid = expected_cm * (1 - tolerance)
    max_valid = expected_cm * (1 + tolerance)

    if not (min_valid <= (scale_factor * 100) <= max_valid):
        logger.warning(
            f"Scaling factor may be incorrect: {scale_factor:.4f} cm/pixel. "
            f"Expected body length: {expected_cm}±{tolerance*100}% cm"
        )

    return True


def validate_keypoints_present(data: Dict, required_keypoints: List[str]) -> bool:
    """
    Validate that all required keypoints are present in data.

    Args:
        data: Dictionary of keypoint data
        required_keypoints: List of required keypoint names

    Returns:
        True if valid

    Raises:
        ValidationError: If required keypoints are missing
    """
    missing_keypoints = [kp for kp in required_keypoints if kp not in data]

    if missing_keypoints:
        raise ValidationError(
            f"Missing required keypoints: {', '.join(missing_keypoints)}"
        )

    return True


def validate_data_completeness(data: np.ndarray, min_valid_ratio: float = 0.7) -> bool:
    """
    Validate that data has sufficient non-NaN values.

    Args:
        data: Data array
        min_valid_ratio: Minimum ratio of valid (non-NaN) values

    Returns:
        True if valid

    Raises:
        ValidationError: If data has too many missing values
    """
    valid_ratio = np.sum(~np.isnan(data)) / len(data)

    if valid_ratio < min_valid_ratio:
        raise ValidationError(
            f"Insufficient valid data: {valid_ratio*100:.1f}% valid, "
            f"minimum required: {min_valid_ratio*100:.1f}%"
        )

    return True


def validate_sufficient_strides(num_strides: int, min_strides: int = 3) -> bool:
    """
    Validate sufficient number of strides for analysis.

    Args:
        num_strides: Number of detected strides
        min_strides: Minimum required strides

    Returns:
        True if valid

    Raises:
        ValidationError: If insufficient strides
    """
    if num_strides < min_strides:
        raise ValidationError(
            f"Insufficient strides for analysis: {num_strides} detected, "
            f"minimum required: {min_strides}"
        )

    return True


def validate_time_windows(windows: List[Tuple[int, int]], max_frames: int) -> bool:
    """
    Validate time windows are within valid range.

    Args:
        windows: List of (start, end) frame tuples
        max_frames: Maximum valid frame index

    Returns:
        True if valid

    Raises:
        ValidationError: If windows are invalid
    """
    for start, end in windows:
        if start < 0 or end >= max_frames or start >= end:
            raise ValidationError(
                f"Invalid time window: ({start}, {end}), max frames: {max_frames}"
            )

    return True


def check_data_quality(data: np.ndarray) -> Dict[str, float]:
    """
    Compute data quality metrics.

    Args:
        data: Data array

    Returns:
        Dictionary of quality metrics
    """
    total = len(data)
    valid = np.sum(~np.isnan(data))
    missing = total - valid

    quality_metrics = {
        'total_frames': total,
        'valid_frames': valid,
        'missing_frames': missing,
        'valid_ratio': valid / total if total > 0 else 0,
        'missing_ratio': missing / total if total > 0 else 0,
    }

    return quality_metrics


def log_validation_warnings(issues: List[str]) -> None:
    """
    Log validation warnings for non-critical issues.

    Args:
        issues: List of warning messages
    """
    if issues:
        logger.warning("Data quality warnings:")
        for issue in issues:
            logger.warning(f"  - {issue}")

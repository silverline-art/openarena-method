"""Signal processing utilities for gait analysis"""
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from typing import Tuple, Optional


def apply_savgol_filter(data: np.ndarray, window_length: int = 11, polyorder: int = 3) -> np.ndarray:
    """
    Apply Savitzky-Golay filter for smoothing trajectories.

    Args:
        data: 1D array of data points
        window_length: Length of the filter window (must be odd)
        polyorder: Order of the polynomial fit

    Returns:
        Smoothed data array
    """
    if len(data) < window_length:
        window_length = len(data) if len(data) % 2 == 1 else len(data) - 1
        if window_length < polyorder + 2:
            return data

    return signal.savgol_filter(data, window_length, polyorder, mode='nearest')


def interpolate_missing_values(data: np.ndarray, max_gap: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate missing values (NaN) in trajectory data.

    Args:
        data: 1D array with potential NaN values
        max_gap: Maximum gap size to interpolate (frames)

    Returns:
        Tuple of (interpolated_data, mask_of_valid_values)
    """
    mask = ~np.isnan(data)
    indices = np.arange(len(data))

    if not mask.any():
        return data, mask

    valid_indices = indices[mask]
    valid_data = data[mask]

    if len(valid_indices) < 2:
        return data, mask

    interpolator = interp1d(valid_indices, valid_data, kind='linear',
                           bounds_error=False, fill_value=np.nan)
    interpolated = interpolator(indices)

    gaps = np.diff(np.concatenate([[0], valid_indices, [len(data)]]))
    large_gaps = gaps > max_gap

    result = data.copy()
    for i in range(len(valid_indices) - 1):
        if gaps[i + 1] <= max_gap:
            start = valid_indices[i]
            end = valid_indices[i + 1]
            result[start:end + 1] = interpolated[start:end + 1]

    return result, ~np.isnan(result)


def compute_mad(data: np.ndarray, scale: float = 1.4826) -> float:
    """
    Compute Median Absolute Deviation (MAD).

    Args:
        data: 1D array of data points
        scale: Scale factor for normal distribution consistency (default 1.4826)

    Returns:
        MAD value
    """
    median = np.nanmedian(data)
    mad = np.nanmedian(np.abs(data - median))
    return mad * scale


def detect_outliers_mad(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """
    Detect outliers using MAD-based thresholding.

    Args:
        data: 1D array of data points
        threshold: Number of MAD units for outlier detection

    Returns:
        Boolean mask where True indicates outlier
    """
    median = np.nanmedian(data)
    mad = compute_mad(data)

    if mad == 0:
        return np.zeros(len(data), dtype=bool)

    z_score = np.abs(data - median) / mad
    return z_score > threshold


def filter_outliers_mad(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """
    Filter outliers using MAD-based thresholding by replacing with NaN.

    Args:
        data: 1D array of data points
        threshold: Number of MAD units for outlier detection

    Returns:
        Array with outliers replaced by NaN
    """
    result = data.copy()
    outliers = detect_outliers_mad(data, threshold)
    result[outliers] = np.nan
    return result


def compute_velocity(positions: np.ndarray, fps: float = 120.0) -> np.ndarray:
    """
    Compute velocity from position data.

    Args:
        positions: 1D array of positions (cm)
        fps: Frame rate (Hz)

    Returns:
        Velocity array (cm/s)
    """
    dt = 1.0 / fps
    velocity = np.gradient(positions) / dt
    return velocity


def compute_angular_velocity(angles: np.ndarray, fps: float = 120.0) -> np.ndarray:
    """
    Compute angular velocity from angle data.

    Args:
        angles: 1D array of angles (degrees)
        fps: Frame rate (Hz)

    Returns:
        Angular velocity array (degrees/s)
    """
    dt = 1.0 / fps
    angular_velocity = np.gradient(angles) / dt
    return angular_velocity


def detect_peaks_adaptive(signal_data: np.ndarray,
                          min_prominence: Optional[float] = None,
                          min_distance: Optional[int] = None) -> np.ndarray:
    """
    Detect peaks in signal using adaptive thresholding.

    Args:
        signal_data: 1D signal array
        min_prominence: Minimum peak prominence (if None, uses MAD-based)
        min_distance: Minimum distance between peaks in samples

    Returns:
        Array of peak indices
    """
    if min_prominence is None:
        min_prominence = compute_mad(signal_data) * 0.5

    peaks, _ = signal.find_peaks(signal_data,
                                  prominence=min_prominence,
                                  distance=min_distance)
    return peaks


def smooth_binary_classification(binary_signal: np.ndarray,
                                 window_size: int = 30) -> np.ndarray:
    """
    Smooth binary classification using moving average.

    Args:
        binary_signal: Binary array (0/1 or False/True)
        window_size: Size of smoothing window in samples

    Returns:
        Smoothed binary signal
    """
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(binary_signal.astype(float), kernel, mode='same')
    return smoothed > 0.5


def smooth_velocity_ema(positions: np.ndarray, alpha: float = 0.35, fps: float = 120.0) -> np.ndarray:
    """
    Compute velocity using Exponential Moving Average (EMA) smoothing (v1.2.0).

    EMA provides less dampening than Savitzky-Golay, preserving dynamic peaks
    while still reducing noise. Suitable for velocity calculations where peak
    preservation is critical.

    Args:
        positions: 1D array of positions (any unit)
        alpha: EMA smoothing factor (0-1). Higher = less smoothing.
               Default 0.35 balances noise reduction with peak preservation.
        fps: Frame rate (Hz)

    Returns:
        Smoothed velocity array (units/s)
    """
    # Compute raw velocity
    dt = 1.0 / fps
    raw_velocity = np.gradient(positions) / dt

    # Apply EMA smoothing
    smoothed_velocity = np.zeros_like(raw_velocity)
    smoothed_velocity[0] = raw_velocity[0]

    for i in range(1, len(raw_velocity)):
        if np.isnan(raw_velocity[i]):
            smoothed_velocity[i] = smoothed_velocity[i-1]
        else:
            smoothed_velocity[i] = alpha * raw_velocity[i] + (1 - alpha) * smoothed_velocity[i-1]

    return smoothed_velocity


def smooth_trajectory_adaptive(
    trajectory: np.ndarray,
    data_completeness: float,
    window_size_base: int = 7,
    polyorder: int = 3
) -> np.ndarray:
    """
    Apply adaptive smoothing based on data quality (v1.2.0).

    Adjusts smoothing window based on data completeness:
    - High quality (>0.9): Reduce window by 2 frames (more responsive)
    - Medium quality (0.7-0.9): Use base window
    - Low quality (<0.7): Increase window by 2 frames (more robust)

    Args:
        trajectory: 1D array of trajectory data
        data_completeness: Fraction of valid data points (0-1)
        window_size_base: Base window size for Savitzky-Golay filter
        polyorder: Polynomial order for filter

    Returns:
        Smoothed trajectory array
    """
    # Adjust window size based on data quality
    if data_completeness >= 0.9:
        # High quality: less smoothing needed
        window_size = max(polyorder + 2, window_size_base - 2)
    elif data_completeness >= 0.7:
        # Medium quality: use base window
        window_size = window_size_base
    else:
        # Low quality: more smoothing for robustness
        window_size = window_size_base + 2

    # Ensure window is odd and valid
    if window_size % 2 == 0:
        window_size += 1

    # Apply Savitzky-Golay with adaptive window
    return apply_savgol_filter(trajectory, window_size, polyorder)

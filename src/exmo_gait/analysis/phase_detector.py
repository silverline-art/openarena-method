"""Detection of stationary and walking phases"""
import numpy as np
import logging
from typing import List, Tuple
from ..utils.signal_processing import (
    compute_mad,
    smooth_binary_classification,
    compute_velocity
)
from ..utils.geometry import compute_trajectory_speed
from ..constants import (
    FPS_DEFAULT,
    MAD_THRESHOLD_STATIONARY_DEFAULT,
    MAD_THRESHOLD_WALKING_DEFAULT,
    MIN_WALKING_DURATION_SEC,
    MIN_STATIONARY_DURATION_SEC,
    ADAPTIVE_PERCENTILE_DEFAULT,
    MIN_THRESHOLD_PX_PER_FRAME
)

logger = logging.getLogger(__name__)


class PhaseDetector:
    """Detect stationary and walking phases from CoM trajectory"""

    def __init__(self,
                 fps: float = FPS_DEFAULT,
                 stationary_mad_threshold: float = MAD_THRESHOLD_STATIONARY_DEFAULT,
                 walking_mad_threshold: float = MAD_THRESHOLD_WALKING_DEFAULT,
                 min_walking_duration: float = MIN_WALKING_DURATION_SEC,
                 min_stationary_duration: float = MIN_STATIONARY_DURATION_SEC,
                 smoothing_window_ms: float = 250.0,
                 merge_gap_ms: float = 100.0,
                 use_hybrid_threshold: bool = False,
                 adaptive_percentile: float = ADAPTIVE_PERCENTILE_DEFAULT,
                 min_threshold_px_per_frame: float = MIN_THRESHOLD_PX_PER_FRAME):
        """
        Initialize phase detector.

        Args:
            fps: Frame rate (Hz)
            stationary_mad_threshold: MAD multiplier for stationary detection
            walking_mad_threshold: MAD multiplier for walking detection
            min_walking_duration: Minimum walking duration (seconds)
            min_stationary_duration: Minimum stationary duration (seconds)
            smoothing_window_ms: Smoothing window size (milliseconds)
            merge_gap_ms: Gap size for merging adjacent windows (milliseconds)
            use_hybrid_threshold: Use hybrid MAD + percentile method (v1.2.0)
            adaptive_percentile: Percentile for adaptive threshold (v1.2.0)
            min_threshold_px_per_frame: Safety lower bound for threshold (v1.2.0)
        """
        self.fps = fps
        self.stationary_mad_threshold = stationary_mad_threshold
        self.walking_mad_threshold = walking_mad_threshold
        self.min_walking_duration = min_walking_duration
        self.min_stationary_duration = min_stationary_duration
        self.smoothing_window_frames = int(smoothing_window_ms / 1000 * fps)
        self.merge_gap_frames = int(merge_gap_ms / 1000 * fps)
        self.use_hybrid_threshold = use_hybrid_threshold
        self.adaptive_percentile = adaptive_percentile
        self.min_threshold_px_per_frame = min_threshold_px_per_frame

    def compute_com_speed(self, com_trajectory: np.ndarray) -> np.ndarray:
        """
        Compute center of mass speed.

        Args:
            com_trajectory: CoM trajectory of shape (N, 2)

        Returns:
            Speed array (cm/s)
        """
        speeds = compute_trajectory_speed(com_trajectory, self.fps)
        return speeds

    def detect_stationary_phase(self, com_speed: np.ndarray) -> np.ndarray:
        """
        Detect stationary phase using MAD-based thresholding.

        Args:
            com_speed: CoM speed array (cm/s)

        Returns:
            Boolean array where True indicates stationary
        """
        mad = compute_mad(com_speed)
        threshold = mad * self.stationary_mad_threshold

        stationary = com_speed < threshold

        if self.smoothing_window_frames > 1:
            stationary = smooth_binary_classification(stationary, self.smoothing_window_frames)

        logger.info(
            f"Detected stationary phase: threshold={threshold:.2f} cm/s, "
            f"{np.sum(stationary)/len(stationary)*100:.1f}% of frames"
        )

        return stationary

    def compute_hybrid_threshold(self, com_speed: np.ndarray, mad_multiplier: float) -> float:
        """
        Compute threshold using hybrid MAD + percentile method (v1.2.0).

        Combines MAD-based robust statistic with percentile-based adaptive
        threshold, then applies safety lower bound.

        Args:
            com_speed: CoM speed array (cm/s)
            mad_multiplier: Multiplier for MAD component

        Returns:
            Hybrid threshold value
        """
        # MAD component
        mad = compute_mad(com_speed)
        mad_threshold = mad * mad_multiplier

        # Percentile component
        percentile_threshold = np.percentile(com_speed, self.adaptive_percentile)

        # Hybrid: average of both methods
        hybrid_threshold = (mad_threshold + percentile_threshold) / 2

        # Apply safety lower bound
        final_threshold = max(hybrid_threshold, self.min_threshold_px_per_frame)

        logger.debug(
            f"[v1.2.0 Hybrid] MAD: {mad_threshold:.3f}, "
            f"Percentile: {percentile_threshold:.3f}, "
            f"Hybrid: {hybrid_threshold:.3f}, "
            f"Final: {final_threshold:.3f} cm/s"
        )

        return final_threshold

    def detect_walking_phase(self, com_speed: np.ndarray) -> np.ndarray:
        """
        Detect walking phase using dynamic MAD-based thresholding.

        Args:
            com_speed: CoM speed array (cm/s)

        Returns:
            Boolean array where True indicates walking
        """
        if self.use_hybrid_threshold:
            # v1.2.0: Hybrid MAD + percentile threshold
            threshold = self.compute_hybrid_threshold(com_speed, self.walking_mad_threshold)
        else:
            # v1.1.0: Pure MAD threshold
            mad = compute_mad(com_speed)
            threshold = mad * self.walking_mad_threshold

        walking = com_speed > threshold

        if self.smoothing_window_frames > 1:
            walking = smooth_binary_classification(walking, self.smoothing_window_frames)

        logger.info(
            f"Detected walking phase: threshold={threshold:.2f} cm/s "
            f"({'hybrid' if self.use_hybrid_threshold else 'MAD-only'}), "
            f"{np.sum(walking)/len(walking)*100:.1f}% of frames"
        )

        return walking

    def extract_time_windows(self,
                            binary_phase: np.ndarray,
                            min_duration_sec: float) -> List[Tuple[int, int]]:
        """
        Extract continuous time windows from binary phase classification.

        Args:
            binary_phase: Boolean array indicating phase
            min_duration_sec: Minimum window duration (seconds)

        Returns:
            List of (start_frame, end_frame) tuples
        """
        min_duration_frames = int(min_duration_sec * self.fps)

        windows = []
        in_window = False
        start_frame = 0

        for i in range(len(binary_phase)):
            if binary_phase[i] and not in_window:
                start_frame = i
                in_window = True
            elif not binary_phase[i] and in_window:
                duration = i - start_frame
                if duration >= min_duration_frames:
                    windows.append((start_frame, i - 1))
                in_window = False

        if in_window:
            duration = len(binary_phase) - start_frame
            if duration >= min_duration_frames:
                windows.append((start_frame, len(binary_phase) - 1))

        return windows

    def merge_close_windows(self,
                           windows: List[Tuple[int, int]],
                           max_gap_frames: int = None) -> List[Tuple[int, int]]:
        """
        Merge windows that are close together.

        Args:
            windows: List of (start, end) tuples
            max_gap_frames: Maximum gap to merge (if None, uses self.merge_gap_frames)

        Returns:
            List of merged windows
        """
        if not windows:
            return windows

        if max_gap_frames is None:
            max_gap_frames = self.merge_gap_frames

        merged = [windows[0]]

        for current_start, current_end in windows[1:]:
            last_start, last_end = merged[-1]

            gap = current_start - last_end

            if gap <= max_gap_frames:
                merged[-1] = (last_start, current_end)
            else:
                merged.append((current_start, current_end))

        return merged

    def detect_stationary_windows(self, com_trajectory: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect stationary time windows.

        Args:
            com_trajectory: CoM trajectory of shape (N, 2)

        Returns:
            List of (start_frame, end_frame) tuples for stationary periods
        """
        com_speed = self.compute_com_speed(com_trajectory)
        stationary_phase = self.detect_stationary_phase(com_speed)

        windows = self.extract_time_windows(
            stationary_phase,
            self.min_stationary_duration
        )

        windows = self.merge_close_windows(windows)

        logger.info(f"Detected {len(windows)} stationary windows")

        return windows

    def detect_walking_windows(self, com_trajectory: np.ndarray) -> List[Tuple[int, int]]:
        """
        Detect walking time windows.

        Args:
            com_trajectory: CoM trajectory of shape (N, 2)

        Returns:
            List of (start_frame, end_frame) tuples for walking periods
        """
        com_speed = self.compute_com_speed(com_trajectory)
        walking_phase = self.detect_walking_phase(com_speed)

        windows = self.extract_time_windows(
            walking_phase,
            self.min_walking_duration
        )

        windows = self.merge_close_windows(windows)

        total_walking_frames = sum(end - start + 1 for start, end in windows)
        total_walking_duration = total_walking_frames / self.fps

        logger.info(
            f"Detected {len(windows)} walking windows, "
            f"total duration: {total_walking_duration:.2f} sec"
        )

        return windows

    def get_window_durations(self, windows: List[Tuple[int, int]]) -> np.ndarray:
        """
        Compute duration of each window in seconds.

        Args:
            windows: List of (start, end) frame tuples

        Returns:
            Array of durations in seconds
        """
        durations = np.array([(end - start + 1) / self.fps for start, end in windows])
        return durations

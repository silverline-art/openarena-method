"""Step detection and foot strike event extraction"""
import numpy as np
import logging
from typing import List, Tuple, Dict
from scipy import signal
from ..utils.signal_processing import compute_mad, detect_peaks_adaptive
from ..utils.validation import validate_sufficient_strides

logger = logging.getLogger(__name__)


class StepDetector:
    """Detect foot strikes and step events from paw trajectories"""

    def __init__(self,
                 fps: float = 120.0,
                 min_stride_duration: float = 0.1,
                 max_stride_duration: float = 1.0,
                 prominence_multiplier: float = 0.5,
                 allow_micro_steps: bool = False,
                 micro_step_threshold_cm: float = 1.0):
        """
        Initialize step detector.

        Args:
            fps: Frame rate (Hz)
            min_stride_duration: Minimum stride duration (seconds)
            max_stride_duration: Maximum stride duration (seconds)
            prominence_multiplier: Multiplier for MAD-based peak prominence
            allow_micro_steps: Include strides < micro_step_threshold (v1.2.0)
            micro_step_threshold_cm: Threshold for micro-step labeling (v1.2.0)
        """
        self.fps = fps
        self.min_stride_frames = int(min_stride_duration * fps)
        self.max_stride_frames = int(max_stride_duration * fps)
        self.prominence_multiplier = prominence_multiplier
        self.allow_micro_steps = allow_micro_steps
        self.micro_step_threshold_cm = micro_step_threshold_cm

    def detect_foot_strikes_vertical(self,
                                    paw_trajectory: np.ndarray,
                                    use_y: bool = True) -> np.ndarray:
        """
        Detect foot strikes using vertical position oscillations.

        Args:
            paw_trajectory: Array of shape (N, 2) with (x, y) coordinates
            use_y: If True, use y-coordinate for detection (typical for side/bottom views)

        Returns:
            Array of frame indices where foot strikes occur
        """
        coord_idx = 1 if use_y else 0
        vertical_pos = paw_trajectory[:, coord_idx]

        valid_mask = ~np.isnan(vertical_pos)
        if np.sum(valid_mask) < 100:
            logger.warning("Insufficient valid data for foot strike detection")
            return np.array([])

        inverted_signal = -vertical_pos.copy()
        inverted_signal[~valid_mask] = np.nanmin(inverted_signal)

        mad = compute_mad(vertical_pos[valid_mask])
        min_prominence = mad * self.prominence_multiplier

        peaks = detect_peaks_adaptive(
            inverted_signal,
            min_prominence=min_prominence,
            min_distance=self.min_stride_frames
        )

        peaks = peaks[peaks < len(vertical_pos)]

        valid_peaks = peaks[valid_mask[peaks]]

        return valid_peaks

    def detect_foot_strikes_velocity(self, paw_trajectory: np.ndarray) -> np.ndarray:
        """
        Detect foot strikes using velocity-based method.

        Args:
            paw_trajectory: Array of shape (N, 2) with (x, y) coordinates

        Returns:
            Array of frame indices where foot strikes occur
        """
        speed = np.sqrt(np.sum(np.diff(paw_trajectory, axis=0, prepend=paw_trajectory[:1]) ** 2, axis=1))

        valid_mask = ~np.isnan(speed)
        if np.sum(valid_mask) < 100:
            logger.warning("Insufficient valid data for velocity-based detection")
            return np.array([])

        inverted_speed = -speed.copy()
        inverted_speed[~valid_mask] = np.nanmin(inverted_speed)

        mad = compute_mad(speed[valid_mask])
        min_prominence = mad * self.prominence_multiplier

        peaks = detect_peaks_adaptive(
            inverted_speed,
            min_prominence=min_prominence,
            min_distance=self.min_stride_frames
        )

        valid_peaks = peaks[valid_mask[peaks]]

        return valid_peaks

    def detect_swing_stance_phases(self,
                                  paw_trajectory: np.ndarray,
                                  foot_strikes: np.ndarray) -> Dict[str, List[Tuple[int, int]]]:
        """
        Detect swing and stance phases for a paw.

        Args:
            paw_trajectory: Array of shape (N, 2) with (x, y) coordinates
            foot_strikes: Array of foot strike frame indices

        Returns:
            Dictionary with 'swing' and 'stance' lists of (start, end) tuples
        """
        if len(foot_strikes) < 2:
            return {'swing': [], 'stance': []}

        vertical_pos = paw_trajectory[:, 1]
        speed = np.sqrt(np.sum(np.diff(paw_trajectory, axis=0, prepend=paw_trajectory[:1]) ** 2, axis=1))

        speed_threshold = np.nanmedian(speed) + compute_mad(speed[~np.isnan(speed)]) * 0.5

        swing_phases = []
        stance_phases = []

        for i in range(len(foot_strikes) - 1):
            stride_start = foot_strikes[i]
            stride_end = foot_strikes[i + 1]

            stride_speed = speed[stride_start:stride_end]
            stride_vertical = vertical_pos[stride_start:stride_end]

            if len(stride_speed) < 10:
                continue

            high_speed = stride_speed > speed_threshold

            swing_start = None
            for j in range(len(high_speed)):
                if high_speed[j] and swing_start is None:
                    swing_start = stride_start + j
                elif not high_speed[j] and swing_start is not None:
                    swing_end = stride_start + j - 1
                    swing_phases.append((swing_start, swing_end))

                    if len(stance_phases) == 0 or stance_phases[-1][1] < swing_start - 1:
                        stance_start = stance_phases[-1][1] + 1 if stance_phases else stride_start
                        stance_phases.append((stance_start, swing_start - 1))

                    swing_start = None

            if swing_start is not None:
                swing_phases.append((swing_start, stride_end))

            if len(stance_phases) == 0 or stance_phases[-1][1] < stride_end - 1:
                stance_start = stance_phases[-1][1] + 1 if stance_phases else stride_start
                stance_phases.append((stance_start, stride_end))

        return {'swing': swing_phases, 'stance': stance_phases}

    def compute_stride_times(self, foot_strikes: np.ndarray) -> np.ndarray:
        """
        Compute stride times from foot strike indices.

        Args:
            foot_strikes: Array of foot strike frame indices

        Returns:
            Array of stride times in seconds
        """
        if len(foot_strikes) < 2:
            return np.array([])

        stride_frames = np.diff(foot_strikes)
        stride_times = stride_frames / self.fps

        valid_strides = (stride_times >= self.min_stride_frames / self.fps) & \
                       (stride_times <= self.max_stride_frames / self.fps)

        return stride_times[valid_strides]

    def compute_stride_info_v2(self,
                              foot_strikes: np.ndarray,
                              paw_trajectory: np.ndarray,
                              scale_factor: float = 1.0) -> List[Dict]:
        """
        Compute stride information with micro-step labeling (v1.2.0).

        Args:
            foot_strikes: Array of foot strike frame indices
            paw_trajectory: Array of shape (N, 2) with paw trajectory
            scale_factor: Scaling factor for cm conversion (default 1.0 if already in cm)

        Returns:
            List of stride info dictionaries with keys:
                - start_frame: Start frame index
                - end_frame: End frame index
                - duration_sec: Stride duration in seconds
                - length_cm: Stride length in cm
                - is_micro_step: Boolean flag for micro-steps
        """
        if len(foot_strikes) < 2:
            return []

        stride_info_list = []

        for i in range(len(foot_strikes) - 1):
            start_frame = foot_strikes[i]
            end_frame = foot_strikes[i + 1]

            # Compute duration
            duration_sec = (end_frame - start_frame) / self.fps

            # Compute stride length
            start_pos = paw_trajectory[start_frame]
            end_pos = paw_trajectory[end_frame]

            if np.isnan(start_pos).any() or np.isnan(end_pos).any():
                stride_length_cm = np.nan
            else:
                stride_length_px = np.linalg.norm(end_pos - start_pos)
                stride_length_cm = stride_length_px * scale_factor

            # Check if micro-step
            is_micro = False
            if self.allow_micro_steps and not np.isnan(stride_length_cm):
                is_micro = stride_length_cm < self.micro_step_threshold_cm

            # Create stride info
            stride_info = {
                'start_frame': int(start_frame),
                'end_frame': int(end_frame),
                'duration_sec': float(duration_sec),
                'length_cm': float(stride_length_cm) if not np.isnan(stride_length_cm) else None,
                'is_micro_step': bool(is_micro)
            }

            stride_info_list.append(stride_info)

        return stride_info_list

    def compute_cadence(self, foot_strikes: np.ndarray, total_duration: float) -> float:
        """
        Compute cadence (steps per minute).

        Args:
            foot_strikes: Array of foot strike frame indices
            total_duration: Total duration in seconds

        Returns:
            Cadence in steps/minute
        """
        if len(foot_strikes) < 2 or total_duration == 0:
            return np.nan

        num_steps = len(foot_strikes) - 1
        cadence = (num_steps / total_duration) * 60

        return cadence

    def detect_all_limbs(self,
                        paw_trajectories: Dict[str, np.ndarray],
                        walking_windows: List[Tuple[int, int]]) -> Dict[str, Dict]:
        """
        Detect foot strikes for all limbs during walking windows.

        Args:
            paw_trajectories: Dictionary mapping limb names to trajectories
            walking_windows: List of (start, end) walking window tuples

        Returns:
            Dictionary mapping limb names to detection results
        """
        results = {}

        for limb_name, trajectory in paw_trajectories.items():
            logger.info(f"Detecting steps for {limb_name}")

            all_foot_strikes = []

            for start, end in walking_windows:
                window_traj = trajectory[start:end + 1]

                foot_strikes_vertical = self.detect_foot_strikes_vertical(window_traj)
                foot_strikes = foot_strikes_vertical + start

                all_foot_strikes.extend(foot_strikes.tolist())

            all_foot_strikes = np.array(sorted(set(all_foot_strikes)))

            stride_times = self.compute_stride_times(all_foot_strikes)

            swing_stance = self.detect_swing_stance_phases(trajectory, all_foot_strikes)

            results[limb_name] = {
                'foot_strikes': all_foot_strikes,
                'stride_times': stride_times,
                'num_strides': len(stride_times),
                'swing_phases': swing_stance['swing'],
                'stance_phases': swing_stance['stance']
            }

            logger.info(
                f"{limb_name}: {len(all_foot_strikes)} foot strikes, "
                f"{len(stride_times)} valid strides"
            )

        return results

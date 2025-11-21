"""Step detection and foot strike event extraction"""
import numpy as np
import logging
from typing import List, Tuple, Dict
from scipy import signal
from ..utils.signal_processing import compute_mad, detect_peaks_adaptive
from ..utils.validation import validate_sufficient_strides
from ..constants import FPS_DEFAULT, MIN_STRIDE_DURATION_SEC, MAX_STRIDE_DURATION_SEC

logger = logging.getLogger(__name__)


class StepDetector:
    """Detect foot strikes and step events from paw trajectories"""

    def __init__(self,
                 fps: float = FPS_DEFAULT,
                 min_stride_duration: float = MIN_STRIDE_DURATION_SEC,
                 max_stride_duration: float = MAX_STRIDE_DURATION_SEC,
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
        Detect foot strikes using vertical position oscillations (v1.3.0 dynamic prominence).

        Two-pass detection:
        1. Initial pass with minimal threshold to estimate stride frequency
        2. Compute median prominence from detected peaks
        3. Final detection with threshold = 0.3 × median_prominence

        Args:
            paw_trajectory: Array of shape (N, 2) with (x, y) coordinates
            use_y: If True, use y-coordinate for detection (typical for side/bottom views)

        Returns:
            Array of frame indices where foot strikes occur
        """
        coord_idx = 1 if use_y else 0
        vertical_pos = paw_trajectory[:, coord_idx]

        valid_mask = ~np.isnan(vertical_pos)
        if np.sum(valid_mask) < 30:
            logger.warning("Insufficient valid data for foot strike detection")
            return np.array([])

        inverted_signal = -vertical_pos.copy()
        inverted_signal[~valid_mask] = np.nanmin(inverted_signal)

        # v1.3.0: Two-pass adaptive prominence detection
        signal_range = np.ptp(vertical_pos[valid_mask])

        # First pass: minimal threshold to find all potential peaks
        initial_prominence = 0.1 * signal_range
        initial_peaks, properties = signal.find_peaks(
            inverted_signal,
            prominence=initial_prominence,
            distance=max(5, self.min_stride_frames // 2)  # Relaxed distance
        )

        if len(initial_peaks) < 2:
            logger.warning(f"[v1.3.0] First pass found <2 peaks, using fallback")
            min_prominence = max(signal_range * 0.15, 0.1)
        else:
            # Compute prominence distribution
            prominences = properties.get('prominences', np.array([]))
            median_prominence = np.median(prominences)

            # v1.3.0: Threshold = prominence_multiplier × median_prominence
            min_prominence = self.prominence_multiplier * median_prominence

            # Safety clamp: [0.05×range, 0.5×range]
            min_prominence = np.clip(min_prominence, 0.05 * signal_range, 0.5 * signal_range)

            logger.debug(
                f"[v1.3.0] Step detection: initial_peaks={len(initial_peaks)}, "
                f"median_prominence={median_prominence:.3f}, threshold={min_prominence:.3f} cm"
            )

        # Second pass: final detection with adaptive prominence
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
        # Reduced from 100 to 30 frames (0.25s at 120fps) to handle fragmented walking windows
        if np.sum(valid_mask) < 30:
            logger.warning("Insufficient valid data for velocity-based detection")
            return np.array([])

        inverted_speed = -speed.copy()
        inverted_speed[~valid_mask] = np.nanmin(inverted_speed)

        mad = compute_mad(speed[valid_mask])
        # Improved: use adaptive prominence with min floor
        min_prominence = max(mad * self.prominence_multiplier, 0.5)  # 0.5cm/s floor

        peaks = detect_peaks_adaptive(
            inverted_speed,
            min_prominence=min_prominence,
            min_distance=self.min_stride_frames
        )

        valid_peaks = peaks[valid_mask[peaks]]

        return valid_peaks

    def detect_foot_strikes_bottom_view(self, paw_trajectory: np.ndarray) -> np.ndarray:
        """
        Detect foot strikes for bottom view using contact-based approach (v1.3.0 fix).

        Bottom camera looks UP at paws, so vertical oscillation doesn't work.
        Instead, detect based on speed minima (paw planted → speed ≈ 0).

        Args:
            paw_trajectory: Array of shape (N, 2) with (x, y) coordinates

        Returns:
            Array of frame indices where foot strikes occur
        """
        # Compute 2D speed (paw movement in bottom view plane)
        speed = np.sqrt(np.sum(np.diff(paw_trajectory, axis=0, prepend=paw_trajectory[:1]) ** 2, axis=1))

        valid_mask = ~np.isnan(speed)
        if np.sum(valid_mask) < 30:
            logger.warning("[Bottom View] Insufficient valid data for step detection")
            return np.array([])

        # Smooth speed to reduce noise
        from scipy.ndimage import gaussian_filter1d
        speed_smoothed = speed.copy()
        speed_smoothed[valid_mask] = gaussian_filter1d(speed[valid_mask], sigma=2)

        # Invert speed to find minima (stance = low speed)
        inverted_speed = -speed_smoothed.copy()
        inverted_speed[~valid_mask] = np.nanmin(inverted_speed)

        # Two-pass adaptive detection similar to vertical method
        signal_range = np.ptp(speed_smoothed[valid_mask])

        # First pass: find all potential contact points
        initial_prominence = 0.1 * signal_range
        initial_peaks, properties = signal.find_peaks(
            inverted_speed,
            prominence=initial_prominence,
            distance=max(5, self.min_stride_frames // 2)
        )

        if len(initial_peaks) < 2:
            logger.warning(f"[Bottom View] First pass found <2 peaks, using fallback")
            min_prominence = max(signal_range * 0.15, 0.2)  # Lower floor for bottom view
        else:
            prominences = properties.get('prominences', np.array([]))
            median_prominence = np.median(prominences)
            min_prominence = self.prominence_multiplier * median_prominence
            min_prominence = np.clip(min_prominence, 0.05 * signal_range, 0.5 * signal_range)

            logger.debug(
                f"[Bottom View] Step detection: initial_peaks={len(initial_peaks)}, "
                f"median_prominence={median_prominence:.3f}, threshold={min_prominence:.3f} cm/s"
            )

        # Second pass: final detection
        peaks = detect_peaks_adaptive(
            inverted_speed,
            min_prominence=min_prominence,
            min_distance=self.min_stride_frames
        )

        peaks = peaks[peaks < len(speed)]
        valid_peaks = peaks[valid_mask[peaks]]

        logger.info(f"[Bottom View] Detected {len(valid_peaks)} foot strikes using speed minima")

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

    def _detect_with_threshold_vertical(self,
                                        paw_trajectory: np.ndarray,
                                        min_prominence: float) -> np.ndarray:
        """
        Detect foot strikes using pre-computed threshold (vertical/side view, v1.3.2).

        Args:
            paw_trajectory: Array of shape (N, 2) with (x, y) coordinates
            min_prominence: Pre-computed prominence threshold

        Returns:
            Array of frame indices where foot strikes occur
        """
        coord_idx = 1  # y-coordinate
        vertical_pos = paw_trajectory[:, coord_idx]

        valid_mask = ~np.isnan(vertical_pos)
        if np.sum(valid_mask) < 30:
            return np.array([])

        inverted_signal = -vertical_pos.copy()
        inverted_signal[~valid_mask] = np.nanmin(inverted_signal)

        # Use pre-computed threshold
        peaks = detect_peaks_adaptive(
            inverted_signal,
            min_prominence=min_prominence,
            min_distance=self.min_stride_frames
        )

        peaks = peaks[peaks < len(vertical_pos)]
        valid_peaks = peaks[valid_mask[peaks]]

        return valid_peaks

    def _detect_with_threshold_bottom(self,
                                     paw_trajectory: np.ndarray,
                                     min_prominence: float) -> np.ndarray:
        """
        Detect foot strikes using pre-computed threshold (bottom view, v1.3.2).

        Args:
            paw_trajectory: Array of shape (N, 2) with (x, y) coordinates
            min_prominence: Pre-computed prominence threshold

        Returns:
            Array of frame indices where foot strikes occur
        """
        # Compute 2D speed
        speed = np.sqrt(np.sum(np.diff(paw_trajectory, axis=0, prepend=paw_trajectory[:1]) ** 2, axis=1))

        valid_mask = ~np.isnan(speed)
        if np.sum(valid_mask) < 30:
            return np.array([])

        # Smooth speed
        from scipy.ndimage import gaussian_filter1d
        speed_smoothed = speed.copy()
        speed_smoothed[valid_mask] = gaussian_filter1d(speed[valid_mask], sigma=2)

        # Invert speed to find minima (stance = low speed)
        inverted_speed = -speed_smoothed.copy()
        inverted_speed[~valid_mask] = np.nanmin(inverted_speed)

        # Use pre-computed threshold
        peaks = detect_peaks_adaptive(
            inverted_speed,
            min_prominence=min_prominence,
            min_distance=self.min_stride_frames
        )

        peaks = peaks[peaks < len(speed)]
        valid_peaks = peaks[valid_mask[peaks]]

        return valid_peaks

    def _compute_global_threshold(self,
                                 trajectory: np.ndarray,
                                 walking_windows: List[Tuple[int, int]],
                                 is_bottom_view: bool = True) -> float:
        """
        Compute global adaptive threshold across ALL walking windows (v1.3.2 fix).

        This fixes per-window threshold variability by analyzing all walking data together.

        Args:
            trajectory: Full trajectory array
            walking_windows: List of (start, end) walking window tuples
            is_bottom_view: If True, use speed-based detection; else use vertical position

        Returns:
            Global prominence threshold in cm or cm/s
        """
        all_window_signals = []
        all_prominences = []

        for start, end in walking_windows:
            window_traj = trajectory[start:end + 1]

            if len(window_traj) < 30:
                continue

            if is_bottom_view:
                # Speed-based signal for bottom view
                speed = np.sqrt(np.sum(np.diff(window_traj, axis=0, prepend=window_traj[:1]) ** 2, axis=1))
                valid_mask = ~np.isnan(speed)
                if np.sum(valid_mask) < 30:
                    continue

                # Smooth speed
                from scipy.ndimage import gaussian_filter1d
                speed_smoothed = speed.copy()
                speed_smoothed[valid_mask] = gaussian_filter1d(speed[valid_mask], sigma=2)

                # Invert for peak detection (stance = low speed)
                signal = -speed_smoothed
                signal[~valid_mask] = np.nanmin(signal)
            else:
                # Vertical position signal for side view
                coord_idx = 1  # y-coordinate
                vertical_pos = window_traj[:, coord_idx]
                valid_mask = ~np.isnan(vertical_pos)
                if np.sum(valid_mask) < 30:
                    continue

                # Invert for peak detection (foot strike = minimum y)
                signal = -vertical_pos.copy()
                signal[~valid_mask] = np.nanmin(signal)

            # First pass: find all potential peaks with minimal threshold
            signal_range = np.ptp(signal[~np.isnan(signal)])
            initial_prominence = 0.05 * signal_range

            from scipy import signal as scipy_signal
            peaks, properties = scipy_signal.find_peaks(
                signal,
                prominence=initial_prominence,
                distance=max(5, self.min_stride_frames // 2)
            )

            if len(peaks) >= 2 and 'prominences' in properties:
                all_prominences.extend(properties['prominences'].tolist())
                all_window_signals.append(signal_range)

        # Compute global threshold from all prominences
        if len(all_prominences) >= 3:
            median_prominence = np.median(all_prominences)
            global_threshold = self.prominence_multiplier * median_prominence

            # Safety clamps based on view type
            if is_bottom_view:
                # Bottom view: speed-based, minimum 0.3 cm/s
                min_floor = max(0.3, 0.05 * np.mean(all_window_signals))
                max_ceil = 0.5 * np.mean(all_window_signals)
            else:
                # Side/vertical view: position-based, minimum 0.5 cm
                min_floor = max(0.5, 0.05 * np.mean(all_window_signals))
                max_ceil = 0.5 * np.mean(all_window_signals)

            global_threshold = np.clip(global_threshold, min_floor, max_ceil)

            logger.info(
                f"[Global Threshold] Analyzed {len(all_prominences)} peaks across {len(all_window_signals)} windows, "
                f"median_prominence={median_prominence:.3f}, threshold={global_threshold:.3f} {'cm/s' if is_bottom_view else 'cm'}"
            )
        else:
            # Fallback: use reasonable defaults
            if is_bottom_view:
                global_threshold = 0.5  # 0.5 cm/s for bottom view
            else:
                global_threshold = 0.8  # 0.8 cm for vertical view

            logger.warning(
                f"[Global Threshold] Insufficient data for adaptive threshold, using fallback={global_threshold:.3f}"
            )

        return global_threshold

    def detect_all_limbs(self,
                        paw_trajectories: Dict[str, np.ndarray],
                        walking_windows: List[Tuple[int, int]]) -> Dict[str, Dict]:
        """
        Detect foot strikes for all limbs during walking windows (v1.3.2: unified cross-limb threshold).

        Args:
            paw_trajectories: Dictionary mapping limb names to trajectories
            walking_windows: List of (start, end) walking window tuples

        Returns:
            Dictionary mapping limb names to detection results
        """
        results = {}

        # v1.3.2 CRITICAL FIX: Compute ONE unified threshold for ALL limbs
        # This ensures consistent detection across all paws
        logger.info("[Unified Threshold] Computing cross-limb threshold for all paws together...")

        all_limb_prominences = []
        all_limb_signals = []

        # Pool data from ALL limbs to compute unified threshold
        for limb_name, trajectory in paw_trajectories.items():
            is_bottom_view = limb_name in ['paw_RR', 'paw_RL', 'paw_FR', 'paw_FL']

            for start, end in walking_windows:
                window_traj = trajectory[start:end + 1]

                if len(window_traj) < 30:
                    continue

                if is_bottom_view:
                    speed = np.sqrt(np.sum(np.diff(window_traj, axis=0, prepend=window_traj[:1]) ** 2, axis=1))
                    valid_mask = ~np.isnan(speed)
                    if np.sum(valid_mask) < 30:
                        continue

                    from scipy.ndimage import gaussian_filter1d
                    speed_smoothed = speed.copy()
                    speed_smoothed[valid_mask] = gaussian_filter1d(speed[valid_mask], sigma=2)

                    signal = -speed_smoothed
                    signal[~valid_mask] = np.nanmin(signal)
                else:
                    vertical_pos = window_traj[:, 1]
                    valid_mask = ~np.isnan(vertical_pos)
                    if np.sum(valid_mask) < 30:
                        continue

                    signal = -vertical_pos.copy()
                    signal[~valid_mask] = np.nanmin(signal)

                signal_range = np.ptp(signal[~np.isnan(signal)])
                initial_prominence = 0.05 * signal_range

                from scipy import signal as scipy_signal
                peaks, properties = scipy_signal.find_peaks(
                    signal,
                    prominence=initial_prominence,
                    distance=max(5, self.min_stride_frames // 2)
                )

                if len(peaks) >= 2 and 'prominences' in properties:
                    all_limb_prominences.extend(properties['prominences'].tolist())
                    all_limb_signals.append(signal_range)

        # Compute UNIFIED threshold from pooled data
        if len(all_limb_prominences) >= 5:
            median_prominence = np.median(all_limb_prominences)
            unified_threshold = self.prominence_multiplier * median_prominence

            # Safety clamp for bottom view
            min_floor = max(0.4, 0.05 * np.mean(all_limb_signals))  # Increased from 0.3 to 0.4
            max_ceil = 0.5 * np.mean(all_limb_signals)
            unified_threshold = np.clip(unified_threshold, min_floor, max_ceil)

            logger.info(
                f"[Unified Threshold] Analyzed {len(all_limb_prominences)} peaks from ALL limbs, "
                f"median_prominence={median_prominence:.3f}, unified_threshold={unified_threshold:.3f} cm/s"
            )
        else:
            unified_threshold = 0.5
            logger.warning(f"[Unified Threshold] Insufficient data, using fallback={unified_threshold:.3f} cm/s")

        # Now detect with SAME threshold for ALL limbs
        for limb_name, trajectory in paw_trajectories.items():
            logger.info(f"Detecting steps for {limb_name} (unified_threshold={unified_threshold:.3f})")

            is_bottom_view = limb_name in ['paw_RR', 'paw_RL', 'paw_FR', 'paw_FL']

            all_foot_strikes = []

            for start, end in walking_windows:
                window_traj = trajectory[start:end + 1]

                if is_bottom_view:
                    foot_strikes = self._detect_with_threshold_bottom(window_traj, unified_threshold)
                else:
                    foot_strikes = self._detect_with_threshold_vertical(window_traj, unified_threshold)

                foot_strikes = foot_strikes + start
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

        # Validate cross-limb consistency
        stride_counts = [res['num_strides'] for res in results.values()]
        if len(stride_counts) > 1:
            cv = np.std(stride_counts) / np.mean(stride_counts) if np.mean(stride_counts) > 0 else 0
            logger.info(
                f"[Cross-Limb Validation] Stride counts: {stride_counts}, "
                f"mean={np.mean(stride_counts):.1f}, CV={cv:.3f}"
            )
            if cv > 0.3:
                logger.warning(
                    f"[Cross-Limb Validation] High variability (CV={cv:.3f}) - "
                    f"consider adjusting prominence_multiplier or checking data quality"
                )

        return results

"""Computation of gait and range of motion metrics"""
import numpy as np
import logging
from typing import Dict, List, Tuple
from ..utils.geometry import (
    compute_stride_length,
    compute_trajectory_speed,
    compute_angle_3points,
    compute_range_of_motion,
    compute_symmetry_index,
    compute_lateral_deviation
)
from ..utils.signal_processing import compute_mad
from ..constants import FPS_DEFAULT

logger = logging.getLogger(__name__)


class GaitMetricsComputer:
    """Compute comprehensive gait analysis metrics"""

    def __init__(self, fps: float = FPS_DEFAULT):
        """
        Initialize metrics computer.

        Args:
            fps: Frame rate (Hz)
        """
        self.fps = fps

    def compute_cadence(self, foot_strikes: np.ndarray, duration: float) -> float:
        """
        Compute cadence (steps per minute).

        Args:
            foot_strikes: Array of foot strike indices
            duration: Total duration in seconds

        Returns:
            Cadence in steps/minute
        """
        if len(foot_strikes) < 2 or duration == 0:
            return np.nan

        num_steps = len(foot_strikes) - 1
        cadence = (num_steps / duration) * 60
        return cadence

    def compute_duty_cycle(self,
                          stance_phases: List[Tuple[int, int]],
                          stride_times: np.ndarray) -> float:
        """
        Compute duty cycle (stance time / stride time × 100%).

        Args:
            stance_phases: List of (start, end) stance phase tuples
            stride_times: Array of stride times in seconds

        Returns:
            Duty cycle percentage
        """
        if len(stance_phases) == 0 or len(stride_times) == 0:
            return np.nan

        stance_durations = [(end - start + 1) / self.fps for start, end in stance_phases]
        avg_stance = np.mean(stance_durations)
        avg_stride = np.mean(stride_times)

        if avg_stride == 0:
            return np.nan

        duty_cycle = (avg_stance / avg_stride) * 100
        return duty_cycle

    def compute_average_speed(self, trajectory: np.ndarray) -> float:
        """
        Compute average trajectory speed.

        Args:
            trajectory: Array of shape (N, 2) with (x, y) coordinates

        Returns:
            Average speed in cm/s
        """
        speeds = compute_trajectory_speed(trajectory, self.fps)
        valid_speeds = speeds[~np.isnan(speeds)]

        if len(valid_speeds) == 0:
            return np.nan

        return np.mean(valid_speeds)

    def compute_stride_lengths(self,
                               foot_strikes: np.ndarray,
                               trajectory: np.ndarray) -> np.ndarray:
        """
        Compute stride lengths.

        Args:
            foot_strikes: Array of foot strike indices
            trajectory: Paw trajectory

        Returns:
            Array of stride lengths in cm
        """
        return compute_stride_length(foot_strikes, trajectory)

    def compute_regularity_index(self,
                                diagonal_pair_strikes: Tuple[np.ndarray, np.ndarray],
                                tolerance_frames: int = 6) -> float:
        """
        Compute regularity index for diagonal limb pair.

        RI = proportion of steps matching normal pattern.

        Args:
            diagonal_pair_strikes: Tuple of (limb1_strikes, limb2_strikes)
            tolerance_frames: Temporal tolerance for matching (frames)

        Returns:
            Regularity index (0-1)
        """
        strikes1, strikes2 = diagonal_pair_strikes

        if len(strikes1) < 2 or len(strikes2) < 2:
            return np.nan

        expected_offset = len(strikes1) / 2

        matched = 0
        total = min(len(strikes1), len(strikes2))

        for i, strike1 in enumerate(strikes1[:total]):
            for strike2 in strikes2:
                if abs(strike1 - strike2) <= tolerance_frames:
                    matched += 1
                    break

        regularity = matched / total if total > 0 else 0
        return regularity

    def compute_phase_dispersion(self,
                                diagonal_pair_strikes: Tuple[np.ndarray, np.ndarray],
                                stride_times: np.ndarray) -> float:
        """
        Compute phase dispersion (coordination timing).

        Args:
            diagonal_pair_strikes: Tuple of (limb1_strikes, limb2_strikes)
            stride_times: Array of stride times

        Returns:
            Phase dispersion value
        """
        strikes1, strikes2 = diagonal_pair_strikes

        if len(strikes1) < 2 or len(strikes2) < 2 or len(stride_times) == 0:
            return np.nan

        avg_stride_frames = np.mean(stride_times) * self.fps

        phase_delays = []
        for strike1 in strikes1:
            closest_strike2 = strikes2[np.argmin(np.abs(strikes2 - strike1))]
            delay = abs(strike1 - closest_strike2)
            phase = delay / avg_stride_frames
            phase_delays.append(phase)

        return np.mean(phase_delays)

    def compute_swing_stance_ratio(self,
                                  swing_phases: List[Tuple[int, int]],
                                  stance_phases: List[Tuple[int, int]]) -> float:
        """
        Compute swing to stance ratio.

        Args:
            swing_phases: List of swing phase windows
            stance_phases: List of stance phase windows

        Returns:
            Swing/stance ratio
        """
        if len(swing_phases) == 0 or len(stance_phases) == 0:
            return np.nan

        swing_duration = sum((end - start + 1) for start, end in swing_phases) / self.fps
        stance_duration = sum((end - start + 1) for start, end in stance_phases) / self.fps

        if stance_duration == 0:
            return np.nan

        return swing_duration / stance_duration

    def compute_all_gait_metrics(self,
                                step_results: Dict[str, Dict],
                                paw_trajectories: Dict[str, np.ndarray],
                                com_trajectory: np.ndarray,
                                walking_windows: List[Tuple[int, int]]) -> Dict:
        """
        Compute all gait metrics for all limbs.

        Args:
            step_results: Dictionary with step detection results per limb
            paw_trajectories: Dictionary of paw trajectories
            com_trajectory: Center of mass trajectory
            walking_windows: List of walking windows

        Returns:
            Dictionary of all gait metrics
        """
        logger.info("Computing comprehensive gait metrics")

        total_walking_frames = sum(end - start + 1 for start, end in walking_windows)
        total_walking_duration = total_walking_frames / self.fps

        metrics = {}

        for limb in ['paw_RR', 'paw_RL']:
            if limb not in step_results:
                continue

            limb_results = step_results[limb]
            limb_traj = paw_trajectories[limb]

            foot_strikes = limb_results['foot_strikes']
            stride_times = limb_results['stride_times']
            swing_phases = limb_results['swing_phases']
            stance_phases = limb_results['stance_phases']

            cadence = self.compute_cadence(foot_strikes, total_walking_duration)

            duty_cycle = self.compute_duty_cycle(stance_phases, stride_times)

            avg_speed = self.compute_average_speed(limb_traj)

            stride_lengths = self.compute_stride_lengths(foot_strikes, limb_traj)

            swing_stance_ratio = self.compute_swing_stance_ratio(swing_phases, stance_phases)

            metrics[limb] = {
                'cadence': cadence,
                'duty_cycle': duty_cycle,
                'avg_speed': avg_speed,
                'stride_lengths': stride_lengths,
                'stride_times': stride_times,
                'swing_stance_ratio': swing_stance_ratio,
                'num_strides': len(stride_times)
            }

        if 'paw_RR' in step_results and 'paw_FL' in step_results:
            diagonal_pair_1 = (step_results['paw_RR']['foot_strikes'],
                             step_results['paw_FL']['foot_strikes'])
            ri_1 = self.compute_regularity_index(diagonal_pair_1)
            phase_disp_1 = self.compute_phase_dispersion(
                diagonal_pair_1,
                step_results['paw_RR']['stride_times']
            )
            metrics['diagonal_RR_FL'] = {
                'regularity_index': ri_1,
                'phase_dispersion': phase_disp_1
            }

        if 'paw_RL' in step_results and 'paw_FR' in step_results:
            diagonal_pair_2 = (step_results['paw_RL']['foot_strikes'],
                             step_results['paw_FR']['foot_strikes'])
            ri_2 = self.compute_regularity_index(diagonal_pair_2)
            phase_disp_2 = self.compute_phase_dispersion(
                diagonal_pair_2,
                step_results['paw_RL']['stride_times']
            )
            metrics['diagonal_RL_FR'] = {
                'regularity_index': ri_2,
                'phase_dispersion': phase_disp_2
            }

        all_hind_limb_duty_cycles = []
        for limb in ['paw_RR', 'paw_RL']:
            if limb in metrics and 'duty_cycle' in metrics[limb]:
                if not np.isnan(metrics[limb]['duty_cycle']):
                    all_hind_limb_duty_cycles.append(metrics[limb]['duty_cycle'])

        if all_hind_limb_duty_cycles:
            metrics['quadrupedal'] = {
                'avg_duty_cycle': np.mean(all_hind_limb_duty_cycles)
            }

        com_speed = self.compute_average_speed(com_trajectory)
        metrics['whole_body'] = {
            'com_avg_speed': com_speed
        }

        logger.info("Gait metrics computation complete")

        return metrics


class ROMMetricsComputer:
    """Compute range of motion metrics"""

    def __init__(self, fps: float = FPS_DEFAULT):
        """
        Initialize ROM metrics computer.

        Args:
            fps: Frame rate (Hz)
        """
        self.fps = fps

    def compute_com_sway(self, com_trajectory: np.ndarray) -> Dict[str, float]:
        """
        Compute CoM lateral and AP sway.

        Args:
            com_trajectory: CoM trajectory of shape (N, 2)

        Returns:
            Dictionary with ML and AP sway values
        """
        ml_mean, ml_std = compute_lateral_deviation(com_trajectory, axis=0)
        ap_mean, ap_std = compute_lateral_deviation(com_trajectory, axis=1)

        return {
            'ml_sway_cm': ml_std,
            'ap_sway_cm': ap_std
        }

    def compute_hip_asymmetry(self,
                             left_hip_angles: np.ndarray,
                             right_hip_angles: np.ndarray) -> float:
        """
        Compute hip asymmetry index.

        Args:
            left_hip_angles: Left hip angle trajectory
            right_hip_angles: Right hip angle trajectory

        Returns:
            Symmetry index
        """
        return compute_symmetry_index(left_hip_angles, right_hip_angles)

    def compute_joint_rom_and_velocity(self,
                                      p1: np.ndarray,
                                      p2: np.ndarray,
                                      p3: np.ndarray) -> Dict[str, float]:
        """
        Compute joint ROM and angular velocity.

        Args:
            p1: Proximal point trajectory (N, 2)
            p2: Joint center trajectory (N, 2)
            p3: Distal point trajectory (N, 2)

        Returns:
            Dictionary with ROM and angular velocity metrics
        """
        angles = compute_angle_3points(p1, p2, p3)
        valid_angles = angles[~np.isnan(angles)]

        if len(valid_angles) == 0:
            return {'rom': np.nan, 'angular_velocity_mean': np.nan, 'angular_velocity_max': np.nan}

        rom = compute_range_of_motion(angles)

        angular_velocity = np.gradient(angles) * self.fps
        valid_ang_vel = angular_velocity[~np.isnan(angular_velocity)]

        return {
            'rom': rom,
            'angular_velocity_mean': np.mean(np.abs(valid_ang_vel)) if len(valid_ang_vel) > 0 else np.nan,
            'angular_velocity_max': np.max(np.abs(valid_ang_vel)) if len(valid_ang_vel) > 0 else np.nan
        }

    def compute_all_rom_metrics(self,
                               keypoints: Dict[str, np.ndarray],
                               com_trajectory: np.ndarray) -> Dict:
        """
        Compute all ROM metrics.

        Args:
            keypoints: Dictionary of keypoint trajectories
            com_trajectory: CoM trajectory

        Returns:
            Dictionary of ROM metrics
        """
        logger.info("Computing range of motion metrics")

        metrics = {}

        com_sway = self.compute_com_sway(com_trajectory)
        metrics['com_sway'] = com_sway

        if all(k in keypoints for k in ['shoulder_R', 'elbow_R', 'paw_FR']):
            elbow_r_metrics = self.compute_joint_rom_and_velocity(
                keypoints['shoulder_R'],
                keypoints['elbow_R'],
                keypoints['paw_FR']
            )
            metrics['elbow_R'] = elbow_r_metrics

        if all(k in keypoints for k in ['shoulder_L', 'elbow_L', 'paw_FL']):
            elbow_l_metrics = self.compute_joint_rom_and_velocity(
                keypoints['shoulder_L'],
                keypoints['elbow_L'],
                keypoints['paw_FL']
            )
            metrics['elbow_L'] = elbow_l_metrics

        # Hip ROM computation (v1.2.0 fix - now using knee keypoints)
        if all(k in keypoints for k in ['hip_center', 'hip_R', 'knee_R']):
            hip_r_metrics = self.compute_joint_rom_and_velocity(
                keypoints['hip_center'],
                keypoints['hip_R'],
                keypoints['knee_R']
            )
            metrics['hip_R'] = hip_r_metrics
            logger.info("Computed hip_R ROM metrics")
        else:
            logger.info("hip_R ROM skipped: missing keypoints (need hip_center, hip_R, knee_R)")

        if all(k in keypoints for k in ['hip_center', 'hip_L', 'knee_L']):
            hip_l_metrics = self.compute_joint_rom_and_velocity(
                keypoints['hip_center'],
                keypoints['hip_L'],
                keypoints['knee_L']
            )
            metrics['hip_L'] = hip_l_metrics
            logger.info("Computed hip_L ROM metrics")
        else:
            logger.info("hip_L ROM skipped: missing keypoints (need hip_center, hip_L, knee_L)")

        # Compute hip asymmetry if both hips were calculated (v1.2.0 fix)
        if 'hip_R' in metrics and 'hip_L' in metrics:
            # Recompute angles for asymmetry calculation
            hip_r_angles = compute_angle_3points(
                keypoints['hip_center'],
                keypoints['hip_R'],
                keypoints['knee_R']
            )
            hip_l_angles = compute_angle_3points(
                keypoints['hip_center'],
                keypoints['hip_L'],
                keypoints['knee_L']
            )
            hip_asymmetry = self.compute_hip_asymmetry(hip_l_angles, hip_r_angles)
            metrics['hip_asymmetry'] = {'asymmetry_index': hip_asymmetry}
            logger.info(f"Computed hip asymmetry index: {hip_asymmetry:.4f}")

        logger.info("ROM metrics computation complete")

        return metrics

    def compute_com_3d(self,
                       top_keypoints: Dict[str, np.ndarray],
                       side_keypoints: Dict[str, np.ndarray],
                       weights: Dict[str, float] = None) -> np.ndarray:
        """
        Compute 3D center of mass from TOP and SIDE views (v1.2.0).

        Combines XY coordinates from TOP view with Z (vertical) from SIDE view
        for full 3D COM trajectory calculation.

        Args:
            top_keypoints: Dictionary of keypoints from TOP view (XY plane)
            side_keypoints: Dictionary of keypoints from SIDE view (XZ plane, Z=vertical)
            weights: Optional dictionary of keypoint weights for COM calculation

        Returns:
            Array of shape (N, 3) with (x, y, z) COM coordinates
        """
        # Default weights if not provided
        if weights is None:
            weights = {
                'spine1': 0.15,
                'spine2': 0.20,
                'spine3': 0.20,
                'tailbase': 0.15,
                'nose': 0.10,
                'hip_R': 0.10,
                'hip_L': 0.10
            }

        # Get frame count
        n_frames = len(next(iter(top_keypoints.values())))

        # Initialize 3D COM
        com_3d = np.zeros((n_frames, 3))

        for frame_idx in range(n_frames):
            # Collect XY from TOP view
            top_points = []
            top_weights = []

            for keypoint_name, weight in weights.items():
                if keypoint_name in top_keypoints:
                    pos_xy = top_keypoints[keypoint_name][frame_idx]
                    if not np.isnan(pos_xy).any():
                        top_points.append(pos_xy)
                        top_weights.append(weight)

            # Collect Z from SIDE view (use Y coordinate as vertical Z)
            side_z_values = []
            side_weights = []

            for keypoint_name, weight in weights.items():
                if keypoint_name in side_keypoints:
                    pos_side = side_keypoints[keypoint_name][frame_idx]
                    if not np.isnan(pos_side).any():
                        # SIDE view: X is horizontal, Y is vertical (Z in 3D)
                        side_z_values.append(pos_side[1])
                        side_weights.append(weight)

            # Compute weighted COM for XY
            if len(top_points) > 0:
                top_points = np.array(top_points)
                top_weights = np.array(top_weights)
                top_weights = top_weights / np.sum(top_weights)
                com_xy = np.average(top_points, axis=0, weights=top_weights)
                com_3d[frame_idx, 0:2] = com_xy
            else:
                com_3d[frame_idx, 0:2] = np.nan

            # Compute weighted COM for Z
            if len(side_z_values) > 0:
                side_z_values = np.array(side_z_values)
                side_weights = np.array(side_weights)
                side_weights = side_weights / np.sum(side_weights)
                com_z = np.average(side_z_values, weights=side_weights)
                com_3d[frame_idx, 2] = com_z
            else:
                com_3d[frame_idx, 2] = np.nan

        logger.info("[v1.2.0] Computed 3D COM trajectory from TOP+SIDE views")

        return com_3d

    def compute_velocity_3d(self,
                           com_3d: np.ndarray,
                           fps: float = FPS_DEFAULT,
                           scaling_factor: float = 1.0) -> np.ndarray:
        """
        Compute 3D velocity from 3D COM trajectory (v1.2.0).

        Args:
            com_3d: Array of shape (N, 3) with (x, y, z) coordinates
            fps: Frame rate (Hz)
            scaling_factor: Scaling factor for unit conversion (cm/pixel)

        Returns:
            Array of shape (N,) with 3D speed values (cm/s)
        """
        # Compute 3D displacement
        displacement = np.diff(com_3d, axis=0, prepend=com_3d[:1])

        # Compute 3D speed: sqrt(dx² + dy² + dz²)
        speed_3d = np.sqrt(np.sum(displacement ** 2, axis=1))

        # Convert to cm/s
        speed_3d_cm_s = speed_3d * scaling_factor * fps

        logger.info(f"[v1.2.0] Computed 3D velocity (mean: {np.nanmean(speed_3d_cm_s):.2f} cm/s)")

        return speed_3d_cm_s

    def compute_joint_angle_triplet(self,
                                   proximal: np.ndarray,
                                   joint: np.ndarray,
                                   distal: np.ndarray) -> np.ndarray:
        """
        Compute joint angles using 3-point (triplet) method (v1.2.0).

        Calculates angle at joint vertex formed by proximal-joint-distal.
        More anatomically accurate than 2-point displacement methods.

        Args:
            proximal: Array of shape (N, 2) for proximal point
            joint: Array of shape (N, 2) for joint vertex
            distal: Array of shape (N, 2) for distal point

        Returns:
            Array of shape (N,) with joint angles in degrees
        """
        n_frames = len(joint)
        angles = np.zeros(n_frames)

        for i in range(n_frames):
            p1 = proximal[i]
            p2 = joint[i]
            p3 = distal[i]

            if np.isnan(p1).any() or np.isnan(p2).any() or np.isnan(p3).any():
                angles[i] = np.nan
            else:
                angles[i] = compute_angle_3points(p1, p2, p3)

        return angles

    def compute_rom_v2(self,
                      side_keypoints: Dict[str, np.ndarray],
                      smoothing_window: int = 3) -> Dict[str, Dict]:
        """
        Compute ROM using triplet angle method with minimal smoothing (v1.2.0).

        Uses 3-point angle calculation for anatomically accurate joint angles:
        - Hip: spine3 → hip → knee
        - Elbow: shoulder → elbow → paw

        Args:
            side_keypoints: Dictionary of keypoints from SIDE view
            smoothing_window: Minimal smoothing window (default 3 frames)

        Returns:
            Dictionary with ROM metrics for each joint
        """
        from ..utils.signal_processing import apply_savgol_filter, compute_angular_velocity

        logger.info("[v1.2.0] Computing ROM with triplet angle method")

        rom_metrics = {}

        # Hip ROM (if available)
        if all(k in side_keypoints for k in ['spine3', 'hip_R', 'knee_R']):
            hip_angles = self.compute_joint_angle_triplet(
                side_keypoints['spine3'],
                side_keypoints['hip_R'],
                side_keypoints['knee_R']
            )

            # Minimal smoothing
            valid_mask = ~np.isnan(hip_angles)
            if np.sum(valid_mask) > smoothing_window:
                smoothed = apply_savgol_filter(
                    hip_angles[valid_mask],
                    smoothing_window,
                    2
                )
                hip_angles_smooth = hip_angles.copy()
                hip_angles_smooth[valid_mask] = smoothed
            else:
                hip_angles_smooth = hip_angles

            # Compute ROM and angular velocity
            hip_rom = compute_range_of_motion(hip_angles_smooth)
            hip_angular_vel = compute_angular_velocity(hip_angles_smooth, self.fps)

            rom_metrics['hip_R'] = {
                'rom_degrees': float(hip_rom) if not np.isnan(hip_rom) else None,
                'mean_angle_degrees': float(np.nanmean(hip_angles_smooth)),
                'angular_velocity_mean': float(np.nanmean(np.abs(hip_angular_vel))),
                'angular_velocity_max': float(np.nanmax(np.abs(hip_angular_vel)))
            }

            logger.info(f"[v1.2.0] Hip ROM: {hip_rom:.1f}°, "
                       f"Angular vel: {np.nanmean(np.abs(hip_angular_vel)):.1f} °/s")

        # Elbow ROM (if available)
        if all(k in side_keypoints for k in ['shoulder_R', 'elbow_R', 'paw_FR']):
            elbow_angles = self.compute_joint_angle_triplet(
                side_keypoints['shoulder_R'],
                side_keypoints['elbow_R'],
                side_keypoints['paw_FR']
            )

            # Minimal smoothing
            valid_mask = ~np.isnan(elbow_angles)
            if np.sum(valid_mask) > smoothing_window:
                smoothed = apply_savgol_filter(
                    elbow_angles[valid_mask],
                    smoothing_window,
                    2
                )
                elbow_angles_smooth = elbow_angles.copy()
                elbow_angles_smooth[valid_mask] = smoothed
            else:
                elbow_angles_smooth = elbow_angles

            # Compute ROM and angular velocity
            elbow_rom = compute_range_of_motion(elbow_angles_smooth)
            elbow_angular_vel = compute_angular_velocity(elbow_angles_smooth, self.fps)

            rom_metrics['elbow_R'] = {
                'rom_degrees': float(elbow_rom) if not np.isnan(elbow_rom) else None,
                'mean_angle_degrees': float(np.nanmean(elbow_angles_smooth)),
                'angular_velocity_mean': float(np.nanmean(np.abs(elbow_angular_vel))),
                'angular_velocity_max': float(np.nanmax(np.abs(elbow_angular_vel)))
            }

            logger.info(f"[v1.2.0] Elbow ROM: {elbow_rom:.1f}°, "
                       f"Angular vel: {np.nanmean(np.abs(elbow_angular_vel)):.1f} °/s")

        logger.info("[v1.2.0] ROM v2 computation complete")

        return rom_metrics

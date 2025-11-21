"""Data preprocessing: smoothing, scaling, and CoM computation"""
import numpy as np
import logging
from typing import Dict, Tuple
from ..utils.signal_processing import (
    apply_savgol_filter,
    interpolate_missing_values,
    filter_outliers_mad
)
from ..utils.geometry import (
    compute_scaling_factor,
    compute_scaling_factor_v2,
    pixels_to_cm,
    compute_center_of_mass,
    compute_distance_2d
)
from ..utils.validation import validate_scaling_factor
from ..constants import (
    SAVGOL_WINDOW_SIZE_DEFAULT,
    SAVGOL_POLY_ORDER_DEFAULT,
    LEGACY_SPINE_LENGTH_CM,
    DEFAULT_MOUSE_BODY_LENGTH_CM,
    MIN_LIKELIHOOD_SCALING,
    SCALING_TOLERANCE_DEFAULT,
    DATA_COMPLETENESS_MEDIUM
)

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Preprocess pose estimation data for analysis"""

    def __init__(self,
                 smoothing_window: int = SAVGOL_WINDOW_SIZE_DEFAULT,
                 smoothing_poly: int = SAVGOL_POLY_ORDER_DEFAULT,
                 outlier_threshold: float = 3.0,
                 max_interpolation_gap: int = 5):
        """
        Initialize preprocessor.

        Args:
            smoothing_window: Window length for Savitzky-Golay filter
            smoothing_poly: Polynomial order for Savitzky-Golay filter
            outlier_threshold: MAD threshold for outlier detection
            max_interpolation_gap: Maximum gap size for interpolation (frames)
        """
        self.smoothing_window = smoothing_window
        self.smoothing_poly = smoothing_poly
        self.outlier_threshold = outlier_threshold
        self.max_interpolation_gap = max_interpolation_gap
        self.scale_factor = None  # Legacy: single scaling factor
        self.scaling_factors = {}  # v1.3.0: Per-view scaling factors {top, side, bottom}

    def compute_scale_factor(self,
                            point1: np.ndarray,
                            point2: np.ndarray,
                            known_distance_cm: float = LEGACY_SPINE_LENGTH_CM) -> float:
        """
        Compute scaling factor from known distance (LEGACY v1.1).

        Args:
            point1: Array of shape (N, 2) for first point (e.g., snout)
            point2: Array of shape (N, 2) for second point (e.g., tail_base)
            known_distance_cm: Known real-world distance (default 8 cm)

        Returns:
            Scaling factor (cm/pixel)
        """
        valid_mask = ~(np.isnan(point1).any(axis=1) | np.isnan(point2).any(axis=1))

        if np.sum(valid_mask) < 100:
            logger.warning("Insufficient valid points for robust scaling factor computation")

        valid_p1 = point1[valid_mask]
        valid_p2 = point2[valid_mask]

        self.scale_factor = compute_scaling_factor(valid_p1, valid_p2, known_distance_cm)

        validate_scaling_factor(self.scale_factor, known_distance_cm)

        logger.info(f"Computed scaling factor: {self.scale_factor:.6f} cm/pixel")

        return self.scale_factor

    def compute_scale_factor_v2(self,
                               snout: np.ndarray,
                               tailbase: np.ndarray,
                               likelihood_snout: np.ndarray = None,
                               likelihood_tail: np.ndarray = None,
                               expected_body_length_cm: float = DEFAULT_MOUSE_BODY_LENGTH_CM,
                               min_likelihood: float = MIN_LIKELIHOOD_SCALING,
                               tolerance: float = SCALING_TOLERANCE_DEFAULT) -> Tuple[float, Dict]:
        """
        Compute scaling factor using full-body measurement (v1.2.0).

        Uses snout→tailbase distance for more accurate body length estimation
        with likelihood filtering and robust outlier removal.

        Args:
            snout: Array of shape (N, 2) for snout coordinates
            tailbase: Array of shape (N, 2) for tailbase coordinates
            likelihood_snout: Optional likelihood array for snout
            likelihood_tail: Optional likelihood array for tailbase
            expected_body_length_cm: Expected body length (default 10.0 cm for adult mice)
            min_likelihood: Minimum confidence threshold (default 0.9)
            tolerance: Allowed variance from median (default 0.25 = ±25%)

        Returns:
            Tuple of (scaling_factor, diagnostics_dict)
        """
        self.scale_factor, diagnostics = compute_scaling_factor_v2(
            snout, tailbase,
            likelihood_snout, likelihood_tail,
            expected_body_length_cm,
            min_likelihood,
            tolerance
        )

        validate_scaling_factor(self.scale_factor, expected_body_length_cm)

        logger.info(f"[v1.2.0] Computed scaling factor: {self.scale_factor:.6f} cm/pixel")
        logger.info(f"[v1.2.0] Used {diagnostics['frames_used']}/{diagnostics['frames_total']} frames "
                   f"(outliers removed: {diagnostics['outliers_removed']})")

        return self.scale_factor, diagnostics

    def compute_per_view_scaling(self,
                                 view_keypoints: Dict[str, Dict[str, np.ndarray]],
                                 expected_body_length_cm: float = DEFAULT_MOUSE_BODY_LENGTH_CM,
                                 min_likelihood: float = MIN_LIKELIHOOD_SCALING,
                                 tolerance: float = SCALING_TOLERANCE_DEFAULT) -> Tuple[Dict[str, float], Dict]:
        """
        Compute independent scaling factors for each camera view (v1.3.0).

        Each view has different camera distance/focal length, requiring
        view-specific pixel-to-cm conversion.

        Args:
            view_keypoints: Dict with keys 'top', 'side', 'bottom', each containing:
                - 'point1': Reference point 1 (e.g., snout)
                - 'point2': Reference point 2 (e.g., tail_base)
                - 'likelihood1': Optional likelihood for point1
                - 'likelihood2': Optional likelihood for point2
            expected_body_length_cm: Expected body length (default 10.0 cm)
            min_likelihood: Minimum confidence threshold
            tolerance: Allowed variance from median

        Returns:
            Tuple of (scaling_factors_dict, diagnostics_dict)
            scaling_factors_dict: {'top': cm/px, 'side': cm/px, 'bottom': cm/px}
            diagnostics_dict: Per-view diagnostics and cross-view validation
        """
        scaling_factors = {}
        per_view_diagnostics = {}

        for view_name, keypoints in view_keypoints.items():
            point1 = keypoints.get('point1')
            point2 = keypoints.get('point2')
            likelihood1 = keypoints.get('likelihood1')
            likelihood2 = keypoints.get('likelihood2')

            if point1 is None or point2 is None:
                logger.warning(f"Missing keypoints for {view_name} view, skipping")
                continue

            # Compute scaling factor for this view
            scale_factor, view_diag = compute_scaling_factor_v2(
                point1, point2,
                likelihood1, likelihood2,
                expected_body_length_cm,
                min_likelihood,
                tolerance
            )

            scaling_factors[view_name] = scale_factor
            per_view_diagnostics[view_name] = view_diag

            logger.info(f"[v1.3.0] {view_name.upper()} view scaling: {scale_factor:.6f} cm/pixel "
                       f"(used {view_diag['frames_used']}/{view_diag['frames_total']} frames)")

        # Store in instance
        self.scaling_factors = scaling_factors

        # For backward compatibility, set legacy scale_factor to TOP view
        if 'top' in scaling_factors:
            self.scale_factor = scaling_factors['top']

        # Cross-view validation
        validation = self._validate_cross_view_scaling(scaling_factors)

        diagnostics = {
            'per_view': per_view_diagnostics,
            'validation': validation,
            'scaling_factors': scaling_factors
        }

        return scaling_factors, diagnostics

    def _validate_cross_view_scaling(self, scaling_factors: Dict[str, float]) -> Dict:
        """
        Validate consistency across camera views (v1.3.0).

        Args:
            scaling_factors: Dict of {view_name: cm/pixel}

        Returns:
            Validation diagnostics
        """
        validation = {
            'cross_view_consistency': 'PASS',
            'warnings': [],
            'physical_plausibility': 'PASS'
        }

        # Expected range for mouse videos: [0.02, 0.15] cm/px
        PHYSICAL_MIN = 0.02
        PHYSICAL_MAX = 0.15
        CONSISTENCY_THRESHOLD = 0.02  # 0.02 cm/px difference

        # Physical plausibility check
        for view, scale in scaling_factors.items():
            if scale < PHYSICAL_MIN:
                validation['physical_plausibility'] = 'FAIL'
                validation['warnings'].append(
                    f"{view.upper()}: {scale:.4f} cm/px < {PHYSICAL_MIN} cm/px - Camera too far or keypoints incorrect"
                )
            elif scale > PHYSICAL_MAX:
                validation['physical_plausibility'] = 'FAIL'
                validation['warnings'].append(
                    f"{view.upper()}: {scale:.4f} cm/px > {PHYSICAL_MAX} cm/px - Camera too close or keypoints incorrect"
                )

        # Cross-view consistency check
        if 'top' in scaling_factors and 'side' in scaling_factors:
            diff_top_side = abs(scaling_factors['top'] - scaling_factors['side'])
            if diff_top_side > CONSISTENCY_THRESHOLD:
                validation['cross_view_consistency'] = 'WARNING'
                validation['warnings'].append(
                    f"TOP-SIDE differ by {diff_top_side:.4f} cm/px (>{CONSISTENCY_THRESHOLD}) - Check camera alignment"
                )

        if 'side' in scaling_factors and 'bottom' in scaling_factors:
            diff_side_bottom = abs(scaling_factors['side'] - scaling_factors['bottom'])
            if diff_side_bottom > CONSISTENCY_THRESHOLD:
                validation['cross_view_consistency'] = 'WARNING'
                validation['warnings'].append(
                    f"SIDE-BOTTOM differ by {diff_side_bottom:.4f} cm/px (>{CONSISTENCY_THRESHOLD}) - Check camera alignment"
                )

        # Log validation results
        if validation['warnings']:
            for warning in validation['warnings']:
                logger.warning(f"[v1.3.0] Cross-view validation: {warning}")
        else:
            logger.info("[v1.3.0] Cross-view validation: All checks PASS")

        return validation

    def preprocess_trajectory(self, trajectory: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Preprocess single trajectory (x or y coordinates).

        Steps:
        1. Filter outliers using MAD
        2. Interpolate missing values
        3. Apply Savitzky-Golay smoothing

        Args:
            trajectory: 1D array of coordinates

        Returns:
            Tuple of (preprocessed_trajectory, quality_metrics)
        """
        original_valid = np.sum(~np.isnan(trajectory))

        filtered = filter_outliers_mad(trajectory, self.outlier_threshold)
        outliers_removed = original_valid - np.sum(~np.isnan(filtered))

        interpolated, valid_mask = interpolate_missing_values(
            filtered,
            self.max_interpolation_gap
        )
        points_interpolated = np.sum(valid_mask) - np.sum(~np.isnan(filtered))

        valid_data = interpolated[valid_mask]
        if len(valid_data) > self.smoothing_window:
            smoothed_valid = apply_savgol_filter(
                valid_data,
                self.smoothing_window,
                self.smoothing_poly
            )
            result = interpolated.copy()
            result[valid_mask] = smoothed_valid
        else:
            result = interpolated

        quality_metrics = {
            'original_valid': original_valid,
            'outliers_removed': outliers_removed,
            'points_interpolated': points_interpolated,
            'final_valid': np.sum(~np.isnan(result)),
            'completeness': np.sum(~np.isnan(result)) / len(trajectory)
        }

        return result, quality_metrics

    def preprocess_keypoint_2d(self, keypoint_data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Preprocess 2D keypoint data (x, y coordinates).

        Args:
            keypoint_data: Array of shape (N, 2) with (x, y) coordinates

        Returns:
            Tuple of (preprocessed_data, quality_metrics)
        """
        x_processed, x_metrics = self.preprocess_trajectory(keypoint_data[:, 0])
        y_processed, y_metrics = self.preprocess_trajectory(keypoint_data[:, 1])

        processed = np.column_stack([x_processed, y_processed])

        quality_metrics = {
            'x_metrics': x_metrics,
            'y_metrics': y_metrics,
            'overall_completeness': (x_metrics['completeness'] + y_metrics['completeness']) / 2
        }

        return processed, quality_metrics

    def convert_to_cm(self, pixel_data: np.ndarray, view: str = None) -> np.ndarray:
        """
        Convert pixel coordinates to centimeters (v1.3.0: view-specific scaling).

        Args:
            pixel_data: Array of coordinates in pixels
            view: Optional view name ('top', 'side', 'bottom') for per-view scaling.
                  If None, uses legacy single scale_factor.

        Returns:
            Array of coordinates in centimeters
        """
        if view is not None and view in self.scaling_factors:
            # v1.3.0: Use view-specific scaling factor
            scale_factor = self.scaling_factors[view]
        elif self.scale_factor is not None:
            # Legacy: Use single scaling factor
            scale_factor = self.scale_factor
        else:
            raise ValueError("Scale factor not computed. Call compute_scale_factor or compute_per_view_scaling first.")

        return pixels_to_cm(pixel_data, scale_factor)

    def compute_com_trajectory(self,
                               snout: np.ndarray,
                               neck: np.ndarray,
                               rib_center: np.ndarray,
                               hip_center: np.ndarray) -> np.ndarray:
        """
        Compute center of mass trajectory using weighted average (v1.2.0 physics-based).

        Physics: COM = weighted_average([snout, neck, rib, hip])
        Thoracic mass ~35-40% body mass → rib_center has highest weight

        Args:
            snout: Array of shape (N, 2) for snout position
            neck: Array of shape (N, 2) for neck position
            rib_center: Array of shape (N, 2) for rib center position
            hip_center: Array of shape (N, 2) for hip center position

        Returns:
            CoM trajectory of shape (N, 2)
        """
        from ..constants import (
            COM_WEIGHT_SNOUT,
            COM_WEIGHT_NECK,
            COM_WEIGHT_RIB_CENTER,
            COM_WEIGHT_HIP
        )

        weights = np.array([
            COM_WEIGHT_SNOUT,      # 0.10 - head/snout
            COM_WEIGHT_NECK,       # 0.15 - neck region
            COM_WEIGHT_RIB_CENTER, # 0.25 - thoracic (largest mass)
            COM_WEIGHT_HIP         # 0.20 - pelvic region
        ])

        com_trajectory = np.zeros_like(hip_center)

        for i in range(len(hip_center)):
            keypoints_available = [snout[i], neck[i], rib_center[i], hip_center[i]]

            # Check if all keypoints are valid
            if not any(np.isnan(kp).any() for kp in keypoints_available):
                points = np.array(keypoints_available)
                com_trajectory[i] = compute_center_of_mass(points, weights)
            else:
                # Fallback: use available keypoints with renormalized weights
                valid_points = []
                valid_weights = []

                if not np.isnan(snout[i]).any():
                    valid_points.append(snout[i])
                    valid_weights.append(COM_WEIGHT_SNOUT)
                if not np.isnan(neck[i]).any():
                    valid_points.append(neck[i])
                    valid_weights.append(COM_WEIGHT_NECK)
                if not np.isnan(rib_center[i]).any():
                    valid_points.append(rib_center[i])
                    valid_weights.append(COM_WEIGHT_RIB_CENTER)
                if not np.isnan(hip_center[i]).any():
                    valid_points.append(hip_center[i])
                    valid_weights.append(COM_WEIGHT_HIP)

                if len(valid_points) >= 2:
                    points = np.array(valid_points)
                    weights_array = np.array(valid_weights)
                    com_trajectory[i] = compute_center_of_mass(points, weights_array)
                else:
                    com_trajectory[i] = np.nan

        com_processed, _ = self.preprocess_keypoint_2d(com_trajectory)

        logger.info("[v1.2.0] Computed physics-based CoM trajectory with weighted average")
        logger.info(f"[v1.2.0] Weights: snout={COM_WEIGHT_SNOUT}, neck={COM_WEIGHT_NECK}, "
                   f"rib={COM_WEIGHT_RIB_CENTER}, hip={COM_WEIGHT_HIP}")

        return com_processed

    def compute_hip_center(self, hip_left: np.ndarray, hip_right: np.ndarray) -> np.ndarray:
        """
        Compute hip center from left and right hip positions.

        Args:
            hip_left: Array of shape (N, 2) for left hip
            hip_right: Array of shape (N, 2) for right hip

        Returns:
            Hip center trajectory of shape (N, 2)
        """
        hip_center = np.zeros_like(hip_left)

        for i in range(len(hip_left)):
            if not (np.isnan(hip_left[i]).any() or np.isnan(hip_right[i]).any()):
                points = np.array([hip_left[i], hip_right[i]])
                hip_center[i] = compute_center_of_mass(points)
            else:
                hip_center[i] = np.nan

        return hip_center

    def batch_preprocess_keypoints(self, keypoints_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Preprocess multiple keypoints in batch.

        Args:
            keypoints_dict: Dictionary mapping keypoint names to (N, 2) arrays

        Returns:
            Dictionary of preprocessed keypoints
        """
        logger.info(f"Preprocessing {len(keypoints_dict)} keypoints")

        preprocessed = {}
        for name, data in keypoints_dict.items():
            processed, metrics = self.preprocess_keypoint_2d(data)
            preprocessed[name] = processed

            if metrics['overall_completeness'] < DATA_COMPLETENESS_MEDIUM:
                logger.warning(
                    f"Low data completeness for {name}: {metrics['overall_completeness']*100:.1f}%"
                )

        logger.info("Batch preprocessing complete")

        return preprocessed

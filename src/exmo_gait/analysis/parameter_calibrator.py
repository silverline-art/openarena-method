"""Auto-calibration for step detection parameters (v1.3.2)"""
import numpy as np
import logging
from typing import Dict, List, Tuple
from .step_detector import StepDetector

logger = logging.getLogger(__name__)


class ParameterCalibrator:
    """
    Auto-calibrate step detection parameters to minimize cross-limb variance.

    Tests different prominence_multiplier values and selects the one that produces
    most consistent stride counts across all limbs.
    """

    def __init__(self, fps: float = 120.0):
        self.fps = fps

    def calibrate_prominence_multiplier(self,
                                      paw_trajectories: Dict[str, np.ndarray],
                                      walking_windows: List[Tuple[int, int]],
                                      test_values: List[float] = None) -> Dict:
        """
        Test different prominence_multiplier values and select optimal.

        Args:
            paw_trajectories: Dictionary mapping limb names to trajectories
            walking_windows: List of (start, end) walking window tuples
            test_values: List of prominence_multiplier values to test (default: [0.2..0.8])

        Returns:
            Dictionary with 'optimal_multiplier', 'stride_counts', and 'score'
        """
        if test_values is None:
            test_values = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

        logger.info(f"[Calibration] Testing {len(test_values)} prominence multiplier values...")

        results = []

        for multiplier in test_values:
            # Create detector with this multiplier
            detector = StepDetector(fps=self.fps, prominence_multiplier=multiplier)

            # Detect steps for all limbs
            step_results = detector.detect_all_limbs(paw_trajectories, walking_windows)

            # Extract stride counts
            stride_counts = [res['num_strides'] for res in step_results.values()]

            if len(stride_counts) == 0:
                continue

            # Compute consistency score (lower is better)
            mean_strides = np.mean(stride_counts)
            variance = np.var(stride_counts)
            cv = np.std(stride_counts) / mean_strides if mean_strides > 0 else 999

            # Penalize if cadence is implausible (mice: ~100-200 steps/min)
            total_duration = sum([(end - start) / self.fps for start, end in walking_windows])
            avg_cadence = (sum(stride_counts) / total_duration) * 60 if total_duration > 0 else 0

            cadence_penalty = 0
            if avg_cadence < 80 or avg_cadence > 250:
                cadence_penalty = abs(avg_cadence - 150) / 150  # Penalize deviation from 150 steps/min

            # Combined score: coefficient of variation + cadence penalty
            score = cv + cadence_penalty

            results.append({
                'multiplier': multiplier,
                'stride_counts': stride_counts,
                'mean_strides': mean_strides,
                'variance': variance,
                'cv': cv,
                'cadence': avg_cadence,
                'score': score
            })

            logger.debug(
                f"  multiplier={multiplier:.1f}: strides={stride_counts}, "
                f"mean={mean_strides:.1f}, CV={cv:.3f}, cadence={avg_cadence:.1f}, score={score:.3f}"
            )

        if len(results) == 0:
            logger.warning("[Calibration] No valid results, using default multiplier=0.5")
            return {'optimal_multiplier': 0.5, 'stride_counts': [], 'score': 999}

        # Select best result (lowest score)
        best = min(results, key=lambda x: x['score'])

        logger.info(
            f"[Calibration] Optimal multiplier={best['multiplier']:.1f}, "
            f"stride_counts={best['stride_counts']}, CV={best['cv']:.3f}, cadence={best['cadence']:.1f}"
        )

        return {
            'optimal_multiplier': best['multiplier'],
            'stride_counts': best['stride_counts'],
            'score': best['score'],
            'cv': best['cv'],
            'cadence': best['cadence']
        }

"""Data loading and synchronization for multi-view CSV files"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Tuple, List
from ..utils.validation import validate_frame_rate, validate_keypoints_present

logger = logging.getLogger(__name__)


class MultiViewDataLoader:
    """Load and synchronize 3-view pose estimation data"""

    REQUIRED_KEYPOINTS = [
        'snout', 'neck', 'rib_center', 'hip_R', 'hip_L', 'tail_base',
        'elbow_R', 'elbow_L', 'paw_FR', 'paw_FL', 'paw_RR', 'paw_RL'
    ]

    def __init__(self, expected_fps: float = 120.0):
        """
        Initialize data loader.

        Args:
            expected_fps: Expected frame rate (Hz)
        """
        self.expected_fps = expected_fps
        self.data = {}
        self.metadata = {}

    def load_csv(self, filepath: Path, view_name: str) -> Dict[str, np.ndarray]:
        """
        Load single CSV file with pose estimation data.

        Args:
            filepath: Path to CSV file
            view_name: Name of view ('top', 'side', 'bottom')

        Returns:
            Dictionary mapping keypoint names to (x, y, likelihood) arrays
        """
        logger.info(f"Loading {view_name} view from {filepath}")

        df = pd.read_csv(filepath, header=[0, 1, 2])

        scorer = df.columns[1][0]
        bodyparts = df.columns.get_level_values(1)
        coords = df.columns.get_level_values(2)

        unique_bodyparts = bodyparts.unique()
        unique_bodyparts = [bp for bp in unique_bodyparts if bp != 'bodyparts']

        keypoint_data = {}
        for bodypart in unique_bodyparts:
            try:
                x_col = (scorer, bodypart, 'x')
                y_col = (scorer, bodypart, 'y')
                likelihood_col = (scorer, bodypart, 'likelihood')

                x = df[x_col].values.astype(float)
                y = df[y_col].values.astype(float)
                likelihood = df[likelihood_col].values.astype(float)

                x[likelihood < 0.5] = np.nan
                y[likelihood < 0.5] = np.nan

                keypoint_data[bodypart] = {
                    'x': x,
                    'y': y,
                    'likelihood': likelihood
                }
            except KeyError as e:
                logger.warning(f"Could not load keypoint {bodypart}: {e}")
                continue

        n_frames = len(df)
        logger.info(f"Loaded {view_name} view: {n_frames} frames, {len(keypoint_data)} keypoints")

        return keypoint_data, n_frames

    def load_all_views(self,
                       top_path: Path,
                       side_path: Path,
                       bottom_path: Path) -> None:
        """
        Load all three views and synchronize.

        Args:
            top_path: Path to top view CSV
            side_path: Path to side view CSV
            bottom_path: Path to bottom view CSV
        """
        self.data['top'], n_top = self.load_csv(top_path, 'top')
        self.data['side'], n_side = self.load_csv(side_path, 'side')
        self.data['bottom'], n_bottom = self.load_csv(bottom_path, 'bottom')

        if not (n_top == n_side == n_bottom):
            logger.warning(
                f"Frame count mismatch: top={n_top}, side={n_side}, bottom={n_bottom}. "
                f"Using minimum: {min(n_top, n_side, n_bottom)}"
            )
            self.n_frames = min(n_top, n_side, n_bottom)
            self._truncate_to_min_frames()
        else:
            self.n_frames = n_top

        validate_frame_rate(self.expected_fps, self.expected_fps)

        self.metadata = {
            'n_frames': self.n_frames,
            'fps': self.expected_fps,
            'duration_sec': self.n_frames / self.expected_fps,
            'views': list(self.data.keys())
        }

        logger.info(f"Synchronized all views: {self.n_frames} frames, {self.metadata['duration_sec']:.2f} sec")

    def _truncate_to_min_frames(self) -> None:
        """Truncate all view data to minimum frame count"""
        for view in self.data:
            for keypoint in self.data[view]:
                for coord in ['x', 'y', 'likelihood']:
                    self.data[view][keypoint][coord] = \
                        self.data[view][keypoint][coord][:self.n_frames]

    def get_keypoint(self, view: str, keypoint: str) -> np.ndarray:
        """
        Get keypoint data as (N, 2) array of (x, y) coordinates.

        Args:
            view: View name ('top', 'side', 'bottom')
            keypoint: Keypoint name

        Returns:
            Array of shape (N, 2) with (x, y) coordinates
        """
        if view not in self.data:
            raise ValueError(f"View '{view}' not loaded")

        if keypoint not in self.data[view]:
            raise ValueError(f"Keypoint '{keypoint}' not found in {view} view")

        x = self.data[view][keypoint]['x']
        y = self.data[view][keypoint]['y']

        return np.column_stack([x, y])

    def get_keypoint_coord(self, view: str, keypoint: str, coord: str) -> np.ndarray:
        """
        Get single coordinate (x or y) for a keypoint.

        Args:
            view: View name
            keypoint: Keypoint name
            coord: 'x' or 'y'

        Returns:
            1D array of coordinate values
        """
        if view not in self.data:
            raise ValueError(f"View '{view}' not loaded")

        if keypoint not in self.data[view]:
            raise ValueError(f"Keypoint '{keypoint}' not found in {view} view")

        return self.data[view][keypoint][coord]

    def get_likelihood(self, view: str, keypoint: str) -> np.ndarray:
        """
        Get likelihood values for a keypoint.

        Args:
            view: View name
            keypoint: Keypoint name

        Returns:
            1D array of likelihood values
        """
        if view not in self.data:
            raise ValueError(f"View '{view}' not loaded")

        if keypoint not in self.data[view]:
            raise ValueError(f"Keypoint '{keypoint}' not found in {view} view")

        return self.data[view][keypoint]['likelihood']

    def get_available_keypoints(self, view: str) -> List[str]:
        """
        Get list of available keypoints for a view.

        Args:
            view: View name

        Returns:
            List of keypoint names
        """
        if view not in self.data:
            return []

        return list(self.data[view].keys())

    def validate_required_keypoints(self) -> bool:
        """
        Validate that all required keypoints are present in at least one view.

        Returns:
            True if validation passes
        """
        all_keypoints = set()
        for view in self.data:
            all_keypoints.update(self.data[view].keys())

        missing = [kp for kp in self.REQUIRED_KEYPOINTS if kp not in all_keypoints]

        if missing:
            logger.error(f"Missing required keypoints: {', '.join(missing)}")
            return False

        logger.info("All required keypoints present")
        return True

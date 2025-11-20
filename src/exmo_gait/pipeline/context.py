"""Pipeline context for state flow between stages.

This module defines the PipelineContext dataclass that carries state
through the pipeline stages, replacing the previous procedural approach
with explicit state management.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import logging


@dataclass
class PipelineContext:
    """Immutable state container for pipeline execution.

    Each stage receives a context and returns a new context with updates.
    This ensures explicit state flow and makes dependencies clear.

    Attributes:
        input_paths: Paths to input CSV files (top, side, bottom)
        output_dir: Directory for results
        config: Configuration dictionary
        logger: Logger instance

        # Stage outputs (populated during execution)
        loader: MultiViewDataLoader instance
        keypoints: Raw keypoint trajectories
        keypoints_preprocessed: Smoothed/scaled keypoints
        scale_factor: Spatial scaling factor (cm/pixel)
        com_trajectory: Center of mass trajectory
        walking_windows: Detected walking periods
        stationary_windows: Detected stationary periods
        step_results: Foot strike detection results
        gait_metrics: Computed gait metrics
        rom_metrics: Range of motion metrics
        aggregated_gait: Aggregated gait statistics
        aggregated_rom: Aggregated ROM statistics
        metadata: Pipeline metadata
        output_files: Generated output file paths
    """

    # Input parameters
    input_paths: Tuple[Path, Path, Path]  # (top, side, bottom)
    output_dir: Path
    config: Dict[str, Any]
    logger: logging.Logger

    # Stage outputs
    loader: Any = None
    keypoints: Dict[str, np.ndarray] = field(default_factory=dict)
    keypoints_preprocessed: Dict[str, np.ndarray] = field(default_factory=dict)
    scale_factor: float = 0.1
    scaling_diagnostics: Dict[str, Any] = field(default_factory=dict)
    com_trajectory: Optional[np.ndarray] = None
    walking_windows: List[Tuple[int, int]] = field(default_factory=list)
    stationary_windows: List[Tuple[int, int]] = field(default_factory=list)
    step_results: Dict[str, Any] = field(default_factory=dict)
    gait_metrics: Dict[str, Any] = field(default_factory=dict)
    rom_metrics: Dict[str, Any] = field(default_factory=dict)
    aggregated_gait: Dict[str, Any] = field(default_factory=dict)
    aggregated_rom: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    output_files: Dict[str, Any] = field(default_factory=dict)

    def get_global_settings(self) -> Dict[str, Any]:
        """Extract global_settings from config with safe defaults."""
        return self.config.get('global_settings', {}) if self.config else {}

    def update(self, **kwargs) -> 'PipelineContext':
        """Create new context with updated fields.

        This maintains immutability by returning a new instance rather
        than modifying in place.

        Args:
            **kwargs: Fields to update

        Returns:
            New PipelineContext with updated fields
        """
        # Create a shallow copy with updates
        import copy
        new_ctx = copy.copy(self)
        for key, value in kwargs.items():
            if hasattr(new_ctx, key):
                setattr(new_ctx, key, value)
            else:
                raise AttributeError(f"PipelineContext has no attribute '{key}'")
        return new_ctx

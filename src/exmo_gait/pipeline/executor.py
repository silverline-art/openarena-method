"""Pipeline executor orchestrates stage execution.

This module contains the PipelineExecutor class that replaces the
monolithic run_pipeline function with a clean pipeline pattern.
"""

from pathlib import Path
from typing import Dict, List, Any
import logging
from datetime import datetime

from .context import PipelineContext
from .stages import (
    ConfigurationStage,
    DataLoadingStage,
    SpatialScalingStage,
    PreprocessingStage,
    PhaseDetectionStage,
    MetricsComputationStage,
    StatisticsAggregationStage,
    ExportStage
)


class PipelineExecutor:
    """Orchestrates pipeline stage execution.

    The executor maintains a list of stages and executes them sequentially,
    passing context between stages. This replaces the 232-line procedural
    run_pipeline function with a clean, testable architecture.

    Attributes:
        stages: Ordered list of pipeline stages to execute
    """

    def __init__(self):
        """Initialize executor with default stages."""
        self.stages = [
            ConfigurationStage(),
            DataLoadingStage(),
            SpatialScalingStage(),
            PreprocessingStage(),
            PhaseDetectionStage(),
            MetricsComputationStage(),
            StatisticsAggregationStage(),
            ExportStage()
        ]

    def execute(
        self,
        top_path: Path,
        side_path: Path,
        bottom_path: Path,
        output_dir: Path,
        verbose: bool = False,
        config: Dict = None
    ) -> Dict[str, Any]:
        """Execute complete gait analysis pipeline.

        This method replaces the original run_pipeline function with a
        clean stage-based architecture. Each stage is responsible for
        one aspect of the analysis.

        Args:
            top_path: Path to top view CSV
            side_path: Path to side view CSV
            bottom_path: Path to bottom view CSV
            output_dir: Output directory for results
            verbose: Enable verbose logging
            config: Optional configuration dictionary

        Returns:
            Dictionary with analysis results and metadata:
            {
                'status': 'success' or 'error',
                'metadata': {...},
                'output_files': {...},
                'error': '...' (if status == 'error')
            }
        """
        logger = self._setup_logging(output_dir, verbose)

        # Initialize pipeline context
        ctx = PipelineContext(
            input_paths=(top_path, side_path, bottom_path),
            output_dir=output_dir,
            config=config or {},
            logger=logger
        )

        try:
            # Execute stages sequentially
            for stage in self.stages:
                stage_name = stage.__class__.__name__
                logger.debug(f"Executing {stage_name}")
                ctx = stage.execute(ctx)

            return {
                'status': 'success',
                'metadata': ctx.metadata,
                'output_files': ctx.output_files
            }

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'error': str(e)
            }

    def _setup_logging(self, output_dir: Path, verbose: bool = False) -> logging.Logger:
        """Setup logging configuration.

        Args:
            output_dir: Directory for log files
            verbose: Enable debug level logging

        Returns:
            Logger instance
        """
        log_dir = output_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)

        log_file = log_dir / f'analysis_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'

        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        return logging.getLogger(__name__)

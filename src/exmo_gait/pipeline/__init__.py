"""Pipeline architecture for EXMO gait analysis.

This package contains the modular pipeline stages that replace the monolithic
run_pipeline function. Each stage has a single responsibility and clear
input/output contracts.
"""

from .executor import PipelineExecutor
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

__all__ = [
    'PipelineExecutor',
    'ConfigurationStage',
    'DataLoadingStage',
    'SpatialScalingStage',
    'PreprocessingStage',
    'PhaseDetectionStage',
    'MetricsComputationStage',
    'StatisticsAggregationStage',
    'ExportStage'
]

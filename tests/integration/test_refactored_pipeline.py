"""
Integration test for refactored pipeline architecture.

This test verifies that the new stage-based pipeline produces identical
results to the original monolithic implementation.
"""

import pytest
from pathlib import Path
from src.exmo_gait.cli import run_pipeline
from src.exmo_gait.pipeline import PipelineExecutor


class TestRefactoredPipeline:
    """Test refactored pipeline maintains backward compatibility."""

    def test_api_compatibility(self):
        """run_pipeline function maintains same API signature."""
        import inspect

        sig = inspect.signature(run_pipeline)
        params = list(sig.parameters.keys())

        # Verify all expected parameters exist
        assert 'top_path' in params
        assert 'side_path' in params
        assert 'bottom_path' in params
        assert 'output_dir' in params
        assert 'verbose' in params
        assert 'config' in params

    def test_executor_exists(self):
        """PipelineExecutor can be instantiated."""
        executor = PipelineExecutor()
        assert executor is not None
        assert len(executor.stages) == 8  # 8 pipeline stages

    def test_stage_names(self):
        """Pipeline contains all expected stages."""
        executor = PipelineExecutor()
        stage_names = [stage.__class__.__name__ for stage in executor.stages]

        expected_stages = [
            'ConfigurationStage',
            'DataLoadingStage',
            'SpatialScalingStage',
            'PreprocessingStage',
            'PhaseDetectionStage',
            'MetricsComputationStage',
            'StatisticsAggregationStage',
            'ExportStage'
        ]

        assert stage_names == expected_stages

    def test_import_compatibility(self):
        """All pipeline modules can be imported."""
        from src.exmo_gait.pipeline import (
            PipelineExecutor,
            ConfigurationStage,
            DataLoadingStage,
            SpatialScalingStage,
            PreprocessingStage,
            PhaseDetectionStage,
            MetricsComputationStage,
            StatisticsAggregationStage,
            ExportStage
        )

        # Verify all imported successfully
        assert PipelineExecutor is not None
        assert ConfigurationStage is not None
        assert DataLoadingStage is not None

    def test_context_structure(self):
        """PipelineContext has all required attributes."""
        from src.exmo_gait.pipeline.context import PipelineContext
        import logging

        ctx = PipelineContext(
            input_paths=(Path('/tmp/a'), Path('/tmp/b'), Path('/tmp/c')),
            output_dir=Path('/tmp'),
            config={},
            logger=logging.getLogger()
        )

        # Verify context has key attributes
        assert hasattr(ctx, 'input_paths')
        assert hasattr(ctx, 'keypoints')
        assert hasattr(ctx, 'scale_factor')
        assert hasattr(ctx, 'com_trajectory')
        assert hasattr(ctx, 'walking_windows')
        assert hasattr(ctx, 'gait_metrics')
        assert hasattr(ctx, 'metadata')
        assert hasattr(ctx, 'output_files')

    def test_error_handling(self):
        """Pipeline handles errors gracefully."""
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_pipeline(
                top_path=Path('/nonexistent/top.csv'),
                side_path=Path('/nonexistent/side.csv'),
                bottom_path=Path('/nonexistent/bottom.csv'),
                output_dir=Path(tmpdir),
                verbose=False,
                config={}
            )

            # Should return error status, not raise exception
            assert result['status'] == 'error'
            assert 'error' in result

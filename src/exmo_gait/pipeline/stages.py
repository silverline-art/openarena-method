"""Pipeline stages for EXMO gait analysis.

Each stage is a focused, cohesive unit with a single responsibility.
Stages follow the pattern: receive context -> process -> return updated context.
"""

import numpy as np
from typing import Dict
from datetime import datetime

from ..core.data_loader import MultiViewDataLoader
from ..core.preprocessor import DataPreprocessor
from ..analysis.phase_detector import PhaseDetector
from ..analysis.step_detector import StepDetector
from ..analysis.metrics_computer import GaitMetricsComputer, ROMMetricsComputer
from ..statistics.aggregator import StatisticsAggregator
from ..export.xlsx_exporter import XLSXExporter
from ..export.visualizer import DashboardVisualizer
from ..export.visualizer_enhanced import EnhancedDashboardVisualizer
from ..constants import (
    FPS_DEFAULT,
    DEFAULT_MOUSE_BODY_LENGTH_CM,
    LEGACY_SPINE_LENGTH_CM,
    PLOT_DPI
)

from .context import PipelineContext


class ConfigurationStage:
    """Validate configuration and log pipeline settings.

    Responsibility: Extract and validate global settings from config,
    log which v1.1.0 vs v1.2.0 methods will be used.
    """

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Log pipeline configuration.

        Args:
            ctx: Pipeline context with config

        Returns:
            Context with validated configuration
        """
        gs = ctx.get_global_settings()

        ctx.logger.info("=" * 80)
        ctx.logger.info("EXMO Animal Gait Analysis Pipeline - Starting")
        ctx.logger.info("=" * 80)

        # Extract configuration flags
        scaling_method = gs.get('scaling_method', 'spine_only')
        use_adaptive_smoothing = gs.get('smoothing_adaptive', False)
        use_ema_velocity = gs.get('velocity_smoothing_method', 'savgol') == 'ema'
        use_hybrid_threshold = gs.get('use_hybrid_threshold', False)
        use_3d_com = gs.get('use_3d_com', False)
        include_ci = gs.get('aggregation_include_ci', False)

        ctx.logger.info("Pipeline Configuration:")

        if scaling_method == 'full_body':
            ctx.logger.info("  - Scaling: v1.2.0 full-body method (expected +20-25% distance accuracy)")
        else:
            ctx.logger.info("  - Scaling: v1.1.0 spine-only method (legacy mode)")

        if use_adaptive_smoothing:
            ctx.logger.info("  - Smoothing: v1.2.0 adaptive method (expected +15-25% peak preservation)")
        else:
            ctx.logger.info("  - Smoothing: v1.1.0 fixed window method (legacy mode)")

        if use_ema_velocity:
            ctx.logger.info("  - Velocity: v1.2.0 EMA smoothing (expected +10-20% velocity accuracy)")
        else:
            ctx.logger.info("  - Velocity: v1.1.0 Savitzky-Golay method (legacy mode)")

        if use_hybrid_threshold:
            ctx.logger.info("  - Phase Detection: v1.2.0 hybrid threshold (expected +10-20% walking detection)")
        else:
            ctx.logger.info("  - Phase Detection: v1.1.0 MAD-only threshold (legacy mode)")

        if use_3d_com:
            ctx.logger.info("  - COM Calculation: v1.2.0 3D method (expected +10-20% speed accuracy)")
        else:
            ctx.logger.info("  - COM Calculation: v1.1.0 2D method (legacy mode)")

        if include_ci:
            ctx.logger.info("  - Statistics: v1.2.0 enhanced with CI (includes 95% confidence intervals)")
        else:
            ctx.logger.info("  - Statistics: v1.1.0 basic statistics (legacy mode)")

        return ctx


class DataLoadingStage:
    """Load and validate multi-view CSV data.

    Responsibility: Load CSV files, validate required keypoints,
    extract trajectories with view priority.
    """

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Load multi-view data and extract keypoints.

        Args:
            ctx: Pipeline context with input paths

        Returns:
            Context with loader and keypoints populated
        """
        gs = ctx.get_global_settings()
        top_path, side_path, bottom_path = ctx.input_paths

        ctx.logger.info("Step 1/10: Loading multi-view data")
        loader = MultiViewDataLoader(expected_fps=gs.get('fps', FPS_DEFAULT))
        loader.load_all_views(top_path, side_path, bottom_path)

        if not loader.validate_required_keypoints():
            raise ValueError("Required keypoints missing from data")

        ctx.logger.info("Step 2/10: Extracting keypoint trajectories")
        view_priority = {
            'top': ['snout', 'neck', 'tail_base', 'rib_center'],
            'bottom': ['paw_RR', 'paw_RL', 'paw_FR', 'paw_FL'],
            'side': ['hip_R', 'hip_L', 'hip_center', 'knee_R', 'knee_L',
                    'elbow_R', 'elbow_L', 'shoulder_R', 'shoulder_L']
        }

        keypoints = {}
        for view, kp_list in view_priority.items():
            for kp in kp_list:
                if kp in loader.get_available_keypoints(view):
                    keypoints[kp] = loader.get_keypoint(view, kp)

        ctx.logger.info(f"Extracted {len(keypoints)} keypoint trajectories")

        return ctx.update(loader=loader, keypoints=keypoints)


class SpatialScalingStage:
    """Compute spatial scaling factor (pixels to centimeters).

    Responsibility: Calculate scale factor using either v1.1.0 spine-only
    or v1.2.0 full-body method based on configuration.
    """

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Compute scale factor from snout-tail distance.

        Args:
            ctx: Pipeline context with keypoints

        Returns:
            Context with scale_factor populated
        """
        gs = ctx.get_global_settings()
        scaling_method = gs.get('scaling_method', 'spine_only')

        ctx.logger.info("Step 3/10: Computing spatial scaling")

        preprocessor = DataPreprocessor(
            smoothing_window=gs.get('smoothing_window', 11),
            smoothing_poly=gs.get('smoothing_poly', 3),
            outlier_threshold=gs.get('outlier_threshold', 3.0),
            max_interpolation_gap=gs.get('max_interpolation_gap', 5)
        )

        snout = ctx.keypoints.get('snout')
        tail_base = ctx.keypoints.get('tail_base')

        if snout is not None and tail_base is not None:
            if scaling_method == 'full_body':
                # v1.2.0: Full-body scaling with likelihood filtering
                expected_body_length = gs.get('expected_body_length_cm', DEFAULT_MOUSE_BODY_LENGTH_CM)
                min_likelihood = gs.get('scaling_min_likelihood', 0.9)
                tolerance = gs.get('scaling_tolerance', 0.25)

                scale_factor, diagnostics = preprocessor.compute_scale_factor_v2(
                    snout, tail_base,
                    expected_body_length_cm=expected_body_length,
                    min_likelihood=min_likelihood,
                    tolerance=tolerance
                )
                ctx.logger.info(f"[v1.2.0] Full-body scaling: {scale_factor:.6f} cm/pixel "
                              f"(used {diagnostics['frames_used']}/{diagnostics['frames_total']} frames)")

                return ctx.update(
                    scale_factor=scale_factor,
                    scaling_diagnostics=diagnostics
                )
            else:
                # v1.1.0: Legacy spine-only scaling
                known_distance_cm = gs.get('expected_body_length_cm', LEGACY_SPINE_LENGTH_CM)
                scale_factor = preprocessor.compute_scale_factor(
                    snout, tail_base,
                    known_distance_cm=known_distance_cm
                )
                ctx.logger.info(f"[v1.1.0] Spine-only scaling: {scale_factor:.6f} cm/pixel")

                return ctx.update(scale_factor=scale_factor)
        else:
            ctx.logger.warning("Could not compute scale factor, using default")
            return ctx.update(scale_factor=0.1)


class PreprocessingStage:
    """Preprocess trajectories: smoothing, scaling, COM calculation.

    Responsibility: Apply smoothing (adaptive or fixed), convert to cm,
    calculate center of mass (2D or 3D).
    """

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Preprocess keypoints and compute COM.

        Args:
            ctx: Pipeline context with keypoints and scale_factor

        Returns:
            Context with preprocessed keypoints and COM trajectory
        """
        gs = ctx.get_global_settings()
        use_adaptive_smoothing = gs.get('smoothing_adaptive', False)
        use_3d_com = gs.get('use_3d_com', False)

        preprocessor = DataPreprocessor(
            smoothing_window=gs.get('smoothing_window', 11),
            smoothing_poly=gs.get('smoothing_poly', 3),
            outlier_threshold=gs.get('outlier_threshold', 3.0),
            max_interpolation_gap=gs.get('max_interpolation_gap', 5)
        )
        preprocessor.scale_factor = ctx.scale_factor

        # Apply smoothing
        if use_adaptive_smoothing:
            # v1.2.0: Adaptive smoothing
            from ..utils.signal_processing import smooth_trajectory_adaptive

            keypoints_preprocessed = {}
            for kp_name, kp_data in ctx.keypoints.items():
                data_completeness = np.sum(~np.isnan(kp_data[:, 0])) / len(kp_data)

                smoothed_x = smooth_trajectory_adaptive(
                    kp_data[:, 0],
                    data_completeness,
                    window_size_base=gs.get('smoothing_window', 7),
                    polyorder=gs.get('smoothing_poly', 3)
                )
                smoothed_y = smooth_trajectory_adaptive(
                    kp_data[:, 1],
                    data_completeness,
                    window_size_base=gs.get('smoothing_window', 7),
                    polyorder=gs.get('smoothing_poly', 3)
                )

                keypoints_preprocessed[kp_name] = np.column_stack([smoothed_x, smoothed_y])

            ctx.logger.info("[v1.2.0] Applied adaptive smoothing based on data quality")
        else:
            # v1.1.0: Fixed window smoothing
            keypoints_preprocessed = preprocessor.batch_preprocess_keypoints(ctx.keypoints)
            ctx.logger.info("[v1.1.0] Applied fixed window smoothing")

        # Convert to cm
        for kp_name in keypoints_preprocessed:
            keypoints_preprocessed[kp_name] = preprocessor.convert_to_cm(
                keypoints_preprocessed[kp_name]
            )

        # Compute center of mass
        ctx.logger.info("Step 4/10: Computing center of mass")

        if use_3d_com:
            # v1.2.0: 3D COM
            gait_computer = GaitMetricsComputer(fps=gs.get('fps', FPS_DEFAULT))

            top_keypoints = {k: v for k, v in keypoints_preprocessed.items()
                           if k in ['snout', 'neck', 'tail_base', 'rib_center', 'spine1', 'spine2', 'spine3']}
            side_keypoints = {k: v for k, v in keypoints_preprocessed.items()
                            if k in ['hip_R', 'hip_L', 'shoulder_R', 'shoulder_L', 'knee_R', 'knee_L']}

            if top_keypoints and side_keypoints:
                com_weights = gs.get('com_weights', None)
                com_trajectory = gait_computer.compute_com_3d(
                    top_keypoints,
                    side_keypoints,
                    weights=com_weights
                )
                ctx.logger.info("[v1.2.0] Computed 3D COM from TOP+SIDE views")
            else:
                ctx.logger.warning("Insufficient keypoints for 3D COM, falling back to 2D")
                use_3d_com = False

        if not use_3d_com:
            # v1.1.0: 2D COM
            if 'hip_R' in keypoints_preprocessed and 'hip_L' in keypoints_preprocessed:
                hip_center = preprocessor.compute_hip_center(
                    keypoints_preprocessed['hip_L'],
                    keypoints_preprocessed['hip_R']
                )
            else:
                ctx.logger.warning("Hip keypoints not available, using approximation")
                hip_center = keypoints_preprocessed.get('rib_center',
                                                       np.zeros((ctx.loader.n_frames, 2)))

            rib_center = keypoints_preprocessed.get('rib_center', hip_center)
            com_trajectory = preprocessor.compute_com_trajectory(hip_center, rib_center)
            ctx.logger.info("[v1.1.0] Computed 2D COM trajectory")

        return ctx.update(
            keypoints_preprocessed=keypoints_preprocessed,
            com_trajectory=com_trajectory
        )


class PhaseDetectionStage:
    """Detect walking and stationary phases, identify foot strikes.

    Responsibility: Detect walking/stationary windows using phase detector,
    identify foot strikes using step detector.
    """

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Detect phases and foot strikes.

        Args:
            ctx: Pipeline context with COM trajectory

        Returns:
            Context with walking/stationary windows and step results
        """
        gs = ctx.get_global_settings()
        use_hybrid_threshold = gs.get('use_hybrid_threshold', False)

        ctx.logger.info("Step 5/10: Detecting phases")

        phase_detector = PhaseDetector(
            fps=gs.get('fps', FPS_DEFAULT),
            stationary_mad_threshold=gs.get('stationary_mad_threshold', 1.5),
            walking_mad_threshold=gs.get('walking_mad_threshold', 2.0),
            min_walking_duration=gs.get('min_walking_duration', 0.3),
            min_stationary_duration=gs.get('min_stationary_duration', 0.25),
            use_hybrid_threshold=use_hybrid_threshold,
            adaptive_percentile=gs.get('adaptive_percentile', 75),
            min_threshold_px_per_frame=gs.get('min_threshold_px_per_frame', 1.0)
        )

        walking_windows = phase_detector.detect_walking_windows(ctx.com_trajectory)
        stationary_windows = phase_detector.detect_stationary_windows(ctx.com_trajectory)

        ctx.logger.info("Step 6/10: Detecting foot strikes")
        step_detector = StepDetector(
            fps=gs.get('fps', FPS_DEFAULT),
            min_stride_duration=gs.get('min_stride_duration', 0.1),
            max_stride_duration=gs.get('max_stride_duration', 1.0),
            prominence_multiplier=gs.get('prominence_multiplier', 0.5)
        )

        paw_trajectories = {
            kp: ctx.keypoints_preprocessed[kp]
            for kp in ['paw_RR', 'paw_RL', 'paw_FR', 'paw_FL']
            if kp in ctx.keypoints_preprocessed
        }

        step_results = step_detector.detect_all_limbs(paw_trajectories, walking_windows)

        return ctx.update(
            walking_windows=walking_windows,
            stationary_windows=stationary_windows,
            step_results=step_results
        )


class MetricsComputationStage:
    """Compute gait and range of motion metrics.

    Responsibility: Calculate gait metrics (stride, velocity, etc.) and
    ROM metrics (joint angles, limb movements).
    """

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Compute all metrics.

        Args:
            ctx: Pipeline context with step results and trajectories

        Returns:
            Context with gait_metrics and rom_metrics populated
        """
        gs = ctx.get_global_settings()
        use_ema_velocity = gs.get('velocity_smoothing_method', 'savgol') == 'ema'

        ctx.logger.info("Step 7/10: Computing gait metrics")

        gait_computer = GaitMetricsComputer(fps=gs.get('fps', FPS_DEFAULT))

        if use_ema_velocity:
            ema_alpha = gs.get('velocity_ema_alpha', 0.35)
            gait_computer.velocity_smoothing_method = 'ema'
            gait_computer.ema_alpha = ema_alpha
            ctx.logger.info(f"[v1.2.0] Using EMA velocity smoothing (alpha={ema_alpha})")
        else:
            gait_computer.velocity_smoothing_method = 'savgol'
            ctx.logger.info("[v1.1.0] Using Savitzky-Golay velocity smoothing")

        paw_trajectories = {
            kp: ctx.keypoints_preprocessed[kp]
            for kp in ['paw_RR', 'paw_RL', 'paw_FR', 'paw_FL']
            if kp in ctx.keypoints_preprocessed
        }

        gait_metrics = gait_computer.compute_all_gait_metrics(
            ctx.step_results,
            paw_trajectories,
            ctx.com_trajectory,
            ctx.walking_windows
        )

        ctx.logger.info("Step 8/10: Computing range of motion metrics")
        rom_computer = ROMMetricsComputer(fps=gs.get('fps', FPS_DEFAULT))
        rom_metrics = rom_computer.compute_all_rom_metrics(
            ctx.keypoints_preprocessed,
            ctx.com_trajectory
        )

        return ctx.update(gait_metrics=gait_metrics, rom_metrics=rom_metrics)


class StatisticsAggregationStage:
    """Aggregate metrics into summary statistics.

    Responsibility: Compute summary statistics (mean, std, CI) from
    per-stride metrics using basic or enhanced methods.
    """

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Aggregate statistics.

        Args:
            ctx: Pipeline context with gait_metrics and rom_metrics

        Returns:
            Context with aggregated statistics
        """
        gs = ctx.get_global_settings()
        include_ci = gs.get('aggregation_include_ci', False)

        ctx.logger.info("Step 9/10: Aggregating statistics")

        aggregator = StatisticsAggregator()

        if include_ci:
            # v1.2.0: Enhanced statistics
            ci_percentile = gs.get('aggregation_ci_percentile', 95)
            trim_percent = gs.get('aggregation_trim_percent', 5)

            aggregator.use_enhanced_stats = True
            aggregator.ci_percentile = ci_percentile
            aggregator.trim_percent = trim_percent

            ctx.logger.info(f"[v1.2.0] Using enhanced statistics (CI={ci_percentile}%, trim={trim_percent}%)")
        else:
            aggregator.use_enhanced_stats = False
            ctx.logger.info("[v1.1.0] Using basic statistics")

        aggregated_gait = aggregator.aggregate_gait_metrics(ctx.gait_metrics)
        aggregated_rom = aggregator.aggregate_rom_metrics(ctx.rom_metrics)

        return ctx.update(
            aggregated_gait=aggregated_gait,
            aggregated_rom=aggregated_rom
        )


class ExportStage:
    """Export results to Excel, plots, and JSON.

    Responsibility: Generate all output files (XLSX, PNG plots, JSON metadata),
    save intermediate data.
    """

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Export all results.

        Args:
            ctx: Pipeline context with all computed metrics

        Returns:
            Context with output_files and metadata populated
        """
        gs = ctx.get_global_settings()

        ctx.logger.info("Step 10/10: Exporting results")

        # Build metadata
        top_path, side_path, bottom_path = ctx.input_paths

        scaling_method = gs.get('scaling_method', 'spine_only')
        use_adaptive_smoothing = gs.get('smoothing_adaptive', False)
        use_ema_velocity = gs.get('velocity_smoothing_method', 'savgol') == 'ema'
        use_hybrid_threshold = gs.get('use_hybrid_threshold', False)
        use_3d_com = gs.get('use_3d_com', False)
        include_ci = gs.get('aggregation_include_ci', False)

        metadata = {
            'analysis_date': datetime.now().isoformat(),
            'top_view': str(top_path),
            'side_view': str(side_path),
            'bottom_view': str(bottom_path),
            'fps': gs.get('fps', FPS_DEFAULT),
            'n_frames': ctx.loader.n_frames,
            'duration_sec': ctx.loader.n_frames / gs.get('fps', FPS_DEFAULT),
            'scale_factor_cm_per_pixel': ctx.scale_factor,
            'n_walking_windows': len(ctx.walking_windows),
            'n_stationary_windows': len(ctx.stationary_windows),
            'pipeline_version': 'v1.2.0' if any([
                scaling_method == 'full_body',
                use_adaptive_smoothing,
                use_ema_velocity,
                use_hybrid_threshold,
                use_3d_com,
                include_ci
            ]) else 'v1.1.0',
            'methods_used': {
                'scaling': scaling_method,
                'adaptive_smoothing': use_adaptive_smoothing,
                'ema_velocity': use_ema_velocity,
                'hybrid_threshold': use_hybrid_threshold,
                '3d_com': use_3d_com,
                'enhanced_stats': include_ci
            }
        }

        # Export Excel
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        xlsx_filename = f"Gait_Analysis_{timestamp}.xlsx"
        xlsx_path = ctx.output_dir / xlsx_filename

        aggregator = StatisticsAggregator()
        summary_data = aggregator.create_summary_table(
            ctx.aggregated_gait,
            ctx.aggregated_rom
        )

        exporter = XLSXExporter(xlsx_path)
        exporter.export(
            summary_data,
            ctx.aggregated_gait,
            ctx.aggregated_rom,
            ctx.step_results,
            ctx.walking_windows,
            metadata
        )

        paw_trajectories = {
            kp: ctx.keypoints_preprocessed[kp]
            for kp in ['paw_RR', 'paw_RL', 'paw_FR', 'paw_FL']
            if kp in ctx.keypoints_preprocessed
        }

        exporter.save_intermediate_data(
            ctx.com_trajectory,
            paw_trajectories,
            ctx.output_dir
        )

        # Generate plots
        use_enhanced = gs.get('use_enhanced_plots', False)
        plot_dpi = gs.get('plot_dpi', PLOT_DPI)

        if use_enhanced:
            visualizer = EnhancedDashboardVisualizer(
                ctx.output_dir,
                dpi=plot_dpi,
                marker_size=gs.get('plot_marker_size', 60),
                font_scale=gs.get('plot_font_scale', 1.0),
                annotate_median=gs.get('plot_annotate_median', True),
                add_reference_bands=gs.get('plot_reference_bands', True)
            )
        else:
            visualizer = DashboardVisualizer(ctx.output_dir, dpi=plot_dpi)

        plot_paths = visualizer.generate_all_dashboards(
            ctx.gait_metrics,
            ctx.rom_metrics,
            ctx.aggregated_gait,
            ctx.aggregated_rom
        )

        output_files = {
            'xlsx': str(xlsx_path),
            'plots': [str(p) for p in plot_paths]
        }

        ctx.logger.info("=" * 80)
        ctx.logger.info("Analysis complete!")
        ctx.logger.info(f"Results saved to: {ctx.output_dir}")
        ctx.logger.info(f"  - Excel: {xlsx_filename}")
        ctx.logger.info(f"  - Plots: {len(plot_paths)} PNG files")
        ctx.logger.info(f"  - Pipeline Version: {metadata['pipeline_version']}")
        ctx.logger.info("=" * 80)

        return ctx.update(metadata=metadata, output_files=output_files)

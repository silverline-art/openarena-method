"""Command-line interface for EXMO gait analyzer"""
import argparse
import logging
import json
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict

from .core.data_loader import MultiViewDataLoader
from .core.preprocessor import DataPreprocessor
from .analysis.phase_detector import PhaseDetector
from .analysis.step_detector import StepDetector
from .analysis.metrics_computer import GaitMetricsComputer, ROMMetricsComputer
from .statistics.aggregator import StatisticsAggregator
from .export.xlsx_exporter import XLSXExporter
from .export.visualizer import DashboardVisualizer
from .export.visualizer_enhanced import EnhancedDashboardVisualizer


def setup_logging(output_dir: Path, verbose: bool = False):
    """Setup logging configuration"""
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


def run_pipeline(top_path: Path,
                side_path: Path,
                bottom_path: Path,
                output_dir: Path,
                verbose: bool = False,
                config: Dict = None) -> Dict:
    """
    Run complete gait analysis pipeline.

    Args:
        top_path: Path to top view CSV
        side_path: Path to side view CSV
        bottom_path: Path to bottom view CSV
        output_dir: Output directory
        verbose: Enable verbose logging
        config: Optional configuration dictionary with global_settings

    Returns:
        Dictionary with analysis results and metadata
    """
    # Get global settings from config or use defaults
    if config and 'global_settings' in config:
        gs = config['global_settings']
    else:
        gs = {}
    logger = setup_logging(output_dir, verbose)
    logger.info("=" * 80)
    logger.info("EXMO Animal Gait Analysis Pipeline - Starting")
    logger.info("=" * 80)

    results = {}

    try:
        logger.info("Step 1/10: Loading multi-view data")
        loader = MultiViewDataLoader(expected_fps=120.0)
        loader.load_all_views(top_path, side_path, bottom_path)

        if not loader.validate_required_keypoints():
            raise ValueError("Required keypoints missing from data")

        logger.info("Step 2/10: Extracting keypoint trajectories")
        view_priority = {'top': ['snout', 'neck', 'tail_base', 'rib_center'],
                        'bottom': ['paw_RR', 'paw_RL', 'paw_FR', 'paw_FL'],
                        'side': ['hip_R', 'hip_L', 'hip_center', 'knee_R', 'knee_L',
                                'elbow_R', 'elbow_L', 'shoulder_R', 'shoulder_L']}

        keypoints = {}
        for view, kp_list in view_priority.items():
            for kp in kp_list:
                if kp in loader.get_available_keypoints(view):
                    keypoints[kp] = loader.get_keypoint(view, kp)

        logger.info(f"Extracted {len(keypoints)} keypoint trajectories")

        logger.info("Step 3/10: Preprocessing data")
        preprocessor = DataPreprocessor(
            smoothing_window=gs.get('smoothing_window', 11),
            smoothing_poly=gs.get('smoothing_poly', 3),
            outlier_threshold=gs.get('outlier_threshold', 3.0),
            max_interpolation_gap=gs.get('max_interpolation_gap', 5)
        )

        snout = keypoints.get('snout')
        tail_base = keypoints.get('tail_base')
        if snout is not None and tail_base is not None:
            scale_factor = preprocessor.compute_scale_factor(snout, tail_base, known_distance_cm=8.0)
        else:
            logger.warning("Could not compute scale factor, using default")
            scale_factor = 0.1

        keypoints_preprocessed = preprocessor.batch_preprocess_keypoints(keypoints)

        for kp_name in keypoints_preprocessed:
            keypoints_preprocessed[kp_name] = preprocessor.convert_to_cm(
                keypoints_preprocessed[kp_name]
            )

        logger.info("Step 4/10: Computing center of mass")
        if 'hip_R' in keypoints_preprocessed and 'hip_L' in keypoints_preprocessed:
            hip_center = preprocessor.compute_hip_center(
                keypoints_preprocessed['hip_L'],
                keypoints_preprocessed['hip_R']
            )
        else:
            logger.warning("Hip keypoints not available, using approximation")
            hip_center = keypoints_preprocessed.get('rib_center',
                                                   np.zeros((loader.n_frames, 2)))

        rib_center = keypoints_preprocessed.get('rib_center', hip_center)
        com_trajectory = preprocessor.compute_com_trajectory(hip_center, rib_center)

        logger.info("Step 5/10: Detecting phases")
        phase_detector = PhaseDetector(
            fps=gs.get('fps', 120.0),
            stationary_mad_threshold=gs.get('stationary_mad_threshold', 1.5),
            walking_mad_threshold=gs.get('walking_mad_threshold', 2.0),
            min_walking_duration=gs.get('min_walking_duration', 0.3),
            min_stationary_duration=gs.get('min_stationary_duration', 0.25)
        )

        walking_windows = phase_detector.detect_walking_windows(com_trajectory)
        stationary_windows = phase_detector.detect_stationary_windows(com_trajectory)

        results['walking_windows'] = walking_windows
        results['stationary_windows'] = stationary_windows

        logger.info("Step 6/10: Detecting foot strikes")
        step_detector = StepDetector(
            fps=gs.get('fps', 120.0),
            min_stride_duration=gs.get('min_stride_duration', 0.1),
            max_stride_duration=gs.get('max_stride_duration', 1.0),
            prominence_multiplier=gs.get('prominence_multiplier', 0.5)
        )

        paw_trajectories = {
            kp: keypoints_preprocessed[kp]
            for kp in ['paw_RR', 'paw_RL', 'paw_FR', 'paw_FL']
            if kp in keypoints_preprocessed
        }

        step_results = step_detector.detect_all_limbs(paw_trajectories, walking_windows)
        results['step_results'] = step_results

        logger.info("Step 7/10: Computing gait metrics")
        gait_computer = GaitMetricsComputer(fps=120.0)
        gait_metrics = gait_computer.compute_all_gait_metrics(
            step_results,
            paw_trajectories,
            com_trajectory,
            walking_windows
        )
        results['gait_metrics'] = gait_metrics

        logger.info("Step 8/10: Computing range of motion metrics")
        rom_computer = ROMMetricsComputer(fps=120.0)
        rom_metrics = rom_computer.compute_all_rom_metrics(
            keypoints_preprocessed,
            com_trajectory
        )
        results['rom_metrics'] = rom_metrics

        logger.info("Step 9/10: Aggregating statistics")
        aggregator = StatisticsAggregator()
        aggregated_gait = aggregator.aggregate_gait_metrics(gait_metrics)
        aggregated_rom = aggregator.aggregate_rom_metrics(rom_metrics)
        summary_data = aggregator.create_summary_table(aggregated_gait, aggregated_rom)

        results['aggregated_gait'] = aggregated_gait
        results['aggregated_rom'] = aggregated_rom

        logger.info("Step 10/10: Exporting results")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        xlsx_filename = f"Gait_Analysis_{timestamp}.xlsx"
        xlsx_path = output_dir / xlsx_filename

        metadata = {
            'analysis_date': datetime.now().isoformat(),
            'top_view': str(top_path),
            'side_view': str(side_path),
            'bottom_view': str(bottom_path),
            'fps': 120.0,
            'n_frames': loader.n_frames,
            'duration_sec': loader.n_frames / 120.0,
            'scale_factor_cm_per_pixel': scale_factor,
            'n_walking_windows': len(walking_windows),
            'n_stationary_windows': len(stationary_windows)
        }

        exporter = XLSXExporter(xlsx_path)
        exporter.export(
            summary_data,
            aggregated_gait,
            aggregated_rom,
            step_results,
            walking_windows,
            metadata
        )

        exporter.save_intermediate_data(
            com_trajectory,
            paw_trajectories,
            output_dir
        )

        # Choose visualizer based on config
        use_enhanced = gs.get('use_enhanced_plots', False)
        plot_dpi = gs.get('plot_dpi', 300)

        if use_enhanced:
            visualizer = EnhancedDashboardVisualizer(
                output_dir,
                dpi=plot_dpi,
                marker_size=gs.get('plot_marker_size', 60),
                font_scale=gs.get('plot_font_scale', 1.0),
                annotate_median=gs.get('plot_annotate_median', True),
                add_reference_bands=gs.get('plot_reference_bands', True)
            )
        else:
            visualizer = DashboardVisualizer(output_dir, dpi=plot_dpi)

        plot_paths = visualizer.generate_all_dashboards(
            gait_metrics,
            rom_metrics,
            aggregated_gait,
            aggregated_rom
        )

        results['output_files'] = {
            'xlsx': str(xlsx_path),
            'plots': [str(p) for p in plot_paths]
        }

        logger.info("=" * 80)
        logger.info("Analysis complete!")
        logger.info(f"Results saved to: {output_dir}")
        logger.info(f"  - Excel: {xlsx_filename}")
        logger.info(f"  - Plots: {len(plot_paths)} PNG files")
        logger.info("=" * 80)

        return {
            'status': 'success',
            'metadata': metadata,
            'output_files': results['output_files']
        }

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        return {
            'status': 'error',
            'error': str(e)
        }


def main():
    """Main entry point for CLI"""
    parser = argparse.ArgumentParser(
        description='EXMO Animal Gait Analysis Pipeline - Production-Grade System',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--top', type=Path, required=True,
                       help='Path to top view CSV file')
    parser.add_argument('--side', type=Path, required=True,
                       help='Path to side view CSV file')
    parser.add_argument('--bottom', type=Path, required=True,
                       help='Path to bottom view CSV file')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output directory for results')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')

    args = parser.parse_args()

    for path in [args.top, args.side, args.bottom]:
        if not path.exists():
            print(f"Error: File not found: {path}", file=sys.stderr)
            sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)

    result = run_pipeline(
        args.top,
        args.side,
        args.bottom,
        args.output,
        args.verbose
    )

    print(json.dumps(result, indent=2))

    sys.exit(0 if result['status'] == 'success' else 2)


if __name__ == '__main__':
    main()

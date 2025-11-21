#!/usr/bin/env python3
"""
EXMO Gait Analysis - Enhanced Batch Processing Script
Processes multiple samples with adaptive thresholds and publication-grade visualization
Version: 1.1.0
"""

import yaml
import argparse
import logging
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from typing import Dict, List, Tuple, Optional
import sys
import time
from tqdm import tqdm

# Import the main pipeline
from src.exmo_gait.cli import run_pipeline


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════

def load_config(config_path: Path) -> Dict:
    """
    Load configuration from YAML file with validation.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate required fields
    required_fields = ['input_dir', 'output_dir', 'experiment_groups',
                      'file_patterns', 'global_settings']
    missing = [field for field in required_fields if field not in config]
    if missing:
        raise ValueError(f"Missing required config fields: {missing}")

    return config


# ═══════════════════════════════════════════════════════════════════════════
# FILE DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════

def find_matching_files(input_dir: Path, sample_id: str, config: Dict) -> Tuple[Path, Path, Path]:
    """
    Find matching TOP, SIDE, BOTTOM CSV files for a sample ID.
    Supports multiple prefix variations and flexible file naming.

    Args:
        input_dir: Base input directory
        sample_id: Sample identifier (e.g., "1grade_3")
        config: Configuration dictionary

    Returns:
        Tuple of (top_path, side_path, bottom_path)

    Raises:
        FileNotFoundError: If any view files are missing
    """
    patterns = config['file_patterns']
    prefixes = config['prefix_variations']

    top_file = None
    side_file = None
    bottom_file = None

    # Try each prefix variation
    for prefix in prefixes:
        # Try to find TOP file
        if not top_file:
            pattern = patterns['top'].format(prefix=prefix, sample_id=sample_id)
            matches = list((input_dir / 'TOP').glob(pattern))
            if matches:
                top_file = matches[0]

        # Try to find SIDE file
        if not side_file:
            pattern = patterns['side'].format(prefix=prefix, sample_id=sample_id)
            matches = list((input_dir / 'SIDE').glob(pattern))
            if matches:
                side_file = matches[0]

        # Try to find BOTTOM file
        if not bottom_file:
            pattern = patterns['bottom'].format(prefix=prefix, sample_id=sample_id)
            matches = list((input_dir / 'BOTTOM').glob(pattern))
            if matches:
                bottom_file = matches[0]

    if not (top_file and side_file and bottom_file):
        missing = []
        if not top_file: missing.append('TOP')
        if not side_file: missing.append('SIDE')
        if not bottom_file: missing.append('BOTTOM')

        raise FileNotFoundError(
            f"Could not find {', '.join(missing)} view file(s) for sample '{sample_id}'\n"
            f"  TOP: {top_file if top_file else 'NOT FOUND'}\n"
            f"  SIDE: {side_file if side_file else 'NOT FOUND'}\n"
            f"  BOTTOM: {bottom_file if bottom_file else 'NOT FOUND'}\n"
            f"  Searched in: {input_dir}"
        )

    return top_file, side_file, bottom_file


# ═══════════════════════════════════════════════════════════════════════════
# SAMPLE PROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def process_sample(sample_id: str, group_name: str, config: Dict,
                  progress_callback: Optional[callable] = None) -> Dict:
    """
    Process a single sample with comprehensive error handling.

    Args:
        sample_id: Sample identifier
        group_name: Experiment group name
        config: Configuration dictionary
        progress_callback: Optional callback for progress updates

    Returns:
        Dictionary with processing results and metadata
    """
    input_dir = Path(config['input_dir'])
    output_base = Path(config['output_dir'])

    start_time = time.time()

    # Create output directory structure
    if config['output_structure']['create_group_folders']:
        output_dir = output_base / group_name
    else:
        output_dir = output_base

    if config['output_structure']['create_sample_folders']:
        output_dir = output_dir / sample_id

    output_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"batch.{sample_id}")
    logger.info(f"Processing sample: {sample_id} (group: {group_name})")

    try:
        # Find input files
        top_path, side_path, bottom_path = find_matching_files(input_dir, sample_id, config)

        logger.info(f"  TOP: {top_path.name}")
        logger.info(f"  SIDE: {side_path.name}")
        logger.info(f"  BOTTOM: {bottom_path.name}")

        # Run pipeline with config
        result = run_pipeline(
            top_path=top_path,
            side_path=side_path,
            bottom_path=bottom_path,
            output_dir=output_dir,
            config=config,
            verbose=config['global_settings'].get('verbose', False)
        )

        # Calculate processing time
        processing_time = time.time() - start_time

        # Enhance result with additional metadata
        result['sample_id'] = sample_id
        result['group_name'] = group_name
        result['processing_time_sec'] = round(processing_time, 2)
        result['config_name'] = config.get('config_name', 'default')
        result['adaptive_mode'] = config['global_settings'].get('adaptive_thresholding', False)
        result['enhanced_plots'] = config['global_settings'].get('use_enhanced_plots', False)

        if result['status'] == 'success':
            logger.info(f"✓ Sample {sample_id} completed in {processing_time:.1f}s")
        else:
            logger.error(f"✗ Sample {sample_id} failed: {result.get('error', 'Unknown error')}")

        if progress_callback:
            progress_callback(sample_id, 'completed', result)

        return result

    except FileNotFoundError as e:
        logger.error(f"✗ Sample {sample_id}: File not found - {str(e)}")
        processing_time = time.time() - start_time
        return {
            'sample_id': sample_id,
            'group_name': group_name,
            'status': 'error',
            'error': f"File not found: {str(e)}",
            'error_type': 'FileNotFoundError',
            'processing_time_sec': round(processing_time, 2)
        }

    except Exception as e:
        logger.error(f"✗ Sample {sample_id} failed with exception: {str(e)}")
        processing_time = time.time() - start_time
        return {
            'sample_id': sample_id,
            'group_name': group_name,
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__,
            'processing_time_sec': round(processing_time, 2)
        }


# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY REPORTING
# ═══════════════════════════════════════════════════════════════════════════

def collect_individual_stride_data(results: List[Dict], output_path: Path) -> pd.DataFrame:
    """
    Collect aggregated stride statistics from all successfully processed samples.

    Args:
        results: List of processing results
        output_path: Base output path

    Returns:
        DataFrame with aggregated stride statistics (or empty DataFrame if none found)
    """
    logger = logging.getLogger("batch.stride_collection")
    all_strides = []

    for result in results:
        if result.get('status') != 'success':
            continue

        sample_id = result.get('sample_id', 'Unknown')
        group_name = result.get('group_name', 'Unknown')

        # Find the sample's Gait_Analysis Excel file
        sample_output_dir = output_path / group_name / sample_id
        xlsx_files = list(sample_output_dir.glob('Gait_Analysis_*.xlsx'))

        if not xlsx_files:
            logger.warning(f"No Gait_Analysis file found for {sample_id}")
            continue

        xlsx_file = xlsx_files[0]

        try:
            # Read the Summary sheet
            df_summary = pd.read_excel(xlsx_file, sheet_name='Summary')

            # Extract aggregated data
            for limb in ['paw_RR', 'paw_RL']:
                # Stride lengths
                stride_length_data = df_summary[
                    (df_summary['category'] == 'Gait') &
                    (df_summary['limb/joint'] == limb) &
                    (df_summary['metric'] == 'stride_lengths')
                ]

                if not stride_length_data.empty:
                    row_data = stride_length_data.iloc[0]
                    all_strides.append({
                        'sample_id': sample_id,
                        'group': group_name,
                        'limb': limb,
                        'metric': 'stride_length',
                        'mean_value': row_data.get('value', None),
                        'median_value': row_data.get('median', None),
                        'std': row_data.get('std', None),
                        'mad': row_data.get('mad', None),
                        'unit': row_data.get('unit', 'cm')
                    })

                # Stride times
                stride_time_data = df_summary[
                    (df_summary['category'] == 'Gait') &
                    (df_summary['limb/joint'] == limb) &
                    (df_summary['metric'] == 'stride_times')
                ]

                if not stride_time_data.empty:
                    row_data = stride_time_data.iloc[0]
                    all_strides.append({
                        'sample_id': sample_id,
                        'group': group_name,
                        'limb': limb,
                        'metric': 'stride_time',
                        'mean_value': row_data.get('value', None),
                        'median_value': row_data.get('median', None),
                        'std': row_data.get('std', None),
                        'mad': row_data.get('mad', None),
                        'unit': row_data.get('unit', 's')
                    })

        except Exception as e:
            logger.warning(f"Error reading stride data from {sample_id}: {e}")
            continue

    if not all_strides:
        logger.warning("No stride data collected - returning empty DataFrame")
        return pd.DataFrame()

    return pd.DataFrame(all_strides)


def collect_raw_stride_data(results: List[Dict], output_path: Path) -> pd.DataFrame:
    """
    Collect raw stride-by-stride data from all successfully processed samples.

    Args:
        results: List of processing results
        output_path: Base output path

    Returns:
        DataFrame with individual stride values (or empty DataFrame if none found)
    """
    logger = logging.getLogger("batch.raw_stride_collection")
    all_raw_strides = []

    for result in results:
        if result.get('status') != 'success':
            continue

        sample_id = result.get('sample_id', 'Unknown')
        group_name = result.get('group_name', 'Unknown')

        # Find the sample's intermediates directory
        sample_output_dir = output_path / group_name / sample_id
        intermediates_dir = sample_output_dir / 'intermediates'

        if not intermediates_dir.exists():
            logger.warning(f"No intermediates directory for {sample_id}")
            continue

        # Read stride data CSV files for hindlimbs
        for limb in ['paw_RR', 'paw_RL']:
            stride_file = intermediates_dir / f'stride_data_{limb}.csv'

            if not stride_file.exists():
                logger.warning(f"No stride data file for {sample_id} {limb}")
                continue

            try:
                df_stride = pd.read_csv(stride_file)

                for _, row in df_stride.iterrows():
                    all_raw_strides.append({
                        'sample_id': sample_id,
                        'group': group_name,
                        'limb': limb,
                        'stride_number': row.get('stride_number', None),
                        'stride_length_cm': row.get('stride_length_cm', None),
                        'stride_time_s': row.get('stride_time_s', None),
                        'start_frame': row.get('start_frame', None),
                        'end_frame': row.get('end_frame', None)
                    })

            except Exception as e:
                logger.warning(f"Error reading stride file for {sample_id} {limb}: {e}")
                continue

    if not all_raw_strides:
        logger.warning("No raw stride data collected - returning empty DataFrame")
        return pd.DataFrame()

    logger.info(f"Collected {len(all_raw_strides)} raw stride records from {len(set(r['sample_id'] for r in all_raw_strides))} samples")
    return pd.DataFrame(all_raw_strides)


def generate_summary_report(results: List[Dict], config: Dict, output_path: Path):
    """
    Generate comprehensive batch processing summary report.

    Args:
        results: List of processing results
        config: Configuration dictionary
        output_path: Path for summary report
    """
    logger = logging.getLogger("batch.summary")
    logger.info("Generating summary report...")

    # Create summary DataFrame
    summary_data = []

    for result in results:
        row = {
            'sample_id': result.get('sample_id', 'Unknown'),
            'group': result.get('group_name', 'Unknown'),
            'status': result.get('status', 'Unknown'),
            'error': result.get('error', ''),
            'error_type': result.get('error_type', ''),
            'processing_time_sec': result.get('processing_time_sec', 0),
        }

        # Add metadata if available
        if 'metadata' in result:
            meta = result['metadata']
            row['n_frames'] = meta.get('n_frames', 0)
            row['duration_sec'] = meta.get('duration_sec', 0)
            row['n_walking_windows'] = meta.get('n_walking_windows', 0)
            row['scale_factor'] = meta.get('scale_factor_cm_per_pixel', 0)
            row['walking_threshold'] = meta.get('walking_threshold_cm_s', 0)

        # Add configuration metadata
        row['adaptive_mode'] = result.get('adaptive_mode', False)
        row['enhanced_plots'] = result.get('enhanced_plots', False)

        summary_data.append(row)

    df = pd.DataFrame(summary_data)

    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_filename = config.get('summary_report', {}).get('output_filename',
                                                            'Batch_Summary_Report_{timestamp}.xlsx')
    summary_filename = summary_filename.format(timestamp=timestamp)
    summary_path = output_path / summary_filename

    with pd.ExcelWriter(summary_path, engine='openpyxl') as writer:
        # Sheet 1: Detailed Summary
        df.to_excel(writer, sheet_name='Summary', index=False)

        # Sheet 2: Group Statistics
        if config.get('summary_report', {}).get('compare_groups', True):
            group_stats = df[df['status'] == 'success'].groupby('group').agg({
                'sample_id': 'count',
                'n_frames': 'mean',
                'duration_sec': 'mean',
                'n_walking_windows': 'mean',
                'processing_time_sec': 'mean'
            }).reset_index()
            group_stats.columns = ['group', 'successful_samples', 'avg_frames',
                                  'avg_duration_sec', 'avg_walking_windows',
                                  'avg_processing_time_sec']
            group_stats.to_excel(writer, sheet_name='Group Statistics', index=False)

        # Sheet 3: Aggregated Stride Statistics
        try:
            stride_data = collect_individual_stride_data(results, output_path)
            if not stride_data.empty:
                stride_data.to_excel(writer, sheet_name='Stride Statistics', index=False)
                logger.info(f"Added {len(stride_data)} aggregated stride statistic records")
        except Exception as e:
            logger.warning(f"Could not add stride statistics: {e}")

        # Sheet 4: Raw Stride Data (ALL individual values)
        try:
            raw_stride_data = collect_raw_stride_data(results, output_path)
            if not raw_stride_data.empty:
                raw_stride_data.to_excel(writer, sheet_name='Raw Stride Data', index=False)
                logger.info(f"Added {len(raw_stride_data)} raw stride records (stride-by-stride)")
        except Exception as e:
            logger.warning(f"Could not add raw stride data: {e}")

        # Sheet 5: Errors (if any)
        errors_df = df[df['status'] != 'success']
        if not errors_df.empty:
            errors_df.to_excel(writer, sheet_name='Errors', index=False)

    logger.info(f"Summary report saved: {summary_path}")

    # Generate group comparison plots (v1.4.0)
    if config.get('summary_report', {}).get('compare_groups', True):
        try:
            logger.info("Generating group comparison plots...")
            from src.exmo_gait.statistics.group_comparator import GroupComparator

            comparator = GroupComparator(output_path)
            comparator.load_group_data(config['experiment_groups'])

            # Compute statistics
            stats_df = comparator.compute_group_statistics()
            stats_df = comparator.compute_statistical_significance(stats_df)

            # Create plots directory
            plots_dir = output_path / 'GroupComparison'
            plots_dir.mkdir(exist_ok=True)

            # Generate plots
            comparator.create_group_comparison_plots(stats_df, plots_dir)

            # Export statistics table
            stats_path = output_path / f'Group_Statistics_{timestamp}.xlsx'
            comparator.export_statistics_table(stats_df, stats_path)

            logger.info(f"Group comparison plots saved to {plots_dir}")
            logger.info(f"Group statistics table saved to {stats_path}")
        except Exception as e:
            logger.error(f"Failed to generate group comparison: {e}")
            import traceback
            traceback.print_exc()

    # Print summary to console
    print_summary_console(df, results)


def print_summary_console(df: pd.DataFrame, results: List[Dict]):
    """Print formatted summary to console"""
    successful = df[df['status'] == 'success']
    failed = df[df['status'] != 'success']

    total_time = sum(r.get('processing_time_sec', 0) for r in results)

    print("\n" + "="*80)
    print("BATCH PROCESSING SUMMARY")
    print("="*80)
    print(f"\nTotal samples processed: {len(results)}")
    print(f"Successful: {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
    print(f"Failed: {len(failed)} ({len(failed)/len(results)*100:.1f}%)")
    print(f"Total processing time: {total_time/60:.1f} minutes")

    if not successful.empty:
        print(f"\nAverage processing time: {successful['processing_time_sec'].mean():.1f}s per sample")
        print(f"Average walking windows: {successful['n_walking_windows'].mean():.1f}")

    print(f"\nBy Group:")
    group_summary = df.groupby('group')['status'].value_counts().unstack(fill_value=0)
    print(group_summary)

    if not failed.empty:
        print(f"\n⚠️  Failed Samples:")
        for _, row in failed.iterrows():
            print(f"  - {row['sample_id']} ({row['group']}): {row['error_type']}")

    print("="*80 + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Enhanced batch processing entry point with progress tracking"""
    parser = argparse.ArgumentParser(
        description='EXMO Gait Analysis - Enhanced Batch Processing with Adaptive Thresholds',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single sample
  python batch_process.py --sample control_5

  # Process group with adaptive config
  python batch_process.py --config config_adaptive.yaml --group control

  # Process all samples in parallel
  python batch_process.py --config config_adaptive.yaml --batch --parallel 4

  # Dry run to check configuration
  python batch_process.py --batch --dry-run
        """
    )

    parser.add_argument('--config', type=Path, default='config.yaml',
                       help='Path to configuration YAML file (default: config.yaml)')
    parser.add_argument('--sample', type=str,
                       help='Process single sample by ID')
    parser.add_argument('--group', type=str,
                       help='Process all samples in experiment group')
    parser.add_argument('--batch', action='store_true',
                       help='Process all samples in all groups')
    parser.add_argument('--parallel', type=int, default=None,
                       help='Number of parallel jobs (overrides config)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be processed without actually processing')
    parser.add_argument('--continue-on-error', action='store_true',
                       help='Continue processing even if samples fail')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("batch")

    # Print configuration info
    print("\n" + "="*80)
    print("EXMO GAIT ANALYSIS - BATCH PROCESSING")
    print("="*80)
    print(f"Config file: {args.config}")
    print(f"Adaptive thresholds: {config['global_settings'].get('adaptive_thresholding', False)}")
    print(f"Enhanced plots: {config['global_settings'].get('use_enhanced_plots', False)}")
    print(f"Plot DPI: {config['global_settings'].get('plot_dpi', 300)}")
    print("="*80 + "\n")

    # Determine samples to process
    samples_to_process = []

    if args.sample:
        # Single sample
        for group_name, group_config in config['experiment_groups'].items():
            if args.sample in group_config['samples']:
                samples_to_process.append((args.sample, group_name))
                break

        if not samples_to_process:
            logger.error(f"Sample '{args.sample}' not found in any group")
            sys.exit(1)

    elif args.group:
        # Single group
        if args.group not in config['experiment_groups']:
            logger.error(f"Group '{args.group}' not found in configuration")
            logger.info(f"Available groups: {', '.join(config['experiment_groups'].keys())}")
            sys.exit(1)

        group_config = config['experiment_groups'][args.group]
        for sample in group_config['samples']:
            samples_to_process.append((sample, args.group))

    elif args.batch:
        # All samples
        for group_name, group_config in config['experiment_groups'].items():
            for sample in group_config['samples']:
                samples_to_process.append((sample, group_name))

    else:
        parser.print_help()
        sys.exit(1)

    # Dry run mode
    if args.dry_run:
        print(f"\n📋 DRY RUN: Would process {len(samples_to_process)} samples:\n")
        for sample, group in samples_to_process:
            print(f"  - {sample} (group: {group})")
        print(f"\nParallel jobs: {args.parallel or config['batch_processing']['parallel_jobs']}")
        print("\nRun without --dry-run to start processing.\n")
        sys.exit(0)

    logger.info(f"Processing {len(samples_to_process)} samples")

    # Process samples
    parallel_jobs = args.parallel or config['batch_processing'].get('parallel_jobs', 1)
    results = []

    if parallel_jobs > 1:
        # Parallel processing with progress bar
        logger.info(f"Using {parallel_jobs} parallel workers")

        with ProcessPoolExecutor(max_workers=parallel_jobs) as executor:
            futures = {
                executor.submit(process_sample, sample, group, config): (sample, group)
                for sample, group in samples_to_process
            }

            # Progress bar
            with tqdm(total=len(samples_to_process), desc="Processing samples",
                     unit="sample") as pbar:
                for future in as_completed(futures):
                    sample, group = futures[future]
                    try:
                        result = future.result()
                        results.append(result)

                        # Update progress bar with status
                        status_icon = "✓" if result['status'] == 'success' else "✗"
                        pbar.set_postfix_str(f"{status_icon} {sample}")
                        pbar.update(1)

                    except Exception as e:
                        logger.error(f"Sample {sample} raised exception: {e}")
                        results.append({
                            'sample_id': sample,
                            'group_name': group,
                            'status': 'error',
                            'error': str(e),
                            'error_type': 'ExecutionError'
                        })
                        pbar.update(1)
    else:
        # Sequential processing with progress bar
        logger.info("Sequential processing (parallel=1)")

        with tqdm(samples_to_process, desc="Processing samples", unit="sample") as pbar:
            for sample, group in pbar:
                pbar.set_postfix_str(f"Current: {sample}")
                result = process_sample(sample, group, config)
                results.append(result)

                # Update with status
                status_icon = "✓" if result['status'] == 'success' else "✗"
                pbar.set_postfix_str(f"{status_icon} {sample}")

    # Generate summary report
    if config.get('batch_processing', {}).get('generate_summary_report', True):
        output_dir = Path(config['output_dir'])
        generate_summary_report(results, config, output_dir)

    # Exit with appropriate code
    failed = sum(1 for r in results if r['status'] != 'success')

    if failed > 0 and not args.continue_on_error:
        logger.error(f"{failed} sample(s) failed. Use --continue-on-error to ignore failures.")
        sys.exit(1)
    else:
        logger.info(f"Batch processing complete. {len(results)-failed}/{len(results)} successful.")
        sys.exit(0)


if __name__ == '__main__':
    main()

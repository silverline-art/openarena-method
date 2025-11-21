"""Excel export functionality for analysis results"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List
from datetime import datetime

logger = logging.getLogger(__name__)


class XLSXExporter:
    """Export analysis results to Excel workbook"""

    def __init__(self, output_path: Path):
        """
        Initialize XLSX exporter.

        Args:
            output_path: Path for output XLSX file
        """
        self.output_path = output_path
        self.writer = None

    def create_summary_sheet(self, summary_data: List[Dict]) -> pd.DataFrame:
        """
        Create summary sheet DataFrame.

        Args:
            summary_data: List of summary records

        Returns:
            Summary DataFrame
        """
        df = pd.DataFrame(summary_data)
        return df

    def create_gait_metrics_sheet(self, gait_metrics: Dict) -> pd.DataFrame:
        """
        Create gait metrics sheet.

        Args:
            gait_metrics: Aggregated gait metrics dictionary

        Returns:
            Gait metrics DataFrame
        """
        rows = []

        for limb, metrics in gait_metrics.items():
            for metric_name, stats in metrics.items():
                if isinstance(stats, dict):
                    row = {
                        'limb': limb,
                        'metric': metric_name
                    }

                    if 'value' in stats:
                        row['value'] = stats['value']
                    if 'median' in stats:
                        row['median'] = stats['median']
                    if 'mean' in stats:
                        row['mean'] = stats['mean']
                    if 'std' in stats:
                        row['std'] = stats['std']
                    if 'mad' in stats:
                        row['mad'] = stats['mad']
                    if 'min' in stats:
                        row['min'] = stats['min']
                    if 'max' in stats:
                        row['max'] = stats['max']
                    if 'count' in stats:
                        row['count'] = stats['count']

                    rows.append(row)

        df = pd.DataFrame(rows)
        return df

    def create_rom_metrics_sheet(self, rom_metrics: Dict) -> pd.DataFrame:
        """
        Create ROM metrics sheet.

        Args:
            rom_metrics: Aggregated ROM metrics dictionary

        Returns:
            ROM metrics DataFrame
        """
        rows = []

        for joint, metrics in rom_metrics.items():
            for metric_name, stats in metrics.items():
                if isinstance(stats, dict):
                    row = {
                        'joint': joint,
                        'metric': metric_name
                    }

                    if 'value' in stats:
                        row['value'] = stats['value']
                    if 'median' in stats:
                        row['median'] = stats['median']
                    if 'mean' in stats:
                        row['mean'] = stats['mean']
                    if 'std' in stats:
                        row['std'] = stats['std']
                    if 'mad' in stats:
                        row['mad'] = stats['mad']
                    if 'min' in stats:
                        row['min'] = stats['min']
                    if 'max' in stats:
                        row['max'] = stats['max']

                    rows.append(row)

        df = pd.DataFrame(rows)
        return df

    def create_step_events_sheet(self, step_results: Dict) -> pd.DataFrame:
        """
        Create step events sheet.

        Args:
            step_results: Step detection results

        Returns:
            Step events DataFrame
        """
        rows = []

        for limb, results in step_results.items():
            foot_strikes = results.get('foot_strikes', [])
            stride_times = results.get('stride_times', [])

            for i, strike_frame in enumerate(foot_strikes):
                row = {
                    'limb': limb,
                    'step_number': i + 1,
                    'frame': strike_frame,
                    'time_sec': strike_frame / 120.0
                }

                if i < len(stride_times):
                    row['stride_time'] = stride_times[i]

                rows.append(row)

        df = pd.DataFrame(rows)
        return df

    def create_walking_windows_sheet(self, walking_windows: List) -> pd.DataFrame:
        """
        Create walking windows sheet.

        Args:
            walking_windows: List of (start, end) tuples

        Returns:
            Walking windows DataFrame
        """
        rows = []

        for i, (start, end) in enumerate(walking_windows):
            duration_frames = end - start + 1
            duration_sec = duration_frames / 120.0

            rows.append({
                'window_number': i + 1,
                'start_frame': start,
                'end_frame': end,
                'duration_frames': duration_frames,
                'duration_sec': duration_sec
            })

        df = pd.DataFrame(rows)
        return df

    def export(self,
              summary_data: List[Dict],
              gait_metrics: Dict,
              rom_metrics: Dict,
              step_results: Dict,
              walking_windows: List,
              metadata: Dict = None) -> None:
        """
        Export all data to Excel workbook.

        Args:
            summary_data: Summary table data
            gait_metrics: Aggregated gait metrics
            rom_metrics: Aggregated ROM metrics
            step_results: Step detection results
            walking_windows: Walking windows
            metadata: Optional metadata dictionary
        """
        logger.info(f"Exporting results to {self.output_path}")

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(self.output_path, engine='openpyxl') as writer:
            summary_df = self.create_summary_sheet(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

            gait_df = self.create_gait_metrics_sheet(gait_metrics)
            gait_df.to_excel(writer, sheet_name='Gait Metrics', index=False)

            rom_df = self.create_rom_metrics_sheet(rom_metrics)
            rom_df.to_excel(writer, sheet_name='ROM Metrics', index=False)

            events_df = self.create_step_events_sheet(step_results)
            events_df.to_excel(writer, sheet_name='Step Events', index=False)

            windows_df = self.create_walking_windows_sheet(walking_windows)
            windows_df.to_excel(writer, sheet_name='Walking Windows', index=False)

            if metadata:
                meta_rows = [{'key': k, 'value': v} for k, v in metadata.items()]
                meta_df = pd.DataFrame(meta_rows)
                meta_df.to_excel(writer, sheet_name='Metadata', index=False)

        logger.info(f"Export complete: {self.output_path}")

    def save_stride_data(self,
                        step_results: Dict,
                        output_dir: Path) -> None:
        """
        Save raw stride data arrays to CSV files.

        Args:
            step_results: Step detection results with stride arrays
            output_dir: Output directory for stride data files
        """
        intermediates_dir = output_dir / 'intermediates'
        intermediates_dir.mkdir(parents=True, exist_ok=True)

        for limb, results in step_results.items():
            stride_lengths = results.get('stride_lengths', np.array([]))
            stride_times = results.get('stride_times', np.array([]))
            foot_strikes = results.get('foot_strikes', np.array([]))

            if len(stride_lengths) == 0:
                continue

            rows = []
            for i in range(len(stride_lengths)):
                row = {
                    'stride_number': i + 1,
                    'stride_length_cm': stride_lengths[i] if i < len(stride_lengths) else np.nan,
                    'stride_time_s': stride_times[i] if i < len(stride_times) else np.nan,
                }

                # Add frame information if available
                if i < len(foot_strikes):
                    row['start_frame'] = foot_strikes[i]
                if i + 1 < len(foot_strikes):
                    row['end_frame'] = foot_strikes[i + 1]

                rows.append(row)

            if rows:
                stride_df = pd.DataFrame(rows)
                stride_df.to_csv(intermediates_dir / f'stride_data_{limb}.csv', index=False)
                logger.info(f"Saved {len(rows)} stride records for {limb}")

    def save_intermediate_data(self,
                              com_trajectory: np.ndarray,
                              paw_trajectories: Dict[str, np.ndarray],
                              output_dir: Path) -> None:
        """
        Save intermediate data as CSV files.

        Args:
            com_trajectory: CoM trajectory
            paw_trajectories: Dictionary of paw trajectories
            output_dir: Output directory for intermediate files
        """
        intermediates_dir = output_dir / 'intermediates'
        intermediates_dir.mkdir(parents=True, exist_ok=True)

        com_df = pd.DataFrame(com_trajectory, columns=['x', 'y'])
        com_df.to_csv(intermediates_dir / 'com_trajectory.csv', index=False)

        for limb_name, traj in paw_trajectories.items():
            traj_df = pd.DataFrame(traj, columns=['x', 'y'])
            traj_df.to_csv(intermediates_dir / f'{limb_name}_trajectory.csv', index=False)

        logger.info(f"Saved intermediate data to {intermediates_dir}")

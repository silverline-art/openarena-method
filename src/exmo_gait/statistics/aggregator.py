"""Statistical aggregation: median, SD, MAD computation"""
import numpy as np
import logging
from typing import Dict, Any, List
from ..utils.signal_processing import compute_mad
from ..constants import CI_PERCENTILE_DEFAULT, TRIM_PERCENT_DEFAULT

logger = logging.getLogger(__name__)


class StatisticsAggregator:
    """Aggregate metrics with statistical measures"""

    def __init__(self):
        """Initialize aggregator with optional v1.2.0 enhanced stats"""
        self.use_enhanced_stats = False
        self.ci_percentile = CI_PERCENTILE_DEFAULT
        self.trim_percent = TRIM_PERCENT_DEFAULT

    @staticmethod
    def compute_summary_stats(values: np.ndarray) -> Dict[str, float]:
        """
        Compute summary statistics for an array of values (LEGACY v1.1).

        Args:
            values: Array of numeric values

        Returns:
            Dictionary with median, SD, MAD, mean, min, max
        """
        valid_values = values[~np.isnan(values)]

        if len(valid_values) == 0:
            return {
                'median': np.nan,
                'std': np.nan,
                'mad': np.nan,
                'mean': np.nan,
                'min': np.nan,
                'max': np.nan,
                'count': 0
            }

        return {
            'median': np.median(valid_values),
            'std': np.std(valid_values),
            'mad': compute_mad(valid_values),
            'mean': np.mean(valid_values),
            'min': np.min(valid_values),
            'max': np.max(valid_values),
            'count': len(valid_values)
        }

    @staticmethod
    def compute_summary_stats_v2(values: np.ndarray,
                                 include_ci: bool = True,
                                 ci_percentile: float = CI_PERCENTILE_DEFAULT,
                                 trim_percent: float = TRIM_PERCENT_DEFAULT) -> Dict[str, float]:
        """
        Compute enhanced summary statistics with confidence intervals (v1.2.0).

        Args:
            values: Array of numeric values
            include_ci: Include 95% confidence intervals
            ci_percentile: Confidence interval percentile (default 95)
            trim_percent: Percentage to trim for corrected mean (default 5)

        Returns:
            Dictionary with median, SD, MAD, mean, CI, corrected_mean
        """
        from scipy import stats

        valid_values = values[~np.isnan(values)]

        if len(valid_values) == 0:
            return {
                'median': np.nan,
                'std': np.nan,
                'mad': np.nan,
                'mean': np.nan,
                'min': np.nan,
                'max': np.nan,
                'count': 0,
                'corrected_mean': np.nan,
                'ci_low': np.nan,
                'ci_high': np.nan,
                'ci_range': np.nan
            }

        # Basic stats
        median_val = np.median(valid_values)
        std_val = np.std(valid_values)
        mad_val = compute_mad(valid_values)
        mean_val = np.mean(valid_values)
        min_val = np.min(valid_values)
        max_val = np.max(valid_values)
        count = len(valid_values)

        # Trimmed mean (corrected mean - removes extreme 5%)
        trim_fraction = trim_percent / 100.0
        corrected_mean = stats.trim_mean(valid_values, trim_fraction)

        # Confidence intervals
        if include_ci and count > 1:
            # Compute percentile-based CI
            ci_lower_pct = (100 - ci_percentile) / 2
            ci_upper_pct = 100 - ci_lower_pct

            ci_low = np.percentile(valid_values, ci_lower_pct)
            ci_high = np.percentile(valid_values, ci_upper_pct)
            ci_range = ci_high - ci_low
        else:
            ci_low = np.nan
            ci_high = np.nan
            ci_range = np.nan

        return {
            'median': float(median_val),
            'std': float(std_val),
            'mad': float(mad_val),
            'mean': float(mean_val),
            'min': float(min_val),
            'max': float(max_val),
            'count': int(count),
            'corrected_mean': float(corrected_mean),  # NEW
            'ci_low': float(ci_low),                  # NEW
            'ci_high': float(ci_high),                # NEW
            'ci_range': float(ci_range)               # NEW
        }

    def aggregate_gait_metrics(self, gait_metrics: Dict) -> Dict:
        """
        Aggregate gait metrics with summary statistics.

        Args:
            gait_metrics: Dictionary of gait metrics from GaitMetricsComputer

        Returns:
            Dictionary of aggregated metrics
        """
        logger.info("Aggregating gait metrics")

        aggregated = {}

        for limb, metrics in gait_metrics.items():
            if limb not in aggregated:
                aggregated[limb] = {}

            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    aggregated[limb][metric_name] = {
                        'value': metric_value
                    }
                elif isinstance(metric_value, np.ndarray):
                    # Use enhanced stats if enabled (v1.2.0)
                    if self.use_enhanced_stats:
                        stats = self.compute_summary_stats_v2(
                            metric_value,
                            include_ci=True,
                            ci_percentile=self.ci_percentile,
                            trim_percent=self.trim_percent
                        )
                    else:
                        stats = self.compute_summary_stats(metric_value)
                    aggregated[limb][metric_name] = stats
                else:
                    aggregated[limb][metric_name] = metric_value

        logger.info("Gait metrics aggregation complete")

        return aggregated

    def aggregate_rom_metrics(self, rom_metrics: Dict) -> Dict:
        """
        Aggregate ROM metrics with summary statistics.

        Args:
            rom_metrics: Dictionary of ROM metrics from ROMMetricsComputer

        Returns:
            Dictionary of aggregated metrics
        """
        logger.info("Aggregating ROM metrics")

        aggregated = {}

        for joint, metrics in rom_metrics.items():
            if joint not in aggregated:
                aggregated[joint] = {}

            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)):
                    aggregated[joint][metric_name] = {
                        'value': metric_value
                    }
                elif isinstance(metric_value, np.ndarray):
                    # Use enhanced stats if enabled (v1.2.0)
                    if self.use_enhanced_stats:
                        stats = self.compute_summary_stats_v2(
                            metric_value,
                            include_ci=True,
                            ci_percentile=self.ci_percentile,
                            trim_percent=self.trim_percent
                        )
                    else:
                        stats = self.compute_summary_stats(metric_value)
                    aggregated[joint][metric_name] = stats
                else:
                    aggregated[joint][metric_name] = metric_value

        logger.info("ROM metrics aggregation complete")

        return aggregated

    def create_summary_table(self,
                           aggregated_gait: Dict,
                           aggregated_rom: Dict) -> List[Dict[str, Any]]:
        """
        Create summary table for export.

        Args:
            aggregated_gait: Aggregated gait metrics
            aggregated_rom: Aggregated ROM metrics

        Returns:
            List of dictionaries suitable for DataFrame conversion
        """
        summary = []

        for limb, metrics in aggregated_gait.items():
            for metric_name, stats in metrics.items():
                if isinstance(stats, dict) and 'value' in stats:
                    summary.append({
                        'category': 'Gait',
                        'limb/joint': limb,
                        'metric': metric_name,
                        'value': stats['value'],
                        'median': np.nan,
                        'std': np.nan,
                        'mad': np.nan,
                        'unit': self._get_unit(metric_name)
                    })
                elif isinstance(stats, dict) and 'median' in stats:
                    summary.append({
                        'category': 'Gait',
                        'limb/joint': limb,
                        'metric': metric_name,
                        'value': stats['mean'],
                        'median': stats['median'],
                        'std': stats['std'],
                        'mad': stats['mad'],
                        'unit': self._get_unit(metric_name)
                    })

        for joint, metrics in aggregated_rom.items():
            for metric_name, stats in metrics.items():
                if isinstance(stats, dict) and 'value' in stats:
                    summary.append({
                        'category': 'ROM',
                        'limb/joint': joint,
                        'metric': metric_name,
                        'value': stats['value'],
                        'median': np.nan,
                        'std': np.nan,
                        'mad': np.nan,
                        'unit': self._get_unit(metric_name)
                    })
                elif isinstance(stats, dict) and 'median' in stats:
                    summary.append({
                        'category': 'ROM',
                        'limb/joint': joint,
                        'metric': metric_name,
                        'value': stats['mean'],
                        'median': stats['median'],
                        'std': stats['std'],
                        'mad': stats['mad'],
                        'unit': self._get_unit(metric_name)
                    })

        return summary

    @staticmethod
    def _get_unit(metric_name: str) -> str:
        """Get unit for metric name"""
        units = {
            'cadence': 'steps/min',
            'duty_cycle': '%',
            'avg_speed': 'cm/s',
            'com_avg_speed': 'cm/s',
            'stride_lengths': 'cm',
            'stride_times': 's',
            'swing_stance_ratio': 'ratio',
            'regularity_index': '0-1',
            'phase_dispersion': 'ratio',
            'ml_sway_cm': 'cm',
            'ap_sway_cm': 'cm',
            'rom': 'degrees',
            'angular_velocity_mean': 'deg/s',
            'angular_velocity_max': 'deg/s'
        }
        return units.get(metric_name, '')

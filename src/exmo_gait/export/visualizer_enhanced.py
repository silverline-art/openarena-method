"""
EXMO Gait Analysis - Enhanced Publication-Grade Visualization
High-fidelity dashboard generation with scientific styling
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from .style import (
    EXMOPlotStyle, EXMO_COLORS, create_figure,
    get_limb_display_name, REFERENCE_BANDS
)

logger = logging.getLogger(__name__)


class EnhancedDashboardVisualizer:
    """
    Enhanced dashboard visualizer with publication-grade styling.
    Implements PRD specifications for high-fidelity scientific plots.
    """

    def __init__(self,
                 output_dir: Path,
                 dpi: int = 600,
                 marker_size: int = 60,
                 font_scale: float = 1.0,
                 annotate_median: bool = True,
                 add_reference_bands: bool = True):
        """
        Initialize enhanced visualizer.

        Args:
            output_dir: Output directory for plots
            dpi: Resolution (300=screen, 600=publication)
            marker_size: Base marker size
            font_scale: Global font size multiplier
            annotate_median: Add median value annotations
            add_reference_bands: Add normal range reference bands
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize style manager
        self.style = EXMOPlotStyle(
            dpi=dpi,
            marker_size=marker_size,
            font_scale=font_scale,
            use_exmo_palette=True,
            annotate_median=annotate_median
        )

        self.dpi = dpi
        self.add_reference_bands = add_reference_bands

    def plot_coordination_dashboard(self,
                                   gait_metrics: Dict,
                                   aggregated_gait: Dict) -> Path:
        """
        Enhanced coordination dashboard with publication styling.

        Args:
            gait_metrics: Raw gait metrics
            aggregated_gait: Aggregated gait metrics

        Returns:
            Path to saved plot
        """
        fig, axes = create_figure(3, 1, self.dpi)
        fig.suptitle('Coordination Dashboard',
                    fontsize=18, fontweight='bold',
                    y=0.98)

        self._plot_cadence_enhanced(axes[0], gait_metrics, aggregated_gait)
        self._plot_duty_cycle_enhanced(axes[1], gait_metrics, aggregated_gait)
        self._plot_regularity_index_enhanced(axes[2], gait_metrics, aggregated_gait)

        output_path = self.output_dir / 'plot_coordination.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()

        logger.info(f"Saved enhanced coordination dashboard: {output_path}")
        return output_path

    def _plot_cadence_enhanced(self, ax, gait_metrics: Dict, aggregated_gait: Dict):
        """Enhanced cadence subplot with reference bands"""
        limbs = ['paw_RR', 'paw_RL']
        x_positions = []
        colors_list = []
        medians = []
        mads = []
        labels = []

        for i, limb in enumerate(limbs):
            if limb in aggregated_gait and 'cadence' in aggregated_gait[limb]:
                stats = aggregated_gait[limb]['cadence']
                if isinstance(stats, dict) and 'value' in stats:
                    x_positions.append(i)
                    colors_list.append(self.style.get_limb_color(limb))
                    medians.append(stats['value'])
                    mads.append(0)
                    labels.append(get_limb_display_name(limb))

        if medians:
            # Add reference band for normal cadence
            if self.add_reference_bands:
                self.style.add_reference_band(ax, 'cadence')

            # Plot bars with enhanced styling
            self.style.plot_bar_with_error(
                ax, x_positions, medians, mads, colors_list, labels
            )

            # Add sample badge
            self.style.add_sample_badge(ax, n_samples=len(medians))

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylabel('Cadence (steps/min)', fontweight='semibold')
        ax.set_title('Cadence (Hind Limbs)', fontsize=14, fontweight='semibold')

        # Apply EXMO styling
        self.style.apply_to_axis(ax)

        # Format y-axis range
        if medians:
            self.style.format_axis_range(ax, medians, zero_baseline=True)

    def _plot_duty_cycle_enhanced(self, ax, gait_metrics: Dict, aggregated_gait: Dict):
        """Enhanced duty cycle subplot"""
        x_positions = []
        values = []
        errors = []
        colors = []
        labels = []

        for i, limb in enumerate(['paw_RR', 'paw_RL']):
            if limb in aggregated_gait and 'duty_cycle' in aggregated_gait[limb]:
                stats = aggregated_gait[limb]['duty_cycle']
                if isinstance(stats, dict) and 'value' in stats:
                    x_positions.append(i)
                    values.append(stats['value'])
                    errors.append(0)
                    colors.append(self.style.get_limb_color(limb))
                    labels.append(get_limb_display_name(limb))

        # Add quadrupedal average
        if 'quadrupedal' in aggregated_gait and 'avg_duty_cycle' in aggregated_gait['quadrupedal']:
            stats = aggregated_gait['quadrupedal']['avg_duty_cycle']
            if isinstance(stats, dict) and 'value' in stats:
                x_positions.append(len(x_positions))
                values.append(stats['value'])
                errors.append(0)
                colors.append('#666666')
                labels.append('Quadruple')

        if values:
            # Add reference band
            if self.add_reference_bands:
                self.style.add_reference_band(ax, 'duty_cycle')

            self.style.plot_bar_with_error(
                ax, x_positions, values, errors, colors, labels
            )
            self.style.add_sample_badge(ax, n_samples=len(values))

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylabel('Duty Cycle (%)', fontweight='semibold')
        ax.set_title('Duty Cycle', fontsize=14, fontweight='semibold')
        ax.set_ylim([0, 100])

        self.style.apply_to_axis(ax)

    def _plot_regularity_index_enhanced(self, ax, gait_metrics: Dict, aggregated_gait: Dict):
        """Enhanced regularity index subplot"""
        x_positions = []
        values = []
        errors = []
        colors = []
        labels = []

        diagonal_map = {
            'diagonal_RR_FL': ('RH-LF', '#e41a1c'),
            'diagonal_RL_FR': ('LH-RF', '#377eb8')
        }

        for i, (diag_key, (label, color)) in enumerate(diagonal_map.items()):
            if diag_key in aggregated_gait and 'regularity_index' in aggregated_gait[diag_key]:
                stats = aggregated_gait[diag_key]['regularity_index']
                if isinstance(stats, dict) and 'value' in stats:
                    x_positions.append(i)
                    values.append(stats['value'])
                    errors.append(0)
                    colors.append(color)
                    labels.append(label)

        if values:
            # Add reference band for healthy regularity
            if self.add_reference_bands:
                self.style.add_reference_band(ax, 'regularity_index')

            self.style.plot_bar_with_error(
                ax, x_positions, values, errors, colors, labels
            )
            self.style.add_sample_badge(ax, n_samples=len(values))

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylabel('Regularity Index (0-1)', fontweight='semibold')
        ax.set_title('Regularity Index (Diagonal Pairs)', fontsize=14, fontweight='semibold')
        ax.set_ylim([0, 1.1])

        self.style.apply_to_axis(ax)

    def plot_speed_spatial_dashboard(self,
                                    gait_metrics: Dict,
                                    aggregated_gait: Dict) -> Path:
        """Enhanced speed & spatial dashboard"""
        fig, axes = create_figure(3, 1, self.dpi)
        fig.suptitle('Speed & Spatial Metrics Dashboard',
                    fontsize=18, fontweight='bold',
                    y=0.98)

        self._plot_avg_speed_enhanced(axes[0], gait_metrics, aggregated_gait)
        self._plot_stride_length_enhanced(axes[1], gait_metrics, aggregated_gait)
        self._plot_stride_time_enhanced(axes[2], gait_metrics, aggregated_gait)

        output_path = self.output_dir / 'plot_speed_spatial.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()

        logger.info(f"Saved enhanced speed & spatial dashboard: {output_path}")
        return output_path

    def _plot_avg_speed_enhanced(self, ax, gait_metrics: Dict, aggregated_gait: Dict):
        """Enhanced average speed subplot"""
        x_positions = []
        values = []
        errors = []
        colors = []
        labels = []

        # CoM speed first
        if 'whole_body' in aggregated_gait and 'com_avg_speed' in aggregated_gait['whole_body']:
            stats = aggregated_gait['whole_body']['com_avg_speed']
            if isinstance(stats, dict) and 'value' in stats:
                x_positions.append(0)
                values.append(stats['value'])
                errors.append(0)
                colors.append(EXMO_COLORS['COM'])
                labels.append('CoM')

        # Limb speeds
        for i, limb in enumerate(['paw_RR', 'paw_RL'], start=len(x_positions)):
            if limb in aggregated_gait and 'avg_speed' in aggregated_gait[limb]:
                stats = aggregated_gait[limb]['avg_speed']
                if isinstance(stats, dict) and 'value' in stats:
                    x_positions.append(i)
                    values.append(stats['value'])
                    errors.append(0)
                    colors.append(self.style.get_limb_color(limb))
                    labels.append(get_limb_display_name(limb))

        if values:
            self.style.plot_bar_with_error(
                ax, x_positions, values, errors, colors, labels
            )
            self.style.add_sample_badge(ax, n_samples=len(values))
            self.style.format_axis_range(ax, values, zero_baseline=True)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylabel('Speed (cm/s)', fontweight='semibold')
        ax.set_title('Average Speed', fontsize=14, fontweight='semibold')

        self.style.apply_to_axis(ax)

    def _plot_stride_length_enhanced(self, ax, gait_metrics: Dict, aggregated_gait: Dict):
        """Enhanced stride length subplot with MAD error bars"""
        x_positions = []
        medians = []
        mads = []
        colors = []
        labels = []

        for i, limb in enumerate(['paw_RR', 'paw_RL']):
            if limb in aggregated_gait and 'stride_lengths' in aggregated_gait[limb]:
                stats = aggregated_gait[limb]['stride_lengths']
                if isinstance(stats, dict) and 'median' in stats:
                    x_positions.append(i)
                    medians.append(stats['median'])
                    mads.append(stats.get('mad', 0))
                    colors.append(self.style.get_limb_color(limb))
                    labels.append(get_limb_display_name(limb))

        if medians:
            self.style.plot_bar_with_error(
                ax, x_positions, medians, mads, colors, labels
            )
            self.style.add_sample_badge(ax, n_samples=len(medians))
            self.style.format_axis_range(ax, medians, zero_baseline=True)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylabel('Stride Length (cm)', fontweight='semibold')
        ax.set_title('Stride Length', fontsize=14, fontweight='semibold')

        self.style.apply_to_axis(ax)

    def _plot_stride_time_enhanced(self, ax, gait_metrics: Dict, aggregated_gait: Dict):
        """Enhanced stride time subplot with MAD error bars"""
        x_positions = []
        medians = []
        mads = []
        colors = []
        labels = []

        for i, limb in enumerate(['paw_RR', 'paw_RL']):
            if limb in aggregated_gait and 'stride_times' in aggregated_gait[limb]:
                stats = aggregated_gait[limb]['stride_times']
                if isinstance(stats, dict) and 'median' in stats:
                    x_positions.append(i)
                    medians.append(stats['median'])
                    mads.append(stats.get('mad', 0))
                    colors.append(self.style.get_limb_color(limb))
                    labels.append(get_limb_display_name(limb))

        if medians:
            self.style.plot_bar_with_error(
                ax, x_positions, medians, mads, colors, labels
            )
            self.style.add_sample_badge(ax, n_samples=len(medians))
            self.style.format_axis_range(ax, medians, zero_baseline=True)

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylabel('Stride Time (s)', fontweight='semibold')
        ax.set_title('Stride Time', fontsize=14, fontweight='semibold')

        self.style.apply_to_axis(ax)

    def plot_phase_timing_dashboard(self,
                                   gait_metrics: Dict,
                                   aggregated_gait: Dict) -> Path:
        """Enhanced phase & timing dashboard"""
        fig, axes = create_figure(2, 1, self.dpi)
        fig.suptitle('Phase & Step Timing Dashboard',
                    fontsize=18, fontweight='bold',
                    y=0.98)

        self._plot_swing_stance_enhanced(axes[0], gait_metrics, aggregated_gait)
        self._plot_phase_dispersion_enhanced(axes[1], gait_metrics, aggregated_gait)

        output_path = self.output_dir / 'plot_phase_timing.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()

        logger.info(f"Saved enhanced phase & timing dashboard: {output_path}")
        return output_path

    def _plot_swing_stance_enhanced(self, ax, gait_metrics: Dict, aggregated_gait: Dict):
        """Enhanced swing/stance ratio with asymmetry highlighting"""
        x_positions = []
        values = []
        errors = []
        colors = []
        labels = []

        for i, limb in enumerate(['paw_RR', 'paw_RL']):
            if limb in aggregated_gait and 'swing_stance_ratio' in aggregated_gait[limb]:
                stats = aggregated_gait[limb]['swing_stance_ratio']
                if isinstance(stats, dict) and 'value' in stats:
                    x_positions.append(i)
                    values.append(stats['value'])
                    errors.append(0)
                    colors.append(self.style.get_limb_color(limb))
                    labels.append(get_limb_display_name(limb))

        if values:
            self.style.plot_bar_with_error(
                ax, x_positions, values, errors, colors, labels
            )
            self.style.add_sample_badge(ax, n_samples=len(values))

            # Highlight asymmetry if >20% difference
            if len(values) == 2:
                asymmetry = abs(values[0] - values[1]) / np.mean(values)
                if asymmetry > 0.20:
                    ax.text(0.5, 0.98, f'⚠️ Asymmetry: {asymmetry*100:.1f}%',
                           transform=ax.transAxes,
                           ha='center', va='top',
                           fontsize=10, color='#e41a1c',
                           bbox=dict(boxstyle='round,pad=0.5',
                                   facecolor='#ffe6e6',
                                   edgecolor='#e41a1c',
                                   linewidth=1))

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylabel('Swing/Stance Ratio', fontweight='semibold')
        ax.set_title('Swing vs Stance', fontsize=14, fontweight='semibold')

        self.style.apply_to_axis(ax)

    def _plot_phase_dispersion_enhanced(self, ax, gait_metrics: Dict, aggregated_gait: Dict):
        """Enhanced phase dispersion"""
        x_positions = []
        values = []
        errors = []
        colors = []
        labels = []

        diagonal_map = {
            'diagonal_RR_FL': ('RH-LF', '#e41a1c'),
            'diagonal_RL_FR': ('LH-RF', '#377eb8')
        }

        for i, (diag_key, (label, color)) in enumerate(diagonal_map.items()):
            if diag_key in aggregated_gait and 'phase_dispersion' in aggregated_gait[diag_key]:
                stats = aggregated_gait[diag_key]['phase_dispersion']
                if isinstance(stats, dict) and 'value' in stats:
                    x_positions.append(i)
                    values.append(stats['value'])
                    errors.append(0)
                    colors.append(color)
                    labels.append(label)

        if values:
            self.style.plot_bar_with_error(
                ax, x_positions, values, errors, colors, labels
            )
            self.style.add_sample_badge(ax, n_samples=len(values))

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylabel('Phase Dispersion (0-1)', fontweight='semibold')
        ax.set_title('Phase Dispersion (Diagonal Pairs)', fontsize=14, fontweight='semibold')
        ax.set_ylim([0, 1.1])

        self.style.apply_to_axis(ax)

    def plot_rom_dashboard(self,
                          rom_metrics: Dict,
                          aggregated_rom: Dict) -> Path:
        """Enhanced ROM dashboard"""
        fig, axes = create_figure(2, 2, self.dpi)
        fig.suptitle('Range of Motion Dashboard',
                    fontsize=18, fontweight='bold',
                    y=0.99)

        # Flatten axes for easier indexing
        axes_flat = axes.flatten()

        self._plot_com_sway_enhanced(axes_flat[0], rom_metrics, aggregated_rom)
        self._plot_hip_asymmetry_enhanced(axes_flat[1], rom_metrics, aggregated_rom)
        self._plot_elbow_rom_enhanced(axes_flat[2], rom_metrics, aggregated_rom)
        self._plot_elbow_angular_velocity_enhanced(axes_flat[3], rom_metrics, aggregated_rom)

        output_path = self.output_dir / 'plot_range_of_motion.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()

        logger.info(f"Saved enhanced ROM dashboard: {output_path}")
        return output_path

    def _plot_com_sway_enhanced(self, ax, rom_metrics: Dict, aggregated_rom: Dict):
        """Enhanced CoM sway subplot"""
        if 'com_sway' in aggregated_rom:
            sway_data = aggregated_rom['com_sway']
            ml_sway = sway_data.get('ml_sway_cm', {}).get('value', 0)
            ap_sway = sway_data.get('ap_sway_cm', {}).get('value', 0)

            values = [ml_sway, ap_sway]
            colors = ['#4daf4a', '#e41a1c']
            labels = ['ML Sway', 'AP Sway']

            self.style.plot_bar_with_error(
                ax, [0, 1], values, [0, 0], colors, labels
            )
            self.style.add_sample_badge(ax, n_samples=2)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['ML Sway', 'AP Sway'])
        ax.set_ylabel('Sway (cm)', fontweight='semibold')
        ax.set_title('Center of Mass Sway', fontsize=14, fontweight='semibold')

        self.style.apply_to_axis(ax)

    def _plot_hip_asymmetry_enhanced(self, ax, rom_metrics: Dict, aggregated_rom: Dict):
        """Enhanced hip asymmetry (placeholder)"""
        ax.text(0.5, 0.5, 'Hip Asymmetry\n(Requires Angle Data)',
               ha='center', va='center',
               fontsize=12, color='#666666',
               transform=ax.transAxes)
        ax.set_title('Hip Asymmetry', fontsize=14, fontweight='semibold')
        ax.axis('off')

    def _plot_elbow_rom_enhanced(self, ax, rom_metrics: Dict, aggregated_rom: Dict):
        """Enhanced elbow ROM"""
        x_positions = []
        values = []
        errors = []
        colors = []
        labels = []

        for i, elbow in enumerate(['elbow_R', 'elbow_L']):
            if elbow in aggregated_rom and 'rom' in aggregated_rom[elbow]:
                stats = aggregated_rom[elbow]['rom']
                if isinstance(stats, dict) and 'value' in stats:
                    x_positions.append(i)
                    values.append(stats['value'])
                    errors.append(0)
                    colors.append('#984ea3' if elbow == 'elbow_R' else '#4daf4a')
                    labels.append(get_limb_display_name(elbow))

        if values:
            self.style.plot_bar_with_error(
                ax, x_positions, values, errors, colors, labels
            )
            self.style.add_sample_badge(ax, n_samples=len(values))

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylabel('ROM (degrees)', fontweight='semibold')
        ax.set_title('Elbow ROM', fontsize=14, fontweight='semibold')

        self.style.apply_to_axis(ax)

    def _plot_elbow_angular_velocity_enhanced(self, ax, rom_metrics: Dict, aggregated_rom: Dict):
        """Enhanced elbow angular velocity"""
        x_positions = []
        values = []
        errors = []
        colors = []
        labels = []

        for i, elbow in enumerate(['elbow_R', 'elbow_L']):
            if elbow in aggregated_rom and 'angular_velocity_mean' in aggregated_rom[elbow]:
                stats = aggregated_rom[elbow]['angular_velocity_mean']
                if isinstance(stats, dict) and 'value' in stats:
                    x_positions.append(i)
                    values.append(stats['value'])
                    errors.append(0)
                    colors.append('#984ea3' if elbow == 'elbow_R' else '#4daf4a')
                    labels.append(get_limb_display_name(elbow))

        if values:
            self.style.plot_bar_with_error(
                ax, x_positions, values, errors, colors, labels
            )
            self.style.add_sample_badge(ax, n_samples=len(values))

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylabel('Angular Velocity (deg/s)', fontweight='semibold')
        ax.set_title('Elbow Angular Velocity', fontsize=14, fontweight='semibold')

        self.style.apply_to_axis(ax)

    def generate_all_dashboards(self,
                               gait_metrics: Dict,
                               rom_metrics: Dict,
                               aggregated_gait: Dict,
                               aggregated_rom: Dict,
                               walking_windows: List[Tuple[int, int]] = None,
                               stationary_windows: List[Tuple[int, int]] = None,
                               step_results: Dict[str, Any] = None,
                               fps: float = 120.0) -> List[Path]:
        """
        Generate all enhanced dashboard plots (v1.3.0: added phase timeline).

        Args:
            gait_metrics: Raw gait metrics
            rom_metrics: Raw ROM metrics
            aggregated_gait: Aggregated gait metrics
            aggregated_rom: Aggregated ROM metrics
            walking_windows: Walking phase windows (v1.3.0)
            stationary_windows: Stationary phase windows (v1.3.0)
            step_results: Step detection results per limb (v1.3.0)
            fps: Frames per second for time conversion (v1.3.0)

        Returns:
            List of paths to generated plots
        """
        logger.info("Generating all enhanced dashboard plots")

        plots = []

        plots.append(self.plot_coordination_dashboard(gait_metrics, aggregated_gait))
        plots.append(self.plot_speed_spatial_dashboard(gait_metrics, aggregated_gait))
        plots.append(self.plot_phase_timing_dashboard(gait_metrics, aggregated_gait))
        plots.append(self.plot_rom_dashboard(rom_metrics, aggregated_rom))

        # v1.3.0: Add phase timeline plot if data is available
        # Use the base visualizer's phase timeline (no enhanced version needed yet)
        if walking_windows is not None and stationary_windows is not None and step_results is not None:
            from .visualizer import DashboardVisualizer
            base_viz = DashboardVisualizer(self.output_dir, dpi=self.dpi)
            plots.append(base_viz.plot_phase_timeline(walking_windows, stationary_windows, step_results, fps))

        logger.info(f"Generated {len(plots)} enhanced dashboard plots")

        return plots

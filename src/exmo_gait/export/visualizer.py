"""Visualization dashboard generation"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
from ..constants import PLOT_DPI

logger = logging.getLogger(__name__)

COLORS = {
    'LH': 'blue',
    'RH': 'red',
    'LF': 'green',
    'RF': 'purple',
    'COM': 'black',
    'paw_RL': 'blue',
    'paw_RR': 'red',
    'paw_FL': 'green',
    'paw_FR': 'purple'
}


class DashboardVisualizer:
    """Generate PNG dashboard plots"""

    def __init__(self, output_dir: Path, dpi: int = PLOT_DPI):
        """
        Initialize visualizer.

        Args:
            output_dir: Output directory for plots
            dpi: Resolution for PNG files
        """
        self.output_dir = output_dir
        self.dpi = dpi
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_coordination_dashboard(self,
                                   gait_metrics: Dict,
                                   aggregated_gait: Dict) -> Path:
        """
        Plot coordination dashboard (cadence, duty cycle, regularity).

        Args:
            gait_metrics: Raw gait metrics
            aggregated_gait: Aggregated gait metrics

        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Coordination Dashboard', fontsize=16, fontweight='bold')

        self._plot_cadence(axes[0], gait_metrics, aggregated_gait)
        self._plot_duty_cycle(axes[1], gait_metrics, aggregated_gait)
        self._plot_regularity_index(axes[2], gait_metrics, aggregated_gait)

        plt.tight_layout()

        output_path = self.output_dir / 'plot_coordination.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved coordination dashboard: {output_path}")
        return output_path

    def _plot_cadence(self, ax, gait_metrics: Dict, aggregated_gait: Dict):
        """Plot cadence subplot"""
        limbs = ['paw_RR', 'paw_RL']
        x_positions = []
        colors_list = []
        medians = []
        mads = []

        for i, limb in enumerate(limbs):
            if limb in aggregated_gait and 'cadence' in aggregated_gait[limb]:
                stats = aggregated_gait[limb]['cadence']
                if isinstance(stats, dict) and 'value' in stats:
                    x_positions.append(i)
                    colors_list.append(COLORS.get(limb, 'gray'))
                    medians.append(stats['value'])
                    mads.append(0)

        if medians:
            ax.bar(x_positions, medians, color=colors_list, alpha=0.7, edgecolor='black')
            ax.errorbar(x_positions, medians, yerr=mads, fmt='none', ecolor='black', capsize=5)

        ax.set_xticks(range(len(limbs)))
        ax.set_xticklabels(['RH', 'LH'])
        ax.set_ylabel('Cadence (steps/min)', fontweight='bold')
        ax.set_title('Cadence (Hind Limbs)')
        ax.grid(axis='y', alpha=0.3)

    def _plot_duty_cycle(self, ax, gait_metrics: Dict, aggregated_gait: Dict):
        """Plot duty cycle subplot with MEAN bars and individual scatter points"""
        x_pos_map = {}
        current_x = 0

        # Plot individual limbs with scatter + mean
        for limb in ['paw_RR', 'paw_RL']:
            if limb in aggregated_gait and 'duty_cycle' in aggregated_gait[limb]:
                stats = aggregated_gait[limb]['duty_cycle']
                if isinstance(stats, dict) and 'value' in stats:
                    label = 'RH' if limb == 'paw_RR' else 'LH'
                    color = COLORS.get(limb, 'gray')

                    # Mean bar
                    ax.bar(current_x, stats['value'], color=color, alpha=0.5,
                          edgecolor='black', label=label, width=0.6)

                    # Scatter individual duty cycle values per stride
                    if limb in gait_metrics and 'duty_cycle_per_stride' in gait_metrics[limb]:
                        raw_values = gait_metrics[limb]['duty_cycle_per_stride']
                        if hasattr(raw_values, '__iter__') and len(raw_values) > 0:
                            x_scatter = np.full(len(raw_values), current_x) + np.random.normal(0, 0.05, len(raw_values))
                            ax.scatter(x_scatter, raw_values, color='black', alpha=0.6,
                                     s=30, zorder=3, edgecolors='white', linewidth=0.5)

                    x_pos_map[label] = current_x
                    current_x += 1

        # Add quadruple average if available
        if 'quadrupedal' in aggregated_gait and 'avg_duty_cycle' in aggregated_gait['quadrupedal']:
            stats = aggregated_gait['quadrupedal']['avg_duty_cycle']
            if isinstance(stats, dict) and 'value' in stats:
                ax.bar(current_x, stats['value'], color='gray', alpha=0.5,
                      edgecolor='black', label='Quadruple', width=0.6)
                x_pos_map['Quadruple'] = current_x
                current_x += 1

        ax.set_xticks(list(x_pos_map.values()))
        ax.set_xticklabels(list(x_pos_map.keys()))
        ax.set_ylabel('Duty Cycle (%)', fontweight='bold')
        ax.set_title('Duty Cycle')
        ax.grid(axis='y', alpha=0.3)

    def _plot_regularity_index(self, ax, gait_metrics: Dict, aggregated_gait: Dict):
        """Plot regularity index subplot with MEAN bars, scatter points, and Quadruple"""
        x_pos_map = {}
        current_x = 0
        colors_map = {'RH-LF': 'purple', 'LH-RF': 'orange', 'Quadruple': 'gray'}

        # Diagonal pairs
        diagonal_pairs = [
            ('diagonal_RR_FL', 'RH-LF'),
            ('diagonal_RL_FR', 'LH-RF')
        ]

        for diag_key, label in diagonal_pairs:
            if diag_key in aggregated_gait and 'regularity_index' in aggregated_gait[diag_key]:
                stats = aggregated_gait[diag_key]['regularity_index']
                if isinstance(stats, dict) and 'value' in stats:
                    color = colors_map.get(label, 'gray')

                    # Mean bar
                    ax.bar(current_x, stats['value'], color=color, alpha=0.5,
                          edgecolor='black', label=label, width=0.6)

                    # Scatter individual regularity values per stride pair
                    if diag_key in gait_metrics and 'regularity_index_per_stride' in gait_metrics[diag_key]:
                        raw_values = gait_metrics[diag_key]['regularity_index_per_stride']
                        if hasattr(raw_values, '__iter__') and len(raw_values) > 0:
                            x_scatter = np.full(len(raw_values), current_x) + np.random.normal(0, 0.05, len(raw_values))
                            ax.scatter(x_scatter, raw_values, color='black', alpha=0.6,
                                     s=30, zorder=3, edgecolors='white', linewidth=0.5)

                    x_pos_map[label] = current_x
                    current_x += 1

        # Add quadruple regularity index if available
        if 'quadrupedal' in aggregated_gait and 'regularity_index' in aggregated_gait['quadrupedal']:
            stats = aggregated_gait['quadrupedal']['regularity_index']
            if isinstance(stats, dict) and 'value' in stats:
                ax.bar(current_x, stats['value'], color='gray', alpha=0.5,
                      edgecolor='black', label='Quadruple', width=0.6)
                x_pos_map['Quadruple'] = current_x
                current_x += 1

        ax.set_xticks(list(x_pos_map.values()))
        ax.set_xticklabels(list(x_pos_map.keys()))
        ax.set_ylabel('Regularity Index (0-1)', fontweight='bold')
        ax.set_title('Regularity Index')
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)

    def plot_speed_spatial_dashboard(self,
                                    gait_metrics: Dict,
                                    aggregated_gait: Dict) -> Path:
        """
        Plot speed and spatial metrics dashboard.

        Args:
            gait_metrics: Raw gait metrics
            aggregated_gait: Aggregated gait metrics

        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Speed & Spatial Metrics Dashboard', fontsize=16, fontweight='bold')

        self._plot_avg_speed(axes[0], gait_metrics, aggregated_gait)
        self._plot_stride_length(axes[1], gait_metrics, aggregated_gait)
        self._plot_stride_time(axes[2], gait_metrics, aggregated_gait)

        plt.tight_layout()

        output_path = self.output_dir / 'plot_speed_spatial.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved speed & spatial dashboard: {output_path}")
        return output_path

    def _plot_avg_speed(self, ax, gait_metrics: Dict, aggregated_gait: Dict):
        """Plot average speed subplot with MEAN bars and individual scatter points"""
        x_pos_map = {}
        current_x = 0

        # CoM average speed
        if 'whole_body' in aggregated_gait and 'com_avg_speed' in aggregated_gait['whole_body']:
            stats = aggregated_gait['whole_body']['com_avg_speed']
            if isinstance(stats, dict) and 'value' in stats:
                ax.bar(current_x, stats['value'], color='black', alpha=0.5,
                      edgecolor='black', label='CoM', width=0.6)

                # Scatter individual CoM speed values per stride/step
                if 'whole_body' in gait_metrics and 'com_speed_per_stride' in gait_metrics['whole_body']:
                    raw_values = gait_metrics['whole_body']['com_speed_per_stride']
                    if hasattr(raw_values, '__iter__') and len(raw_values) > 0:
                        x_scatter = np.full(len(raw_values), current_x) + np.random.normal(0, 0.05, len(raw_values))
                        ax.scatter(x_scatter, raw_values, color='gray', alpha=0.6,
                                 s=30, zorder=3, edgecolors='white', linewidth=0.5)

                x_pos_map['CoM'] = current_x
                current_x += 1

        # Individual limb average speeds
        for limb in ['paw_RR', 'paw_RL']:
            if limb in aggregated_gait and 'avg_speed' in aggregated_gait[limb]:
                stats = aggregated_gait[limb]['avg_speed']
                if isinstance(stats, dict) and 'value' in stats:
                    label = 'RH' if limb == 'paw_RR' else 'LH'
                    color = COLORS.get(limb, 'gray')

                    ax.bar(current_x, stats['value'], color=color, alpha=0.5,
                          edgecolor='black', label=label, width=0.6)

                    # Scatter individual speed values per stride
                    if limb in gait_metrics and 'avg_speed_per_stride' in gait_metrics[limb]:
                        raw_values = gait_metrics[limb]['avg_speed_per_stride']
                        if hasattr(raw_values, '__iter__') and len(raw_values) > 0:
                            x_scatter = np.full(len(raw_values), current_x) + np.random.normal(0, 0.05, len(raw_values))
                            ax.scatter(x_scatter, raw_values, color='black', alpha=0.6,
                                     s=30, zorder=3, edgecolors='white', linewidth=0.5)

                    x_pos_map[label] = current_x
                    current_x += 1

        ax.set_xticks(list(x_pos_map.values()))
        ax.set_xticklabels(list(x_pos_map.keys()))
        ax.set_ylabel('Speed (cm/s)', fontweight='bold')
        ax.set_title('Average Speed (Walking)')
        ax.grid(axis='y', alpha=0.3)

    def _plot_stride_length(self, ax, gait_metrics: Dict, aggregated_gait: Dict):
        """Plot stride length subplot with MEDIAN, MAD error bars, and individual data points"""
        for limb in ['paw_RR', 'paw_RL']:
            if limb in aggregated_gait and 'stride_lengths' in aggregated_gait[limb]:
                stats = aggregated_gait[limb]['stride_lengths']
                if isinstance(stats, dict) and 'median' in stats and 'mad' in stats:
                    label = 'RH' if limb == 'paw_RR' else 'LH'
                    color = COLORS.get(limb, 'gray')

                    x_pos = 0 if limb == 'paw_RR' else 1
                    # Bar shows MEDIAN with MAD error bars (robust statistics for stride events)
                    ax.bar(x_pos, stats['median'], color=color, alpha=0.5,
                          edgecolor='black', label=label, width=0.6)
                    ax.errorbar(x_pos, stats['median'], yerr=stats['mad'],
                              fmt='none', ecolor='black', capsize=5, linewidth=2)

                    # Scatter individual stride values
                    if limb in gait_metrics and 'stride_lengths' in gait_metrics[limb]:
                        raw_values = gait_metrics[limb]['stride_lengths']
                        if hasattr(raw_values, '__iter__') and len(raw_values) > 0:
                            # Add jitter to x position for visibility
                            x_scatter = np.full(len(raw_values), x_pos) + np.random.normal(0, 0.05, len(raw_values))
                            ax.scatter(x_scatter, raw_values, color='black', alpha=0.6,
                                     s=30, zorder=3, edgecolors='white', linewidth=0.5)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['RH', 'LH'])
        ax.set_ylabel('Stride Length (cm)', fontweight='bold')
        ax.set_title('Stride Length')
        ax.grid(axis='y', alpha=0.3)

    def _plot_stride_time(self, ax, gait_metrics: Dict, aggregated_gait: Dict):
        """Plot stride time subplot with MEDIAN, MAD error bars, and individual data points"""
        for limb in ['paw_RR', 'paw_RL']:
            if limb in aggregated_gait and 'stride_times' in aggregated_gait[limb]:
                stats = aggregated_gait[limb]['stride_times']
                if isinstance(stats, dict) and 'median' in stats and 'mad' in stats:
                    label = 'RH' if limb == 'paw_RR' else 'LH'
                    color = COLORS.get(limb, 'gray')

                    x_pos = 0 if limb == 'paw_RR' else 1
                    # Bar shows MEDIAN with MAD error bars (robust statistics for stride events)
                    ax.bar(x_pos, stats['median'], color=color, alpha=0.5,
                          edgecolor='black', label=label, width=0.6)
                    ax.errorbar(x_pos, stats['median'], yerr=stats['mad'],
                              fmt='none', ecolor='black', capsize=5, linewidth=2)

                    # Scatter individual stride values
                    if limb in gait_metrics and 'stride_times' in gait_metrics[limb]:
                        raw_values = gait_metrics[limb]['stride_times']
                        if hasattr(raw_values, '__iter__') and len(raw_values) > 0:
                            # Add jitter to x position for visibility
                            x_scatter = np.full(len(raw_values), x_pos) + np.random.normal(0, 0.05, len(raw_values))
                            ax.scatter(x_scatter, raw_values, color='black', alpha=0.6,
                                     s=30, zorder=3, edgecolors='white', linewidth=0.5)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['RH', 'LH'])
        ax.set_ylabel('Stride Time (s)', fontweight='bold')
        ax.set_title('Stride Time')
        ax.grid(axis='y', alpha=0.3)

    def plot_phase_timing_dashboard(self,
                                   gait_metrics: Dict,
                                   aggregated_gait: Dict) -> Path:
        """
        Plot phase and timing dashboard.

        Args:
            gait_metrics: Raw gait metrics
            aggregated_gait: Aggregated gait metrics

        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle('Phase & Step Timing Dashboard', fontsize=16, fontweight='bold')

        self._plot_swing_stance(axes[0], gait_metrics, aggregated_gait)
        self._plot_phase_dispersion(axes[1], gait_metrics, aggregated_gait)

        plt.tight_layout()

        output_path = self.output_dir / 'plot_phase_timing.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved phase & timing dashboard: {output_path}")
        return output_path

    def _plot_swing_stance(self, ax, gait_metrics: Dict, aggregated_gait: Dict):
        """Plot swing vs stance subplot with MEAN bars and individual scatter points"""
        for limb in ['paw_RR', 'paw_RL']:
            if limb in aggregated_gait and 'swing_stance_ratio' in aggregated_gait[limb]:
                stats = aggregated_gait[limb]['swing_stance_ratio']
                if isinstance(stats, dict) and 'value' in stats:
                    label = 'RH' if limb == 'paw_RR' else 'LH'
                    color = COLORS.get(limb, 'gray')

                    x_pos = 0 if limb == 'paw_RR' else 1
                    # Mean bar
                    ax.bar(x_pos, stats['value'], color=color, alpha=0.5,
                          edgecolor='black', label=label, width=0.6)

                    # Scatter individual swing/stance ratio values per stride
                    if limb in gait_metrics and 'swing_stance_ratio_per_stride' in gait_metrics[limb]:
                        raw_values = gait_metrics[limb]['swing_stance_ratio_per_stride']
                        if hasattr(raw_values, '__iter__') and len(raw_values) > 0:
                            x_scatter = np.full(len(raw_values), x_pos) + np.random.normal(0, 0.05, len(raw_values))
                            ax.scatter(x_scatter, raw_values, color='black', alpha=0.6,
                                     s=30, zorder=3, edgecolors='white', linewidth=0.5)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['RH', 'LH'])
        ax.set_ylabel('Swing/Stance Ratio', fontweight='bold')
        ax.set_title('Swing vs Stance')
        ax.grid(axis='y', alpha=0.3)

    def _plot_phase_dispersion(self, ax, gait_metrics: Dict, aggregated_gait: Dict):
        """Plot phase dispersion subplot with MEAN bars and individual scatter points"""
        x_pos_map = {}
        current_x = 0
        colors_map = {'RH-LF': 'purple', 'LH-RF': 'orange'}

        diagonal_pairs = [
            ('diagonal_RR_FL', 'RH-LF'),
            ('diagonal_RL_FR', 'LH-RF')
        ]

        for diag_key, label in diagonal_pairs:
            if diag_key in aggregated_gait and 'phase_dispersion' in aggregated_gait[diag_key]:
                stats = aggregated_gait[diag_key]['phase_dispersion']
                if isinstance(stats, dict) and 'value' in stats:
                    color = colors_map.get(label, 'gray')

                    # Mean bar
                    ax.bar(current_x, stats['value'], color=color, alpha=0.5,
                          edgecolor='black', label=label, width=0.6)

                    # Scatter individual phase dispersion values per stride pair
                    if diag_key in gait_metrics and 'phase_dispersion_per_stride' in gait_metrics[diag_key]:
                        raw_values = gait_metrics[diag_key]['phase_dispersion_per_stride']
                        if hasattr(raw_values, '__iter__') and len(raw_values) > 0:
                            x_scatter = np.full(len(raw_values), current_x) + np.random.normal(0, 0.05, len(raw_values))
                            ax.scatter(x_scatter, raw_values, color='black', alpha=0.6,
                                     s=30, zorder=3, edgecolors='white', linewidth=0.5)

                    x_pos_map[label] = current_x
                    current_x += 1

        ax.set_xticks(list(x_pos_map.values()))
        ax.set_xticklabels(list(x_pos_map.keys()))
        ax.set_ylabel('Phase Dispersion', fontweight='bold')
        ax.set_title('Phase Dispersion (Diagonal Pairs)')
        ax.grid(axis='y', alpha=0.3)

    def plot_rom_dashboard(self,
                          rom_metrics: Dict,
                          aggregated_rom: Dict) -> Path:
        """
        Plot range of motion dashboard.

        Args:
            rom_metrics: Raw ROM metrics
            aggregated_rom: Aggregated ROM metrics

        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Range of Motion Dashboard', fontsize=16, fontweight='bold')

        self._plot_com_sway(axes[0, 0], rom_metrics, aggregated_rom)
        self._plot_hip_asymmetry(axes[0, 1], rom_metrics, aggregated_rom)
        self._plot_elbow_rom(axes[1, 0], rom_metrics, aggregated_rom)
        self._plot_elbow_angular_velocity(axes[1, 1], rom_metrics, aggregated_rom)

        plt.tight_layout()

        output_path = self.output_dir / 'plot_range_of_motion.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved ROM dashboard: {output_path}")
        return output_path

    def _plot_com_sway(self, ax, rom_metrics: Dict, aggregated_rom: Dict):
        """Plot CoM sway subplot with MEAN bars and individual scatter points"""
        if 'com_sway' in aggregated_rom:
            sway_data = aggregated_rom['com_sway']
            ml_sway = sway_data.get('ml_sway_cm', {}).get('value', 0)
            ap_sway = sway_data.get('ap_sway_cm', {}).get('value', 0)

            # Mean bars
            ax.bar([0, 1], [ml_sway, ap_sway],
                  color=['steelblue', 'coral'], alpha=0.5, edgecolor='black', width=0.6)

            # Scatter individual sway values per stride/step
            if 'com_sway' in rom_metrics:
                # ML sway per stride
                if 'ml_sway_per_stride' in rom_metrics['com_sway']:
                    ml_values = rom_metrics['com_sway']['ml_sway_per_stride']
                    if hasattr(ml_values, '__iter__') and len(ml_values) > 0:
                        x_scatter = np.full(len(ml_values), 0) + np.random.normal(0, 0.05, len(ml_values))
                        ax.scatter(x_scatter, ml_values, color='black', alpha=0.6,
                                 s=30, zorder=3, edgecolors='white', linewidth=0.5)

                # AP sway per stride
                if 'ap_sway_per_stride' in rom_metrics['com_sway']:
                    ap_values = rom_metrics['com_sway']['ap_sway_per_stride']
                    if hasattr(ap_values, '__iter__') and len(ap_values) > 0:
                        x_scatter = np.full(len(ap_values), 1) + np.random.normal(0, 0.05, len(ap_values))
                        ax.scatter(x_scatter, ap_values, color='black', alpha=0.6,
                                 s=30, zorder=3, edgecolors='white', linewidth=0.5)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['ML Sway', 'AP Sway'])
        ax.set_ylabel('Sway (cm)', fontweight='bold')
        ax.set_title('Center of Mass Sway')
        ax.grid(axis='y', alpha=0.3)

    def _plot_hip_asymmetry(self, ax, rom_metrics: Dict, aggregated_rom: Dict):
        """Plot hip asymmetry subplot"""
        ax.text(0.5, 0.5, 'Hip asymmetry\nrequires angle data',
               ha='center', va='center', fontsize=12)
        ax.set_title('Hip Asymmetry')
        ax.axis('off')

    def _plot_elbow_rom(self, ax, rom_metrics: Dict, aggregated_rom: Dict):
        """Plot elbow ROM subplot with MEAN bars and individual scatter points"""
        x_pos_map = {}
        current_x = 0
        colors_map = {'Right': 'purple', 'Left': 'green'}

        for elbow in ['elbow_R', 'elbow_L']:
            if elbow in aggregated_rom and 'rom' in aggregated_rom[elbow]:
                stats = aggregated_rom[elbow]['rom']
                if isinstance(stats, dict) and 'value' in stats:
                    label = 'Right' if elbow == 'elbow_R' else 'Left'
                    color = colors_map.get(label, 'gray')

                    # Mean bar
                    ax.bar(current_x, stats['value'], color=color, alpha=0.5,
                          edgecolor='black', label=label, width=0.6)

                    # Scatter individual ROM values per frame/stride
                    if elbow in rom_metrics and 'rom_per_frame' in rom_metrics[elbow]:
                        raw_values = rom_metrics[elbow]['rom_per_frame']
                        if hasattr(raw_values, '__iter__') and len(raw_values) > 0:
                            x_scatter = np.full(len(raw_values), current_x) + np.random.normal(0, 0.05, len(raw_values))
                            ax.scatter(x_scatter, raw_values, color='black', alpha=0.6,
                                     s=30, zorder=3, edgecolors='white', linewidth=0.5)

                    x_pos_map[label] = current_x
                    current_x += 1

        ax.set_xticks(list(x_pos_map.values()))
        ax.set_xticklabels(list(x_pos_map.keys()))
        ax.set_ylabel('ROM (degrees)', fontweight='bold')
        ax.set_title('Elbow ROM')
        ax.grid(axis='y', alpha=0.3)

    def _plot_elbow_angular_velocity(self, ax, rom_metrics: Dict, aggregated_rom: Dict):
        """Plot elbow angular velocity subplot"""
        velocities = []
        labels = []

        for elbow in ['elbow_R', 'elbow_L']:
            if elbow in aggregated_rom and 'angular_velocity_mean' in aggregated_rom[elbow]:
                stats = aggregated_rom[elbow]['angular_velocity_mean']
                if isinstance(stats, dict) and 'value' in stats:
                    velocities.append(stats['value'])
                    labels.append('Right' if elbow == 'elbow_R' else 'Left')

        if velocities:
            ax.bar(range(len(velocities)), velocities,
                  color=['purple', 'green'][:len(velocities)],
                  alpha=0.7, edgecolor='black')

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylabel('Angular Velocity (deg/s)', fontweight='bold')
        ax.set_title('Elbow Angular Velocity')
        ax.grid(axis='y', alpha=0.3)

    def plot_phase_timeline(self,
                           walking_windows: List[Tuple[int, int]],
                           stationary_windows: List[Tuple[int, int]],
                           step_results: Dict[str, Any],
                           fps: float = 120.0) -> Path:
        """
        Plot walking/stationary phase timeline with step frequency (v1.3.0).

        Shows temporal distribution of walking vs stationary phases with
        step event markers to visualize gait activity over time.

        Args:
            walking_windows: List of (start_frame, end_frame) for walking periods
            stationary_windows: List of (start_frame, end_frame) for stationary periods
            step_results: Dict containing foot strike data per limb
            fps: Frames per second for time conversion

        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(1, 1, figsize=(14, 6))
        fig.suptitle('Phase Timeline & Step Frequency', fontsize=16, fontweight='bold')

        # Determine total recording duration
        max_frame = 0
        for start, end in walking_windows + stationary_windows:
            max_frame = max(max_frame, end)

        # Add step events to determine range
        for limb_name, limb_data in step_results.items():
            if 'foot_strikes' in limb_data and limb_data['foot_strikes'] is not None:
                strikes = limb_data['foot_strikes']
                if len(strikes) > 0:
                    max_frame = max(max_frame, np.max(strikes))

        if max_frame == 0:
            max_frame = 1000  # Default fallback

        time_max = max_frame / fps

        # Plot phase background shading
        for start, end in stationary_windows:
            ax.axvspan(start / fps, end / fps, alpha=0.2, color='gray', label='Stationary' if start == stationary_windows[0][0] else '')

        for start, end in walking_windows:
            ax.axvspan(start / fps, end / fps, alpha=0.3, color='green', label='Walking' if start == walking_windows[0][0] else '')

        # Plot step events as scatter points
        y_positions = {'paw_RR': 3, 'paw_RL': 2, 'paw_FR': 1, 'paw_FL': 0}
        y_labels = ['FL', 'FR', 'LH', 'RH']

        for limb_name, y_pos in y_positions.items():
            if limb_name in step_results and 'foot_strikes' in step_results[limb_name]:
                strikes = step_results[limb_name]['foot_strikes']
                if strikes is not None and len(strikes) > 0:
                    strike_times = strikes / fps
                    color = COLORS.get(limb_name, 'black')
                    ax.scatter(strike_times, [y_pos] * len(strike_times),
                             color=color, s=50, alpha=0.7, marker='|', linewidths=2,
                             label=y_labels[y_pos])

        # Formatting
        ax.set_xlabel('Time (s)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Limb', fontweight='bold', fontsize=12)
        ax.set_yticks(list(y_positions.values()))
        ax.set_yticklabels(y_labels)
        ax.set_xlim(0, time_max)
        ax.set_ylim(-0.5, 3.5)
        ax.grid(axis='x', alpha=0.3)

        # Legend
        handles, labels = ax.get_legend_handles_labels()
        # Remove duplicate labels
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10)

        plt.tight_layout()

        output_path = self.output_dir / 'plot_phase_timeline.png'
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved phase timeline plot: {output_path}")
        return output_path

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
        Generate all dashboard plots (v1.3.0: added phase timeline).

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
        logger.info("Generating all dashboard plots")

        plots = []

        plots.append(self.plot_coordination_dashboard(gait_metrics, aggregated_gait))
        plots.append(self.plot_speed_spatial_dashboard(gait_metrics, aggregated_gait))
        plots.append(self.plot_phase_timing_dashboard(gait_metrics, aggregated_gait))
        plots.append(self.plot_rom_dashboard(rom_metrics, aggregated_rom))

        # v1.3.0: Add phase timeline plot if data is available
        if walking_windows is not None and stationary_windows is not None and step_results is not None:
            plots.append(self.plot_phase_timeline(walking_windows, stationary_windows, step_results, fps))

        logger.info(f"Generated {len(plots)} dashboard plots")

        return plots

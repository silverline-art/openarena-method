"""Visualization dashboard generation"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List
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
        """Plot duty cycle subplot"""
        pairs = []
        labels = []

        for limb in ['paw_RR', 'paw_RL']:
            if limb in aggregated_gait and 'duty_cycle' in aggregated_gait[limb]:
                stats = aggregated_gait[limb]['duty_cycle']
                if isinstance(stats, dict) and 'value' in stats:
                    pairs.append(stats['value'])
                    labels.append('RH' if limb == 'paw_RR' else 'LH')

        if 'quadrupedal' in aggregated_gait and 'avg_duty_cycle' in aggregated_gait['quadrupedal']:
            stats = aggregated_gait['quadrupedal']['avg_duty_cycle']
            if isinstance(stats, dict) and 'value' in stats:
                pairs.append(stats['value'])
                labels.append('Quadruple')

        if pairs:
            colors_plot = ['red', 'blue', 'gray'][:len(pairs)]
            ax.bar(range(len(pairs)), pairs, color=colors_plot, alpha=0.7, edgecolor='black')

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylabel('Duty Cycle (%)', fontweight='bold')
        ax.set_title('Duty Cycle')
        ax.grid(axis='y', alpha=0.3)

    def _plot_regularity_index(self, ax, gait_metrics: Dict, aggregated_gait: Dict):
        """Plot regularity index subplot"""
        pairs = []
        labels = []

        if 'diagonal_RR_FL' in aggregated_gait and 'regularity_index' in aggregated_gait['diagonal_RR_FL']:
            stats = aggregated_gait['diagonal_RR_FL']['regularity_index']
            if isinstance(stats, dict) and 'value' in stats:
                pairs.append(stats['value'])
                labels.append('RH-LF')

        if 'diagonal_RL_FR' in aggregated_gait and 'regularity_index' in aggregated_gait['diagonal_RL_FR']:
            stats = aggregated_gait['diagonal_RL_FR']['regularity_index']
            if isinstance(stats, dict) and 'value' in stats:
                pairs.append(stats['value'])
                labels.append('LH-RF')

        if pairs:
            ax.bar(range(len(pairs)), pairs, color=['purple', 'orange'][:len(pairs)],
                  alpha=0.7, edgecolor='black')

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylabel('Regularity Index (0-1)', fontweight='bold')
        ax.set_title('Regularity Index (Diagonal Pairs)')
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
        """Plot average speed subplot"""
        speeds = []
        labels = []
        colors_list = []

        if 'whole_body' in aggregated_gait and 'com_avg_speed' in aggregated_gait['whole_body']:
            stats = aggregated_gait['whole_body']['com_avg_speed']
            if isinstance(stats, dict) and 'value' in stats:
                speeds.append(stats['value'])
                labels.append('CoM')
                colors_list.append('black')

        for limb in ['paw_RR', 'paw_RL']:
            if limb in aggregated_gait and 'avg_speed' in aggregated_gait[limb]:
                stats = aggregated_gait[limb]['avg_speed']
                if isinstance(stats, dict) and 'value' in stats:
                    speeds.append(stats['value'])
                    labels.append('RH' if limb == 'paw_RR' else 'LH')
                    colors_list.append(COLORS.get(limb, 'gray'))

        if speeds:
            ax.bar(range(len(speeds)), speeds, color=colors_list, alpha=0.7, edgecolor='black')

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylabel('Speed (cm/s)', fontweight='bold')
        ax.set_title('Average Speed')
        ax.grid(axis='y', alpha=0.3)

    def _plot_stride_length(self, ax, gait_metrics: Dict, aggregated_gait: Dict):
        """Plot stride length subplot"""
        for limb in ['paw_RR', 'paw_RL']:
            if limb in aggregated_gait and 'stride_lengths' in aggregated_gait[limb]:
                stats = aggregated_gait[limb]['stride_lengths']
                if isinstance(stats, dict) and 'median' in stats:
                    label = 'RH' if limb == 'paw_RR' else 'LH'
                    color = COLORS.get(limb, 'gray')

                    x_pos = 0 if limb == 'paw_RR' else 1
                    ax.bar(x_pos, stats['median'], color=color, alpha=0.7,
                          edgecolor='black', label=label)
                    ax.errorbar(x_pos, stats['median'], yerr=stats['mad'],
                              fmt='none', ecolor='black', capsize=5)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['RH', 'LH'])
        ax.set_ylabel('Stride Length (cm)', fontweight='bold')
        ax.set_title('Stride Length')
        ax.grid(axis='y', alpha=0.3)

    def _plot_stride_time(self, ax, gait_metrics: Dict, aggregated_gait: Dict):
        """Plot stride time subplot"""
        for limb in ['paw_RR', 'paw_RL']:
            if limb in aggregated_gait and 'stride_times' in aggregated_gait[limb]:
                stats = aggregated_gait[limb]['stride_times']
                if isinstance(stats, dict) and 'median' in stats:
                    label = 'RH' if limb == 'paw_RR' else 'LH'
                    color = COLORS.get(limb, 'gray')

                    x_pos = 0 if limb == 'paw_RR' else 1
                    ax.bar(x_pos, stats['median'], color=color, alpha=0.7,
                          edgecolor='black', label=label)
                    ax.errorbar(x_pos, stats['median'], yerr=stats['mad'],
                              fmt='none', ecolor='black', capsize=5)

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
        """Plot swing vs stance subplot"""
        for limb in ['paw_RR', 'paw_RL']:
            if limb in aggregated_gait and 'swing_stance_ratio' in aggregated_gait[limb]:
                stats = aggregated_gait[limb]['swing_stance_ratio']
                if isinstance(stats, dict) and 'value' in stats:
                    label = 'RH' if limb == 'paw_RR' else 'LH'
                    color = COLORS.get(limb, 'gray')

                    x_pos = 0 if limb == 'paw_RR' else 1
                    ax.bar(x_pos, stats['value'], color=color, alpha=0.7,
                          edgecolor='black', label=label)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(['RH', 'LH'])
        ax.set_ylabel('Swing/Stance Ratio', fontweight='bold')
        ax.set_title('Swing vs Stance')
        ax.grid(axis='y', alpha=0.3)

    def _plot_phase_dispersion(self, ax, gait_metrics: Dict, aggregated_gait: Dict):
        """Plot phase dispersion subplot"""
        dispersions = []
        labels = []

        if 'diagonal_RR_FL' in aggregated_gait and 'phase_dispersion' in aggregated_gait['diagonal_RR_FL']:
            stats = aggregated_gait['diagonal_RR_FL']['phase_dispersion']
            if isinstance(stats, dict) and 'value' in stats:
                dispersions.append(stats['value'])
                labels.append('RH-LF')

        if 'diagonal_RL_FR' in aggregated_gait and 'phase_dispersion' in aggregated_gait['diagonal_RL_FR']:
            stats = aggregated_gait['diagonal_RL_FR']['phase_dispersion']
            if isinstance(stats, dict) and 'value' in stats:
                dispersions.append(stats['value'])
                labels.append('LH-RF')

        if dispersions:
            ax.bar(range(len(dispersions)), dispersions,
                  color=['purple', 'orange'][:len(dispersions)],
                  alpha=0.7, edgecolor='black')

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
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
        """Plot CoM sway subplot"""
        if 'com_sway' in aggregated_rom:
            sway_data = aggregated_rom['com_sway']
            ml_sway = sway_data.get('ml_sway_cm', {}).get('value', 0)
            ap_sway = sway_data.get('ap_sway_cm', {}).get('value', 0)

            ax.bar([0, 1], [ml_sway, ap_sway],
                  color=['steelblue', 'coral'], alpha=0.7, edgecolor='black')

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
        """Plot elbow ROM subplot"""
        roms = []
        labels = []

        for elbow in ['elbow_R', 'elbow_L']:
            if elbow in aggregated_rom and 'rom' in aggregated_rom[elbow]:
                stats = aggregated_rom[elbow]['rom']
                if isinstance(stats, dict) and 'value' in stats:
                    roms.append(stats['value'])
                    labels.append('Right' if elbow == 'elbow_R' else 'Left')

        if roms:
            ax.bar(range(len(roms)), roms, color=['purple', 'green'][:len(roms)],
                  alpha=0.7, edgecolor='black')

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
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

    def generate_all_dashboards(self,
                               gait_metrics: Dict,
                               rom_metrics: Dict,
                               aggregated_gait: Dict,
                               aggregated_rom: Dict) -> List[Path]:
        """
        Generate all dashboard plots.

        Args:
            gait_metrics: Raw gait metrics
            rom_metrics: Raw ROM metrics
            aggregated_gait: Aggregated gait metrics
            aggregated_rom: Aggregated ROM metrics

        Returns:
            List of paths to generated plots
        """
        logger.info("Generating all dashboard plots")

        plots = []

        plots.append(self.plot_coordination_dashboard(gait_metrics, aggregated_gait))
        plots.append(self.plot_speed_spatial_dashboard(gait_metrics, aggregated_gait))
        plots.append(self.plot_phase_timing_dashboard(gait_metrics, aggregated_gait))
        plots.append(self.plot_rom_dashboard(rom_metrics, aggregated_rom))

        logger.info(f"Generated {len(plots)} dashboard plots")

        return plots

"""
EXMO Gait Analysis - Publication-Grade Visual Style System
Professional, color-blind-safe styling for scientific visualization
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict, Optional
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# EXMO SCIENTIFIC COLOR PALETTE (Color-Blind Safe)
# ═══════════════════════════════════════════════════════════════════════════

EXMO_COLORS = {
    # Limb-specific colors (Paul Tol's colorblind-safe palette)
    'paw_RL': '#377eb8',  # Blue (Left Hind)
    'paw_RR': '#e41a1c',  # Red (Right Hind)
    'paw_FL': '#4daf4a',  # Green (Left Fore)
    'paw_FR': '#984ea3',  # Purple (Right Fore)

    # Aliases for compatibility
    'LH': '#377eb8',
    'RH': '#e41a1c',
    'LF': '#4daf4a',
    'RF': '#984ea3',

    # Body/CoM
    'COM': '#000000',
    'body': '#000000',

    # Diagonal pairs
    'diagonal_RR_FL': '#e41a1c',  # Red-Green
    'diagonal_RL_FR': '#377eb8',  # Blue-Purple

    # Grid and background
    'grid': '#bbbbbb',
    'background': '#ffffff',

    # Reference bands
    'ref_band': '#e0e0e0',
    'ref_line': '#666666'
}


# ═══════════════════════════════════════════════════════════════════════════
# TYPOGRAPHY HIERARCHY
# ═══════════════════════════════════════════════════════════════════════════

FONT_SIZES = {
    'title': 18,           # Main dashboard title
    'subtitle': 14,        # Subplot titles
    'axis_label': 12,      # X/Y axis labels
    'tick_label': 11,      # Tick marks
    'legend': 11,          # Legend text
    'annotation': 9,       # Inline annotations
    'badge': 9             # Sample count badges
}

FONT_WEIGHTS = {
    'title': 'bold',
    'subtitle': 'semibold',
    'axis_label': 'semibold',
    'normal': 'normal'
}


# ═══════════════════════════════════════════════════════════════════════════
# MARKER & LINE SPECIFICATIONS
# ═══════════════════════════════════════════════════════════════════════════

MARKER_SPECS = {
    'size': 60,                  # Scatter marker size
    'size_median': 120,          # Median marker size (larger)
    'edge_width': 0.3,           # Marker edge line width
    'edge_color': '#000000',     # Marker edge color
    'alpha': 0.75,               # Marker transparency
    'median_marker': 'D',        # Diamond for median
    'data_marker': 'o'           # Circle for data points
}

LINE_SPECS = {
    'width_body': 2.0,           # CoM/body lines
    'width_limb': 1.5,           # Limb trajectory lines
    'width_error': 1.2,          # Error bar line width
    'width_ref': 1.0,            # Reference line width
    'style_grid': '--',          # Grid line style (dashed)
    'alpha_line': 0.8,           # Line transparency
    'alpha_shadow': 0.1          # Shadow transparency
}

ERROR_BAR_SPECS = {
    'capsize': 5,                # Error bar cap size
    'linewidth': 1.2,            # Error bar line width
    'alpha': 0.85                # Error bar transparency
}


# ═══════════════════════════════════════════════════════════════════════════
# LAYOUT SPECIFICATIONS
# ═══════════════════════════════════════════════════════════════════════════

LAYOUT_SPECS = {
    'dpi': 600,                  # Publication-grade DPI
    'dpi_screen': 300,           # Screen/preview DPI
    'fig_width': 12,             # Standard figure width (inches)
    'fig_height_single': 6,      # Single-row dashboard height
    'fig_height_double': 12,     # Double-row dashboard height
    'wspace': 0.25,              # Horizontal subplot spacing
    'hspace': 0.30,              # Vertical subplot spacing
    'margin': 0.10,              # Axis margin (10%)
    'use_constrained_layout': True
}


# ═══════════════════════════════════════════════════════════════════════════
# GRID & BACKGROUND SPECIFICATIONS
# ═══════════════════════════════════════════════════════════════════════════

GRID_SPECS = {
    'color': '#bbbbbb',
    'linestyle': '--',
    'linewidth': 0.6,
    'alpha': 0.3,
    'axis': 'y',                 # Grid only on Y-axis
    'which': 'major'
}


# ═══════════════════════════════════════════════════════════════════════════
# REFERENCE BANDS (Normal Ranges)
# ═══════════════════════════════════════════════════════════════════════════

REFERENCE_BANDS = {
    'cadence': {
        'min': 180,              # steps/min
        'max': 240,
        'label': 'Normal Range'
    },
    'duty_cycle': {
        'min': 50,               # %
        'max': 70,
        'label': 'Normal Range'
    },
    'regularity_index': {
        'min': 0.8,              # 0-1 scale
        'max': 1.0,
        'label': 'Healthy'
    }
}


# ═══════════════════════════════════════════════════════════════════════════
# PLOT FACTORY CLASS
# ═══════════════════════════════════════════════════════════════════════════

class EXMOPlotStyle:
    """
    Centralized style management for EXMO Gait Analysis plots.
    Applies publication-grade styling consistently across all dashboards.
    """

    def __init__(self,
                 dpi: int = 600,
                 marker_size: int = 60,
                 font_scale: float = 1.0,
                 use_exmo_palette: bool = True,
                 annotate_median: bool = True):
        """
        Initialize EXMO plot style manager.

        Args:
            dpi: Resolution for output (300=screen, 600=publication)
            marker_size: Base marker size for scatter plots
            font_scale: Global font size multiplier
            use_exmo_palette: Use EXMO color-blind-safe palette
            annotate_median: Add median value annotations
        """
        self.dpi = dpi
        self.marker_size = marker_size
        self.font_scale = font_scale
        self.use_exmo_palette = use_exmo_palette
        self.annotate_median = annotate_median

        # Apply global matplotlib style
        self._configure_matplotlib()

    def _configure_matplotlib(self):
        """Configure global matplotlib settings"""
        mpl.rcParams['font.family'] = 'DejaVu Sans'
        mpl.rcParams['font.size'] = FONT_SIZES['tick_label'] * self.font_scale
        mpl.rcParams['axes.labelsize'] = FONT_SIZES['axis_label'] * self.font_scale
        mpl.rcParams['axes.titlesize'] = FONT_SIZES['subtitle'] * self.font_scale
        mpl.rcParams['xtick.labelsize'] = FONT_SIZES['tick_label'] * self.font_scale
        mpl.rcParams['ytick.labelsize'] = FONT_SIZES['tick_label'] * self.font_scale
        mpl.rcParams['legend.fontsize'] = FONT_SIZES['legend'] * self.font_scale
        mpl.rcParams['figure.titlesize'] = FONT_SIZES['title'] * self.font_scale

        # Modern clean style
        mpl.rcParams['axes.spines.top'] = False
        mpl.rcParams['axes.spines.right'] = False
        mpl.rcParams['axes.edgecolor'] = '#333333'
        mpl.rcParams['axes.linewidth'] = 1.0

        # Grid
        mpl.rcParams['grid.color'] = GRID_SPECS['color']
        mpl.rcParams['grid.linestyle'] = GRID_SPECS['linestyle']
        mpl.rcParams['grid.linewidth'] = GRID_SPECS['linewidth']
        mpl.rcParams['grid.alpha'] = GRID_SPECS['alpha']

    def apply_to_axis(self, ax):
        """
        Apply EXMO styling to a matplotlib axis.

        Args:
            ax: Matplotlib axis object
        """
        # Grid styling
        ax.grid(axis=GRID_SPECS['axis'],
               which=GRID_SPECS['which'],
               linestyle=GRID_SPECS['linestyle'],
               linewidth=GRID_SPECS['linewidth'],
               alpha=GRID_SPECS['alpha'],
               color=GRID_SPECS['color'])

        # Spine styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.0)
        ax.spines['left'].set_linewidth(1.0)
        ax.spines['bottom'].set_color('#333333')
        ax.spines['left'].set_color('#333333')

        # Background
        ax.set_facecolor(EXMO_COLORS['background'])

        # Tick styling
        ax.tick_params(width=1.0, length=4, color='#333333')

    def get_limb_color(self, limb: str) -> str:
        """
        Get color for specific limb.

        Args:
            limb: Limb identifier (e.g., 'paw_RR', 'RH')

        Returns:
            Hex color code
        """
        return EXMO_COLORS.get(limb, '#666666')

    def plot_bar_with_error(self,
                           ax,
                           x_positions,
                           values,
                           errors,
                           colors,
                           labels=None):
        """
        Plot bar chart with error bars using EXMO styling.

        Args:
            ax: Matplotlib axis
            x_positions: X positions for bars
            values: Bar heights
            errors: Error bar magnitudes (MAD)
            colors: Bar colors
            labels: Optional labels for legend
        """
        # Plot bars
        bars = ax.bar(x_positions, values,
                     color=colors,
                     alpha=MARKER_SPECS['alpha'],
                     edgecolor=MARKER_SPECS['edge_color'],
                     linewidth=MARKER_SPECS['edge_width'] * 2)

        # Plot error bars
        ax.errorbar(x_positions, values, yerr=errors,
                   fmt='none',
                   ecolor='#000000',
                   capsize=ERROR_BAR_SPECS['capsize'],
                   linewidth=ERROR_BAR_SPECS['linewidth'],
                   alpha=ERROR_BAR_SPECS['alpha'])

        # Add median markers
        ax.scatter(x_positions, values,
                  s=MARKER_SPECS['size_median'],
                  marker=MARKER_SPECS['median_marker'],
                  c=colors,
                  edgecolors='#000000',
                  linewidths=MARKER_SPECS['edge_width'] * 3,
                  alpha=1.0,
                  zorder=10)

        # Optional annotations
        if self.annotate_median:
            for x, y in zip(x_positions, values):
                ax.annotate(f'{y:.1f}',
                          xy=(x, y),
                          xytext=(0, 8),
                          textcoords='offset points',
                          ha='center',
                          fontsize=FONT_SIZES['annotation'] * self.font_scale,
                          color='#333333')

        return bars

    def add_reference_band(self,
                          ax,
                          metric_name: str,
                          orientation='horizontal'):
        """
        Add reference band for normal/healthy range.

        Args:
            ax: Matplotlib axis
            metric_name: Metric name ('cadence', 'duty_cycle', etc.)
            orientation: 'horizontal' or 'vertical'
        """
        if metric_name not in REFERENCE_BANDS:
            return

        band = REFERENCE_BANDS[metric_name]

        if orientation == 'horizontal':
            ax.axhspan(band['min'], band['max'],
                      color=EXMO_COLORS['ref_band'],
                      alpha=0.2,
                      zorder=0,
                      label=band['label'])
        else:
            ax.axvspan(band['min'], band['max'],
                      color=EXMO_COLORS['ref_band'],
                      alpha=0.2,
                      zorder=0,
                      label=band['label'])

    def add_sample_badge(self, ax, n_samples: int):
        """
        Add sample count badge to top-right corner.

        Args:
            ax: Matplotlib axis
            n_samples: Number of samples
        """
        ax.text(0.98, 0.98, f'N = {n_samples}',
               transform=ax.transAxes,
               fontsize=FONT_SIZES['badge'] * self.font_scale,
               ha='right', va='top',
               bbox=dict(boxstyle='round,pad=0.4',
                        facecolor='white',
                        edgecolor='#cccccc',
                        linewidth=0.5,
                        alpha=0.9))

    def format_axis_range(self, ax, data, margin=0.10, zero_baseline=False):
        """
        Format axis range with appropriate margins.

        Args:
            ax: Matplotlib axis
            data: Data array for range calculation
            margin: Margin fraction (default 10%)
            zero_baseline: Force y-axis to start at 0
        """
        if len(data) == 0:
            return

        data_min = np.nanmin(data)
        data_max = np.nanmax(data)
        data_range = data_max - data_min

        if zero_baseline:
            y_min = 0
            y_max = data_max + (data_range * margin)
        else:
            y_min = data_min - (data_range * margin)
            y_max = data_max + (data_range * margin)

        ax.set_ylim([y_min, y_max])


# ═══════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def create_figure(n_subplots_x: int,
                 n_subplots_y: int = 1,
                 dpi: int = 600) -> tuple:
    """
    Create publication-grade figure with EXMO styling.

    Args:
        n_subplots_x: Number of horizontal subplots
        n_subplots_y: Number of vertical subplots
        dpi: Resolution

    Returns:
        (fig, axes) tuple
    """
    width = LAYOUT_SPECS['fig_width']
    height = (LAYOUT_SPECS['fig_height_single'] if n_subplots_y == 1
              else LAYOUT_SPECS['fig_height_double'])

    fig, axes = plt.subplots(n_subplots_y, n_subplots_x,
                            figsize=(width, height),
                            dpi=dpi,
                            constrained_layout=LAYOUT_SPECS['use_constrained_layout'])

    fig.patch.set_facecolor(EXMO_COLORS['background'])

    return fig, axes


def get_limb_display_name(limb: str) -> str:
    """
    Convert limb ID to display name.

    Args:
        limb: Limb ID (e.g., 'paw_RR', 'paw_RL')

    Returns:
        Display name (e.g., 'RH', 'LH')
    """
    mapping = {
        'paw_RR': 'RH',
        'paw_RL': 'LH',
        'paw_FR': 'RF',
        'paw_FL': 'LF',
        'elbow_R': 'Right',
        'elbow_L': 'Left'
    }
    return mapping.get(limb, limb)

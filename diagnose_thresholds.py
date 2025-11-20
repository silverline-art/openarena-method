#!/usr/bin/env python3
"""
EXMO Gait Analysis - Threshold Diagnostic Tool
Analyzes CoM speed distribution and recommends optimal thresholds
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from exmo_gait.core.data_loader import MultiViewDataLoader
from exmo_gait.core.preprocessor import DataPreprocessor
from exmo_gait.utils.signal_processing import compute_mad


def analyze_com_speed_distribution(top_path, side_path, bottom_path):
    """
    Analyze CoM speed distribution and recommend thresholds.

    Args:
        top_path: Path to top view CSV
        side_path: Path to side view CSV
        bottom_path: Path to bottom view CSV

    Returns:
        Dictionary with analysis results and recommendations
    """
    print("="*80)
    print("THRESHOLD DIAGNOSTIC ANALYSIS")
    print("="*80)

    # Load data
    print("\n[1/5] Loading multi-view data...")
    loader = MultiViewDataLoader(expected_fps=120.0)
    loader.load_all_views(top_path, side_path, bottom_path)

    # Extract keypoints
    print("[2/5] Extracting keypoint trajectories...")
    keypoints = {}
    view_priority = {'top': ['snout', 'neck', 'tail_base', 'rib_center'],
                    'bottom': ['paw_RR', 'paw_RL'],
                    'side': ['hip_R', 'hip_L']}

    for view, kp_list in view_priority.items():
        for kp in kp_list:
            if kp in loader.get_available_keypoints(view):
                keypoints[kp] = loader.get_keypoint(view, kp)

    # Preprocess
    print("[3/5] Preprocessing data...")
    preprocessor = DataPreprocessor()

    snout = keypoints.get('snout')
    tail_base = keypoints.get('tail_base')
    if snout is not None and tail_base is not None:
        scale_factor = preprocessor.compute_scale_factor(snout, tail_base)
    else:
        scale_factor = 0.1

    keypoints_preprocessed = preprocessor.batch_preprocess_keypoints(keypoints)
    for kp_name in keypoints_preprocessed:
        keypoints_preprocessed[kp_name] = preprocessor.convert_to_cm(
            keypoints_preprocessed[kp_name]
        )

    # Compute CoM
    print("[4/5] Computing center of mass...")
    if 'hip_R' in keypoints_preprocessed and 'hip_L' in keypoints_preprocessed:
        hip_center = preprocessor.compute_hip_center(
            keypoints_preprocessed['hip_L'],
            keypoints_preprocessed['hip_R']
        )
    else:
        hip_center = keypoints_preprocessed.get('rib_center',
                                               np.zeros((loader.n_frames, 2)))

    rib_center = keypoints_preprocessed.get('rib_center', hip_center)
    com_trajectory = preprocessor.compute_com_trajectory(hip_center, rib_center)

    # Compute CoM speed
    print("[5/5] Analyzing CoM speed distribution...")
    diffs = np.diff(com_trajectory, axis=0, prepend=com_trajectory[:1])
    distances = np.sqrt(np.sum(diffs ** 2, axis=1))
    com_speeds = distances * 120.0  # Convert to cm/s

    valid_speeds = com_speeds[~np.isnan(com_speeds)]

    # Compute statistics
    median_speed = np.median(valid_speeds)
    mean_speed = np.mean(valid_speeds)
    std_speed = np.std(valid_speeds)
    mad_speed = compute_mad(valid_speeds)

    percentiles = np.percentile(valid_speeds, [25, 50, 75, 90, 95, 99])

    # Print analysis
    print("\n" + "="*80)
    print("SPEED DISTRIBUTION ANALYSIS")
    print("="*80)
    print(f"\nTotal frames: {len(com_speeds)}")
    print(f"Valid frames: {len(valid_speeds)} ({len(valid_speeds)/len(com_speeds)*100:.1f}%)")
    print(f"\nSpeed Statistics (cm/s):")
    print(f"  Median: {median_speed:.3f}")
    print(f"  Mean:   {mean_speed:.3f}")
    print(f"  Std:    {std_speed:.3f}")
    print(f"  MAD:    {mad_speed:.3f}")
    print(f"\nPercentiles (cm/s):")
    print(f"  P25:  {percentiles[0]:.3f}")
    print(f"  P50:  {percentiles[1]:.3f}")
    print(f"  P75:  {percentiles[2]:.3f}")
    print(f"  P90:  {percentiles[3]:.3f}")
    print(f"  P95:  {percentiles[4]:.3f}")
    print(f"  P99:  {percentiles[5]:.3f}")

    # Current thresholds
    print("\n" + "="*80)
    print("THRESHOLD ANALYSIS")
    print("="*80)

    current_stationary = mad_speed * 1.5
    current_walking = mad_speed * 2.0

    print(f"\nCurrent Config (CONSERVATIVE):")
    print(f"  Stationary threshold: {current_stationary:.3f} cm/s (MAD × 1.5)")
    print(f"  Walking threshold:    {current_walking:.3f} cm/s (MAD × 2.0)")

    frames_below_stationary = np.sum(valid_speeds < current_stationary)
    frames_above_walking = np.sum(valid_speeds > current_walking)

    print(f"\n  Frames classified as stationary: {frames_below_stationary} ({frames_below_stationary/len(valid_speeds)*100:.1f}%)")
    print(f"  Frames classified as walking:    {frames_above_walking} ({frames_above_walking/len(valid_speeds)*100:.1f}%)")

    # Adaptive thresholds
    adaptive_stationary = mad_speed * 1.0
    adaptive_walking = mad_speed * 1.2

    print(f"\nAdaptive Config (RELAXED):")
    print(f"  Stationary threshold: {adaptive_stationary:.3f} cm/s (MAD × 1.0)")
    print(f"  Walking threshold:    {adaptive_walking:.3f} cm/s (MAD × 1.2)")

    frames_below_adaptive_stat = np.sum(valid_speeds < adaptive_stationary)
    frames_above_adaptive_walk = np.sum(valid_speeds > adaptive_walking)

    print(f"\n  Frames classified as stationary: {frames_below_adaptive_stat} ({frames_below_adaptive_stat/len(valid_speeds)*100:.1f}%)")
    print(f"  Frames classified as walking:    {frames_above_adaptive_walk} ({frames_above_adaptive_walk/len(valid_speeds)*100:.1f}%)")

    # Percentile-based recommendation
    p75_threshold = percentiles[2]
    p90_threshold = percentiles[3]

    print(f"\nPercentile-based (ADAPTIVE):")
    print(f"  Stationary threshold: {p75_threshold:.3f} cm/s (P75)")
    print(f"  Walking threshold:    {p90_threshold:.3f} cm/s (P90)")

    frames_below_p75 = np.sum(valid_speeds < p75_threshold)
    frames_above_p90 = np.sum(valid_speeds > p90_threshold)

    print(f"\n  Frames classified as stationary: {frames_below_p75} ({frames_below_p75/len(valid_speeds)*100:.1f}%)")
    print(f"  Frames classified as walking:    {frames_above_p90} ({frames_above_p90/len(valid_speeds)*100:.1f}%)")

    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    if frames_above_walking < len(valid_speeds) * 0.05:
        print("\n⚠️  WARNING: Very low activity detected!")
        print(f"   Only {frames_above_walking/len(valid_speeds)*100:.1f}% of frames classified as walking")
        print("\n✅  RECOMMENDED ACTION:")
        print("   1. Use config_adaptive.yaml (already created)")
        print("   2. Or add to config.yaml:")
        print(f"      walking_mad_threshold: 1.0  # (threshold: {adaptive_walking:.3f} cm/s)")
        print(f"      min_walking_duration: 0.15")
    else:
        print("\n✅  Activity level looks good!")
        print("   Current thresholds should work fine.")

    # Generate diagnostic plot
    print("\n[6/6] Generating diagnostic plot...")

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Histogram
    ax = axes[0]
    ax.hist(valid_speeds, bins=100, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(current_stationary, color='red', linestyle='--', linewidth=2,
              label=f'Current Stationary ({current_stationary:.2f} cm/s)')
    ax.axvline(current_walking, color='green', linestyle='--', linewidth=2,
              label=f'Current Walking ({current_walking:.2f} cm/s)')
    ax.axvline(adaptive_stationary, color='orange', linestyle=':', linewidth=2,
              label=f'Adaptive Stationary ({adaptive_stationary:.2f} cm/s)')
    ax.axvline(adaptive_walking, color='purple', linestyle=':', linewidth=2,
              label=f'Adaptive Walking ({adaptive_walking:.2f} cm/s)')
    ax.set_xlabel('CoM Speed (cm/s)', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('CoM Speed Distribution with Threshold Comparison', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Time series
    ax = axes[1]
    time = np.arange(len(com_speeds)) / 120.0  # Convert to seconds
    ax.plot(time, com_speeds, color='blue', alpha=0.5, linewidth=0.5)
    ax.axhline(current_walking, color='green', linestyle='--', linewidth=2,
              label=f'Current Walking ({current_walking:.2f} cm/s)')
    ax.axhline(adaptive_walking, color='purple', linestyle=':', linewidth=2,
              label=f'Adaptive Walking ({adaptive_walking:.2f} cm/s)')
    ax.set_xlabel('Time (seconds)', fontweight='bold')
    ax.set_ylabel('CoM Speed (cm/s)', fontweight='bold')
    ax.set_title('CoM Speed Over Time', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    output_path = Path('diagnostic_thresholds.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅  Diagnostic plot saved: {output_path}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80 + "\n")

    return {
        'median_speed': median_speed,
        'mad_speed': mad_speed,
        'current_walking_threshold': current_walking,
        'adaptive_walking_threshold': adaptive_walking,
        'percentile_walking_threshold': p90_threshold,
        'walking_frames_current': frames_above_walking,
        'walking_frames_adaptive': frames_above_adaptive_walk,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Diagnostic tool for analyzing CoM speed and recommending thresholds'
    )

    parser.add_argument('--top', type=Path, required=True,
                       help='Path to top view CSV')
    parser.add_argument('--side', type=Path, required=True,
                       help='Path to side view CSV')
    parser.add_argument('--bottom', type=Path, required=True,
                       help='Path to bottom view CSV')

    args = parser.parse_args()

    # Run analysis
    results = analyze_com_speed_distribution(args.top, args.side, args.bottom)

    sys.exit(0)


if __name__ == '__main__':
    main()

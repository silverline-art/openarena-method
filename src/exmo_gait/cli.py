"""Command-line interface for EXMO gait analyzer"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict

from .pipeline.executor import PipelineExecutor


def run_pipeline(top_path: Path,
                side_path: Path,
                bottom_path: Path,
                output_dir: Path,
                verbose: bool = False,
                config: Dict = None) -> Dict:
    """
    Run complete gait analysis pipeline.

    This function now delegates to PipelineExecutor, which uses a clean
    stage-based architecture replacing the original 232-line monolithic
    implementation.

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
    executor = PipelineExecutor()
    return executor.execute(
        top_path=top_path,
        side_path=side_path,
        bottom_path=bottom_path,
        output_dir=output_dir,
        verbose=verbose,
        config=config
    )


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

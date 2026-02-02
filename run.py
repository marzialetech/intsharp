#!/usr/bin/env python3
"""
CLI entry point for running simulations.

Usage:
    python run.py config.yaml

Creates a new timestamped run folder, moves the YAML into it, and writes
all monitor output (GIF, PNG, HDF5, etc.) into that same folder.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run a 1D interface-sharpening simulation from YAML config.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run.py config.yaml
    python run.py unit_tests/tanh_one_rev_no_sharpening.yaml

Creates a new run folder in the same directory as the YAML, named
<yaml_stem>_YYYY-MM-DD_HH:MM:SS (e.g. unit_tests/tanh_one_rev_no_sharpening_2025-02-01_19:15:30),
moves the YAML into it, and writes all outputs there.
        """,
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()
    config_path = args.config.resolve()

    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}", file=sys.stderr)
        return 1

    try:
        from intsharp.config import load_config
        from intsharp.runner import run_simulation

        print(f"Loading configuration: {config_path}")
        config = load_config(config_path)

        # Create new run folder: yaml_name + YYYY-MM-DD_HH:MM:SS
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        run_dir = config_path.parent / f"{config_path.stem}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Move the YAML into the run folder (single copy)
        dest_yaml = run_dir / config_path.name
        shutil.move(str(config_path), str(dest_yaml))
        print(f"Run folder: {run_dir} (moved config to {dest_yaml.name})")

        # Point config output to this folder and run
        config.output.directory = str(run_dir)
        fields = run_simulation(config)

        print(f"Simulation complete. Outputs in {run_dir}. Final fields: {list(fields.keys())}")
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

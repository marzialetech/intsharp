#!/usr/bin/env python3
"""
CLI entry point for running simulations.

Usage:
    python run.py config.yaml

Creates a new timestamped run folder, copies the YAML into it, and writes
all monitor output (GIF, PNG, HDF5, etc.) into that same folder.
The original YAML remains in place for easy re-running.
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
copies the YAML into it, and writes all outputs there. The original YAML
remains in place for easy re-running.
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

        # Copy the YAML into the run folder (keep original for re-running)
        dest_yaml = run_dir / config_path.name
        shutil.copy2(str(config_path), str(dest_yaml))
        print(f"Run folder: {run_dir} (copied config to {dest_yaml.name})")

        # Copy any initial-condition images into the run folder (for archival)
        config_dir = config_path.parent
        seen_images: set[Path] = set()
        for field_cfg in config.fields:
            img_path = getattr(field_cfg, "initial_condition_image", None)
            if not img_path:
                continue
            src = (config_dir / img_path).resolve()
            if not src.exists():
                continue
            if src not in seen_images:
                seen_images.add(src)
                dest = run_dir / Path(img_path).name
                shutil.copy2(str(src), str(dest))
                print(f"Copied initial-condition image to {dest.name}")

        # Point config output to this folder and run
        config.output.directory = str(run_dir)
        fields = run_simulation(config, config_dir=config_path.parent)

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

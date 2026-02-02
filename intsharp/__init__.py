"""
IntSharp: Interface Sharpening Simulation Framework

A modular framework for 1D advection simulations with interface sharpening.

Usage:
    from intsharp.runner import run_from_file
    fields = run_from_file("config.yaml")

Or via CLI:
    python run.py config.yaml
"""

from .config import SimulationConfig, load_config
from .domain import Domain1D, create_domain
from .fields import Field, create_fields
from .runner import run_simulation, run_from_file

__version__ = "0.1.0"

__all__ = [
    "SimulationConfig",
    "load_config",
    "Domain1D",
    "create_domain",
    "Field",
    "create_fields",
    "run_simulation",
    "run_from_file",
]

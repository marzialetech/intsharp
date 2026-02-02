"""
Main simulation runner (1D and 2D).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .boundary import apply_bc, apply_bc_2d
from .config import SimulationConfig, load_config
from .domain import Domain1D, Domain2D, create_domain
from .fields import Field, create_fields
from .registry import get_sharpening, get_solver

# Import submodules to trigger registration of solvers/monitors/sharpening/timesteppers
from . import solvers  # noqa: F401
from . import monitors  # noqa: F401
from . import sharpening as sharpening_module  # noqa: F401
from . import timesteppers as timesteppers_module  # noqa: F401

if TYPE_CHECKING:
    from .monitors.base import Monitor


def create_monitors(
    config: SimulationConfig,
    output_dir: Path,
) -> list["Monitor"]:
    """
    Create monitor instances from configuration.

    Parameters
    ----------
    config : SimulationConfig
        Simulation configuration.
    output_dir : Path
        Base output directory.

    Returns
    -------
    list[Monitor]
        List of monitor instances.
    """
    from .registry import get_monitor

    monitor_list = []

    for mon_cfg in config.output.monitors:
        monitor_cls = get_monitor(mon_cfg.type)

        # Build kwargs for monitor
        mon_output_dir = output_dir
        if mon_cfg.type == "txt":
            mon_output_dir = output_dir / "txt"
        kwargs = {
            "output_dir": mon_output_dir,
            "every_n_steps": mon_cfg.every_n_steps,
            "at_times": mon_cfg.at_times,
        }

        # Add type-specific kwargs
        if mon_cfg.type == "console":
            kwargs["total_steps"] = config.time.n_steps
        elif mon_cfg.type in ("png", "pdf", "svg"):
            kwargs["field"] = mon_cfg.field
            kwargs["show_colorbar"] = mon_cfg.show_colorbar
            kwargs["show_annotations"] = mon_cfg.show_annotations
        elif mon_cfg.type in ("gif", "mp4"):
            # Unified gif/mp4: single-field or compare mode
            kwargs["field"] = mon_cfg.field
            kwargs["style"] = mon_cfg.style
            kwargs["contour_levels"] = mon_cfg.contour_levels
            kwargs["colormap"] = mon_cfg.colormap
            kwargs["contour_overlay_color"] = mon_cfg.contour_overlay_color
            kwargs["contour_color"] = mon_cfg.contour_color
            kwargs["background_color"] = mon_cfg.background_color
            kwargs["show_colorbar"] = mon_cfg.show_colorbar
            kwargs["show_annotations"] = mon_cfg.show_annotations
            kwargs["output_format"] = mon_cfg.type  # "gif" or "mp4"
            if mon_cfg.compare_fields:
                kwargs["compare_fields"] = [
                    {
                        "field": cf.field,
                        "contour_levels": cf.contour_levels,
                        "color": cf.color,
                        "linestyle": cf.linestyle,
                    }
                    for cf in mon_cfg.compare_fields
                ]
        elif mon_cfg.type in ("txt", "curve"):
            kwargs["field"] = mon_cfg.field
            kwargs["fields"] = mon_cfg.fields
        elif mon_cfg.type == "hdf5":
            kwargs["fields"] = mon_cfg.fields

        monitor_list.append(monitor_cls(**kwargs))

    return monitor_list


def run_simulation(
    config: SimulationConfig,
    config_dir: Path | None = None,
) -> dict[str, Field]:
    """
    Run the simulation.

    Parameters
    ----------
    config : SimulationConfig
        Complete simulation configuration.
    config_dir : Path or None
        Directory of the config file for resolving relative image paths.

    Returns
    -------
    dict[str, Field]
        Final field states.
    """
    # Create domain
    domain = create_domain(config.domain)
    ndim = domain.ndim

    # Create fields
    fields = create_fields(config.fields, domain, config_dir)

    # Get solver based on dimension
    if ndim == 1:
        solver_fn = get_solver(config.solver.type)
    else:
        # For 2D, use the 2D version of the solver
        solver_type_2d = config.solver.type + "_2d"
        try:
            solver_fn = get_solver(solver_type_2d)
        except KeyError:
            # Fall back to the specified solver if no 2D version exists
            solver_fn = get_solver(config.solver.type)

    # Get sharpening method if needed
    # Sharpening is needed if: global enabled OR any field has per-field sharpening=True
    sharpening_fn = None
    needs_sharpening = False
    if config.sharpening:
        if config.sharpening.enabled:
            needs_sharpening = True
        else:
            # Check if any field has per-field sharpening enabled
            for field_cfg in config.fields:
                if field_cfg.sharpening is True:
                    needs_sharpening = True
                    break

    if needs_sharpening and config.sharpening:
        if ndim == 1:
            sharpening_fn = get_sharpening(config.sharpening.method)
        else:
            # For 2D, use the 2D version of sharpening
            sharpening_method_2d = config.sharpening.method + "_2d"
            try:
                sharpening_fn = get_sharpening(sharpening_method_2d)
            except KeyError:
                sharpening_fn = get_sharpening(config.sharpening.method)

    # Create output directory
    output_dir = Path(config.output.directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create monitors
    monitor_list = create_monitors(config, output_dir)

    # Extract parameters
    dt = config.time.dt
    n_steps = config.time.n_steps
    dx = domain.dx

    if ndim == 1:
        velocity = config.velocity[0]
        dy = 0.0
    else:
        velocity = (config.velocity[0], config.velocity[1])
        dy = domain.dy  # type: ignore

    # Initialize monitors
    for monitor in monitor_list:
        monitor.on_start(fields, domain)

    # Output initial conditions (step 0)
    t = 0.0
    for monitor in monitor_list:
        monitor.on_step(0, t, fields, domain)

    # Time loop
    for step in range(1, n_steps + 1):
        # Advect each field
        for name, field in fields.items():
            if ndim == 1:
                # 1D advection
                new_values = solver_fn(
                    field.values,
                    velocity,
                    dx,
                    dt,
                    field.bc,
                )
                # Apply boundary conditions
                new_values = apply_bc(new_values, field.bc, dx)
            else:
                # 2D advection
                new_values = solver_fn(
                    field.values,
                    velocity,
                    dx,
                    dy,
                    dt,
                    field.bc,
                )
                # Apply boundary conditions
                new_values = apply_bc_2d(new_values, field.bc, dx, dy)

            field.values = new_values

        # Sharpening post-step
        # Per-field sharpening: field.sharpening_enabled can override global setting
        if config.sharpening is not None:
            for name, field in fields.items():
                # Determine if sharpening should be applied to this field
                if field.sharpening_enabled is True:
                    # Per-field override: force sharpening on
                    apply_sharpening = True
                elif field.sharpening_enabled is False:
                    # Per-field override: force sharpening off
                    apply_sharpening = False
                else:
                    # Use global setting
                    apply_sharpening = config.sharpening.enabled and sharpening_fn is not None

                if apply_sharpening and sharpening_fn is not None:
                    if ndim == 1:
                        field.values = sharpening_fn(
                            field.values,
                            dx,
                            dt,
                            config.sharpening.eps_target,
                            config.sharpening.strength,
                            field.bc,
                        )
                    else:
                        field.values = sharpening_fn(
                            field.values,
                            dx,
                            dy,
                            dt,
                            config.sharpening.eps_target,
                            config.sharpening.strength,
                            field.bc,
                        )

        # Update time
        t += dt

        # Call monitors
        for monitor in monitor_list:
            monitor.on_step(step, t, fields, domain)

    # Finalize monitors
    for monitor in monitor_list:
        monitor.on_end(fields, domain)

    return fields


def run_from_file(config_path: str | Path) -> dict[str, Field]:
    """
    Load configuration from file and run simulation.

    Parameters
    ----------
    config_path : str or Path
        Path to YAML configuration file.

    Returns
    -------
    dict[str, Field]
        Final field states.
    """
    config = load_config(config_path)
    return run_simulation(config)

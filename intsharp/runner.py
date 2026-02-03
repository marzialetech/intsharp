"""
Main simulation runner (1D and 2D).

Supports two physics modes:
- advection_only: Scalar advection with optional sharpening
- euler: Compressible Euler equations with AUSM+UP flux
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .boundary import apply_bc, apply_bc_2d
from .config import (
    SimulationConfig,
    load_config,
    ConstantVelocityConfig,
    ExpressionVelocityConfig,
)
from .domain import Domain1D, Domain2D, create_domain
from .fields import (
    Field,
    create_fields,
    evaluate_velocity_expression_1d,
    evaluate_velocity_expression_2d,
)
from .registry import get_sharpening, get_solver
from .surface_tension import compute_surface_tension_diagnostics_2d
from .solvers.euler_1d import (
    EulerState1D,
    euler_step_1d,
    create_initial_state_riemann_1d,
    check_cfl_euler_1d,
    TwoPhaseEulerState1D,
    euler_step_two_phase_1d,
    create_initial_state_riemann_two_phase_1d,
    check_cfl_euler_two_phase_1d,
)
from .solvers.euler_5eq_1d import (
    FiveEqState1D,
    euler_step_5eq_1d,
    create_initial_state_riemann_5eq_1d,
    check_cfl_5eq_1d,
)
from .eos import total_energy_from_primitives, mixture_pressure_from_conservatives

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
            kwargs["quiver_overlay_x"] = mon_cfg.quiver_overlay_x
            kwargs["quiver_overlay_y"] = mon_cfg.quiver_overlay_y
            kwargs["quiver_skip"] = mon_cfg.quiver_skip
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


def run_advection_simulation(
    config: SimulationConfig,
    config_dir: Path | None = None,
) -> dict[str, Field]:
    """
    Run advection-only simulation (original mode).

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
        dy = 0.0
    else:
        dy = domain.dy  # type: ignore

    # Velocity configuration: constant or expression
    vel_config = config.velocity
    is_constant_velocity = isinstance(vel_config, ConstantVelocityConfig)

    # For constant velocity, extract scalar values once
    if is_constant_velocity:
        if ndim == 1:
            const_velocity = vel_config.value[0]
        else:
            const_velocity = (vel_config.value[0], vel_config.value[1])

    # Track primary field names (only these are advected)
    primary_field_names = set(fields.keys())

    # Initialize monitors
    for monitor in monitor_list:
        monitor.on_start(fields, domain)

    # Surface tension diagnostics at t=0 (so monitors can output curvature etc. from start)
    t = 0.0
    if ndim == 2 and config.surface_tension and config.surface_tension.enabled:
        source_name = config.surface_tension.source_field
        if source_name in fields:
            source = fields[source_name]
            sigma = config.surface_tension.sigma
            smoothing_sigma = config.surface_tension.smoothing_sigma
            band_min = config.surface_tension.interface_band_alpha_min
            band_max = config.surface_tension.interface_band_alpha_max
            st_diags = compute_surface_tension_diagnostics_2d(
                source.values, sigma, dx, dy, source.bc,
                smoothing_sigma=smoothing_sigma,
                interface_band_alpha_min=band_min,
                interface_band_alpha_max=band_max,
            )
            for diag_name, diag_values in st_diags.items():
                fields[diag_name] = Field(diag_name, diag_values, source.bc, None)

    # Output initial conditions (step 0)
    for monitor in monitor_list:
        monitor.on_step(0, t, fields, domain)

    # Time loop
    for step in range(1, n_steps + 1):
        # Compute velocity for this step
        if is_constant_velocity:
            velocity = const_velocity
        else:
            # Expression velocity: evaluate at current time t
            if ndim == 1:
                velocity = evaluate_velocity_expression_1d(
                    vel_config.u,
                    domain.x,
                    t,
                )
            else:
                velocity = evaluate_velocity_expression_2d(
                    vel_config.u,
                    vel_config.v,  # type: ignore (validated to exist for 2D)
                    domain.X,
                    domain.Y,
                    t,
                )

        # Advect each primary field (skip derived diagnostic fields)
        for name in primary_field_names:
            field = fields[name]
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

        # Sharpening post-step (only primary fields)
        # Per-field sharpening: field.sharpening_enabled can override global setting
        if config.sharpening is not None:
            for name in primary_field_names:
                field = fields[name]
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

        # Surface tension diagnostics (2D only)
        if ndim == 2 and config.surface_tension and config.surface_tension.enabled:
            source_name = config.surface_tension.source_field
            if source_name in fields:
                source = fields[source_name]
                sigma = config.surface_tension.sigma
                
                # Compute all surface tension diagnostics
                smoothing_sigma = config.surface_tension.smoothing_sigma
                band_min = config.surface_tension.interface_band_alpha_min
                band_max = config.surface_tension.interface_band_alpha_max
                st_diags = compute_surface_tension_diagnostics_2d(
                    source.values, sigma, dx, dy, source.bc,
                    smoothing_sigma=smoothing_sigma,
                    interface_band_alpha_min=band_min,
                    interface_band_alpha_max=band_max,
                )
                
                # Add as derived fields (use source BC, no sharpening)
                for diag_name, diag_values in st_diags.items():
                    fields[diag_name] = Field(diag_name, diag_values, source.bc, None)

        # Update time
        t += dt

        # Call monitors
        for monitor in monitor_list:
            monitor.on_step(step, t, fields, domain)

    # Finalize monitors
    for monitor in monitor_list:
        monitor.on_end(fields, domain)

    return fields


def run_euler_simulation(
    config: SimulationConfig,
    config_dir: Path | None = None,
) -> dict[str, Field]:
    """
    Run compressible Euler simulation (single-phase or two-phase).

    Parameters
    ----------
    config : SimulationConfig
        Complete simulation configuration.
    config_dir : Path or None
        Directory of the config file (unused for Euler mode).

    Returns
    -------
    dict[str, Field]
        Final field states (rho, u, p, E, and alpha for two-phase).
    """
    from .boundary import BoundaryCondition

    # Create domain
    domain = create_domain(config.domain)
    
    # Currently only 1D
    if domain.ndim != 1:
        raise ValueError("Euler mode currently only supports 1D")

    # Extract physics config
    physics = config.physics
    euler_ic = physics.euler_initial_conditions
    euler_bc = physics.euler_bc

    # Determine if single-phase or two-phase
    is_two_phase = physics.is_two_phase

    if is_two_phase:
        # Two-phase mode
        gamma1 = physics.phase1.gamma
        p_inf1 = physics.phase1.p_infinity
        gamma2 = physics.phase2.gamma
        p_inf2 = physics.phase2.p_infinity
        two_phase_model = physics.two_phase_model
        use_muscl = physics.use_muscl

        # Get alpha from IC (default to 1.0 on left, 0.0 on right if not specified)
        if euler_ic.type != "riemann":
            raise ValueError(f"Two-phase Euler only supports 'riemann' IC, got: {euler_ic.type}")
        
        alpha_L = euler_ic.left.alpha if euler_ic.left.alpha is not None else 1.0
        alpha_R = euler_ic.right.alpha if euler_ic.right.alpha is not None else 0.0

        dummy_bc = BoundaryCondition("neumann", gradient_left=0.0, gradient_right=0.0)

        if two_phase_model == "5eq":
            # 5-equation model
            # For 5-eq, we need phase densities, not mixture density
            # Left: if alpha=1, pure phase 1, so rho1=rho, rho2 is ambient
            # Right: if alpha=0, pure phase 2, so rho2=rho, rho1 is ambient
            rho1_L = euler_ic.left.rho if alpha_L > 0.5 else euler_ic.left.rho
            rho2_L = euler_ic.left.rho if alpha_L < 0.5 else euler_ic.left.rho
            rho1_R = euler_ic.right.rho if alpha_R > 0.5 else euler_ic.right.rho
            rho2_R = euler_ic.right.rho if alpha_R < 0.5 else euler_ic.right.rho
            
            # Actually, for water-air: left is water (alpha=1), right is air (alpha=0)
            # rho_L = 1000 is water density, rho_R = 10 is air density
            # So: rho1_L = 1000 (water), rho2_L = any (doesn't matter, alpha2=0)
            #     rho1_R = any (doesn't matter, alpha1=0), rho2_R = 10 (air)
            # For generality, use rho from IC as the dominant phase density
            rho1_L = euler_ic.left.rho   # Phase 1 density on left
            rho2_L = euler_ic.right.rho  # Use air density as phase 2 on left (trace)
            rho1_R = euler_ic.left.rho   # Use water density as phase 1 on right (trace)
            rho2_R = euler_ic.right.rho  # Phase 2 density on right

            state = create_initial_state_riemann_5eq_1d(
                x=domain.x,
                x_discontinuity=euler_ic.x_discontinuity,
                rho1_L=rho1_L,
                rho2_L=rho2_L,
                u_L=euler_ic.left.u,
                p_L=euler_ic.left.p,
                alpha1_L=alpha_L,
                rho1_R=rho1_R,
                rho2_R=rho2_R,
                u_R=euler_ic.right.u,
                p_R=euler_ic.right.p,
                alpha1_R=alpha_R,
                gamma1=gamma1,
                gamma2=gamma2,
                p_inf1=p_inf1,
                p_inf2=p_inf2,
            )

            def state_to_fields(s: FiveEqState1D) -> dict[str, Field]:
                gamma_eff = s.alpha1 * gamma1 + s.alpha2 * gamma2
                p_inf_eff = s.alpha1 * p_inf1 + s.alpha2 * p_inf2
                e = (s.p + gamma_eff * p_inf_eff) / ((gamma_eff - 1.0) * s.rho + 1e-30)
                return {
                    "rho": Field("rho", s.rho.copy(), dummy_bc, None),
                    "u": Field("u", s.u.copy(), dummy_bc, None),
                    "p": Field("p", s.p.copy(), dummy_bc, None),
                    "E": Field("E", s.E.copy(), dummy_bc, None),
                    "e_int": Field("e_int", e.copy(), dummy_bc, None),
                    "alpha": Field("alpha", s.alpha1.copy(), dummy_bc, None),
                    "alpha1_rho1": Field("alpha1_rho1", s.alpha1_rho1.copy(), dummy_bc, None),
                    "alpha2_rho2": Field("alpha2_rho2", s.alpha2_rho2.copy(), dummy_bc, None),
                    "rho_u": Field("rho_u", s.rho_u.copy(), dummy_bc, None),
                }

            def step_fn(s, dx, dt, bc):
                return euler_step_5eq_1d(s, dx, dt, bc, use_muscl=use_muscl)

            def cfl_fn(s, dx, dt):
                return check_cfl_5eq_1d(s, dx, dt)

        else:
            # Mixture model (original two-phase)
            state = create_initial_state_riemann_two_phase_1d(
                x=domain.x,
                x_discontinuity=euler_ic.x_discontinuity,
                rho_L=euler_ic.left.rho,
                u_L=euler_ic.left.u,
                p_L=euler_ic.left.p,
                alpha_L=alpha_L,
                rho_R=euler_ic.right.rho,
                u_R=euler_ic.right.u,
                p_R=euler_ic.right.p,
                alpha_R=alpha_R,
                gamma1=gamma1,
                gamma2=gamma2,
                p_inf1=p_inf1,
                p_inf2=p_inf2,
            )

            def state_to_fields(s: TwoPhaseEulerState1D) -> dict[str, Field]:
                gamma_eff = s.alpha * gamma1 + (1.0 - s.alpha) * gamma2
                p_inf_eff = s.alpha * p_inf1 + (1.0 - s.alpha) * p_inf2
                e = (s.p + gamma_eff * p_inf_eff) / ((gamma_eff - 1.0) * s.rho + 1e-30)
                return {
                    "rho": Field("rho", s.rho.copy(), dummy_bc, None),
                    "u": Field("u", s.u.copy(), dummy_bc, None),
                    "p": Field("p", s.p.copy(), dummy_bc, None),
                    "E": Field("E", s.E.copy(), dummy_bc, None),
                    "e_int": Field("e_int", e.copy(), dummy_bc, None),
                    "alpha": Field("alpha", s.alpha.copy(), dummy_bc, None),
                    "rho_u": Field("rho_u", s.rho_u.copy(), dummy_bc, None),
                }

            def step_fn(s, dx, dt, bc):
                return euler_step_two_phase_1d(s, dx, dt, bc, use_muscl=use_muscl)

            def cfl_fn(s, dx, dt):
                return check_cfl_euler_two_phase_1d(s, dx, dt)

    else:
        # Single-phase mode
        material = physics.material
        gamma = material.gamma
        p_inf = material.p_infinity

        # Create initial state
        if euler_ic.type == "riemann":
            state = create_initial_state_riemann_1d(
                x=domain.x,
                x_discontinuity=euler_ic.x_discontinuity,
                rho_L=euler_ic.left.rho,
                u_L=euler_ic.left.u,
                p_L=euler_ic.left.p,
                rho_R=euler_ic.right.rho,
                u_R=euler_ic.right.u,
                p_R=euler_ic.right.p,
                gamma=gamma,
                p_inf=p_inf,
            )
        elif euler_ic.type == "uniform":
            n = len(domain.x)
            rho = np.full(n, euler_ic.rho)
            u = np.full(n, euler_ic.u)
            p = np.full(n, euler_ic.p)
            rho_u = rho * u
            E = total_energy_from_primitives(rho, u, p, gamma, p_inf)
            state = EulerState1D(rho=rho, rho_u=rho_u, E=E, gamma=gamma, p_inf=p_inf)
        else:
            raise ValueError(f"Unknown Euler IC type: {euler_ic.type}")

        # State to fields converter for single-phase
        dummy_bc = BoundaryCondition("neumann", gradient_left=0.0, gradient_right=0.0)

        def state_to_fields(s: EulerState1D) -> dict[str, Field]:
            e = (s.p + gamma * p_inf) / ((gamma - 1.0) * s.rho)
            return {
                "rho": Field("rho", s.rho.copy(), dummy_bc, None),
                "u": Field("u", s.u.copy(), dummy_bc, None),
                "p": Field("p", s.p.copy(), dummy_bc, None),
                "E": Field("E", s.E.copy(), dummy_bc, None),
                "e_int": Field("e_int", e.copy(), dummy_bc, None),
                "rho_u": Field("rho_u", s.rho_u.copy(), dummy_bc, None),
            }

        # Time stepping function
        use_muscl = physics.use_muscl
        def step_fn(s, dx, dt, bc):
            return euler_step_1d(s, dx, dt, bc, use_muscl=use_muscl)

        # CFL check function
        def cfl_fn(s, dx, dt):
            return check_cfl_euler_1d(s, dx, dt)

    # Create output directory
    output_dir = Path(config.output.directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create monitors
    monitor_list = create_monitors(config, output_dir)

    # Extract time parameters
    dt = config.time.dt
    n_steps = config.time.n_steps
    dx = domain.dx

    # Check CFL at start
    cfl_fn(state, dx, dt)

    # Initialize monitors
    fields = state_to_fields(state)
    for monitor in monitor_list:
        monitor.on_start(fields, domain)

    # Output initial conditions (step 0)
    t = 0.0
    for monitor in monitor_list:
        monitor.on_step(0, t, fields, domain)

    # Time loop
    for step in range(1, n_steps + 1):
        # Euler step
        state = step_fn(state, dx, dt, euler_bc)
        t += dt

        # Convert to fields for monitors
        fields = state_to_fields(state)

        # Call monitors
        for monitor in monitor_list:
            monitor.on_step(step, t, fields, domain)

    # Finalize monitors
    for monitor in monitor_list:
        monitor.on_end(fields, domain)

    return fields


def run_simulation(
    config: SimulationConfig,
    config_dir: Path | None = None,
) -> dict[str, Field]:
    """
    Run the simulation.

    Dispatches to the appropriate runner based on physics mode.

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
    mode = config.physics_mode

    if mode == "euler":
        return run_euler_simulation(config, config_dir)
    else:
        return run_advection_simulation(config, config_dir)


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

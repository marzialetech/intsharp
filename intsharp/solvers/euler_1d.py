"""
1D compressible Euler solver with modular intercell fluxes.

Solves the 1D Euler equations:
    ∂ρ/∂t + ∂(ρu)/∂x = 0
    ∂(ρu)/∂t + ∂(ρu² + p)/∂x = 0
    ∂E/∂t + ∂((E + p)u)/∂x = 0

With stiffened gas EOS:
    p = (γ - 1) ρ e - γ p_∞
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ..eos import (
    sound_speed,
    pressure_from_total_energy,
    total_energy_from_primitives,
    primitives_from_conservatives,
    mixture_density,
    mixture_total_energy,
    mixture_pressure_from_conservatives,
    mixture_sound_speed_wood,
)
from ..flux_ausm import ausm_plus_up_flux_1d, compute_interface_states_1d
from ..flux_hllc import hllc_flux_1d


@dataclass
class EulerState1D:
    """Container for 1D Euler conserved and primitive variables."""

    # Conserved variables
    rho: NDArray[np.float64]      # Density
    rho_u: NDArray[np.float64]    # Momentum (ρu)
    E: NDArray[np.float64]        # Total energy

    # EOS parameters
    gamma: float
    p_inf: float = 0.0

    @property
    def u(self) -> NDArray[np.float64]:
        """Velocity."""
        return self.rho_u / self.rho

    @property
    def p(self) -> NDArray[np.float64]:
        """Pressure."""
        return pressure_from_total_energy(
            self.rho, self.rho_u, self.E, self.gamma, self.p_inf
        )

    @property
    def c(self) -> NDArray[np.float64]:
        """Sound speed."""
        return sound_speed(self.rho, self.p, self.gamma, self.p_inf)

    def copy(self) -> "EulerState1D":
        """Create a copy of the state."""
        return EulerState1D(
            rho=self.rho.copy(),
            rho_u=self.rho_u.copy(),
            E=self.E.copy(),
            gamma=self.gamma,
            p_inf=self.p_inf,
        )


def check_cfl_euler_1d(
    state: EulerState1D,
    dx: float,
    dt: float,
) -> float:
    """
    Compute CFL number and warn if > 1.

    CFL = max(|u| + c) * dt / dx

    Parameters
    ----------
    state : EulerState1D
        Current state.
    dx : float
        Grid spacing.
    dt : float
        Time step.

    Returns
    -------
    float
        CFL number.
    """
    max_speed = np.max(np.abs(state.u) + state.c)
    cfl = max_speed * dt / dx

    if cfl > 1.0:
        dt_stable = 0.9 * dx / max_speed
        warnings.warn(
            f"CFL = {cfl:.3f} > 1.0: simulation may be unstable. "
            f"Consider reducing dt to < {dt_stable:.6e}"
        )

    return cfl


def apply_bc_euler_1d(
    state: EulerState1D,
    bc_type: Literal["transmissive", "reflective", "periodic"],
) -> tuple[NDArray[np.float64], ...]:
    """
    Apply boundary conditions to 1D Euler state.

    Returns ghost cell values for left and right boundaries.

    Parameters
    ----------
    state : EulerState1D
        Current state.
    bc_type : str
        Boundary condition type:
        - "transmissive": Zero-gradient (copy edge values)
        - "reflective": Mirror with reversed velocity
        - "periodic": Wrap around

    Returns
    -------
    tuple
        (rho_ghost_L, rho_u_ghost_L, E_ghost_L,
         rho_ghost_R, rho_u_ghost_R, E_ghost_R)
    """
    if bc_type == "transmissive":
        # Zero-gradient: copy edge values
        return (
            state.rho[0], state.rho_u[0], state.E[0],
            state.rho[-1], state.rho_u[-1], state.E[-1],
        )
    elif bc_type == "reflective":
        # Mirror with reversed velocity
        return (
            state.rho[0], -state.rho_u[0], state.E[0],
            state.rho[-1], -state.rho_u[-1], state.E[-1],
        )
    elif bc_type == "periodic":
        # Wrap around
        return (
            state.rho[-1], state.rho_u[-1], state.E[-1],
            state.rho[0], state.rho_u[0], state.E[0],
        )
    else:
        raise ValueError(f"Unknown BC type: {bc_type}")


def _compute_euler_flux_1d(
    rho_L: NDArray[np.float64],
    u_L: NDArray[np.float64],
    p_L: NDArray[np.float64],
    E_L: NDArray[np.float64],
    c_L: NDArray[np.float64],
    rho_R: NDArray[np.float64],
    u_R: NDArray[np.float64],
    p_R: NDArray[np.float64],
    E_R: NDArray[np.float64],
    c_R: NDArray[np.float64],
    flux_calculator: Literal["ausm_plus_up", "hllc"],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Dispatch Euler flux computation to the selected calculator."""
    if flux_calculator == "ausm_plus_up":
        return ausm_plus_up_flux_1d(
            rho_L, u_L, p_L, E_L, c_L,
            rho_R, u_R, p_R, E_R, c_R,
        )
    if flux_calculator == "hllc":
        return hllc_flux_1d(
            rho_L, u_L, p_L, E_L, c_L,
            rho_R, u_R, p_R, E_R, c_R,
        )
    raise ValueError(f"Unknown flux calculator: {flux_calculator}")


def euler_step_1d(
    state: EulerState1D,
    dx: float,
    dt: float,
    bc_type: Literal["transmissive", "reflective", "periodic"] = "transmissive",
    use_muscl: bool = True,
    flux_calculator: Literal["ausm_plus_up", "hllc"] = "ausm_plus_up",
) -> EulerState1D:
    """
    Perform one explicit Euler time step for 1D compressible flow.

    Uses selected intercell flux and forward Euler time integration.

    Parameters
    ----------
    state : EulerState1D
        Current state (conserved variables + EOS params).
    dx : float
        Grid spacing.
    dt : float
        Time step.
    bc_type : str
        Boundary condition type.
    use_muscl : bool
        If True, use MUSCL reconstruction with Barth-Jespersen limiter.
    flux_calculator : {"ausm_plus_up", "hllc"}
        Intercell flux calculator.

    Returns
    -------
    EulerState1D
        Updated state after one time step.
    """
    n = len(state.rho)

    # Get primitive variables
    rho = state.rho
    u = state.u
    p = state.p
    E = state.E
    c = state.c
    gamma = state.gamma
    p_inf = state.p_inf

    # Apply boundary conditions to get ghost values
    bc = apply_bc_euler_1d(state, bc_type)
    rho_gL, rho_u_gL, E_gL = bc[0], bc[1], bc[2]
    rho_gR, rho_u_gR, E_gR = bc[3], bc[4], bc[5]

    # Compute ghost primitives
    u_gL = rho_u_gL / rho_gL
    u_gR = rho_u_gR / rho_gR
    p_gL = pressure_from_total_energy(rho_gL, rho_u_gL, E_gL, gamma, p_inf)
    p_gR = pressure_from_total_energy(rho_gR, rho_u_gR, E_gR, gamma, p_inf)
    c_gL = sound_speed(rho_gL, p_gL, gamma, p_inf)
    c_gR = sound_speed(rho_gR, p_gR, gamma, p_inf)

    # Extend arrays with ghost cells for flux computation
    rho_ext = np.concatenate([[rho_gL], rho, [rho_gR]])
    u_ext = np.concatenate([[u_gL], u, [u_gR]])
    p_ext = np.concatenate([[p_gL], p, [p_gR]])
    E_ext = np.concatenate([[E_gL], E, [E_gR]])
    c_ext = np.concatenate([[c_gL], c, [c_gR]])

    # Get left and right states at all interfaces (n+1 interfaces for n cells + 2 ghosts)
    rho_L, u_L, p_L, E_L, c_L, rho_R, u_R, p_R, E_R, c_R = compute_interface_states_1d(
        rho_ext, u_ext, p_ext, E_ext, c_ext, use_muscl=use_muscl
    )

    # Compute interface fluxes at all interfaces
    F_rho, F_rho_u, F_E = _compute_euler_flux_1d(
        rho_L, u_L, p_L, E_L, c_L,
        rho_R, u_R, p_R, E_R, c_R,
        flux_calculator=flux_calculator,
    )

    # F has n+1 elements (interfaces between ghost and interior cells)
    # For cell i (interior), flux difference is F[i+1] - F[i]
    # Interior cells are indices 1 to n in extended array, so fluxes are F[1:n+1] - F[0:n]
    # But our F array corresponds to interfaces 0 to n (between cells 0-1, 1-2, ..., n-n+1)
    # For interior cell i (0-indexed in original), the bounding fluxes are F[i] and F[i+1]

    # Update conserved variables: U^{n+1} = U^n - dt/dx * (F_{i+1/2} - F_{i-1/2})
    rho_new = rho - dt / dx * (F_rho[1:n+1] - F_rho[0:n])
    rho_u_new = state.rho_u - dt / dx * (F_rho_u[1:n+1] - F_rho_u[0:n])
    E_new = E - dt / dx * (F_E[1:n+1] - F_E[0:n])

    # Ensure positivity (numerical safeguard)
    rho_new = np.maximum(rho_new, 1e-10)
    # Ensure positive pressure (check and warn)
    p_new = pressure_from_total_energy(rho_new, rho_u_new, E_new, gamma, p_inf)
    if np.any(p_new < 0):
        warnings.warn("Negative pressure detected. Consider reducing time step.")
        # Clamp to small positive value
        min_p = 1e-6
        e_int = np.maximum(p_new + gamma * p_inf, min_p * (gamma - 1)) / ((gamma - 1) * rho_new)
        kinetic = 0.5 * rho_u_new ** 2 / rho_new
        E_new = rho_new * e_int + kinetic

    return EulerState1D(
        rho=rho_new,
        rho_u=rho_u_new,
        E=E_new,
        gamma=gamma,
        p_inf=p_inf,
    )


def create_initial_state_riemann_1d(
    x: NDArray[np.float64],
    x_discontinuity: float,
    rho_L: float,
    u_L: float,
    p_L: float,
    rho_R: float,
    u_R: float,
    p_R: float,
    gamma: float,
    p_inf: float = 0.0,
) -> EulerState1D:
    """
    Create initial state for a Riemann problem (shock tube).

    Parameters
    ----------
    x : NDArray
        Cell center coordinates.
    x_discontinuity : float
        Location of the initial discontinuity.
    rho_L, u_L, p_L : float
        Left state (density, velocity, pressure).
    rho_R, u_R, p_R : float
        Right state.
    gamma : float
        Heat capacity ratio.
    p_inf : float
        Stiffness parameter.

    Returns
    -------
    EulerState1D
        Initial state.
    """
    n = len(x)
    rho = np.where(x < x_discontinuity, rho_L, rho_R)
    u = np.where(x < x_discontinuity, u_L, u_R)
    p = np.where(x < x_discontinuity, p_L, p_R)

    # Convert to conserved variables
    rho_u = rho * u
    E = total_energy_from_primitives(rho, u, p, gamma, p_inf)

    return EulerState1D(rho=rho, rho_u=rho_u, E=E, gamma=gamma, p_inf=p_inf)


def run_euler_1d(
    state: EulerState1D,
    x: NDArray[np.float64],
    dx: float,
    dt: float,
    n_steps: int,
    bc_type: Literal["transmissive", "reflective", "periodic"] = "transmissive",
    callback: callable = None,
) -> EulerState1D:
    """
    Run 1D Euler simulation for multiple time steps.

    Parameters
    ----------
    state : EulerState1D
        Initial state.
    x : NDArray
        Cell center coordinates.
    dx : float
        Grid spacing.
    dt : float
        Time step.
    n_steps : int
        Number of time steps.
    bc_type : str
        Boundary condition type.
    callback : callable or None
        Optional callback(step, t, state) called each step.

    Returns
    -------
    EulerState1D
        Final state.
    """
    t = 0.0

    # Check CFL at start
    check_cfl_euler_1d(state, dx, dt)

    if callback:
        callback(0, t, state)

    for step in range(1, n_steps + 1):
        state = euler_step_1d(state, dx, dt, bc_type)
        t += dt

        if callback:
            callback(step, t, state)

    return state


# ---------------------------------------------------------------------------
# Two-Phase Euler Solver (Pressure Equilibrium)
# ---------------------------------------------------------------------------


@dataclass
class TwoPhaseEulerState1D:
    """Container for 1D two-phase Euler conserved and primitive variables."""

    # Conserved variables
    rho: NDArray[np.float64]      # Mixture density
    rho_u: NDArray[np.float64]    # Momentum (ρu)
    E: NDArray[np.float64]        # Total energy
    alpha: NDArray[np.float64]    # Volume fraction of phase 1

    # EOS parameters (required - no defaults)
    gamma1: float
    gamma2: float
    
    # Stiffness parameters (optional with defaults)
    p_inf1: float = 0.0
    p_inf2: float = 0.0

    @property
    def u(self) -> NDArray[np.float64]:
        """Velocity."""
        return self.rho_u / (self.rho + 1e-30)

    @property
    def p(self) -> NDArray[np.float64]:
        """Pressure (from mixture EOS)."""
        return mixture_pressure_from_conservatives(
            self.rho, self.rho_u, self.E, self.alpha,
            self.gamma1, self.gamma2, self.p_inf1, self.p_inf2
        )

    @property
    def c(self) -> NDArray[np.float64]:
        """Sound speed (mixture, using effective gamma approximation)."""
        # For simplicity, use volume-weighted effective gamma
        # More accurate: use Wood's formula with phase sound speeds
        gamma_eff = self.alpha * self.gamma1 + (1.0 - self.alpha) * self.gamma2
        p_inf_eff = self.alpha * self.p_inf1 + (1.0 - self.alpha) * self.p_inf2
        return sound_speed(self.rho, self.p, gamma_eff, p_inf_eff)

    def copy(self) -> "TwoPhaseEulerState1D":
        """Create a copy of the state."""
        return TwoPhaseEulerState1D(
            rho=self.rho.copy(),
            rho_u=self.rho_u.copy(),
            E=self.E.copy(),
            alpha=self.alpha.copy(),
            gamma1=self.gamma1,
            p_inf1=self.p_inf1,
            gamma2=self.gamma2,
            p_inf2=self.p_inf2,
        )


def check_cfl_euler_two_phase_1d(
    state: TwoPhaseEulerState1D,
    dx: float,
    dt: float,
) -> float:
    """Compute CFL number and warn if > 1."""
    max_speed = np.max(np.abs(state.u) + state.c)
    cfl = max_speed * dt / dx

    if cfl > 1.0:
        dt_stable = 0.9 * dx / max_speed
        warnings.warn(
            f"CFL = {cfl:.3f} > 1.0: simulation may be unstable. "
            f"Consider reducing dt to < {dt_stable:.6e}"
        )

    return cfl


def apply_bc_euler_two_phase_1d(
    state: TwoPhaseEulerState1D,
    bc_type: Literal["transmissive", "reflective", "periodic"],
) -> tuple[NDArray[np.float64], ...]:
    """Apply BCs to two-phase Euler state. Returns ghost values."""
    if bc_type == "transmissive":
        return (
            state.rho[0], state.rho_u[0], state.E[0], state.alpha[0],
            state.rho[-1], state.rho_u[-1], state.E[-1], state.alpha[-1],
        )
    elif bc_type == "reflective":
        return (
            state.rho[0], -state.rho_u[0], state.E[0], state.alpha[0],
            state.rho[-1], -state.rho_u[-1], state.E[-1], state.alpha[-1],
        )
    elif bc_type == "periodic":
        return (
            state.rho[-1], state.rho_u[-1], state.E[-1], state.alpha[-1],
            state.rho[0], state.rho_u[0], state.E[0], state.alpha[0],
        )
    else:
        raise ValueError(f"Unknown BC type: {bc_type}")


def euler_step_two_phase_1d(
    state: TwoPhaseEulerState1D,
    dx: float,
    dt: float,
    bc_type: Literal["transmissive", "reflective", "periodic"] = "transmissive",
    use_muscl: bool = True,
    flux_calculator: Literal["ausm_plus_up", "hllc"] = "ausm_plus_up",
) -> TwoPhaseEulerState1D:
    """
    Perform one explicit Euler time step for 1D two-phase flow.

    Uses selected intercell flux with effective mixture properties.
    Alpha is advected with the flow velocity (non-conservative transport).
    
    Parameters
    ----------
    state : TwoPhaseEulerState1D
        Current state.
    dx : float
        Grid spacing.
    dt : float
        Time step.
    bc_type : str
        Boundary condition type.
    use_muscl : bool
        If True, use MUSCL reconstruction with Barth-Jespersen limiter.
        If False, use first-order (piecewise constant).
    """
    n = len(state.rho)
    gamma1, gamma2 = state.gamma1, state.gamma2
    p_inf1, p_inf2 = state.p_inf1, state.p_inf2

    # Get primitive variables
    rho = state.rho
    u = state.u
    p = state.p
    E = state.E
    c = state.c
    alpha = state.alpha

    # Effective gamma for flux computation (volume-weighted)
    gamma_eff = alpha * gamma1 + (1.0 - alpha) * gamma2
    p_inf_eff = alpha * p_inf1 + (1.0 - alpha) * p_inf2

    # Apply boundary conditions
    bc = apply_bc_euler_two_phase_1d(state, bc_type)
    rho_gL, rho_u_gL, E_gL, alpha_gL = bc[0], bc[1], bc[2], bc[3]
    rho_gR, rho_u_gR, E_gR, alpha_gR = bc[4], bc[5], bc[6], bc[7]

    # Compute ghost primitives with effective gamma
    u_gL = rho_u_gL / (rho_gL + 1e-30)
    u_gR = rho_u_gR / (rho_gR + 1e-30)
    gamma_gL = alpha_gL * gamma1 + (1.0 - alpha_gL) * gamma2
    gamma_gR = alpha_gR * gamma1 + (1.0 - alpha_gR) * gamma2
    p_inf_gL = alpha_gL * p_inf1 + (1.0 - alpha_gL) * p_inf2
    p_inf_gR = alpha_gR * p_inf1 + (1.0 - alpha_gR) * p_inf2
    p_gL = mixture_pressure_from_conservatives(
        rho_gL, rho_u_gL, E_gL, alpha_gL, gamma1, gamma2, p_inf1, p_inf2
    )
    p_gR = mixture_pressure_from_conservatives(
        rho_gR, rho_u_gR, E_gR, alpha_gR, gamma1, gamma2, p_inf1, p_inf2
    )
    c_gL = sound_speed(rho_gL, p_gL, gamma_gL, p_inf_gL)
    c_gR = sound_speed(rho_gR, p_gR, gamma_gR, p_inf_gR)

    # Extend arrays with ghost cells
    rho_ext = np.concatenate([[rho_gL], rho, [rho_gR]])
    u_ext = np.concatenate([[u_gL], u, [u_gR]])
    p_ext = np.concatenate([[p_gL], p, [p_gR]])
    E_ext = np.concatenate([[E_gL], E, [E_gR]])
    c_ext = np.concatenate([[c_gL], c, [c_gR]])
    alpha_ext = np.concatenate([[alpha_gL], alpha, [alpha_gR]])

    # Interface states (MUSCL or first-order)
    rho_L, u_L, p_L, E_L, c_L, rho_R, u_R, p_R, E_R, c_R = compute_interface_states_1d(
        rho_ext, u_ext, p_ext, E_ext, c_ext, use_muscl=use_muscl
    )

    # Compute interface fluxes
    F_rho, F_rho_u, F_E = _compute_euler_flux_1d(
        rho_L, u_L, p_L, E_L, c_L,
        rho_R, u_R, p_R, E_R, c_R,
        flux_calculator=flux_calculator,
    )

    # Update conserved variables
    rho_new = rho - dt / dx * (F_rho[1:n+1] - F_rho[0:n])
    rho_u_new = state.rho_u - dt / dx * (F_rho_u[1:n+1] - F_rho_u[0:n])
    E_new = E - dt / dx * (F_E[1:n+1] - F_E[0:n])

    # Advect alpha (non-conservative transport: ∂α/∂t + u ∂α/∂x = 0)
    # Use upwind scheme with optional MUSCL reconstruction
    if use_muscl:
        from ..limiters import muscl_reconstruct_1d
        alpha_L_muscl, alpha_R_muscl = muscl_reconstruct_1d(alpha_ext)
        # Upwind: use left state if velocity > 0, right state if velocity < 0
        u_face = 0.5 * (u_L + u_R)  # Face velocities from MUSCL states
        alpha_flux = np.where(u_face >= 0, u_face * alpha_L_muscl, u_face * alpha_R_muscl)
    else:
        alpha_L_fo = alpha_ext[:-1]  # Left neighbors (first order)
        alpha_R_fo = alpha_ext[1:]   # Right neighbors
        u_face = 0.5 * (u_ext[:-1] + u_ext[1:])  # Face velocities
        alpha_flux = np.where(u_face >= 0, u_face * alpha_L_fo, u_face * alpha_R_fo)

    # Update alpha: α_new = α - dt/dx * (F_{i+1/2} - F_{i-1/2})
    alpha_new = alpha - dt / dx * (alpha_flux[1:n+1] - alpha_flux[0:n])

    # Clamp alpha to [0, 1]
    alpha_new = np.clip(alpha_new, 0.0, 1.0)

    # Ensure positivity
    rho_new = np.maximum(rho_new, 1e-10)

    # Check for negative pressure
    p_new = mixture_pressure_from_conservatives(
        rho_new, rho_u_new, E_new, alpha_new, gamma1, gamma2, p_inf1, p_inf2
    )
    if np.any(p_new < 0):
        warnings.warn("Negative pressure detected. Consider reducing time step.")
        # Try to fix by limiting energy
        p_new = np.maximum(p_new, 1e-6)
        E_new = mixture_total_energy(rho_new, rho_u_new / rho_new, p_new, alpha_new,
                                      gamma1, gamma2, p_inf1, p_inf2)

    return TwoPhaseEulerState1D(
        rho=rho_new,
        rho_u=rho_u_new,
        E=E_new,
        alpha=alpha_new,
        gamma1=gamma1,
        gamma2=gamma2,
        p_inf1=p_inf1,
        p_inf2=p_inf2,
    )


def create_initial_state_riemann_two_phase_1d(
    x: NDArray[np.float64],
    x_discontinuity: float,
    rho_L: float,
    u_L: float,
    p_L: float,
    alpha_L: float,
    rho_R: float,
    u_R: float,
    p_R: float,
    alpha_R: float,
    gamma1: float,
    gamma2: float,
    p_inf1: float = 0.0,
    p_inf2: float = 0.0,
) -> TwoPhaseEulerState1D:
    """
    Create initial state for a two-phase Riemann problem.

    Parameters
    ----------
    x : NDArray
        Cell center coordinates.
    x_discontinuity : float
        Location of the discontinuity.
    rho_L, u_L, p_L, alpha_L : float
        Left state (mixture density, velocity, pressure, volume fraction).
    rho_R, u_R, p_R, alpha_R : float
        Right state.
    gamma1, gamma2 : float
        Phase heat capacity ratios.
    p_inf1, p_inf2 : float
        Phase stiffness parameters.

    Returns
    -------
    TwoPhaseEulerState1D
        Initial state.
    """
    rho = np.where(x < x_discontinuity, rho_L, rho_R)
    u = np.where(x < x_discontinuity, u_L, u_R)
    p = np.where(x < x_discontinuity, p_L, p_R)
    alpha = np.where(x < x_discontinuity, alpha_L, alpha_R)

    # Convert to conserved variables
    rho_u = rho * u
    E = mixture_total_energy(rho, u, p, alpha, gamma1, gamma2, p_inf1, p_inf2)

    return TwoPhaseEulerState1D(
        rho=rho,
        rho_u=rho_u,
        E=E,
        alpha=alpha,
        gamma1=gamma1,
        gamma2=gamma2,
        p_inf1=p_inf1,
        p_inf2=p_inf2,
    )

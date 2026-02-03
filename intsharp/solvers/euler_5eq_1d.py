"""
5-Equation Model for 1D Two-Phase Compressible Flow.

The 5-equation model (Allaire, Clerc, Kokh 2002; Kapila et al. 2001) tracks:
    1. α₁ρ₁  - Partial density of phase 1
    2. α₂ρ₂  - Partial density of phase 2
    3. ρu    - Mixture momentum
    4. E     - Mixture total energy
    5. α₁    - Volume fraction of phase 1

Conservation equations:
    ∂(α₁ρ₁)/∂t + ∂(α₁ρ₁u)/∂x = 0
    ∂(α₂ρ₂)/∂t + ∂(α₂ρ₂u)/∂x = 0
    ∂(ρu)/∂t + ∂(ρu² + p)/∂x = 0
    ∂E/∂t + ∂((E+p)u)/∂x = 0
    ∂α₁/∂t + u·∂α₁/∂x = 0  (non-conservative)

Pressure equilibrium assumption: p₁ = p₂ = p
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ..eos import sound_speed
from ..flux_ausm import ausm_plus_up_flux_1d, compute_interface_states_1d
from ..limiters import muscl_reconstruct_1d


@dataclass
class FiveEqState1D:
    """
    Container for 5-equation model state.
    
    Conserved variables:
        alpha1_rho1 : Partial density of phase 1 (α₁ρ₁)
        alpha2_rho2 : Partial density of phase 2 (α₂ρ₂)
        rho_u       : Mixture momentum (ρu)
        E           : Mixture total energy
        alpha1      : Volume fraction of phase 1
    """
    # Conserved variables
    alpha1_rho1: NDArray[np.float64]  # α₁ρ₁
    alpha2_rho2: NDArray[np.float64]  # α₂ρ₂
    rho_u: NDArray[np.float64]        # ρu
    E: NDArray[np.float64]            # Total energy
    alpha1: NDArray[np.float64]       # α₁ (volume fraction of phase 1)
    
    # EOS parameters
    gamma1: float
    gamma2: float
    p_inf1: float = 0.0
    p_inf2: float = 0.0
    
    @property
    def rho(self) -> NDArray[np.float64]:
        """Mixture density: ρ = α₁ρ₁ + α₂ρ₂."""
        return self.alpha1_rho1 + self.alpha2_rho2
    
    @property
    def alpha2(self) -> NDArray[np.float64]:
        """Volume fraction of phase 2."""
        return 1.0 - self.alpha1
    
    @property
    def rho1(self) -> NDArray[np.float64]:
        """Density of phase 1: ρ₁ = (α₁ρ₁)/α₁."""
        return self.alpha1_rho1 / (self.alpha1 + 1e-30)
    
    @property
    def rho2(self) -> NDArray[np.float64]:
        """Density of phase 2: ρ₂ = (α₂ρ₂)/α₂."""
        return self.alpha2_rho2 / (self.alpha2 + 1e-30)
    
    @property
    def u(self) -> NDArray[np.float64]:
        """Velocity: u = ρu/ρ."""
        return self.rho_u / (self.rho + 1e-30)
    
    @property
    def p(self) -> NDArray[np.float64]:
        """
        Pressure from energy equation (pressure equilibrium).
        
        E - ½ρu² = α₁(p + γ₁p∞₁)/(γ₁-1) + α₂(p + γ₂p∞₂)/(γ₂-1)
        
        Solving for p:
        E_int = p·[α₁/(γ₁-1) + α₂/(γ₂-1)] + [α₁γ₁p∞₁/(γ₁-1) + α₂γ₂p∞₂/(γ₂-1)]
        """
        rho = self.rho
        u = self.u
        kinetic = 0.5 * rho * u * u
        E_int = self.E - kinetic
        
        alpha1 = self.alpha1
        alpha2 = self.alpha2
        gamma1, gamma2 = self.gamma1, self.gamma2
        p_inf1, p_inf2 = self.p_inf1, self.p_inf2
        
        # Coefficients: E_int = p*A + B
        A = alpha1 / (gamma1 - 1.0) + alpha2 / (gamma2 - 1.0)
        B = alpha1 * gamma1 * p_inf1 / (gamma1 - 1.0) + alpha2 * gamma2 * p_inf2 / (gamma2 - 1.0)
        
        p = (E_int - B) / (A + 1e-30)
        return np.maximum(p, 1e-10)
    
    @property
    def c(self) -> NDArray[np.float64]:
        """
        Mixture sound speed (Wood's formula approximation).
        
        For simplicity, use volume-weighted effective gamma.
        """
        gamma_eff = self.alpha1 * self.gamma1 + self.alpha2 * self.gamma2
        p_inf_eff = self.alpha1 * self.p_inf1 + self.alpha2 * self.p_inf2
        return sound_speed(self.rho, self.p, gamma_eff, p_inf_eff)
    
    def copy(self) -> "FiveEqState1D":
        """Create a copy of the state."""
        return FiveEqState1D(
            alpha1_rho1=self.alpha1_rho1.copy(),
            alpha2_rho2=self.alpha2_rho2.copy(),
            rho_u=self.rho_u.copy(),
            E=self.E.copy(),
            alpha1=self.alpha1.copy(),
            gamma1=self.gamma1,
            gamma2=self.gamma2,
            p_inf1=self.p_inf1,
            p_inf2=self.p_inf2,
        )


def check_cfl_5eq_1d(state: FiveEqState1D, dx: float, dt: float) -> float:
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


def apply_bc_5eq_1d(
    state: FiveEqState1D,
    bc_type: Literal["transmissive", "reflective", "periodic"],
) -> tuple:
    """
    Apply boundary conditions to 5-equation state.
    
    Returns ghost cell values for left and right boundaries.
    """
    if bc_type == "transmissive":
        return (
            state.alpha1_rho1[0], state.alpha2_rho2[0], state.rho_u[0], state.E[0], state.alpha1[0],
            state.alpha1_rho1[-1], state.alpha2_rho2[-1], state.rho_u[-1], state.E[-1], state.alpha1[-1],
        )
    elif bc_type == "reflective":
        return (
            state.alpha1_rho1[0], state.alpha2_rho2[0], -state.rho_u[0], state.E[0], state.alpha1[0],
            state.alpha1_rho1[-1], state.alpha2_rho2[-1], -state.rho_u[-1], state.E[-1], state.alpha1[-1],
        )
    elif bc_type == "periodic":
        return (
            state.alpha1_rho1[-1], state.alpha2_rho2[-1], state.rho_u[-1], state.E[-1], state.alpha1[-1],
            state.alpha1_rho1[0], state.alpha2_rho2[0], state.rho_u[0], state.E[0], state.alpha1[0],
        )
    else:
        raise ValueError(f"Unknown BC type: {bc_type}")


def compute_pressure_5eq(
    alpha1_rho1: NDArray,
    alpha2_rho2: NDArray,
    rho_u: NDArray,
    E: NDArray,
    alpha1: NDArray,
    gamma1: float,
    gamma2: float,
    p_inf1: float,
    p_inf2: float,
) -> NDArray:
    """Compute pressure from 5-equation conserved variables."""
    rho = alpha1_rho1 + alpha2_rho2
    u = rho_u / (rho + 1e-30)
    kinetic = 0.5 * rho * u * u
    E_int = E - kinetic
    
    alpha2 = 1.0 - alpha1
    A = alpha1 / (gamma1 - 1.0) + alpha2 / (gamma2 - 1.0)
    B = alpha1 * gamma1 * p_inf1 / (gamma1 - 1.0) + alpha2 * gamma2 * p_inf2 / (gamma2 - 1.0)
    
    p = (E_int - B) / (A + 1e-30)
    return np.maximum(p, 1e-10)


def euler_step_5eq_1d(
    state: FiveEqState1D,
    dx: float,
    dt: float,
    bc_type: Literal["transmissive", "reflective", "periodic"] = "transmissive",
    use_muscl: bool = True,
) -> FiveEqState1D:
    """
    Perform one time step for 5-equation model.
    
    Uses AUSM+UP flux for conservative equations.
    Alpha is advected non-conservatively.
    """
    n = len(state.alpha1_rho1)
    gamma1, gamma2 = state.gamma1, state.gamma2
    p_inf1, p_inf2 = state.p_inf1, state.p_inf2
    
    # Get state variables
    alpha1_rho1 = state.alpha1_rho1
    alpha2_rho2 = state.alpha2_rho2
    rho_u = state.rho_u
    E = state.E
    alpha1 = state.alpha1
    
    # Derived quantities
    rho = state.rho
    u = state.u
    p = state.p
    c = state.c
    
    # Apply boundary conditions
    bc = apply_bc_5eq_1d(state, bc_type)
    a1r1_gL, a2r2_gL, rhou_gL, E_gL, a1_gL = bc[0], bc[1], bc[2], bc[3], bc[4]
    a1r1_gR, a2r2_gR, rhou_gR, E_gR, a1_gR = bc[5], bc[6], bc[7], bc[8], bc[9]
    
    # Ghost cell derived quantities
    rho_gL = a1r1_gL + a2r2_gL
    rho_gR = a1r1_gR + a2r2_gR
    u_gL = rhou_gL / (rho_gL + 1e-30)
    u_gR = rhou_gR / (rho_gR + 1e-30)
    p_gL = compute_pressure_5eq(a1r1_gL, a2r2_gL, rhou_gL, E_gL, a1_gL, gamma1, gamma2, p_inf1, p_inf2)
    p_gR = compute_pressure_5eq(a1r1_gR, a2r2_gR, rhou_gR, E_gR, a1_gR, gamma1, gamma2, p_inf1, p_inf2)
    gamma_gL = a1_gL * gamma1 + (1.0 - a1_gL) * gamma2
    gamma_gR = a1_gR * gamma1 + (1.0 - a1_gR) * gamma2
    p_inf_gL = a1_gL * p_inf1 + (1.0 - a1_gL) * p_inf2
    p_inf_gR = a1_gR * p_inf1 + (1.0 - a1_gR) * p_inf2
    c_gL = sound_speed(rho_gL, p_gL, gamma_gL, p_inf_gL)
    c_gR = sound_speed(rho_gR, p_gR, gamma_gR, p_inf_gR)
    
    # Extend arrays with ghost cells
    alpha1_rho1_ext = np.concatenate([[a1r1_gL], alpha1_rho1, [a1r1_gR]])
    alpha2_rho2_ext = np.concatenate([[a2r2_gL], alpha2_rho2, [a2r2_gR]])
    rho_ext = np.concatenate([[rho_gL], rho, [rho_gR]])
    u_ext = np.concatenate([[u_gL], u, [u_gR]])
    p_ext = np.concatenate([[p_gL], p, [p_gR]])
    E_ext = np.concatenate([[E_gL], E, [E_gR]])
    c_ext = np.concatenate([[c_gL], c, [c_gR]])
    alpha1_ext = np.concatenate([[a1_gL], alpha1, [a1_gR]])
    
    # MUSCL reconstruction for flux computation
    if use_muscl:
        rho_L, rho_R = muscl_reconstruct_1d(rho_ext)
        u_L, u_R = muscl_reconstruct_1d(u_ext)
        p_L, p_R = muscl_reconstruct_1d(p_ext)
        E_L, E_R = muscl_reconstruct_1d(E_ext)
        c_L, c_R = muscl_reconstruct_1d(c_ext)
        alpha1_rho1_L, alpha1_rho1_R = muscl_reconstruct_1d(alpha1_rho1_ext)
        alpha2_rho2_L, alpha2_rho2_R = muscl_reconstruct_1d(alpha2_rho2_ext)
        alpha1_L, alpha1_R = muscl_reconstruct_1d(alpha1_ext)
        
        # Ensure positivity
        rho_L = np.maximum(rho_L, 1e-10)
        rho_R = np.maximum(rho_R, 1e-10)
        p_L = np.maximum(p_L, 1e-10)
        p_R = np.maximum(p_R, 1e-10)
        c_L = np.maximum(c_L, 1e-10)
        c_R = np.maximum(c_R, 1e-10)
        alpha1_rho1_L = np.maximum(alpha1_rho1_L, 0.0)
        alpha1_rho1_R = np.maximum(alpha1_rho1_R, 0.0)
        alpha2_rho2_L = np.maximum(alpha2_rho2_L, 0.0)
        alpha2_rho2_R = np.maximum(alpha2_rho2_R, 0.0)
        alpha1_L = np.clip(alpha1_L, 0.0, 1.0)
        alpha1_R = np.clip(alpha1_R, 0.0, 1.0)
    else:
        # First-order
        rho_L, rho_R = rho_ext[:-1], rho_ext[1:]
        u_L, u_R = u_ext[:-1], u_ext[1:]
        p_L, p_R = p_ext[:-1], p_ext[1:]
        E_L, E_R = E_ext[:-1], E_ext[1:]
        c_L, c_R = c_ext[:-1], c_ext[1:]
        alpha1_rho1_L, alpha1_rho1_R = alpha1_rho1_ext[:-1], alpha1_rho1_ext[1:]
        alpha2_rho2_L, alpha2_rho2_R = alpha2_rho2_ext[:-1], alpha2_rho2_ext[1:]
        alpha1_L, alpha1_R = alpha1_ext[:-1], alpha1_ext[1:]
    
    # Compute AUSM+UP fluxes for mixture (ρ, ρu, E)
    F_rho, F_rho_u, F_E = ausm_plus_up_flux_1d(
        rho_L, u_L, p_L, E_L, c_L,
        rho_R, u_R, p_R, E_R, c_R,
    )
    
    # Compute fluxes for partial densities (α₁ρ₁ and α₂ρ₂)
    # These are advected with velocity u: F = (αᵢρᵢ)·u
    # Use upwind based on interface velocity
    u_face = 0.5 * (u_L + u_R)
    
    F_alpha1_rho1 = np.where(u_face >= 0, 
                             alpha1_rho1_L * u_L,
                             alpha1_rho1_R * u_R)
    
    F_alpha2_rho2 = np.where(u_face >= 0,
                             alpha2_rho2_L * u_L,
                             alpha2_rho2_R * u_R)
    
    # Update conserved variables
    alpha1_rho1_new = alpha1_rho1 - dt / dx * (F_alpha1_rho1[1:n+1] - F_alpha1_rho1[0:n])
    alpha2_rho2_new = alpha2_rho2 - dt / dx * (F_alpha2_rho2[1:n+1] - F_alpha2_rho2[0:n])
    rho_u_new = rho_u - dt / dx * (F_rho_u[1:n+1] - F_rho_u[0:n])
    E_new = E - dt / dx * (F_E[1:n+1] - F_E[0:n])
    
    # Advect alpha1 non-conservatively: ∂α₁/∂t + u·∂α₁/∂x = 0
    # For non-conservative transport, we use: α_new = α - dt × u × (∂α/∂x)
    # Use upwind for the gradient
    # Cell-centered velocity
    u_cell = rho_u / (rho + 1e-30)
    
    # Alpha gradient using upwind (based on local velocity)
    # For interface i+1/2, if u > 0: grad uses (α_i - α_{i-1}), else (α_{i+1} - α_i)
    # But we need cell-centered update, so use face values
    alpha1_L_face = alpha1_L  # Alpha at left of each interface
    alpha1_R_face = alpha1_R  # Alpha at right of each interface
    
    # Non-conservative: ∂α/∂t + u·∂α/∂x = 0
    # Discretize as: (α_new - α) / dt + u * (α_{i+1/2} - α_{i-1/2}) / dx = 0
    # Use upwind for face values
    u_face = 0.5 * (u_L + u_R)
    alpha1_face = np.where(u_face >= 0, alpha1_L_face, alpha1_R_face)
    
    # Gradient term: (α_{i+1/2} - α_{i-1/2}) / dx, multiplied by cell velocity
    # This is still not quite right for non-conservative form
    # Better approach: use cell velocity and upwind gradient
    alpha1_ext_interior = alpha1_ext[1:-1]  # Interior cells
    alpha1_left = alpha1_ext[:-2]   # Left neighbors
    alpha1_right = alpha1_ext[2:]   # Right neighbors
    
    # Upwind gradient based on cell velocity
    grad_alpha1 = np.where(
        u_cell >= 0,
        (alpha1_ext_interior - alpha1_left) / dx,
        (alpha1_right - alpha1_ext_interior) / dx
    )
    
    alpha1_new = alpha1 - dt * u_cell * grad_alpha1
    
    # Clamp values
    alpha1_rho1_new = np.maximum(alpha1_rho1_new, 0.0)
    alpha2_rho2_new = np.maximum(alpha2_rho2_new, 0.0)
    alpha1_new = np.clip(alpha1_new, 1e-10, 1.0 - 1e-10)  # Avoid exact 0 or 1
    
    # Check for negative pressure
    p_new = compute_pressure_5eq(
        alpha1_rho1_new, alpha2_rho2_new, rho_u_new, E_new, alpha1_new,
        gamma1, gamma2, p_inf1, p_inf2
    )
    if np.any(p_new < 0):
        warnings.warn("Negative pressure detected. Consider reducing time step.")
    
    return FiveEqState1D(
        alpha1_rho1=alpha1_rho1_new,
        alpha2_rho2=alpha2_rho2_new,
        rho_u=rho_u_new,
        E=E_new,
        alpha1=alpha1_new,
        gamma1=gamma1,
        gamma2=gamma2,
        p_inf1=p_inf1,
        p_inf2=p_inf2,
    )


def create_initial_state_riemann_5eq_1d(
    x: NDArray[np.float64],
    x_discontinuity: float,
    rho1_L: float,
    rho2_L: float,
    u_L: float,
    p_L: float,
    alpha1_L: float,
    rho1_R: float,
    rho2_R: float,
    u_R: float,
    p_R: float,
    alpha1_R: float,
    gamma1: float,
    gamma2: float,
    p_inf1: float = 0.0,
    p_inf2: float = 0.0,
) -> FiveEqState1D:
    """
    Create initial state for 5-equation Riemann problem.
    
    Parameters
    ----------
    x : NDArray
        Cell center coordinates.
    x_discontinuity : float
        Location of the discontinuity.
    rho1_L, rho2_L : float
        Phase densities on the left.
    u_L, p_L : float
        Velocity and pressure on the left.
    alpha1_L : float
        Volume fraction of phase 1 on the left.
    rho1_R, rho2_R : float
        Phase densities on the right.
    u_R, p_R : float
        Velocity and pressure on the right.
    alpha1_R : float
        Volume fraction of phase 1 on the right.
    gamma1, gamma2 : float
        Heat capacity ratios.
    p_inf1, p_inf2 : float
        Stiffness parameters.
    
    Returns
    -------
    FiveEqState1D
        Initial state.
    """
    # Volume fractions
    alpha1 = np.where(x < x_discontinuity, alpha1_L, alpha1_R)
    alpha2 = 1.0 - alpha1
    
    # Phase densities
    rho1 = np.where(x < x_discontinuity, rho1_L, rho1_R)
    rho2 = np.where(x < x_discontinuity, rho2_L, rho2_R)
    
    # Partial densities
    alpha1_rho1 = alpha1 * rho1
    alpha2_rho2 = alpha2 * rho2
    
    # Mixture density and velocity
    rho = alpha1_rho1 + alpha2_rho2
    u = np.where(x < x_discontinuity, u_L, u_R)
    p = np.where(x < x_discontinuity, p_L, p_R)
    
    # Momentum
    rho_u = rho * u
    
    # Total energy
    # E = α₁(p + γ₁p∞₁)/(γ₁-1) + α₂(p + γ₂p∞₂)/(γ₂-1) + ½ρu²
    E_int = alpha1 * (p + gamma1 * p_inf1) / (gamma1 - 1.0) + \
            alpha2 * (p + gamma2 * p_inf2) / (gamma2 - 1.0)
    E = E_int + 0.5 * rho * u * u
    
    return FiveEqState1D(
        alpha1_rho1=alpha1_rho1,
        alpha2_rho2=alpha2_rho2,
        rho_u=rho_u,
        E=E,
        alpha1=alpha1,
        gamma1=gamma1,
        gamma2=gamma2,
        p_inf1=p_inf1,
        p_inf2=p_inf2,
    )

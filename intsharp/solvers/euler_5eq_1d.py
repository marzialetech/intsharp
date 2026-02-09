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

from ..eos import sound_speed, mixture_sound_speed_wood
from ..flux_ausm import ausm_plus_up_flux_1d_with_v_riem
from ..flux_hllc import hllc_flux_1d_with_v_riem
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
        Mixture sound speed via Wood's formula.
        
        1/(ρ_mix c_mix²) = α₁/(ρ₁ c₁²) + α₂/(ρ₂ c₂²)
        """
        rho1 = np.maximum(self.rho1, 1e-10)
        rho2 = np.maximum(self.rho2, 1e-10)
        p = self.p
        c1 = sound_speed(rho1, p, self.gamma1, self.p_inf1)
        c2 = sound_speed(rho2, p, self.gamma2, self.p_inf2)
        c_mix = mixture_sound_speed_wood(
            self.rho, self.alpha1, rho1, rho2, c1, c2
        )
        return np.maximum(np.minimum(c_mix, 1e10), 1e-10)
    
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
    flux_calculator: Literal["ausm_plus_up", "hllc"] = "ausm_plus_up",
) -> FiveEqState1D:
    """
    Perform one time step for 5-equation model.
    
    Uses selected intercell flux for conservative equations.
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
        alpha1_rho1_L, alpha1_rho1_R = muscl_reconstruct_1d(alpha1_rho1_ext)
        alpha2_rho2_L, alpha2_rho2_R = muscl_reconstruct_1d(alpha2_rho2_ext)
        alpha1_L, alpha1_R = muscl_reconstruct_1d(alpha1_ext)
        
        # Ensure positivity
        rho_L = np.maximum(rho_L, 1e-10)
        rho_R = np.maximum(rho_R, 1e-10)
        p_L = np.maximum(p_L, 1e-10)
        p_R = np.maximum(p_R, 1e-10)
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
        alpha1_rho1_L, alpha1_rho1_R = alpha1_rho1_ext[:-1], alpha1_rho1_ext[1:]
        alpha2_rho2_L, alpha2_rho2_R = alpha2_rho2_ext[:-1], alpha2_rho2_ext[1:]
        alpha1_L, alpha1_R = alpha1_ext[:-1], alpha1_ext[1:]
    
    # Interface sound speed via Wood's formula (phase-specific c1, c2)
    alpha2_L = 1.0 - alpha1_L
    alpha2_R = 1.0 - alpha1_R
    rho1_L = np.maximum(alpha1_rho1_L / (alpha1_L + 1e-30), 1e-10)
    rho2_L = np.maximum(alpha2_rho2_L / (alpha2_L + 1e-30), 1e-10)
    rho1_R = np.maximum(alpha1_rho1_R / (alpha1_R + 1e-30), 1e-10)
    rho2_R = np.maximum(alpha2_rho2_R / (alpha2_R + 1e-30), 1e-10)
    c1_L = sound_speed(rho1_L, p_L, gamma1, p_inf1)
    c2_L = sound_speed(rho2_L, p_L, gamma2, p_inf2)
    c1_R = sound_speed(rho1_R, p_R, gamma1, p_inf1)
    c2_R = sound_speed(rho2_R, p_R, gamma2, p_inf2)
    c_L = mixture_sound_speed_wood(rho_L, alpha1_L, rho1_L, rho2_L, c1_L, c2_L)
    c_R = mixture_sound_speed_wood(rho_R, alpha1_R, rho1_R, rho2_R, c1_R, c2_R)
    c_L = np.maximum(np.minimum(c_L, 1e10), 1e-10)
    c_R = np.maximum(np.minimum(c_R, 1e10), 1e-10)
    
    # Compute interface fluxes for mixture (ρ, ρu, E) with v_riem for flux vector splitting
    if flux_calculator == "ausm_plus_up":
        F_rho, F_rho_u, F_E, v_riem = ausm_plus_up_flux_1d_with_v_riem(
            rho_L, u_L, p_L, E_L, c_L,
            rho_R, u_R, p_R, E_R, c_R,
        )
    elif flux_calculator == "hllc":
        F_rho, F_rho_u, F_E, v_riem = hllc_flux_1d_with_v_riem(
            rho_L, u_L, p_L, E_L, c_L,
            rho_R, u_R, p_R, E_R, c_R,
        )
    else:
        raise ValueError(f"Unknown flux calculator: {flux_calculator}")
    
    # Flux vector splitting with ℓ± and v_riem: use v_riem = m_dot/ρ for convective fluxes
    # Partial density fluxes: F_{αᵢρᵢ} = (αᵢρᵢ)_upwind * v_riem (ensures F_α₁ρ₁+F_α₂ρ₂=F_ρ)
    alpha1_rho1_upwind = np.where(v_riem >= 0, alpha1_rho1_L, alpha1_rho1_R)
    alpha2_rho2_upwind = np.where(v_riem >= 0, alpha2_rho2_L, alpha2_rho2_R)
    
    F_alpha1_rho1 = alpha1_rho1_upwind * v_riem
    F_alpha2_rho2 = alpha2_rho2_upwind * v_riem
    
    # Corrective vector for α_k: dissipative term ∝ (u_L - u_R) at interface for stability
    # Added to the α flux in the non-conservative equation (see alpha1 update below)
    
    # Update conserved variables
    alpha1_rho1_new = alpha1_rho1 - dt / dx * (F_alpha1_rho1[1:n+1] - F_alpha1_rho1[0:n])
    alpha2_rho2_new = alpha2_rho2 - dt / dx * (F_alpha2_rho2[1:n+1] - F_alpha2_rho2[0:n])
    rho_u_new = rho_u - dt / dx * (F_rho_u[1:n+1] - F_rho_u[0:n])
    E_new = E - dt / dx * (F_E[1:n+1] - F_E[0:n])
    
    # Advect alpha1 non-conservatively: ∂α₁/∂t + v_riem·∂α₁/∂x = 0
    # Use v_riem (interface velocity) for flux vector splitting
    alpha1_upwind = np.where(v_riem >= 0, alpha1_L, alpha1_R)
    F_alpha1 = alpha1_upwind * v_riem
    
    # Corrective vector for α_k: dissipative term ∝ (u_L - u_R) for stability at interfaces
    # Proportional to relative velocity (volume fraction coupling from enhanced AUSM+-up)
    K_alpha = 0.1
    alpha_bar = 0.5 * (alpha1_L + alpha1_R)
    correction_alpha = K_alpha * (u_L - u_R) * alpha_bar * (1.0 - alpha_bar)
    F_alpha1 = F_alpha1 + correction_alpha
    
    alpha1_new = alpha1 - dt / dx * (F_alpha1[1:n+1] - F_alpha1[0:n])
    
    # Post-step α_k normalization: ensure consistency with partial densities
    # α₁ = (α₁ρ₁) / ρ_mix (redefine from conserved variables for robustness)
    rho_new = alpha1_rho1_new + alpha2_rho2_new
    alpha1_new = np.where(
        rho_new > 1e-30,
        alpha1_rho1_new / rho_new,
        alpha1_new
    )
    
    # Clamp and ensure α₁ + α₂ = 1
    alpha1_rho1_new = np.maximum(alpha1_rho1_new, 0.0)
    alpha2_rho2_new = np.maximum(alpha2_rho2_new, 0.0)
    alpha1_new = np.clip(alpha1_new, 1e-10, 1.0 - 1e-10)
    
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

"""
AUSM+UP flux scheme for compressible Euler equations.

AUSM+UP (Advection Upstream Splitting Method with pressure-based improvements)
is an all-speed flux scheme that works from incompressible to supersonic flows.

References:
    - Liou, M.S. (2006). A sequel to AUSM, Part II: AUSM+-up for all speeds.
      Journal of Computational Physics, 214(1), 137-170.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .eos import sound_speed, enthalpy, pressure_from_total_energy


# ---------------------------------------------------------------------------
# AUSM+ Mach number and pressure splitting functions
# ---------------------------------------------------------------------------


def _mach_plus_1(M: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
    """First-order Mach splitting M₁⁺ = 0.5(M + |M|)."""
    return 0.5 * (M + np.abs(M))


def _mach_minus_1(M: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
    """First-order Mach splitting M₁⁻ = 0.5(M - |M|)."""
    return 0.5 * (M - np.abs(M))


def _mach_plus_2(M: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
    """Second-order Mach splitting M₂⁺ = 0.25(M + 1)²."""
    return 0.25 * (M + 1.0) ** 2


def _mach_minus_2(M: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
    """Second-order Mach splitting M₂⁻ = -0.25(M - 1)²."""
    return -0.25 * (M - 1.0) ** 2


def _mach_plus_4(M: NDArray[np.float64] | float, beta: float = 1.0 / 8.0) -> NDArray[np.float64] | float:
    """
    Fourth-order Mach splitting M₄⁺.

    M₄⁺ = M₁⁺           if |M| >= 1
        = M₂⁺(1 - 16β M₂⁻)  otherwise
    """
    M = np.asarray(M)
    result = np.where(
        np.abs(M) >= 1.0,
        _mach_plus_1(M),
        _mach_plus_2(M) * (1.0 - 16.0 * beta * _mach_minus_2(M))
    )
    return result


def _mach_minus_4(M: NDArray[np.float64] | float, beta: float = 1.0 / 8.0) -> NDArray[np.float64] | float:
    """
    Fourth-order Mach splitting M₄⁻.

    M₄⁻ = M₁⁻           if |M| >= 1
        = M₂⁻(1 + 16β M₂⁺)  otherwise
    """
    M = np.asarray(M)
    result = np.where(
        np.abs(M) >= 1.0,
        _mach_minus_1(M),
        _mach_minus_2(M) * (1.0 + 16.0 * beta * _mach_plus_2(M))
    )
    return result


def _pressure_plus_5(M: NDArray[np.float64] | float, alpha: float = 3.0 / 16.0) -> NDArray[np.float64] | float:
    """
    Fifth-order pressure splitting P₅⁺.

    P₅⁺ = 1/M * M₁⁺                           if |M| >= 1
        = M₂⁺((±2 - M) ∓ 16α M M₂⁻)          otherwise

    For AUSM+UP, this is: M₂⁺ * ((2 - M) - 16α M M₂⁻)
    """
    M = np.asarray(M)
    result = np.where(
        np.abs(M) >= 1.0,
        np.where(M != 0, _mach_plus_1(M) / M, 0.5),
        _mach_plus_2(M) * ((2.0 - M) - 16.0 * alpha * M * _mach_minus_2(M))
    )
    return result


def _pressure_minus_5(M: NDArray[np.float64] | float, alpha: float = 3.0 / 16.0) -> NDArray[np.float64] | float:
    """
    Fifth-order pressure splitting P₅⁻.

    P₅⁻ = 1/M * M₁⁻                           if |M| >= 1
        = M₂⁻((-2 - M) + 16α M M₂⁺)          otherwise
    """
    M = np.asarray(M)
    result = np.where(
        np.abs(M) >= 1.0,
        np.where(M != 0, _mach_minus_1(M) / M, 0.5),
        _mach_minus_2(M) * ((-2.0 - M) + 16.0 * alpha * M * _mach_plus_2(M))
    )
    return result


# ---------------------------------------------------------------------------
# AUSM+UP interface flux (1D)
# ---------------------------------------------------------------------------


def ausm_plus_up_flux_1d(
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
    Kp: float = 0.25,
    Ku: float = 0.75,
    sigma: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute AUSM+UP interface fluxes for 1D Euler equations.

    Parameters
    ----------
    rho_L, rho_R : NDArray
        Left and right densities at each interface.
    u_L, u_R : NDArray
        Left and right velocities at each interface.
    p_L, p_R : NDArray
        Left and right pressures at each interface.
    E_L, E_R : NDArray
        Left and right total energies at each interface.
    c_L, c_R : NDArray
        Left and right sound speeds at each interface.
    Kp : float
        Pressure diffusion coefficient (default 0.25).
    Ku : float
        Velocity diffusion coefficient (default 0.75).
    sigma : float
        Density ratio scaling (default 1.0).

    Returns
    -------
    tuple[NDArray, NDArray, NDArray]
        Interface fluxes for (F_rho, F_rho_u, F_E) at each interface.
    """
    # Interface sound speed (simple average)
    a_half = 0.5 * (c_L + c_R)

    # Local Mach numbers
    M_L = u_L / a_half
    M_R = u_R / a_half

    # Average density for scaling
    rho_avg = 0.5 * (rho_L + rho_R)

    # Mach number splitting (M₄±)
    M_plus = _mach_plus_4(M_L)
    M_minus = _mach_minus_4(M_R)

    # Pressure diffusion term (low-speed improvement)
    # M_p = -Kp * max(1 - M_bar², 0) * (p_R - p_L) / (rho_avg * a_half²)
    M_bar_sq = 0.5 * (M_L ** 2 + M_R ** 2)
    # Clamp to avoid issues
    M_bar_sq = np.minimum(M_bar_sq, 1.0)
    
    # Pressure diffusion
    p_diff = (p_R - p_L) / (rho_avg * a_half ** 2 + 1e-30)
    M_p = -Kp * np.maximum(1.0 - M_bar_sq, 0.0) * p_diff

    # Interface Mach number
    M_half = M_plus + M_minus + M_p

    # Pressure splitting (P₅±)
    P_plus = _pressure_plus_5(M_L)
    P_minus = _pressure_minus_5(M_R)

    # Velocity diffusion term (low-speed improvement)
    # p_u = -Ku * P₅⁺ * P₅⁻ * (rho_L + rho_R) * a_half * (u_R - u_L)
    p_u = -Ku * P_plus * P_minus * 2.0 * rho_avg * a_half * (u_R - u_L)

    # Interface pressure
    p_half = P_plus * p_L + P_minus * p_R + p_u

    # Mass flux: ṁ = a_half * M_half * ρ (upwind)
    # Upwind density based on sign of M_half
    rho_upwind = np.where(M_half >= 0, rho_L, rho_R)
    m_dot = a_half * M_half * rho_upwind

    # Convective fluxes (upwind)
    # Ψ = [1, u, H]ᵀ based on upwind direction
    H_L = enthalpy(rho_L, p_L, E_L)
    H_R = enthalpy(rho_R, p_R, E_R)

    u_upwind = np.where(M_half >= 0, u_L, u_R)
    H_upwind = np.where(M_half >= 0, H_L, H_R)

    # Fluxes
    F_rho = m_dot
    F_rho_u = m_dot * u_upwind + p_half
    F_E = m_dot * H_upwind

    return F_rho, F_rho_u, F_E


def compute_interface_states_1d(
    rho: NDArray[np.float64],
    u: NDArray[np.float64],
    p: NDArray[np.float64],
    E: NDArray[np.float64],
    c: NDArray[np.float64],
    use_muscl: bool = False,
) -> tuple:
    """
    Extract left and right states at cell interfaces for 1D.

    For cell i, interface i+1/2 has:
        Left state = cell i
        Right state = cell i+1

    Parameters
    ----------
    rho, u, p, E, c : NDArray
        Primitive and derived variables at cell centers (with ghost cells).
        For n interior cells, this should have n+2 elements.
    use_muscl : bool
        If True, use MUSCL reconstruction with Barth-Jespersen limiter.
        If False, use first-order (piecewise constant).

    Returns
    -------
    tuple
        (rho_L, u_L, p_L, E_L, c_L, rho_R, u_R, p_R, E_R, c_R)
        Each array has n+1 elements for n interior cells (interfaces 0 to n).
    """
    if use_muscl:
        from .limiters import muscl_reconstruct_1d
        
        # MUSCL reconstruction for each variable
        rho_L, rho_R = muscl_reconstruct_1d(rho)
        u_L, u_R = muscl_reconstruct_1d(u)
        p_L, p_R = muscl_reconstruct_1d(p)
        E_L, E_R = muscl_reconstruct_1d(E)
        c_L, c_R = muscl_reconstruct_1d(c)
        
        # Ensure positivity
        rho_L = np.maximum(rho_L, 1e-10)
        rho_R = np.maximum(rho_R, 1e-10)
        p_L = np.maximum(p_L, 1e-10)
        p_R = np.maximum(p_R, 1e-10)
        c_L = np.maximum(c_L, 1e-10)
        c_R = np.maximum(c_R, 1e-10)
        
        return rho_L, u_L, p_L, E_L, c_L, rho_R, u_R, p_R, E_R, c_R
    
    # First-order: Left states are cells 0 to n-2, Right states are cells 1 to n-1
    rho_L = rho[:-1]
    u_L = u[:-1]
    p_L = p[:-1]
    E_L = E[:-1]
    c_L = c[:-1]

    # Right states: cells 1 to n-1
    rho_R = rho[1:]
    u_R = u[1:]
    p_R = p[1:]
    E_R = E[1:]
    c_R = c[1:]

    return rho_L, u_L, p_L, E_L, c_L, rho_R, u_R, p_R, E_R, c_R

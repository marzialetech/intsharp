"""
Equation of State (EOS) for compressible flow.

Implements stiffened gas EOS which generalizes to ideal gas when p_infinity = 0.

Stiffened Gas EOS:
    p = (γ - 1) ρ e - γ p_∞

Where:
    p = pressure
    ρ = density
    e = specific internal energy
    γ = heat capacity ratio
    p_∞ = stiffness parameter (0 for ideal gas)

For water: γ ≈ 4.4, p_∞ ≈ 6×10^8 Pa
For air:   γ = 1.4,  p_∞ = 0
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def pressure_from_energy(
    rho: NDArray[np.float64] | float,
    e: NDArray[np.float64] | float,
    gamma: float,
    p_inf: float = 0.0,
) -> NDArray[np.float64] | float:
    """
    Compute pressure from density and specific internal energy.

    p = (γ - 1) ρ e - γ p_∞

    Parameters
    ----------
    rho : array or float
        Density.
    e : array or float
        Specific internal energy.
    gamma : float
        Heat capacity ratio.
    p_inf : float
        Stiffness parameter (default 0 for ideal gas).

    Returns
    -------
    array or float
        Pressure.
    """
    return (gamma - 1.0) * rho * e - gamma * p_inf


def internal_energy_from_pressure(
    rho: NDArray[np.float64] | float,
    p: NDArray[np.float64] | float,
    gamma: float,
    p_inf: float = 0.0,
) -> NDArray[np.float64] | float:
    """
    Compute specific internal energy from density and pressure.

    e = (p + γ p_∞) / ((γ - 1) ρ)

    Parameters
    ----------
    rho : array or float
        Density.
    p : array or float
        Pressure.
    gamma : float
        Heat capacity ratio.
    p_inf : float
        Stiffness parameter (default 0 for ideal gas).

    Returns
    -------
    array or float
        Specific internal energy.
    """
    return (p + gamma * p_inf) / ((gamma - 1.0) * rho)


def sound_speed(
    rho: NDArray[np.float64] | float,
    p: NDArray[np.float64] | float,
    gamma: float,
    p_inf: float = 0.0,
) -> NDArray[np.float64] | float:
    """
    Compute sound speed.

    c = sqrt(γ (p + p_∞) / ρ)

    Parameters
    ----------
    rho : array or float
        Density.
    p : array or float
        Pressure.
    gamma : float
        Heat capacity ratio.
    p_inf : float
        Stiffness parameter (default 0 for ideal gas).

    Returns
    -------
    array or float
        Sound speed.
    """
    return np.sqrt(gamma * (p + p_inf) / rho)


def total_energy_from_primitives(
    rho: NDArray[np.float64] | float,
    u: NDArray[np.float64] | float,
    p: NDArray[np.float64] | float,
    gamma: float,
    p_inf: float = 0.0,
    v: NDArray[np.float64] | float | None = None,
) -> NDArray[np.float64] | float:
    """
    Compute total energy per unit volume from primitive variables.

    E = ρ e + ½ ρ (u² + v²)
      = (p + γ p_∞) / (γ - 1) + ½ ρ (u² + v²)

    Parameters
    ----------
    rho : array or float
        Density.
    u : array or float
        Velocity in x.
    p : array or float
        Pressure.
    gamma : float
        Heat capacity ratio.
    p_inf : float
        Stiffness parameter.
    v : array or float or None
        Velocity in y (for 2D).

    Returns
    -------
    array or float
        Total energy per unit volume.
    """
    e = internal_energy_from_pressure(rho, p, gamma, p_inf)
    kinetic = 0.5 * rho * u * u
    if v is not None:
        kinetic = kinetic + 0.5 * rho * v * v
    return rho * e + kinetic


def pressure_from_total_energy(
    rho: NDArray[np.float64] | float,
    rho_u: NDArray[np.float64] | float,
    E: NDArray[np.float64] | float,
    gamma: float,
    p_inf: float = 0.0,
    rho_v: NDArray[np.float64] | float | None = None,
) -> NDArray[np.float64] | float:
    """
    Compute pressure from conserved variables.

    p = (γ - 1) [E - ½ ρ (u² + v²)] - γ p_∞

    Parameters
    ----------
    rho : array or float
        Density.
    rho_u : array or float
        Momentum in x (ρu).
    E : array or float
        Total energy per unit volume.
    gamma : float
        Heat capacity ratio.
    p_inf : float
        Stiffness parameter.
    rho_v : array or float or None
        Momentum in y (ρv) for 2D.

    Returns
    -------
    array or float
        Pressure.
    """
    u = rho_u / rho
    kinetic = 0.5 * rho * u * u
    if rho_v is not None:
        v = rho_v / rho
        kinetic = kinetic + 0.5 * rho * v * v
    rho_e = E - kinetic
    e = rho_e / rho
    return pressure_from_energy(rho, e, gamma, p_inf)


def enthalpy(
    rho: NDArray[np.float64] | float,
    p: NDArray[np.float64] | float,
    E: NDArray[np.float64] | float,
) -> NDArray[np.float64] | float:
    """
    Compute specific total enthalpy.

    H = (E + p) / ρ

    Parameters
    ----------
    rho : array or float
        Density.
    p : array or float
        Pressure.
    E : array or float
        Total energy per unit volume.

    Returns
    -------
    array or float
        Specific total enthalpy.
    """
    return (E + p) / rho


def primitives_from_conservatives(
    rho: NDArray[np.float64],
    rho_u: NDArray[np.float64],
    E: NDArray[np.float64],
    gamma: float,
    p_inf: float = 0.0,
    rho_v: NDArray[np.float64] | None = None,
) -> tuple:
    """
    Convert conserved variables to primitives.

    Parameters
    ----------
    rho : NDArray
        Density.
    rho_u : NDArray
        Momentum in x.
    E : NDArray
        Total energy.
    gamma : float
        Heat capacity ratio.
    p_inf : float
        Stiffness parameter.
    rho_v : NDArray or None
        Momentum in y (for 2D).

    Returns
    -------
    tuple
        (rho, u, p) for 1D or (rho, u, v, p) for 2D.
    """
    u = rho_u / rho
    p = pressure_from_total_energy(rho, rho_u, E, gamma, p_inf, rho_v)
    if rho_v is not None:
        v = rho_v / rho
        return rho, u, v, p
    return rho, u, p


def conservatives_from_primitives(
    rho: NDArray[np.float64],
    u: NDArray[np.float64],
    p: NDArray[np.float64],
    gamma: float,
    p_inf: float = 0.0,
    v: NDArray[np.float64] | None = None,
) -> tuple:
    """
    Convert primitive variables to conservatives.

    Parameters
    ----------
    rho : NDArray
        Density.
    u : NDArray
        Velocity in x.
    p : NDArray
        Pressure.
    gamma : float
        Heat capacity ratio.
    p_inf : float
        Stiffness parameter.
    v : NDArray or None
        Velocity in y (for 2D).

    Returns
    -------
    tuple
        (rho, rho_u, E) for 1D or (rho, rho_u, rho_v, E) for 2D.
    """
    rho_u = rho * u
    E = total_energy_from_primitives(rho, u, p, gamma, p_inf, v)
    if v is not None:
        rho_v = rho * v
        return rho, rho_u, rho_v, E
    return rho, rho_u, E


# ---------------------------------------------------------------------------
# Two-Phase Mixture EOS (Pressure Equilibrium Model)
# ---------------------------------------------------------------------------


def mixture_density(
    alpha: NDArray[np.float64],
    rho1: NDArray[np.float64] | float,
    rho2: NDArray[np.float64] | float,
) -> NDArray[np.float64]:
    """
    Compute mixture density from volume fraction.

    ρ_mix = α ρ₁ + (1 - α) ρ₂

    Parameters
    ----------
    alpha : NDArray
        Volume fraction of phase 1 (0 ≤ α ≤ 1).
    rho1 : NDArray or float
        Density of phase 1.
    rho2 : NDArray or float
        Density of phase 2.

    Returns
    -------
    NDArray
        Mixture density.
    """
    return alpha * rho1 + (1.0 - alpha) * rho2


def mixture_gamma_effective(
    alpha: NDArray[np.float64],
    gamma1: float,
    gamma2: float,
    p_inf1: float = 0.0,
    p_inf2: float = 0.0,
    p: NDArray[np.float64] | float = 1e5,
) -> NDArray[np.float64]:
    """
    Compute effective gamma for mixture (volume-weighted).

    This is an approximation for numerical convenience.
    For more accurate results, use the full mixture EOS.

    γ_eff ≈ α γ₁ + (1 - α) γ₂

    Returns
    -------
    NDArray
        Effective gamma.
    """
    return alpha * gamma1 + (1.0 - alpha) * gamma2


def mixture_p_infinity_effective(
    alpha: NDArray[np.float64],
    p_inf1: float,
    p_inf2: float,
) -> NDArray[np.float64]:
    """
    Compute effective p_infinity for mixture (volume-weighted).

    p_∞_eff = α p_∞₁ + (1 - α) p_∞₂

    Returns
    -------
    NDArray
        Effective p_infinity.
    """
    return alpha * p_inf1 + (1.0 - alpha) * p_inf2


def mixture_sound_speed_wood(
    rho_mix: NDArray[np.float64],
    alpha: NDArray[np.float64],
    rho1: NDArray[np.float64] | float,
    rho2: NDArray[np.float64] | float,
    c1: NDArray[np.float64] | float,
    c2: NDArray[np.float64] | float,
) -> NDArray[np.float64]:
    """
    Compute mixture sound speed using Wood's formula.

    1/(ρ_mix c_mix²) = α/(ρ₁ c₁²) + (1-α)/(ρ₂ c₂²)

    This formula comes from assuming pressure equilibrium and
    isentropic compression of both phases.

    Parameters
    ----------
    rho_mix : NDArray
        Mixture density.
    alpha : NDArray
        Volume fraction of phase 1.
    rho1, rho2 : NDArray or float
        Phase densities.
    c1, c2 : NDArray or float
        Phase sound speeds.

    Returns
    -------
    NDArray
        Mixture sound speed.
    """
    # Avoid division by zero; handle pure phases (alpha=0 or 1)
    rho1_safe = np.maximum(np.asarray(rho1, dtype=np.float64), 1e-30)
    rho2_safe = np.maximum(np.asarray(rho2, dtype=np.float64), 1e-30)
    c1_sq = np.maximum(np.asarray(c1, dtype=np.float64) ** 2, 1e-30)
    c2_sq = np.maximum(np.asarray(c2, dtype=np.float64) ** 2, 1e-30)

    # Pure phase 1: c_mix = c1; pure phase 2: c_mix = c2
    alpha_arr = np.asarray(alpha, dtype=np.float64)
    inv_rho_c2 = alpha_arr / (rho1_safe * c1_sq) + (1.0 - alpha_arr) / (rho2_safe * c2_sq)
    c_mix_sq = 1.0 / (np.asarray(rho_mix, dtype=np.float64) * inv_rho_c2 + 1e-30)
    c_mix = np.sqrt(np.maximum(c_mix_sq, 1e-30))

    # Override for pure phases to avoid 0/0 or inf
    c1_val = np.sqrt(c1_sq)
    c2_val = np.sqrt(c2_sq)
    c_mix = np.where(alpha_arr >= 1.0 - 1e-10, c1_val, c_mix)
    c_mix = np.where(alpha_arr <= 1e-10, c2_val, c_mix)
    return c_mix


def phase_densities_from_pressure(
    p: NDArray[np.float64],
    e1: NDArray[np.float64] | float,
    e2: NDArray[np.float64] | float,
    gamma1: float,
    gamma2: float,
    p_inf1: float = 0.0,
    p_inf2: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute phase densities from pressure and specific internal energies.

    From p = (γ-1)ρe - γp_∞, we get:
    ρ = (p + γp_∞) / ((γ-1)e)

    Parameters
    ----------
    p : NDArray
        Pressure (same for both phases in equilibrium).
    e1, e2 : NDArray or float
        Specific internal energies of phases.
    gamma1, gamma2 : float
        Heat capacity ratios.
    p_inf1, p_inf2 : float
        Stiffness parameters.

    Returns
    -------
    tuple[NDArray, NDArray]
        (rho1, rho2) phase densities.
    """
    rho1 = (p + gamma1 * p_inf1) / ((gamma1 - 1.0) * e1 + 1e-30)
    rho2 = (p + gamma2 * p_inf2) / ((gamma2 - 1.0) * e2 + 1e-30)
    return rho1, rho2


def mixture_total_energy(
    rho_mix: NDArray[np.float64],
    u: NDArray[np.float64],
    p: NDArray[np.float64],
    alpha: NDArray[np.float64],
    gamma1: float,
    gamma2: float,
    p_inf1: float = 0.0,
    p_inf2: float = 0.0,
) -> NDArray[np.float64]:
    """
    Compute mixture total energy from primitives (pressure equilibrium).

    E = α ρ₁ e₁ + (1-α) ρ₂ e₂ + ½ ρ_mix u²

    For stiffened gas: ρe = (p + γp_∞)/(γ-1)

    So: E = α(p + γ₁p_∞₁)/(γ₁-1) + (1-α)(p + γ₂p_∞₂)/(γ₂-1) + ½ρ_mix u²

    Parameters
    ----------
    rho_mix : NDArray
        Mixture density.
    u : NDArray
        Velocity.
    p : NDArray
        Pressure.
    alpha : NDArray
        Volume fraction of phase 1.
    gamma1, gamma2 : float
        Heat capacity ratios.
    p_inf1, p_inf2 : float
        Stiffness parameters.

    Returns
    -------
    NDArray
        Mixture total energy per unit volume.
    """
    # Internal energy contributions: ρe = (p + γp_∞)/(γ-1)
    rho_e_1 = (p + gamma1 * p_inf1) / (gamma1 - 1.0)
    rho_e_2 = (p + gamma2 * p_inf2) / (gamma2 - 1.0)

    # Mixture internal energy
    rho_e_mix = alpha * rho_e_1 + (1.0 - alpha) * rho_e_2

    # Add kinetic energy
    E = rho_e_mix + 0.5 * rho_mix * u * u

    return E


def mixture_pressure_from_conservatives(
    rho_mix: NDArray[np.float64],
    rho_u: NDArray[np.float64],
    E: NDArray[np.float64],
    alpha: NDArray[np.float64],
    gamma1: float,
    gamma2: float,
    p_inf1: float = 0.0,
    p_inf2: float = 0.0,
) -> NDArray[np.float64]:
    """
    Compute pressure from mixture conserved variables (pressure equilibrium).

    Inverts the mixture energy equation:
    E = α(p + γ₁p_∞₁)/(γ₁-1) + (1-α)(p + γ₂p_∞₂)/(γ₂-1) + ½ρ_mix u²

    Solving for p:
    E - ½ρu² = α(p + γ₁p_∞₁)/(γ₁-1) + (1-α)(p + γ₂p_∞₂)/(γ₂-1)
             = p [α/(γ₁-1) + (1-α)/(γ₂-1)] + [αγ₁p_∞₁/(γ₁-1) + (1-α)γ₂p_∞₂/(γ₂-1)]

    Parameters
    ----------
    rho_mix : NDArray
        Mixture density.
    rho_u : NDArray
        Momentum.
    E : NDArray
        Total energy.
    alpha : NDArray
        Volume fraction of phase 1.
    gamma1, gamma2 : float
        Heat capacity ratios.
    p_inf1, p_inf2 : float
        Stiffness parameters.

    Returns
    -------
    NDArray
        Pressure.
    """
    u = rho_u / (rho_mix + 1e-30)
    kinetic = 0.5 * rho_mix * u * u

    # Internal energy
    rho_e_mix = E - kinetic

    # Coefficients for p
    # rho_e_mix = p * A + B
    # where A = α/(γ₁-1) + (1-α)/(γ₂-1)
    # and B = αγ₁p_∞₁/(γ₁-1) + (1-α)γ₂p_∞₂/(γ₂-1)
    A = alpha / (gamma1 - 1.0) + (1.0 - alpha) / (gamma2 - 1.0)
    B = alpha * gamma1 * p_inf1 / (gamma1 - 1.0) + (1.0 - alpha) * gamma2 * p_inf2 / (gamma2 - 1.0)

    p = (rho_e_mix - B) / (A + 1e-30)

    return np.maximum(p, 1e-10)  # Ensure positive pressure

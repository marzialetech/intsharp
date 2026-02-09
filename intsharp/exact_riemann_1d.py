"""
Compact exact 1D Riemann solution utilities (ideal-gas Euler).

Currently includes Sod-type exact solution for a single ideal gas.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _prefun(
    p: float,
    rho_k: float,
    p_k: float,
    a_k: float,
    gamma: float,
) -> tuple[float, float]:
    """Toro pressure function f_k(p) and derivative for side k."""
    if p > p_k:
        # Shock
        A = 2.0 / ((gamma + 1.0) * rho_k)
        B = (gamma - 1.0) / (gamma + 1.0) * p_k
        sqrt_term = np.sqrt(A / (p + B))
        f = (p - p_k) * sqrt_term
        df = sqrt_term * (1.0 - 0.5 * (p - p_k) / (p + B))
    else:
        # Rarefaction
        pr = p / p_k
        expo = (gamma - 1.0) / (2.0 * gamma)
        f = 2.0 * a_k / (gamma - 1.0) * (pr ** expo - 1.0)
        df = (1.0 / (rho_k * a_k)) * pr ** (-(gamma + 1.0) / (2.0 * gamma))
    return float(f), float(df)


def _solve_star_region(
    rho_L: float,
    u_L: float,
    p_L: float,
    rho_R: float,
    u_R: float,
    p_R: float,
    gamma: float,
) -> tuple[float, float]:
    """Solve for p* and u* using Newton iterations."""
    a_L = np.sqrt(gamma * p_L / rho_L)
    a_R = np.sqrt(gamma * p_R / rho_R)

    # PVRS initial guess (Toro)
    p_pv = 0.5 * (p_L + p_R) - 0.125 * (u_R - u_L) * (rho_L + rho_R) * (a_L + a_R)
    p_old = max(1e-12, p_pv)

    for _ in range(60):
        fL, dfL = _prefun(p_old, rho_L, p_L, a_L, gamma)
        fR, dfR = _prefun(p_old, rho_R, p_R, a_R, gamma)
        num = fL + fR + (u_R - u_L)
        den = dfL + dfR
        p_new = p_old - num / (den + 1e-30)
        p_new = max(1e-12, p_new)
        if 2.0 * abs(p_new - p_old) / (p_new + p_old + 1e-30) < 1e-10:
            p_old = p_new
            break
        p_old = p_new

    p_star = p_old
    fL, _ = _prefun(p_star, rho_L, p_L, a_L, gamma)
    fR, _ = _prefun(p_star, rho_R, p_R, a_R, gamma)
    u_star = 0.5 * (u_L + u_R + fR - fL)
    return float(p_star), float(u_star)


def exact_sod_1d(
    x: NDArray[np.float64],
    t: float,
    x_discontinuity: float,
    rho_L: float,
    u_L: float,
    p_L: float,
    rho_R: float,
    u_R: float,
    p_R: float,
    gamma: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Exact Sod/Riemann solution for ideal-gas Euler at positions x and time t.

    Assumes a single ideal gas (p_infinity = 0).
    """
    x = np.asarray(x, dtype=np.float64)
    if t <= 0.0:
        rho = np.where(x < x_discontinuity, rho_L, rho_R)
        u = np.where(x < x_discontinuity, u_L, u_R)
        p = np.where(x < x_discontinuity, p_L, p_R)
        return rho, u, p

    a_L = np.sqrt(gamma * p_L / rho_L)
    a_R = np.sqrt(gamma * p_R / rho_R)
    p_star, u_star = _solve_star_region(rho_L, u_L, p_L, rho_R, u_R, p_R, gamma)

    S = (x - x_discontinuity) / t
    rho = np.empty_like(x)
    u = np.empty_like(x)
    p = np.empty_like(x)

    for i, s in enumerate(S):
        if s <= u_star:
            # Left side
            if p_star <= p_L:
                # Left rarefaction
                s_hl = u_L - a_L
                a_star_l = a_L * (p_star / p_L) ** ((gamma - 1.0) / (2.0 * gamma))
                s_tl = u_star - a_star_l
                if s <= s_hl:
                    rho[i], u[i], p[i] = rho_L, u_L, p_L
                elif s > s_tl:
                    rho[i] = rho_L * (p_star / p_L) ** (1.0 / gamma)
                    u[i] = u_star
                    p[i] = p_star
                else:
                    u_i = 2.0 / (gamma + 1.0) * (a_L + 0.5 * (gamma - 1.0) * u_L + s)
                    a_i = 2.0 / (gamma + 1.0) * (a_L + 0.5 * (gamma - 1.0) * (u_L - s))
                    rho_i = rho_L * (a_i / a_L) ** (2.0 / (gamma - 1.0))
                    p_i = p_L * (a_i / a_L) ** (2.0 * gamma / (gamma - 1.0))
                    rho[i], u[i], p[i] = rho_i, u_i, p_i
            else:
                # Left shock
                s_l = u_L - a_L * np.sqrt((gamma + 1.0) / (2.0 * gamma) * (p_star / p_L) + (gamma - 1.0) / (2.0 * gamma))
                if s <= s_l:
                    rho[i], u[i], p[i] = rho_L, u_L, p_L
                else:
                    num = p_star / p_L + (gamma - 1.0) / (gamma + 1.0)
                    den = (gamma - 1.0) / (gamma + 1.0) * (p_star / p_L) + 1.0
                    rho[i] = rho_L * (num / den)
                    u[i] = u_star
                    p[i] = p_star
        else:
            # Right side
            if p_star <= p_R:
                # Right rarefaction
                s_hr = u_R + a_R
                a_star_r = a_R * (p_star / p_R) ** ((gamma - 1.0) / (2.0 * gamma))
                s_tr = u_star + a_star_r
                if s >= s_hr:
                    rho[i], u[i], p[i] = rho_R, u_R, p_R
                elif s <= s_tr:
                    rho[i] = rho_R * (p_star / p_R) ** (1.0 / gamma)
                    u[i] = u_star
                    p[i] = p_star
                else:
                    u_i = 2.0 / (gamma + 1.0) * (-a_R + 0.5 * (gamma - 1.0) * u_R + s)
                    a_i = 2.0 / (gamma + 1.0) * (a_R - 0.5 * (gamma - 1.0) * (u_R - s))
                    rho_i = rho_R * (a_i / a_R) ** (2.0 / (gamma - 1.0))
                    p_i = p_R * (a_i / a_R) ** (2.0 * gamma / (gamma - 1.0))
                    rho[i], u[i], p[i] = rho_i, u_i, p_i
            else:
                # Right shock
                s_r = u_R + a_R * np.sqrt((gamma + 1.0) / (2.0 * gamma) * (p_star / p_R) + (gamma - 1.0) / (2.0 * gamma))
                if s >= s_r:
                    rho[i], u[i], p[i] = rho_R, u_R, p_R
                else:
                    num = p_star / p_R + (gamma - 1.0) / (gamma + 1.0)
                    den = (gamma - 1.0) / (gamma + 1.0) * (p_star / p_R) + 1.0
                    rho[i] = rho_R * (num / den)
                    u[i] = u_star
                    p[i] = p_star

    return rho, u, p

"""
HLLC flux scheme for compressible Euler equations (1D).

HLLC (Harten-Lax-van Leer-Contact) restores the contact wave missing in HLL
and computes upwinded star states across three waves: S_L, S_*, S_R.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _safe_denom(x: NDArray[np.float64], eps: float = 1e-30) -> NDArray[np.float64]:
    """Avoid division by zero while preserving sign."""
    return np.where(np.abs(x) < eps, np.where(x >= 0.0, eps, -eps), x)


def hllc_flux_1d(
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
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute HLLC interface fluxes for 1D Euler equations.

    Returns (F_rho, F_rho_u, F_E) at each interface.
    """
    # Physical fluxes from left/right states
    F_rho_L = rho_L * u_L
    F_rho_u_L = rho_L * u_L * u_L + p_L
    F_E_L = (E_L + p_L) * u_L

    F_rho_R = rho_R * u_R
    F_rho_u_R = rho_R * u_R * u_R + p_R
    F_E_R = (E_R + p_R) * u_R

    # Wave-speed estimates (Davis bounds)
    S_L = np.minimum(u_L - c_L, u_R - c_R)
    S_R = np.maximum(u_L + c_L, u_R + c_R)

    # Contact wave speed
    sm_num = (
        p_R - p_L
        + rho_L * u_L * (S_L - u_L)
        - rho_R * u_R * (S_R - u_R)
    )
    sm_den = _safe_denom(rho_L * (S_L - u_L) - rho_R * (S_R - u_R))
    S_M = sm_num / sm_den

    # Star pressure (from left relation)
    p_star = p_L + rho_L * (S_L - u_L) * (S_M - u_L)

    # Star states U*_L and U*_R
    den_L = _safe_denom(S_L - S_M)
    den_R = _safe_denom(S_R - S_M)

    rho_star_L = rho_L * (S_L - u_L) / den_L
    rho_star_R = rho_R * (S_R - u_R) / den_R

    mom_star_L = rho_star_L * S_M
    mom_star_R = rho_star_R * S_M

    E_star_L = ((S_L - u_L) * E_L - p_L * u_L + p_star * S_M) / den_L
    E_star_R = ((S_R - u_R) * E_R - p_R * u_R + p_star * S_M) / den_R

    # HLLC flux piecewise selection
    F_rho = np.where(
        0.0 <= S_L,
        F_rho_L,
        np.where(
            0.0 <= S_M,
            F_rho_L + S_L * (rho_star_L - rho_L),
            np.where(
                0.0 <= S_R,
                F_rho_R + S_R * (rho_star_R - rho_R),
                F_rho_R,
            ),
        ),
    )

    F_rho_u = np.where(
        0.0 <= S_L,
        F_rho_u_L,
        np.where(
            0.0 <= S_M,
            F_rho_u_L + S_L * (mom_star_L - rho_L * u_L),
            np.where(
                0.0 <= S_R,
                F_rho_u_R + S_R * (mom_star_R - rho_R * u_R),
                F_rho_u_R,
            ),
        ),
    )

    F_E = np.where(
        0.0 <= S_L,
        F_E_L,
        np.where(
            0.0 <= S_M,
            F_E_L + S_L * (E_star_L - E_L),
            np.where(
                0.0 <= S_R,
                F_E_R + S_R * (E_star_R - E_R),
                F_E_R,
            ),
        ),
    )

    return F_rho, F_rho_u, F_E


def hllc_flux_1d_with_v_riem(
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
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    HLLC flux with interface transport velocity v_riem for two-phase splitting.

    v_riem is computed from the mass flux and upwind density so that
    F_{alpha1*rho1} + F_{alpha2*rho2} = F_rho when upwind partial densities are used.
    """
    F_rho, F_rho_u, F_E = hllc_flux_1d(
        rho_L, u_L, p_L, E_L, c_L,
        rho_R, u_R, p_R, E_R, c_R,
    )
    rho_upwind = np.where(F_rho >= 0.0, rho_L, rho_R)
    v_riem = F_rho / (rho_upwind + 1e-30)
    return F_rho, F_rho_u, F_E, v_riem

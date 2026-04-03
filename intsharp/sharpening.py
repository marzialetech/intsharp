"""
Interface sharpening source terms.

Implements:
- PM (Parameswaran-Mandal)
- CL (Chiu-Lin, isotropic diffusion form)
- Olsson-Kreiss (CLS [2] 2007, anisotropic diffusion)
- ACLS (Desjardins et al. [5] 2008, algebraic phi_inv normal)
- CLS 2010 ([3], non-conservative with mapped normal)
- LCLS 2012 ([7], localized with beta = 4 psi(1-psi))
- LCLS 2014 ([4], variable pseudo-time localization)
- CLS 2015 ([10], mapping function phi_map)
- CLS 2017 ([11], inverse transform with cosh)
- SCLS (Chiodi-Desjardins [8] 2018, self-correcting)
- VCAC (Volume-Conserving Allen-Cahn, AI-proposed)
- GLCLS (Gradient-Localized CLS, AI-proposed)

Applied post-step via operator splitting.
Supports both 1D and 2D.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from .boundary import BoundaryCondition
from .registry import register_sharpening


# ---------------------------------------------------------------------------
# 1D Gradient/Divergence Helpers
# ---------------------------------------------------------------------------

def _grad_periodic(f: NDArray[np.float64], dx: float) -> NDArray[np.float64]:
    """Central difference gradient with periodic BC (1D)."""
    return (np.roll(f, -1) - np.roll(f, 1)) / (2 * dx)


def _div_periodic(q: NDArray[np.float64], dx: float) -> NDArray[np.float64]:
    """Central difference divergence with periodic BC (1D)."""
    return (np.roll(q, -1) - np.roll(q, 1)) / (2 * dx)


def _grad_nonperiodic(
    f: NDArray[np.float64],
    dx: float,
    bc: BoundaryCondition,
) -> NDArray[np.float64]:
    """Central difference gradient with non-periodic BC (1D)."""
    grad = np.zeros_like(f)
    # Interior: central difference
    grad[1:-1] = (f[2:] - f[:-2]) / (2 * dx)
    # Boundaries: one-sided
    grad[0] = (f[1] - f[0]) / dx
    grad[-1] = (f[-1] - f[-2]) / dx
    return grad


# ---------------------------------------------------------------------------
# 2D Gradient/Divergence Helpers
# ---------------------------------------------------------------------------

def _grad_periodic_2d(
    f: NDArray[np.float64],
    dx: float,
    dy: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Central difference gradient with periodic BC (2D).
    
    Returns (df/dx, df/dy).
    """
    # df/dx: roll along axis=1 (x-direction)
    dfdx = (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2 * dx)
    # df/dy: roll along axis=0 (y-direction)
    dfdy = (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2 * dy)
    return dfdx, dfdy


def _div_periodic_2d(
    qx: NDArray[np.float64],
    qy: NDArray[np.float64],
    dx: float,
    dy: float,
) -> NDArray[np.float64]:
    """
    Central difference divergence with periodic BC (2D).
    
    div(q) = dqx/dx + dqy/dy
    """
    dqx_dx = (np.roll(qx, -1, axis=1) - np.roll(qx, 1, axis=1)) / (2 * dx)
    dqy_dy = (np.roll(qy, -1, axis=0) - np.roll(qy, 1, axis=0)) / (2 * dy)
    return dqx_dx + dqy_dy


def _grad_nonperiodic_2d(
    f: NDArray[np.float64],
    dx: float,
    dy: float,
    bc: BoundaryCondition,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Central difference gradient with non-periodic BC (2D).
    
    Returns (df/dx, df/dy).
    """
    ny, nx = f.shape
    dfdx = np.zeros_like(f)
    dfdy = np.zeros_like(f)
    
    # df/dx: interior central, boundaries one-sided
    dfdx[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2 * dx)
    dfdx[:, 0] = (f[:, 1] - f[:, 0]) / dx
    dfdx[:, -1] = (f[:, -1] - f[:, -2]) / dx
    
    # df/dy: interior central, boundaries one-sided
    dfdy[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2 * dy)
    dfdy[0, :] = (f[1, :] - f[0, :]) / dy
    dfdy[-1, :] = (f[-1, :] - f[-2, :]) / dy
    
    return dfdx, dfdy


def _div_nonperiodic_2d(
    qx: NDArray[np.float64],
    qy: NDArray[np.float64],
    dx: float,
    dy: float,
    bc: BoundaryCondition,
) -> NDArray[np.float64]:
    """
    Central difference divergence with non-periodic BC (2D).
    
    div(q) = dqx/dx + dqy/dy
    """
    ny, nx = qx.shape
    dqx_dx = np.zeros_like(qx)
    dqy_dy = np.zeros_like(qy)
    
    # dqx/dx
    dqx_dx[:, 1:-1] = (qx[:, 2:] - qx[:, :-2]) / (2 * dx)
    dqx_dx[:, 0] = (qx[:, 1] - qx[:, 0]) / dx
    dqx_dx[:, -1] = (qx[:, -1] - qx[:, -2]) / dx
    
    # dqy/dy
    dqy_dy[1:-1, :] = (qy[2:, :] - qy[:-2, :]) / (2 * dy)
    dqy_dy[0, :] = (qy[1, :] - qy[0, :]) / dy
    dqy_dy[-1, :] = (qy[-1, :] - qy[-2, :]) / dy
    
    return dqx_dx + dqy_dy


@register_sharpening("pm")
def pm_sharpening(
    psi: NDArray[np.float64],
    dx: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    Parameswaran-Mandal sharpening.

    RHS = -K * psi * (1-psi) * (1-2*psi) + eps * (1-2*psi) * |grad(psi)|

    where K = 1 / (4 * eps^2).

    Parameters
    ----------
    psi : NDArray
        Volume fraction field.
    dx : float
        Grid spacing.
    dt : float
        Time step for sharpening sub-step.
    eps_target : float
        Target interface thickness (epsilon).
    strength : float
        Sharpening strength multiplier (Gamma).
    bc : BoundaryCondition
        Boundary condition.

    Returns
    -------
    NDArray
        Updated psi.
    """
    # Compute gradient
    if bc.bc_type == "periodic":
        grad_psi = _grad_periodic(psi, dx)
    else:
        grad_psi = _grad_nonperiodic(psi, dx, bc)

    abs_grad = np.abs(grad_psi)

    # PM sharpening coefficient
    K = 1.0 / (4.0 * eps_target**2)

    # RHS
    rhs = (
        -K * psi * (1 - psi) * (1 - 2*psi)
        + eps_target * (1 - 2*psi) * abs_grad
    )

    # Update
    psi_new = psi + dt * strength * rhs

    # Clip to [0, 1]
    return np.clip(psi_new, 0.0, 1.0)


@register_sharpening("cl")
def cl_sharpening(
    psi: NDArray[np.float64],
    dx: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    Chiu-Lin sharpening.

    RHS = div(eps * grad(psi) - psi * (1-psi) * n_hat)

    where n_hat = grad(psi) / |grad(psi)|

    The pseudo-timestep is set by the diffusion CFL of the eps*Laplacian
    term: theta = CFL_SAFETY * dx^2 / eps.  The physical dt is unused;
    `strength` (Gamma) is a dimensionless multiplier on theta.

    Parameters
    ----------
    psi : NDArray
        Volume fraction field.
    dx : float
        Grid spacing.
    dt : float
        Physical time step (unused; kept for API compatibility).
    eps_target : float
        Target interface thickness.
    strength : float
        Dimensionless sharpening strength (Gamma).
    bc : BoundaryCondition
        Boundary condition.

    Returns
    -------
    NDArray
        Updated psi.
    """
    theta = _CFL_DIFF * dx * dx / eps_target

    if bc.bc_type == "periodic":
        grad_psi = _grad_periodic(psi, dx)
    else:
        grad_psi = _grad_nonperiodic(psi, dx, bc)

    n_hat = grad_psi / (np.abs(grad_psi) + _ETA)

    if bc.bc_type == "periodic":
        rhs = _div_split_rusanov_periodic(psi, n_hat, eps_target, dx)
    else:
        rhs = _div_split_rusanov_nonperiodic(psi, n_hat, eps_target, dx, bc)

    return np.clip(psi + strength * theta * rhs, 0.0, 1.0)


# Calibrated PM constant from dissertation (200 * 1.97715965626)
C_PM_CALIBRATED = 395.43193125


@register_sharpening("pm_cal")
def pm_sharpening_calibrated(
    psi: NDArray[np.float64],
    dx: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    Parameswaran-Mandal sharpening with calibrated constant (1D).

    RHS = -C_PM * psi * (1-psi) * (1-2*psi) + eps * (1-2*psi) * |grad(psi)|

    where C_PM ≈ 395.4 is a calibrated constant (independent of eps).

    Parameters
    ----------
    psi : NDArray
        Volume fraction field.
    dx : float
        Grid spacing.
    dt : float
        Time step for sharpening sub-step.
    eps_target : float
        Target interface thickness (epsilon).
    strength : float
        Sharpening strength multiplier (Gamma).
    bc : BoundaryCondition
        Boundary condition.

    Returns
    -------
    NDArray
        Updated psi.
    """
    # Compute gradient
    if bc.bc_type == "periodic":
        grad_psi = _grad_periodic(psi, dx)
    else:
        grad_psi = _grad_nonperiodic(psi, dx, bc)

    abs_grad = np.abs(grad_psi)

    # RHS with calibrated constant
    rhs = (
        -C_PM_CALIBRATED * psi * (1 - psi) * (1 - 2*psi)
        + eps_target * (1 - 2*psi) * abs_grad
    )

    # Update
    psi_new = psi + dt * strength * rhs

    # Clip to [0, 1]
    return np.clip(psi_new, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 2D Sharpening Methods
# ---------------------------------------------------------------------------


@register_sharpening("pm_2d")
def pm_sharpening_2d(
    psi: NDArray[np.float64],
    dx: float,
    dy: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    Parameswaran-Mandal sharpening (2D).

    RHS = -K * psi * (1-psi) * (1-2*psi) + eps * (1-2*psi) * |grad(psi)|

    where K = 1 / (4 * eps^2) and |grad| = sqrt((dpsi/dx)^2 + (dpsi/dy)^2).

    Parameters
    ----------
    psi : NDArray
        Volume fraction field (shape: ny, nx).
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.
    dt : float
        Time step for sharpening sub-step.
    eps_target : float
        Target interface thickness (epsilon).
    strength : float
        Sharpening strength multiplier (Gamma).
    bc : BoundaryCondition
        Boundary condition.

    Returns
    -------
    NDArray
        Updated psi.
    """
    eta = 1e-12  # Small number to avoid division by zero

    # Compute gradient
    if bc.bc_type == "periodic":
        dfdx, dfdy = _grad_periodic_2d(psi, dx, dy)
    else:
        dfdx, dfdy = _grad_nonperiodic_2d(psi, dx, dy, bc)

    # Gradient magnitude
    abs_grad = np.sqrt(dfdx**2 + dfdy**2) + eta

    # PM sharpening coefficient
    K = 1.0 / (4.0 * eps_target**2)

    # RHS
    rhs = (
        -K * psi * (1 - psi) * (1 - 2*psi)
        + eps_target * (1 - 2*psi) * abs_grad
    )

    # Update
    psi_new = psi + dt * strength * rhs

    # Clip to [0, 1]
    return np.clip(psi_new, 0.0, 1.0)


@register_sharpening("pm_cal_2d")
def pm_sharpening_calibrated_2d(
    psi: NDArray[np.float64],
    dx: float,
    dy: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    Parameswaran-Mandal sharpening with calibrated constant (2D).

    RHS = -C_PM * psi * (1-psi) * (1-2*psi) + eps * (1-2*psi) * |grad(psi)|

    where C_PM ≈ 395.4 is a calibrated constant (independent of eps).
    |grad| = sqrt((dpsi/dx)^2 + (dpsi/dy)^2).

    Parameters
    ----------
    psi : NDArray
        Volume fraction field (shape: ny, nx).
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.
    dt : float
        Time step for sharpening sub-step.
    eps_target : float
        Target interface thickness (epsilon).
    strength : float
        Sharpening strength multiplier (Gamma).
    bc : BoundaryCondition
        Boundary condition.

    Returns
    -------
    NDArray
        Updated psi.
    """
    eta = 1e-12  # Small number to avoid division by zero

    # Compute gradient
    if bc.bc_type == "periodic":
        dfdx, dfdy = _grad_periodic_2d(psi, dx, dy)
    else:
        dfdx, dfdy = _grad_nonperiodic_2d(psi, dx, dy, bc)

    # Gradient magnitude
    abs_grad = np.sqrt(dfdx**2 + dfdy**2) + eta

    # RHS with calibrated constant
    rhs = (
        -C_PM_CALIBRATED * psi * (1 - psi) * (1 - 2*psi)
        + eps_target * (1 - 2*psi) * abs_grad
    )

    # Update
    psi_new = psi + dt * strength * rhs

    # Clip to [0, 1]
    return np.clip(psi_new, 0.0, 1.0)


@register_sharpening("cl_2d")
def cl_sharpening_2d(
    psi: NDArray[np.float64],
    dx: float,
    dy: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    Chiu-Lin sharpening (2D).

    RHS = div(eps * grad(psi) - psi * (1-psi) * n_hat)

    Uses Rusanov face-flux for compressive term, central diff for diffusion.
    """
    if bc.bc_type == "periodic":
        dfdx, dfdy = _grad_periodic_2d(psi, dx, dy)
    else:
        dfdx, dfdy = _grad_nonperiodic_2d(psi, dx, dy, bc)

    abs_grad = np.sqrt(dfdx**2 + dfdy**2) + _ETA
    nx = dfdx / abs_grad
    ny = dfdy / abs_grad

    if bc.bc_type == "periodic":
        rhs = _div_split_rusanov_periodic_2d(psi, nx, ny, eps_target, dx, dy)
    else:
        rhs = _div_split_rusanov_nonperiodic_2d(psi, nx, ny, eps_target, dx, dy, bc)

    return np.clip(psi + dt * strength * rhs, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Shared Utilities for Literature Methods
# ---------------------------------------------------------------------------

_ETA = 1e-12
_CFL_DIFF = 0.2   # diffusion CFL safety factor for CLS-family pseudo-timestep
_CFL_ADV = 0.2    # advection CFL safety factor for PM-family pseudo-timestep


def _phi_inv(psi: NDArray[np.float64], eps: float) -> NDArray[np.float64]:
    """Algebraic signed-distance inversion: phi = eps * ln(psi / (1-psi))."""
    psi_c = np.clip(psi, _ETA, 1.0 - _ETA)
    return eps * np.log(psi_c / (1.0 - psi_c))


def _laplacian_periodic(f: NDArray[np.float64], dx: float) -> NDArray[np.float64]:
    """Second-order Laplacian with periodic BC (1D)."""
    return (np.roll(f, -1) - 2.0 * f + np.roll(f, 1)) / (dx * dx)


def _laplacian_nonperiodic(
    f: NDArray[np.float64],
    dx: float,
    bc: BoundaryCondition,
) -> NDArray[np.float64]:
    """Second-order Laplacian with non-periodic BC (1D)."""
    lap = np.zeros_like(f)
    lap[1:-1] = (f[2:] - 2.0 * f[1:-1] + f[:-2]) / (dx * dx)
    lap[0] = (f[1] - 2.0 * f[0] + f[1]) / (dx * dx)
    lap[-1] = (f[-2] - 2.0 * f[-1] + f[-2]) / (dx * dx)
    return lap


def _laplacian_periodic_2d(
    f: NDArray[np.float64],
    dx: float,
    dy: float,
) -> NDArray[np.float64]:
    """Second-order Laplacian with periodic BC (2D)."""
    d2f_dx2 = (np.roll(f, -1, axis=1) - 2.0 * f + np.roll(f, 1, axis=1)) / (dx * dx)
    d2f_dy2 = (np.roll(f, -1, axis=0) - 2.0 * f + np.roll(f, 1, axis=0)) / (dy * dy)
    return d2f_dx2 + d2f_dy2


def _laplacian_nonperiodic_2d(
    f: NDArray[np.float64],
    dx: float,
    dy: float,
    bc: BoundaryCondition,
) -> NDArray[np.float64]:
    """Second-order Laplacian with non-periodic BC (2D)."""
    lap = np.zeros_like(f)
    # x-direction
    lap[:, 1:-1] += (f[:, 2:] - 2.0 * f[:, 1:-1] + f[:, :-2]) / (dx * dx)
    lap[:, 0] += (f[:, 1] - 2.0 * f[:, 0] + f[:, 1]) / (dx * dx)
    lap[:, -1] += (f[:, -2] - 2.0 * f[:, -1] + f[:, -2]) / (dx * dx)
    # y-direction
    lap[1:-1, :] += (f[2:, :] - 2.0 * f[1:-1, :] + f[:-2, :]) / (dy * dy)
    lap[0, :] += (f[1, :] - 2.0 * f[0, :] + f[1, :]) / (dy * dy)
    lap[-1, :] += (f[-2, :] - 2.0 * f[-1, :] + f[-2, :]) / (dy * dy)
    return lap


def _div_nonperiodic(
    q: NDArray[np.float64],
    dx: float,
    bc: BoundaryCondition,
) -> NDArray[np.float64]:
    """Central difference divergence with non-periodic BC (1D)."""
    d = np.zeros_like(q)
    d[1:-1] = (q[2:] - q[:-2]) / (2 * dx)
    d[0] = (q[1] - q[0]) / dx
    d[-1] = (q[-1] - q[-2]) / dx
    return d


# ---------------------------------------------------------------------------
# Rusanov (local Lax-Friedrichs) face-flux for the compressive term
# ---------------------------------------------------------------------------
# The compressive flux F_comp = psi*(1-psi)*n has wavespeed (1-2*psi)*n.
# Central-difference divergence creates oscillations for this hyperbolic
# term. Rusanov face-fluxes add just enough dissipation to eliminate them.
# ---------------------------------------------------------------------------


def _div_split_rusanov_periodic(
    psi: NDArray[np.float64],
    n_hat: NDArray[np.float64],
    eps: float,
    dx: float,
) -> NDArray[np.float64]:
    """
    Divergence of CLS flux [eps*grad(psi) - psi*(1-psi)*n] using
    Rusanov face-flux for the compressive part, central diff for diffusion (1D).
    """
    # Diffusive part: eps * d2psi/dx2 (compact stencil, always stable)
    diff = eps * (np.roll(psi, -1) - 2.0 * psi + np.roll(psi, 1)) / (dx * dx)

    # Compressive part: div(psi*(1-psi)*n) via Rusanov face flux
    F = psi * (1.0 - psi) * n_hat

    F_L = F                     # left cell value at face i+1/2
    F_R = np.roll(F, -1)        # right cell value at face i+1/2
    psi_L = psi
    psi_R = np.roll(psi, -1)

    # Local max wavespeed: max(|1-2*psi_L|, |1-2*psi_R|)
    alpha = np.maximum(np.abs(1.0 - 2.0 * psi_L), np.abs(1.0 - 2.0 * psi_R))

    # Rusanov face flux: F_{i+1/2}
    F_face = 0.5 * (F_L + F_R) - 0.5 * alpha * (psi_R - psi_L)

    # Divergence: (F_{i+1/2} - F_{i-1/2}) / dx
    comp = (F_face - np.roll(F_face, 1)) / dx

    return diff - comp


def _div_split_rusanov_nonperiodic(
    psi: NDArray[np.float64],
    n_hat: NDArray[np.float64],
    eps: float,
    dx: float,
    bc: BoundaryCondition,
) -> NDArray[np.float64]:
    """
    Same as _div_split_rusanov_periodic but with non-periodic BC (1D).
    """
    n = len(psi)

    # Diffusive part
    diff = np.zeros_like(psi)
    diff[1:-1] = eps * (psi[2:] - 2.0 * psi[1:-1] + psi[:-2]) / (dx * dx)
    diff[0] = eps * (psi[1] - 2.0 * psi[0] + psi[1]) / (dx * dx)
    diff[-1] = eps * (psi[-2] - 2.0 * psi[-1] + psi[-2]) / (dx * dx)

    # Compressive Rusanov face fluxes (n-1 faces)
    F = psi * (1.0 - psi) * n_hat
    F_face = np.zeros(n + 1)
    for i in range(n - 1):
        a = max(abs(1.0 - 2.0 * psi[i]), abs(1.0 - 2.0 * psi[i + 1]))
        F_face[i + 1] = 0.5 * (F[i] + F[i + 1]) - 0.5 * a * (psi[i + 1] - psi[i])
    F_face[0] = F[0]
    F_face[n] = F[-1]

    comp = np.zeros_like(psi)
    for i in range(n):
        comp[i] = (F_face[i + 1] - F_face[i]) / dx

    return diff - comp


def _div_split_rusanov_periodic_2d(
    psi: NDArray[np.float64],
    nx_hat: NDArray[np.float64],
    ny_hat: NDArray[np.float64],
    eps: float,
    dx: float,
    dy: float,
) -> NDArray[np.float64]:
    """
    Divergence of CLS flux [eps*grad(psi) - psi*(1-psi)*n] using
    Rusanov face-flux for the compressive part, central diff for diffusion (2D).
    """
    # Diffusive: eps * laplacian(psi)
    d2x = (np.roll(psi, -1, axis=1) - 2.0 * psi + np.roll(psi, 1, axis=1)) / (dx * dx)
    d2y = (np.roll(psi, -1, axis=0) - 2.0 * psi + np.roll(psi, 1, axis=0)) / (dy * dy)
    diff = eps * (d2x + d2y)

    compressive = psi * (1.0 - psi)

    # x-direction compressive Rusanov
    Fx = compressive * nx_hat
    Fx_L = Fx
    Fx_R = np.roll(Fx, -1, axis=1)
    psi_L_x = psi
    psi_R_x = np.roll(psi, -1, axis=1)
    alpha_x = np.maximum(np.abs(1.0 - 2.0 * psi_L_x), np.abs(1.0 - 2.0 * psi_R_x))
    Fx_face = 0.5 * (Fx_L + Fx_R) - 0.5 * alpha_x * (psi_R_x - psi_L_x)
    comp_x = (Fx_face - np.roll(Fx_face, 1, axis=1)) / dx

    # y-direction compressive Rusanov
    Fy = compressive * ny_hat
    Fy_L = Fy
    Fy_R = np.roll(Fy, -1, axis=0)
    psi_L_y = psi
    psi_R_y = np.roll(psi, -1, axis=0)
    alpha_y = np.maximum(np.abs(1.0 - 2.0 * psi_L_y), np.abs(1.0 - 2.0 * psi_R_y))
    Fy_face = 0.5 * (Fy_L + Fy_R) - 0.5 * alpha_y * (psi_R_y - psi_L_y)
    comp_y = (Fy_face - np.roll(Fy_face, 1, axis=0)) / dy

    return diff - (comp_x + comp_y)


def _div_split_rusanov_nonperiodic_2d(
    psi: NDArray[np.float64],
    nx_hat: NDArray[np.float64],
    ny_hat: NDArray[np.float64],
    eps: float,
    dx: float,
    dy: float,
    bc: BoundaryCondition,
) -> NDArray[np.float64]:
    """
    Same as _div_split_rusanov_periodic_2d but non-periodic BC (2D).
    Uses central diff for diffusion; 1st-order upwind for compressive part
    at boundaries, Rusanov in interior.
    """
    ny_dim, nx_dim = psi.shape

    # Diffusive: laplacian
    diff = _laplacian_nonperiodic_2d(psi, dx, dy, bc) * eps

    compressive = psi * (1.0 - psi)

    # x-direction Rusanov face fluxes
    Fx = compressive * nx_hat
    comp_x = np.zeros_like(psi)
    # Interior faces
    for j in range(nx_dim - 1):
        a = np.maximum(np.abs(1.0 - 2.0 * psi[:, j]), np.abs(1.0 - 2.0 * psi[:, j + 1]))
        Ff = 0.5 * (Fx[:, j] + Fx[:, j + 1]) - 0.5 * a * (psi[:, j + 1] - psi[:, j])
        comp_x[:, j] += Ff / dx
        comp_x[:, j + 1] -= Ff / dx
    # Boundary correction
    comp_x[:, 0] += -Fx[:, 0] / dx
    comp_x[:, -1] += Fx[:, -1] / dx

    # y-direction Rusanov face fluxes
    Fy = compressive * ny_hat
    comp_y = np.zeros_like(psi)
    for i in range(ny_dim - 1):
        a = np.maximum(np.abs(1.0 - 2.0 * psi[i, :]), np.abs(1.0 - 2.0 * psi[i + 1, :]))
        Ff = 0.5 * (Fy[i, :] + Fy[i + 1, :]) - 0.5 * a * (psi[i + 1, :] - psi[i, :])
        comp_y[i, :] += Ff / dy
        comp_y[i + 1, :] -= Ff / dy
    comp_y[0, :] += -Fy[0, :] / dy
    comp_y[-1, :] += Fy[-1, :] / dy

    return diff - (comp_x + comp_y)


def _compressive_rusanov_2d(
    psi: NDArray[np.float64],
    nx_hat: NDArray[np.float64],
    ny_hat: NDArray[np.float64],
    dx: float,
    dy: float,
    bc: BoundaryCondition,
) -> NDArray[np.float64]:
    """
    Rusanov face-flux divergence of the compressive flux psi*(1-psi)*n (2D).
    Returns div(psi*(1-psi)*n) so callers can do  rhs = diffusion - compression.
    """
    compressive = psi * (1.0 - psi)

    if bc.bc_type == "periodic":
        # x-direction
        Fx = compressive * nx_hat
        Fx_L = Fx
        Fx_R = np.roll(Fx, -1, axis=1)
        psi_Lx = psi
        psi_Rx = np.roll(psi, -1, axis=1)
        ax = np.maximum(np.abs(1.0 - 2.0 * psi_Lx), np.abs(1.0 - 2.0 * psi_Rx))
        Fx_face = 0.5 * (Fx_L + Fx_R) - 0.5 * ax * (psi_Rx - psi_Lx)
        comp_x = (Fx_face - np.roll(Fx_face, 1, axis=1)) / dx

        # y-direction
        Fy = compressive * ny_hat
        Fy_L = Fy
        Fy_R = np.roll(Fy, -1, axis=0)
        psi_Ly = psi
        psi_Ry = np.roll(psi, -1, axis=0)
        ay = np.maximum(np.abs(1.0 - 2.0 * psi_Ly), np.abs(1.0 - 2.0 * psi_Ry))
        Fy_face = 0.5 * (Fy_L + Fy_R) - 0.5 * ay * (psi_Ry - psi_Ly)
        comp_y = (Fy_face - np.roll(Fy_face, 1, axis=0)) / dy

        return comp_x + comp_y
    else:
        ny_dim, nx_dim = psi.shape
        Fx = compressive * nx_hat
        comp_x = np.zeros_like(psi)
        for j in range(nx_dim - 1):
            a = np.maximum(np.abs(1.0 - 2.0 * psi[:, j]),
                           np.abs(1.0 - 2.0 * psi[:, j + 1]))
            Ff = (0.5 * (Fx[:, j] + Fx[:, j + 1])
                  - 0.5 * a * (psi[:, j + 1] - psi[:, j]))
            comp_x[:, j] += Ff / dx
            comp_x[:, j + 1] -= Ff / dx
        comp_x[:, 0] += -Fx[:, 0] / dx
        comp_x[:, -1] += Fx[:, -1] / dx

        Fy = compressive * ny_hat
        comp_y = np.zeros_like(psi)
        for i in range(ny_dim - 1):
            a = np.maximum(np.abs(1.0 - 2.0 * psi[i, :]),
                           np.abs(1.0 - 2.0 * psi[i + 1, :]))
            Ff = (0.5 * (Fy[i, :] + Fy[i + 1, :])
                  - 0.5 * a * (psi[i + 1, :] - psi[i, :]))
            comp_y[i, :] += Ff / dy
            comp_y[i + 1, :] -= Ff / dy
        comp_y[0, :] += -Fy[0, :] / dy
        comp_y[-1, :] += Fy[-1, :] / dy

        return comp_x + comp_y


# ---------------------------------------------------------------------------
# CLS [2] (2007) — Olsson-Kreiss
# ---------------------------------------------------------------------------


@register_sharpening("olsson_kreiss")
def olsson_kreiss_sharpening(
    psi: NDArray[np.float64],
    dx: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    Olsson-Kreiss CLS reinitialization (1D).

    RHS = div[eps (grad psi . n) n  -  psi(1-psi) n]

    Normal n is frozen at the start of the sub-step (n = sign(grad psi)).
    In 1D the anisotropic diffusion reduces to isotropic, so this is
    equivalent to CL with a frozen normal.
    Uses Rusanov face-flux for the compressive term.

    Pseudo-timestep matches 1D CL: theta = CFL_SAFETY * dx^2 / eps_target.
    The physical ``dt`` is unused (same convention as ``cl_sharpening``).
    Using ``dt * strength`` here would violate explicit-diffusion stability
    when dt ~ dx and eps_target ~ dx.
    """
    theta = _CFL_DIFF * dx * dx / eps_target

    if bc.bc_type == "periodic":
        grad_psi = _grad_periodic(psi, dx)
    else:
        grad_psi = _grad_nonperiodic(psi, dx, bc)

    n = grad_psi / (np.abs(grad_psi) + _ETA)

    if bc.bc_type == "periodic":
        rhs = _div_split_rusanov_periodic(psi, n, eps_target, dx)
    else:
        rhs = _div_split_rusanov_nonperiodic(psi, n, eps_target, dx, bc)

    return np.clip(psi + strength * theta * rhs, 0.0, 1.0)


@register_sharpening("olsson_kreiss_2d")
def olsson_kreiss_sharpening_2d(
    psi: NDArray[np.float64],
    dx: float,
    dy: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    Olsson-Kreiss CLS reinitialization (2D).

    RHS = div[eps (grad psi . n) n  -  psi(1-psi) n]

    Normal n is frozen at the start of the sub-step.
    Diffusion is anisotropic (only along the interface normal).
    Compressive term uses Rusanov face-flux; diffusion uses central diff.
    """
    if bc.bc_type == "periodic":
        dfdx, dfdy = _grad_periodic_2d(psi, dx, dy)
    else:
        dfdx, dfdy = _grad_nonperiodic_2d(psi, dx, dy, bc)

    abs_grad = np.sqrt(dfdx**2 + dfdy**2) + _ETA
    nx = dfdx / abs_grad
    ny = dfdy / abs_grad

    # Anisotropic diffusion: div(eps*(grad_psi.n)*n) via central diff
    grad_dot_n = dfdx * nx + dfdy * ny
    diff_flux_x = eps_target * grad_dot_n * nx
    diff_flux_y = eps_target * grad_dot_n * ny

    if bc.bc_type == "periodic":
        rhs_diff = _div_periodic_2d(diff_flux_x, diff_flux_y, dx, dy)
    else:
        rhs_diff = _div_nonperiodic_2d(diff_flux_x, diff_flux_y, dx, dy, bc)

    # Compressive: div(psi*(1-psi)*n) via Rusanov
    rhs_comp = _compressive_rusanov_2d(psi, nx, ny, dx, dy, bc)

    return np.clip(psi + dt * strength * (rhs_diff - rhs_comp), 0.0, 1.0)


# ---------------------------------------------------------------------------
# ACLS [5] (2008) — Desjardins et al.
# ---------------------------------------------------------------------------


@register_sharpening("acls")
def acls_sharpening(
    psi: NDArray[np.float64],
    dx: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    Accurate Conservative Level Set reinitialization (1D).

    Same flux form as Olsson-Kreiss, but normal n is computed from the
    algebraic signed-distance inversion phi_inv = eps * ln(psi/(1-psi)).
    Uses Rusanov face-flux for the compressive term.

    Pseudo-timestep matches CL/OK: theta = CFL_SAFETY * dx^2 / eps_target.
    """
    theta = _CFL_DIFF * dx * dx / eps_target

    phi = _phi_inv(psi, eps_target)

    if bc.bc_type == "periodic":
        grad_phi = _grad_periodic(phi, dx)
    else:
        grad_phi = _grad_nonperiodic(phi, dx, bc)

    n = grad_phi / (np.abs(grad_phi) + _ETA)

    if bc.bc_type == "periodic":
        rhs = _div_split_rusanov_periodic(psi, n, eps_target, dx)
    else:
        rhs = _div_split_rusanov_nonperiodic(psi, n, eps_target, dx, bc)

    return np.clip(psi + strength * theta * rhs, 0.0, 1.0)


@register_sharpening("acls_2d")
def acls_sharpening_2d(
    psi: NDArray[np.float64],
    dx: float,
    dy: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    Accurate Conservative Level Set reinitialization (2D).

    Normal n from phi_inv = eps * ln(psi/(1-psi));
    flux = eps (grad psi . n) n  -  psi(1-psi) n.
    Compressive term uses Rusanov face-flux.
    """
    phi = _phi_inv(psi, eps_target)

    if bc.bc_type == "periodic":
        dphi_dx, dphi_dy = _grad_periodic_2d(phi, dx, dy)
        dfdx, dfdy = _grad_periodic_2d(psi, dx, dy)
    else:
        dphi_dx, dphi_dy = _grad_nonperiodic_2d(phi, dx, dy, bc)
        dfdx, dfdy = _grad_nonperiodic_2d(psi, dx, dy, bc)

    abs_grad_phi = np.sqrt(dphi_dx**2 + dphi_dy**2) + _ETA
    nx = dphi_dx / abs_grad_phi
    ny = dphi_dy / abs_grad_phi

    # Anisotropic diffusion via central diff
    grad_dot_n = dfdx * nx + dfdy * ny
    diff_flux_x = eps_target * grad_dot_n * nx
    diff_flux_y = eps_target * grad_dot_n * ny

    if bc.bc_type == "periodic":
        rhs_diff = _div_periodic_2d(diff_flux_x, diff_flux_y, dx, dy)
    else:
        rhs_diff = _div_nonperiodic_2d(diff_flux_x, diff_flux_y, dx, dy, bc)

    # Compressive via Rusanov
    rhs_comp = _compressive_rusanov_2d(psi, nx, ny, dx, dy, bc)

    return np.clip(psi + dt * strength * (rhs_diff - rhs_comp), 0.0, 1.0)


# ---------------------------------------------------------------------------
# CLS [3] (2010) — Non-conservative form with mapped normal
# ---------------------------------------------------------------------------


@register_sharpening("cls_2010")
def cls_2010_sharpening(
    psi: NDArray[np.float64],
    dx: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    CLS [3] reinitialization (1D).

    RHS = n . grad[eps |grad psi| - psi(1-psi)]

    Normal n from mapped field phi(psi) = psi^alpha / (psi^alpha + (1-psi)^alpha).

    Pseudo-timestep: theta = CFL_SAFETY * dx^2 / eps_target (same as CL).
    """
    theta = _CFL_DIFF * dx * dx / eps_target
    alpha = kwargs.get("mapping_alpha", 2.0)

    psi_c = np.clip(psi, _ETA, 1.0 - _ETA)
    phi_map = psi_c**alpha / (psi_c**alpha + (1.0 - psi_c)**alpha)

    if bc.bc_type == "periodic":
        grad_phi = _grad_periodic(phi_map, dx)
        grad_psi = _grad_periodic(psi, dx)
    else:
        grad_phi = _grad_nonperiodic(phi_map, dx, bc)
        grad_psi = _grad_nonperiodic(psi, dx, bc)

    abs_grad_phi = np.abs(grad_phi) + _ETA
    n = grad_phi / abs_grad_phi

    # Scalar quantity: eps*|grad psi| - psi*(1-psi)
    scalar = eps_target * np.abs(grad_psi) - psi * (1.0 - psi)

    # Gradient of scalar
    if bc.bc_type == "periodic":
        grad_scalar = _grad_periodic(scalar, dx)
    else:
        grad_scalar = _grad_nonperiodic(scalar, dx, bc)

    rhs = n * grad_scalar

    return np.clip(psi + strength * theta * rhs, 0.0, 1.0)


@register_sharpening("cls_2010_2d")
def cls_2010_sharpening_2d(
    psi: NDArray[np.float64],
    dx: float,
    dy: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    CLS [3] reinitialization (2D).

    RHS = n . grad[eps |grad psi| - psi(1-psi)]

    Normal n from mapped field phi(psi) = psi^alpha / (psi^alpha + (1-psi)^alpha).
    """
    alpha = kwargs.get("mapping_alpha", 2.0)

    psi_c = np.clip(psi, _ETA, 1.0 - _ETA)
    phi_map = psi_c**alpha / (psi_c**alpha + (1.0 - psi_c)**alpha)

    if bc.bc_type == "periodic":
        dphi_dx, dphi_dy = _grad_periodic_2d(phi_map, dx, dy)
        dfdx, dfdy = _grad_periodic_2d(psi, dx, dy)
    else:
        dphi_dx, dphi_dy = _grad_nonperiodic_2d(phi_map, dx, dy, bc)
        dfdx, dfdy = _grad_nonperiodic_2d(psi, dx, dy, bc)

    abs_grad_phi = np.sqrt(dphi_dx**2 + dphi_dy**2) + _ETA
    nx = dphi_dx / abs_grad_phi
    ny = dphi_dy / abs_grad_phi

    # Scalar: eps*|grad psi| - psi*(1-psi)
    abs_grad_psi = np.sqrt(dfdx**2 + dfdy**2)
    scalar = eps_target * abs_grad_psi - psi * (1.0 - psi)

    if bc.bc_type == "periodic":
        ds_dx, ds_dy = _grad_periodic_2d(scalar, dx, dy)
    else:
        ds_dx, ds_dy = _grad_nonperiodic_2d(scalar, dx, dy, bc)

    rhs = nx * ds_dx + ny * ds_dy

    return np.clip(psi + dt * strength * rhs, 0.0, 1.0)


# ---------------------------------------------------------------------------
# LCLS [7] (2012) — Localized CLS
# ---------------------------------------------------------------------------


@register_sharpening("lcls_2012")
def lcls_2012_sharpening(
    psi: NDArray[np.float64],
    dx: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    Localized CLS reinitialization (1D).

    RHS = beta * div[eps * grad(psi) - psi(1-psi) n]

    Localization: beta = 4 psi (1-psi), which peaks at the interface
    and vanishes in bulk regions.
    Uses Rusanov face-flux for the compressive term.

    Pseudo-timestep: theta = CFL_SAFETY * dx^2 / eps_target (same as CL).
    """
    theta = _CFL_DIFF * dx * dx / eps_target
    beta = 4.0 * psi * (1.0 - psi)

    if bc.bc_type == "periodic":
        grad_psi = _grad_periodic(psi, dx)
    else:
        grad_psi = _grad_nonperiodic(psi, dx, bc)

    n = grad_psi / (np.abs(grad_psi) + _ETA)

    if bc.bc_type == "periodic":
        div_flux = _div_split_rusanov_periodic(psi, n, eps_target, dx)
    else:
        div_flux = _div_split_rusanov_nonperiodic(psi, n, eps_target, dx, bc)

    rhs = beta * div_flux

    return np.clip(psi + strength * theta * rhs, 0.0, 1.0)


@register_sharpening("lcls_2012_2d")
def lcls_2012_sharpening_2d(
    psi: NDArray[np.float64],
    dx: float,
    dy: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    Localized CLS reinitialization (2D).

    RHS = beta * div[eps * grad(psi) - psi(1-psi) n]

    beta = 4 psi (1-psi).
    Compressive term uses Rusanov face-flux.
    """
    beta = 4.0 * psi * (1.0 - psi)

    if bc.bc_type == "periodic":
        dfdx, dfdy = _grad_periodic_2d(psi, dx, dy)
    else:
        dfdx, dfdy = _grad_nonperiodic_2d(psi, dx, dy, bc)

    abs_grad = np.sqrt(dfdx**2 + dfdy**2) + _ETA
    nx = dfdx / abs_grad
    ny = dfdy / abs_grad

    if bc.bc_type == "periodic":
        div_flux = _div_split_rusanov_periodic_2d(psi, nx, ny, eps_target, dx, dy)
    else:
        div_flux = _div_split_rusanov_nonperiodic_2d(psi, nx, ny, eps_target, dx, dy, bc)

    rhs = beta * div_flux

    return np.clip(psi + dt * strength * rhs, 0.0, 1.0)


# ---------------------------------------------------------------------------
# LCLS [4] (2014) — Variable pseudo-time localization
# ---------------------------------------------------------------------------


@register_sharpening("lcls_2014")
def lcls_2014_sharpening(
    psi: NDArray[np.float64],
    dx: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    LCLS [4] reinitialization with variable pseudo-time (1D).

    RHS = a_tilde * div[eps (grad psi . n) n  -  psi(1-psi) n]

    Simplified localization weight a_tilde = 4 psi (1-psi).
    Uses Rusanov face-flux for the compressive term.

    Pseudo-timestep: theta = CFL_SAFETY * dx^2 / eps_target (same as CL).
    """
    theta = _CFL_DIFF * dx * dx / eps_target
    a_tilde = 4.0 * psi * (1.0 - psi)

    if bc.bc_type == "periodic":
        grad_psi = _grad_periodic(psi, dx)
    else:
        grad_psi = _grad_nonperiodic(psi, dx, bc)

    n = grad_psi / (np.abs(grad_psi) + _ETA)

    if bc.bc_type == "periodic":
        div_flux = _div_split_rusanov_periodic(psi, n, eps_target, dx)
    else:
        div_flux = _div_split_rusanov_nonperiodic(psi, n, eps_target, dx, bc)

    rhs = a_tilde * div_flux

    return np.clip(psi + strength * theta * rhs, 0.0, 1.0)


@register_sharpening("lcls_2014_2d")
def lcls_2014_sharpening_2d(
    psi: NDArray[np.float64],
    dx: float,
    dy: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    LCLS [4] reinitialization with variable pseudo-time (2D).

    RHS = a_tilde * div[eps (grad psi . n) n  -  psi(1-psi) n]

    a_tilde = 4 psi (1-psi).
    Compressive term uses Rusanov face-flux; diffusion uses central diff.
    """
    a_tilde = 4.0 * psi * (1.0 - psi)

    if bc.bc_type == "periodic":
        dfdx, dfdy = _grad_periodic_2d(psi, dx, dy)
    else:
        dfdx, dfdy = _grad_nonperiodic_2d(psi, dx, dy, bc)

    abs_grad = np.sqrt(dfdx**2 + dfdy**2) + _ETA
    nx = dfdx / abs_grad
    ny = dfdy / abs_grad

    # Anisotropic diffusion via central diff
    grad_dot_n = dfdx * nx + dfdy * ny
    diff_flux_x = eps_target * grad_dot_n * nx
    diff_flux_y = eps_target * grad_dot_n * ny

    if bc.bc_type == "periodic":
        rhs_diff = _div_periodic_2d(diff_flux_x, diff_flux_y, dx, dy)
    else:
        rhs_diff = _div_nonperiodic_2d(diff_flux_x, diff_flux_y, dx, dy, bc)

    # Compressive via Rusanov
    rhs_comp = _compressive_rusanov_2d(psi, nx, ny, dx, dy, bc)

    rhs = a_tilde * (rhs_diff - rhs_comp)

    return np.clip(psi + dt * strength * rhs, 0.0, 1.0)


# ---------------------------------------------------------------------------
# CLS [10] (2015) — Mapping function approach
# ---------------------------------------------------------------------------


@register_sharpening("cls_2015")
def cls_2015_sharpening(
    psi: NDArray[np.float64],
    dx: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    CLS [10] mapping-function reinitialization (1D).

    RHS = div[psi(1-psi) (|grad phi_map| - 1) n_Gamma]

    phi_map = (psi+eps)^gamma / ((psi+eps)^gamma + (1-psi+eps)^gamma)
    n_Gamma = grad(phi_map) / |grad(phi_map)|

    Split into diffusive (central diff) and compressive (Rusanov) parts.
    """
    gamma = kwargs.get("mapping_gamma", 2.0)
    eps = eps_target

    psi_c = np.clip(psi, 0.0, 1.0)
    a = (psi_c + eps)**gamma
    b = (1.0 - psi_c + eps)**gamma
    phi_map = a / (a + b)

    if bc.bc_type == "periodic":
        grad_phi = _grad_periodic(phi_map, dx)
    else:
        grad_phi = _grad_nonperiodic(phi_map, dx, bc)

    abs_grad_phi = np.abs(grad_phi) + _ETA
    n = grad_phi / abs_grad_phi

    # Expand: psi*(1-psi)*(|grad_phi|-1)*n = psi*(1-psi)*|grad_phi|*n - psi*(1-psi)*n
    # Diffusive part: psi*(1-psi)*|grad_phi|*n (central diff ok)
    diff_flux = psi * (1.0 - psi) * abs_grad_phi * n
    if bc.bc_type == "periodic":
        rhs_diff = _div_periodic(diff_flux, dx)
    else:
        rhs_diff = _div_nonperiodic(diff_flux, dx, bc)

    # Compressive part: psi*(1-psi)*n (Rusanov)
    F_comp = psi * (1.0 - psi) * n
    if bc.bc_type == "periodic":
        F_L = F_comp
        F_R = np.roll(F_comp, -1)
        psi_L = psi
        psi_R = np.roll(psi, -1)
        alpha_lf = np.maximum(np.abs(1.0 - 2.0 * psi_L), np.abs(1.0 - 2.0 * psi_R))
        F_face = 0.5 * (F_L + F_R) - 0.5 * alpha_lf * (psi_R - psi_L)
        rhs_comp = (F_face - np.roll(F_face, 1)) / dx
    else:
        nn = len(psi)
        F_face_arr = np.zeros(nn + 1)
        for i in range(nn - 1):
            a_lf = max(abs(1.0 - 2.0 * psi[i]), abs(1.0 - 2.0 * psi[i + 1]))
            F_face_arr[i + 1] = 0.5 * (F_comp[i] + F_comp[i + 1]) - 0.5 * a_lf * (psi[i + 1] - psi[i])
        F_face_arr[0] = F_comp[0]
        F_face_arr[nn] = F_comp[-1]
        rhs_comp = np.zeros_like(psi)
        for i in range(nn):
            rhs_comp[i] = (F_face_arr[i + 1] - F_face_arr[i]) / dx

    rhs = rhs_diff - rhs_comp

    return np.clip(psi + dt * strength * rhs, 0.0, 1.0)


@register_sharpening("cls_2015_2d")
def cls_2015_sharpening_2d(
    psi: NDArray[np.float64],
    dx: float,
    dy: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    CLS [10] mapping-function reinitialization (2D).

    RHS = div[psi(1-psi) (|grad phi_map| - 1) n_Gamma]

    Split into diffusive (central diff) and compressive (Rusanov) parts.
    """
    gamma = kwargs.get("mapping_gamma", 2.0)
    eps = eps_target

    psi_c = np.clip(psi, 0.0, 1.0)
    a = (psi_c + eps)**gamma
    b = (1.0 - psi_c + eps)**gamma
    phi_map = a / (a + b)

    if bc.bc_type == "periodic":
        dphi_dx, dphi_dy = _grad_periodic_2d(phi_map, dx, dy)
    else:
        dphi_dx, dphi_dy = _grad_nonperiodic_2d(phi_map, dx, dy, bc)

    abs_grad_phi = np.sqrt(dphi_dx**2 + dphi_dy**2) + _ETA
    nx = dphi_dx / abs_grad_phi
    ny = dphi_dy / abs_grad_phi

    # Diffusive part: psi*(1-psi)*|grad_phi|*n (central diff)
    diff_flux_x = psi * (1.0 - psi) * abs_grad_phi * nx
    diff_flux_y = psi * (1.0 - psi) * abs_grad_phi * ny

    if bc.bc_type == "periodic":
        rhs_diff = _div_periodic_2d(diff_flux_x, diff_flux_y, dx, dy)
    else:
        rhs_diff = _div_nonperiodic_2d(diff_flux_x, diff_flux_y, dx, dy, bc)

    # Compressive: psi*(1-psi)*n via Rusanov
    rhs_comp = _compressive_rusanov_2d(psi, nx, ny, dx, dy, bc)

    return np.clip(psi + dt * strength * (rhs_diff - rhs_comp), 0.0, 1.0)


# ---------------------------------------------------------------------------
# CLS [11] (2017) — Inverse transform with cosh
# ---------------------------------------------------------------------------


@register_sharpening("cls_2017")
def cls_2017_sharpening(
    psi: NDArray[np.float64],
    dx: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    CLS [11] inverse-transform reinitialization (1D).

    RHS = div[ 1/(4 cosh^2(phi_inv / (2 eps)))  *  (|grad phi_inv| - 1)  *  n ]

    phi_inv = eps * ln(psi/(1-psi)),  n = grad(phi_inv) / |grad(phi_inv)|.

    Rewritten using weight = psi*(1-psi), and splitting into standard CLS
    form with a correction. Rusanov face-flux for the compressive term.

    Pseudo-timestep: theta = CFL_SAFETY * dx^2 / eps_target (same as CL).
    """
    eps = eps_target
    phi = _phi_inv(psi, eps)

    if bc.bc_type == "periodic":
        grad_phi = _grad_periodic(phi, dx)
    else:
        grad_phi = _grad_nonperiodic(phi, dx, bc)

    abs_grad_phi = np.abs(grad_phi) + _ETA
    n = grad_phi / abs_grad_phi

    # weight = 1/(4*cosh^2) = psi*(1-psi)
    # flux = psi*(1-psi) * (|grad_phi| - 1) * n
    # Expand: psi*(1-psi)*|grad_phi|*n - psi*(1-psi)*n
    # The second term is the standard compressive flux.
    # The first term is diffusion-like: psi*(1-psi)*|grad_phi|*n = psi*(1-psi)*grad_phi (in 1D)
    # Rewrite: flux = psi*(1-psi)*grad_phi - psi*(1-psi)*n

    # Diffusion part: compute pointwise psi*(1-psi)*|grad_phi| and use as effective eps
    # in the Laplacian approximation for stability.
    # Simpler approach: compute RHS as product of two terms.
    arg = np.clip(phi / (2.0 * eps), -50.0, 50.0)
    weight = 1.0 / (4.0 * np.cosh(arg)**2)

    # Use Rusanov for the full div(weight*(|grad_phi|-1)*n)
    # Treat it as: div(F) where F = weight*(|grad_phi|-1)*n
    # The "compressive" part is weight*(-1)*n = -psi*(1-psi)*n (standard)
    # The "diffusive" part is weight*|grad_phi|*n

    # Diffusive contribution (central diff is fine)
    diff_flux = weight * abs_grad_phi * n
    if bc.bc_type == "periodic":
        rhs_diff = _div_periodic(diff_flux, dx)
    else:
        rhs_diff = _div_nonperiodic(diff_flux, dx, bc)

    # Compressive contribution via Rusanov
    F_comp = psi * (1.0 - psi) * n
    if bc.bc_type == "periodic":
        F_L = F_comp
        F_R = np.roll(F_comp, -1)
        psi_L = psi
        psi_R = np.roll(psi, -1)
        alpha = np.maximum(np.abs(1.0 - 2.0 * psi_L), np.abs(1.0 - 2.0 * psi_R))
        F_face = 0.5 * (F_L + F_R) - 0.5 * alpha * (psi_R - psi_L)
        rhs_comp = (F_face - np.roll(F_face, 1)) / dx
    else:
        nn = len(psi)
        F_face = np.zeros(nn + 1)
        for i in range(nn - 1):
            a = max(abs(1.0 - 2.0 * psi[i]), abs(1.0 - 2.0 * psi[i + 1]))
            F_face[i + 1] = 0.5 * (F_comp[i] + F_comp[i + 1]) - 0.5 * a * (psi[i + 1] - psi[i])
        F_face[0] = F_comp[0]
        F_face[nn] = F_comp[-1]
        rhs_comp = np.zeros_like(psi)
        for i in range(nn):
            rhs_comp[i] = (F_face[i + 1] - F_face[i]) / dx

    theta = _CFL_DIFF * dx * dx / eps_target
    rhs = rhs_diff - rhs_comp

    return np.clip(psi + strength * theta * rhs, 0.0, 1.0)


@register_sharpening("cls_2017_2d")
def cls_2017_sharpening_2d(
    psi: NDArray[np.float64],
    dx: float,
    dy: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    CLS [11] inverse-transform reinitialization (2D).

    RHS = div[ weight * (|grad phi_inv| - 1) * n ]

    where weight = 1/(4 cosh^2(phi/(2*eps))) = psi*(1-psi).
    Split into diffusive (central diff) and compressive (Rusanov) parts.
    """
    eps = eps_target
    phi = _phi_inv(psi, eps)

    if bc.bc_type == "periodic":
        dphi_dx, dphi_dy = _grad_periodic_2d(phi, dx, dy)
    else:
        dphi_dx, dphi_dy = _grad_nonperiodic_2d(phi, dx, dy, bc)

    abs_grad_phi = np.sqrt(dphi_dx**2 + dphi_dy**2) + _ETA
    nx = dphi_dx / abs_grad_phi
    ny = dphi_dy / abs_grad_phi

    arg = np.clip(phi / (2.0 * eps), -50.0, 50.0)
    weight = 1.0 / (4.0 * np.cosh(arg)**2)

    # Diffusive part: weight*|grad_phi|*n (central diff)
    diff_flux_x = weight * abs_grad_phi * nx
    diff_flux_y = weight * abs_grad_phi * ny

    if bc.bc_type == "periodic":
        rhs_diff = _div_periodic_2d(diff_flux_x, diff_flux_y, dx, dy)
    else:
        rhs_diff = _div_nonperiodic_2d(diff_flux_x, diff_flux_y, dx, dy, bc)

    # Compressive: psi*(1-psi)*n via Rusanov (weight ≈ psi*(1-psi))
    rhs_comp = _compressive_rusanov_2d(psi, nx, ny, dx, dy, bc)

    return np.clip(psi + dt * strength * (rhs_diff - rhs_comp), 0.0, 1.0)


# ---------------------------------------------------------------------------
# SCLS [8] (2018) — Self-Correcting Level Set
# ---------------------------------------------------------------------------


@register_sharpening("scls")
def scls_sharpening(
    psi: NDArray[np.float64],
    dx: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    Self-Correcting Level Set reinitialization (1D).

    RHS = -div(psi(1-psi) m) + div(eps (grad psi . m) m) + div((1 - |m|^2) eps grad psi)

    m = eps grad(psi) / sqrt(eps^2 |grad psi|^2  +  alpha^2 exp(-beta eps^2 |grad psi|^2))

    Compressive term uses Rusanov face-flux; diffusive terms use central diff.

    Pseudo-timestep: theta = CFL_SAFETY * dx^2 / eps_target (same as CL).
    """
    theta = _CFL_DIFF * dx * dx / eps_target
    alpha_sc = kwargs.get("scls_alpha", 1e-3)
    beta_sc = kwargs.get("scls_beta", 1e3)
    eps = eps_target

    if bc.bc_type == "periodic":
        grad_psi = _grad_periodic(psi, dx)
    else:
        grad_psi = _grad_nonperiodic(psi, dx, bc)

    grad_sq = grad_psi**2
    eps_grad_sq = eps**2 * grad_sq

    denom = np.sqrt(eps_grad_sq + alpha_sc**2 * np.exp(-beta_sc * eps_grad_sq)) + _ETA
    m = eps * grad_psi / denom
    m_sq = m**2

    # Term 1 (compressive): -div(psi(1-psi) m) via Rusanov
    F_comp = psi * (1.0 - psi) * m
    if bc.bc_type == "periodic":
        F_L = F_comp
        F_R = np.roll(F_comp, -1)
        psi_L = psi
        psi_R = np.roll(psi, -1)
        alpha_lf = np.maximum(np.abs(1.0 - 2.0 * psi_L), np.abs(1.0 - 2.0 * psi_R))
        F_face = 0.5 * (F_L + F_R) - 0.5 * alpha_lf * (psi_R - psi_L)
        div_comp = (F_face - np.roll(F_face, 1)) / dx
    else:
        nn = len(psi)
        F_face = np.zeros(nn + 1)
        for i in range(nn - 1):
            a = max(abs(1.0 - 2.0 * psi[i]), abs(1.0 - 2.0 * psi[i + 1]))
            F_face[i + 1] = 0.5 * (F_comp[i] + F_comp[i + 1]) - 0.5 * a * (psi[i + 1] - psi[i])
        F_face[0] = F_comp[0]
        F_face[nn] = F_comp[-1]
        div_comp = np.zeros_like(psi)
        for i in range(nn):
            div_comp[i] = (F_face[i + 1] - F_face[i]) / dx

    # Terms 2+3 (diffusive): central difference
    grad_dot_m = grad_psi * m
    flux_diff = eps * grad_dot_m * m + (1.0 - m_sq) * eps * grad_psi

    if bc.bc_type == "periodic":
        div_diff = _div_periodic(flux_diff, dx)
    else:
        div_diff = _div_nonperiodic(flux_diff, dx, bc)

    rhs = -div_comp + div_diff

    return np.clip(psi + strength * theta * rhs, 0.0, 1.0)


@register_sharpening("scls_2d")
def scls_sharpening_2d(
    psi: NDArray[np.float64],
    dx: float,
    dy: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    Self-Correcting Level Set reinitialization (2D).

    RHS = -div(psi(1-psi) m) + div(eps (grad psi . m) m) + div((1 - |m|^2) eps grad psi)

    m = eps grad(psi) / sqrt(eps^2 |grad psi|^2  +  alpha^2 exp(-beta eps^2 |grad psi|^2))

    Compressive term uses Rusanov face-flux; diffusive terms use central diff.
    """
    alpha_sc = kwargs.get("scls_alpha", 1e-3)
    beta_sc = kwargs.get("scls_beta", 1e3)
    eps = eps_target

    if bc.bc_type == "periodic":
        dfdx, dfdy = _grad_periodic_2d(psi, dx, dy)
    else:
        dfdx, dfdy = _grad_nonperiodic_2d(psi, dx, dy, bc)

    grad_sq = dfdx**2 + dfdy**2
    eps_grad_sq = eps**2 * grad_sq

    denom = np.sqrt(eps_grad_sq + alpha_sc**2 * np.exp(-beta_sc * eps_grad_sq)) + _ETA
    mx = eps * dfdx / denom
    my = eps * dfdy / denom
    m_sq = mx**2 + my**2

    # Term 1 (compressive): -div(psi(1-psi) m) via Rusanov
    rhs_comp = _compressive_rusanov_2d(psi, mx, my, dx, dy, bc)

    # Terms 2+3 (diffusive): central diff
    grad_dot_m = dfdx * mx + dfdy * my
    diff_fx = eps * grad_dot_m * mx + (1.0 - m_sq) * eps * dfdx
    diff_fy = eps * grad_dot_m * my + (1.0 - m_sq) * eps * dfdy

    if bc.bc_type == "periodic":
        rhs_diff = _div_periodic_2d(diff_fx, diff_fy, dx, dy)
    else:
        rhs_diff = _div_nonperiodic_2d(diff_fx, diff_fy, dx, dy, bc)

    return np.clip(psi + dt * strength * (-rhs_comp + rhs_diff), 0.0, 1.0)


# ---------------------------------------------------------------------------
# FIRST-ORDER DERIVATION 1: Localized PM (LPM)
# ---------------------------------------------------------------------------
# Derived from LCLS [7] (Eq. B.1) by applying PM approximations (13),(15).
#
# Start: dψ/dτ = β [ε ∇²ψ − ∇·(ψ(1−ψ)n)], β = 4ψ(1−ψ)
#
# Step 1: Expand non-conservatively, remove curvature advection (same as
#         PM paper §2.1, Eq. 10):
#   dψ/dτ = β [ε (n₀·∇(∇ψ))·n₀ − (1−2ψ) ∇ψ·n₀]
#
# Step 2: Apply approximation (13): ∇ψ·n₀ ≈ ψ(1−ψ)/ε
#   Compression becomes: −(1−2ψ) · ψ(1−ψ)/ε    [ZEROTH order]
#
# Step 3: Apply approximation (15): ε(n₀·∇(∇ψ))·n₀ ≈ (1−2ψ)|∇ψ|
#   Diffusion becomes: (1−2ψ)|∇ψ|               [FIRST order]
#
# Result: dψ/dτ = 4ψ(1−ψ)(1−2ψ) [|∇ψ| − ψ(1−ψ)/ε]
#
# Properties:
#   - First order (only |∇ψ|, no Laplacian)
#   - Quartic localization 4ψ(1−ψ) suppresses far from interface
#   - Reaction: −4ψ²(1−ψ)²(1−2ψ)/ε  (zeroth-order compression)
#   - Diffusion: 4ψ(1−ψ)(1−2ψ)|∇ψ|  (first-order balancing)
#   - Steady state: |∇ψ| = ψ(1−ψ)/ε  (same tanh profile as PM/CLS)
#   - NOT conservative (non-divergence form after approximation)
# ---------------------------------------------------------------------------

@register_sharpening("lpm")
def lpm_sharpening(
    psi: NDArray[np.float64],
    dx: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    Localized Parameswaran-Mandal sharpening (1D).

    Derived from LCLS by applying PM approximations (13) and (15).
    RHS = 4*psi*(1-psi)*(1-2*psi)*[|grad psi| - psi*(1-psi)/eps]
    """
    if bc.bc_type == "periodic":
        grad_psi = _grad_periodic(psi, dx)
    else:
        grad_psi = _grad_nonperiodic(psi, dx, bc)

    abs_grad = np.abs(grad_psi)
    beta = 4.0 * psi * (1.0 - psi)
    departure = abs_grad - psi * (1.0 - psi) / eps_target
    rhs = beta * (1.0 - 2.0 * psi) * departure

    return np.clip(psi + dt * strength * rhs, 0.0, 1.0)


@register_sharpening("lpm_2d")
def lpm_sharpening_2d(
    psi: NDArray[np.float64],
    dx: float,
    dy: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    Localized Parameswaran-Mandal sharpening (2D).

    RHS = 4*psi*(1-psi)*(1-2*psi)*[|grad psi| - psi*(1-psi)/eps]
    """
    if bc.bc_type == "periodic":
        dfdx, dfdy = _grad_periodic_2d(psi, dx, dy)
    else:
        dfdx, dfdy = _grad_nonperiodic_2d(psi, dx, dy, bc)

    abs_grad = np.sqrt(dfdx**2 + dfdy**2)
    beta = 4.0 * psi * (1.0 - psi)
    departure = abs_grad - psi * (1.0 - psi) / eps_target
    rhs = beta * (1.0 - 2.0 * psi) * departure

    return np.clip(psi + dt * strength * rhs, 0.0, 1.0)


# ---------------------------------------------------------------------------
# FIRST-ORDER DERIVATION 2: Conservative First-Order (CFO)
# ---------------------------------------------------------------------------
# Derived from the CLS equilibrium condition and flux-potential theory.
#
# Start: CLS equilibrium (Eq. 22): ε|∇ψ| = ψ(1−ψ)
#   Equivalently: ψ(1−ψ) − ε|∇ψ| = 0
#
# Step 1: Define a scalar potential whose roots encode the equilibrium:
#   g(ψ) = ψ(1−ψ)[ψ(1−ψ) − ε]
#   g = 0  iff  ψ ∈ {0, 1} or ψ(1−ψ) = ε
#
# Step 2: Form the flux F = g(ψ) · n̂  where n̂ = ∇ψ/|∇ψ|.
#   This is a CONSERVATIVE flux (divergence form).
#
# Step 3: The reinitialization equation dψ/dτ + ∇·F = 0 gives:
#   dψ/dτ = −g'(ψ)|∇ψ| = −(1−2ψ)[2ψ(1−ψ) − ε]|∇ψ|
#
# Splitting the RHS into compression and diffusion:
#   Compression: −2ψ(1−ψ)(1−2ψ)|∇ψ|  (drives ψ → {0,1})
#   Diffusion:   +ε(1−2ψ)|∇ψ|          (balances, prevents overshoot)
#
# Both are FIRST-ORDER (only |∇ψ|, no Laplacian).
#
# Properties:
#   - First order
#   - CONSERVATIVE (divergence form ∇·(g·n̂), mass preserved on periodic)
#   - Self-localizing (g → 0 at ψ = 0, 1)
#   - Steady state: g(ψ) = 0, i.e., ψ(1−ψ) = ε → very sharp interface
#   - Discretized via Rusanov face-flux for stability
# ---------------------------------------------------------------------------

def _cfo_rusanov_periodic(
    psi: NDArray[np.float64],
    n_hat: NDArray[np.float64],
    eps: float,
    dx: float,
) -> NDArray[np.float64]:
    """Rusanov face-flux divergence for the CFO flux g(psi)*n_hat (1D periodic)."""
    g = psi * (1.0 - psi) * (psi * (1.0 - psi) - eps)
    F = g * n_hat

    F_L = F
    F_R = np.roll(F, -1)
    psi_L = psi
    psi_R = np.roll(psi, -1)

    p1 = psi_L * (1.0 - psi_L)
    p2 = psi_R * (1.0 - psi_R)
    alpha = np.maximum(
        np.abs((1.0 - 2.0 * psi_L) * (2.0 * p1 - eps)),
        np.abs((1.0 - 2.0 * psi_R) * (2.0 * p2 - eps)),
    )

    F_face = 0.5 * (F_L + F_R) - 0.5 * alpha * (psi_R - psi_L)
    return (F_face - np.roll(F_face, 1)) / dx


def _cfo_rusanov_nonperiodic(
    psi: NDArray[np.float64],
    n_hat: NDArray[np.float64],
    eps: float,
    dx: float,
    bc: BoundaryCondition,
) -> NDArray[np.float64]:
    """Rusanov face-flux divergence for the CFO flux g(psi)*n_hat (1D non-periodic)."""
    n = len(psi)
    g = psi * (1.0 - psi) * (psi * (1.0 - psi) - eps)
    F = g * n_hat

    F_face = np.zeros(n + 1)
    for i in range(n - 1):
        pL, pR = psi[i], psi[i + 1]
        p1 = pL * (1.0 - pL)
        p2 = pR * (1.0 - pR)
        a = max(abs((1.0 - 2.0 * pL) * (2.0 * p1 - eps)),
                abs((1.0 - 2.0 * pR) * (2.0 * p2 - eps)))
        F_face[i + 1] = 0.5 * (F[i] + F[i + 1]) - 0.5 * a * (pR - pL)
    F_face[0] = F_face[1]
    F_face[n] = F_face[n - 1]

    return (F_face[1:] - F_face[:-1]) / dx


@register_sharpening("cfo")
def cfo_sharpening(
    psi: NDArray[np.float64],
    dx: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    Conservative First-Order sharpening (1D).

    Derived from the CLS equilibrium via flux potential g(psi) = psi(1-psi)[psi(1-psi)-eps].
    RHS = -div(g(psi)*n_hat) via Rusanov face-flux.
    """
    if bc.bc_type == "periodic":
        grad_psi = _grad_periodic(psi, dx)
    else:
        grad_psi = _grad_nonperiodic(psi, dx, bc)

    n_hat = grad_psi / (np.abs(grad_psi) + _ETA)

    if bc.bc_type == "periodic":
        div_flux = _cfo_rusanov_periodic(psi, n_hat, eps_target, dx)
    else:
        div_flux = _cfo_rusanov_nonperiodic(psi, n_hat, eps_target, dx, bc)

    return np.clip(psi + dt * strength * (-div_flux), 0.0, 1.0)


@register_sharpening("cfo_2d")
def cfo_sharpening_2d(
    psi: NDArray[np.float64],
    dx: float,
    dy: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    Conservative First-Order sharpening (2D).

    Flux F = g(psi)*n_hat where g = psi(1-psi)[psi(1-psi)-eps], discretized via Rusanov.
    """
    if bc.bc_type == "periodic":
        dfdx, dfdy = _grad_periodic_2d(psi, dx, dy)
    else:
        dfdx, dfdy = _grad_nonperiodic_2d(psi, dx, dy, bc)

    abs_grad = np.sqrt(dfdx**2 + dfdy**2) + _ETA
    nx = dfdx / abs_grad
    ny = dfdy / abs_grad

    g = psi * (1.0 - psi) * (psi * (1.0 - psi) - eps_target)
    Fx = g * nx
    Fy = g * ny

    p1 = psi * (1.0 - psi)
    wavespeed = np.abs((1.0 - 2.0 * psi) * (2.0 * p1 - eps_target))

    if bc.bc_type == "periodic":
        # x-direction Rusanov
        Fx_L, Fx_R = Fx, np.roll(Fx, -1, axis=1)
        psi_L, psi_R = psi, np.roll(psi, -1, axis=1)
        ws_R = np.roll(wavespeed, -1, axis=1)
        alpha_x = np.maximum(wavespeed, ws_R)
        Fx_face = 0.5 * (Fx_L + Fx_R) - 0.5 * alpha_x * (psi_R - psi_L)
        div_x = (Fx_face - np.roll(Fx_face, 1, axis=1)) / dx

        # y-direction Rusanov
        Fy_L, Fy_R = Fy, np.roll(Fy, -1, axis=0)
        psi_L, psi_R = psi, np.roll(psi, -1, axis=0)
        ws_R = np.roll(wavespeed, -1, axis=0)
        alpha_y = np.maximum(wavespeed, ws_R)
        Fy_face = 0.5 * (Fy_L + Fy_R) - 0.5 * alpha_y * (psi_R - psi_L)
        div_y = (Fy_face - np.roll(Fy_face, 1, axis=0)) / dy
    else:
        div_x = np.zeros_like(psi)
        div_y = np.zeros_like(psi)
        ny_, nx_ = psi.shape

        for i in range(nx_ - 1):
            aL = wavespeed[:, i]
            aR = wavespeed[:, i + 1]
            a = np.maximum(aL, aR)
            ff = 0.5 * (Fx[:, i] + Fx[:, i + 1]) - 0.5 * a * (psi[:, i + 1] - psi[:, i])
            div_x[:, i] += ff / dx
            div_x[:, i + 1] -= ff / dx

        for j in range(ny_ - 1):
            aL = wavespeed[j, :]
            aR = wavespeed[j + 1, :]
            a = np.maximum(aL, aR)
            ff = 0.5 * (Fy[j, :] + Fy[j + 1, :]) - 0.5 * a * (psi[j + 1, :] - psi[j, :])
            div_y[j, :] += ff / dy
            div_y[j + 1, :] -= ff / dy

    return np.clip(psi + dt * strength * (-(div_x + div_y)), 0.0, 1.0)


# ---------------------------------------------------------------------------
# NEW PROPOSAL 1: Volume-Conserving Allen-Cahn (VCAC)
# ---------------------------------------------------------------------------
# Combines PM's cubic reaction ψ(1-ψ)(1-2ψ) with Laplacian diffusion and a
# Lagrange multiplier λ = -mean(R) that enforces exact discrete mass
# conservation.  This is the Allen-Cahn equation with volume preservation
# (Rubinstein-Sternberg 1992, Brassel-Bretin 2011), adapted to the
# CLS reinitialization context.
#
# Continuum PDE:
#   ∂ψ/∂τ = -K ψ(1-ψ)(1-2ψ) + ε ∇²ψ + λ(τ)
#   where  λ = -1/|Ω| ∫ [-K ψ(1-ψ)(1-2ψ) + ε ∇²ψ] dx
#
# Properties:
#   - Mass-conserving by construction (∑ψ' = ∑ψ exactly)
#   - Sharp: cubic reaction drives ψ → {0, 1}
#   - Diffusion regularises the interface to width ~ ε
#   - Bounded under CFL on dt, eps
# ---------------------------------------------------------------------------

@register_sharpening("vcac")
def vcac_sharpening(
    psi: NDArray[np.float64],
    dx: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    Volume-Conserving Allen-Cahn sharpening (1D).

    RHS = -K ψ(1-ψ)(1-2ψ) + ε ∇²ψ + λ
    where K = 1/(4ε²) and λ = -mean(RHS_without_λ).
    """
    K = 1.0 / (4.0 * eps_target**2)

    reaction = -K * psi * (1.0 - psi) * (1.0 - 2.0 * psi)

    if bc.bc_type == "periodic":
        diffusion = eps_target * _laplacian_periodic(psi, dx)
    else:
        diffusion = eps_target * _laplacian_nonperiodic(psi, dx, bc)

    rhs_raw = reaction + diffusion
    lam = -np.mean(rhs_raw)
    rhs = rhs_raw + lam

    return np.clip(psi + dt * strength * rhs, 0.0, 1.0)


@register_sharpening("vcac_2d")
def vcac_sharpening_2d(
    psi: NDArray[np.float64],
    dx: float,
    dy: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    Volume-Conserving Allen-Cahn sharpening (2D).

    RHS = -K ψ(1-ψ)(1-2ψ) + ε ∇²ψ + λ
    where K = 1/(4ε²) and λ = -mean(RHS_without_λ).
    """
    K = 1.0 / (4.0 * eps_target**2)

    reaction = -K * psi * (1.0 - psi) * (1.0 - 2.0 * psi)

    if bc.bc_type == "periodic":
        diffusion = eps_target * _laplacian_periodic_2d(psi, dx, dy)
    else:
        diffusion = eps_target * _laplacian_nonperiodic_2d(psi, dx, dy, bc)

    rhs_raw = reaction + diffusion
    lam = -np.mean(rhs_raw)
    rhs = rhs_raw + lam

    return np.clip(psi + dt * strength * rhs, 0.0, 1.0)


# ---------------------------------------------------------------------------
# NEW PROPOSAL 2: Gradient-Localized CLS (GLCLS)
# ---------------------------------------------------------------------------
# Standard CLS with a gradient-magnitude localization weight instead of
# the field-value-based 4ψ(1-ψ) used by LCLS.
#
# Continuum PDE:
#   ∂ψ/∂τ + div(w · F_CLS) = 0
#   F_CLS = ε∇ψ − ψ(1-ψ)n
#   w = |∇ψ|² / (|∇ψ|² + δ²)
#
# Motivation: gradient-based localization is invariant to the functional
# form of the profile; it activates wherever the interface exists,
# regardless of whether ψ is close to 0.5 or not. This avoids the
# issue where 4ψ(1-ψ) suppresses the operator too aggressively in
# transition regions with ψ close to 0 or 1 but nonzero gradient.
#
# Properties:
#   - Conservative (divergence form, periodic ⟹ ∑ψ preserved up to
#     the localization weight, which is close to conservative for smooth w)
#   - Localized: only active near the interface
#   - δ controls the localization width (default: 1/dx)
# ---------------------------------------------------------------------------

@register_sharpening("glcls")
def glcls_sharpening(
    psi: NDArray[np.float64],
    dx: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    Gradient-Localized CLS sharpening (1D).

    RHS = w · div(ε grad(ψ) - ψ(1-ψ) n)
    where w = |∇ψ|² / (|∇ψ|² + δ²), δ = glcls_delta (default 1/dx).
    """
    delta = kwargs.get("glcls_delta", 1.0 / dx)

    if bc.bc_type == "periodic":
        grad_psi = _grad_periodic(psi, dx)
    else:
        grad_psi = _grad_nonperiodic(psi, dx, bc)

    grad_sq = grad_psi**2
    w = grad_sq / (grad_sq + delta**2)

    n_hat = grad_psi / (np.abs(grad_psi) + _ETA)

    if bc.bc_type == "periodic":
        div_flux = _div_split_rusanov_periodic(psi, n_hat, eps_target, dx)
    else:
        div_flux = _div_split_rusanov_nonperiodic(psi, n_hat, eps_target, dx, bc)

    rhs = w * div_flux

    return np.clip(psi + dt * strength * rhs, 0.0, 1.0)


@register_sharpening("glcls_2d")
def glcls_sharpening_2d(
    psi: NDArray[np.float64],
    dx: float,
    dy: float,
    dt: float,
    eps_target: float,
    strength: float,
    bc: BoundaryCondition,
    **kwargs,
) -> NDArray[np.float64]:
    """
    Gradient-Localized CLS sharpening (2D).

    RHS = w · div(ε grad(ψ) - ψ(1-ψ) n)
    where w = |∇ψ|² / (|∇ψ|² + δ²), δ = glcls_delta.
    """
    delta = kwargs.get("glcls_delta", 1.0 / min(dx, dy))

    if bc.bc_type == "periodic":
        dfdx, dfdy = _grad_periodic_2d(psi, dx, dy)
    else:
        dfdx, dfdy = _grad_nonperiodic_2d(psi, dx, dy, bc)

    grad_sq = dfdx**2 + dfdy**2
    w = grad_sq / (grad_sq + delta**2)

    abs_grad = np.sqrt(grad_sq) + _ETA
    nx = dfdx / abs_grad
    ny = dfdy / abs_grad

    rhs_flux = _compressive_rusanov_2d(psi, nx, ny, dx, dy, bc)

    if bc.bc_type == "periodic":
        diff = eps_target * _laplacian_periodic_2d(psi, dx, dy)
    else:
        diff = eps_target * _laplacian_nonperiodic_2d(psi, dx, dy, bc)

    rhs = w * (diff - rhs_flux)

    return np.clip(psi + dt * strength * rhs, 0.0, 1.0)

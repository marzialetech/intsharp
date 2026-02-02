"""
Interface sharpening source terms.

Implements:
- PM (Parameswaran-Mandal)
- CL (Chiu-Lin)

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
) -> NDArray[np.float64]:
    """
    Chiu-Lin sharpening.

    RHS = div(eps * grad(psi) - psi * (1-psi) * n_hat)

    where n_hat = grad(psi) / |grad(psi)|

    Parameters
    ----------
    psi : NDArray
        Volume fraction field.
    dx : float
        Grid spacing.
    dt : float
        Time step for sharpening sub-step.
    eps_target : float
        Target interface thickness.
    strength : float
        Sharpening strength multiplier.
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
        grad_psi = _grad_periodic(psi, dx)
    else:
        grad_psi = _grad_nonperiodic(psi, dx, bc)

    abs_grad = np.abs(grad_psi) + eta

    # Unit normal (sign of gradient)
    n_hat = grad_psi / abs_grad

    # Flux: eps * grad(psi) - psi * (1-psi) * n_hat
    flux = eps_target * grad_psi - psi * (1 - psi) * n_hat

    # Divergence of flux
    if bc.bc_type == "periodic":
        rhs = _div_periodic(flux, dx)
    else:
        # Central difference for interior, one-sided for boundaries
        rhs = np.zeros_like(psi)
        rhs[1:-1] = (flux[2:] - flux[:-2]) / (2 * dx)
        rhs[0] = (flux[1] - flux[0]) / dx
        rhs[-1] = (flux[-1] - flux[-2]) / dx

    # Update
    psi_new = psi + dt * strength * rhs

    # Clip to [0, 1]
    return np.clip(psi_new, 0.0, 1.0)


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
) -> NDArray[np.float64]:
    """
    Chiu-Lin sharpening (2D).

    RHS = div(eps * grad(psi) - psi * (1-psi) * n_hat)

    where n_hat = grad(psi) / |grad(psi)|.

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
        Target interface thickness.
    strength : float
        Sharpening strength multiplier.
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

    # Unit normal vector
    nx = dfdx / abs_grad
    ny = dfdy / abs_grad

    # Flux components: eps * grad(psi) - psi * (1-psi) * n_hat
    flux_x = eps_target * dfdx - psi * (1 - psi) * nx
    flux_y = eps_target * dfdy - psi * (1 - psi) * ny

    # Divergence of flux
    if bc.bc_type == "periodic":
        rhs = _div_periodic_2d(flux_x, flux_y, dx, dy)
    else:
        rhs = _div_nonperiodic_2d(flux_x, flux_y, dx, dy, bc)

    # Update
    psi_new = psi + dt * strength * rhs

    # Clip to [0, 1]
    return np.clip(psi_new, 0.0, 1.0)

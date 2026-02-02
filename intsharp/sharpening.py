"""
Interface sharpening source terms.

Implements:
- PM (Parameswaran-Mandal)
- CL (Corrective flux / Chiu-Lin)

Applied post-step via operator splitting.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .boundary import BoundaryCondition
from .registry import register_sharpening


def _grad_periodic(f: NDArray[np.float64], dx: float) -> NDArray[np.float64]:
    """Central difference gradient with periodic BC."""
    return (np.roll(f, -1) - np.roll(f, 1)) / (2 * dx)


def _div_periodic(q: NDArray[np.float64], dx: float) -> NDArray[np.float64]:
    """Central difference divergence with periodic BC."""
    return (np.roll(q, -1) - np.roll(q, 1)) / (2 * dx)


def _grad_nonperiodic(
    f: NDArray[np.float64],
    dx: float,
    bc: BoundaryCondition,
) -> NDArray[np.float64]:
    """Central difference gradient with non-periodic BC."""
    grad = np.zeros_like(f)
    # Interior: central difference
    grad[1:-1] = (f[2:] - f[:-2]) / (2 * dx)
    # Boundaries: one-sided
    grad[0] = (f[1] - f[0]) / dx
    grad[-1] = (f[-1] - f[-2]) / dx
    return grad


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

    where K is a constant (typically large, e.g., 200 * something).

    This returns the updated psi after applying dt * strength * RHS.

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

    # PM sharpening coefficient (empirical)
    # K ~ 1 / (4 * eps^2) for tanh profiles
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
    Corrective flux / Chiu-Lin sharpening.

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

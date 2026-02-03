"""
Surface tension diagnostic computations.

Computes curvature, interface normal, and CSF (Continuum Surface Force) fields
from a volume fraction field. These are diagnostic outputs; velocity remains prescribed.

Mathematical formulation:
    Gradient:        grad_alpha = (dalpha/dx, dalpha/dy)
    Gradient magnitude: |grad_alpha| = sqrt((dalpha/dx)^2 + (dalpha/dy)^2)
    Unit normal:     n = grad_alpha / (|grad_alpha| + eta)
    Curvature:       kappa = -div(n) = -(dn_x/dx + dn_y/dy)
    CSF force:       F = sigma * kappa * grad_alpha

Brackbill-style smoothing (optional):
    alpha_smooth = Gaussian_blur(alpha)
    kappa and grad computed from alpha_smooth
    F = sigma * kappa_smooth * grad_alpha_smooth
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from .boundary import BoundaryCondition


# ---------------------------------------------------------------------------
# Gaussian smoothing (Brackbill-style auxiliary field)
# ---------------------------------------------------------------------------

def gaussian_smooth_2d(
    f: NDArray[np.float64],
    sigma: float,
    bc: BoundaryCondition,
) -> NDArray[np.float64]:
    """
    Apply Gaussian smoothing to a 2D field.

    Uses periodic boundary conditions (mode='wrap') for periodic BC,
    otherwise mode='reflect'.

    Parameters
    ----------
    f : NDArray
        2D field (shape: ny, nx).
    sigma : float
        Gaussian sigma in grid cells (e.g., 2.0 = 2 cells).
    bc : BoundaryCondition
        Boundary condition (affects filter mode).

    Returns
    -------
    NDArray
        Smoothed field (same shape as f).
    """
    mode = "wrap" if bc.bc_type == "periodic" else "reflect"
    return gaussian_filter(f, sigma=(sigma, sigma), mode=mode)


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
    dfdx = (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2 * dx)
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
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Central difference gradient with non-periodic BC (2D).
    Uses one-sided differences at boundaries.
    
    Returns (df/dx, df/dy).
    """
    ny, nx = f.shape
    
    # df/dx
    dfdx = np.zeros_like(f)
    dfdx[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2 * dx)
    dfdx[:, 0] = (f[:, 1] - f[:, 0]) / dx
    dfdx[:, -1] = (f[:, -1] - f[:, -2]) / dx
    
    # df/dy
    dfdy = np.zeros_like(f)
    dfdy[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2 * dy)
    dfdy[0, :] = (f[1, :] - f[0, :]) / dy
    dfdy[-1, :] = (f[-1, :] - f[-2, :]) / dy
    
    return dfdx, dfdy


def _div_nonperiodic_2d(
    qx: NDArray[np.float64],
    qy: NDArray[np.float64],
    dx: float,
    dy: float,
) -> NDArray[np.float64]:
    """
    Central difference divergence with non-periodic BC (2D).
    Uses one-sided differences at boundaries.
    
    div(q) = dqx/dx + dqy/dy
    """
    ny, nx = qx.shape
    
    # dqx/dx
    dqx_dx = np.zeros_like(qx)
    dqx_dx[:, 1:-1] = (qx[:, 2:] - qx[:, :-2]) / (2 * dx)
    dqx_dx[:, 0] = (qx[:, 1] - qx[:, 0]) / dx
    dqx_dx[:, -1] = (qx[:, -1] - qx[:, -2]) / dx
    
    # dqy/dy
    dqy_dy = np.zeros_like(qy)
    dqy_dy[1:-1, :] = (qy[2:, :] - qy[:-2, :]) / (2 * dy)
    dqy_dy[0, :] = (qy[1, :] - qy[0, :]) / dy
    dqy_dy[-1, :] = (qy[-1, :] - qy[-2, :]) / dy
    
    return dqx_dx + dqy_dy


# ---------------------------------------------------------------------------
# Surface Tension Computations
# ---------------------------------------------------------------------------

def compute_gradient_2d(
    alpha: NDArray[np.float64],
    dx: float,
    dy: float,
    bc: BoundaryCondition,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute gradient of alpha field using central differences.
    
    Parameters
    ----------
    alpha : NDArray
        Volume fraction field (shape: ny, nx).
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.
    bc : BoundaryCondition
        Boundary condition.
    
    Returns
    -------
    Tuple[NDArray, NDArray]
        (dalpha/dx, dalpha/dy) gradient components.
    """
    if bc.bc_type == "periodic":
        return _grad_periodic_2d(alpha, dx, dy)
    else:
        return _grad_nonperiodic_2d(alpha, dx, dy)


def compute_normal_2d(
    alpha: NDArray[np.float64],
    dx: float,
    dy: float,
    bc: BoundaryCondition,
    eta: float = 1e-8,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute unit normal vector from volume fraction field.
    
    n = grad_alpha / |grad_alpha|
    
    Parameters
    ----------
    alpha : NDArray
        Volume fraction field (shape: ny, nx).
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.
    bc : BoundaryCondition
        Boundary condition.
    eta : float
        Small regularization constant to avoid division by zero.
    
    Returns
    -------
    Tuple[NDArray, NDArray]
        (normal_x, normal_y) unit normal components.
    """
    grad_x, grad_y = compute_gradient_2d(alpha, dx, dy, bc)
    
    # Gradient magnitude with regularization
    grad_mag = np.sqrt(grad_x**2 + grad_y**2) + eta
    
    # Unit normal
    normal_x = grad_x / grad_mag
    normal_y = grad_y / grad_mag
    
    return normal_x, normal_y


def compute_curvature_2d(
    alpha: NDArray[np.float64],
    dx: float,
    dy: float,
    bc: BoundaryCondition,
    eta: float = 1e-8,
) -> NDArray[np.float64]:
    """
    Compute mean curvature from volume fraction field.
    
    kappa = -div(n) where n = grad_alpha / |grad_alpha|
    
    Parameters
    ----------
    alpha : NDArray
        Volume fraction field (shape: ny, nx).
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.
    bc : BoundaryCondition
        Boundary condition.
    eta : float
        Small regularization constant to avoid division by zero.
    
    Returns
    -------
    NDArray
        Curvature field (shape: ny, nx).
    """
    # Compute unit normal
    normal_x, normal_y = compute_normal_2d(alpha, dx, dy, bc, eta)
    
    # Compute divergence of normal
    if bc.bc_type == "periodic":
        div_n = _div_periodic_2d(normal_x, normal_y, dx, dy)
    else:
        div_n = _div_nonperiodic_2d(normal_x, normal_y, dx, dy)
    
    # Curvature is negative divergence of normal
    kappa = -div_n
    
    return kappa


def compute_csf_force_2d(
    alpha: NDArray[np.float64],
    kappa: NDArray[np.float64],
    sigma: float,
    dx: float,
    dy: float,
    bc: BoundaryCondition,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Compute CSF (Continuum Surface Force) volume force.
    
    F = sigma * kappa * grad_alpha
    
    Parameters
    ----------
    alpha : NDArray
        Volume fraction field (shape: ny, nx).
    kappa : NDArray
        Curvature field (shape: ny, nx).
    sigma : float
        Surface tension coefficient.
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.
    bc : BoundaryCondition
        Boundary condition.
    
    Returns
    -------
    Tuple[NDArray, NDArray]
        (csf_x, csf_y) CSF force components.
    """
    grad_x, grad_y = compute_gradient_2d(alpha, dx, dy, bc)
    
    # CSF force: F = sigma * kappa * grad_alpha
    csf_x = sigma * kappa * grad_x
    csf_y = sigma * kappa * grad_y
    
    return csf_x, csf_y


def compute_surface_tension_diagnostics_2d(
    alpha: NDArray[np.float64],
    sigma: float,
    dx: float,
    dy: float,
    bc: BoundaryCondition,
    eta: float = 1e-8,
    smoothing_sigma: float | None = None,
    interface_band_alpha_min: float | None = None,
    interface_band_alpha_max: float | None = None,
) -> dict[str, NDArray[np.float64]]:
    """
    Compute all surface tension diagnostic fields.

    When smoothing_sigma is set, uses Brackbill-style auxiliary smoothed field:
    alpha_smooth = Gaussian_blur(alpha), then kappa and grad from alpha_smooth,
    F = sigma * kappa_smooth * grad_alpha_smooth. Reduces grid-alignment artifacts.

    When interface_band_alpha_min/max are set, zeros out diagnostics outside
    the band (alpha_min <= alpha <= alpha_max) using raw alpha. Restricts
    surface force to a finite volume enclosing the interface.

    Parameters
    ----------
    alpha : NDArray
        Volume fraction field (shape: ny, nx).
    sigma : float
        Surface tension coefficient.
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.
    bc : BoundaryCondition
        Boundary condition.
    eta : float
        Small regularization constant to avoid division by zero.
    smoothing_sigma : float or None
        If set, Gaussian smoothing sigma (in grid cells) for Brackbill-style
        auxiliary field. None = no smoothing (raw alpha).
    interface_band_alpha_min : float or None
        If set with interface_band_alpha_max, mask to alpha >= this (raw alpha).
    interface_band_alpha_max : float or None
        If set with interface_band_alpha_min, mask to alpha <= this (raw alpha).

    Returns
    -------
    dict[str, NDArray]
        Dictionary with keys: 'kappa', 'normal_x', 'normal_y', 'csf_x', 'csf_y'.
    """
    # Optional Brackbill-style smoothing
    if smoothing_sigma is not None and smoothing_sigma > 0:
        alpha_work = gaussian_smooth_2d(alpha, smoothing_sigma, bc)
    else:
        alpha_work = alpha

    # Compute normal and curvature from (possibly smoothed) field
    normal_x, normal_y = compute_normal_2d(alpha_work, dx, dy, bc, eta)

    if bc.bc_type == "periodic":
        div_n = _div_periodic_2d(normal_x, normal_y, dx, dy)
    else:
        div_n = _div_nonperiodic_2d(normal_x, normal_y, dx, dy)
    kappa = -div_n

    # CSF force: F = sigma * kappa * grad_alpha (all from smoothed field)
    grad_x, grad_y = compute_gradient_2d(alpha_work, dx, dy, bc)
    csf_x = sigma * kappa * grad_x
    csf_y = sigma * kappa * grad_y

    # Interface band mask: zero diagnostics outside (alpha_min, alpha_max) using raw alpha
    if (
        interface_band_alpha_min is not None
        and interface_band_alpha_max is not None
    ):
        mask = (alpha >= interface_band_alpha_min) & (alpha <= interface_band_alpha_max)
        kappa = np.where(mask, kappa, 0.0)
        normal_x = np.where(mask, normal_x, 0.0)
        normal_y = np.where(mask, normal_y, 0.0)
        csf_x = np.where(mask, csf_x, 0.0)
        csf_y = np.where(mask, csf_y, 0.0)

    return {
        "kappa": kappa,
        "normal_x": normal_x,
        "normal_y": normal_y,
        "csf_x": csf_x,
        "csf_y": csf_y,
    }

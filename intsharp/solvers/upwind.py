"""
First-order upwind advection scheme (1D and 2D).

Supports both constant velocity (scalar) and spatially-varying velocity (arrays).
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np
from numpy.typing import NDArray

from ..boundary import (
    BoundaryCondition,
    get_ghost_values,
    get_ghost_values_2d_x,
    get_ghost_values_2d_y,
)
from ..registry import register_solver


@register_solver("upwind")
def upwind_advect(
    field_values: NDArray[np.float64],
    velocity: float,
    dx: float,
    dt: float,
    bc: BoundaryCondition,
) -> NDArray[np.float64]:
    """
    Compute one step of first-order upwind advection.

    Solves: ∂f/∂t + u ∂f/∂x = 0

    Notes
    -----
    First-order upwind has a numerical phase (dispersion) error: the numerical
    wave travels slightly slower than the exact speed, so the profile lags
    (e.g. drifts left for u > 0) over many revolutions. This is expected, not
    a bug. For 1D periodic advection, CFL = 1 gives an exact integer shift
    per step and no phase error. The 1D domain is cell-centered with
    dx = L/n_points; use dt = dx/|u| and n_points steps per revolution.

    Parameters
    ----------
    field_values : NDArray
        Current field values.
    velocity : float
        Advection velocity (1D, scalar).
    dx : float
        Grid spacing.
    dt : float
        Time step.
    bc : BoundaryCondition
        Boundary condition.

    Returns
    -------
    NDArray
        Updated field values after one advection step.
    """
    n = len(field_values)
    f_new = np.empty_like(field_values)

    # Get ghost values for boundary handling
    left_ghost, right_ghost = get_ghost_values(field_values, bc, dx)

    # CFL number
    cfl = velocity * dt / dx

    if velocity >= 0:
        # Upwind from the left: f[i] - cfl * (f[i] - f[i-1])
        for i in range(n):
            if i == 0:
                f_left = left_ghost
            else:
                f_left = field_values[i - 1]
            f_new[i] = field_values[i] - cfl * (field_values[i] - f_left)
    else:
        # Upwind from the right: f[i] - cfl * (f[i+1] - f[i])
        for i in range(n):
            if i == n - 1:
                f_right = right_ghost
            else:
                f_right = field_values[i + 1]
            f_new[i] = field_values[i] - cfl * (f_right - field_values[i])

    return f_new


@register_solver("upwind_vectorized")
def upwind_advect_vectorized(
    field_values: NDArray[np.float64],
    velocity: float,
    dx: float,
    dt: float,
    bc: BoundaryCondition,
) -> NDArray[np.float64]:
    """
    Vectorized first-order upwind advection (faster for large grids).

    Same as upwind_advect but uses numpy vectorization.
    """
    cfl = velocity * dt / dx

    if bc.bc_type == "periodic":
        if velocity >= 0:
            # f[i] - cfl * (f[i] - f[i-1])
            return field_values - cfl * (field_values - np.roll(field_values, 1))
        else:
            # f[i] - cfl * (f[i+1] - f[i])
            return field_values - cfl * (np.roll(field_values, -1) - field_values)
    else:
        # Non-periodic: use ghost values
        left_ghost, right_ghost = get_ghost_values(field_values, bc, dx)
        
        if velocity >= 0:
            f_left = np.concatenate([[left_ghost], field_values[:-1]])
            return field_values - cfl * (field_values - f_left)
        else:
            f_right = np.concatenate([field_values[1:], [right_ghost]])
            return field_values - cfl * (f_right - field_values)


# ---------------------------------------------------------------------------
# 2D Upwind Advection
# ---------------------------------------------------------------------------


@register_solver("upwind_2d")
def upwind_advect_2d(
    field_values: NDArray[np.float64],
    velocity: Union[Tuple[float, float], Tuple[NDArray[np.float64], NDArray[np.float64]]],
    dx: float,
    dy: float,
    dt: float,
    bc: BoundaryCondition,
) -> NDArray[np.float64]:
    """
    Compute one step of first-order upwind advection in 2D.

    Solves: ∂f/∂t + u ∂f/∂x + v ∂f/∂y = 0

    Uses dimension-by-dimension splitting:
    1. First advect in x: f* = f - (u*dt/dx) * (f - f_upwind_x)
    2. Then advect in y: f** = f* - (v*dt/dy) * (f* - f*_upwind_y)

    Supports both constant velocity (scalars) and spatially-varying velocity (arrays).
    For spatially-varying velocity, each cell uses its local velocity for upwinding.

    Parameters
    ----------
    field_values : NDArray
        Current field values (shape: ny, nx).
    velocity : Tuple[float, float] or Tuple[NDArray, NDArray]
        Advection velocity (u, v). Can be scalars (constant) or 2D arrays (varying).
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.
    dt : float
        Time step.
    bc : BoundaryCondition
        Boundary condition (applied to all edges).

    Returns
    -------
    NDArray
        Updated field values after one advection step.
    """
    u, v = velocity
    f = field_values
    ny, nx = f.shape

    # Convert scalars to arrays for uniform handling
    u_is_array = isinstance(u, np.ndarray)
    v_is_array = isinstance(v, np.ndarray)

    if not u_is_array:
        u_arr = np.full((ny, nx), u, dtype=np.float64)
    else:
        u_arr = u

    if not v_is_array:
        v_arr = np.full((ny, nx), v, dtype=np.float64)
    else:
        v_arr = v

    # --- X-direction advection (cell-by-cell upwinding) ---
    cfl_x = u_arr * dt / dx

    if bc.bc_type == "periodic":
        # Get upwind neighbors
        f_left = np.roll(f, 1, axis=1)   # f[j, i-1]
        f_right = np.roll(f, -1, axis=1)  # f[j, i+1]
    else:
        # Non-periodic: use ghost values
        left_ghost, right_ghost = get_ghost_values_2d_x(field_values, bc, dx)
        f_left = np.empty_like(f)
        f_left[:, 1:] = f[:, :-1]
        f_left[:, 0] = left_ghost
        f_right = np.empty_like(f)
        f_right[:, :-1] = f[:, 1:]
        f_right[:, -1] = right_ghost

    # Upwind based on sign of u at each cell
    # u >= 0: upwind from left, u < 0: upwind from right
    u_pos = u_arr >= 0
    f = np.where(
        u_pos,
        f - cfl_x * (f - f_left),
        f - cfl_x * (f_right - f)
    )

    # --- Y-direction advection (cell-by-cell upwinding) ---
    if dy > 0:
        cfl_y = v_arr * dt / dy

        if bc.bc_type == "periodic":
            f_bottom = np.roll(f, 1, axis=0)   # f[j-1, i]
            f_top = np.roll(f, -1, axis=0)     # f[j+1, i]
        else:
            bottom_ghost, top_ghost = get_ghost_values_2d_y(field_values, bc, dy)
            f_bottom = np.empty_like(f)
            f_bottom[1:, :] = f[:-1, :]
            f_bottom[0, :] = bottom_ghost
            f_top = np.empty_like(f)
            f_top[:-1, :] = f[1:, :]
            f_top[-1, :] = top_ghost

        # Upwind based on sign of v at each cell
        v_pos = v_arr >= 0
        f = np.where(
            v_pos,
            f - cfl_y * (f - f_bottom),
            f - cfl_y * (f_top - f)
        )

    return f

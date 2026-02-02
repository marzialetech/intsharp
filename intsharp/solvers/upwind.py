"""
First-order upwind advection scheme (1D and 2D).
"""

from __future__ import annotations

from typing import Tuple

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
    velocity: Tuple[float, float],
    dx: float,
    dy: float,
    dt: float,
    bc: BoundaryCondition,
) -> NDArray[np.float64]:
    """
    Compute one step of first-order upwind advection in 2D.

    Solves: ∂f/∂t + u ∂f/∂x + v ∂f/∂y = 0

    Uses dimension-by-dimension splitting:
    1. First advect in x: f* = f - cfl_x * (f - f_upwind_x)
    2. Then advect in y: f** = f* - cfl_y * (f* - f*_upwind_y)

    Parameters
    ----------
    field_values : NDArray
        Current field values (shape: ny, nx).
    velocity : Tuple[float, float]
        Advection velocity (u, v).
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

    # --- X-direction advection ---
    cfl_x = u * dt / dx
    if bc.bc_type == "periodic":
        if u >= 0:
            # Upwind from the left (i-1): roll by +1 along axis=1
            f = f - cfl_x * (f - np.roll(f, 1, axis=1))
        else:
            # Upwind from the right (i+1): roll by -1 along axis=1
            f = f - cfl_x * (np.roll(f, -1, axis=1) - f)
    else:
        # Non-periodic: use ghost values
        left_ghost, right_ghost = get_ghost_values_2d_x(field_values, bc, dx)
        if u >= 0:
            f_left = np.empty_like(f)
            f_left[:, 1:] = f[:, :-1]
            f_left[:, 0] = left_ghost
            f = f - cfl_x * (f - f_left)
        else:
            f_right = np.empty_like(f)
            f_right[:, :-1] = f[:, 1:]
            f_right[:, -1] = right_ghost
            f = f - cfl_x * (f_right - f)

    # --- Y-direction advection ---
    cfl_y = v * dt / dy if dy > 0 else 0.0
    if abs(v) > 1e-14 and dy > 0:
        if bc.bc_type == "periodic":
            if v >= 0:
                # Upwind from the bottom (j-1): roll by +1 along axis=0
                f = f - cfl_y * (f - np.roll(f, 1, axis=0))
            else:
                # Upwind from the top (j+1): roll by -1 along axis=0
                f = f - cfl_y * (np.roll(f, -1, axis=0) - f)
        else:
            # Non-periodic: use ghost values
            bottom_ghost, top_ghost = get_ghost_values_2d_y(field_values, bc, dy)
            if v >= 0:
                f_bottom = np.empty_like(f)
                f_bottom[1:, :] = f[:-1, :]
                f_bottom[0, :] = bottom_ghost
                f = f - cfl_y * (f - f_bottom)
            else:
                f_top = np.empty_like(f)
                f_top[:-1, :] = f[1:, :]
                f_top[-1, :] = top_ghost
                f = f - cfl_y * (f_top - f)

    return f

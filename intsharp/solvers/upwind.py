"""
First-order upwind advection scheme.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..boundary import BoundaryCondition, get_ghost_values
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

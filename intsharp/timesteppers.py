"""
Time integration methods.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from .registry import register_timestepper


# Type alias for RHS function: (field_values, t) -> d(field)/dt
RHSFunction = Callable[[NDArray[np.float64], float], NDArray[np.float64]]


@register_timestepper("euler")
def euler_step(
    field_values: NDArray[np.float64],
    t: float,
    dt: float,
    rhs_fn: RHSFunction,
) -> NDArray[np.float64]:
    """
    Forward Euler time step.

    f^{n+1} = f^n + dt * rhs(f^n, t^n)

    Parameters
    ----------
    field_values : NDArray
        Current field values.
    t : float
        Current time.
    dt : float
        Time step.
    rhs_fn : Callable
        Function computing the RHS (negative of advection term + any sources).

    Returns
    -------
    NDArray
        Updated field values.
    """
    return field_values + dt * rhs_fn(field_values, t)


@register_timestepper("rk4")
def rk4_step(
    field_values: NDArray[np.float64],
    t: float,
    dt: float,
    rhs_fn: RHSFunction,
) -> NDArray[np.float64]:
    """
    4th-order Runge-Kutta time step.

    Parameters
    ----------
    field_values : NDArray
        Current field values.
    t : float
        Current time.
    dt : float
        Time step.
    rhs_fn : Callable
        Function computing the RHS.

    Returns
    -------
    NDArray
        Updated field values.
    """
    k1 = rhs_fn(field_values, t)
    k2 = rhs_fn(field_values + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = rhs_fn(field_values + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = rhs_fn(field_values + dt * k3, t + dt)

    return field_values + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

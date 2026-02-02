"""
Boundary condition implementations.

Supports:
- Periodic: wraps values around
- Neumann: specified gradient at boundaries
- Dirichlet: specified values at boundaries
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .config import BoundaryConfig


@dataclass
class BoundaryCondition:
    """
    Boundary condition handler.

    Attributes
    ----------
    bc_type : str
        Type of BC: "periodic", "neumann", or "dirichlet".
    gradient_left : float or None
        Gradient at left boundary (Neumann).
    gradient_right : float or None
        Gradient at right boundary (Neumann).
    value_left : float or None
        Value at left boundary (Dirichlet).
    value_right : float or None
        Value at right boundary (Dirichlet).
    """
    bc_type: Literal["periodic", "neumann", "dirichlet"]
    gradient_left: float | None = None
    gradient_right: float | None = None
    value_left: float | None = None
    value_right: float | None = None


def create_bc(config: BoundaryConfig) -> BoundaryCondition:
    """Create a BoundaryCondition from config."""
    return BoundaryCondition(
        bc_type=config.type,
        gradient_left=config.gradient_left,
        gradient_right=config.gradient_right,
        value_left=config.value_left,
        value_right=config.value_right,
    )


def apply_bc(
    field: NDArray[np.float64],
    bc: BoundaryCondition,
    dx: float,
) -> NDArray[np.float64]:
    """
    Apply boundary conditions to a field (in-place modification).

    For periodic: no explicit action needed during advection (handled by roll).
    For Neumann/Dirichlet: sets boundary values after advection step.

    Parameters
    ----------
    field : NDArray
        The field values (modified in-place).
    bc : BoundaryCondition
        The boundary condition.
    dx : float
        Grid spacing.

    Returns
    -------
    NDArray
        The field with BCs applied.
    """
    if bc.bc_type == "periodic":
        # Periodic BCs are handled implicitly via np.roll in solvers
        pass

    elif bc.bc_type == "neumann":
        # Neumann: set boundary values based on gradient
        # f[0] = f[1] - gradient_left * dx
        # f[-1] = f[-2] + gradient_right * dx
        assert bc.gradient_left is not None
        assert bc.gradient_right is not None
        field[0] = field[1] - bc.gradient_left * dx
        field[-1] = field[-2] + bc.gradient_right * dx

    elif bc.bc_type == "dirichlet":
        # Dirichlet: set boundary values directly
        assert bc.value_left is not None
        assert bc.value_right is not None
        field[0] = bc.value_left
        field[-1] = bc.value_right

    return field


def get_ghost_values(
    field: NDArray[np.float64],
    bc: BoundaryCondition,
    dx: float,
) -> tuple[float, float]:
    """
    Get ghost cell values for left and right boundaries.

    Used by solvers that need values "outside" the domain.

    Parameters
    ----------
    field : NDArray
        The field values.
    bc : BoundaryCondition
        The boundary condition.
    dx : float
        Grid spacing.

    Returns
    -------
    tuple[float, float]
        (left_ghost, right_ghost) values.
    """
    if bc.bc_type == "periodic":
        return field[-1], field[0]

    elif bc.bc_type == "neumann":
        assert bc.gradient_left is not None
        assert bc.gradient_right is not None
        left_ghost = field[0] - bc.gradient_left * dx
        right_ghost = field[-1] + bc.gradient_right * dx
        return left_ghost, right_ghost

    elif bc.bc_type == "dirichlet":
        assert bc.value_left is not None
        assert bc.value_right is not None
        # Extrapolate ghost to maintain boundary value
        left_ghost = 2 * bc.value_left - field[0]
        right_ghost = 2 * bc.value_right - field[-1]
        return left_ghost, right_ghost

    else:
        raise ValueError(f"Unknown BC type: {bc.bc_type}")

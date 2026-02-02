"""
Field containers with expression-based initial condition evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Any, Union

import numpy as np
from numpy.typing import NDArray

from .boundary import BoundaryCondition, create_bc
from .config import FieldConfig
from .domain import Domain1D, Domain2D, Domain


# ---------------------------------------------------------------------------
# Safe expression evaluation
# ---------------------------------------------------------------------------

# Whitelist of allowed names in IC expressions
SAFE_NAMESPACE: dict[str, Any] = {
    # Math functions
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "exp": np.exp,
    "log": np.log,
    "log10": np.log10,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "tanh": np.tanh,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "floor": np.floor,
    "ceil": np.ceil,
    "sign": np.sign,
    "heaviside": np.heaviside,
    # Constants
    "pi": np.pi,
    "e": np.e,
    # NumPy
    "np": np,
    # Builtins needed for expressions
    "min": np.minimum,
    "max": np.maximum,
}


def evaluate_expression_1d(
    expr: str,
    x: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Safely evaluate a math expression with 'x' as the spatial coordinate (1D).

    Parameters
    ----------
    expr : str
        Expression string (e.g., "0.5 * (1 + tanh((0.025 - abs(x)) / 0.002))").
    x : NDArray
        1D grid coordinates.

    Returns
    -------
    NDArray
        Evaluated values at each grid point.

    Raises
    ------
    ValueError
        If the expression is invalid or uses disallowed functions.
    """
    # Create namespace with x
    namespace = SAFE_NAMESPACE.copy()
    namespace["x"] = x

    try:
        result = eval(expr, {"__builtins__": {}}, namespace)
    except Exception as e:
        raise ValueError(
            f"Failed to evaluate initial condition expression: {expr!r}\n"
            f"Error: {e}"
        ) from e

    # Ensure result is an array of the right shape
    result = np.asarray(result, dtype=np.float64)
    if result.shape != x.shape:
        # Handle scalar expressions (constant IC)
        if result.ndim == 0:
            result = np.full_like(x, result)
        else:
            raise ValueError(
                f"IC expression result has shape {result.shape}, "
                f"expected {x.shape}"
            )

    return result


def evaluate_expression_2d(
    expr: str,
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Safely evaluate a math expression with 'x', 'y', and 'r' as spatial coordinates (2D).

    Parameters
    ----------
    expr : str
        Expression string (e.g., "0.5 * (1 + tanh((0.15 - r) / 0.02))").
    X : NDArray
        2D meshgrid of x coordinates (shape: ny, nx).
    Y : NDArray
        2D meshgrid of y coordinates (shape: ny, nx).

    Returns
    -------
    NDArray
        Evaluated values at each grid point (shape: ny, nx).

    Raises
    ------
    ValueError
        If the expression is invalid or uses disallowed functions.
    """
    # Create namespace with x, y, and r
    namespace = SAFE_NAMESPACE.copy()
    namespace["x"] = X
    namespace["y"] = Y
    namespace["r"] = np.sqrt(X**2 + Y**2)

    try:
        result = eval(expr, {"__builtins__": {}}, namespace)
    except Exception as e:
        raise ValueError(
            f"Failed to evaluate initial condition expression: {expr!r}\n"
            f"Error: {e}"
        ) from e

    # Ensure result is an array of the right shape
    result = np.asarray(result, dtype=np.float64)
    if result.shape != X.shape:
        # Handle scalar expressions (constant IC)
        if result.ndim == 0:
            result = np.full_like(X, result)
        else:
            raise ValueError(
                f"IC expression result has shape {result.shape}, "
                f"expected {X.shape}"
            )

    return result


# ---------------------------------------------------------------------------
# Field container
# ---------------------------------------------------------------------------

@dataclass
class Field:
    """
    A scalar field on the 1D or 2D domain.

    Attributes
    ----------
    name : str
        Field name.
    values : NDArray[np.float64]
        Current field values. Shape is (n_points,) for 1D, (ny, nx) for 2D.
    bc : BoundaryCondition
        Boundary condition for this field.
    sharpening_enabled : bool or None
        Per-field sharpening override. None means use global setting.
    """
    name: str
    values: NDArray[np.float64]
    bc: BoundaryCondition
    sharpening_enabled: bool | None = None

    def copy(self) -> "Field":
        """Return a copy of this field."""
        return Field(
            name=self.name,
            values=self.values.copy(),
            bc=self.bc,
            sharpening_enabled=self.sharpening_enabled,
        )


def create_field(config: FieldConfig, domain: Domain) -> Field:
    """
    Create a field from configuration and domain.

    Parameters
    ----------
    config : FieldConfig
        Field configuration.
    domain : Domain1D or Domain2D
        The computational domain.

    Returns
    -------
    Field
        Initialized field.
    """
    # Evaluate initial condition based on domain dimension
    if isinstance(domain, Domain2D):
        values = evaluate_expression_2d(config.initial_condition, domain.X, domain.Y)
    else:
        values = evaluate_expression_1d(config.initial_condition, domain.x)

    # Create boundary condition
    bc = create_bc(config.boundary)

    return Field(
        name=config.name,
        values=values,
        bc=bc,
        sharpening_enabled=config.sharpening,
    )


def create_fields(
    configs: list[FieldConfig],
    domain: Domain,
) -> dict[str, Field]:
    """
    Create all fields from configuration.

    Parameters
    ----------
    configs : list[FieldConfig]
        List of field configurations.
    domain : Domain1D or Domain2D
        The computational domain.

    Returns
    -------
    dict[str, Field]
        Dictionary mapping field names to Field objects.
    """
    fields = {}
    for cfg in configs:
        fields[cfg.name] = create_field(cfg, domain)
    return fields

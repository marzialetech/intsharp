"""
1D and 2D domain and grid setup.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
from numpy.typing import NDArray

from .config import DomainConfig


@dataclass
class Domain1D:
    """
    1D computational domain.

    Attributes
    ----------
    x : NDArray[np.float64]
        Grid point coordinates.
    dx : float
        Grid spacing.
    n_points : int
        Number of grid points.
    x_min : float
        Left boundary.
    x_max : float
        Right boundary.
    """
    x: NDArray[np.float64]
    dx: float
    n_points: int
    x_min: float
    x_max: float

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions."""
        return 1

    @property
    def L(self) -> float:
        """Domain length."""
        return self.x_max - self.x_min


@dataclass
class Domain2D:
    """
    2D computational domain.

    Attributes
    ----------
    x : NDArray[np.float64]
        1D array of x coordinates.
    y : NDArray[np.float64]
        1D array of y coordinates.
    X : NDArray[np.float64]
        2D meshgrid of x coordinates (shape: ny, nx).
    Y : NDArray[np.float64]
        2D meshgrid of y coordinates (shape: ny, nx).
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.
    nx : int
        Number of grid points in x.
    ny : int
        Number of grid points in y.
    x_min : float
        Left boundary.
    x_max : float
        Right boundary.
    y_min : float
        Bottom boundary.
    y_max : float
        Top boundary.
    """
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    X: NDArray[np.float64]
    Y: NDArray[np.float64]
    dx: float
    dy: float
    nx: int
    ny: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions."""
        return 2

    @property
    def Lx(self) -> float:
        """Domain length in x."""
        return self.x_max - self.x_min

    @property
    def Ly(self) -> float:
        """Domain length in y."""
        return self.y_max - self.y_min


# Type alias for domain
Domain = Union[Domain1D, Domain2D]


def create_domain(config: DomainConfig) -> Domain:
    """
    Create a 1D or 2D domain from configuration.

    Parameters
    ----------
    config : DomainConfig
        Domain configuration.

    Returns
    -------
    Domain1D or Domain2D
        The computational domain.
    """
    if config.ndim == 1:
        # 1D domain
        n = config.nx
        x = np.linspace(config.x_min, config.x_max, n)
        dx = x[1] - x[0] if n > 1 else config.x_max - config.x_min

        return Domain1D(
            x=x,
            dx=dx,
            n_points=n,
            x_min=config.x_min,
            x_max=config.x_max,
        )
    else:
        # 2D domain
        nx = config.nx
        ny = config.ny
        x = np.linspace(config.x_min, config.x_max, nx)
        y = np.linspace(config.y_min, config.y_max, ny)  # type: ignore
        dx = x[1] - x[0] if nx > 1 else config.x_max - config.x_min
        dy = y[1] - y[0] if ny > 1 else config.y_max - config.y_min  # type: ignore

        # Create meshgrid (X, Y) with indexing='xy' for standard (ny, nx) shape
        X, Y = np.meshgrid(x, y, indexing="xy")

        return Domain2D(
            x=x,
            y=y,
            X=X,
            Y=Y,
            dx=dx,
            dy=dy,
            nx=nx,
            ny=ny,
            x_min=config.x_min,
            x_max=config.x_max,
            y_min=config.y_min,  # type: ignore
            y_max=config.y_max,  # type: ignore
        )

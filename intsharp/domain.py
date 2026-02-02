"""
1D domain and grid setup.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    def L(self) -> float:
        """Domain length."""
        return self.x_max - self.x_min


def create_domain(config: DomainConfig) -> Domain1D:
    """
    Create a 1D domain from configuration.

    Parameters
    ----------
    config : DomainConfig
        Domain configuration.

    Returns
    -------
    Domain1D
        The computational domain.
    """
    x = np.linspace(config.x_min, config.x_max, config.n_points)
    dx = x[1] - x[0] if config.n_points > 1 else config.x_max - config.x_min

    return Domain1D(
        x=x,
        dx=dx,
        n_points=config.n_points,
        x_min=config.x_min,
        x_max=config.x_max,
    )

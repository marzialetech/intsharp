"""
Spatial discretization solvers for advection and compressible flow.

Import all solvers here to register them automatically.
"""

from . import upwind
from . import euler_1d
from . import euler_5eq_1d

__all__ = ["upwind", "euler_1d"]

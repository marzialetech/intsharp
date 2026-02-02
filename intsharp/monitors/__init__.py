"""
Output monitors for simulation results.

Import all monitors here to register them automatically.
"""

from . import base, console, curve, gif, hdf5, image, txt

__all__ = ["base", "console", "curve", "gif", "hdf5", "image", "txt"]

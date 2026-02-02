"""
Output monitors for simulation results.

Import all monitors here to register them automatically.
"""

from . import base, console, contour_gif, curve, gif, hdf5, image, txt

__all__ = ["base", "console", "contour_gif", "curve", "gif", "hdf5", "image", "txt"]

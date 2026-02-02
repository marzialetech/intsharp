"""
HDF5 data output monitor.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..registry import register_monitor
from .base import Monitor

if TYPE_CHECKING:
    from ..domain import Domain1D
    from ..fields import Field


@register_monitor("hdf5")
class HDF5Monitor(Monitor):
    """
    HDF5 data output.

    Saves field data at specified intervals.
    """

    def __init__(
        self,
        output_dir: Path,
        every_n_steps: int | None = None,
        at_times: list[float] | None = None,
        fields: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(output_dir, every_n_steps, at_times)
        self.field_names = fields or []
        self._h5file = None
        self._frame_count = 0

    def on_start(
        self,
        fields: dict[str, "Field"],
        domain: "Domain1D",
    ) -> None:
        """Create output directory and initialize HDF5 file."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "h5py is required for HDF5 output. Install with: pip install h5py"
            )

        filepath = self.output_dir / "simulation.h5"
        self._h5file = h5py.File(filepath, "w")

        # Store grid
        self._h5file.create_dataset("x", data=domain.x)
        self._h5file.attrs["n_points"] = domain.n_points
        self._h5file.attrs["x_min"] = domain.x_min
        self._h5file.attrs["x_max"] = domain.x_max
        self._h5file.attrs["dx"] = domain.dx

        # Create groups for each field
        for name in self.field_names:
            self._h5file.create_group(name)

    def on_step(
        self,
        step: int,
        t: float,
        fields: dict[str, "Field"],
        domain: "Domain1D",
    ) -> None:
        """Save field data if output is due."""
        if self._h5file is None:
            return

        dt = 0.001  # Approximate
        if not self.should_output(step, t, dt):
            return

        for name in self.field_names:
            if name not in fields:
                continue
            
            group = self._h5file[name]
            dataset_name = f"step_{self._frame_count:06d}"
            ds = group.create_dataset(dataset_name, data=fields[name].values)
            ds.attrs["step"] = step
            ds.attrs["time"] = t

        self._frame_count += 1

    def on_end(
        self,
        fields: dict[str, "Field"],
        domain: "Domain1D",
    ) -> None:
        """Close HDF5 file."""
        if self._h5file is not None:
            self._h5file.close()
            self._h5file = None

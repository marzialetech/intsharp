"""
Curve (.curve) output monitor.

Writes one .curve file per output time per field: two columns (x, value),
suitable for plotters and analysis tools that expect curve data.
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


@register_monitor("curve")
class CurveMonitor(Monitor):
    """
    Curve-format output.

    One .curve file per output time per field: two columns (x, value),
    no header, whitespace-separated. Filename: {field}_{frame:05d}.curve.
    """

    def __init__(
        self,
        output_dir: Path,
        every_n_steps: int | None = None,
        at_times: list[float] | None = None,
        field: str | None = None,
        fields: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(output_dir, every_n_steps, at_times)
        if field is not None:
            self.field_names = [field]
        elif fields is not None:
            self.field_names = list(fields)
        else:
            self.field_names = []
        self._frame_count = 0

    def on_start(
        self,
        fields: dict[str, "Field"],
        domain: "Domain1D",
    ) -> None:
        """Create output directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def on_step(
        self,
        step: int,
        t: float,
        fields: dict[str, "Field"],
        domain: "Domain1D",
    ) -> None:
        """Write .curve snapshot(s) if output is due."""
        dt = 0.001
        if not self.should_output(step, t, dt):
            return

        if not self.field_names:
            return

        x = domain.x
        for name in self.field_names:
            if name not in fields:
                raise KeyError(f"Field '{name}' not found for curve output")
            data = np.column_stack([x, fields[name].values])
            filename = f"{name}_{self._frame_count:05d}.curve"
            filepath = self.output_dir / filename
            np.savetxt(filepath, data, fmt="%.6e", delimiter="\t")

        self._frame_count += 1

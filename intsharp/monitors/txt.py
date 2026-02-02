"""
Plain-text (.txt) output monitor.

Writes columnar field data: one .txt file per output time with optional
header (step, t) and columns x, field values.
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


@register_monitor("txt")
class TxtMonitor(Monitor):
    """
    Plain-text columnar output.

    One .txt file per output time with header line (# step=... t=...)
    and columns: x, then requested field(s). Delimiter is whitespace.
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
        """Write .txt snapshot if output is due."""
        dt = 0.001
        if not self.should_output(step, t, dt):
            return

        if not self.field_names:
            return

        for name in self.field_names:
            if name not in fields:
                raise KeyError(f"Field '{name}' not found for txt output")

        x = domain.x
        header_parts = [f"# step={step}", f"t={t:.6e}"]
        col_names = ["x"] + self.field_names
        cols = [x] + [fields[name].values for name in self.field_names]
        data = np.column_stack(cols)

        filename = f"snapshot_{self._frame_count:05d}.txt"
        filepath = self.output_dir / filename
        with open(filepath, "w") as f:
            f.write(" ".join(header_parts) + "\n")
            f.write("# " + " ".join(col_names) + "\n")
            np.savetxt(f, data, fmt="%.6e", delimiter="\t")

        self._frame_count += 1

"""
Image output monitors (PNG, PDF, SVG).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from ..registry import register_monitor
from .base import Monitor

if TYPE_CHECKING:
    from ..domain import Domain1D
    from ..fields import Field


class ImageMonitor(Monitor):
    """
    Base class for image output (PNG or PDF).
    """

    def __init__(
        self,
        output_dir: Path,
        every_n_steps: int | None = None,
        at_times: list[float] | None = None,
        field: str | None = None,
        extension: str = "png",
        **kwargs,
    ):
        super().__init__(output_dir, every_n_steps, at_times)
        self.field_name = field
        self.extension = extension
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
        """Save image if output is due."""
        dt = 0.001  # Approximate dt for time check (will be passed properly in runner)
        if not self.should_output(step, t, dt):
            return

        if self.field_name is None:
            return

        if self.field_name not in fields:
            raise KeyError(f"Field '{self.field_name}' not found for image output")

        field = fields[self.field_name]

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(domain.x, field.values, "b-", linewidth=1.5)
        ax.set_xlabel("x")
        ax.set_ylabel(field.name)
        ax.set_title(f"{field.name} at t = {t:.4f}")
        ax.set_ylim(-0.1, 1.1)
        ax.grid(True, alpha=0.3)

        # Save
        filename = f"{self.field_name}_{self._frame_count:05d}.{self.extension}"
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)

        self._frame_count += 1


@register_monitor("png")
class PNGMonitor(ImageMonitor):
    """PNG image output."""

    def __init__(self, **kwargs):
        super().__init__(extension="png", **kwargs)


@register_monitor("pdf")
class PDFMonitor(ImageMonitor):
    """PDF image output."""

    def __init__(self, **kwargs):
        super().__init__(extension="pdf", **kwargs)


@register_monitor("svg")
class SVGMonitor(ImageMonitor):
    """SVG image output."""

    def __init__(self, **kwargs):
        super().__init__(extension="svg", **kwargs)

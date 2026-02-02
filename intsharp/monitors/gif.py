"""
Animated GIF output monitor.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from ..registry import register_monitor
from .base import Monitor

if TYPE_CHECKING:
    from ..domain import Domain1D
    from ..fields import Field


@register_monitor("gif")
class GIFMonitor(Monitor):
    """
    Animated GIF output.

    Accumulates frames during simulation and writes GIF at the end.
    """

    def __init__(
        self,
        output_dir: Path,
        every_n_steps: int | None = None,
        at_times: list[float] | None = None,
        field: str | None = None,
        fps: int = 10,
        **kwargs,
    ):
        super().__init__(output_dir, every_n_steps, at_times)
        self.field_name = field
        self.fps = fps
        self._frames: list[np.ndarray] = []
        self._times: list[float] = []
        self._domain_x: np.ndarray | None = None

    def on_start(
        self,
        fields: dict[str, "Field"],
        domain: "Domain1D",
    ) -> None:
        """Initialize and create output directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._domain_x = domain.x.copy()
        self._frames = []
        self._times = []

    def on_step(
        self,
        step: int,
        t: float,
        fields: dict[str, "Field"],
        domain: "Domain1D",
    ) -> None:
        """Capture frame if output is due."""
        dt = 0.001  # Approximate
        if not self.should_output(step, t, dt):
            return

        if self.field_name is None:
            return

        if self.field_name not in fields:
            raise KeyError(f"Field '{self.field_name}' not found for GIF output")

        # Store field values and time
        self._frames.append(fields[self.field_name].values.copy())
        self._times.append(t)

    def on_end(
        self,
        fields: dict[str, "Field"],
        domain: "Domain1D",
    ) -> None:
        """Create and save the GIF."""
        if not self._frames or self.field_name is None:
            return

        try:
            import imageio.v2 as imageio
        except ImportError:
            import imageio

        images = []
        for i, (values, t) in enumerate(zip(self._frames, self._times)):
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(self._domain_x, values, "b-", linewidth=1.5)
            ax.set_xlabel("x")
            ax.set_ylabel(self.field_name)
            ax.set_title(f"{self.field_name} at t = {t:.4f}")
            ax.set_ylim(-0.1, 1.1)
            ax.grid(True, alpha=0.3)

            # Render to image array
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            images.append(img)
            plt.close(fig)

        # Save GIF (duration in ms per frame = 1000 / fps)
        filepath = self.output_dir / f"{self.field_name}.gif"
        duration = int(1000 / self.fps) if self.fps > 0 else 100
        imageio.mimsave(filepath, images, duration=duration)

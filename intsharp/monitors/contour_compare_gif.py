"""
Contour Compare GIF output monitor (2D only).

Compares multiple fields by overlaying their contour lines on a single plot.
Each field can have its own contour levels, color, and linestyle.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from ..registry import register_monitor
from .base import Monitor

if TYPE_CHECKING:
    from ..domain import Domain
    from ..fields import Field


# Default color cycle for fields
DEFAULT_COLORS = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray"]


@register_monitor("contour_compare_gif")
class ContourCompareGIFMonitor(Monitor):
    """
    Animated GIF comparing contour lines of multiple fields.

    Each field can have its own contour levels, color, and linestyle.
    Only supports 2D domains.
    """

    def __init__(
        self,
        output_dir: Path,
        every_n_steps: int | None = None,
        at_times: list[float] | None = None,
        compare_fields: list[dict] | None = None,
        fps: int = 10,
        **kwargs,
    ):
        super().__init__(output_dir, every_n_steps, at_times)
        self.compare_fields = compare_fields or []
        self.fps = fps
        # Storage for frames: list of dicts {field_name: values}
        self._frames: list[dict[str, np.ndarray]] = []
        self._times: list[float] = []
        self._domain_X: np.ndarray | None = None
        self._domain_Y: np.ndarray | None = None
        self._x_min: float = 0.0
        self._x_max: float = 1.0
        self._y_min: float = 0.0
        self._y_max: float = 1.0

    def on_start(
        self,
        fields: dict[str, "Field"],
        domain: "Domain",
    ) -> None:
        """Initialize and create output directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._frames = []
        self._times = []

        if domain.ndim != 2:
            raise ValueError("ContourCompareGIFMonitor only supports 2D domains")

        self._domain_X = domain.X.copy()  # type: ignore
        self._domain_Y = domain.Y.copy()  # type: ignore
        self._x_min = domain.x_min
        self._x_max = domain.x_max
        self._y_min = domain.y_min  # type: ignore
        self._y_max = domain.y_max  # type: ignore

    def on_step(
        self,
        step: int,
        t: float,
        fields: dict[str, "Field"],
        domain: "Domain",
    ) -> None:
        """Capture frame if output is due."""
        dt = 0.001  # Approximate
        if not self.should_output(step, t, dt):
            return

        # Store all field values for this frame
        frame_data: dict[str, np.ndarray] = {}
        for field_cfg in self.compare_fields:
            field_name = field_cfg.get("field", "")
            if field_name in fields:
                frame_data[field_name] = fields[field_name].values.copy()

        if frame_data:
            self._frames.append(frame_data)
            self._times.append(t)

    def on_end(
        self,
        fields: dict[str, "Field"],
        domain: "Domain",
    ) -> None:
        """Create and save the contour comparison GIF."""
        if not self._frames:
            return

        try:
            import imageio.v2 as imageio
        except ImportError:
            import imageio

        images = []
        for i, (frame_data, t) in enumerate(zip(self._frames, self._times)):
            fig, ax = plt.subplots(figsize=(6, 6))

            # Plot each field's contours
            for j, field_cfg in enumerate(self.compare_fields):
                field_name = field_cfg.get("field", "")
                if field_name not in frame_data:
                    continue

                values = frame_data[field_name]
                contour_levels = field_cfg.get("contour_levels", [0.5])
                color = field_cfg.get("color") or DEFAULT_COLORS[j % len(DEFAULT_COLORS)]
                linestyle = field_cfg.get("linestyle", "-")

                ax.contour(
                    self._domain_X,
                    self._domain_Y,
                    values,
                    levels=contour_levels,
                    colors=[color],
                    linestyles=[linestyle],
                    linewidths=2,
                )

            ax.set_xlim(self._x_min, self._x_max)
            ax.set_ylim(self._y_min, self._y_max)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(f"Contour comparison at t = {t:.4f}")
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)

            # Render to image array
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            images.append(img)
            plt.close(fig)

        # Save GIF (duration in ms per frame = 1000 / fps)
        filepath = self.output_dir / "contour_compare.gif"
        duration = int(1000 / self.fps) if self.fps > 0 else 100
        imageio.mimsave(filepath, images, duration=duration)

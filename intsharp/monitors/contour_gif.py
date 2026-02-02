"""
Contour GIF output monitor (2D only).

Draws contour lines (e.g., 0.5 level) with optional centroid and crosshairs.
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


def compute_centroid(values: np.ndarray, X: np.ndarray, Y: np.ndarray, dx: float, dy: float) -> tuple[float, float]:
    """
    Compute the center of mass of a 2D field.

    Parameters
    ----------
    values : np.ndarray
        Field values (shape: ny, nx).
    X : np.ndarray
        X meshgrid.
    Y : np.ndarray
        Y meshgrid.
    dx : float
        Grid spacing in x.
    dy : float
        Grid spacing in y.

    Returns
    -------
    tuple[float, float]
        (x_centroid, y_centroid)
    """
    total_mass = np.sum(values) * dx * dy
    if total_mass < 1e-12:
        # Avoid division by zero; return domain center
        return float(np.mean(X)), float(np.mean(Y))

    x_cm = np.sum(X * values) * dx * dy / total_mass
    y_cm = np.sum(Y * values) * dx * dy / total_mass
    return float(x_cm), float(y_cm)


@register_monitor("contour_gif")
class ContourGIFMonitor(Monitor):
    """
    Animated GIF with contour lines, optional centroid marker, and crosshairs.

    Only supports 2D domains.
    """

    def __init__(
        self,
        output_dir: Path,
        every_n_steps: int | None = None,
        at_times: list[float] | None = None,
        field: str | None = None,
        fps: int = 10,
        contour_level: float = 0.5,
        show_centroid: bool = False,
        show_crosshairs: bool = False,
        **kwargs,
    ):
        super().__init__(output_dir, every_n_steps, at_times)
        self.field_name = field
        self.fps = fps
        self.contour_level = contour_level
        self.show_centroid = show_centroid
        self.show_crosshairs = show_crosshairs
        self._frames: list[np.ndarray] = []
        self._times: list[float] = []
        self._domain_X: np.ndarray | None = None
        self._domain_Y: np.ndarray | None = None
        self._dx: float = 1.0
        self._dy: float = 1.0
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
            raise ValueError("ContourGIFMonitor only supports 2D domains")

        self._domain_X = domain.X.copy()  # type: ignore
        self._domain_Y = domain.Y.copy()  # type: ignore
        self._dx = domain.dx
        self._dy = domain.dy
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

        if self.field_name is None:
            return

        if self.field_name not in fields:
            raise KeyError(f"Field '{self.field_name}' not found for contour GIF output")

        # Store field values and time
        self._frames.append(fields[self.field_name].values.copy())
        self._times.append(t)

    def on_end(
        self,
        fields: dict[str, "Field"],
        domain: "Domain",
    ) -> None:
        """Create and save the contour GIF."""
        if not self._frames or self.field_name is None:
            return

        try:
            import imageio.v2 as imageio
        except ImportError:
            import imageio

        images = []
        for i, (values, t) in enumerate(zip(self._frames, self._times)):
            fig, ax = plt.subplots(figsize=(6, 6))

            # Draw contour at specified level
            cs = ax.contour(
                self._domain_X,
                self._domain_Y,
                values,
                levels=[self.contour_level],
                colors=["blue"],
                linewidths=2,
            )

            # Compute centroid if needed
            if self.show_centroid or self.show_crosshairs:
                x_cm, y_cm = compute_centroid(
                    values, self._domain_X, self._domain_Y, self._dx, self._dy
                )

                if self.show_crosshairs:
                    # Draw vertical and horizontal lines through centroid
                    ax.axvline(x_cm, color="red", linestyle="--", linewidth=1, alpha=0.7)
                    ax.axhline(y_cm, color="red", linestyle="--", linewidth=1, alpha=0.7)

                if self.show_centroid:
                    # Draw centroid marker
                    ax.plot(x_cm, y_cm, "ro", markersize=8, markeredgecolor="darkred", markeredgewidth=1.5)

            ax.set_xlim(self._x_min, self._x_max)
            ax.set_ylim(self._y_min, self._y_max)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(f"{self.field_name} contour at t = {t:.4f}")
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)

            # Render to image array
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            images.append(img)
            plt.close(fig)

        # Save GIF (duration in ms per frame = 1000 / fps)
        filepath = self.output_dir / f"{self.field_name}_contour.gif"
        duration = int(1000 / self.fps) if self.fps > 0 else 100
        imageio.mimsave(filepath, images, duration=duration)

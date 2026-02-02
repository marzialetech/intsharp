"""
Unified animated output monitor (1D and 2D).

Supports GIF and MP4 output formats.

Modes:
- Single-field (field=...):
  - 1D: line plot
  - 2D style="pcolormesh": heatmap (default)
  - 2D style="contour": contour lines at contour_levels
- Multi-field (compare_fields=[...]):
  - 1D: overlaid line plots
  - 2D: overlaid contour lines
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


DEFAULT_COLORS = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray"]


@register_monitor("gif")
@register_monitor("mp4")
class GIFMonitor(Monitor):
    """
    Unified animated output monitor (GIF or MP4).

    Single-field mode (field=...):
      - 1D: line plot
      - 2D: pcolormesh (default) or contour (style="contour")

    Multi-field mode (compare_fields=[...]):
      - 1D: overlaid line plots
      - 2D: overlaid contour lines

    Output format can be 'gif' (default) or 'mp4'.
    """

    def __init__(
        self,
        output_dir: Path,
        every_n_steps: int | None = None,
        at_times: list[float] | None = None,
        field: str | None = None,
        compare_fields: list[dict] | None = None,
        style: str = "pcolormesh",
        contour_levels: list[float] | None = None,
        output_format: str = "gif",
        fps: int = 10,
        **kwargs,
    ):
        super().__init__(output_dir, every_n_steps, at_times)
        self.field_name = field
        self.compare_fields = compare_fields or []
        self.style = style
        self.contour_levels = contour_levels if contour_levels else [0.5]
        self.output_format = output_format.lower()
        self.fps = fps

        # Mode: "single" or "compare"
        self._mode = "compare" if self.compare_fields else "single"

        # Storage
        self._frames_single: list[np.ndarray] = []
        self._frames_compare: list[dict[str, np.ndarray]] = []
        self._times: list[float] = []

        # Domain info
        self._ndim: int = 1
        self._domain_x: np.ndarray | None = None
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
        self._frames_single = []
        self._frames_compare = []
        self._times = []
        self._ndim = domain.ndim

        if domain.ndim == 1:
            self._domain_x = domain.x.copy()
            self._x_min = float(np.min(domain.x))
            self._x_max = float(np.max(domain.x))
        else:
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
        dt = 0.001
        if not self.should_output(step, t, dt):
            return

        if self._mode == "single":
            if self.field_name is None or self.field_name not in fields:
                return
            self._frames_single.append(fields[self.field_name].values.copy())
            self._times.append(t)
        else:
            # Compare mode
            frame_data: dict[str, np.ndarray] = {}
            for field_cfg in self.compare_fields:
                fname = field_cfg.get("field", "")
                if fname in fields:
                    frame_data[fname] = fields[fname].values.copy()
            if frame_data:
                self._frames_compare.append(frame_data)
                self._times.append(t)

    def on_end(
        self,
        fields: dict[str, "Field"],
        domain: "Domain",
    ) -> None:
        """Create and save the GIF."""
        if self._mode == "single":
            self._save_single_field()
        else:
            self._save_compare_fields()

    def _save_single_field(self) -> None:
        """Save single-field GIF."""
        if not self._frames_single or self.field_name is None:
            return

        try:
            import imageio.v2 as imageio
        except ImportError:
            import imageio

        images = []
        for values, t in zip(self._frames_single, self._times):
            if self._ndim == 1:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(self._domain_x, values, "b-", linewidth=1.5)
                ax.set_xlabel("x")
                ax.set_ylabel(self.field_name)
                ax.set_title(f"{self.field_name} at t = {t:.4f}")
                ax.set_xlim(self._x_min, self._x_max)
                ax.set_ylim(-0.1, 1.1)
                ax.grid(True, alpha=0.3)
            else:
                fig, ax = plt.subplots(figsize=(6, 6))
                if self.style == "contour":
                    ax.contour(
                        self._domain_X,
                        self._domain_Y,
                        values,
                        levels=self.contour_levels,
                        colors=["blue"],
                        linewidths=2,
                    )
                else:
                    pcm = ax.pcolormesh(
                        self._domain_X,
                        self._domain_Y,
                        values,
                        cmap="viridis",
                        vmin=0.0,
                        vmax=1.0,
                        shading="auto",
                    )
                    fig.colorbar(pcm, ax=ax, label=self.field_name)
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_title(f"{self.field_name} at t = {t:.4f}")
                ax.set_xlim(self._x_min, self._x_max)
                ax.set_ylim(self._y_min, self._y_max)
                ax.set_aspect("equal")
                ax.grid(True, alpha=0.3)

            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            images.append(img)
            plt.close(fig)

        ext = "mp4" if self.output_format == "mp4" else "gif"
        filepath = self.output_dir / f"{self.field_name}.{ext}"
        if self.output_format == "mp4":
            imageio.mimsave(filepath, images, fps=self.fps, codec="libx264")
        else:
            duration = int(1000 / self.fps) if self.fps > 0 else 100
            imageio.mimsave(filepath, images, duration=duration)

    def _save_compare_fields(self) -> None:
        """Save multi-field comparison GIF."""
        if not self._frames_compare:
            return

        try:
            import imageio.v2 as imageio
        except ImportError:
            import imageio

        images = []
        for frame_data, t in zip(self._frames_compare, self._times):
            if self._ndim == 1:
                fig, ax = plt.subplots(figsize=(8, 4))
                for j, field_cfg in enumerate(self.compare_fields):
                    fname = field_cfg.get("field", "")
                    if fname not in frame_data:
                        continue
                    values = frame_data[fname]
                    color = field_cfg.get("color") or DEFAULT_COLORS[j % len(DEFAULT_COLORS)]
                    linestyle = field_cfg.get("linestyle", "-")
                    ax.plot(
                        self._domain_x,
                        values,
                        color=color,
                        linestyle=linestyle,
                        linewidth=1.5,
                        label=fname,
                    )
                ax.set_xlim(self._x_min, self._x_max)
                ax.set_ylim(-0.1, 1.1)
                ax.set_xlabel("x")
                ax.set_ylabel("value")
                ax.set_title(f"Comparison at t = {t:.4f}")
                ax.grid(True, alpha=0.3)
                ax.legend(loc="upper right", fontsize=8)
            else:
                # 2D: overlaid contours
                fig, ax = plt.subplots(figsize=(6, 6))
                for j, field_cfg in enumerate(self.compare_fields):
                    fname = field_cfg.get("field", "")
                    if fname not in frame_data:
                        continue
                    values = frame_data[fname]
                    levels = field_cfg.get("contour_levels", self.contour_levels)
                    color = field_cfg.get("color") or DEFAULT_COLORS[j % len(DEFAULT_COLORS)]
                    linestyle = field_cfg.get("linestyle", "-")
                    ax.contour(
                        self._domain_X,
                        self._domain_Y,
                        values,
                        levels=levels,
                        colors=[color],
                        linestyles=[linestyle],
                        linewidths=2,
                    )
                ax.set_xlim(self._x_min, self._x_max)
                ax.set_ylim(self._y_min, self._y_max)
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_title(f"Comparison at t = {t:.4f}")
                ax.set_aspect("equal")
                ax.grid(True, alpha=0.3)

            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            images.append(img)
            plt.close(fig)

        ext = "mp4" if self.output_format == "mp4" else "gif"
        filepath = self.output_dir / f"compare.{ext}"
        if self.output_format == "mp4":
            imageio.mimsave(filepath, images, fps=self.fps, codec="libx264")
        else:
            duration = int(1000 / self.fps) if self.fps > 0 else 100
            imageio.mimsave(filepath, images, duration=duration)

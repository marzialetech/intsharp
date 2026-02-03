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

import io
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
        colormap: str | None = None,
        contour_overlay_color: str | None = None,
        contour_color: str | None = None,
        background_color: str | None = None,
        show_colorbar: bool | None = None,
        show_annotations: bool | None = None,
        output_format: str = "gif",
        fps: int = 10,
        quiver_overlay_x: str | None = None,
        quiver_overlay_y: str | None = None,
        quiver_skip: int | None = None,
        **kwargs,
    ):
        super().__init__(output_dir, every_n_steps, at_times)
        self.field_name = field
        self.compare_fields = compare_fields or []
        self.style = style
        self.contour_levels = contour_levels if contour_levels else [0.5]
        self.colormap = colormap or "viridis"
        self.contour_overlay_color = contour_overlay_color
        self.contour_color = contour_color or "blue"
        self.background_color = background_color
        self.show_colorbar = show_colorbar if show_colorbar is not None else True
        self.show_annotations = show_annotations if show_annotations is not None else True
        self.output_format = output_format.lower()
        self.fps = fps
        self.quiver_overlay_x = quiver_overlay_x
        self.quiver_overlay_y = quiver_overlay_y
        self.quiver_skip = quiver_skip if quiver_skip is not None else 4

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

    def _append_frame(
        self,
        images: list,
        fig: "plt.Figure",
        imageio_module,
    ) -> None:
        """Append current figure as RGB array; use tight crop when colorbar and annotations hidden."""
        if not self.show_colorbar and not self.show_annotations:
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=150)
            buf.seek(0)
            img = imageio_module.imread(buf)
            if img.ndim == 3 and img.shape[2] == 4:
                img = img[:, :, :3]
            images.append(img)
        else:
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            images.append(img)

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
            frame_data: dict = {"values": fields[self.field_name].values.copy()}
            if self.quiver_overlay_x and self.quiver_overlay_y:
                if self.quiver_overlay_x in fields and self.quiver_overlay_y in fields:
                    frame_data["quiver_x"] = fields[self.quiver_overlay_x].values.copy()
                    frame_data["quiver_y"] = fields[self.quiver_overlay_y].values.copy()
            self._frames_single.append(frame_data)
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

        # Compute global min/max across all frames for consistent axis bounds
        all_values = [
            (frame_data["values"] if isinstance(frame_data, dict) else frame_data)
            for frame_data in self._frames_single
        ]
        global_min = min(float(np.min(v)) for v in all_values)
        global_max = max(float(np.max(v)) for v in all_values)
        # Add 5% padding
        y_range = global_max - global_min if global_max > global_min else 1.0
        y_min = global_min - 0.05 * y_range
        y_max = global_max + 0.05 * y_range

        images = []
        for frame_data, t in zip(self._frames_single, self._times):
            values = frame_data["values"] if isinstance(frame_data, dict) else frame_data
            quiver_x = frame_data.get("quiver_x") if isinstance(frame_data, dict) else None
            quiver_y = frame_data.get("quiver_y") if isinstance(frame_data, dict) else None

            if self._ndim == 1:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(self._domain_x, values, "b-", linewidth=1.5)
                if self.show_annotations:
                    ax.set_xlabel("x")
                    ax.set_ylabel(self.field_name)
                    ax.set_title(f"{self.field_name} at t = {t:.4f}")
                else:
                    ax.set_xticks([])
                    ax.set_yticks([])
                ax.set_xlim(self._x_min, self._x_max)
                ax.set_ylim(y_min, y_max)
                ax.grid(True, alpha=0.3)
            else:
                fig, ax = plt.subplots(figsize=(6, 6))
                if self.style == "contour":
                    if self.background_color:
                        ax.set_facecolor(self.background_color)
                        fig.patch.set_facecolor(self.background_color)
                    ax.contour(
                        self._domain_X,
                        self._domain_Y,
                        values,
                        levels=self.contour_levels,
                        colors=[self.contour_color],
                        linewidths=2,
                    )
                else:
                    # Use computed global bounds for consistent coloring across frames
                    pcm = ax.pcolormesh(
                        self._domain_X,
                        self._domain_Y,
                        values,
                        cmap=self.colormap,
                        vmin=y_min,
                        vmax=y_max,
                        shading="auto",
                    )
                    if self.show_colorbar:
                        fig.colorbar(pcm, ax=ax, label=self.field_name)
                    if self.contour_overlay_color:
                        ax.contour(
                            self._domain_X,
                            self._domain_Y,
                            values,
                            levels=self.contour_levels,
                            colors=[self.contour_overlay_color],
                            linewidths=2,
                        )
                    # Quiver overlay (e.g. surface force vectors)
                    if quiver_x is not None and quiver_y is not None:
                        skip = self.quiver_skip
                        U = quiver_x[::skip, ::skip]
                        V = quiver_y[::skip, ::skip]
                        mag = np.sqrt(U**2 + V**2)
                        max_mag = float(np.max(mag)) if np.any(mag > 0) else 1.0
                        ny, nx = self._domain_X.shape
                        dx = (self._x_max - self._x_min) / max(nx - 1, 1)
                        dy = (self._y_max - self._y_min) / max(ny - 1, 1)
                        max_arrow_length = 6 * min(dx, dy)  # ~6 grid cells
                        scale = max_mag / max_arrow_length if max_mag > 0 else 1.0
                        U_norm = U / scale
                        V_norm = V / scale
                        ax.quiver(
                            self._domain_X[::skip, ::skip],
                            self._domain_Y[::skip, ::skip],
                            U_norm,
                            V_norm,
                            color="black",
                            scale_units="xy",
                            angles="xy",
                        )
                if self.show_annotations:
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.set_title(f"{self.field_name} at t = {t:.4f}")
                else:
                    ax.set_xticks([])
                    ax.set_yticks([])
                ax.set_xlim(self._x_min, self._x_max)
                ax.set_ylim(self._y_min, self._y_max)
                ax.set_aspect("equal")
                ax.grid(True, alpha=0.3)

            self._append_frame(images, fig, imageio)
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

        # Compute global min/max across all frames and all fields
        all_values = []
        for frame_data in self._frames_compare:
            for fname in frame_data:
                all_values.append(frame_data[fname])
        if all_values:
            global_min = min(float(np.min(v)) for v in all_values)
            global_max = max(float(np.max(v)) for v in all_values)
        else:
            global_min, global_max = 0.0, 1.0
        y_range = global_max - global_min if global_max > global_min else 1.0
        y_min = global_min - 0.05 * y_range
        y_max = global_max + 0.05 * y_range

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
                ax.set_ylim(y_min, y_max)
                if self.show_annotations:
                    ax.set_xlabel("x")
                    ax.set_ylabel("value")
                    ax.set_title(f"Comparison at t = {t:.4f}")
                    ax.legend(loc="upper right", fontsize=8)
                else:
                    ax.set_xticks([])
                    ax.set_yticks([])
                ax.grid(True, alpha=0.3)
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
                if self.show_annotations:
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.set_title(f"Comparison at t = {t:.4f}")
                else:
                    ax.set_xticks([])
                    ax.set_yticks([])
                ax.set_aspect("equal")
                ax.grid(True, alpha=0.3)

            self._append_frame(images, fig, imageio)
            plt.close(fig)

        ext = "mp4" if self.output_format == "mp4" else "gif"
        filepath = self.output_dir / f"compare.{ext}"
        if self.output_format == "mp4":
            imageio.mimsave(filepath, images, fps=self.fps, codec="libx264")
        else:
            duration = int(1000 / self.fps) if self.fps > 0 else 100
            imageio.mimsave(filepath, images, duration=duration)

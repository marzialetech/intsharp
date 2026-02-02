"""
Console output monitor with progress bar.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from tqdm import tqdm

from ..registry import register_monitor
from .base import Monitor

if TYPE_CHECKING:
    from ..domain import Domain1D
    from ..fields import Field


@register_monitor("console")
class ConsoleMonitor(Monitor):
    """
    Console output with tqdm progress bar.

    Prints step/time info at specified intervals.
    """

    def __init__(
        self,
        output_dir: Path,
        every_n_steps: int | None = None,
        at_times: list[float] | None = None,
        total_steps: int | None = None,
        **kwargs,
    ):
        super().__init__(output_dir, every_n_steps, at_times)
        self.total_steps = total_steps
        self._pbar: tqdm | None = None

    def on_start(
        self,
        fields: dict[str, "Field"],
        domain: "Domain1D",
    ) -> None:
        """Initialize progress bar."""
        if self.total_steps is not None:
            self._pbar = tqdm(total=self.total_steps, desc="Simulating", unit="step")

    def on_step(
        self,
        step: int,
        t: float,
        fields: dict[str, "Field"],
        domain: "Domain1D",
    ) -> None:
        """Update progress bar and optionally print info."""
        if self._pbar is not None:
            self._pbar.update(1)
            self._pbar.set_postfix({"t": f"{t:.4f}"})

    def on_end(
        self,
        fields: dict[str, "Field"],
        domain: "Domain1D",
    ) -> None:
        """Close progress bar."""
        if self._pbar is not None:
            self._pbar.close()

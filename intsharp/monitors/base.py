"""
Base class for output monitors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..domain import Domain1D
    from ..fields import Field


class Monitor(ABC):
    """
    Abstract base class for output monitors.

    Monitors are called at each time step and decide whether to output
    based on their configuration (every_n_steps or at_times).
    """

    def __init__(
        self,
        output_dir: Path,
        every_n_steps: int | None = None,
        at_times: list[float] | None = None,
    ):
        """
        Initialize monitor.

        Parameters
        ----------
        output_dir : Path
            Directory for output files.
        every_n_steps : int or None
            Output every N steps.
        at_times : list[float] or None
            Output at specific times.
        """
        self.output_dir = Path(output_dir)
        self.every_n_steps = every_n_steps
        self.at_times = at_times or []
        self._times_output: set[float] = set()

    def should_output(self, step: int, t: float, dt: float) -> bool:
        """
        Check if output should occur at this step/time.

        Parameters
        ----------
        step : int
            Current step number.
        t : float
            Current simulation time.
        dt : float
            Time step size.

        Returns
        -------
        bool
            True if output should occur.
        """
        # Check every_n_steps
        if self.every_n_steps is not None and step % self.every_n_steps == 0:
            return True

        # Check at_times (with tolerance for floating point)
        for target_t in self.at_times:
            if target_t not in self._times_output:
                if abs(t - target_t) < 0.5 * dt:
                    self._times_output.add(target_t)
                    return True

        return False

    @abstractmethod
    def on_step(
        self,
        step: int,
        t: float,
        fields: dict[str, "Field"],
        domain: "Domain1D",
    ) -> None:
        """
        Called at each step; output if should_output() returns True.

        Parameters
        ----------
        step : int
            Current step number.
        t : float
            Current simulation time.
        fields : dict[str, Field]
            All simulation fields.
        domain : Domain1D
            The computational domain.
        """
        pass

    def on_start(
        self,
        fields: dict[str, "Field"],
        domain: "Domain1D",
    ) -> None:
        """
        Called at simulation start. Override for setup.

        Parameters
        ----------
        fields : dict[str, Field]
            All simulation fields.
        domain : Domain1D
            The computational domain.
        """
        pass

    def on_end(
        self,
        fields: dict[str, "Field"],
        domain: "Domain1D",
    ) -> None:
        """
        Called at simulation end. Override for finalization.

        Parameters
        ----------
        fields : dict[str, Field]
            All simulation fields.
        domain : Domain1D
            The computational domain.
        """
        pass

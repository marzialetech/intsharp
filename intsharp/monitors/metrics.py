"""
Metrics monitor — tracks ε_char, ε_char/ε_target, δ*_2, δ*_∞ per field.

Writes a single TSV file per monitored field:

    step  t  eps_char  eps_char_over_target  delta_2  delta_inf

The file is incrementally appended so data is available even if the run
is interrupted.

YAML example (static / SIDI)
-----------------------------

.. code-block:: yaml

    output:
      monitors:
        - type: metrics
          every_n_steps: 1
          fields: [alpha_pm_cal, alpha_cl]
          interface_radius: 0.025

YAML example (advection with periodic wrapping)
------------------------------------------------

.. code-block:: yaml

    output:
      monitors:
        - type: metrics
          every_n_steps: 100
          fields: [alpha_cl]
          interface_radius: 0.025
          advection_velocity: 1.0   # shifts field back to center before metrics
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..metrics import compute_alpha_true, compute_delta_2, compute_delta_inf, compute_eps_char
from ..registry import register_monitor
from .base import Monitor

if TYPE_CHECKING:
    from ..domain import Domain
    from ..fields import Field


@register_monitor("metrics")
class MetricsMonitor(Monitor):
    """
    Per-step interface-quality metrics.

    Parameters
    ----------
    interface_radius : float
        R — radius of the α = 0.5 contour in the reference hat profile.
    eps_target : float or None
        If given, an extra column ε_char / ε_target is written.
        When *None*, the column is still present but filled with *nan*.
    advection_velocity : float or None
        When set, the field is ``np.roll``-ed back to center before metrics
        are computed so that periodic boundary wrapping does not confuse the
        contour finder.  The shift is ``round(velocity * t / dx)`` cells.
    initial_center : float
        Initial x-position of the hat center (default 0.0).
    """

    def __init__(
        self,
        output_dir: Path,
        every_n_steps: int | None = None,
        at_times: list[float] | None = None,
        field: str | None = None,
        fields: list[str] | None = None,
        interface_radius: float = 0.025,
        eps_target: float | None = None,
        advection_velocity: float | None = None,
        initial_center: float = 0.0,
        **kwargs,
    ):
        super().__init__(output_dir, every_n_steps, at_times)
        if field is not None:
            self.field_names = [field]
        elif fields is not None:
            self.field_names = list(fields)
        else:
            self.field_names = []

        self.R = interface_radius
        self.eps_target = eps_target
        self.advection_velocity = advection_velocity
        self.initial_center = initial_center
        self._handles: dict[str, object] = {}

    # ── helpers ────────────────────────────────────────────────────

    def _shift_to_center(
        self,
        psi: np.ndarray,
        t: float,
        dx: float,
    ) -> np.ndarray:
        """Roll *psi* so that the advected hat is back at its initial center."""
        if self.advection_velocity is None:
            return psi
        shift_cells = round(self.advection_velocity * t / dx)
        return np.roll(psi, -shift_cells)

    # ── lifecycle ─────────────────────────────────────────────────

    def on_start(
        self,
        fields: dict[str, "Field"],
        domain: "Domain",
    ) -> None:
        out = self.output_dir / "metrics"
        out.mkdir(parents=True, exist_ok=True)

        header = "step\tt\teps_char\teps_char_over_target\tdelta_2\tdelta_inf\n"
        for name in self.field_names:
            fp = out / f"{name}.tsv"
            fh = open(fp, "w")  # noqa: SIM115
            fh.write(header)
            fh.flush()
            self._handles[name] = fh

    def on_step(
        self,
        step: int,
        t: float,
        fields: dict[str, "Field"],
        domain: "Domain",
    ) -> None:
        dt = 0.001
        if not self.should_output(step, t, dt):
            return

        x = domain.x
        dx = domain.dx

        for name in self.field_names:
            if name not in fields:
                continue

            psi = np.asarray(fields[name].values, dtype=np.float64)
            psi = self._shift_to_center(psi, t, dx)

            ec = compute_eps_char(psi, x, self.R)
            ratio = ec / self.eps_target if self.eps_target else np.nan

            if np.isfinite(ec) and ec > 0:
                ref = compute_alpha_true(x, self.R, ec)
                d2 = compute_delta_2(psi, ref, dx)
                dinf = compute_delta_inf(psi, ref)
            else:
                d2 = np.nan
                dinf = np.nan

            fh = self._handles.get(name)
            if fh is not None:
                fh.write(
                    f"{step}\t{t:.8e}\t{ec:.8e}\t{ratio:.8e}\t{d2:.8e}\t{dinf:.8e}\n"
                )
                fh.flush()

    def on_end(
        self,
        fields: dict[str, "Field"],
        domain: "Domain",
    ) -> None:
        for fh in self._handles.values():
            fh.close()
        self._handles.clear()

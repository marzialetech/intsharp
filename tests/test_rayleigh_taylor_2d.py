"""
Unit test: 2D Rayleigh-Taylor instability (two-phase 5-equation Euler, FV).

Validates that the 2D two-phase path runs with and without gravity and
keeps core thermodynamic/admissibility quantities finite and bounded.
"""

import tempfile
from pathlib import Path

import numpy as np


CONFIG_PATH = Path(__file__).resolve().parent.parent / "unit_tests" / "rayleigh_taylor_2d.yaml"


def _run_case(gravity_enabled: bool):
    from intsharp.config import load_config
    from intsharp.runner import run_simulation

    config = load_config(CONFIG_PATH)
    config.physics.gravity.enabled = gravity_enabled
    config.domain.n_points_x = 48
    config.domain.n_points_y = 96
    config.output.monitors = []
    config.time.n_steps = 60

    with tempfile.TemporaryDirectory() as tmpdir:
        config.output.directory = tmpdir
        return run_simulation(config)


def test_rayleigh_taylor_2d_runs_with_gravity():
    fields = _run_case(gravity_enabled=True)

    rho = fields["rho"].values
    p = fields["p"].values
    alpha = fields["alpha"].values

    assert rho.ndim == 2
    assert np.all(np.isfinite(rho))
    assert np.all(np.isfinite(p))
    assert np.all(np.isfinite(alpha))
    assert np.all(rho > 0.0)
    assert np.all(p > 0.0)
    assert np.all((alpha >= 0.0) & (alpha <= 1.0))


def test_rayleigh_taylor_2d_runs_without_gravity():
    fields = _run_case(gravity_enabled=False)

    rho = fields["rho"].values
    p = fields["p"].values
    alpha = fields["alpha"].values

    assert rho.ndim == 2
    assert np.all(np.isfinite(rho))
    assert np.all(np.isfinite(p))
    assert np.all(np.isfinite(alpha))
    assert np.all(rho > 0.0)
    assert np.all(p > 0.0)
    assert np.all((alpha >= 0.0) & (alpha <= 1.0))

"""
Unit test: Sod shock tube (1D compressible Euler).

Validates that the Sod shock tube produces results matching the reference snapshot.
Run: pytest tests/test_sod_shock_tube.py -v
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

# Reference snapshot (from sod_shock_tube_1d_2026-02-02_21:19:55)
REFERENCE_DIR = Path(__file__).resolve().parent.parent / "unit_tests" / "sod_shock_tube_1d_2026-02-02_21:19:55"
CONFIG_PATH = Path(__file__).resolve().parent.parent / "unit_tests" / "sod_shock_tube_1d.yaml"


def _load_snapshot(path: Path) -> tuple[np.ndarray, dict]:
    """Load snapshot_00004.txt and return (x, data_dict)."""
    data = {}
    cols = []
    with open(path / "txt" / "snapshot_00004.txt") as f:
        for line in f:
            if line.startswith("#"):
                if "x " in line or line.strip() == "# x rho u p E e_int rho_u":
                    parts_header = line.strip().split()
                    cols = [p for p in parts_header if p != "#" and "=" not in p]
                    for c in cols:
                        data[c] = []
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                for i, col in enumerate(cols):
                    if i < len(parts):
                        data[col].append(float(parts[i]))
    for k in data:
        data[k] = np.array(data[k])
    x = data.get("x", np.arange(len(data.get("rho", []))))
    return x, data


@pytest.mark.skipif(not REFERENCE_DIR.exists(), reason="Reference snapshot not found")
def test_sod_shock_tube_matches_reference():
    """Run Sod shock tube and verify output matches reference snapshot."""
    from intsharp.config import load_config
    from intsharp.runner import run_simulation

    config = load_config(CONFIG_PATH)
    with tempfile.TemporaryDirectory() as tmpdir:
        config.output.directory = tmpdir
        run_simulation(config)

        out_dir = Path(tmpdir)
        assert (out_dir / "txt" / "snapshot_00004.txt").exists()

        x_ref, data_ref = _load_snapshot(REFERENCE_DIR)
        x_new, data_new = _load_snapshot(out_dir)

        np.testing.assert_allclose(x_ref, x_new, err_msg="x coordinates differ")
        for key in data_ref:
            assert key in data_new, f"Missing field {key}"
            np.testing.assert_allclose(
                data_ref[key], data_new[key],
                rtol=1e-14, atol=1e-14,
                err_msg=f"Field {key} differs from reference"
            )


@pytest.mark.parametrize("dg_order", [1, 2, 3])
def test_sod_shock_tube_dg_runs(dg_order: int):
    """Run Sod shock tube with DG spatial discretization and check admissibility."""
    from intsharp.config import load_config
    from intsharp.runner import run_simulation

    config = load_config(CONFIG_PATH)
    config.physics.euler_spatial_discretization = "dg"
    config.physics.dg_order = dg_order
    config.output.monitors = []
    config.time.n_steps = 600

    with tempfile.TemporaryDirectory() as tmpdir:
        config.output.directory = tmpdir
        fields = run_simulation(config)

    rho = fields["rho"].values
    p = fields["p"].values
    assert np.all(np.isfinite(rho))
    assert np.all(np.isfinite(p))
    assert np.all(rho > 0.0)
    assert np.all(p > 0.0)

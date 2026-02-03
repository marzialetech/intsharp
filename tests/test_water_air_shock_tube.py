"""
Unit test: Water-air shock tube (1D two-phase 5-equation Euler).

Validates that the water-air shock tube produces results matching the reference snapshot.
Run: pytest tests/test_water_air_shock_tube.py -v
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

# Reference snapshot (from water_air_shock_tube_1d_2026-02-02_22:56:33)
REFERENCE_DIR = Path(__file__).resolve().parent.parent / "unit_tests" / "water_air_shock_tube_1d_2026-02-02_22:56:33"
CONFIG_PATH = Path(__file__).resolve().parent.parent / "unit_tests" / "water_air_shock_tube_1d.yaml"


def _load_snapshot(path: Path) -> tuple[np.ndarray, dict]:
    """Load snapshot_00004.txt and return (x, data_dict)."""
    data = {}
    cols = []
    with open(path / "txt" / "snapshot_00004.txt") as f:
        for line in f:
            if line.startswith("#"):
                if "x " in line or "=" not in line:
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
def test_water_air_shock_tube_matches_reference():
    """Run water-air shock tube and verify output matches reference snapshot."""
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
                rtol=1e-12, atol=1e-8,
                err_msg=f"Field {key} differs from reference"
            )

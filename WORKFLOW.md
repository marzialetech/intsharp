# IntSharp: YAML-Driven Interface Sharpening Framework

## Repository Structure

```
intsharp/
├── run.py               # CLI entry point
├── config.yaml          # Example simulation configuration
├── intsharp/            # Main Python package
│   ├── __init__.py
│   ├── config.py        # YAML parsing and validation (pydantic)
│   ├── domain.py        # 1D grid setup
│   ├── fields.py        # Field containers with IC evaluation
│   ├── boundary.py      # Periodic/Neumann/Dirichlet BCs
│   ├── registry.py      # Plugin registration utilities
│   ├── runner.py        # Main simulation loop
│   ├── sharpening.py    # PM and CL sharpening methods
│   ├── timesteppers.py  # Euler and RK4
│   ├── utils.py         # Shared utilities (from old notebooks)
│   ├── solvers/         # Advection discretizations
│   │   ├── __init__.py
│   │   └── upwind.py    # First-order upwind
│   └── monitors/        # Output handlers
│       ├── __init__.py
│       ├── base.py      # Monitor base class
│       ├── console.py   # Progress bar output
│       ├── image.py     # PNG/PDF frame output
│       ├── gif.py       # Animated GIF
│       └── hdf5.py      # HDF5 data output
├── results/             # Simulation outputs (generated)
├── Makefile
└── requirements.txt
```

## Running Simulations

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run with a YAML config
python run.py config.yaml

# Or use make (runs config.yaml)
make run
```

Each run creates a **new timestamped folder in the same directory as the YAML** (e.g. `unit_tests/run_20250201_191530` when running `unit_tests/tanh_one_rev_no_sharpening.yaml`), **moves** the YAML into that folder (single copy), and writes all monitor output (GIF, PNG, HDF5, etc.) there. The config in the YAML is not modified; only the output directory used for that run is set to the new folder.

### Configuration

Edit `config.yaml` to configure:

- **domain**: Grid extent and resolution
- **time**: Time step and number of steps
- **velocity**: Advection velocity (1D vector)
- **fields**: Field definitions with ICs and BCs
- **solver**: Advection discretization (e.g., `upwind`)
- **timestepper**: Time integration (e.g., `euler`, `rk4`)
- **sharpening**: Optional interface sharpening (PM or CL)
- **output**: Results directory and monitors

Example:
```yaml
domain:
  x_min: -0.5
  x_max: 0.5
  n_points: 200

time:
  dt: 0.001
  n_steps: 500

velocity: [0.5]

fields:
  - name: alpha
    initial_condition: "0.5 * (tanh((0.1 + x) / 0.02) + tanh((0.1 - x) / 0.02))"
    boundary:
      type: periodic

solver:
  type: upwind

sharpening:
  enabled: true
  method: cl
  eps_target: 0.01
  strength: 1.0

output:
  directory: ./results
  monitors:
    - type: console
    - type: png
      every_n_steps: 100
      field: alpha
    - type: gif
      every_n_steps: 10
      field: alpha
```

### Output

Results are saved to `./results/` (or configured directory):
- `alpha_XXXXX.png` - Frame snapshots
- `alpha.gif` - Animated visualization
- `simulation.h5` - HDF5 data for post-processing

## Extending the Framework

### Adding a Solver

```python
# intsharp/solvers/my_solver.py
from ..registry import register_solver

@register_solver("my_solver")
def my_solver(field_values, velocity, dx, dt, bc):
    # Implement discretization
    return new_values
```

### Adding a Sharpening Method

```python
# In intsharp/sharpening.py
@register_sharpening("my_method")
def my_sharpening(psi, dx, dt, eps_target, strength, bc):
    # Implement sharpening RHS
    return new_psi
```

## Make Targets

- `make run` - Run simulation with `config.yaml`
- `make install` - Install Python dependencies
- `make clean` - Remove build artifacts

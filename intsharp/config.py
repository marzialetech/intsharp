"""
Configuration parsing and validation for the simulation framework.

Uses pydantic for strict schema validation with helpful error messages.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class DomainConfig(BaseModel):
    """1D domain configuration."""
    x_min: float = Field(..., description="Left boundary of domain")
    x_max: float = Field(..., description="Right boundary of domain")
    n_points: int = Field(..., gt=1, description="Number of grid points")

    @property
    def dx(self) -> float:
        """Grid spacing."""
        return (self.x_max - self.x_min) / (self.n_points - 1)

    @property
    def L(self) -> float:
        """Domain length."""
        return self.x_max - self.x_min


class TimeConfig(BaseModel):
    """Time-stepping configuration."""
    dt: float = Field(..., gt=0, description="Time step size")
    n_steps: int = Field(..., gt=0, description="Number of time steps")

    @property
    def t_final(self) -> float:
        """Final simulation time."""
        return self.dt * self.n_steps


class BoundaryConfig(BaseModel):
    """Boundary condition configuration for a field."""
    type: Literal["periodic", "neumann", "dirichlet"] = Field(
        ..., description="Boundary condition type"
    )
    # For Neumann: specify gradient at boundaries
    gradient_left: Optional[float] = Field(
        None, description="Gradient at left boundary (Neumann)"
    )
    gradient_right: Optional[float] = Field(
        None, description="Gradient at right boundary (Neumann)"
    )
    # For Dirichlet: specify values at boundaries
    value_left: Optional[float] = Field(
        None, description="Value at left boundary (Dirichlet)"
    )
    value_right: Optional[float] = Field(
        None, description="Value at right boundary (Dirichlet)"
    )

    @model_validator(mode="after")
    def validate_bc_params(self) -> "BoundaryConfig":
        if self.type == "neumann":
            if self.gradient_left is None or self.gradient_right is None:
                raise ValueError(
                    "Neumann BC requires 'gradient_left' and 'gradient_right'"
                )
        elif self.type == "dirichlet":
            if self.value_left is None or self.value_right is None:
                raise ValueError(
                    "Dirichlet BC requires 'value_left' and 'value_right'"
                )
        return self


class FieldConfig(BaseModel):
    """Configuration for a single field (e.g., volume fraction)."""
    name: str = Field(..., description="Field name (e.g., 'alpha')")
    initial_condition: str = Field(
        ..., description="Expression for initial condition (uses 'x' as variable)"
    )
    boundary: BoundaryConfig = Field(..., description="Boundary condition")


class SolverConfig(BaseModel):
    """Spatial discretization (advection solver) configuration."""
    type: str = Field(..., description="Solver type (e.g., 'upwind')")


class TimeStepperConfig(BaseModel):
    """Time integration method configuration."""
    type: Literal["euler", "rk4"] = Field(
        "euler", description="Time-stepping method"
    )


class SharpeningConfig(BaseModel):
    """Interface sharpening configuration (optional)."""
    enabled: bool = Field(False, description="Enable sharpening")
    method: Literal["pm", "cl"] = Field("cl", description="Sharpening method")
    eps_target: float = Field(
        ..., gt=0, description="Target interface thickness"
    )
    strength: float = Field(1.0, gt=0, description="Sharpening strength (Gamma)")


class MonitorConfig(BaseModel):
    """Output monitor configuration."""
    type: Literal["console", "png", "pdf", "gif", "hdf5", "txt", "curve"] = Field(
        ..., description="Monitor type"
    )
    every_n_steps: Optional[int] = Field(
        None, gt=0, description="Output every N steps"
    )
    at_times: Optional[list[float]] = Field(
        None, description="Output at specific times"
    )
    field: Optional[str] = Field(
        None, description="Field to output (for image/gif/txt/curve)"
    )
    fields: Optional[list[str]] = Field(
        None, description="Fields to output (for hdf5)"
    )

    @model_validator(mode="after")
    def validate_output_trigger(self) -> "MonitorConfig":
        if self.every_n_steps is None and self.at_times is None:
            if self.type != "console":
                raise ValueError(
                    f"Monitor '{self.type}' requires 'every_n_steps' or 'at_times'"
                )
        return self


class OutputConfig(BaseModel):
    """Output configuration."""
    directory: str = Field("./results", description="Output directory")
    monitors: list[MonitorConfig] = Field(
        default_factory=list, description="List of output monitors"
    )


# ---------------------------------------------------------------------------
# Main configuration
# ---------------------------------------------------------------------------

class SimulationConfig(BaseModel):
    """Complete simulation configuration."""
    domain: DomainConfig
    time: TimeConfig
    velocity: list[float] = Field(..., description="Advection velocity vector")
    fields: list[FieldConfig] = Field(..., min_length=1)
    solver: SolverConfig
    timestepper: TimeStepperConfig = Field(default_factory=TimeStepperConfig)
    sharpening: Optional[SharpeningConfig] = Field(None)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @field_validator("velocity")
    @classmethod
    def validate_velocity_dimension(cls, v: list[float]) -> list[float]:
        if len(v) != 1:
            raise ValueError(
                f"velocity has {len(v)} components but domain is 1D (expected 1)"
            )
        return v

    @model_validator(mode="after")
    def validate_cfl_warning(self) -> "SimulationConfig":
        """Warn if CFL > 1."""
        u = abs(self.velocity[0])
        dt = self.time.dt
        dx = self.domain.dx
        cfl = u * dt / dx
        if cfl > 1.0:
            import warnings
            warnings.warn(
                f"CFL = {cfl:.3f} > 1, simulation may be unstable. "
                f"Consider reducing dt or increasing n_points."
            )
        return self

    @model_validator(mode="after")
    def validate_sharpening_eps(self) -> "SimulationConfig":
        """Validate sharpening eps_target is set if enabled."""
        if self.sharpening and self.sharpening.enabled:
            if self.sharpening.eps_target is None:
                raise ValueError(
                    "sharpening.eps_target is required when sharpening is enabled"
                )
        return self


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> SimulationConfig:
    """
    Load and validate a YAML configuration file.

    Parameters
    ----------
    path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    SimulationConfig
        Validated configuration object.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    pydantic.ValidationError
        If the configuration is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    return SimulationConfig.model_validate(raw)

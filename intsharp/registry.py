"""
Registry pattern for extensible solvers, timesteppers, sharpening, and monitors.

Usage:
    from intsharp.registry import register_solver, get_solver

    @register_solver("upwind")
    def upwind_advect(field, velocity, dx, dt, bc):
        ...

    solver_fn = get_solver("upwind")
"""

from typing import Any, Callable, TypeVar

# ---------------------------------------------------------------------------
# Generic registry
# ---------------------------------------------------------------------------

T = TypeVar("T", bound=Callable[..., Any])


class Registry:
    """Generic registry for named callables."""

    def __init__(self, name: str):
        self.name = name
        self._registry: dict[str, Callable[..., Any]] = {}

    def register(self, name: str) -> Callable[[T], T]:
        """Decorator to register a callable under a name."""
        def decorator(fn: T) -> T:
            if name in self._registry:
                raise ValueError(
                    f"{self.name} '{name}' is already registered"
                )
            self._registry[name] = fn
            return fn
        return decorator

    def get(self, name: str) -> Callable[..., Any]:
        """Get a registered callable by name."""
        if name not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise KeyError(
                f"Unknown {self.name}: '{name}'. Available: {available}"
            )
        return self._registry[name]

    def list_available(self) -> list[str]:
        """List all registered names."""
        return sorted(self._registry.keys())


# ---------------------------------------------------------------------------
# Specific registries
# ---------------------------------------------------------------------------

SOLVERS = Registry("solver")
TIMESTEPPERS = Registry("timestepper")
SHARPENING_METHODS = Registry("sharpening method")
MONITORS = Registry("monitor")


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def register_solver(name: str) -> Callable[[T], T]:
    """Decorator to register a solver."""
    return SOLVERS.register(name)


def get_solver(name: str) -> Callable[..., Any]:
    """Get a registered solver by name."""
    return SOLVERS.get(name)


def register_timestepper(name: str) -> Callable[[T], T]:
    """Decorator to register a timestepper."""
    return TIMESTEPPERS.register(name)


def get_timestepper(name: str) -> Callable[..., Any]:
    """Get a registered timestepper by name."""
    return TIMESTEPPERS.get(name)


def register_sharpening(name: str) -> Callable[[T], T]:
    """Decorator to register a sharpening method."""
    return SHARPENING_METHODS.register(name)


def get_sharpening(name: str) -> Callable[..., Any]:
    """Get a registered sharpening method by name."""
    return SHARPENING_METHODS.get(name)


def register_monitor(name: str) -> Callable[[T], T]:
    """Decorator to register a monitor."""
    return MONITORS.register(name)


def get_monitor(name: str) -> Callable[..., Any]:
    """Get a registered monitor by name."""
    return MONITORS.get(name)

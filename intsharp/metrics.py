"""
Interface-sharpening quality metrics.

Provides functions for measuring interface thickness (ε_char),
reconstructing the reference tanh profile, and computing L2 / L∞
shape errors against it.

Reference
---------
ε_char is derived from the half-space tanh profile

    α = ½[1 + tanh((R − r) / (2ε))]

so that  |r_{0.9} − r_{0.1}| = 4 ε ln 3, giving

    ε_char = |r_{0.9} − r_{0.1}| / (4 ln 3).

The reference hat profile (1-D symmetric about origin) is

    α_true(x; R, ε) = ½[tanh((R+x)/(2ε)) + tanh((R−x)/(2ε))].

Shape errors δ*_2 and δ*_∞ compare the numerical profile ψ against
α_true evaluated with the *measured* ε_char.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_LN3 = np.log(3.0)


# ── contour finding ────────────────────────────────────────────────


def _find_contour_1d(
    x: NDArray[np.float64],
    psi: NDArray[np.float64],
    level: float,
) -> list[float]:
    """Return all *x* locations where *psi* crosses *level* (linear interp)."""
    crossings: list[float] = []
    for i in range(len(psi) - 1):
        if (psi[i] - level) * (psi[i + 1] - level) <= 0 and psi[i] != psi[i + 1]:
            frac = (level - psi[i]) / (psi[i + 1] - psi[i])
            crossings.append(x[i] + frac * (x[i + 1] - x[i]))
    return crossings


# ── ε_char ─────────────────────────────────────────────────────────


def compute_eps_char(
    psi: NDArray[np.float64],
    x: NDArray[np.float64],
    R: float,
) -> float:
    r"""
    Characteristic interface thickness from the 0.1 / 0.9 contours.

    Measures at the **right** interface (closest to *x = R*).

    .. math::
        \varepsilon_{\mathrm{char}}
        = \frac{|r_{0.9} - r_{0.1}|}{4\ln 3}

    Parameters
    ----------
    psi : 1-D array
        Volume-fraction field.
    x : 1-D array
        Cell-centre coordinates (same length as *psi*).
    R : float
        Radius of the α = 0.5 contour (hat half-width).

    Returns
    -------
    float
        ε_char, or *nan* if contours cannot be found.
    """
    r01 = _find_contour_1d(x, psi, 0.1)
    r09 = _find_contour_1d(x, psi, 0.9)

    if not r01 or not r09:
        return np.nan

    # Pick the crossing closest to R for each level
    r01_right = min(r01, key=lambda v: abs(v - R))
    r09_right = min(r09, key=lambda v: abs(v - R))

    return abs(r09_right - r01_right) / (4.0 * _LN3)


# ── reference profile ─────────────────────────────────────────────


def compute_alpha_true(
    x: NDArray[np.float64],
    R: float,
    eps_char: float,
) -> NDArray[np.float64]:
    r"""
    Exact tanh hat profile with thickness ε_char.

    .. math::
        \alpha_{\mathrm{true}}(x)
        = \tfrac12\bigl[\tanh\!\bigl(\frac{R+x}{2\varepsilon}\bigr)
                       + \tanh\!\bigl(\frac{R-x}{2\varepsilon}\bigr)\bigr]

    Parameters
    ----------
    x : 1-D array
        Cell-centre coordinates.
    R : float
        Radius of the α = 0.5 contour.
    eps_char : float
        Interface thickness parameter.

    Returns
    -------
    1-D array
        Reference volume-fraction profile.
    """
    two_eps = 2.0 * eps_char
    return 0.5 * (np.tanh((R + x) / two_eps) + np.tanh((R - x) / two_eps))


# ── shape errors ───────────────────────────────────────────────────


def compute_delta_2(
    psi: NDArray[np.float64],
    alpha_true: NDArray[np.float64],
    dx: float,
) -> float:
    r"""
    L2 shape error.

    .. math::
        \delta^*_2
        = \Bigl(\sum_i [\alpha_i - \alpha_{\mathrm{true},i}]^2\,h\Bigr)^{1/2}
    """
    return float(np.sqrt(np.sum((psi - alpha_true) ** 2) * dx))


def compute_delta_inf(
    psi: NDArray[np.float64],
    alpha_true: NDArray[np.float64],
) -> float:
    r"""
    L∞ shape error.

    .. math::
        L_\infty = \max_i |\alpha_i - \alpha_{\mathrm{true},i}|
    """
    return float(np.max(np.abs(psi - alpha_true)))

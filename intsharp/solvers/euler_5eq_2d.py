"""
5-Equation Model for 2D Two-Phase Compressible Flow (Finite Volume).

State variables (cell-centered):
    1. alpha1_rho1 : partial density of phase 1 (alpha1 * rho1)
    2. alpha2_rho2 : partial density of phase 2 (alpha2 * rho2)
    3. rho_u       : mixture x-momentum
    4. rho_v       : mixture y-momentum
    5. E           : mixture total energy
    6. alpha1      : volume fraction of phase 1

Conservative transport is split along x and y directions using
modular 1D Riemann fluxes (AUSM+UP or HLLC) applied to face-normal states.
Gravity source terms can be added as a modular explicit source.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ..eos import sound_speed, mixture_sound_speed_wood
from ..flux_ausm import ausm_plus_up_flux_1d_with_v_riem
from ..flux_hllc import hllc_flux_1d_with_v_riem


BCType = Literal["transmissive", "reflective", "periodic"]
FluxType = Literal["ausm_plus_up", "hllc"]


@dataclass
class FiveEqState2D:
    """Container for 2D 5-equation model state."""

    alpha1_rho1: NDArray[np.float64]
    alpha2_rho2: NDArray[np.float64]
    rho_u: NDArray[np.float64]
    rho_v: NDArray[np.float64]
    E: NDArray[np.float64]
    alpha1: NDArray[np.float64]
    gamma1: float
    gamma2: float
    p_inf1: float = 0.0
    p_inf2: float = 0.0

    @property
    def rho(self) -> NDArray[np.float64]:
        return self.alpha1_rho1 + self.alpha2_rho2

    @property
    def alpha2(self) -> NDArray[np.float64]:
        return 1.0 - self.alpha1

    @property
    def rho1(self) -> NDArray[np.float64]:
        return self.alpha1_rho1 / (self.alpha1 + 1e-30)

    @property
    def rho2(self) -> NDArray[np.float64]:
        return self.alpha2_rho2 / (self.alpha2 + 1e-30)

    @property
    def u(self) -> NDArray[np.float64]:
        return self.rho_u / (self.rho + 1e-30)

    @property
    def v(self) -> NDArray[np.float64]:
        return self.rho_v / (self.rho + 1e-30)

    @property
    def p(self) -> NDArray[np.float64]:
        return compute_pressure_5eq_2d(
            self.alpha1_rho1,
            self.alpha2_rho2,
            self.rho_u,
            self.rho_v,
            self.E,
            self.alpha1,
            self.gamma1,
            self.gamma2,
            self.p_inf1,
            self.p_inf2,
        )

    @property
    def c(self) -> NDArray[np.float64]:
        rho1 = np.maximum(self.rho1, 1e-10)
        rho2 = np.maximum(self.rho2, 1e-10)
        p = self.p
        c1 = sound_speed(rho1, p, self.gamma1, self.p_inf1)
        c2 = sound_speed(rho2, p, self.gamma2, self.p_inf2)
        c_mix = mixture_sound_speed_wood(self.rho, self.alpha1, rho1, rho2, c1, c2)
        return np.maximum(np.minimum(c_mix, 1e10), 1e-10)

    def copy(self) -> "FiveEqState2D":
        return FiveEqState2D(
            alpha1_rho1=self.alpha1_rho1.copy(),
            alpha2_rho2=self.alpha2_rho2.copy(),
            rho_u=self.rho_u.copy(),
            rho_v=self.rho_v.copy(),
            E=self.E.copy(),
            alpha1=self.alpha1.copy(),
            gamma1=self.gamma1,
            gamma2=self.gamma2,
            p_inf1=self.p_inf1,
            p_inf2=self.p_inf2,
        )


def compute_pressure_5eq_2d(
    alpha1_rho1: NDArray[np.float64],
    alpha2_rho2: NDArray[np.float64],
    rho_u: NDArray[np.float64],
    rho_v: NDArray[np.float64],
    E: NDArray[np.float64],
    alpha1: NDArray[np.float64],
    gamma1: float,
    gamma2: float,
    p_inf1: float,
    p_inf2: float,
) -> NDArray[np.float64]:
    """Pressure closure from 5-equation conserved variables (2D)."""
    rho = alpha1_rho1 + alpha2_rho2
    u = rho_u / (rho + 1e-30)
    v = rho_v / (rho + 1e-30)
    kinetic = 0.5 * rho * (u * u + v * v)
    E_int = E - kinetic

    alpha2 = 1.0 - alpha1
    A = alpha1 / (gamma1 - 1.0) + alpha2 / (gamma2 - 1.0)
    B = alpha1 * gamma1 * p_inf1 / (gamma1 - 1.0) + alpha2 * gamma2 * p_inf2 / (gamma2 - 1.0)
    p = (E_int - B) / (A + 1e-30)
    return np.maximum(p, 1e-10)


def _pad_x(q: NDArray[np.float64], bc_x: BCType, is_normal_momentum: bool = False) -> NDArray[np.float64]:
    """Pad one ghost cell in x-direction (axis=1)."""
    if bc_x == "periodic":
        left = q[:, -1:]
        right = q[:, :1]
    elif bc_x == "transmissive":
        left = q[:, :1]
        right = q[:, -1:]
    elif bc_x == "reflective":
        left = q[:, :1].copy()
        right = q[:, -1:].copy()
        if is_normal_momentum:
            left *= -1.0
            right *= -1.0
    else:
        raise ValueError(f"Unknown x-BC: {bc_x}")
    return np.concatenate([left, q, right], axis=1)


def _pad_y(q: NDArray[np.float64], bc_y: BCType, is_normal_momentum: bool = False) -> NDArray[np.float64]:
    """Pad one ghost cell in y-direction (axis=0)."""
    if bc_y == "periodic":
        bottom = q[-1:, :]
        top = q[:1, :]
    elif bc_y == "transmissive":
        bottom = q[:1, :]
        top = q[-1:, :]
    elif bc_y == "reflective":
        bottom = q[:1, :].copy()
        top = q[-1:, :].copy()
        if is_normal_momentum:
            bottom *= -1.0
            top *= -1.0
    else:
        raise ValueError(f"Unknown y-BC: {bc_y}")
    return np.concatenate([bottom, q, top], axis=0)


def _compute_mixture_sound_speed(
    a1r1: NDArray[np.float64],
    a2r2: NDArray[np.float64],
    alpha1: NDArray[np.float64],
    p: NDArray[np.float64],
    gamma1: float,
    gamma2: float,
    p_inf1: float,
    p_inf2: float,
) -> NDArray[np.float64]:
    rho = a1r1 + a2r2
    alpha2 = 1.0 - alpha1
    rho1 = np.maximum(a1r1 / (alpha1 + 1e-30), 1e-10)
    rho2 = np.maximum(a2r2 / (alpha2 + 1e-30), 1e-10)
    c1 = sound_speed(rho1, p, gamma1, p_inf1)
    c2 = sound_speed(rho2, p, gamma2, p_inf2)
    c = mixture_sound_speed_wood(rho, alpha1, rho1, rho2, c1, c2)
    return np.maximum(np.minimum(c, 1e10), 1e-10)


def _flux_with_vriem(
    flux_calculator: FluxType,
    rho_L: NDArray[np.float64],
    u_L: NDArray[np.float64],
    p_L: NDArray[np.float64],
    E_L: NDArray[np.float64],
    c_L: NDArray[np.float64],
    rho_R: NDArray[np.float64],
    u_R: NDArray[np.float64],
    p_R: NDArray[np.float64],
    E_R: NDArray[np.float64],
    c_R: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    if flux_calculator == "ausm_plus_up":
        return ausm_plus_up_flux_1d_with_v_riem(
            rho_L, u_L, p_L, E_L, c_L, rho_R, u_R, p_R, E_R, c_R
        )
    if flux_calculator == "hllc":
        return hllc_flux_1d_with_v_riem(
            rho_L, u_L, p_L, E_L, c_L, rho_R, u_R, p_R, E_R, c_R
        )
    raise ValueError(f"Unknown flux calculator: {flux_calculator}")


def _sweep_x(
    state: FiveEqState2D,
    dx: float,
    dt: float,
    bc_x: BCType,
    flux_calculator: FluxType,
    K_alpha: float,
) -> FiveEqState2D:
    """One conservative x-direction FV sweep."""
    a1r1 = state.alpha1_rho1
    a2r2 = state.alpha2_rho2
    ru = state.rho_u
    rv = state.rho_v
    E = state.E
    a1 = state.alpha1
    g1, g2 = state.gamma1, state.gamma2
    pi1, pi2 = state.p_inf1, state.p_inf2

    # Ghost-padding in x
    a1r1e = _pad_x(a1r1, bc_x)
    a2r2e = _pad_x(a2r2, bc_x)
    rue = _pad_x(ru, bc_x, is_normal_momentum=True)
    rve = _pad_x(rv, bc_x, is_normal_momentum=False)
    Ee = _pad_x(E, bc_x)
    a1e = _pad_x(a1, bc_x)

    rhoe = a1r1e + a2r2e
    ue = rue / (rhoe + 1e-30)
    ve = rve / (rhoe + 1e-30)
    pe = compute_pressure_5eq_2d(a1r1e, a2r2e, rue, rve, Ee, a1e, g1, g2, pi1, pi2)
    ce = _compute_mixture_sound_speed(a1r1e, a2r2e, a1e, pe, g1, g2, pi1, pi2)

    # Interface states i+1/2 from neighboring cells (first-order)
    rho_L, rho_R = rhoe[:, :-1], rhoe[:, 1:]
    u_L, u_R = ue[:, :-1], ue[:, 1:]
    v_L, v_R = ve[:, :-1], ve[:, 1:]
    p_L, p_R = pe[:, :-1], pe[:, 1:]
    E_L, E_R = Ee[:, :-1], Ee[:, 1:]
    c_L, c_R = ce[:, :-1], ce[:, 1:]
    a1_L, a1_R = a1e[:, :-1], a1e[:, 1:]
    a1r1_L, a1r1_R = a1r1e[:, :-1], a1r1e[:, 1:]
    a2r2_L, a2r2_R = a2r2e[:, :-1], a2r2e[:, 1:]

    F_rho, F_ru, F_E, vriem = _flux_with_vriem(
        flux_calculator, rho_L, u_L, p_L, E_L, c_L, rho_R, u_R, p_R, E_R, c_R
    )

    # Tangential momentum flux (rho*v) in x-sweep
    v_up = np.where(vriem >= 0.0, v_L, v_R)
    F_rv = F_rho * v_up

    # Partial density fluxes
    a1r1_up = np.where(vriem >= 0.0, a1r1_L, a1r1_R)
    a2r2_up = np.where(vriem >= 0.0, a2r2_L, a2r2_R)
    F_a1r1 = a1r1_up * vriem
    F_a2r2 = a2r2_up * vriem

    # Non-conservative alpha advection with mild interface-stabilizing correction
    a1_up = np.where(vriem >= 0.0, a1_L, a1_R)
    F_a1 = a1_up * vriem
    a1_bar = 0.5 * (a1_L + a1_R)
    F_a1 += K_alpha * (u_L - u_R) * a1_bar * (1.0 - a1_bar)

    # Divergence update: interfaces in padded field are nx+1 wide
    s = dt / dx
    a1r1_new = a1r1 - s * (F_a1r1[:, 1:] - F_a1r1[:, :-1])
    a2r2_new = a2r2 - s * (F_a2r2[:, 1:] - F_a2r2[:, :-1])
    ru_new = ru - s * (F_ru[:, 1:] - F_ru[:, :-1])
    rv_new = rv - s * (F_rv[:, 1:] - F_rv[:, :-1])
    E_new = E - s * (F_E[:, 1:] - F_E[:, :-1])
    a1_new = a1 - s * (F_a1[:, 1:] - F_a1[:, :-1])

    return FiveEqState2D(
        alpha1_rho1=a1r1_new,
        alpha2_rho2=a2r2_new,
        rho_u=ru_new,
        rho_v=rv_new,
        E=E_new,
        alpha1=a1_new,
        gamma1=g1,
        gamma2=g2,
        p_inf1=pi1,
        p_inf2=pi2,
    )


def _sweep_y(
    state: FiveEqState2D,
    dy: float,
    dt: float,
    bc_y: BCType,
    flux_calculator: FluxType,
    K_alpha: float,
) -> FiveEqState2D:
    """One conservative y-direction FV sweep."""
    a1r1 = state.alpha1_rho1
    a2r2 = state.alpha2_rho2
    ru = state.rho_u
    rv = state.rho_v
    E = state.E
    a1 = state.alpha1
    g1, g2 = state.gamma1, state.gamma2
    pi1, pi2 = state.p_inf1, state.p_inf2

    # Ghost-padding in y
    a1r1e = _pad_y(a1r1, bc_y)
    a2r2e = _pad_y(a2r2, bc_y)
    rue = _pad_y(ru, bc_y, is_normal_momentum=False)
    rve = _pad_y(rv, bc_y, is_normal_momentum=True)
    Ee = _pad_y(E, bc_y)
    a1e = _pad_y(a1, bc_y)

    rhoe = a1r1e + a2r2e
    ue = rue / (rhoe + 1e-30)
    ve = rve / (rhoe + 1e-30)
    pe = compute_pressure_5eq_2d(a1r1e, a2r2e, rue, rve, Ee, a1e, g1, g2, pi1, pi2)
    ce = _compute_mixture_sound_speed(a1r1e, a2r2e, a1e, pe, g1, g2, pi1, pi2)

    # Interface states j+1/2 from neighboring cells (first-order)
    rho_L, rho_R = rhoe[:-1, :], rhoe[1:, :]
    v_L, v_R = ve[:-1, :], ve[1:, :]
    u_L, u_R = ue[:-1, :], ue[1:, :]
    p_L, p_R = pe[:-1, :], pe[1:, :]
    E_L, E_R = Ee[:-1, :], Ee[1:, :]
    c_L, c_R = ce[:-1, :], ce[1:, :]
    a1_L, a1_R = a1e[:-1, :], a1e[1:, :]
    a1r1_L, a1r1_R = a1r1e[:-1, :], a1r1e[1:, :]
    a2r2_L, a2r2_R = a2r2e[:-1, :], a2r2e[1:, :]

    # Re-use 1D normal flux with normal velocity = v
    F_rho, F_rv, F_E, vriem = _flux_with_vriem(
        flux_calculator, rho_L, v_L, p_L, E_L, c_L, rho_R, v_R, p_R, E_R, c_R
    )

    # Tangential momentum flux (rho*u) in y-sweep
    u_up = np.where(vriem >= 0.0, u_L, u_R)
    F_ru = F_rho * u_up

    # Partial density fluxes
    a1r1_up = np.where(vriem >= 0.0, a1r1_L, a1r1_R)
    a2r2_up = np.where(vriem >= 0.0, a2r2_L, a2r2_R)
    F_a1r1 = a1r1_up * vriem
    F_a2r2 = a2r2_up * vriem

    # Alpha advection with correction in normal direction (v)
    a1_up = np.where(vriem >= 0.0, a1_L, a1_R)
    F_a1 = a1_up * vriem
    a1_bar = 0.5 * (a1_L + a1_R)
    F_a1 += K_alpha * (v_L - v_R) * a1_bar * (1.0 - a1_bar)

    s = dt / dy
    a1r1_new = a1r1 - s * (F_a1r1[1:, :] - F_a1r1[:-1, :])
    a2r2_new = a2r2 - s * (F_a2r2[1:, :] - F_a2r2[:-1, :])
    ru_new = ru - s * (F_ru[1:, :] - F_ru[:-1, :])
    rv_new = rv - s * (F_rv[1:, :] - F_rv[:-1, :])
    E_new = E - s * (F_E[1:, :] - F_E[:-1, :])
    a1_new = a1 - s * (F_a1[1:, :] - F_a1[:-1, :])

    return FiveEqState2D(
        alpha1_rho1=a1r1_new,
        alpha2_rho2=a2r2_new,
        rho_u=ru_new,
        rho_v=rv_new,
        E=E_new,
        alpha1=a1_new,
        gamma1=g1,
        gamma2=g2,
        p_inf1=pi1,
        p_inf2=pi2,
    )


def _post_step_cleanup(state: FiveEqState2D) -> FiveEqState2D:
    """Positivity clamps + alpha normalization."""
    a1r1 = np.maximum(state.alpha1_rho1, 0.0)
    a2r2 = np.maximum(state.alpha2_rho2, 0.0)
    rho = a1r1 + a2r2
    a1 = np.where(rho > 1e-30, a1r1 / rho, state.alpha1)
    a1 = np.clip(a1, 1e-10, 1.0 - 1e-10)

    ru = state.rho_u
    rv = state.rho_v
    E = state.E
    p = compute_pressure_5eq_2d(
        a1r1, a2r2, ru, rv, E, a1, state.gamma1, state.gamma2, state.p_inf1, state.p_inf2
    )
    bad = p < 1e-10
    if np.any(bad):
        rho_b = np.maximum(rho[bad], 1e-30)
        kinetic_b = 0.5 * (ru[bad] * ru[bad] + rv[bad] * rv[bad]) / rho_b
        g_eff = a1[bad] * state.gamma1 + (1.0 - a1[bad]) * state.gamma2
        pinf_eff = a1[bad] * state.p_inf1 + (1.0 - a1[bad]) * state.p_inf2
        e_int_floor = (1e-8 + g_eff * pinf_eff) / (g_eff - 1.0)
        E[bad] = e_int_floor + kinetic_b

    return FiveEqState2D(
        alpha1_rho1=a1r1,
        alpha2_rho2=a2r2,
        rho_u=ru,
        rho_v=rv,
        E=E,
        alpha1=a1,
        gamma1=state.gamma1,
        gamma2=state.gamma2,
        p_inf1=state.p_inf1,
        p_inf2=state.p_inf2,
    )


def euler_step_5eq_2d(
    state: FiveEqState2D,
    dx: float,
    dy: float,
    dt: float,
    bc_x: BCType = "periodic",
    bc_y: BCType = "reflective",
    use_muscl: bool = False,
    flux_calculator: FluxType = "hllc",
    gravity_x: float = 0.0,
    gravity_y: float = 0.0,
    gravity_enabled: bool = False,
    K_alpha: float = 0.1,
) -> FiveEqState2D:
    """One operator-split FV step for 2D 5-equation model."""
    if use_muscl:
        warnings.warn("2D 5eq MUSCL reconstruction not yet implemented; using first-order faces.")

    s = state.copy()
    s = _sweep_x(s, dx=dx, dt=dt, bc_x=bc_x, flux_calculator=flux_calculator, K_alpha=K_alpha)
    s = _post_step_cleanup(s)
    s = _sweep_y(s, dy=dy, dt=dt, bc_y=bc_y, flux_calculator=flux_calculator, K_alpha=K_alpha)
    s = _post_step_cleanup(s)

    if gravity_enabled:
        rho = np.maximum(s.rho, 1e-30)
        u = s.rho_u / rho
        v = s.rho_v / rho
        s.rho_u = s.rho_u + dt * rho * gravity_x
        s.rho_v = s.rho_v + dt * rho * gravity_y
        s.E = s.E + dt * rho * (gravity_x * u + gravity_y * v)
        s = _post_step_cleanup(s)

    return s


def check_cfl_5eq_2d(state: FiveEqState2D, dx: float, dy: float, dt: float) -> float:
    """Compute a 2D CFL metric and warn if too large."""
    c = state.c
    u = np.abs(state.u) + c
    v = np.abs(state.v) + c
    cfl = float(np.max(u * dt / dx + v * dt / dy))
    if cfl > 1.0:
        warnings.warn(f"2D CFL={cfl:.3f} > 1.0: simulation may be unstable.")
    return cfl


def create_initial_state_rti_5eq_2d(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    rho1: float,
    rho2: float,
    gamma1: float,
    gamma2: float,
    p_inf1: float,
    p_inf2: float,
    interface_y0: float,
    alpha1_top: float,
    alpha1_bottom: float,
    perturbation_amplitude: float = 0.0,
    perturbation_mode_x: int = 1,
    interface_thickness: float = 0.0,
    p0: float = 1.0e5,
    u0: float = 0.0,
    v0: float = 0.0,
    gravity_y: float = -9.81,
    v_perturbation_amplitude: float = 0.0,
    v_perturbation_mode_x: int = 1,
    v_perturbation_scale_with_sound_speed: bool = False,
) -> FiveEqState2D:
    """
    Build an RTI-like hydrostatic initial state for the 2D 5eq model.

    Pressure is continuous across the interface and piecewise hydrostatic
    in each phase: p = p0 + rho_mix * g_y * (y - y_interface(x)).
    """
    x_min = float(np.min(X))
    x_max = float(np.max(X))
    Lx = max(x_max - x_min, 1e-30)
    phase = 2.0 * np.pi * perturbation_mode_x * (X - x_min) / Lx
    y_if = interface_y0 + perturbation_amplitude * np.cos(phase)

    if interface_thickness > 0.0:
        # Smooth alpha profile using tanh transition
        chi = (Y - y_if) / interface_thickness
        s = 0.5 * (1.0 + np.tanh(chi))
        alpha1 = alpha1_bottom + (alpha1_top - alpha1_bottom) * s
    else:
        alpha1 = np.where(Y >= y_if, alpha1_top, alpha1_bottom)

    alpha1 = np.clip(alpha1, 1e-10, 1.0 - 1e-10)
    alpha2 = 1.0 - alpha1
    a1r1 = alpha1 * rho1
    a2r2 = alpha2 * rho2
    rho_mix = a1r1 + a2r2

    # Piecewise hydrostatic pressure tied continuously to the local interface.
    p = p0 + rho_mix * gravity_y * (Y - y_if)
    p = np.maximum(p, 1e-6)

    # Optional x-periodic perturbation in v to mimic classical RTI setups.
    if v_perturbation_amplitude != 0.0:
        v_shape = np.cos(2.0 * np.pi * v_perturbation_mode_x * (X - x_min) / Lx)
        if v_perturbation_scale_with_sound_speed:
            gamma_eff = alpha1 * gamma1 + alpha2 * gamma2
            p_inf_eff = alpha1 * p_inf1 + alpha2 * p_inf2
            c_local = np.sqrt(np.maximum(gamma_eff * (p + p_inf_eff) / (rho_mix + 1e-30), 1e-12))
            v = v0 + v_perturbation_amplitude * c_local * v_shape
        else:
            v = v0 + v_perturbation_amplitude * v_shape
    else:
        v = np.full_like(rho_mix, v0, dtype=np.float64)

    ru = rho_mix * u0
    rv = rho_mix * v
    kinetic = 0.5 * rho_mix * (u0 * u0 + v * v)
    E_int = alpha1 * (p + gamma1 * p_inf1) / (gamma1 - 1.0) + alpha2 * (p + gamma2 * p_inf2) / (gamma2 - 1.0)
    E = E_int + kinetic

    return FiveEqState2D(
        alpha1_rho1=a1r1.astype(np.float64),
        alpha2_rho2=a2r2.astype(np.float64),
        rho_u=ru.astype(np.float64),
        rho_v=rv.astype(np.float64),
        E=E.astype(np.float64),
        alpha1=alpha1.astype(np.float64),
        gamma1=gamma1,
        gamma2=gamma2,
        p_inf1=p_inf1,
        p_inf2=p_inf2,
    )

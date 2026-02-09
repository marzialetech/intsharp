"""
1D nodal DG solver for single-phase compressible Euler.

Current scope:
- Polynomial orders P1-P3 on LGL nodes
- HLLE interface flux
- SSP-RK3 time integration
- Troubled-cell fallback to limited linear profile
- Positivity scaling of nodal modes toward the cell mean
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from ..eos import pressure_from_total_energy, sound_speed


def _lgl_nodes_weights(order: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return LGL nodes and quadrature weights for orders 1-3."""
    if order == 1:
        r = np.array([-1.0, 1.0], dtype=np.float64)
        w = np.array([1.0, 1.0], dtype=np.float64)
        return r, w
    if order == 2:
        r = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
        w = np.array([1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0], dtype=np.float64)
        return r, w
    if order == 3:
        a = 1.0 / np.sqrt(5.0)
        r = np.array([-1.0, -a, a, 1.0], dtype=np.float64)
        w = np.array([1.0 / 6.0, 5.0 / 6.0, 5.0 / 6.0, 1.0 / 6.0], dtype=np.float64)
        return r, w
    raise ValueError(f"Unsupported DG order: {order}. Supported: 1,2,3")


def _diff_matrix(nodes: NDArray[np.float64]) -> NDArray[np.float64]:
    """Construct nodal differentiation matrix D_ij = l'_j(x_i)."""
    n = len(nodes)
    lambdas = np.ones(n, dtype=np.float64)
    for j in range(n):
        for k in range(n):
            if k != j:
                lambdas[j] /= (nodes[j] - nodes[k])
    D = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = lambdas[j] / (lambdas[i] * (nodes[i] - nodes[j]))
        D[i, i] = -np.sum(D[i, :])
    return D


def _prim_from_cons(
    U: NDArray[np.float64],
    gamma: float,
    p_inf: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Convert conservative vector U=[rho, rho_u, E] to primitives."""
    rho = U[..., 0]
    rho_u = U[..., 1]
    E = U[..., 2]
    u = rho_u / (rho + 1e-30)
    p = pressure_from_total_energy(rho, rho_u, E, gamma, p_inf)
    return rho, u, p


def _flux_euler(U: NDArray[np.float64], gamma: float, p_inf: float) -> NDArray[np.float64]:
    """Physical Euler flux F(U)."""
    rho, u, p = _prim_from_cons(U, gamma, p_inf)
    F = np.empty_like(U)
    F[..., 0] = rho * u
    F[..., 1] = rho * u * u + p
    F[..., 2] = (U[..., 2] + p) * u
    return F


def _hlle_flux(
    U_L: NDArray[np.float64],
    U_R: NDArray[np.float64],
    gamma: float,
    p_inf: float,
) -> NDArray[np.float64]:
    """HLLE Riemann flux."""
    rho_L, u_L, p_L = _prim_from_cons(U_L, gamma, p_inf)
    rho_R, u_R, p_R = _prim_from_cons(U_R, gamma, p_inf)
    c_L = sound_speed(np.maximum(rho_L, 1e-30), np.maximum(p_L, 1e-30), gamma, p_inf)
    c_R = sound_speed(np.maximum(rho_R, 1e-30), np.maximum(p_R, 1e-30), gamma, p_inf)

    s_L = np.minimum(u_L - c_L, u_R - c_R)
    s_R = np.maximum(u_L + c_L, u_R + c_R)

    F_L = _flux_euler(U_L, gamma, p_inf)
    F_R = _flux_euler(U_R, gamma, p_inf)

    den = s_R - s_L
    den = np.where(np.abs(den) < 1e-30, 1e-30, den)
    F_hlle = (s_R[..., None] * F_L - s_L[..., None] * F_R + (s_L * s_R)[..., None] * (U_R - U_L)) / den[..., None]

    return np.where(
        (s_L >= 0.0)[..., None],
        F_L,
        np.where((s_R <= 0.0)[..., None], F_R, F_hlle),
    )


def _minmod3(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    c: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Componentwise minmod of three arrays."""
    same_sign = (np.sign(a) == np.sign(b)) & (np.sign(a) == np.sign(c))
    min_abs = np.minimum(np.abs(a), np.minimum(np.abs(b), np.abs(c)))
    return np.where(same_sign, np.sign(a) * min_abs, 0.0)


@dataclass
class DGEulerState1D:
    """Nodal DG state for 1D Euler."""
    U: NDArray[np.float64]  # shape (n_cells, n_nodes, 3)
    order: int
    gamma: float
    p_inf: float = 0.0

    def _w(self) -> NDArray[np.float64]:
        _, w = _lgl_nodes_weights(self.order)
        return w

    @property
    def U_bar(self) -> NDArray[np.float64]:
        """Cell-mean conserved states."""
        w = self._w()
        return 0.5 * np.einsum("j,ijk->ik", w, self.U)

    @property
    def rho(self) -> NDArray[np.float64]:
        return self.U_bar[:, 0]

    @property
    def rho_u(self) -> NDArray[np.float64]:
        return self.U_bar[:, 1]

    @property
    def E(self) -> NDArray[np.float64]:
        return self.U_bar[:, 2]

    @property
    def u(self) -> NDArray[np.float64]:
        return self.rho_u / (self.rho + 1e-30)

    @property
    def p(self) -> NDArray[np.float64]:
        return pressure_from_total_energy(self.rho, self.rho_u, self.E, self.gamma, self.p_inf)

    @property
    def c(self) -> NDArray[np.float64]:
        return sound_speed(np.maximum(self.rho, 1e-30), np.maximum(self.p, 1e-30), self.gamma, self.p_inf)

    def copy(self) -> "DGEulerState1D":
        return DGEulerState1D(
            U=self.U.copy(),
            order=self.order,
            gamma=self.gamma,
            p_inf=self.p_inf,
        )


def create_initial_state_riemann_dg_1d(
    x: NDArray[np.float64],
    x_discontinuity: float,
    rho_L: float,
    u_L: float,
    p_L: float,
    rho_R: float,
    u_R: float,
    p_R: float,
    gamma: float,
    p_inf: float = 0.0,
    order: int = 1,
) -> DGEulerState1D:
    """Create DG state with piecewise-constant polynomial per cell."""
    rho = np.where(x < x_discontinuity, rho_L, rho_R)
    u = np.where(x < x_discontinuity, u_L, u_R)
    p = np.where(x < x_discontinuity, p_L, p_R)
    E = (p + gamma * p_inf) / (gamma - 1.0) + 0.5 * rho * u * u
    Ubar = np.column_stack([rho, rho * u, E]).astype(np.float64)
    n = len(x)
    n_nodes = order + 1
    U = np.repeat(Ubar[:, None, :], n_nodes, axis=1)
    return DGEulerState1D(U=U, order=order, gamma=gamma, p_inf=p_inf)


def _boundary_ghost_state(
    U_b: NDArray[np.float64],
    bc_type: Literal["transmissive", "reflective", "periodic"],
    side: Literal["left", "right"],
    U_left_edge: NDArray[np.float64],
    U_right_edge: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Construct ghost state for boundary Riemann problem."""
    if bc_type == "transmissive":
        return U_b.copy()
    if bc_type == "reflective":
        g = U_b.copy()
        g[..., 1] *= -1.0
        return g
    if bc_type == "periodic":
        return U_right_edge.copy() if side == "left" else U_left_edge.copy()
    raise ValueError(f"Unknown BC type: {bc_type}")


def _apply_troubled_cell_fallback(
    U: NDArray[np.float64],
    order: int,
    bc_type: Literal["transmissive", "reflective", "periodic"],
    gamma: float,
    p_inf: float,
) -> NDArray[np.float64]:
    """
    If a cell is troubled, locally reduce to limited linear profile.

    This keeps P2/P3 robust for shock problems by using high order in smooth cells
    and P1-limited representation only where needed.
    """
    r, w = _lgl_nodes_weights(order)
    n = U.shape[0]
    Ubar = 0.5 * np.einsum("j,ijk->ik", w, U)

    # Neighbor means with BC extension
    if bc_type == "periodic":
        Ubar_ext = np.vstack([Ubar[-1][None, :], Ubar, Ubar[0][None, :]])
    elif bc_type == "reflective":
        gL = Ubar[0].copy()
        gR = Ubar[-1].copy()
        gL[1] *= -1.0
        gR[1] *= -1.0
        Ubar_ext = np.vstack([gL[None, :], Ubar, gR[None, :]])
    else:
        Ubar_ext = np.vstack([Ubar[0][None, :], Ubar, Ubar[-1][None, :]])

    rho_bar, _, p_bar = _prim_from_cons(Ubar, gamma, p_inf)
    rho_nb_min = np.minimum.reduce([Ubar_ext[0:n, 0], Ubar_ext[1:n + 1, 0], Ubar_ext[2:n + 2, 0]])
    rho_nb_max = np.maximum.reduce([Ubar_ext[0:n, 0], Ubar_ext[1:n + 1, 0], Ubar_ext[2:n + 2, 0]])

    _, _, pL = _prim_from_cons(U[:, 0, :], gamma, p_inf)
    _, _, pR = _prim_from_cons(U[:, -1, :], gamma, p_inf)
    p_ext = np.concatenate([[p_bar[0]], p_bar, [p_bar[-1]]]) if bc_type != "periodic" else np.concatenate([[p_bar[-1]], p_bar, [p_bar[0]]])
    p_nb_min = np.minimum.reduce([p_ext[0:n], p_ext[1:n + 1], p_ext[2:n + 2]])
    p_nb_max = np.maximum.reduce([p_ext[0:n], p_ext[1:n + 1], p_ext[2:n + 2]])

    eps = 1e-10
    troubled = (
        (U[:, 0, 0] < rho_nb_min - eps) | (U[:, 0, 0] > rho_nb_max + eps) |
        (U[:, -1, 0] < rho_nb_min - eps) | (U[:, -1, 0] > rho_nb_max + eps) |
        (pL < p_nb_min - eps) | (pL > p_nb_max + eps) |
        (pR < p_nb_min - eps) | (pR > p_nb_max + eps)
    )
    if not np.any(troubled):
        return U

    Unew = U.copy()
    dminus = Ubar_ext[1:n + 1, :] - Ubar_ext[0:n, :]
    dplus = Ubar_ext[2:n + 2, :] - Ubar_ext[1:n + 1, :]
    # Raw linear slope estimate from endpoint nodes
    s_raw = 0.5 * (U[:, -1, :] - U[:, 0, :])
    s_lim = _minmod3(s_raw, dminus, dplus)

    idx = np.where(troubled)[0]
    Unew[idx, :, :] = Ubar[idx, None, :] + r[None, :, None] * s_lim[idx, None, :]
    return Unew


def _positivity_scale_modes(
    U: NDArray[np.float64],
    order: int,
    gamma: float,
    p_inf: float,
    eps_rho: float = 1e-10,
    eps_p: float = 1e-10,
) -> NDArray[np.float64]:
    """Scale nodal modes toward cell mean so rho,p remain positive at nodes."""
    _, w = _lgl_nodes_weights(order)
    Unew = U.copy()
    Ubar = 0.5 * np.einsum("j,ijk->ik", w, Unew)

    for i in range(Unew.shape[0]):
        mean = Ubar[i]
        dU = Unew[i] - mean[None, :]
        theta = 1.0

        # Density positivity
        rho_vals = mean[0] + dU[:, 0]
        bad_rho = rho_vals < eps_rho
        if np.any(bad_rho):
            den = mean[0] - rho_vals[bad_rho]
            theta_rho = np.min((mean[0] - eps_rho) / (den + 1e-30))
            theta = min(theta, float(theta_rho))

        # Pressure positivity via bisection
        def min_p_for_theta(th: float) -> float:
            vals = mean[None, :] + th * dU
            rho = np.maximum(vals[:, 0], 1e-30)
            p = pressure_from_total_energy(rho, vals[:, 1], vals[:, 2], gamma, p_inf)
            return float(np.min(p))

        if min_p_for_theta(theta) < eps_p:
            lo, hi = 0.0, theta
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if min_p_for_theta(mid) >= eps_p:
                    lo = mid
                else:
                    hi = mid
            theta = lo

        theta = float(np.clip(theta, 0.0, 1.0))
        Unew[i] = mean[None, :] + theta * dU

    return Unew


def _dg_rhs(
    U: NDArray[np.float64],
    order: int,
    dx: float,
    gamma: float,
    p_inf: float,
    bc_type: Literal["transmissive", "reflective", "periodic"],
) -> NDArray[np.float64]:
    """Semi-discrete DG RHS for nodal LGL collocation."""
    _, w = _lgl_nodes_weights(order)
    D = _diff_matrix(_lgl_nodes_weights(order)[0])
    n = U.shape[0]

    F = _flux_euler(U, gamma, p_inf)  # (n, p+1, 3)
    rhs = -(2.0 / dx) * np.einsum("jm,imk->ijk", D, F)

    U_left = U[:, 0, :]
    U_right = U[:, -1, :]

    ghost_L = _boundary_ghost_state(U_left[0], bc_type, "left", U_left[0], U_right[-1])[None, :]
    ghost_R = _boundary_ghost_state(U_right[-1], bc_type, "right", U_left[0], U_right[-1])[None, :]

    UL_if = np.vstack([ghost_L, U_right])   # interface left states i=0..n
    UR_if = np.vstack([U_left, ghost_R])    # interface right states i=0..n
    F_star = _hlle_flux(UL_if, UR_if, gamma, p_inf)

    # Strong-form boundary corrections (only endpoint nodes)
    F_L_star = F_star[0:n, :]
    F_R_star = F_star[1:n + 1, :]
    rhs[:, 0, :] += (2.0 / dx) * (F_L_star - F[:, 0, :]) / w[0]
    rhs[:, -1, :] -= (2.0 / dx) * (F_R_star - F[:, -1, :]) / w[-1]

    return rhs


def dg_step_1d(
    state: DGEulerState1D,
    dx: float,
    dt: float,
    bc_type: Literal["transmissive", "reflective", "periodic"] = "transmissive",
    use_limiter: bool = True,
    use_positivity: bool = True,
) -> DGEulerState1D:
    """One SSP-RK3 step for nodal DG Euler."""
    order = state.order
    gamma, p_inf = state.gamma, state.p_inf

    def stage_condition(Uin: NDArray[np.float64]) -> NDArray[np.float64]:
        U = Uin.copy()
        # Ensure positive means first
        _, w = _lgl_nodes_weights(order)
        Ubar = 0.5 * np.einsum("j,ijk->ik", w, U)
        Ubar[:, 0] = np.maximum(Ubar[:, 0], 1e-10)
        p_bar = pressure_from_total_energy(Ubar[:, 0], Ubar[:, 1], Ubar[:, 2], gamma, p_inf)
        bad = p_bar < 1e-10
        if np.any(bad):
            rho = Ubar[bad, 0]
            rho_u = Ubar[bad, 1]
            kinetic = 0.5 * rho_u * rho_u / (rho + 1e-30)
            rho_e = (1e-10 + gamma * p_inf) / (gamma - 1.0)
            Ubar[bad, 2] = rho_e + kinetic
        # Re-center nodal values around corrected mean
        U = U - (0.5 * np.einsum("j,ijk->ik", w, U))[:, None, :] + Ubar[:, None, :]

        if use_limiter:
            U = _apply_troubled_cell_fallback(U, order, bc_type, gamma, p_inf)
        if use_positivity:
            U = _positivity_scale_modes(U, order, gamma, p_inf)
        return U

    U0 = state.U.copy()

    U0s = stage_condition(U0)
    k1 = _dg_rhs(U0s, order, dx, gamma, p_inf, bc_type)
    U1 = U0 + dt * k1

    U1s = stage_condition(U1)
    k2 = _dg_rhs(U1s, order, dx, gamma, p_inf, bc_type)
    U2 = 0.75 * U0 + 0.25 * (U1 + dt * k2)

    U2s = stage_condition(U2)
    k3 = _dg_rhs(U2s, order, dx, gamma, p_inf, bc_type)
    Unp1 = (1.0 / 3.0) * U0 + (2.0 / 3.0) * (U2 + dt * k3)

    Unp1 = stage_condition(Unp1)
    return DGEulerState1D(U=Unp1, order=order, gamma=gamma, p_inf=p_inf)

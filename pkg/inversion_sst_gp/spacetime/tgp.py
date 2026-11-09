"""
Phase B, stage 1: a space-time Vecchia GP on the tracer T itself, with derivative functionals.

Each row is `(x, y, t, wT, wx, wy, wt)` and represents the linear functional

    wT * T + wx * dT/dx + wy * dT/dy + wt * dT/dt         at (x, y, t).

Observations are `(1, 0, 0, 0)` (T plus noise); predictions of the derivatives that feed the velocity model are
`(0, 1, 0, 0)` etc. The prior is a sum of up to two separable components, each Matern-5/2 in space (isotropic) times a
Matern-nu_t in time (nu_t = 3/2 or 5/2, so first derivatives in time exist):

    K(dx, dy, tau) = sum_c  sigma_c^2 * M52(r / l_c) * Mnu(tau / tau0_c),     dx = x1 - x2 etc.

Derivative covariances come from closed-form partial derivatives of K, with the sign rule that a derivative with respect to
the SECOND point's coordinate is minus the derivative with respect to the lag (the spatial parts reuse
browseGP's `_matern52_AB_njit`: dK/ddx = -dx A, d2K/ddx2 = -A + coeff dx^2, d2K/ddxddy = coeff dx dy).

cov_params layout (see `pack_params_T`): (sigma1, l1, tau1, sigma2, l2, tau2, nu_t). Set sigma2 = 0 for one component.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numba import njit

from browseGP.covariance.helmholtz_numba import _matern52_AB_njit, _scalar_matern_njit

from . import kernels as K
from .model import SpaceTimeGP

N_PARAMS_T = 7
FIELDS_T = {'T': (1, 0, 0, 0), 'Tx': (0, 1, 0, 0), 'Ty': (0, 0, 1, 0), 'Tt': (0, 0, 0, 1)}


def pack_params_T(sigma1, l1, tau1, sigma2=0.0, l2=1.0, tau2=1.0, nu_t=2.5):
    return np.array([sigma1, l1, tau1, sigma2, l2, tau2, nu_t], dtype=float)


def build_T_points(x, y, t):
    """Observation rows: T itself."""
    x, y, t = (np.asarray(v, dtype=float).ravel() for v in (x, y, t))
    n = x.size
    return np.column_stack([x, y, t, np.ones(n), np.zeros(n), np.zeros(n), np.zeros(n)])


def build_T_pred_points(x, y, t, field):
    w = FIELDS_T[field]
    x, y, t = (np.asarray(v, dtype=float).ravel() for v in (x, y, t))
    n = x.size
    return np.column_stack([x, y, t] + [np.full(n, float(wi)) for wi in w])


@njit(cache=True, inline='always')
def _time_derivs(tau, tau0, nu):
    """(K, dK/dtau, d2K/dtau2) of the unit-variance Matern-nu temporal kernel, nu in {3/2, 5/2}."""
    a = abs(tau)
    if nu == 1.5:
        z = np.sqrt(3.0) * a / tau0
        e = np.exp(-z)
        return (1.0 + z) * e, -(3.0 / tau0 ** 2) * tau * e, -(3.0 / tau0 ** 2) * (1.0 - z) * e
    z = np.sqrt(5.0) * a / tau0
    e = np.exp(-z)
    return ((1.0 + z + z * z / 3.0) * e, -(5.0 / (3.0 * tau0 ** 2)) * tau * (1.0 + z) * e,
            -(5.0 / (3.0 * tau0 ** 2)) * (1.0 + z - z * z) * e)


@njit(cache=True)
def T_covfunc(x1, x2, p):
    """Covariance between two functional rows `(x, y, t, wT, wx, wy, wt)`. See module docstring."""
    dx = x1[0] - x2[0]
    dy = x1[1] - x2[1]
    tau = x1[2] - x2[2]
    r = np.hypot(dx, dy)
    a0, ax, ay, at = x1[3], x1[4], x1[5], x1[6]
    b0, bx, by, bt = x2[3], x2[4], x2[5], x2[6]
    nu = p[6]
    total = 0.0
    for c in range(2):
        sigma, ell, tau0 = p[3 * c], p[3 * c + 1], p[3 * c + 2]
        if sigma == 0.0:
            continue
        A, _, coeff = _matern52_AB_njit(r, sigma, ell)
        Ks = _scalar_matern_njit(r, sigma, ell)
        Ks10, Ks01 = -dx * A, -dy * A
        Ks20, Ks02, Ks11 = -A + coeff * dx * dx, -A + coeff * dy * dy, coeff * dx * dy
        Kt, Kt1, Kt2 = _time_derivs(tau, tau0, nu)
        # (point-1 operator i, point-2 operator j); a derivative on point 2 carries a factor -1
        v = a0 * b0 * Ks * Kt
        v += a0 * bx * (-Ks10 * Kt) + a0 * by * (-Ks01 * Kt) + a0 * bt * (-Ks * Kt1)
        v += ax * b0 * (Ks10 * Kt) + ay * b0 * (Ks01 * Kt) + at * b0 * (Ks * Kt1)
        v += ax * bx * (-Ks20 * Kt) + ay * by * (-Ks02 * Kt) + (ax * by + ay * bx) * (-Ks11 * Kt)
        v += (ax * bt + at * bx) * (-Ks10 * Kt1) + (ay * bt + at * by) * (-Ks01 * Kt1)
        v += at * bt * (-Ks * Kt2)
        total += v
    return total


# ---------------------------------------------------------------------------
# Rough-in-space variant for the resolved-gradient mode (fields T and dT/dt only)
# ---------------------------------------------------------------------------

N_PARAMS_T_RES = 9


def pack_params_T_res(sigma1, l1, tau1, sigma2=0.0, l2=1.0, tau2=1.0, nu_t=2.5, nu_s1=2.5, nu_s2=2.5):
    """Like `pack_params_T` plus the structural SPATIAL smoothness of each component (0.5, 1.5 or 2.5)."""
    return np.array([sigma1, l1, tau1, sigma2, l2, tau2, nu_t, nu_s1, nu_s2], dtype=float)


@njit(cache=True, inline='always')
def _matern_space(r, sigma, ell, nu):
    """sigma^2 * Matern-nu(r / ell) for nu in {1/2, 3/2, 5/2} (unit lengthscale convention of the standard Matern)."""
    if nu == 0.5:
        return sigma * sigma * np.exp(-r / ell)
    if nu == 1.5:
        z = np.sqrt(3.0) * r / ell
        return sigma * sigma * (1.0 + z) * np.exp(-z)
    z = np.sqrt(5.0) * r / ell
    return sigma * sigma * (1.0 + z + z * z / 3.0) * np.exp(-z)


@njit(cache=True)
def T_covfunc_res(x1, x2, p):
    """Covariance of rows `(x, y, t, wT, wx, wy, wt)` with wx = wy = 0 (only T and dT/dt), spatial Matern smoothness per component.

    p = (sigma1, l1, tau1, sigma2, l2, tau2, nu_t, nu_s1, nu_s2); components use temporal Matern-nu_t and spatial Matern-nu_s{c}.
    """
    dx = x1[0] - x2[0]
    dy = x1[1] - x2[1]
    tau = x1[2] - x2[2]
    r = np.hypot(dx, dy)
    a0, at = x1[3], x1[6]
    b0, bt = x2[3], x2[6]
    nu = p[6]
    total = 0.0
    for c in range(2):
        sigma, ell, tau0 = p[3 * c], p[3 * c + 1], p[3 * c + 2]
        if sigma == 0.0:
            continue
        Ks = _matern_space(r, sigma, ell, p[7 + c])
        Kt, Kt1, Kt2 = _time_derivs(tau, tau0, nu)
        total += (a0 * b0 * Kt - a0 * bt * Kt1 + at * b0 * Kt1 - at * bt * Kt2) * Ks
    return total


# ---------------------------------------------------------------------------
# Low-rank terms for T (value and first derivatives of the basis are needed)
# ---------------------------------------------------------------------------

@dataclass
class TLowRankTerm:
    """`T_term = Phi(x, y, t) @ w`, `w ~ N(0, scale^2 I)`; `phi` returns `(value, d/dx, d/dy, d/dt)`, each `(n, k)`."""
    name: str
    phi: Callable
    scale: float = 1.0

    def basis(self, rows):
        f, fx, fy, ft = self.phi(rows[:, 0], rows[:, 1], rows[:, 2])
        out = (rows[:, 3:4] * f + rows[:, 4:5] * fx + rows[:, 5:6] * fy + rows[:, 6:7] * ft)
        return self.scale * out


def poly_mean(L=50e3, T0=86400.0):
    """Constant + linear in x, y, t (large-scale trend), coordinates scaled by `L`, `T0`."""
    def phi(x, y, t):
        one, zero = np.ones_like(x), np.zeros_like(x)
        f = np.column_stack([one, x / L, y / L, t / T0])
        fx = np.column_stack([zero, one / L, zero, zero])
        fy = np.column_stack([zero, zero, one / L, zero])
        ft = np.column_stack([zero, zero, zero, one / T0])
        return f, fx, fy, ft
    return phi


def diurnal_T(n_harm=3, period=86400.0, decay=1.0):
    """[cos k w t, sin k w t] harmonics (the diurnal cycle in T itself), column std 1/k**decay."""
    w = 2 * np.pi / period

    def phi(x, y, t):
        cols, dcols = [], []
        for k in range(1, n_harm + 1):
            s = 1.0 / k ** decay
            cols += [s * np.cos(k * w * t), s * np.sin(k * w * t)]
            dcols += [-s * k * w * np.sin(k * w * t), s * k * w * np.cos(k * w * t)]
        f = np.column_stack(cols)
        zero = np.zeros_like(f)
        return f, zero, zero, np.column_stack(dcols)
    return phi


class TGP(SpaceTimeGP):
    """`SpaceTimeGP` for the tracer: data rows are T observations, predictions are T / dT/dx / dT/dy / dT/dt."""

    def __init__(self, pts, T, cov_params, noise, lowrank=(), covfunc=None, **kw):
        super().__init__(pts, T, cov_params, noise, lowrank=lowrank, covfunc=T_covfunc if covfunc is None else covfunc, **kw)

    def _pred_rows(self, field, x, y, t):
        return build_T_pred_points(x, y, t, field)

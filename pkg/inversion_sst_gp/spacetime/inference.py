"""
Observations, Vecchia operator construction (with a locked neighbour array) and
prediction for the space-time transport-equation model. See `kernels.py` for the
covariance and `notes/00_spacetime_vecchia_plan.md` for the design.

Scaling: gpvecchia adds an absolute jitter, so `dTdt` (~1e-6 K/s) is divided by
`zscale` and the gradients likewise (`a = -dTdx / zscale`). Everything fitted lives
in these scaled units; S predictions are multiplied back by `zscale`.

Neighbour array: ordering and nearest neighbours are computed once from guessed
length scales `(Lx, Lt)` (coordinates are scaled `(x/Lx, y/Lx, t/Lt)` for the search
only -- the covariance still sees raw metres and seconds) and then reused for every
hyperparameter update via `VecchiaOperator.update_hyperparams`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from gpvecchia.vecchia_utils import find_nn, get_pred_nn, maximin_order

from . import kernels as K
from .fastop import FastVecchiaOperator


@dataclass
class Observations:
    pts: np.ndarray          # (N, 6) rows (x, y, t, a, b, c), scaled
    z: np.ndarray            # (N,) dTdt / zscale
    index: tuple             # np.nonzero-style indices into the (time, y, x) cube
    shape: tuple
    zscale: float


def make_observations(ds, grad_source='clean', dTdt_source='obs', zscale=1e-6, time_slice=None, resid=False):
    """Valid-pixel observations from a synthetic (or Himawari-style) dataset.

    `grad_source` / `dTdt_source`: 'clean' (from true T) or 'obs' (from noisy T). `dTdt_source='equation'` is a
    diagnostic that removes the finite-difference time error: `dTdt = S - u dTdx - v dTdy` from the TRUE u, v, S and
    the chosen gradients, plus iid noise with the std the hourly difference of noisy T would give
    (`sigma_T / (sqrt(2) dt)`). The model is then exactly specified.
    """
    g = '_clean' if grad_source == 'clean' else ''
    d = '_clean' if dTdt_source in ('clean', 'equation') else ''
    dTdt, dTdx, dTdy = ds['dTdt' + d].values, ds['dTdx' + g].values, ds['dTdy' + g].values
    if dTdt_source == 'equation':
        eq = ds['S_true'].values - ds['u_true'].values * dTdx - ds['v_true'].values * dTdy
        noise = np.random.default_rng(11).standard_normal(eq.shape) * ds.attrs['sigma_T'] / (np.sqrt(2.0) * 3600.0)
        dTdt = np.where(np.isfinite(dTdt), eq + noise, np.nan)
    valid = np.isfinite(dTdt) & np.isfinite(dTdx) & np.isfinite(dTdy)
    if time_slice is not None:
        mask_t = np.zeros(valid.shape[0], bool); mask_t[time_slice] = True
        valid &= mask_t[:, None, None]
    idx = np.nonzero(valid)
    t = ds['time'].values[idx[0]]; y = ds['y'].values[idx[1]]; x = ds['x'].values[idx[2]]
    pts = K.build_points(x, y, t, dTdx[idx] / zscale, dTdy[idx] / zscale, resid=resid)
    return Observations(pts=pts, z=dTdt[idx] / zscale, index=idx, shape=valid.shape, zscale=zscale)


def search_coords(pts, Lx, Lt):
    """Coordinates used only for ordering / neighbour search."""
    return np.column_stack([pts[:, 0] / Lx, pts[:, 1] / Lx, pts[:, 2] / Lt])


def build_operator(pts, cov_params, noise, nnum=30, Lx=20e3, Lt=6 * 3600.0, jitter=1e-8,
                   order_idx=None, nn_array=None, covfunc=None):
    """Vecchia operator with ordering/neighbours locked from guessed scales (Lx, Lt)."""
    s = search_coords(pts, Lx, Lt)
    if order_idx is None:
        order_idx = maximin_order(s)
    if nn_array is None:
        nn_array = find_nn(s[order_idx], nnum, verbose=False)
    covfunc = K.transport_obs_covfunc if covfunc is None else covfunc
    op = FastVecchiaOperator(pts, covfunc, cov_params, noise=noise, nnum=nnum,
                         nn_cols=K.COORD_COLS, order_idx=order_idx, nn_array=nn_array,
                         jitter=jitter)
    op.search_scales = (Lx, Lt)
    return op


def predict(op, z, field, x, y, t, nnum_pred=None, method='sklearn'):
    """Posterior mean / variance of 'u', 'v' or 'S' (scaled units for S) at (x, y, t)."""
    Lx, Lt = op.search_scales
    Xp = K.build_pred_points(x, y, t, field, resid=op.X.shape[1] > 6)
    m = op.nnum if nnum_pred is None else nnum_pred
    nn_pred = get_pred_nn(search_coords(Xp, Lx, Lt), search_coords(op.Xo, Lx, Lt), m=m, method=method)
    return op.predict_local(z, Xp, nn_pred_array=nn_pred)


def loglik(op, z):
    """Vecchia log-likelihood (without the -N/2 log 2pi constant)."""
    return float(np.squeeze(op.loglik(z)))

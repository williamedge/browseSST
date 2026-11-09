"""
Space-time GP = Vecchia block (Helmholtz (u, v) + Matern S) + low-rank terms, combined with
`datide.VecchiaMultiscaleGP` (Woodbury on top of the Vecchia data covariance).

Low-rank terms are bases over (x, y, t) for any of the latent fields. Because every
observation / prediction row carries its coefficients `(a, b, c)`, one function gives the
basis rows for both the data and any prediction target:

    U_term(row) = a * Phi_u(x, y, t) + b * Phi_v(x, y, t) + c * Phi_S(x, y, t)

(predicting u uses `(a, b, c) = (1, 0, 0)`, etc.). The diurnal S cycle is one such term; the
browseGP tidal plane model (shared u, v coefficients) is another and plugs in the same way.

Prediction follows `VecchiaMultiscaleGP.predictive_mean`: condition the Vecchia block locally
on the data with the low-rank estimate removed, then add the low-rank part. The returned
variance is the Vecchia block's conditional variance only (it omits the uncertainty of the
low-rank coefficients) -- see `notes/` for the open item.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
from datide import LowRankScale, VecchiaMultiscaleGP
from gpvecchia.vecchia_utils import get_pred_nn

from . import inference as I
from . import kernels as K

DAY = 86400.0


# ---------------------------------------------------------------------------
# Low-rank terms
# ---------------------------------------------------------------------------

def diurnal_harmonics(n_harm=4, period=DAY, decay=1.0):
    """`(x, y, t) -> (n, 1 + 2 n_harm)`: [1, cos kwt, sin kwt], column std `1 / k**decay` (std 1 for the constant).

    A rectified half-sine (daytime heating, flat night) has harmonics at every integer multiple
    of the daily frequency, so a few harmonics are needed, not just the fundamental.
    """
    w = 2 * np.pi / period

    def phi(x, y, t):
        cols = [np.ones_like(t)]
        for k in range(1, n_harm + 1):
            s = 1.0 / k ** decay
            cols += [s * np.cos(k * w * t), s * np.sin(k * w * t)]
        return np.column_stack(cols)
    return phi


def diurnal_harmonics_daily(knots, n_harm=4, period=DAY, decay=1.0, spacing=DAY):
    """`(x, y, t) -> (n, J * (1 + 2 n_harm))`: the diurnal harmonics of `diurnal_harmonics` multiplied by linear hat functions of time centred on `knots` (spacing `spacing`), so that the
    amplitude of every harmonic can change from day to day (a spatially uniform S whose daily cycle is not the same every day). The hats form a partition of unity between the first and last knot."""
    base = diurnal_harmonics(n_harm, period, decay)
    knots = np.asarray(knots, float)

    def phi(x, y, t):
        t = np.asarray(t, float)
        B = base(x, y, t)
        W = np.maximum(0.0, 1.0 - np.abs(t[:, None] - knots[None, :]) / spacing)
        return (W[:, :, None] * B[:, None, :]).reshape(t.size, -1)
    return phi


@dataclass
class LowRankTerm:
    """A prior `field = Phi(x, y, t) @ w`, `w ~ N(0, scale**2 I)`, for one or more latent fields."""
    name: str
    phi: Dict[str, Callable]        # any of 'u', 'v', 'S' -> callable (x, y, t) -> (n, k); same k for all
    scale: float = 1.0              # prior std multiplier (a hyperparameter)

    def basis(self, rows):
        x, y, t = rows[:, 0], rows[:, 1], rows[:, 2]
        out = 0.0
        for fld, col in (('u', 3), ('v', 4), ('S', 5)):
            if fld in self.phi:
                out = out + rows[:, col:col + 1] * self.phi[fld](x, y, t)
        return self.scale * out


# ---------------------------------------------------------------------------
# GP
# ---------------------------------------------------------------------------

class _DataGP:
    """Minimal adapter exposing what `VecchiaMultiscaleGP` needs from its Vecchia block."""

    def __init__(self, op, z):
        self.op, self.yaug = op, z

    def update_hyperparams(self, cov_params_mo=None, noise=None):
        self.op.update_hyperparams(covparams=cov_params_mo, noise=noise)


class SpaceTimeGP:
    def __init__(self, pts, z, cov_params, noise, lowrank=(), nnum=100, Lx=20e3, Lt=24 * 3600.0,
                 jitter=1e-8, order_idx=None, nn_array=None, covfunc=None):
        self.pts, self.z = pts, np.asarray(z, float)
        self.terms = list(lowrank)
        self.op = I.build_operator(pts, cov_params, noise, nnum=nnum, Lx=Lx, Lt=Lt, jitter=jitter,
                                   order_idx=order_idx, nn_array=nn_array, covfunc=covfunc)
        self.gp = VecchiaMultiscaleGP(_DataGP(self.op, self.z), self._scales())

    def _scales(self):
        return [LowRankScale(t.basis(self.pts)) for t in self.terms]

    def _pred_rows(self, field, x, y, t):
        """Prediction rows for `field` (overridden by models with a different row layout, e.g. `tgp.TGP`)."""
        return K.build_pred_points(x, y, t, field, resid=self.pts.shape[1] > 6)

    def update(self, cov_params=None, noise=None, lowrank_scales: Optional[Dict[str, float]] = None):
        """Refit at new hyperparameters, reusing the locked ordering / neighbour array."""
        scales = None
        if lowrank_scales is not None:
            for t in self.terms:
                t.scale = lowrank_scales.get(t.name, t.scale)
            scales = self._scales()
        self.gp.update_hyperparams(cov_params_mo=cov_params, noise=noise, lowrank_scales=scales)

    def loglik(self):
        """Log marginal likelihood (includes the -N/2 log 2 pi constant)."""
        return self.gp.log_marg_likelihood()

    def pred_neighbours(self, x, y, t, nnum_pred=None, method='sklearn'):
        """Training-neighbour array for prediction locations (depends on (x, y, t) only, so it can be shared by fields)."""
        Lx, Lt = self.op.search_scales
        xyz = np.column_stack([np.asarray(v, float).ravel() for v in (x, y, t)])
        m = self.op.nnum if nnum_pred is None else nnum_pred
        return get_pred_nn(I.search_coords(xyz, Lx, Lt), I.search_coords(self.op.Xo, Lx, Lt), m=m, method=method)

    def predict(self, field, x, y, t, nnum_pred=None, method='sklearn', z=None, nn_array=None):
        """Posterior mean and (Vecchia-block) variance of the named field ('u', 'v', 'S'; S in scaled units).

        z : optional replacement data vector (same length as the training data), e.g. a simulated data set in the
            Matheron-rule sampler (`impute.py`); the hyperparameters and neighbours are unchanged.
        nn_array : optional prediction-neighbour array from `pred_neighbours` (reused across fields / draws).
        """
        z = self.z if z is None else np.asarray(z, float)
        rows = self._pred_rows(field, x, y, t)
        alpha = self.gp._Sinv(z)
        U = self.gp.U
        beta = U.T @ alpha if U.shape[1] else np.zeros(0)
        z_det = z - U @ beta if U.shape[1] else z
        nn = nn_array if nn_array is not None else self.pred_neighbours(x, y, t, nnum_pred, method)
        mean, var = self.op.predict_local(z_det, rows, nn_pred_array=nn)
        if U.shape[1]:
            Up = np.hstack([t_.basis(rows) for t_ in self.terms])
            mean = mean + Up @ beta
        return mean, var

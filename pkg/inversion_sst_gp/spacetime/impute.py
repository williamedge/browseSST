"""
Phase B, stage 1b: posterior SAMPLES of the T-GP derivative fields (multiple imputation) by the Matheron rule.

For data `y = T(X) + e`, a posterior sample of any linear functional `D` of the T-GP is

    D_post = D_prior + E[D | y] - E[D | y_s],        y_s = T_prior(X) + e_s,

where `(T_prior, D_prior)` is ONE joint draw from the prior and `e_s` a draw from the noise. Needed here: the joint prior draw of
`(T, dT/dx, dT/dy, dT/dt)` at the pixel locations. It is drawn from a Vecchia approximation of the joint prior over `4 N` functional
rows (the four functionals of each location are consecutive in the ordering, locations in the data operator's maximin order,
neighbours by (x, y, t) distance), plus the low-rank terms' prior. The derivative rows are scaled to O(1) (`dx` per pixel,
`dt` per hour) so the absolute jitter is harmless.

The `E[. | y_s]` terms reuse the fitted data operator through `SpaceTimeGP.predict(z=...)`, so each draw costs one prediction pass per
target field.

Two ways to get the spatial gradients (see notes/00, "resolved gradients"):
  * analytic: targets ('Tx', 'Ty', 'Tt'). The pointwise derivative of a rough field has variance at scales the grid cannot resolve, so
    the draws are over-dispersed relative to any grid-resolved gradient.
  * resolved: fields ('T', 'Tt') and `grid_gradients` (a centred finite-difference stencil of the latent T on the pixel grid), which
    is the gradient the grid can actually resolve. Fewer rows and prediction passes, too.
"""
from __future__ import annotations

import numpy as np
from gpvecchia.vecchia_utils import find_nn, forward_solve_sp

from . import inference as I
from . import kernels as K
from . import tgp as G

FIELDS = ('T', 'Tx', 'Ty', 'Tt')


def grid_gradients(Tflat, shape, dx, order=4):
    """Centred-difference d/dx, d/dy of a flat `(time, y, x)`-ordered field on the pixel grid; NaN on the x / y border.

    order 2: 1-cell border; order 4: 2-cell border, (-1, 8, -8, 1) / 12 dx. Returns flat `(Tx, Ty)`.
    """
    T = np.asarray(Tflat, float).reshape(shape)
    Tx = np.full_like(T, np.nan); Ty = np.full_like(T, np.nan)
    if order == 2:
        Tx[:, :, 1:-1] = (T[:, :, 2:] - T[:, :, :-2]) / (2 * dx)
        Ty[:, 1:-1, :] = (T[:, 2:, :] - T[:, :-2, :]) / (2 * dx)
    elif order == 4:
        Tx[:, :, 2:-2] = (-T[:, :, 4:] + 8 * T[:, :, 3:-1] - 8 * T[:, :, 1:-3] + T[:, :, :-4]) / (12 * dx)
        Ty[:, 2:-2, :] = (-T[:, 4:, :] + 8 * T[:, 3:-1, :] - 8 * T[:, 1:-3, :] + T[:, :-4, :]) / (12 * dx)
    else:
        raise ValueError("order must be 2 or 4")
    return Tx.ravel(), Ty.ravel()


class DerivativeImputer:
    def __init__(self, gp, x, y, t, dx=4000.0, dt=3600.0, nnum_prior=100, nnum_pred=200, jitter=1e-10, seed=0,
                 fields=FIELDS, targets=None):
        """`fields`: functionals in the joint prior draw (must include 'T', the data field); `targets`: those to predict /
        correct (default: all but 'T')."""
        self.gp, self.dx, self.dt = gp, float(dx), float(dt)
        self.x, self.y, self.t = (np.asarray(v, float).ravel() for v in (x, y, t))
        self.N = self.x.size
        self.nnum_pred = nnum_pred
        self.rng = np.random.default_rng(seed)
        self.fields = tuple(fields)
        assert self.fields[0] == 'T', "'T' must be the first field"
        self.targets = tuple(targets) if targets is not None else tuple(f for f in self.fields if f != 'T')
        n, nf = self.N, len(self.fields)
        wts = {'T': (1.0, 0, 0, 0), 'Tx': (0, self.dx, 0, 0), 'Ty': (0, 0, self.dx, 0), 'Tt': (0, 0, 0, self.dt)}
        self.scale = {'T': 1.0, 'Tx': self.dx, 'Ty': self.dx, 'Tt': self.dt}
        # rows in the layout [field block][location]; the ordering interleaves the functionals of each ordered location
        self.rows = np.vstack([np.column_stack([self.x, self.y, self.t, np.tile(wts[f], (n, 1))]) for f in self.fields])
        order = np.empty(nf * n, dtype=int)
        for j in range(nf):
            order[j::nf] = gp.op.ord + j * n
        Lx, Lt = gp.op.search_scales
        s = I.search_coords(self.rows, Lx, Lt)
        nn = find_nn(s[order], nnum_prior, verbose=False)
        self.prior_op = I.build_operator(self.rows, gp.op.covparams, 0.0, nnum=nnum_prior, Lx=Lx, Lt=Lt, jitter=jitter,
                                         order_idx=order, nn_array=nn, covfunc=gp.op.covfunc)
        self.nn_pred = gp.pred_neighbours(self.x, self.y, self.t, nnum_pred)
        self._mean = None

    def refresh(self):
        """Rebuild the prior operator after a hyperparameter change of the TGP."""
        self.prior_op.update_hyperparams(covparams=self.gp.op.covparams)

    def prior_sample(self):
        """One joint prior draw: dict field -> (N,) in physical units, original order."""
        op = self.prior_op
        eps = self.rng.standard_normal(len(self.fields) * self.N)
        f = forward_solve_sp(op.L, op.nn_array, eps)[op.orig_ord]
        if self.gp.terms:
            Phi = np.hstack([tm.basis(self.rows) for tm in self.gp.terms])
            f = f + Phi @ self.rng.standard_normal(Phi.shape[1])
        n = self.N
        return {name: f[j * n:(j + 1) * n] / self.scale[name] for j, name in enumerate(self.fields)}

    def posterior_mean(self, z=None):
        """E[target | z] at the N locations for each target field (z defaults to the fitted data)."""
        return {f: self.gp.predict(f, self.x, self.y, self.t, z=z, nn_array=self.nn_pred)[0] for f in self.targets}

    def posterior_samples(self, K, noise_var=None, verbose=True):
        """K posterior draws of the target fields at the N locations (Matheron rule). Returns (mean, [draws])."""
        import time
        if self._mean is None:
            self._mean = self.posterior_mean()
        noise_var = float(self.gp.op.noise if noise_var is None else noise_var)
        draws = []
        for k in range(K):
            t0 = time.time()
            ps = self.prior_sample()
            ys = ps['T'] + np.sqrt(noise_var) * self.rng.standard_normal(self.N)
            ms = self.posterior_mean(z=ys)
            draws.append({f: ps[f] + self._mean[f] - ms[f] for f in self.targets})
            if verbose:
                print(f"    draw {k + 1}/{K}: {time.time() - t0:.0f}s", flush=True)
        return self._mean, draws

# Faster `VecchiaOperator.update_hyperparams` (`FastVecchiaOperator`)

Written for: anyone fitting hyperparameters with gpvecchia's `VecchiaOperator` who wants the rebuild speed-up
(drop-in subclass now; candidate patch for gpvecchia itself).

- Code: `pkg/inversion_sst_gp/spacetime/fastop.py`
- Equivalence test: `tests/test_fastop.py` (run `python tests/test_fastop.py`)
- Benchmark that produced the table below: re-create with the snippet at the end of this note
- Written against gpvecchia 0.0.3 (`wedge/gpvecchia`, commit 593d09b)

## Problem

Fitting hyperparameters calls `op.update_hyperparams(covparams, noise)` once per likelihood evaluation, and
finite-difference gradients need ~(n_params + 1) of them per optimiser iteration. `update_hyperparams` calls
`_build_factor`, which does three things:

1. `L_matrix(...)` - numba `prange`, one small Cholesky per point. Parallel and necessary.
2. `_compute_logdet()` - a **serial Python loop** over all N points that calls `K_matrix` and `np.linalg.cholesky`
   again for every point, i.e. it recomputes the local Cholesky factors that step 1 has just produced.
3. `_build_sparse_factor()` - a **Python loop** that appends the factor to three lists entry by entry
   (N x m iterations) and then builds a `csr_matrix`.

Steps 2 and 3 are serial Python and dominated the rebuild. Profiled at N = 2200, m = 100: `L_matrix` 0.53 s,
`_compute_logdet` 1.57 s, `_build_sparse_factor` 0.26 s.

## Change

Both redundant steps follow from `L`, which is already in memory.

**Log-determinant.** `L_matrix` stores, for point i, the weights of its regression-Cholesky row, with
`L[i, 0] = 1 / sigma_i` where `sigma_i^2` is the conditional variance of point i given its neighbours (the last
entry of `Li^{-T} e_last`, reversed). The Vecchia determinant is the product of conditional variances, so

    log|C| = sum_i log sigma_i^2 = -2 * sum_i log L[i, 0].

**Sparse factor.** Row i of the factor is `L[i, :k_i]` placed at columns `nn_array[i, :k_i]`, where `k_i` is the
number of valid (>= 0) neighbours. Boolean-mask indexing does this in one vectorised step.

```python
import numpy as np
from gpvecchia.vecchia_operator import VecchiaOperator, _get_scipy_sparse_ops


class FastVecchiaOperator(VecchiaOperator):
    def _compute_logdet(self):
        if not np.array_equal(self.nn_array[:, 0], np.arange(self.nn_array.shape[0])):
            raise ValueError("needs nn_array[:, 0] == arange(n), as find_nn returns")
        return float(-2.0 * np.sum(np.log(self.L[:, 0])))

    def _build_sparse_factor(self):
        csr_matrix, _, _, _ = _get_scipy_sparse_ops()
        n = self.Xo.shape[0]
        valid = self.nn_array >= 0
        rows, cols = np.nonzero(valid)
        data = self.L[:, :self.nn_array.shape[1]][valid]
        if csr_matrix is None:
            Ld = np.zeros((n, n)); Ld[rows, self.nn_array[valid]] = data
            return Ld
        return csr_matrix((data, (rows, self.nn_array[valid])), shape=(n, n))
```

Use it exactly like `VecchiaOperator` (same constructor and methods). In browseSST,
`spacetime/inference.py::build_operator` already builds a `FastVecchiaOperator`.

## Results are identical

Same data, kernel, ordering and neighbours; base vs fast after one `update_hyperparams` (N points drawn at random in
a 56 km x 56 km x 48 h box, the space-time transport-equation covariance of `spacetime/kernels.py`). The sparse
factor is bitwise identical, the log-determinant agrees to rounding, and `solve` is identical. Timings are one
`update_hyperparams` call on a 16-core WSL2 machine (second call, so numba is compiled).

| N | m (nnum) | base (s) | fast (s) | speed-up | max abs diff in L | rel diff in logdet | max abs diff in solve |
|---|---|---|---|---|---|---|---|
| 2000 | 50 | 1.09 | 0.18 | 6.0x | 0 | 1e-15 | 0 |
| 2000 | 100 | 1.84 | 0.34 | 5.5x | 0 | 2e-15 | 0 |
| 2000 | 200 | 11.29 | 3.59 | 3.1x | 0 | 3e-15 | 0 |
| 6600 | 50 | 2.28 | 0.25 | 9.2x | 0 | 3e-15 | 0 |
| 6600 | 100 | 5.51 | 0.88 | 6.3x | 0 | 1e-15 | 0 |
| 6600 | 200 | 31.98 | 5.42 | 5.9x | 0 | 2e-15 | 0 |

In the real fits (N = 6624, m = 100) one hyperparameter evaluation went from about 4.5 s to about 1.0 s (both measured).
A 372-evaluation fit would therefore have taken roughly 28 min with the base class (extrapolated, that run was stopped
early) against the measured 380 s with the fast class. What remains is `L_matrix` itself, O(N m^3), so the speed-up shrinks as m grows.

## Assumptions and caveats

- **`nn_array[i, 0] == i`.** True for arrays from `gpvecchia.vecchia_utils.find_nn` (each point is first in its own
  neighbour set). The subclass raises if a user-supplied `nn_array` breaks this. The base class does not need it.
- Padding: invalid neighbours must be `-1` and sit at the end of the row (as `find_nn` returns). `L` entries beyond a
  row's valid length are zero and are masked out.
- Tested for the single-output / augmented-column layout used here. The multi-output class `GPtideVecchiaMO` wraps a
  plain `VecchiaOperator`, so it would need to construct a `FastVecchiaOperator` to benefit.
- The log-determinant uses `noise + jitter`, exactly as the base class does.
- No gpvecchia files were edited. To upstream, replace the two methods in `VecchiaOperator` with the bodies above
  (and keep the guard).

## A further saving not yet implemented (identity verified)

`VecchiaOperator.loglik(y)` calls `vecchia_llik`, which again recomputes every local Cholesky (serial per-point
work in numba). But with precision `Q = L^T L`, the Gaussian log-likelihood is available from the factor directly:

    loglik(y) = -0.5 * ( ||L y_ord||^2 + log|C| ),   log|C| = op.logdet()      (no -N/2 log 2 pi constant)

Checked on N = 3000, m = 60: `-3587.696449` from `op.loglik`, `-3587.696449` from the factor, difference 1.2e-7
(consistent with `loglik` not including the 1e-8 jitter). Cost of the factor-based version is one sparse
matrix-vector product. In `SpaceTimeGP` this is not needed because `datide.VecchiaMultiscaleGP` already forms its
likelihood from `op.solve` and `op.logdet()`, but plain `op.loglik` users would gain.

## Reproducing the benchmark

```python
import numpy as np, time
from gpvecchia.vecchia_operator import VecchiaOperator
from gpvecchia.vecchia_utils import find_nn, maximin_order
from inversion_sst_gp.spacetime import kernels as K, inference as I
from inversion_sst_gp.spacetime.fastop import FastVecchiaOperator

rng = np.random.default_rng(0); N, m = 6600, 100
x, y, t = rng.uniform(0, 56e3, N), rng.uniform(0, 56e3, N), rng.uniform(0, 48 * 3600, N)
pts = K.build_points(x, y, t, 5 * rng.standard_normal(N), 5 * rng.standard_normal(N))
p = K.pack_params(2000, 30e3, 350, 15e3, 2.6e5, 4.3e4, 0.3, 2e4, 8.6e4)
s = I.search_coords(pts, 20e3, 24 * 3600.0); order = maximin_order(s); nn = find_nn(s[order], m, verbose=False)
kw = dict(noise=4.0, nnum=m, nn_cols=K.COORD_COLS, order_idx=order, nn_array=nn)
for cls in (VecchiaOperator, FastVecchiaOperator):
    op = cls(pts, K.transport_obs_covfunc, p, **kw)
    op.update_hyperparams(p, 4.5)                      # warm up numba
    t0 = time.time(); op.update_hyperparams(p, 4.5); print(cls.__name__, time.time() - t0)
```

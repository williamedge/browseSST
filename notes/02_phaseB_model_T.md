# Phase B: model T itself -- what it would cost and where it could go wrong

Written for: the project owner deciding whether and how to invest in Phase B (assessment, not yet implemented).

## What Phase B replaces

Phase A (everything built so far) takes finite-difference products `dTdt`, `dTdx`, `dTdy` of the hourly snapshots and treats
the gradients as exact design coefficients: `dTdt = S - u dTdx - v dTdy + error`. Errors in those products (snapshot noise,
differencing truncation) are only handled approximately, via the observation-error terms of experiment 11. Phase B puts a
GP on T so that the derivatives, and their uncertainty, come from a smooth space-time fit instead of differencing.

## The core difficulty: the equation is bilinear

`u dT/dx` multiplies two unknown fields, so the joint posterior of (T, u, v, S) is not Gaussian. Two ways to cope:

### B1: two-stage with uncertainty propagation (recommended first step)

1. **Fit a space-time Vecchia GP to T.** One output, same N points as now. Kernel: Matern-5/2 in space x Matern-3/2 or 5/2
   in time (the temporal kernel must be at least once differentiable, so Matern-1/2 is out), plus low-rank terms for a mean /
   large-scale trend and the diurnal cycle in T. About 6-8 hyperparameters; fit cost per evaluation is the same order as the
   current one (about 1 s at N = 6.6k, m = 100), so a fit is minutes.
2. **Predict derivative functionals** `T_t`, `T_x`, `T_y` at every observation point. These are linear functionals of T, so the
   same pointwise-projection trick applies: rows carry weights `(w_T, w_x, w_y, w_t)` and the covariance function needs the
   kernel's first and second mixed partial derivatives (closed forms exist: the Helmholtz tensor in browseGP already uses the
   Matern-5/2 second derivatives). Three prediction passes at N points, m_pred ~ 100-200: about the cost of today's u, v, S
   predictions.
3. **Run the velocity inversion on the derivative estimates** with the error of those estimates accounted for. The errors are
   correlated across neighbouring points and between `T_t` and `grad T`, so a diagonal noise term is not enough. Options:
   - *Multiple imputation*: draw K posterior samples of T (Vecchia conditional sampling exists in gpvecchia), derive
     `(T_t, grad T)` for each, fit/predict on each, pool (Rubin's rules). Principled, no error-covariance modelling. Cost x K for the
     stage-2 predictions only (hyperparameters can be fitted once on the posterior mean); K ~ 10-20.
   - *Parametric error GP*: an extra correlated-error term in the data covariance, like the nuisance terms of experiment 11.
     Cheap, approximate.

Rough cost versus Phase A today: about 2-3x for the mean-field version, roughly 10-20x for stage-2 predictions with multiple
imputation (hyperparameter fits are not repeated). Stage 1 is also embarrassingly parallel across tiles.

### B2: joint PDE-constrained GP (not recommended yet)

Treat the residual `T_t + u T_x + v T_y - S` as pseudo-observations of zero at collocation points, with `u, v` taken from the
current estimate, and iterate (T given u, v; then u, v given T). Doubles the number of Vecchia points (data + PDE rows), needs
second derivatives of the T kernel, and an outer loop of ~3-6 iterations that each need refits, i.e. roughly 10-30x today's
cost, with no guarantee the iteration converges. Only worth it if B1 leaves large, identifiable gains on the table.

## Where Phase B could go wrong

1. **Stationary T prior vs real SST.** A stationary Matern GP smooths fronts and filaments, and a few strong fronts carry most of
   the useful gradient information. Smoothed gradients bias velocity. Mitigation: multi-scale T (large-scale low-rank term plus
   one or two Matern scales); check on data with a real front.
2. **Advection makes T non-separable.** A separable space-time kernel cannot represent a pattern translating with the flow.
   Hourly sampling is dense relative to advection across a 4 km pixel (~10 h at 0.1 m/s), so this is mild for derivatives, but
   a frozen-flow kernel `K(r - U tau)` with a regional mean flow would be a better T prior.
3. **Diurnal warming in T itself** (order 0.5-2 K for skin SST) must be in the T model or it contaminates `T_t`.
4. **Clouds.** Large gaps make derivative uncertainty large; multiple imputation propagates this correctly, a diagonal noise
   model does not.
5. **Smoothness / differentiability.** `T_t` needs a once-differentiable temporal kernel; velocity from `T_x` needs spatial
   smoothness; wrong smoothness choices bias derivatives.
6. **Implementation effort.** Derivative kernels need symbolic or finite-difference verification against the dense reference
   (the same strategy as `tests/test_spacetime_kernels.py`). Moderate: a few sessions.

## Is it worth doing? Test cheaply first

Whether Phase B pays depends on how much the finite-difference products hurt at *realistic noise*. At sigma_T = 0.01 K, the
synthetic gradients are only ~10% noisy and (dense experiment 01) noisy gradients cost almost nothing. The honest first test
is experiment 12: re-observe the same truth at sigma_T in {0.01, 0.03, 0.1} K (`synthetic.reobserve`), fit with the best
Phase A error model, and compare skill with clean vs noisy gradients. If skill with noisy gradients stays close to skill with
clean gradients, Phase B is a smaller win than it looks; if it collapses at sigma_T >= 0.03 K, Phase B is justified. What
sigma_T the real 4 km products have after averaging is the key unknown (the original package defaults to 1e-2 K).

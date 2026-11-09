# Space-time Vecchia velocity inversion: working plan and decisions

Fork of `inversion_sst_gp` (de Kreij et al.). Goal: replace the single-snapshot dense GP with a
space-time Vecchia GP, tiled over a large domain and run on HPC. Full original plan:
`~/.claude/plans/i-was-thinking-about-melodic-pixel.md` (not in the repo).

## Decisions so far

| Topic | Decision |
|---|---|
| Env | conda `drift` (py3.12, numba, editable `gpvecchia`, `datide`, `browseGP`, `ocean_syndata`); browseSST installed editable with `--no-deps`. No pytest in env; tests are plain functions with a `__main__` runner. |
| (u,v) prior | browseGP Helmholtz (rotational + divergent) space-time tensor, Matern-5/2 space x Matern time. Units: metres, seconds. |
| S prior | Independent of (u,v) (no cross-covariance model exists). Matern space-time in the Vecchia block, plus `datide.LowRankScale` for mean flow and diurnal terms. |
| Observation model, phase A | Keep `z = dT/dt` with `dTdx`, `dTdy` as known design coefficients, but now over (x,y,t). Fastest test of "does time help". |
| Observation model, phase B | Model T itself (T-GP gives gradients + uncertainty, errors-in-variables). Not started. |
| Test bed | Synthetic only (`ocean_syndata` Helmholtz fields + a tracer obeying the transport eq.). No OSSE for now. |
| Grid | Subsample to 4 km (Courant number for the hourly finite-difference derivatives). |
| Tiles | Not fixed. Try ~1 deg x 1 deg x 1 week (instead of 0.5 deg x month), with overlap + partition-of-unity blending. |

## Key design idea: pointwise projected kernel

Each observation is `z_i = a_i u_i + b_i v_i + c_i S_i + noise`, with `a = -dTdx`, `b = -dTdy`, `c = 1`.
Because the design is pointwise, the data covariance is evaluated on the fly:

    Cov(z_i, z_j) = a_i a_j Cuu + b_i b_j Cvv + (a_i b_j + b_i a_j) Cuv + c_i c_j K_S

So each point is one row `(x, y, t, a, b, c)` fed to a single Vecchia operator (`VecchiaOperator`)
with the neighbour search restricted to `nn_cols=[0,1,2]`.
Prediction of any linear functional of (u,v,S) uses the same operator with the prediction
point's coefficients: u = (1,0,0), v = (0,1,0), S = (0,0,1). `predict_local` returns mean and variance.

Low-rank scales (`datide.LowRankScale`) need a basis evaluated at the z rows:
`[a_i * phi(x_i) | b_i * phi(x_i) | c_i * psi(x_i)]`, then `datide.VecchiaMultiscaleGP`.

## Open items / flags

- Noise is a scalar in `VecchiaOperator`; phase B needs heteroscedastic noise.
- `scale_coords` must stay off (Helmholtz has two spatial and two temporal scales), so the
  neighbour metric must be built by hand (`order_idx`, `nn_array`) with a sensible space-time scaling.
- `ocean_syndata` fields are continuous in time, so the tracer forward model can be integrated accurately.
- browseSST now depends on `browseGP.covariance.helmholtz_numba` for the kernel pieces.

## Gotchas found

- `VecchiaOperator` adds an **absolute** jitter (default 1e-8) to the diagonal. Raw `dTdt` is ~1e-6 K/s, so
  the data must be rescaled so z, a, b, c are O(1) (divide z, a, b by a reference scale `zs`; S then lives
  in the same scaled units; noise variance scales by `1/zs**2`), or pass a much smaller `jitter`.
- `VecchiaOperator.loglik` omits the `-0.5 N log(2 pi)` constant (compare differences, or add it back).
- The browseSST repo tracks `__pycache__/*.pyc` files, so running Python dirties `git status`.
- **Numba cache poisoning after editing a covariance function (2026-11-05).** gpvecchia's `cache=True` kernels
  (`L_matrix`, `K_matrix`, ...) take the covariance function as an argument, so their on-disk cache is keyed on that
  function. After editing the covfunc's source, the old index entries can no longer be reloaded, and the next cache miss
  fails with `ReferenceError: underlying object has vanished` (raised from numba `_save_index`, right after a `KeyError`).
  Symptom: every script that builds an operator dies at the first `VecchiaOperator(...)`, even code that worked before.
  Fix: clear the stale cache (`*.nbi`/`*.nbc` in `gpvecchia/src/gpvecchia/__pycache__`, regenerable) or point a run at a
  clean directory with `NUMBA_CACHE_DIR=/some/new/dir`. Verified: with a fresh `NUMBA_CACHE_DIR`, two different covfuncs
  work in one process and across processes. Experiments from 11 on were run with `NUMBA_CACHE_DIR` set; the default
  gpvecchia cache was left untouched and is currently stale for browseSST's covfuncs (clear it before running without the variable).

## Progress log

- 2026-11-03: env set up; `spacetime/kernels.py` (projected covfunc) + `tests/test_spacetime_kernels.py`.
  Verified against dense references (browseGP `helmholtz.py`): covariance matrix to 1e-16, Vecchia
  log-likelihood at full neighbours to 1e-14, `predict_local` for u, v, S to 1e-10. The pointwise
  projection idea works with no changes to gpvecchia.
- 2026-11-03: synthetic generator (`spacetime/synthetic.py`), inference layer (`spacetime/inference.py`,
  neighbour array locked from guessed `(Lx, Lt)`), experiments 01-03 in `experiments/`.
  Test bed: 12x12 px @ 4 km, 24 h hourly, N=2200 interior points, TRUE hyperparameters, oracle noise.

## Findings (experiments 01-03; small test bed, matched prior, true hyperparameters)

1. **Time helps, dense vs dense.** noisy dTdt + clean gradients: u skill 0.90 / v 0.89 for dense
   space-time vs 0.69 / 0.78 for dense snapshot-independent (same spatial kernel, taus -> 1 s).
   Not yet tested: unknown hyperparameters, model mismatch, noisy gradients, real data.
2. **FD products have an error floor.** Noise-free, `dTdt` from hourly centred differences satisfies
   the transport equation only to a relative RMS of 0.07 / 0.11 / 0.23 at dx = 2 / 4 / 8 km
   (`synthetic.equation_residual`). The simulator itself is converged (dt_sim 1800 vs 900 s: 8e-4 K rms).
   The FD residual is smooth in time, so iid noise + a time-correlated GP over-trusts time (clean-products
   case: dense space-time was no better than snapshot-independent). Needs a noise model that knows this.
3. **Vecchia on the data covariance needs many neighbours here.** Loglik deficit vs dense at the
   guessed metric 20 km / 6 h: -68 (nnum 30), -19 (100), -8 (200). The prediction (`predict_local`) depends
   only on `nnum_pred`, not on the likelihood's `nnum`; u rms diff vs dense 0.046 (30) -> 0.025 (200) -> 0.016 (600).
4. **The neighbour metric matters; only the ratio Lx:Lt counts.** Weighting time less is better:
   at nnum = 100 the loglik deficit is -19 at 20 km / 6 h but -0.7 at 20 km / 24 h (0.8 km per hour);
   too little time weight (20 km / 96 h) is worse again (-7). With nnum = 30 nothing gets below about -26.
5. Cost is not the blocker: likelihood factor ~ N m^3 / 3, so m = 100-200 at N = 130k is cheap.
   `predict_local` at m = 200-600 per field is the expensive part. `maximin_order` is O(N^2), needs replacing at scale.

## Decisions added 2026-11-03 (later)

- Diurnal S is **not** a cosine: empirical estimates look like the positive half of a sine (daytime heating)
  with fluctuations about a flat line at night. Synthetic truth is now `max(0, sin)` (`diurnal_shape='halfsine'`,
  8e-7 K/s peak, sunrise 06:00); experiments 01-03 are pinned to the legacy cosine so they stay reproducible.
  Low-rank basis = constant + 4 daily harmonics (a rectified sine has harmonics at every multiple), column std 1/k.
- The browseGP **tidal plane model** will be added later as another `LowRankTerm` (shared u, v coefficients through
  `phi={'u':..., 'v':...}`); it does not need testing now.
- `LowRankTerm.basis(rows) = a*Phi_u + b*Phi_v + c*Phi_S` evaluated at a row's coefficient columns, so one function
  serves data rows and prediction rows (u: (1,0,0), v: (0,1,0), S: (0,0,1)).
- Variance returned by `SpaceTimeGP.predict` is the Vecchia block's conditional variance only; it omits the
  uncertainty of the low-rank coefficients (datide has no `predictive_covariance` yet). S variance is slightly too small.

## Performance finding

`VecchiaOperator._build_factor` (every hyperparameter evaluation) spends most of its time in serial Python:
`_compute_logdet` recomputes every local Cholesky (1.6 of ~2.4 s at N=2200, m=100) and `_build_sparse_factor` loops
entry by entry. `spacetime/fastop.py::FastVecchiaOperator` takes the log-determinant from `L[:,0]` (= 1/sigma_i) and
vectorises the sparse assembly: identical results (tests/test_fastop.py), ~7x faster. Candidate to upstream to gpvecchia.
Before: one rebuild = 4.5 s at N=6624, m=100. Full write-up, benchmark table and upstream patch: `notes/01_fast_vecchia_operator.md`.

## Results: diurnal term + fitting (experiments 04, 05) -- phase A, one synthetic realisation

Setup: 14x14 px @ 4 km, 48 h, N=6624, half-sine diurnal truth, noisy dTdt + clean gradients, Vecchia nnum=100
(metric 20 km : 24 h), nnum_pred=200, oracle noise variance, fit from a perturbed start (most params off by 0.5-2x,
weak log-normal priors sd=1 centred on the start). Skill = 1 - rmse^2/var(truth).

| model | u skill | v skill | S skill | loglik |
|---|---|---|---|---|
| ST, oracle hyperparams, Matern S only | 0.836 | 0.887 | -0.04 | -14636 |
| ST, oracle hyperparams, + diurnal low-rank | 0.845 | 0.899 | +0.39 | -14592 |
| ST, FITTED (exp 04) | 0.830 | 0.899 | -0.53 | -14527 |
| snapshot-independent, oracle, + diurnal | 0.768 | 0.824 | +0.49 | -15485 |
| snapshot-independent, FITTED (exp 05) | 0.753 | 0.820 | +0.48 | -15440 |

- Time helps with fitted hyperparameters too: u/v rmse 17-25% lower, loglik ~900 higher.
- ST fit: 372 evals, 380 s (1.0 s/eval after `FastVecchiaOperator`; 4.5 s before). u, v are robust to hyperparameter
  error, but individual ST hyperparameters drift (sigma_psi 1.9x, l_psi 1.4x, tau0_psi 0.54x, sigma_S 2.6x, l_S 0.54x,
  lr:diurnal 0.57x). The snapshot-independent fit recovers its parameters much better (sigma_psi 1.18x, l_psi 1.02x).
- S is weakly identified (S std 0.45 vs noise std 2.2 in scaled units) and the ST fit moved variance from the diurnal
  term to the Matern S, so S skill got worse than at the oracle values. u, v matter most.
- Coverage of u, v at oracle values is 0.87-0.90 (ST) vs 0.95-0.96 (SI): the ST posterior is somewhat overconfident
  (non-iid finite-difference residual, Vecchia approximation).
- Hypotheses NOT yet tested for the ST parameter drift: (a) tau0_psi = 3 d is longer than the 48 h window, so it is
  weakly identified; (b) a ridge between sigma_psi / l_psi / tau0_psi; (c) the locked neighbour metric is no longer
  ideal once hyperparameters move; (d) local minimum.
- **Not a local minimum (2026-11-05):** `python experiments/04_diurnal_and_fit.py 100 40 truth` (fit started AT the true
  values) moved to the same optimum as the perturbed start: sigma_psi 4331 vs 4051, l_psi 4.43e4 vs 4.31e4, sigma_S 0.773
  vs 0.773, lr:diurnal 0.26 vs 0.28, loglik -14526.4 vs -14526.7 (648 evals). The optimum of this Vecchia (m=100)
  likelihood is well defined and sits away from the truth (loglik +65 above the truth's). Remaining explanations: estimation
  uncertainty of one realisation (ridge), approximation bias (Vecchia m, non-iid FD residual, oracle-noise), see exp 06.

## Is the drift a sigma-ell ridge? (experiment 06, 2026-11-05)

Added browseGP's amplitude parameterisation to `fit.py` (`gamma`, `f_delta` instead of `sigma_psi`, `sigma_phi`;
`to_amplitude_params` / `to_potential_params`) and a finite-difference Laplace covariance (`laplace_covariance`,
`amplitude_uncertainty`). Truth: gamma 0.0949, f_delta 0.100.

- **Same optimum in either parameterisation** (loglik -14526.7 potential vs -14527.1 amplitude; gamma 1.33x vs 1.29x
  truth). Expected: the MLE is parameterisation-invariant. The amplitude form did NOT speed the optimiser up
  (636 evals vs 372). Its value is for priors (gamma is a velocity rms the user can elicit), not for the location of the fit.
- **The ridge is real**: at the optimum corr(log sigma_psi, log l_psi) = +0.96, regression slope ~1.9, so sigma_psi ~ l_psi^2,
  i.e. sigma^2/l^4 (vorticity / strain variance) is pinned but sigma^2/l^2 (velocity variance gamma^2) is not, because
  the 56 km domain is smaller than 2 l_psi. Laplace sd (log units): sigma_psi 0.26, l_psi 0.13, sigma_phi 0.42, l_phi 0.25,
  tau0_psi 0.22, tau0_phi 0.48, sigma_S 0.18, l_S 0.21, tau0_S 0.52, noise 0.02, diurnal 0.29; gamma 0.13, f_delta 0.044.
  Weakest direction: sigma_phi / l_phi / tau0_phi (the divergent trio), as browseGP's notes predict.
- **But the drift is larger than the data's own uncertainty**: fit vs truth is ~5 sd for sigma_S (2.6x), ~2.6 sd for l_psi,
  ~2.8 sd for tau0_psi, ~2 sd for gamma. So estimation noise on the ridge is not the whole story. Hypothesis (exp 07 tests it):
  the smooth-in-time finite-difference residual (~11% of dTdt rms at 4 km) is absorbed by the Matern S.
- The Hessian at the truth is not positive definite (min eigenvalue -8.4): the truth is not a stationary point of this
  likelihood; only the Hessian at the optimum is meaningful.
- Possible fix independent of cause: informative priors on gamma / f_delta (and l, tau), as the original package does
  for sigma_u (0.09 +- 0.02 m/s). For this data a vorticity-amplitude coordinate (sigma/l^2) would align better with the ridge than gamma.

## Exact-equation test (experiment 07, 2026-11-05): the FD residual is what the fit absorbs

Data = S_true - u_true dTdx - v_true dTdy + iid noise (std sigma_T/(sqrt2 dt)), so the model is exactly specified
(`make_observations(..., dTdt_source='equation')`). Same start, priors, neighbours as exp 04. 516 evals; stopped at the
iteration limit (40), not fully converged.

- Hyperparameters fit/truth: sigma_psi 1.41, l_psi 1.26 (the ridge), sigma_phi 1.03, l_phi 1.13, sigma_S 0.77, l_S 0.88,
  noise 0.99, diurnal 0.47 -- but all three temporal scales are ~0.4x: tau0_psi 0.45, tau0_phi 0.39, tau0_S 0.41.
- Skill: u 0.909, v 0.920, S 0.639 (oracle values: 0.906 / 0.913 / 0.668). Coverage u 0.97, v 0.95, S 0.88.
  With FD products (exp 04) it was u 0.830 / v 0.898 / S -0.54 and coverage 0.88 / 0.92 / 0.87.
- So the finite-difference residual (only ~5% of the noise variance, but smooth in time and tied to the gradients) cost
  ~0.08 u skill, made S unusable, and made the posterior overconfident. It is absorbed mainly by the Matern S.
- Unexplained: the ~0.4x temporal scales (not explained by the 48 h window alone, since tau0_phi = 12 h and tau0_S = 24 h
  are shorter than it; the prior is weak, sd 1 in log). The Laplace sd for tau0_phi / tau0_S is ~0.5 (factor 1.6), so
  0.39x is ~2 sd. Worth re-checking with a longer window (1 week) and more iterations before reading anything into it.
- Options for the FD residual: (a) an explicit nuisance residual GP in the Vecchia block so S does not have to absorb it;
  (b) higher-order stencils (less truncation error, more noise amplification); (c) the T-GP route (phase B), which removes
  finite differencing altogether.

## Observation-error models (experiments 08, 09, 11, 12; 2026-11-05)

Structure of the FD residual (exp 08/09, exp-04 dataset): simulator is accurate (uniform-flow exact solution, exp 09);
residual is real differencing error that falls only slowly with dx (T has rough small-scale structure); nearly white in
space (lag-1 autocorrelation 0.06), persistent in time (0.92 at 1 h, 0.88 at 2 h, 0.64 at 4 h); size roughly independent
of |grad T| except in the top quartile; equivalent velocity error rms(res)/|grad T| ~ 0.03-0.08 m/s. Also: dTdt noise from
independent snapshots has covariance -noise/2 at a 2 h lag (shared snapshot), not iid.

Exp 11 (fit from the usual perturbed start, FD2 products, noisy dTdt, clean gradients):

| | V0 iid | V1 + lag-2 diff. noise | V2 + lag-2 + FD residual (white in space x Matern-3/2 in time) |
|---|---|---|---|
| loglik | -14526.7 | **-13555.7** | -13544.5 |
| u / v / S skill | 0.830 / 0.899 / -0.53 | 0.797 / 0.899 / -1.54 | 0.793 / 0.893 / -1.49 |
| coverage u / v / S | 0.88 / 0.92 / 0.88 | 0.85 / 0.91 / 0.88 | 0.85 / 0.91 / 0.86 |

- The lag-2 differencing-noise covariance is overwhelmingly supported (+971 loglik, no extra parameters) -> keep it as the
  default noise model, it is the correct covariance of hourly differences of independent snapshot noise.
- It does NOT improve skill. The FD-residual term adds only +11 loglik (fitted sigma_R 0.47, tau0_R 2 h) and no skill. The
  Matern S still absorbs the residual: l_S shrinks to 5-7 km (~1.5 px), sigma_S ~3x truth. The residual is not white in space at
  nonzero lags (grid-scale features are advected across pixels), which the same-pixel term cannot represent.
- At the TRUE hyperparameters FD2 products already cost skill (exp 04 vs 07 oracle: u rmse 0.0359 vs 0.0280, S skill 0.39 vs
  0.67), so the loss is in the data, not in hyperparameters or the noise model; additive nuisance terms cannot recover it.

Exp 12 (true hyperparameters, sigma_T = 0.01 K, clean gradients, all variants scored on the same 4400 interior points):

| products | u rmse (skill) | v rmse (skill) | S skill | coverage u / v / S |
|---|---|---|---|---|
| (a) 2nd-order (current) | 0.0355 (0.867) | 0.0351 (0.895) | 0.40 | 0.86 / 0.89 / 0.81 |
| (b) 4th-order stencils in x, y, t | **0.0261 (0.928)** | **0.0310 (0.918)** | 0.55 | 0.97 / 0.96 / 0.89 |
| (c) exact-equation ceiling, 2nd-order gradients | 0.0266 (0.925) | 0.0306 (0.920) | 0.70 | 0.97 / 0.94 / 0.95 |
| (c') exact-equation ceiling, 4th-order gradients | 0.0254 (0.932) | 0.0285 (0.931) | 0.69 | 0.97 / 0.94 / 0.92 |

4th-order products recover essentially all of the skill lost to FD2 error at this noise level, despite 1.8x the dTdt noise
variance and fewer points (the 4th-order border drops 2 cells). Rescoring on a common interior mattered little (FD2 u rmse
0.0359 -> 0.0355).

**Noise sweep (exp 12, same setup, sigma_T = 0.01 / 0.03 / 0.1 K, gradients from true T = clean, or from noisy T = obs;
true hyperparameters; oracle iid noise variance; one noise realisation; scored on the same 4400 interior points):**

| sigma_T | gradients | 2nd-order u / v rmse (skill) | 4th-order u / v rmse (skill) | coverage u / v, 2nd-order |
|---|---|---|---|---|
| 0.01 | clean | 0.0355 (0.867) / 0.0351 (0.895) | **0.0261 (0.928) / 0.0310 (0.918)** | 0.86 / 0.89 |
| 0.01 | obs | 0.0349 (0.871) / 0.0357 (0.891) | **0.0287 (0.913) / 0.0348 (0.897)** | 0.87 / 0.88 |
| 0.03 | clean | 0.0376 (0.850) / 0.0412 (0.855) | 0.0361 (0.862) / 0.0435 (0.839) | 0.97 / 0.95 |
| 0.03 | obs | **0.0448 (0.787) / 0.0537 (0.754)** | 0.0486 (0.750) / 0.0613 (0.679) | 0.93 / 0.86 |
| 0.1 | clean | 0.0549 (0.681) / 0.0678 (0.608) | 0.0624 (0.587) / 0.0756 (0.512) | 0.99 / 0.95 |
| 0.1 | obs | **0.0730 (0.435) / 0.0990 (0.164)** | 0.0797 (0.327) / 0.1094 (-0.022) | 0.88 / 0.72 |

- The 4th-order advantage exists only at sigma_T ~ 0.01 K. At 0.03 K it is neutral (clean gradients) or worse (noisy
  gradients), and at 0.1 K it is worse. Do NOT adopt 4th-order stencils without knowing the real noise of the 4 km products.
- Gradient noise is the dominant loss from sigma_T ~ 0.03 K up (2nd-order, clean -> obs gradients): v skill 0.855 -> 0.754 at
  0.03 K and 0.608 -> 0.164 at 0.1 K; u 0.850 -> 0.787 and 0.681 -> 0.435. Coverage falls (v 0.72 at 0.1 K): an iid-noise model
  does not represent noise that multiplies the unknown velocity. This is the errors-in-variables term that Phase B (a latent T
  model) targets. At 0.01 K gradient noise costs almost nothing, so Phase B is mainly justified if real sigma_T >= ~0.03 K.
- The 'exact-equation' rows are not a true ceiling at high noise: their noise is white, whereas differenced snapshot noise is
  high-pass (and the model with iid noise does not exploit that). A model with the lag-2 noise covariance (exp 11 V1) should do
  better than the 'obs' rows above at high noise; not yet tested at oracle hyperparameters.
- Caveats: single realisation, true hyperparameters, matched-prior truth, 14x14 px x 48 h.

## Parked idea (user, 2026-11-05): pool hyperparameters across tiles / reduced-rank large scale

Estimate hyperparameters on a big set of tiles, then smooth the map of parameters (and/or add a reduced-rank large-scale
component). Why it fits what we measured: the sigma_psi-l_psi ridge (corr +0.96) and the short-looking temporal scales are limits
on information per 56 km x 48 h window, not optimiser or data-quality problems (they persisted with exactly-specified data, exp 07).
Pooling attacks that directly:
- precision-weighted smoothing of log-hyperparameters across the tile map, using each tile's Laplace covariance
  (`fit.laplace_covariance`) as the weight: well-determined tiles pull poorly-determined neighbours;
- a smooth parameter field is also what coherent blending of overlapping tiles needs (no seams from parameter jumps);
- a reduced-rank large-scale velocity term (`LowRankTerm` with phi on 'u', 'v': polynomial / low-order modes / the browseGP tidal
  plane model) can carry the energy at scales larger than a tile, which is the weakly determined direction (sigma^2/l^2).
Do after Phase B; none of it is implemented.

## Phase B status (2026-11-05)

Decision: go for the full latent-T model (see `notes/02_phaseB_model_T.md`). Built: `spacetime/tgp.py` (derivative-capable T kernel
with closed-form partial derivatives, `TGP` model, `TLowRankTerm` bases with derivatives), verified against finite differences for all
16 derivative pairs and against a dense GP (`tests/test_tgp.py`). Experiment 13 = stage 1 + plug-in at sigma_T = 0.03 and 0.1 K.
Next (updated): see "Experiments 15-17" below. Original plan: multiple imputation (Matheron-rule posterior samples) if the plug-in shows a real gain; a front /
filament test field; spatially varying noise (atmospheric-effect periods).

### Experiment 13 results (14x14 px x 48 h, one realisation; true velocity hyperparameters; posterior MEANS only)

Derivative error vs 4th-order FD of the true T, as a fraction of the std of the reference, common 4400-point interior:

| sigma_T | method | Tx | Ty | Tt |
|---|---|---|---|---|
| 0.03 K | FD2 | 0.29 | 0.29 | 1.60 |
| | FD4 | 0.32 | 0.31 | 2.15 |
| | T-GP | 0.17 | 0.17 | 0.28 |
| 0.1 K | FD2 | 0.80 | 0.79 | 5.35 |
| | FD4 | 1.05 | 1.04 | 7.16 |
| | T-GP | 0.33 | 0.32 | 0.53 |

Velocity skill (u / v / S) and 95% coverage (u / v / S), scored on the same interior points:

| sigma_T | gradients | u skill | v skill | S skill | cover u / v / S |
|---|---|---|---|---|---|
| 0.03 K | FD2 noisy | 0.787 | 0.754 | 0.171 | 0.93 / 0.86 / 0.79 |
| | FD4 noisy | 0.750 | 0.679 | -0.010 | 0.93 / 0.84 / 0.73 |
| | FD2 clean (bound) | 0.850 | 0.855 | 0.423 | 0.97 / 0.95 / 0.89 |
| | **T-GP plug-in** | **0.872** | **0.867** | 0.197 | 0.79 / 0.74 / 0.69 |
| 0.1 K | FD2 noisy | 0.435 | 0.164 | -0.114 | 0.88 / 0.72 / 0.70 |
| | FD4 noisy | 0.327 | -0.022 | -0.116 | 0.86 / 0.69 / 0.74 |
| | FD2 clean (bound) | 0.681 | 0.608 | 0.272 | 0.99 / 0.95 / 0.84 |
| | **T-GP plug-in** | **0.704** | **0.760** | -0.482 | 0.61 / 0.61 / 0.53 |

Reading:
- u, v: the T-GP beats noisy finite differences clearly (v 0.16 -> 0.76 at 0.1 K) and even beats the FD2 *clean-gradient* "bound",
  because it also removes the FD truncation error and denoises T_t. That bound is therefore not an upper bound for Phase B.
- Coverage is badly over-confident (0.61 at 0.1 K, nominal 0.95). The stage-2 noise here is the empirical residual variance
  (1.4 / 2.8 vs 35 / 384 for FD2), treated as iid, but the T-GP derivative error is smooth and correlated in space and time, and
  the derivative uncertainty is not propagated. This is what multiple imputation is for.
- S gets no better and is worse than predicting the mean at 0.1 K (skill -0.48, -0.75 with all points): with a tiny assumed
  noise the free S field soaks up the T-GP's T_t error. Needs the propagated uncertainty too, then re-check.
- Noise is recovered well by the T-GP fit (0.0292 K for 0.03; 0.0988 K for 0.1).
- Fit: 340 evals, ~600 s each (1.8 s/eval at N = 9408), `success=False` (hit maxiter 30, not converged). The short component's
  length scale sits at the 3 km lower bound (below the 4 km pixel), so it acts like a nugget-like fine-scale term. Hyperparameters
  of the T-GP are therefore not settled.

Vecchia accuracy of the derivative predictions (experiment 14: 1944-point subset vs dense, at the FITTED hyperparameters), rms
difference / std of the dense prediction:

| sigma_T | nnum / nnum_pred | Tx | Ty | Tt |
|---|---|---|---|---|
| 0.03 K | 100 / 100 (used in exp 13) | 0.097 | 0.081 | 0.068 |
| | 100 / 200 | 0.027 | 0.046 | 0.029 |
| | 200 / 200 | 0.027 | 0.047 | 0.030 |
| | 400 / 400 | 0.014 | 0.014 | 0.020 |
| 0.1 K | 100 / 100 | 0.124 | 0.101 | 0.207 |
| | 100 / 200 | 0.046 | 0.061 | 0.125 |
| | 200 / 200 | 0.045 | 0.061 | 0.124 |
| | 400 / 400 | 0.021 | 0.019 | 0.038 |

- At the generic START hyperparameters (sigma 0.5 K, l 40 km, tau 2 d) the same check at 100/100 gave 15% / 15% / 28%, so the
  smooth, low-noise regime is the hard one for Vecchia; the fit moves away from it.
- The gain comes from `nnum_pred` (100 -> 200), not `nnum` (the likelihood neighbours; rows 2 and 3 are nearly identical). So
  experiment 13's derivatives carried Vecchia error comparable to a good part of the 0.17 / 0.33 errors above, and its T-GP numbers
  are likely conservative. Use `nnum_pred >= 200` for derivative prediction.
- Caveat: on 1944 points, m = 400 is 20% of the data, so the large-m rows are optimistic for 9408+ points. Re-check on the full cube.

Cost note (extrapolated, not measured): the stage-1 fit scales roughly linearly in N at fixed m, so a 28 x 28 px x 168 h tile
(131,712 points, 14x more) would be ~25 s per evaluation, i.e. a few hours for 340 evaluations. Fit hyperparameters on a few
windows and reuse them (the parked pooling idea), predict-only per tile.

Big-domain file `data/synth_84px_168h.nc` is generated (3984 s; pad 92, vmax 0.45 m/s). Its FD equation residual is 35% of dTdt rms,
which I have not compared on the same footing with the 14 px file; the Courant number per hour at vmax is about 0.4 pixel.

### Experiments 15-17 (2026-11-05): split bounds, multiple imputation, sampler accuracy

**Exp 15: two components split by bounds** (eddy: l >= 10 km, tau >= 12 h; fast: l in [3, 10] km, tau <= 12 h; warm start from exp 13; suggested
by the browseGP practice of an "eddy" and a "fast red noise" model). Both fits CONVERGED this time (L-BFGS-B tolerance; 260 and 620 evaluations).

| sigma_T | eddy (sigma, l, tau) | fast (sigma, l, tau) | noise sd (true) | loglik (exp 13) |
|---|---|---|---|---|
| 0.03 K | 0.206 K, 13.8 km, 29.3 h | 0.069 K, 3.1 km, 9.5 h | 0.0294 (0.03) | 17496.2 (17487.7) |
| 0.1 K | 0.214 K, 13.7 km, 29.8 h | 0.066 K, 3.5 km, 8.0 h | 0.0988 (0.1) | 7539.5 (7539.3) |

- The exp 13 fits were already at their optimum (`success=False` there was only the iteration cap); the split bounds change nothing
  at 0.1 K. At 0.03 K the fast component stays at the 3 km floor: the data want pixel-scale (or finer) structure.
- 200 prediction neighbours: derivative errors at 0.1 K unchanged (Tx 0.34, Ty 0.32, Tt 0.50 of std); at 0.03 K slightly better
  (0.15 / 0.15 / 0.27 vs 0.17 / 0.17 / 0.28). Plug-in velocity skill at 0.03 K (all points): u 0.901, v 0.895, S 0.375, i.e. the
  FD2 clean-gradient S skill (0.423) is reached on the FD2 point set (S 0.434), cover 0.86 / 0.80 / 0.73. At 0.1 K unchanged:
  u 0.702, v 0.758, S -0.94, cover 0.61 / 0.62 / 0.46.

**Exp 17: Vecchia sampler accuracy** (1944-point subset, dense reference, 0.03 K fit): draw variance / dense posterior variance is 0.98
(Tx 0.93, Ty 1.00, Tt 1.02) with 100 prior neighbours and 1.00 (0.95, 1.02, 1.02) with 200; sample mean within sampling noise;
Vecchia posterior mean within 0.033 sd of dense. 100 prior neighbours suffice. `tests/test_impute.py`: exact regime, prior operator vs
dense 4e-15, draw variance ratio 0.999, both imputation modes.

**Exp 16: multiple imputation of the analytic derivatives FAILS** (0.1 K, K = 10 Matheron draws, true velocity hyperparameters):

| estimator | u skill / cover | v skill / cover | S skill / cover |
|---|---|---|---|
| plug-in (posterior mean), oracle noise | 0.702 / 0.61 | 0.758 / 0.62 | -0.94 / 0.46 |
| plug-in, noise fitted by marginal likelihood (0.56 vs oracle 3.38) | 0.587 / 0.41 | 0.650 / 0.42 | -2.46 / 0.28 |
| one draw, oracle noise | 0.367 / 0.48 | 0.447 / 0.52 | -1.75 / 0.50 |
| Rubin-pooled over 10 draws, oracle noise | 0.445 / 0.62 | 0.429 / 0.59 | -1.84 / 0.76 |

Two separate problems:
1. **The draws are over-dispersed for the spatial gradients**: draw spread / error of the mean is 2.59 (Tx), 2.84 (Ty), 1.15 (Tt)
   (Tx: spread 2.0e-5 vs error 7.7e-6). The sampler itself is exact (exp 17), so this is the MODEL: the fitted prior has gradient sd
   3.1e-5 (fast component alone 2.4e-5) against 2.2e-5 for the 4th-order gradient of the true T. Cause: the true T is a ROUGH
   field (1-D spectral slope -2 to -2.8 from 56 km to the smallest scales; a Matern-5/2 falls as -6; the 84 px file also shows a
   power upturn at 2 dx, grid-scale noise). A two-scale Matern-5/2 mixture mimics the shallow spectrum with a 3 km component whose
   pointwise analytic gradient has variance at scales the 4 km grid cannot resolve. The posterior MEAN is fine; draws are not.
   Real SST has the same kind of spectrum, so expect this on real data.
2. **Even calibrated draws would not be a congenial imputation.** The T-GP does not know the PDE, so a draw's (Tt, grad T) does not
   satisfy the transport equation with the true (u, v, S); stage 2 treats each draw's gradients as exact, and the bilinear u * grad T
   dilutes the estimate. Pooled draws therefore score worse than the plug-in. A noise level fitted on the mean products is also
   too small to use with draws (draws' oracle residual 17.6 vs 3.4 for the mean).

**Exp 18: resolved-gradient imputation WORKS** (fields T and Tt imputed; spatial gradients = 4th-order centred differences of the latent T on the pixel grid;
Tt analytic; K = 10 draws; stage 2 on the 4800 interior points; true velocity hyperparameters; one realisation each).

Draw calibration (spread about the mean / error of the mean, vs the 4th-order gradient of the true T): 0.03 K Tx 0.95, Ty 1.00, Tt 1.16;
0.1 K Tx 0.92, Ty 0.97, Tt 1.14 (analytic gradients, exp 16: 2.59 / 2.84 / 1.15). Mean errors are as good or better (Tx 0.13 of std at 0.03 K,
0.32 at 0.1 K).

Stage-2 noise rule (not an oracle): the equation-error variance induced by the derivative uncertainty,
`var(dTt) + E[u^2] var(dTx) + E[v^2] var(dTy) + 2 E[uv] cov(dTx, dTy)` from the draw spread about the mean and the PRIOR velocity moments
(scaled by 1 / ZS^2). It came out 1.68 (0.03 K) and 5.53 (0.1 K), vs 1.21 and 2.74 for the oracle iid residual of the mean products
(and 2.92 / 8.40 for the draws). The larger value is better than the oracle iid residual because the residual is correlated.

| sigma_T | estimator (noise rule) | u skill / cover | v skill / cover | S skill / cover |
|---|---|---|---|---|
| 0.03 K | plug-in (equation error) | 0.898 / 0.86 | 0.898 / 0.81 | 0.431 / 0.73 |
| | Rubin-pooled draws | 0.882 / 0.94 | 0.875 / 0.91 | 0.197 / 0.86 |
| | **plug-in + propagated variance** | **0.898 / 0.96** | **0.898 / 0.94** | **0.431 / 0.92** |
| | plug-in + propagated, oracle noise | 0.890 / 0.92 | 0.892 / 0.90 | 0.406 / 0.87 |
| 0.1 K | plug-in (equation error) | 0.743 / 0.73 | 0.806 / 0.74 | -0.447 / 0.54 |
| | Rubin-pooled draws | 0.663 / 0.86 | 0.699 / 0.85 | -1.516 / 0.83 |
| | **plug-in + propagated variance** | **0.743 / 0.92** | **0.806 / 0.92** | **-0.447 / 0.94** |
| | plug-in + propagated, oracle noise | 0.709 / 0.86 | 0.778 / 0.86 | -0.788 / 0.87 |

(Earlier plug-in with analytic gradients, oracle noise: 0.03 K 0.901 / 0.895 / 0.375, cover 0.86 / 0.80 / 0.73; 0.1 K 0.702 / 0.758 / -0.94, cover 0.61 / 0.62 / 0.46.)

Recipe that works so far: (1) fit the two-component T-GP to the noisy T; (2) posterior-mean T and Tt (200 prediction neighbours); (3) gradients =
4th-order differences of the latent T; (4) stage 2 on those products with the equation-error noise; (5) K = 10 Matheron draws
(~10 min at 9.4k points) give the between-draw variance B, and the reported variance is `W_plug + (1 + 1/K) B` around the PLUG-IN mean.
Pooling the draw estimates (Rubin) is NOT recommended: it loses accuracy (0.1 K u 0.743 -> 0.663) because the T-GP imputation does not
know the PDE (not congenial) and the bilinear u * grad T dilutes.
Caveats: S at 0.1 K has negative skill though its coverage is fine, so treat S as a nuisance there; stage 2 uses the true velocity
hyperparameters; one realisation; the synthetic T is close to the model class (no fronts, no outliers); the interior only (the outer
2-pixel ring of the tile has no resolved gradient).

**Exp 19: spatial smoothness of the two T-GP components** (resolved-gradient mode; each component's spatial Matern nu in {5/2, 3/2, 1/2}, temporal nu_t = 5/2;
split bounds as exp 15, warm start, all fits converged; `T_covfunc_res` in `tgp.py`, tested against the 5/2 kernel and finite differences).

| combination (eddy / fast) | loglik 0.03 K (vs 5/2:5/2) | loglik 0.1 K (vs 5/2:5/2) |
|---|---|---|
| 5/2 / 5/2 | 17496.2 | 7539.5 |
| 5/2 / 1/2 | 17476.7 (-19.5) | 7532.4 (-7.0) |
| 3/2 / 3/2 | 17476.7 (-19.5) | 7528.6 (-10.9) |
| 3/2 / 1/2 | 17463.2 (-33.0) | 7524.4 (-15.1) |
| 1/2 / 1/2 | 17399.0 (-97.2) | 7459.4 (-80.1; hit the evaluation cap, `success=False`) |

- Marginal likelihood prefers the SMOOTHER kernels at both noise levels, even though the 1-D spectral slope of the true T (about -2 to -2.8) corresponds to nu
  of roughly 1/2 to 1. I do not know why; the slope was measured on single noise-free snapshots between 9 and 56 km, whereas the likelihood is
  of noisy data under a separable space-time kernel (the real T structure is partly advective, not separable), and the fits are constrained by
  the split bounds. So treat the slope-to-nu argument as unproven here, and the likelihood as the thing to use.
- The velocity and gradient results do NOT depend on the choice: 0.03 K u / v skill 0.890-0.891 / 0.891-0.892 and Tx error 0.13 of std for every
  combination (FD4 of the latent T); 0.1 K u / v 0.699-0.709 / 0.764-0.778, Tx 0.32 (the exception is 1/2 : 1/2 at 0.1 K, the worst likelihood: u / v 0.689 / 0.761, Tx 0.33). What fixed the calibration was the resolved-gradient design,
  not the kernel. So: keep nu_s = 5/2 : 5/2 as the default, make nu_s configurable, and let the marginal likelihood choose on real data.
- The fast length scale is exactly at its 3 km floor in the four fits with a 1/2 fast component and a smoother eddy (5/2 / 1/2 and 3/2 / 1/2 at both
  noise levels), and ends between 3.1 and 4.4 km in the others; the eddy is at 10.1-10.9 km (floor 10 km) for 5/2 / 1/2: the split bounds bind. A "fast" scale below the 4 km pixel is not a smooth-field component, it is pixel-scale red noise. Candidate (not tried): a
  same-pixel, temporally correlated term (as the Phase A sigma_R / tau_R) and predict T and Tt from the advected (eddy) part only, e.g. with
  `VecchiaOperator.predict_local_decomposed`; it also matches the physics of real retrieval/atmospheric error that is correlated in time but not advected.
- FD stencil of the latent T: 4th order beats 2nd at 0.03 K (u / v 0.890 / 0.892 vs 0.842 / 0.872; Tx error 0.13 vs 0.21 of std); at 0.1 K 2nd order is
  slightly better for the velocity (0.724 / 0.798 vs 0.709 / 0.778) although its gradient error against the 4th-order truth is a little worse.

**Exp 20: low noise (sigma_T = 0.01 K, the original package's default)**: does the latent-T route still win? Yes, by less. T-GP fit converged
(5/2 : 5/2; noise sd 0.0098 K for a true 0.01; fast scale again at its 3 km floor). Derivative error (fraction of std, vs 4th-order gradient of the true T):
FD2 0.18 / 0.20 / 0.54 (Tx / Ty / Tt), FD4 0.11 / 0.10 / 0.72, latent T with FD4 gradients 0.05 / 0.05 / 0.15 (FD2 gradients of the latent T 0.17 / 0.19 / 0.15).
Velocity stage (true hyperparameters, oracle iid noise, same 4400 points):

| gradients | u skill / cover | v skill / cover | S skill / cover |
|---|---|---|---|
| FD2, noisy | 0.871 / 0.87 | 0.891 / 0.88 | 0.339 / 0.78 |
| FD4, noisy | 0.913 / 0.96 | 0.897 / 0.92 | 0.441 / 0.84 |
| FD2, clean-gradient bound | 0.867 / 0.86 | 0.895 / 0.89 | 0.398 / 0.81 |
| FD4, clean-gradient bound | 0.928 / 0.97 | 0.918 / 0.96 | 0.550 / 0.89 |
| **latent T, FD4 gradients (plug-in)** | **0.922 / 0.86** | **0.925 / 0.82** | **0.586 / 0.75** |
| latent T, FD2 gradients (plug-in) | 0.849 / 0.79 | 0.887 / 0.80 | 0.117 / 0.66 |

- At 0.01 K gradient noise no longer limits finite differences (noisy FD2 = clean FD2 bound); the difference from the latent-T route is the stencil truncation
  error and the noisy T_t. The latent-T plug-in is at the FD4 clean-gradient ceiling (u 0.922 vs 0.928; v 0.925 vs 0.918; S 0.586 vs 0.550) and above noisy FD4
  (u +0.01, v +0.03, S +0.15). The gain over the best finite-difference stencil is small for u at this noise level; the clear gains are at sigma_T >= 0.03 K.
- The advantage is mostly the FD4 resolved gradient of the latent T: with FD2 gradients of the latent T the plug-in is NOT better than FD4 on the noisy T.
  Plug-in coverage is again low (0.75-0.86); the propagated variance (exp 18) is needed.
- Caveat: no front, no outliers, one realisation, and oracle stage-2 noise (the propagated-variance noise rule was not run at 0.01 K).


**Exp 21: fronts and a filament** (`SyntheticConfig(front_amp=0.5, front_width=4000, filament_amp=0.3, filament_width=3500)`; same velocity / S truth as every
earlier run; |grad T| rms 6.6e-5 vs 3.3e-5 K/m, max 2.5e-4; T std 0.49 K vs 0.23 K; `data/synth_14px_48h_fronts.nc`). Scope set by the user: fronts + filaments only
(no cloud gaps, one noise level, no spatially varying diurnal warming). True velocity hyperparameters, oracle iid stage-2 noise, 4400 scored points;
"steep" = the 20% of scored points with the largest |grad T| (880). Skill, all points / steep points (u, v) and S skill:

| sigma_T | gradients | u skill all / steep | v skill all / steep | S |
|---|---|---|---|---|
| 0.01 K | FD2, noisy T | 0.867 / 0.855 | 0.877 / 0.771 | -0.40 |
| | FD4, noisy T | 0.944 / 0.937 | 0.886 / 0.727 | 0.48 |
| | FD4, clean-gradient bound | 0.945 / 0.940 | 0.911 / 0.823 | 0.50 |
| | **latent T, FD4** | 0.941 / 0.936 | **0.927 / 0.872** | 0.51 |
| 0.03 K | FD2, noisy T | 0.877 / 0.870 | 0.751 / 0.393 | -0.24 |
| | FD4, noisy T | 0.877 / 0.835 | 0.676 / 0.168 | 0.22 |
| | FD4, clean-gradient bound | 0.906 / 0.902 | 0.819 / 0.565 | 0.23 |
| | **latent T, FD4** | **0.928 / 0.934** | **0.903 / 0.843** | 0.50 |
| | latent T, FD2 | 0.852 / 0.833 | 0.877 / 0.816 | -0.57 |
| 0.1 K | FD2, noisy T | 0.660 / 0.599 | 0.153 / -0.721 | 0.01 |
| | FD4, noisy T | 0.573 / 0.474 | -0.031 / -1.076 | 0.22 |
| | FD2, clean-gradient bound | 0.785 / 0.769 | 0.562 / -0.006 | 0.05 |
| | FD4, clean-gradient bound | 0.729 / 0.702 | 0.455 / -0.197 | 0.17 |
| | **latent T, FD4** | **0.830 / 0.859** | **0.738 / 0.590** | -0.02 |

- The latent-T route (5/2 : 5/2, 4th-order gradients of the posterior-mean T) is at least as good as every finite-difference option at all three noise levels, including the
  clean-gradient bounds at 0.03 and 0.1 K, and the steep-gradient points are where the margin is largest in v (0.1 K: 0.59 vs -0.72 for the best noisy finite difference).
  At 0.01 K it ties FD4 for u and gains 0.04 in v overall (0.15 at steep points). Plug-in coverage alone is again low (0.84-0.86 at 0.01 and 0.03 K, 0.68-0.72 at 0.1 K);
  the propagated-variance step (exp 18) has not been run on the fronts field.
- A stationary Matern GP does not smooth the fronts away at these noise levels: latent T rms error 0.0052 K (noise sd 0.01; 0.0061 at steep points) and 0.0129 K (noise sd 0.03; 0.0147
  at steep points). Gradient error vs the 4th-order gradient of the true T: 0.06 of std (steep 0.05) at 0.03 K for Tx / Ty. (That reference is itself unreliable across a 2-3 pixel front,
  so the velocity skill is the primary metric.)
- Fits (5/2 : 5/2): 0.03 K eddy 0.313 K, 13.1 km, 42.1 h | fast 0.138 K, 3.13 km, 12.0 h (the fast time scale is at its 12 h bound, the length scale near its 3 km floor), noise sd 0.0295,
  loglik 16780.6; 0.01 K loglik 24891.3; 0.1 K loglik 7056.5. Smoothness again does not matter for the velocity: 3/2 : 3/2 gives the same plug-in skill at every noise level (0.01 K u / v 0.941 / 0.927;
  0.03 K 0.928 / 0.902; 0.1 K 0.829 / 0.736 vs 0.830 / 0.738) with a lower log-likelihood (0.01 K 24875.2, -16.1; 0.03 K 16763.9, -16.7; 0.1 K 7044.2, -12.3). 1/2 : 1/2 at 0.01 K: loglik 24810.4
  (-80.9, same ordering as exp 19); the 0.03 K and 0.1 K 1/2 : 1/2 fronts fits were stopped unfinished (hours under load, ordering already settled; `data/exp21_fit_*.json` holds what finished).
- Stencil on the latent T (fronts field): 4th order is better at <= 0.03 K (0.03 K u / v 0.928 / 0.903 vs 0.852 / 0.877 for 2nd order); at 0.1 K 2nd order is better for v
  (0.787 vs 0.738, u 0.814 vs 0.830), as on the smooth field. A noise-dependent stencil choice may be worth a look; not pursued.

**Exp 22: propagated variance on the fronts field** (exp 18 recipe, 5/2 : 5/2 exp-21 fits, K = 10 draws, equation-error stage-2 noise; skill all / steep, 95% coverage all / steep):

| sigma_T | estimator | u skill | v skill | u cover | v cover | S skill |
|---|---|---|---|---|---|---|
| 0.01 K | plug-in | 0.930 / 0.922 | 0.922 / 0.865 | 0.79 / 0.75 | 0.80 / 0.78 | 0.40 |
| | Rubin-pooled | 0.931 / 0.930 | 0.913 / 0.835 | 0.85 / 0.82 | 0.86 / 0.82 | 0.41 |
| | **plug-in + propagated variance** | 0.930 / 0.922 | 0.922 / 0.865 | **0.86 / 0.82** | **0.88 / 0.87** | 0.40 |
| 0.03 K | plug-in | 0.927 / 0.934 | 0.902 / 0.843 | 0.83 / 0.82 | 0.82 / 0.80 | 0.50 |
| | Rubin-pooled | 0.916 / 0.932 | 0.867 / 0.736 | 0.91 / 0.89 | 0.90 / 0.87 | 0.44 |
| | **plug-in + propagated variance** | 0.927 / 0.934 | 0.902 / 0.843 | **0.93 / 0.91** | **0.94 / 0.95** | 0.50 |
| 0.1 K | plug-in | 0.841 / 0.865 | 0.747 / 0.590 | 0.76 / 0.77 | 0.71 / 0.67 | 0.09 |
| | Rubin-pooled | 0.783 / 0.818 | 0.610 / 0.210 | 0.86 / 0.86 | 0.81 / 0.75 | -0.31 |
| | **plug-in + propagated variance** | 0.841 / 0.865 | 0.747 / 0.590 | **0.92 / 0.92** | **0.90 / 0.91** | 0.09 |

- Draw calibration holds on the fronts field: spread / error of the mean 0.92-1.05 over all points (Tx, Ty, Tt, three noise levels). At the steepest 20% of points the draws are mildly too narrow
  (0.86-0.89 for the gradients, 0.69-0.77 for dT/dt): the actual dT/dt error is 40% larger at fronts (0.01 K: 1.07e-6 vs 7.45e-7) while a stationary model gives the same spread everywhere.
  This is the expected sign of non-stationarity; it does not break the coverage at 0.03 / 0.1 K (steep-point coverage 0.91-0.95 / 0.91-0.92).
- The equation-error noise rule behaved as on the smooth field at 0.03 and 0.1 K (2.17 vs oracle 2.31; 7.50 vs 5.52) but is too small at 0.01 K (0.69 vs oracle 1.49 for the mean products): it contains only
  the derivative-uncertainty part, not the finite-difference discretisation error of the equation itself (32% of dT/dt rms on this field, clean FD2 products), which dominates at low noise. This is why the
  0.01 K coverage stays at 0.86-0.88 (between / within variance 0.45). A floor for the stage-2 noise (discretisation error) would fix it; not tried.
- Skill is the plug-in's (the draws only widen the interval). Rubin pooling again loses accuracy as noise grows (0.1 K v 0.610 vs 0.747; steep 0.21 vs 0.59), so keep the plug-in mean.

**Exp 23: five independent realisations** (fronts + filament field, sigma_T = 0.03 K, seeds 2-6; each seed has its own velocity, source and initial-tracer draw; T-GP 5/2 : 5/2 refitted per seed from a warm start;
true velocity hyperparameters; K = 10 draws; equation-error stage-2 noise for the latent-T rows, oracle noise for the finite-difference rows; `data/exp23_fronts_seed*_sigma0.03.json`,
aggregate with `python experiments/23_multiseed.py agg 0.03 fronts`). Mean +- sd over the 5 seeds, skill (all points; steep-20% mean in brackets) and 95% coverage:

| estimator | u skill | v skill | S skill | cover u / v / S |
|---|---|---|---|---|
| FD2, noisy T | 0.700 +- 0.112 | 0.768 +- 0.108 | -0.17 +- 0.95 | 0.95 / 0.93 / 0.81 |
| FD4, noisy T | 0.720 +- 0.056 | 0.695 +- 0.161 | 0.09 +- 0.36 | 0.97 / 0.93 / 0.87 |
| FD4, clean-gradient bound | 0.783 +- 0.047 | 0.792 +- 0.100 | 0.09 +- 0.31 | 0.99 / 0.97 / 0.86 |
| latent T, FD4 plug-in | 0.802 +- 0.068 (steep 0.774) | 0.851 +- 0.087 (steep 0.874) | -0.01 +- 0.73 | 0.87 / 0.87 / 0.76 |
| **latent T + propagated variance** | 0.802 +- 0.068 | 0.851 +- 0.087 | -0.01 +- 0.73 | **0.94 +- 0.04 / 0.94 +- 0.03 / 0.89 +- 0.12** |

Paired differences (latent-T plug-in minus the finite-difference row, same seed), u / v: vs noisy FD4 mean +0.081 / +0.156 (u better in 4 of 5 seeds, v in 5 of 5; seed 3 u -0.051);
vs the FD4 clean-gradient bound mean +0.018 / +0.059 (u 4 of 5, v 5 of 5, but small: v +0.007 to +0.139).

- The seed-to-seed sd of the skill (0.05-0.16) is as large as many method differences, so single-realisation rankings in exps 13-22 need that caveat. What holds across seeds: the latent-T plug-in is on
  average at or above the CLEAN-gradient FD4 bound, clearly above noisy FD4 and FD2 for v, and by a smaller margin for u (one seed, seed 3, is worse than FD4 for u by 0.051 skill, about 0.003 m/s in rmse
  because that realisation has a small u variance, std 0.070 vs v 0.162).
- The propagated variance gives mean coverage 0.94 / 0.94 (u / v) at the steepest-point level 0.91-0.99; per seed 0.87-0.97 (seed 3 lowest: 0.87 / 0.89, S 0.69). Plug-in alone: 0.87 / 0.87.
- **S is not reliably recovered**: skill varies from -0.97 to +0.5 between seeds (latent-T sd 0.73; FD4 noisy 0.36), so the S = 0.50 of seed 1 (exps 18-22) was a favourable draw. Treat S as a nuisance.
- Fits: noise recovered to 0.0295-0.0299 K in all seeds. The split bounds bind in most: eddy length 10.0-10.3 km (floor 10 km) in 3 of 5, fast time scale at the 12 h bound in 4 of 5, fast length at the 3 km floor in 2 of 5.
  Velocity is insensitive to this (exp 19 / 21), but the fitted scales are bound-driven, not identified.

**Exp 24: ESTIMATED velocity hyperparameters** (the five exp-23 fronts realisations, sigma_T = 0.03 K; stage-1 products and draws as in exp 23; stage-2 noise fixed at the equation-error value;
the Helmholtz + S kernel hyperparameters (gamma, f_delta, l_psi, l_phi, tau_psi, tau_phi, sigma_S, l_S, tau_S) and the diurnal low-rank S scale fitted by the Vecchia marginal likelihood with
log-normal priors (sd 1.0 in log) centred on the start; S is a nuisance and not reported. 'fit A' starts from a generic, roughly right guess (gamma 0.10, f_delta 0.2, l_psi 30 km, l_phi 20 km,
tau 2 d / 1 d, ...); 'fit B' from a deliberately wrong one (gamma 0.05, f_delta 0.5, l_psi 60 km, l_phi 8 km, tau 1 d / 6 h, ...). `data/exp24_fronts_seed*_sigma0.03.json`,
`python experiments/24_estimated_velocity_hyperparams.py agg 0.03`.)

| stage-2 hyperparameters | u skill (steep) | v skill (steep) | cover u / v, plug-in | cover u / v, + propagated variance |
|---|---|---|---|---|
| true | 0.802 +- 0.068 (0.774) | 0.851 +- 0.087 (0.874) | 0.87 / 0.87 | 0.94 / 0.94 |
| fit A (generic start) | 0.806 +- 0.056 (0.783) | 0.844 +- 0.071 (0.850) | 0.87 / 0.85 | 0.93 / 0.91 |
| fit B (wrong start) | 0.807 +- 0.056 (0.785) | 0.843 +- 0.072 (0.850) | 0.87 / 0.84 | (not run) |

- **Estimating the hyperparameters costs essentially nothing in velocity skill**: paired by seed the change from the true values is u +0.005, v -0.007 (fit A), u +0.005, v -0.008 (fit B), worst single seed -0.03.
  Fit A and fit B reach the same optimum (log-likelihoods within 2 units, skill identical to 0.001, parameters within about 10% except tau_phi), so the likelihood, not the prior / start, sets the answer.
- Fitted / true (geometric mean, seed range; A and B agree): gamma 0.75-0.77 (0.66-0.92), f_delta 0.63-0.65 (0.49-0.89), l_psi 1.07-1.09 (0.78-1.49), l_phi 1.07-1.18 (0.90-1.60): the spatial scales and amplitude are recovered
  (the divergent fraction is somewhat low). **tau_psi is 0.31-0.32 of the truth in EVERY seed** (about 0.95 d against 3 d), the same ~0.4x seen in Phase A, so it is systematic, not seed noise; the divergent
  time scale is unidentified (tau_phi 17-26x the truth, 6-16 d; the divergent part holds about 10% of the variance). The nuisance S takes sigma_S x3.9, l_S x0.18 (3.6 km, below a pixel), tau_S x0.26 (6 h):
  it absorbs fast, pixel-scale equation error, as in Phase A. A candidate cause of the short tau_psi (not tested) is leftover temporally correlated equation error that S cannot absorb at those scales and the velocity amplitude / time scale does.
- Propagated-variance coverage with fitted hyperparameters is 0.93 / 0.91 (u / v), a little under 0.94 / 0.94 at the true values.
- Caveat: one small domain (56 km x 48 h, close to 2 psi length scales x 1 psi time scale), so the sigma-l ridge and the time scales are weakly constrained here; longer windows (a 1 week tile) should constrain them better. Not tested.

> **WARNING for forecasting / extrapolation in time (user, 2026-11-06).** The velocity TIME scales are not reliably estimated from a 48 h window: the rotational time scale
> fits at about 0.3x the truth in every seed and the divergent time scale is unidentified (17-26x the truth; the divergent part holds only ~10% of the variance). The spatial scales
> and the amplitude are recovered. This does NOT matter for the intended use, **estimates at the observation points (hindcast / nowcast within the data window)**, where the data
> pin the field and the skill with estimated hyperparameters equals the skill with the true ones (exp 24). It WILL matter for **forecasts, gap-filling over long gaps, or interpolation
> far from data in time**, because the prior time correlation controls how far information is carried (a rotational time scale 3x too short makes the prediction relax to the prior
> mean about 3x too fast; an unconstrained divergent time scale makes the divergent-flow forecast uncertainty meaningless). Anyone using these fits for prediction beyond the data
> should (a) fix the time scales from independent information (e.g. a longer window, other tiles, a pooled regional fit) rather than trusting the single-tile fit, and (b) check
> calibration out of sample. The user expects strong semidiurnal divergent flow (internal tide / tide-driven) in the region of interest, which should make tau_phi (and the tidal
> low-rank term) identifiable on real data; that is untested here because the synthetic field has no such signal.

**Exp 25: scaling of the Vecchia machinery** (`experiments/25_scaling.py`; T-GP 5/2 : 5/2 with the exp-21 hyperparameters, 100 likelihood neighbours, 200 prediction neighbours, 8 Numba threads, idle machine, one process per size):

| N (grid) | construct (maximin order + neighbours + first factor) | factor rebuild = one likelihood evaluation | predict per 1000 points | peak RSS |
|---|---|---|---|---|
| 9,408 (14 x 14 px x 48 h) | 7 s | 0.8 s | 2.4 s | 0.45 GB |
| 37,632 (28 x 28 x 48) | 44 s | 3.3 s | 2.3 s | 0.77 GB |
| 75,264 (28 x 28 x 96) | 160 s | 7.1 s | 2.4 s | 1.2 GB |
| 131,712 (28 x 28 x 168 = 1 deg x 1 deg x 1 week at 4 km) | 474 s | 12.1 s | 2.4 s | 1.8 GB |

- Factor rebuild is linear in N (about 0.09 ms per point); prediction costs 2.4 s per 1000 points independent of N; construction is about N^1.9 (the maximin ordering is O(N^2)) and is the part
  that will limit larger tiles (a cheaper ordering is the next fix). Earlier 20-40 minute fits at 9.4k points were the shared load: uncontended a 500-evaluation fit is about 7 minutes.
- Extrapolation (per-1000-point rates, NOT measured end to end): a 132k-point tile at fixed hyperparameters is about 8 min (construction) + 11 min (T and dT/dt at all points) + a similar stage 2 = about 35 min
  plus about 20 min per uncertainty draw; restricting predictions to the tile core would cut the prediction cost. Fitting the hyperparameters on such a tile would be about 12 s per evaluation (100 min for 500
  evaluations), which is why they should be estimated once on a window and shared.

**Exp 26: tiling seam test** (`experiments/26_tiling.py`; 36 x 36 px (144 km) x 48 h fronts + filament domain, 62,208 points, sigma_T = 0.03 K; SHARED fixed hyperparameters (stage 1: the exp-21 fit from another
window of the same statistics; stage 2: true Helmholtz hyperparameters, noise 2.17); plug-in estimates; each job predicts u, v at every point of its tile; 2 x 2 tiles with cores 18 x 18 px and a buffer of K px on the
sides that border another tile; weights 1 on the core, sin^2 taper across the buffer, normalised by the sum of weights; K = 0 is the hard partition). Scored on 39,600 interior points, 14,256 of them within 3 cells of a seam:

| estimator | N per job | u skill all / seam / away | v skill all / seam / away | rms difference to the single job at seams (u / v) vs its posterior sd | coverage u / v |
|---|---|---|---|---|---|
| single job | 62,208 | 0.816 / 0.785 / 0.822 | 0.842 / 0.797 / 0.848 | - (sd 0.025 / 0.024 m/s) | 0.83 / 0.82 |
| tiles, buffer 0 px | 15,552 | 0.747 / 0.517 / 0.822 | 0.791 / 0.571 / 0.848 | 0.042 / 0.040 m/s (1.7x the sd) | 0.84 / 0.84 |
| tiles, buffer 2 px | 19,200 | 0.816 / 0.781 / 0.823 | 0.842 / 0.790 / 0.850 | 0.012 / 0.010 (0.5x) | 0.85 / 0.84 |
| tiles, buffer 4 px | 23,232 | 0.818 / 0.789 / 0.823 | 0.842 / 0.791 / 0.850 | 0.007 / 0.006 (0.3x) | 0.84 / 0.83 |

- A hard partition loses 0.27 (u) / 0.23 (v) skill in the seam band; a 2-pixel (8 km) buffer already recovers the single-job skill (overall identical to three decimals), a 4-pixel (16 km) buffer brings the seam difference to
  about a quarter of the posterior sd. Away from the seams all methods are identical. Coverage is unchanged by blending (plug-in variance only; the propagated variance was not run here).
- Cost: with a 4-pixel buffer the four tiles contain 93k points in total (1.5x the domain; 1.2x for a 2-pixel buffer) but each job is a third of the size and independent; per tile 84-173 s against 591 s for the single
  job (timings concurrent with the reference job, so only indicative). The buffer is much shorter than the 2-3 correlation lengths (psi length 30 km) suggested earlier and still suffices here, because the tile
  keeps its own T data in the buffer and the hyperparameters are shared.
- Caveats: one realisation, a 144 km domain (core 72 km = 2.4 psi length scales), stationary shared hyperparameters (spatially varying ones are not tested), true stage-2 hyperparameters, plug-in only,
  and the blended posterior variance has not been checked for calibration.

**Why u and v differ in some of these tables (checked, not assumed).** The Helmholtz prior is isotropic, so u and v are exchangeable a priori. A transposition test (swap x <-> y and u <-> v,
FD4 clean-gradient products, 0.03 K) swaps the results exactly: original u / v skill 0.906 / 0.819 (rmse 0.0298 / 0.0460), transposed 0.819 / 0.906 (0.0460 / 0.0298). So the code is
symmetric and the difference comes from the DATA: in this field the true gradient is larger in x than in y (rms Tx 5.1e-5 vs Ty 4.1e-5 K/m over the scored points; 9.3e-5 vs 7.4e-5 at the
steep points), and the data constrain u and v through z = -Tx u - Ty v + S, so the information on each component scales with its gradient squared (1.57x more for u here) and the
weaker component degrades faster as the noise grows (finite differences at 0.1 K: v 0.15, u 0.66). Realised velocity std is also slightly different (u 0.097, v 0.108 m/s). The
smooth-field gradient ratio was not computed; the near-equal u / v skills there (e.g. 0.90 / 0.90 at 0.03 K) are consistent with this but not checked.

## Real-data input: loader written and first real tile (2026-11-06)

**Loader (done, tested).** `pkg/inversion_sst_gp/spacetime/{himawari,geo,tiling}.py`, tests in `tests/test_himawari_tiling.py` (12 tests, plain runner, synthetic pysatview-format files plus the real ones).
- `himawari.load_cube(nc_dir, t0, t1, step, lonlim, latlim, qmin=5, sst_range, stride, reduce)` reads the per-hour NetCDFs that `pysatview` writes (`YYYYmmddHHMMSS.nc`; SST int16 x 0.01 K + 273.15
  and `quality_level`) and also raw global granules (`...-STAR-L3C_....nc`, lat descending, cropped lazily: 1.2 s for a 2 deg box out of the 3 GB file). It needs only xarray / netCDF4 (no earthaccess,
  no network, so it runs on the HPC); the download stays in pysatview (needs an Earthdata `.netrc`: none on this machine and `earthaccess` is not in the `drift` env, so downloading is a step for the user
  or the HPC). Mask = `quality_level >= 5` and finite SST in 268-313 K; missing hours become all-NaN slabs; a different lon / lat grid between files raises. `reduce='stride'` (every 2nd pixel, the settled 4 km
  choice) or `'mean'` (2 x 2 blocks, `min_frac`). `ocean_mask`, `coverage`, `find_windows(cov, length, min_mean, min_each)` pick periods of good coverage.
- `geo.LocalFrame(lon0, lat0)`: WGS84 azimuthal-equidistant frame (pyproj, vectorised; distances from the centre exact, tested against the geodesic to 1 mm); `metric_spacing` gives metres per cell
  east / north per row from the ellipsoid radii of curvature, so gradients are taken along local EAST / NORTH on the native lon / lat grid (`tiling.grid_gradients_metric`, equal to `impute.grid_gradients`
  on a uniform grid) and the (u, v) the velocity stage returns are east / north components. The GP kernels use the frame (x, y) in metres; the frame x axis differs from local east by < 0.5 deg for a
  tile of 1-2 deg (not corrected in the kernels; `LocalFrame.to_east_north` removes it if wanted).
- `tiling.py`: `make_tiles(shape, core, buffer)` on the INDEX grid (cores partition the cube exactly; a remainder < half a core merges into the last core; buffer only on sides bordering another tile),
  `tile_weights` (exp 26 sin^2 taper, also in time), `Blender` (sum w f / sum w, NaN cells get zero weight per field), `write_manifest` / `read_manifest` (JSON), `extract_tile` -> `TileData`
  (x, y in metres from the block centre, t in seconds, anomaly T with the tile mean removed, obs mask, hx / hy, weights, `gradients`, `scatter`, `coverage`). Temporal blending is the same
  construction but its seam behaviour is NOT tested (exp 26 was spatial only).

**Real data available locally:** `../pysatview/test_data/himawari/nc/`: four hourly cropped files 2025-10-01 00-03 UTC (111-114E, 25-20S, 0.02 deg, 250 x 150 px; 93-99.7% of the ocean pixels valid) and one raw granule 2026-02-18 00 UTC. Only quality levels 0 (no data) and 5 occur in both (checked in the cropped files and in the granule's 111-114E, 26-18S box),
so the quality threshold is effectively the clear-sky mask. The SST is quantised to
0.01 K (quantisation sd 0.003 K, negligible here).

**First real tile (exps 27, 28; one 28 x 28 px (4 km, 1.1 deg) x 4 h tile, 100% observed, 3,136 points, 5/2 : 5/2 two-component T-GP, split bounds; time scales 27 h / 9 h and the diurnal term FIXED because 4 h
cannot identify them).**
- **Noise level: the likelihood chooses sigma_T = 0.0126 K (stride) and 0.0081 K (2 x 2 mean) at 4 km**, i.e. about 0.01 K, NOT the 0.03 K of most synthetic tests. Cross-checks: the lag-1 h difference
  (mean removed) gives an upper bound sigma_T <= 0.034 K (it includes real change); small-lag variograms of the raw 2 km field follow gamma ~ h^1.65 down to one pixel with NO visible nugget
  (the L3C product is smooth at the pixel scale, so sigma_T is bounded, not measured, by this route; sqrt(gamma(1 px)) = 0.03 K). Ratio of the two fits 0.64 (white noise would give 0.5), so some of it is
  pixel-correlated. Caveats: a 4 h daytime sample, one tile, kernel misspecification can leak into the noise, and the fit sits at or near the bounds (eddy amplitude 9-10 K against its 10 K bound, l = 214-300 km, i.e. a smooth
  background; fast component at its 10 km length bound), so the split into eddy / fast is bound-driven here, as in the synthetic fits.
- **Consequence for the method (see exps 12, 20):** at sigma_T = 0.01 K finite differences are no longer limited by gradient noise; the latent-T route still gains (v +0.04 overall, +0.15 at steep points,
  u ties FD4 in the synthetic fronts / smooth tests) but by less, and the stage-2 noise rule needs a floor for the finite-difference discretisation error (omitted at 0.01 K, coverage 0.86-0.88; not
  tried). Phase B is mainly justified at sigma_T >= 0.03 K, so the real sigma_T needs confirming on a longer, more varied sample before the choice is final.
- **Leave-out standardised residuals (hyperparameters fixed): calibrated overall, strongly heteroscedastic.** 20-fold leave-out of every pixel: sd(z) = 1.02, but |z| quantiles 50 / 90 / 99 / 99.9% =
  0.53 / 1.50 / 3.44 / 6.45 (Gaussian 0.67 / 1.64 / 2.58 / 3.29) and fraction |z| > 4 = 0.7% (Gaussian 0.006%). The tails are the FRONTS: by local gradient (of the observed anomaly) sd(z) is 0.63
  (below the median gradient), 0.98 (median-90%), 1.70 (90-99%) and 3.0 (top 1%; 36% of those points have |z| > 3); the outliers lie along the strong fronts, persist across the four hours, alternate
  in sign along the front, and about half have a neighbour that is also an outlier (24 of 50). So the stationary GP is over-confident at fronts and under-confident in the smooth bulk; the
  predictive variance (and the draw spread feeding the stage-2 noise rule) inherits that. This is the real-data version of the synthetic fronts finding (dT/dt error 40% larger at fronts).
  Whole-hour removal (time interpolation, hours 1 and 2): sd(z) 0.99-1.05, rmse 0.018 K, but mean z +0.26 / -0.28 (bias: curvature of the diurnal warming with the time scales fixed); random 5% / 4 x 4 px holes
  rmse 0.02 K. `data/exp28_real_tile_stride_z.png` shows the maps.
- Candidate remedies (not tried): a gradient-dependent amplitude for the fast component (non-stationary variance, e.g. sigma2(x) proportional to a smoothed |grad T| from a first pass; needs an amplitude
  column in the Vecchia rows), a heavier-tailed likelihood or a robust outlier screen on leave-out residuals, or calibrating the variance by a gradient-binned inflation factor. To be tested on the synthetic
  fronts (truth known) before real data.

## Barotropic tide: regional least-squares stage 0 (exps 29-32, 2026-11-06; synthetic only)

**Design (user, 2026-11-06).** Barotropic tidal currents are a significant part of the velocity. A separate regional stage fits M2, S2, K1, O1 (kwarg `constituents`) by least squares from SST on a coarse
grid and a long window, with amplitude AND phase of every constituent varying slowly in space (depth changes over the domain), data shallower than 200 m removed (kwarg `min_depth`, default 200 m, `None` keeps
everything). The tide is then subtracted from the tracer equation in each tile and added back to the GP velocity. Sequential, not joint: the GP never sees the tide.

**Model** (`spacetime/tides.py`). dT/dt = -Tx u_bt - Ty v_bt + S_nuis + noise, with u_bt = sum_k [a_k(x, y) cos w_k t + b_k(x, y) sin w_k t] (same for v) and a_k, b_k tensor-product cubic B-splines with
breakpoint spacing `spacing` (kwarg; second-difference P-spline penalty). S_nuis = diurnal-heating harmonics (0..H of the solar day, spatial basis 3x coarser) fitted jointly, because the heating shares
frequencies with K1 / S1 and (its 2nd harmonic) exactly with S2. Linear in the coefficients, normal equations accumulated in row chunks (sparse), optional forked workers (`n_jobs`). Rayleigh limits
(`rayleigh_days`): M2 / S2 14.8 d, K1 / O1 13.7 d, M2 / N2 27.6 d; K1 vs P1 and S2 vs K2 need about 6 months, so the fitted K1 / S2 contain P1 / K2 (and K1 cannot be told from S1 apart from
the gradient coupling). Unit tests `tests/test_tides.py` (exact-consistency rows: recovery of spatially varying amplitude / phase, shallow-row masking, nuisance necessity, pooled = pixel normal equations,
FD4 time derivative keeps M2 to 0.2%).

**Synthetic test bed** (`synthetic.TideField`, `SyntheticConfig(tide=True)`; NOT dynamically consistent). Synthetic depth: a meandering shelf break, 60 m on the shelf to 3000 m, depth 200 m about 37 km
south of the break (22% of the 3 deg x 3 deg domain is < 200 m). Amplitude A200 sqrt(200 / depth) (M2 0.30, S2 0.12, K1 0.10, O1 0.07 m/s at 200 m), phase from a progressive wave crossing the shelf
break (w * integral of ds / sqrt(g depth)), elliptical v with constituent-specific ratio and lag, plus a ROUGH shelf component (40 km wavelength, 50% of the local amplitude, weight 1 / (1 + (depth / 150)^4))
that a smooth regional fit cannot represent (the reason for the depth mask). 84 x 84 px (4 km) x 60 d hourly, sigma_T 0.01 K, fronts + filament, the usual eddy (Helmholtz) field and S; padding 45 cells instead of
the worst-case drift (which would be absurd with a tide); `n_modes` 200, `dt_sim` 1 h (about 1 h of CPU). The default configuration reproduces the earlier datasets bit for bit (checked). RMS: tide 0.10 m/s in
deep water (0.51 m/s on the shelf), eddies 0.13 m/s; equation residual of the FD products 26% of dT/dt rms (hourly semi-Lagrangian step + FD truncation). Data live in `~/browsesst_data/` (outside OneDrive).

**Results of the regional fit** (`experiments/30_tide_regional_ls.py`; 16 x 16 blocks of 20 km, noisy 4 km data, 4th-order differences, rows from days 10-40 unless noted; error = rms of the tidal (u, v) error
in the DEEP region (>= 200 m) as a percentage of the tide rms 0.10 m/s, on the 4 km grid at every 4th hour; one realisation, second 30-day window gives 23% against 21%):

| estimator | deep error | M2 amp error (of 0.110) | S2 phase | K1 phase | O1 phase |
|---|---|---|---|---|---|
| 20 km block MEANS of dT/dt, grad T (the original proposal), 30 d | 21% | 0.013 | 5.4 deg | 8.8 deg | 14.8 deg |
| same, 15 d / 60 d | 28% / 18% | 0.014 / 0.013 | 10.3 / 4.1 | 14.3 / 6.6 | 20.1 / 9.4 |
| same, no nuisance (heating) term | 23% | 0.014 | 6.1 | 13.9 | 14.8 |
| same, NO depth mask | 30% (shelf 47% vs 70%) | 0.018 | 6.3 | 9.4 | 17.6 |
| same, spline spacing 50 / 80 / 120 km | 36% / 21% / 21% | 0.013 / 0.013 / 0.016 | 6.1 / 5.4 / 3.7 | 13.2 / 8.8 / 7.0 | 21.9 / 14.8 / 11.7 |
| same, every 5th pixel (no averaging) | 35% | 0.013 | 7.5 | 13.4 | 23.8 |
| same, 8 km blocks (909k rows, 14 min serial) | 12% | 0.009 | 2.4 | 4.9 | 9.2 |
| same, noise-free T | 21% | | | | |
| **moment-pooled 20 km blocks (`pool_rows`)**, 30 d | **16%** | 0.014 | 2.7 | 4.0 | 5.4 |
| pooled, 15 d / 60 d | 17% / 15% | | | | |
| pooled 40 km blocks | 20% | | | | |
| pooled, noise-free T | 16% | | | | |
| pooled + known depth scaling (depth/1000)^-0.5 (the truth here) / ^-0.25 (misspecified) | 13% / 15% | 0.012 / 0.013 | 2.2 / 2.3 | 3.9 / 3.9 | 5.1 / 5.3 |

- **Plain block means of the equation lose most of the information** (21% vs 12% at 8 km and 35% with no averaging at all): the tide multiplies the gradient at every pixel, but a block mean of the gradient
  cancels opposite-signed gradients. The fix costs nothing in rows: each block and time keeps the 4 x 4 second-moment matrix of (Tx, Ty, 1, dT/dt), factored into 4 pseudo-rows that reproduce the pixel
  normal equations exactly (`pool_rows`, tested). With it the 20 km grid proposed by the user does well (16%, phases 3-5 deg), the window length hardly matters (17% at 15 d, 15% at 60 d: the Rayleigh
  limit of 14.8 d is what counts) and coarser (40 km) blocks lose little (20%).
- The remaining 15-16% is not measurement noise (noise-free T gives the same), not the window, and mostly not spline representation (known depth scaling gains 3 points). Absolute errors: amplitude about 0.01
  m/s (M2) and 0.003-0.005 (others); the M2 amplitude error is the largest part and, in the unpooled fits, is concentrated on the shelf-break slope (200-1600 m: rms 0.016-0.024 against 0.011 on the
  3000 m plateau), where the amplitude changes fastest and the rough shelf term is strongest (leakage from it is a candidate cause, not tested).
- **Masking the shallow data matters (21% vs 30%)** and the cost is only that the fit cannot be used on the shelf (70% error there, flagged by `TideFit.supported` / `velocity(mask_unsupported=True)`).
- The diurnal-heating nuisance term matters mainly for the K1 phase (8.8 vs 13.9 deg). Spline spacing 80-120 km is right for a smoothly varying tide; 50 km fits noise.
- Depth scaling (Green's law) is an option (`depth_scale`, scale the gradients before the regression and the output after): +3 points here, where the truth has exactly that scaling, so treat as an upper bound.

**GP stage after the tide** (`experiments/31_tide_gp_stage.py`; 28 x 28 px x 48 h tiles at day 20, stage 1 refitted on the tile (noise recovered 0.0098 / 0.0099 K, true 0.01), TRUE eddy hyperparameters in stage 2,
oracle-equation-error noise; scored against the true TOTAL velocity; skill relative to the variance of the eddy part so that the tide cannot flatter it):

| tile | tide left in the data (`none`) | regional-LS tide removed (`fitted`) | true tide removed (`oracle`) |
|---|---|---|---|
| deep plateau (3000 m), tide rms 0.068 m/s | rmse 0.0339, skill 0.858 (u 0.882, v 0.818) | 0.0255, 0.920 (0.939, 0.886) | 0.0254, 0.921 |
| shelf-break slope (361-3000 m), tide rms 0.074 | 0.0374, 0.831 (0.825, 0.835) | 0.0280, 0.905 (0.906, 0.905) | 0.0278, 0.907 |

- **The sequential approach works and the regional fit is as good as the true tide**: the fitted-tide response gives the same velocity skill as the oracle (rmse 0.0255 vs 0.0254), although the fitted tide has
  an error of 0.008-0.011 m/s (the GP error is 0.025-0.028). Removing the tide cuts the total-velocity rmse by 25% in both tiles.
- **With the true hyperparameters the GP alone absorbs most of the tide** (skill 0.83-0.86): at sigma_T = 0.01 K the equation constrains the total velocity pointwise, so the prior time scales matter little for
  estimates AT the observation points. The reasons to model the tide anyway are the 25% rmse, the estimated hyperparameters (exp 32 below), forecasting / gap filling, and a clean split into barotropic
  and eddy parts.

**Estimated hyperparameters** (`experiments/32_tide_hyperparams.py`; the shelf-break slope tile of exp 31, cached stage-1 products, stage-2 noise fixed at the exp-31 value; the Helmholtz + S hyperparameters
(9 + the diurnal scale) FITTED by the Vecchia marginal likelihood from the generic start of exp 24 ('fit A'), 25 L-BFGS iterations = 374 evaluations, about 48 min each under heavy machine load; **neither fit
converged** (`success=False`)). Fitted / true:

| | gamma | f_delta | l_psi | l_phi | tau_psi | tau_phi | sigma_S | l_S | tau_S | total rmse (m/s), skill vs eddy variance |
|---|---|---|---|---|---|---|---|---|---|---|
| tide left in the data | 4.59 | 0.05 | 3.91 | 1.20 | 0.30 | 56 | 6.3 | 0.15 | 3.5 | 0.0279, 0.906 |
| regional-LS tide removed first | 1.14 | 0.51 | 1.24 | 1.06 | 0.25 | 19 | 6.0 | 0.15 | 0.71 | 0.0262, 0.917 |
| (exp 31, TRUE hyperparameters: tide in / removed) | | | | | | | | | | 0.0374, 0.831 / 0.0280, 0.905 |

- **An unmodelled tide does drag the SPATIAL hyperparameters**: with the tide in the data the fit inflates the prior rms velocity (gamma 4.6x, 0.44 m/s against a true 0.095), switches the divergent fraction almost off (f_delta 0.05x of the true
  0.1) and lengthens l_psi 3.9x; with the fitted tide removed first, gamma, l_psi and l_phi are within 25% and f_delta within 2x. If the fitted hyperparameters are used for anything beyond the velocity at the
  observation points (energy spectra, the divergent fraction, a regional prior), the tide has to come out first.
- **The TIME scales are not rescued by removing the tide**: tau_psi is 0.25-0.30 of the truth in both variants (the same shortening as exp 24, and this window is 48 h against a true 3 d), tau_phi is 19-56x too
  long (true 12 h; the tide-in fit sits near its 35 d bound), and S is the same in both (sigma_S 6x, l_S 3 km = 0.15x). So the time-scale warning in the exp 24 section stands. **The expectation that a strong
  semidiurnal divergent flow makes tau_phi identifiable is NOT supported by this test**, but the synthetic divergent eddy field has no 12 h periodicity, only a 12 h decorrelation time; if the real divergent
  flow is genuinely semidiurnal, a periodic (cos(omega t) x Matern) temporal factor for the divergent part would be the better model (idea only, untested).
- **Velocity at the observation points hardly cares**: with ESTIMATED hyperparameters the tide-in run gets skill 0.906 (rmse 0.0279), better than the true-hyperparameter tide-in run (0.831, 0.0374) and as
  good as the true-hyperparameter run with the tide removed (0.905). The fit adapts (large gamma, S absorbing error), so the 25% rmse gain of exp 31 shrinks to 6% (0.0279 -> 0.0262, skill 0.906 -> 0.917)
  when the hyperparameters are estimated, and that difference is one tile and one realisation, so it is not established. That estimated hyperparameters beat the true ones is the same observation as in exp 24:
  the true prior is not the best prior for the equation-error structure that is actually present.
- **What this means for the design**: stage 0 is justified as protection of the spatial hyperparameters and as a separate barotropic product (and for forecasting / gap filling), NOT by a large gain in velocity
  accuracy at the observation points. Caveats: one tile, one realisation, unconverged fits, a 48 h window (real tiles are 1 week), synthetic tide.

## The HPC chain: per-tile pipeline, regional tide hook, SLURM drivers (exp 33, 2026-11-06)

**What was built** (all tested locally; nothing run on the HPC yet):
- `spacetime/bathymetry.py` (`load_depth`: GEBCO-style NetCDF -> cell-mean depth on the cube's lon / lat axes; elevation or depth variable, either latitude order, 0..360 longitudes, land = 0),
  `spacetime/regional.py` (`fit_region(cube, depth, TideConfig)` -> `TideModel`: chunked 4th-order products with ellipsoid metric factors, `pool_rows` 20 km blocks, `fit_tides`; `TideModel.velocity(lon, lat, time)`,
  `.tile_velocity(td)`, `.constituents(lon, lat)` amplitude / phase maps, `.save/.load`), `spacetime/pipeline.py` (`Hyper` JSON, `PipelineConfig`, `run_tile`, `fit_hyper`, `pool_hypers`, `write_tile`, `merge_tiles`,
  the `VelocityPrior` interface). `tides.pool_rows` now accepts 2-D pixel positions (lon / lat pixels are not exactly rectilinear in the local frame); `tiling.extract_tile` reads lazily (`isel`).
- `hpc/`: `make_cube.py`, `make_manifest.py` (tiles, `*_tiles.csv`, sample tiles), `run_tide.py`, `fit_hyper.py`, `pool_hyper.py`, `run_tile.py`, `merge.py`, `job_*.sbatch`, `submit_all.sh` (dependency chain),
  `env.sh.example`, `README.md`. Tests: `tests/test_pipeline.py` (7 tests) on top of the loader (12) and tide (7) suites, all passing.

**The chain per tile:** latent-T GP on the observed pixels -> T and dT/dt at every grid point -> 4th-order east / north gradients -> `z = dT/dt + u_bt . grad T` (tide from the regional fit, zero where unsupported)
-> velocity GP (`VelocityPrior`) -> eddy u, v, S, block variances -> total = eddy + tide. Hyperparameters are shared and fixed (`Hyper`), the stage-2 noise is fixed (`Hyper.stage2['noise']`) or fitted by ML per tile.
`fit_hyper` fits them on a few sample tiles (stage 1 by the T-GP likelihood, stage 2 after removing the tide), `pool_hypers` takes the log-median. NOT in the chain: Matheron draws / propagated variance (the returned
variances are the stage-2 block variances only), per-tile stage-1 refits (available, `stage1_maxiter`), the temporal-gap model.

**Covariance model (user, 2026-11-06):** the final model will be a simpler version of `wedge/browseGP`, to be specified later. Its divergent field is a semidiurnal WAVE in time
(`temporal_wave_njit(tau, tau0, omega0, gamma, nu_t)`; M2, and an M4 copy) that represents the internal tide, plus rotational, inertial and fast components, a barotropic tide block and a mean; its inferred parameters from the
in-situ data (small region) can be taken directly: `browseGP/results/mcmc_M4_v0.01/mcmc_results.json` (posterior medians: rotational gamma 0.072, ell 54 km, tau 4.4 d; divergent wave gamma 0.120, ell 33 km, tau 7.1 d;
inertial gamma 0.024, ell 18 km, tau 2.7 d; fast gamma 0.068, ell 2.0 km, tau 1.3 h; M4 gamma 0.025, ell 12 km, tau 3.9 d; sigma_tide 0.046, sigma_mean 0.054, sigma_n 0.014; parameterisation to be mapped onto ours).
The plug-in point is `pipeline.VelocityPrior` (`cov_params`, `covfunc`, `lowrank`, `free`, `bounds`); the default `HelmholtzMaternPrior` is the model used in all experiments so far. A wave-type divergent kernel
changes the exp-32 time-scale conclusion (tau_phi unidentified) only if the real divergent flow is actually periodic; that is then testable on the real data. browseGP was read, not modified.

**End-to-end test on the tide synthetic through the CLI scripts** (`experiments/33_pipeline_synthetic.py`; 3 deg-equivalent box = 84 x 84 px x 30 d mapped onto a lon / lat grid at 123E 14S; GEBCO-style file from the
synthetic depth; shared hyperparameters = exp-31 stage-1 refit + TRUE stage-2 values + the exp-31 noise, so this tests the plumbing and the tide hook, not estimation; tiles 24 h x 28 px cores, 6 h x 4 px buffers, block 36^3):
- Tide stage by `run_tide.py`: 106-122 s on one core (30 d, 84 x 84, `--time-stride 2`, 367k pooled rows, 928 parameters); deep-water error 16% of the tide rms (0.016 of 0.099 m/s), as exp 30, so the lon / lat path
  and the chunked products reproduce the earlier result. Shelf (< 200 m): 66% error, as before (extrapolation).
- **Bug found by this test and fixed:** the default support radius of `fit_tides` is 1.5 x the spline spacing (120 km), so the whole shelf counted as "supported" and received an extrapolated tide (66% error).
  `fit_region` now defaults to one block width (block x pixel spacing = 20 km); the shelf test uses the default.
- Four tiles per variant (t10_01_01, t10_02_01 and the time neighbours t09_02_01, t11_02_01), merged by `merge.py`, scored on the cells of the tile cores (75k cells, all deep water >= 200 m):

| variant | total-velocity rmse (m/s) | skill vs eddy variance | tide-term error (tide rms 0.051) |
|---|---|---|---|
| no tide removal (GP absorbs it) | 0.0393 | 0.820 | 0.0513 (= the tide, not modelled) |
| regional tide removed first | 0.0304 | 0.892 | 0.0135 |

  The tide hook gives the gain seen in exp 31 (23% rmse) on tiles cut and blended by the production scripts.
- **First temporal seam check** (one spatial tile, adjacent weekly-style windows 24 h apart with 6 h buffers, the 12 h around the seam; rmse against the truth, m/s): blended 0.0299 vs hard cut 0.0305 vs tile A
  0.0302 / tile B 0.0311 with the tide, and 0.0391 vs 0.0416 (A 0.0392, B 0.0399) without it. The two windows disagree by 0.0160 (with tide) and 0.0245 (without) rms in the overlap, so removing the tide halves the
  tile-to-tile discrepancy. The blend does not degrade the seam; this is one seam, one realisation, true hyperparameters, a 6 h buffer (the HPC default is 12 h, untested).
- **Cost per tile** (41-47k points, 2 threads, machine under load from other jobs): 7-10 min = stage-1 build 100 s (mostly the O(N^2) maximin ordering) + stage-1 prediction 120-135 s + stage-2 build 65-80 s +
  stage-2 prediction 205-245 s. Peak memory not measured (about 0.9 GB seen mid-run). A real 1 deg x 1 week tile with buffers has ~250k points (5.5x): prediction scales ~linearly, the two orderings as N^2
  (474 s at 132k points in exp 25), so expect roughly 1-2 h per tile at 4 threads before any optimisation; to be timed on the HPC with ONE tile first. Cheap savings not yet taken: reuse the stage-1 ordering for stage 2
  (same grid points when coverage is complete), a cheaper ordering than maximin.
- **Hyperparameter plumbing** (`fit_hyper.py` on 2 small tiles, 2 + 2 iterations from the truth, then `pool_hyper.py`, then `run_tile.py` with the pooled file): runs end to end (50 + 44 evaluations, ~4 min per 11k-point
  tile); two iterations only test the wiring, not the estimates.

**Not yet tested / open for the real data:** GEBCO on the real box and its 200 m mask; cloud gaps in the finite-difference stencils of the tide stage (a missing hour removes 5 consecutive time derivatives) and in the
latent-T GP (the `tt_sd_factor` guard that drops very uncertain stage-2 rows is untested); the step in the subtracted tide at the edge of the support (hard mask; a taper would be smoother); whether the GP stage
should also drop shallow-shelf pixels; real hyperparameters (the defaults are placeholders from synthetic fits); FES2022 (`uwadev/fes22_tides`) as an independent check of the fitted amplitude / phase maps.

## Real test dataset (28 days, 3 deg x 3 deg) and the first real-data results (exps 34-35, 2026-11-06)

**The test file** (`hpc/make_test_dataset.py`; `/mnt/c/Users/00099894/Large_Data/Other/browseSST/test_3x3_may2026.nc`, 4.3 MB; the user will copy it to the HPC): Himawari-9 L3C SST (quality 5) from
`.../Large_Data/Other/browseSST/himawari/nc` (670 continuous hourly files, 2026-05-01 00 UTC to 05-28 21 UTC, 550 x 550 px at 0.02 deg over 114-125E, 21-10S) cropped to lon 121-124, lat -14 to -11, stride 2
(0.04 deg, 4.4 km): `sst` 670 x 75 x 75; `depth` = cell mean of the SUNTANS unstructured bathymetry (`SUNTANS_bathy_unstructured.nc`: 225k wet cell centres `xv, yv, dv`, 10-6000 m) on the same grid (98% of the cells hold
mesh points; the empty ones are reefs / islands), **NaN where shallower than 100 m (4% of the cells)**, plus `depth_raw`. NaN depth means "excluded" in the tide stage (pixels never enter its equations; before this
change NaN meant "unknown, keep"). Box: deep (500-3800 m) in the west, a plateau at 100-300 m in the east. Coverage (valid fraction of the ocean pixels, daily): 0.8-0.99 on days 0-14 and 22-27, a **cloud gap on
days 15-21 (about 0 on May 17-19)**; mean 0.72. `hpc/run_test.py` + `sbat_test_*.sbatch` + `submit_test.sh` are the HPC drivers in the style of `browseGP/scripts` (pilot mode, log files; HPC paths are assumed, see
`hpc/README.md`); the driver ran all stages locally on the real file with small tiles (return code 0). The Himawari downloader stopped by itself at 15:48 after May 28 21:00; the user resumes it overnight.

**The SST-regression tide could not be validated on the real month** (`experiments/34_real_tide_control.py`; fit M2 S2 K1 O1, 80 km splines, nuisance H = 4, `--min-depth 100`, 40 s on one core):
- The fit returns large amplitudes (median over the supported 20 km cells, u / v: M2 0.21 / 0.26, S2 0.21 / 0.23, K1 0.17 / 0.22, O1 0.18 / 0.22 m/s; residual rms of dT/dt 1.95e-5 K/s, three times the synthetic).
  The user's expectation is that barotropic tides here can exceed 0.2 m/s rms and that internal tides can be as strong, so large amplitudes are NOT in themselves implausible (my first reading, that S2 ~ M2 was
  unphysical, was too strong).
- **Control frequencies:** the same code and data with four fictitious constituents (periods 20, 16.4, 13.6, 11.25 h, away from every tidal and solar line) give median amplitudes 0.14-0.23 m/s, mean rms amplitude
  0.34 against 0.40 m/s for the real constituents (ratio 1.17; 1.15-1.23 with spline spacing 160-300 km or with the nuisance term richer, poorer or absent). So the estimator's own scatter is as large as the
  amplitudes the user expects, and the fit cannot tell a tide of that size from the floor.
- **Split-half (even / odd days, same time origin; `34_real_tide_control.py split`):** complex coherence of the fitted amplitude fields 0.31 / 0.14 / 0.10 / 0.15 (M2 / S2 / K1 / O1), median phase difference 75-85 deg,
  against 0.29 / 0.21 / 0.12 / 0.11 and 76-112 deg for the controls: the two halves do not reproduce each other. With a floor of about 0.2 m/s per component per half and a signal of 0.2 m/s the expected coherence is
  about 0.3-0.5, so the test cannot discriminate either. It is also reduced for M2 by the phase wander of internal tides (not coherent over weeks) and by their spatial structure (wavelengths of order 100-200 km,
  comparable to the 80 km spline spacing), neither of which is a barotropic tide.
- **Why the floor is high** (variance decomposition of the 4th-order dT/dt on the test cube): only 41% of the pixel-hours survive the stencil (5 consecutive hours and a 5 x 5 pixel cross; mean coverage is 72%);
  rms dT/dt 2.2e-5 K/s against rms |grad T| 1.1e-5 K/m (so the tide signal at 0.1 m/s is about 1e-6 K/s, 5% of the rms); 34% of the dT/dt variance is spatially uniform per hour (box-mean heating and
  retrieval changes), 66% remains after removing the per-hour mean and 53% after a per-hour plane; the hourly pixel differences with the box mean removed have rms 0.068 K, so the white-noise level is at most 0.048 K
  (the finite-difference noise of dT/dt is then 1.3e-5 K/s, about a third of its variance). A richer or absent diurnal nuisance term changes the residual only from 2.1e-5 to 1.94e-5 K/s.
- **Does a longer record or a stiffer basis fix it?** (controls vs the four tidal constituents, mean rms amplitude over the supported cells, m/s; same code, 80 km or 300 km splines):

  | window | 80 km: controls / tidal / excess in quadrature | 300 km: controls / tidal / excess |
  |---|---|---|
  | 7 d | 0.50 / 0.62 / 0.37 (constituents unresolved) | 0.37 / 0.40 / 0.14 |
  | 14 d | 0.41 / 0.46 / 0.20 | 0.30 / 0.33 / 0.15 |
  | 28 d | 0.34 / 0.40 / 0.21 | 0.25 / 0.28 / 0.14 |

  The control floor falls as T^-0.28 to -0.30 (not T^-1/2: the residual is red, correlation of order a day or more), so 90-120 days would bring it to about 0.23-0.25 (80 km) or 0.16-0.18 m/s (300 km),
  i.e. down by a third, to roughly the size of the ~0.2 m/s signal that the quadrature excess suggests (SNR about 1 per cell, 80 km). The excess is stable at ~0.2 m/s for 14 and 28 d at 80 km (consistent
  with a real signal of the size the user expects; but the controls are a noisy null, and the solar-harmonic lines S2 and K1 may carry more heating residual than the controls do) and lower (0.14-0.15) at
  300 km, where the stiff basis cannot follow a tide that must vary with depth (3500 m in the west, 100-300 m on the eastern plateau, a Green's-law factor of ~3). Stiffening 80 -> 300 km lowers the floor by only 27%
  (0.34 -> 0.25), far less than the ~3.7x that the 14-fold reduction in parameters would give for white noise: the floor is a spatially coherent, large-scale residual, not estimator variance from too many coefficients.
  The tidal / control ratio is 1.1-1.2 in EVERY variant tried (record 7-28 d, spacing 80-300 km, nuisance richer / poorer / absent, hourly mean or plane of dT/dt removed: floor 0.34 -> 0.28 / 0.25 but the
  excess falls with it), so none of these knobs lifts the tide decisively above the floor.
- **The residual is real SST variability, not white noise** (corrects the guess above): the temporal structure function of the pixel SST (box mean removed per hour) is S(1 h) = 0.0047, S(2) = 0.0125, S(3) = 0.0189,
  S(6) = 0.031, S(12) = 0.038 K^2 (saturating at ~12 h). White noise would be flat at 2 sigma^2; the steep rise says hour-to-hour differences are dominated by real, non-advective change (an advective change at
  0.1 m/s is only 0.004 K per hour), with the white-noise sigma_T somewhere between 0 and ~0.03 K (S proportional to L^2 plus a nugget gives 0.032 K). A longer difference lag would therefore not cut the floor much.
- **Consequences / options** (none implemented beyond the hook): (a) the 4-month, 10 deg x 10 deg download (floor about halves for 4 months if the residual stays correlated over days); (b) take the barotropic tide
  from FES2022 (the user has `fes22_tides` / pyTMD; the FES files on this machine are elevation constituents only and pyTMD is not installed, so it was not tried) as an EXTERNAL product written to the cube file
  as `u_tide`, `v_tide` (`regional.ArrayTide`, `--tide-from-cube`, `BSST_TIDE_SOURCE=cube`; tested), which makes the decomposition a clean barotropic (FES) + internal tide / eddies (GP wave kernel) split with
  no SNR problem; (c) fit the regression on the latent-T products of the tile GP (denoised, gap-filled) instead of raw differences, i.e. backfitting (not built).

**Real noise level: still about 0.01 K (corrects an earlier reading).** A 3-iteration pilot stage-1 fit on a 20 x 20 px x 20 h tile had moved the noise to 0.062 K; that was an artefact of the short, unconverged
fit. The 48 h fit (`experiments/35_real_tile_stage1_48h.py`, 28 x 28 px x 48 h, days 3-5, 99% observed, loose prior on the noise, 360 evaluations, **not converged**, `data/exp35_real_stage1_h72_j24_i24.json`) gives
**noise sd 0.0074 K**, in line with exp 27 (0.008-0.013 K), with the hour-scale variability carried by the fast component (sd 0.048 K, 5.6 km, 1.7 h) and a smooth large-scale eddy component (sd 6.4 K, 236 km,
tau 12 h = its lower bound, so the eddy / fast split is bound-driven as in exps 15 / 27 / 28). So the white-noise level is small and most of the hour-to-hour change is genuine SST variability.

**That variability is what dominates stage 2** (`experiments/36_real_stage2_noise.py`, the same tile): the latent dT/dt has rms 4.1e-5 K/s, rms |grad T| is 1.5e-5 K/m, so an advective signal u.grad T for a 0.1 m/s flow is only
1.1e-6 K/s; the ML stage-2 noise at the placeholder S prior is 709 scaled units = 2.7e-5 K/s, **25 times the signal of a 0.1 m/s flow** (the placeholder S prior, sigma_S = 0.3 scaled = 3e-7 K/s, is far too small
for real data, so the noise absorbs the real non-advective dT/dt). With a stage-2 data set of ~27k rows the velocity is then constrained mainly through its long space-time coherence (days, tens of km) against the
short-lived S (1-2 h), i.e. weakly: expect posterior variances close to the prior on small scales until this is tested against independent velocities (drifters). A 3-iteration stage-2 fit with the synthetic noise (2.17)
ran gamma to its bound (2.0), sigma_S to its bound (10 scaled, itself too low for real data) and tau0_S to 2 h; the `hyper` stage of `run_test.py` therefore fits the stage-2 noise by ML per tile first
(`--noise-mode mle`), and the bounds of sigma_S and the diurnal scale were widened. `hpc/hyper_real_start.json` now holds the exp-35 stage-1 values and the exp-36 noise (709) with the placeholder velocity parameters.
(The synthetic exps 22-24 at 0.03 / 0.1 K are closer to this regime than the 0.01 K of exps 31-33.)

### Where the dT/dt variance sits, and structured S in stage 2 (exp 37, 2026-11-06; real tile, no tide removed)

**Where the non-advective dT/dt is** (three 72 h windows, NaN-aware, no gap filling): the spatially uniform per-hour mean carries 74%, 35% and 32% of the dT/dt variance (periods 9-36 h); of the non-uniform
remainder about half is at scales below 18 km; the remainder's variance rises 2.5-5x with |grad T| but a gradient-independent floor of 1.2-1.6e-5 K/s dominates. **Correction to the exp 36 comparison above:**
the "25x" used a 0.1 m/s flow; the fitted velocity has rms 0.2-0.24 m/s, so the advective term is about rms|grad T| x rms u = 1.5e-5 x 0.25 = 3.8e-6 K/s and the equation error (2e-5 K/s) is about 5x it, not 25x.

**Test** (`experiments/37_real_structured_S.py`; tile = hours 72-120, rows / cols 24-52, 28 x 28 px x 48 h, 27648 stage-2 rows, stage-1 parameters from exp 35, ALL stage-2 parameters including the noise free,
maxiter 20, ~1 h per fit, 3 fits in parallel). Variants: `whiteS` = the old `HelmholtzMaternPrior` (Matern S + fixed-shape diurnal low-rank S) started at the exp-36 values (sigma_S 0.3, noise 709); `fixedcycle` and
`daily` = `StructuredSPrior` started at sigma_S 15, l_S 8 km, tau_S 2 h, noise 4, with the diurnal S either one fixed cycle (4 harmonics) or day-by-day amplitudes (`diurnal_harmonics_daily`, linear-hat modulated).

| variant | log-lik | noise (K/s) | sigma_S (K/s) | l_S (km) | tau_S (h) | lr:diurnal | gamma (m/s) | post/prior var u | rms u |
|---|---|---|---|---|---|---|---|---|---|
| whiteS | -75613.5 | 1.0e-6 (bound) | 1.95e-5 | 16.3 | 1.0 | 105 | 0.290 | 0.46 | 0.199 |
| fixedcycle | -75609.2 | 1.0e-6 (bound) | 1.99e-5 | 16.3 | 1.0 | 97 | 0.294 | 0.46 | 0.201 |
| daily | **-75435.4** | 1.0e-6 (bound) | 1.73e-5 | 15.1 | 0.9 | 91 | 0.283 | 0.47 | 0.196 |

Velocity hyperparameters (similar in all three): f_delta 0.35-0.40, l_psi 70-83 km, l_phi 8 km, tau_psi 2.6-2.8 h, tau_phi 1.3 h. None of the three fits converged (maxiter reached, `success=False`, 300-312 evaluations).

- **The equation error is S, not white noise.** In every variant the iid noise went to its lower bound (1 scaled unit = 1e-6 K/s) from a start of 4 (structured) or 709 (whiteS); a Matern S of sd 1.7-2.0e-5 K/s, 15-16 km and
  about 1 h takes over, which is the same size as the 2.7e-5 K/s noise of exp 36 but spatially correlated. (The ~1 h time scale is the sampling interval, so in time it is effectively white.)
- **whiteS and fixedcycle are the same model** (the old prior already has a fixed-shape diurnal term) started 100x apart in noise; they end within 4 log-lik units and 5% in every parameter. That is a reproducibility
  check on the optimiser despite the failed convergence flag, not a model comparison.
- **Day-by-day diurnal amplitudes are preferred: +174 log-lik** over the fixed cycle on identical rows and Vecchia settings (the marginal likelihood already pays for the extra flexibility), consistent with the user's
  observation that the day-to-day heat flux is inconsistent.
- **The velocity does not change.** Posterior-mean u, v correlate 1.00 between the variants; rms differences 0.008 m/s (whiteS vs fixedcycle) and 0.013-0.016 m/s (either vs daily) against an rms of 0.2-0.24 m/s.
  The posterior / prior variance of u is 0.46 in all three: the data remove only about half the prior variance of the velocity on this tile.
- **Reading.** The way S is structured (among these three) is not what limits the velocity on this tile; the information is (strong fronts only). This is a model-adequacy test: there is no velocity truth, so
  "insensitive to S" is not "correct". Caveats: one 48 h window, no tide removed, unconverged fits, noise at its bound. The short velocity time scales (tau_psi 2.6-2.8 h, tau_phi 1.3 h) are EXPECTED
  (user, from the browseGP in-situ work): without a 'fast' red-noise kernel in the velocity prior (browseGP: ~7 cm/s, 2-3 km, ~1.1-1.4 h) the slow rotational / divergent time scales shrink to absorb that
  fast variance (browseGP's rotational tau went 0.4 d -> 1.3-4 d once K1 / O1 and the fast block were added). They are a symptom of the missing block, not physics; the stage-2 prior has no fast block yet.
- **Not done:** the day-by-day diurnal term in the tide regression (it is the natural nuisance there, since S2 and K1 share frequencies with solar heating); a longer window; converged fits.

### Day-by-day diurnal nuisance in the tide regression, and the BB23 mooring tides (exps 38-39, 2026-11-06; real test month)

**Code.** `tides.fit_tides(..., nuisance_t_spacing=86400, nuisance_t_smooth=0)`: each diurnal harmonic's coefficient field is a piecewise-linear function of time with one knot per day (partition of unity, so
the old fixed cycle is a special case); `TideConfig.nuisance_t_spacing / nuisance_t_smooth`, `hpc/run_tide.py --nuisance-t-days`. Parameters: 4960 for the 29-day test month (dense solve; for a 4-month
10 deg box with the same spatial basis it would be ~17k: needs a sparse or Schur-complement solve, or fewer harmonics). `tests/test_tides.py::test_day_by_day_nuisance_handles_inconsistent_heating` (synthetic, heating
amplitude 0.2-1.8x and onset +-1.5 h from day to day): tide error 0.0027 -> 0.0001 m/s; with consistent heating the extra freedom costs nothing. On noiseless algebraic rows the inconsistent heating hurts the
fixed-cycle fit very little (0.5% of the tide rms), so this synthetic test says the option works, not that it matters.

**Real month** (`experiments/38_real_tide_daily_nuisance.py`; M2 S2 K1 O1 against the four fictitious control frequencies of exp 34; min depth 100 m, 80 km splines, 20 km pooled blocks, time stride 2):

| variant | residual rms (K/s) | tidal rms amp (m/s) | control rms amp | ratio | split-half coherence tidal (M2 S2 K1 O1) / control mean |
|---|---|---|---|---|---|
| base (fixed cycle, 4 harmonics) | 1.95e-5 | 0.400 | 0.341 | 1.17 | 0.31 0.14 0.10 0.15 / 0.18 |
| **daily** (4 harmonics) | **1.33e-5** | 0.230 | **0.197** | 1.17 | 0.12 0.13 0.27 0.16 / 0.12 |
| daily, 2 harmonics | 1.52e-5 | 0.258 | 0.225 | 1.15 | |
| daily + per-hour mean removed | 1.33e-5 | 0.229 | 0.197 | 1.17 | same as daily |
| daily + temporal smoothing | 1.41e-5 | 0.250 | 0.215 | 1.16 | |

- **The day-by-day nuisance explains 32% less residual (rms) and cuts the control-frequency floor from 0.34 to 0.20 m/s (-42%), but the tidal / control ratio does not move (1.15-1.17 in every variant) and the split-half
  coherence stays at the control level** (tidal 0.12-0.27, control 0.05-0.17). So the spurious amplitude falls, the signal is still not detected. The per-hour mean adds nothing once the nuisance is day-by-day
  (it already contains the uniform part). The M2 split-half coherence fell from 0.31 to 0.12, so the earlier 0.31 was probably partly a heating pattern shared by the two halves, not tide.
- **Mooring constants (exp 39; BB23 `_long` file, 6 mADCP records of 35-181 days, hourly means, M2 S2 N2 K2 K1 O1 P1 Q1 + a 7-day spline for the sub-tidal flow; the `_corrected` file only differs in Ekman /
  Stokes removal that does not touch the tidal band for moorings):** amplitudes per component, over the sites, **M2 0.13-0.37 m/s, S2 0.09-0.23, K1 0.02-0.06, O1 0.01-0.05**. So the diurnal tides are
  physically negligible here (< 0.05 m/s) and the SST fit's 0.25 m/s for K1 and O1 (and its S2 0.20-0.41) are noise; M2 and S2 (rms over both components ~ 0.2-0.4 m/s) are of the size of the daily-nuisance
  floor (0.20), i.e. at best marginally detectable. Site 0-4 records are 35-108 days: M2 / S2 differ across the 6 sites by 50-90% (the semidiurnal field is not uniform over 50 km, internal tide).
- **The moorings are mostly NOT inside the test box.** BB23 sites: lon 122.84-123.30, lat -14.24 to -13.94; the test cube is lat -14 to -11, so five of six sit 0.05-0.24 deg south of it (only site 5, lat -13.94,
  depth 254 m, is inside, 7 km from the edge). The comparison of SST and mooring constants therefore rests on one site at the edge of the spline domain: not a validation (SST/mooring amplitude ratios 1.2-6 for M2 and
  S2 (daily variant), 5-45 for K1 / O1; phases 20-170 deg off, all consistent with noise). The Himawari download (lon 114-125, lat -21 to -10) does cover the BB23 area: a cube centred on it would give a proper test.
  Time mismatch: SST is May 2026, moorings 2023, so only phase-locked constants (not instantaneous velocities) can be compared; nodal factors (< 1% for M2, ~10 deg in K1 / O1 phase) are ignored.
- **Consequences / options (not implemented):** (a) K1 and O1 should be dropped from the SST regression or given an amplitude prior of ~0.04 m/s from the moorings, otherwise the pipeline subtracts ~0.25 m/s of pure
  noise as "tide" and adds it back to the velocity; (b) an informative amplitude ridge for M2 / S2 (sd ~ 0.25 / 0.17 m/s from BB23) would shrink the estimate towards the in-situ level instead of the floor;
  (c) the in-situ constants themselves (browseGP's tide block) as the tide in this region, with the SST regression only for departures; (d) re-cut the test cube around the moorings (lon ~122-124.5, lat ~-15.5 to
  -12.5).

#### Follow-up (2026-11-06, evening): K1 / O1 dropped, test cube re-cut around the moorings, 80 vs 150 km splines

- **Decisions (user):** K1 and O1 removed for now (`TideConfig.constituents` default is now `("M2", "S2")`, `run_tide.py --constituents M2,S2`; the low-level `fit_tides` default is unchanged); no tight amplitude prior
  around the mooring values (the tide may vary strongly in space); the test cube is re-cut, and the user will download Himawari for **2023** tonight so the BB23 observations overlap in time.
- **Spline spacing was NOT changed to 150 km before this** (it was 80 km in `TideConfig`, the CLI and exps 34-38; exp 34 only tried 300 km, which cut the floor by 27% and biased the amplitude; the 'support radius'
  is a different quantity: 20 km = one block in the pipeline, `1.5 x spacing` only if `fit_tides` is called directly). `hpc/run_test.py` now passes `--spacing 150e3` (and `--nuisance-t-days 1`) to the tide stage after the
  test below; the library default is still 80 km.
- **New cube** `/mnt/c/Users/00099894/Large_Data/Other/browseSST/test_bb23_may2026.nc` (`hpc/make_test_dataset.py --lon 121.6 124.6 --lat -15.4 -12.4 --stride 2 --reduce stride --mask-depth 100`): 75 x 75 x 670, 2026-05-01..28,
  SUNTANS depth on 99.6% of the cells, 70% of the cells deeper than 100 m, SST valid 0.74 (cloud gap May 16-21 as before), all six BB23 moorings inside and 199-295 m deep (kept by the mask). `hpc/run_test.py`,
  `submit_test.sh` and the README now default to it (`run_test_bb23`). For the 2023 download: same command with `--nc-dir <2023 files>`; the in-situ windows are moorings Feb 1 - Jul 31, drifters Apr 21 - Jul 31, ship
  Apr 16 - Jun 3 2023 (the `_corrected` file covers Apr 17 - May 28), so Apr 21 - May 28 has every platform.

| cube / variant (M2 S2, day-by-day nuisance) | tidal rms amp (m/s) | control rms amp | ratio | split-half coherence M2 / S2 vs control | note |
|---|---|---|---|---|---|
| old box, 80 km | 0.227 | 0.197 | 1.16 | 0.06 / 0.15 vs 0.05-0.17 | |
| old box, 150 km | 0.182 | 0.156 | 1.17 | 0.04 / 0.15 vs 0.10-0.42 | floor -21%, no gain in ratio |
| **BB23 box, 80 km** | **0.276** (M2 0.313, S2 0.239) | 0.179 | **1.54** | **0.39** / 0.08 vs 0.04-0.15 (phase diff 53 deg vs ~90 for controls) | |
| **BB23 box, 150 km** | **0.246** (M2 0.292, S2 0.200) | 0.148 | **1.67** | **0.55** / 0.10 vs 0.08-0.12 (43 deg) | floor -17% |

- **On the BB23 box M2 is above the floor for the first time** (quadrature excess sqrt(0.313^2 - 0.18^2) = 0.26 m/s at 80 km; split-half coherence 0.39-0.55 against 0.04-0.15 for the controls). **S2 is not**
  (coherence 0.08-0.10 = control level; the moorings have S2 0.09-0.23 m/s). 150 km splines do better than 80 km here (higher ratio, higher M2 coherence, lower floor) and are neutral on the old box.
- **Against the moorings** (exp 39 on the new cube; all six supported; in-situ M2 u/v 0.13-0.24 / 0.02-0.37, S2 0.09-0.15 / 0.02-0.23; moorings 2023, SST 2026, no nodal correction; sites 3 and 4 are the same position,
  so about five independent points; the moorings differ among themselves by 50-90% over ~50 km, so the barotropic fit at 150 km cannot match the baroclinic part): **the v component of M2 and S2 agrees in phase (M2 -16 deg, S2 +26 deg
  mean difference at 150 km; 5 of 6 sites within ~35-55 deg) with amplitudes 1.4-1.5 x the in-situ mean; the u component has a consistent phase offset of about -70 to -110 deg at 5 of 6 sites (and a different ellipse
  sense), amplitude about right.** Relative error |z_sst - z_obs| / |z_obs| (u and v together): M2 0.86 at 150 km (1.16 at 80 km), S2 1.1 at 150 km; i.e. at best marginally better than predicting no tide for M2 and not
  better for S2. The gradient directions are NOT strongly anisotropic (rms Tx 6.9e-6, Ty 7.7e-6 K/m; eigenvalue ratio 0.79), so poor cross-gradient identifiability does not explain the u-phase offset; the cause is open
  (internal tide / depth of the ADCP, 2023 vs 2026, or the estimator).
- **User's correction (same evening):** the mooring-to-mooring differences are baroclinic. A plane fitted through all the moorings (M2, S2, O1, K1) reproduces the depth-averaged currents' harmonic fit almost perfectly
  at every mooring, so the barotropic target for the SST tide is that PLANE (a smooth field, which is what the 150 km splines represent), not the individual mooring constants used above; the site-to-site residual is the
  baroclinic tide that the semidiurnal oscillating divergent kernel (M2, fixed frequency) of browseGP models. The mooring comparison above should be redone against the plane (exp 39 uses the site mean as a proxy).
- **Implications:** M2 from the SST regression is usable as a rough barotropic estimate (v better than u); S2 is not usable on 28 days; a longer record (the 4-month download) and 150 km splines are the obvious
  next step; the 2023 data will compare the same constants and add direct velocity checks. Candidate for the pipeline default: M2 only (S2 subtraction adds noise), the user has not decided.

### 2023 cube (Himawari Apr 1 - Jun 17 2023): tide validated against the BB23 mooring plane; browseGP kernel plugged in (exps 38-40, 2026-11-07)

**Data.** The user's 2023 download (Apr 1 00:00 - Jun 17 07:00, 1856 hourly files, no gaps; same folder as 2026) cut with `hpc/make_test_dataset.py --lon 121.6 124.6 --lat -15.4 -12.4 --t0 2023-04-01T00:00:00 --t1 2023-06-17T07:00:00`
-> `/mnt/c/Users/00099894/Large_Data/Other/browseSST/test_bb23_2023.nc` (75 x 75 x 1856; SST valid 0.73; Apr 3-15 mostly cloud, good from Apr 16; Apr 21 - May 28 daily coverage 0.76-0.99). The in-situ overlap
(moorings Feb 1 - Jul 31, drifters Apr 21 - Jul 31, ship Apr 16 - Jun 3) is now exact.

**Solver.** `fit_tides(solver='schur')` (default for > 3000 parameters with the day-by-day nuisance): the nuisance block is banded in time once ordered time-major (banded Cholesky), only the tide block (200 coefficients
for M2 + S2 at 150 km) is solved densely; matches the dense solve to 1e-13 (`test_schur_solver_matches_the_dense_solution`); the full 77-day fit (11.6k parameters, 4 M rows pooled) takes 3 min and a few hundred MB
(the dense solve would need ~5 GB); `cov` is the tide block only.

**Tide results, 2023** (`experiments/38_real_tide_daily_nuisance.py`, tag `exp38bb23y_*`; M2 S2, day-by-day nuisance, 150 km splines, min depth 100 m):

| record | tidal / control rms amp | split-half coherence M2 / S2 (controls) | error vs the mooring plane M2 / S2 (u+v, / plane rms) |
|---|---|---|---|
| Apr 21 - May 28 (38 d) | 0.225 / 0.145 = 1.54 | 0.73 (25 deg) / 0.54 (44 deg) (controls 0.07-0.42, 65-106 deg) | 0.55 / 0.79 |
| Apr 1 - Jun 17 (77 d) | 0.201 / 0.129 = 1.56 | 0.56 (27) / 0.55 (45) (controls 0.12-0.46, 68-123 deg) | **0.53 / 0.59** |
| 38 d, 180 m depth crop | 0.193 / 0.150 = 1.29 | | 0.65 / 0.93 |

- Against the PLANE through the moorings (`experiments/39_bb23_mooring_tides.py moor|compare`; the plane = the user's barotropic estimate; the spread of the individual moorings about it, 25-45%, is the baroclinic part):
  77-day M2 amplitude ratio SST / plane 0.95 (u), 1.05 (v), phase difference -10 deg (u), +36 deg (v); S2 0.91 (u), 0.75 (v), +26 deg, -13 deg. **The u-phase offset of ~-90 deg seen when the 2026 SST was compared
  with the 2023 moorings has gone: it came from the year mismatch.** Roughly 72% (M2) and 65% (S2) of the plane's variance is explained at the moorings; unbiased amplitude.
- The 180 m crop is worse (fewer, shallower-tide-bearing blocks lost): keep 100 m. The regional tide stage is therefore OK for M2 and S2 on this region; the HPC defaults (M2 S2, 150 km, day-by-day nuisance, 100 m) stand.

**browseGP kernel as the stage-2 prior** (`kernels.browsegp_transport_covfunc`, `kernels.pack_browsegp`, `pipeline.BrowseGPPrior`): rotational Helmholtz (Matern-5/2 x Matern-3/2) + M2 divergent wave (Matern-5/2 space x
exp(-(tau/tau_d)^2) cos(w_M2 tau)) + M4 wave + fast block + inertial (circularly polarised) block, the browseGP tensors of `scripts/fit_multiscale_real.py` (`covfunc_inertial_fast_m4`), with z = a u + b v + c S + noise.
`BrowseGPPrior.from_results(mcmc_results.json)` takes the posterior MEDIANS of `results/mcmc_M4_v0.01` (22-day BB23 window, 1571 fit obs): rot 0.072 m/s, 54 km, 4.4 d; div(M2) 0.120 m/s, 33 km, 7.1 d; M4 0.025 m/s,
12 km; fast 0.068 m/s, 2.0 km, 1.3 h; inertial 0.024 m/s, 18 km, 2.7 d; amplitude gamma m/s with sigma_potential = gamma l sqrt(3/5). The velocity parameters are fixed, only S (Matern + day-by-day diurnal) and
the equation noise are fitted (`free = sigma_S, l_S, tau0_S, lr:diurnal, noise`; `slots` passed to `fit`). Tests (`tests/test_pipeline.py`): equals the existing transport covfunc for rot + a static divergent block,
equals browseGP's own `rotational_/divergent_velocity_tensor_*` for the full model including the polarised u-v cross term, symmetric PSD Gram matrix, `run_tile` / `fit_hyper` run with it.

**Exp 40: scoring the stage-2 velocity against the BB23 in-situ velocities** (`experiments/40_browsegp_prior_2023.py`; 28 x 28 px x 48 h tiles at the moorings; windows A = Apr 24-26 and B = May 4-6 fitted, plus
W20 W28 W36 W40 W42 (W44 W50 W54 pending) predicted with the parameters fitted in window A; stage 1 with the shared `hyper_real_start.json` values; tide = the 77-day M2 / S2 regional fit, subtracted from dT/dt and
added back; the posterior-mean EDDY velocity is predicted at the position and time of every in-situ observation; moorings are hourly means per site, drifters / ship ADCP as in the file; 1649 observations in
7 windows, results for the `_long` and `_corrected` files agree to +-0.05):

| prior (stage 2) | log-lik (A / B) | rms eddy (m/s) | skill vs tide-only, all platforms, u / v | correlation of (obs - tide) with the eddy, u / v | slope of obs - tide on the eddy, u / v |
|---|---|---|---|---|---|
| browseGP kernel, velocity FIXED at the `mcmc_M4_v0.01` medians, S + noise fitted | -77279 / -83423 | 0.09 / 0.13 | -0.21 / -0.25 | 0.07 / 0.23 | 0.15 / 0.28 |
| our Helmholtz-Matern, all 11 parameters free (exp 37 'daily' structure) | **-76570 / -82624** | 0.15 / 0.21 | -0.77 / -1.07 | 0.06 / 0.17 | 0.07 / 0.12 |

- **The tide stage works against independent velocities:** subtracting the SST-fitted M2 + S2 tide reduces the in-situ rms velocity from 0.218 to 0.160 m/s (u, 46% of the variance) and from 0.211 to 0.173 m/s (v, 33%), pooled
  over moorings, drifters and ship ADCP.
- **The stage-2 EDDY velocity does not beat tide-only:** skill is negative for both priors in every platform and file; only v has a (weak) positive correlation (0.15-0.25; u 0.0-0.1), so there is a small signal but the
  posterior mean is 4-8 times too large for it (regression slope 0.1-0.3: scaled optimally the skill would be 0.7% (u) and 4% (v) of the variance). The 48 h-mean flow at the moorings (33 site-windows) is not predicted either
  (correlation 0.1 / -0.1, rms 0.09 / 0.08 m/s observed vs 0.03 predicted). No u / v swap or sign problem (cross-correlations 0.05-0.14 vs diagonal 0.07 / 0.23).
- **The browseGP kernel is less bad than ours, although it explains dT/dt worse** (log-lik 700-800 lower with 6 fewer free parameters): ours buys its likelihood with 1-2 h velocity time scales (tau_psi 1.7-1.9 h,
  tau_phi 1 h, 77-81 km rotational scale, gamma 0.3) that soak up the fast non-advective dT/dt (the missing fast red-noise block), giving large, wrong velocities; browseGP's fixed 4-7 day scales plus a fast block with
  its own amplitude give smaller velocities. The fitted S is the same in both (sd 1.9-2.4e-5 K/s, 17 km, 0.5-0.6 h; noise 1.8-5 scaled units).
- **Caveats:** in-situ velocities contain the baroclinic tide, inertial motion and depth-dependent flow (the moorings are not at the surface) that SST cannot be expected to see, so every estimator pays that part;
  48 h windows leave few independent samples; S and the velocity are only weakly separable here (S has the same 17 km scale as the velocity; noise sits at its lower bound). Not done: the same test with the S
  parameters fitted per window for the browseGP prior only, a block-by-block decomposition of the predicted velocity (rot / M2 wave / fast / inertial), band-split skill (M2 band vs sub-inertial), longer windows (the
  rotational and divergent blocks have 4-7 day scales, a 48 h tile cannot constrain them), shrinking S's spatial scale, an in-situ-informed prior for S.

**Exp 40 update (all ten windows, 2333 observations, `_long` file):** browseGP kernel, skill vs tide-only u / v -0.16 / -0.11, correlation 0.14 / 0.27, 48 h-mean flow at the moorings (48 site-windows) correlation 0.13 / 0.00;
our free Helmholtz-Matern -0.75 / -0.79, 0.10 / 0.21: the 7-window picture holds (no skill over tide-only; weak positive v correlation).

**User comments on exp 40 (2026-11-07):** (1) not sure that S is small-scale around fronts; that was only seen as a possible explanation, the short-scale part may in fact be frontal CURRENTS (so a short S may be absorbing
real frontal advection that the velocity prior, with its 33-54 km scales, cannot represent); (2) surprised that the baroclinic tide is not seen: it should be strong and clear (to be tested directly: exp 41 band analysis of the
semidiurnal part of obs - tide against the predicted M2-wave block); (3) M4 and inertial blocks dropped (prior = rotational + M2 wave + fast); (4) go ahead with the three-step plan: block / band decomposition, longer
(96 h) windows, longer / smoother S (`experiments/41_blocks_bands_longS.py`, variants bg3, bg3_longS, bg3_noS; `BrowseGPPrior.cov_params_blocks` gives the block-by-block posterior-mean predictions, tested).

**Exp 41 (2026-11-07/08): block / band decomposition, S variants, 96 h windows, M2-band significance** (`experiments/41_blocks_bands_longS.py`; prior = browseGP rotational + M2 wave + fast, velocity FIXED, M4 and inertial dropped
(user); fitted S / diurnal / noise per window A, predict-only with window A's values on the other tiles; 10 windows of 48 h with 2333 observations and 3 windows of 96 h (22 x 22 px) with 1530).
- **S variants (window A, log-lik):** short S (17 km, 38 min, free) -77328; S forced smooth and slow (60 km, 6 h) -85075 (sigma_S at its bound 100, noise 14.7); NO Matern S (white noise + diurnal only) -121297 (noise 331,
  diurnal scale at its bound). The data need a short-scale, hour-scale non-advective term (or a short-scale velocity) far more than white noise; a longer S is strongly rejected. Dropping M4 + inertial costs only 49 log-lik.
  The user's point that this short-scale part may be frontal CURRENTS rather than S is untested (it needs the fast / short-scale velocity block freed, see below).
- **Block attribution (pooled hourly, skill vs tide-only, u / v):** total -0.17 / -0.14, correlation 0.14 / 0.26; by block: rotational rms 0.05 m/s, correlation 0.02 / -0.02; M2 wave rms 0.08 / 0.11, correlation 0.15 / 0.29;
  fast 0.007 / 0.010 (nothing: 2 km is below the 4 km grid). The M2 divergent wave carries all the predicted velocity and all of the correlation.
- **The baroclinic tide IS seen on average.** At the moorings the semidiurnal (M2-frequency) amplitude of obs - tide is 0.12 (u) / 0.20 (v) m/s, the largest part of obs - tide (window mean 0.09 / 0.085, rest 0.075); the predicted
  M2-band amplitude (M2-wave block) is 0.075-0.10 m/s, complex regression slope obs on prediction |b| = 0.78 (bootstrap-over-windows 95% CI 0.58-1.09) at -12 deg (`_corrected`: 0.82 at -17 deg), i.e. about the right
  amplitude and phase. BUT its coherence (0.43, 48 site-windows) equals the cross-window permutation null (mean 0.43, 95th percentile 0.51, p = 0.48): the in-situ M2 phase (epoch-referenced) is strongly locked from window
  to window (phase concentration R = 0.63 vs 0.29 for random) and so is the predicted one (R = 0.56), hence any window pairing gives the same coherence. So: the stable (phase-locked) part of the baroclinic M2 is recovered
  with about unit amplitude; there is NO evidence yet that the estimate tracks window-to-window changes of it. The rotational block contributes nothing at M2 (coherence 0.19 < null 0.26). 96 h windows: |b| 0.93 at -19 deg,
  coherence 0.41 (the permutation test is degenerate with only 3 windows).
- **Longer windows (predict-only, S from window A) do not help:** matched on the same observations, 96 h vs 48 h skill (u / v) -0.18 / -0.20 vs -0.21 / -0.17 (A), -0.26 / -0.60 vs -0.32 / -0.62 (B), -0.59 / -0.10 vs
  -0.64 / -0.12 (W42 + W44); correlations equal to +-0.03. (Not refitted at 96 h; the rotational block has a 4.4 d scale.)
- **Reading:** on average the stage-2 velocity contains a correctly scaled baroclinic M2 tide; the negative hourly skill comes from the rest (the rotational / M2 variability that is not phase-locked, plus noise): the posterior
  mean is 3-5x too large for its correlation (slope 0.25 / 0.36). Next candidates (not run; user wants to free the parameters on the HPC tile jobs): free gamma / length / time of the rotational and M2 blocks (needs slots for
  them in `BrowseGPPrior` and a `--prior browsegp` option in `hpc/`), free the fast / short-scale velocity block to test the frontal-current reading against a short S.

## HPC: browseGP prior with the velocity parameters fitted on the tiles (2026-11-08)

User: "We can free the params on the HPC tile jobs." Built (not run on the HPC; tested locally):
- `kernels.BrowseSlots` / `BG_SLOTS`: free names `gamma_{rot,div,m4,f,in}` (velocity amplitude in m/s; potential sigma = gamma ell sqrt(3/5) is computed AFTER the lengths, so no sigma-ell ridge),
  `ell_*`, `tau_*`, `sigma_S`, `l_S`, `tau0_S`. A block that is switched off stays off whatever the dict holds. `fit.params_to_model_args` calls `slots.post` if the slot map has one.
- `pipeline.BrowseGPPrior(..., free_vel=True | ('rot','div',...), prior_sd_log=...)`: `free` = S + diurnal + noise + the (gamma, ell, tau) of those blocks, bounds in `VEL_BOUNDS`, `complete(h2)` adds the start values
  (browseGP's medians) to `Hyper.stage2`, `cov_params(h2)` takes fitted values from `h2` over `vel`, so the fitted values travel in `hyper.json` to the tile jobs. `fit_hyper` completes h2, uses prior sd 1 (log) about the start
  values (`prior_sd_log` overrides) and records `prior` / `free2` in the meta. `pool_hypers` no longer needs `f_delta`.
- `hpc/`: `--prior browsegp --blocks rot,div,fast --free-vel rot,div,fast --vel-prior-sd 1 --vel-json` on `fit_hyper.py`, `--prior/--blocks` on `run_tile.py` (`_util.add_prior_args/make_prior`); `--fit-hours N` on `fit_hyper.py`
  (central N hours of the tile); `run_test.py` env `BSST_PRIOR=browsegp`, `BSST_BLOCKS`, `BSST_FREE_VEL`, `BSST_FIT_HOURS`; start files `hyper_browsegp_start.json` (stage 1 of `hyper_real_start.json`, S / noise / diurnal from the
  exp-41 window-A fit, velocity = browseGP) and `browsegp_vel.json` (the medians of `results/mcmc_M4_v0.01`, copied so the HPC does not need the browseGP results). README section added. Tests: 2 new in `test_pipeline.py`
  (slot mapping + off-blocks stay off; free-velocity prior, `fit_hyper`, pooling, `run_tile`), all pipeline / tiling / fit / kernel tests pass.
- **Local pilot (2023 cube, tile t11_02_02 = Apr 23-24, 48 h core, 6 h/4 px buffers, 150 km-spline tide, 4 threads, niced; `BSST_CORE=48,20,20 BSST_BUFFER=6,4,4 BSST_FIT_HOURS=36`):** hyper -> pool -> tiles, all stages run.
  Stage 1: 60 evals, 292 s. Stage 2 (14 free parameters, 20.7k rows): 75 evals, 455 s = 6 s per evaluation, 3 iterations (NOT converged). Tile run, 47k points: 360 s (stage-2 predict 176 s). After 3 iterations the velocity
  parameters moved only modestly from browseGP's (gamma_rot 0.072 -> 0.056, ell_rot 54 -> 60 km, gamma_div 0.120 -> 0.092, ell_div 33 -> 37 km, gamma_f 0.068 -> 0.053, ell_f 2.0 -> 2.4 km, tau_f 4.6 -> 5.7 ks; tau_rot,
  tau_div unchanged) and S (sigma 12.8, l 19 km, tau 3.6 ks); **the noise went to its lower bound 1.0** (bound of `StructuredSPrior`; the exp-41 fits had 3.4 / 5.0). That is a pilot, not a result: it only shows the chain works
  and which way the gradient points. The real test of "is there eddy velocity in SST here" is whether the 25+ iteration fits on several tiles keep gamma_rot / gamma_div well above their lower bound (1e-3) and agree between
  tiles (`pool_hyper.py` prints the max / min ratio), and then whether the fitted-parameter eddy velocity has skill against drifters / ADCP (exp 40 / 41 scoring, unchanged).
- Cost on the HPC: a likelihood evaluation scales with the number of rows, so a fit on a full 192 h x 36 x 36 buffered tile is ~10x the pilot (estimate from the row count, not timed: ~1 min per evaluation, ~15 evaluations per iteration): use `--fit-hours 48-96`.

## Status at pause (2026-11-03)

Built and tested (tests/ pass): `spacetime/{kernels,synthetic,inference,model,fit,fastop}.py`.
Experiments 01-05 in `experiments/`. Nothing committed to git. Env: conda `drift`.
Run any script with `python -I` from outside the repo dir (e.g. `cd /tmp`), the package is installed editable.

## Open questions / next

- DONE: diurnal low-rank term (datide `VecchiaMultiscaleGP`), hyperparameter fitting with locked neighbours.
- Resolve the ST hyperparameter drift (diagnostic above; also try a 1-week window, tighter priors on temporal scales).
- Noise model for the finite-difference residual (smooth in time, ~11% of dTdt rms at 4 km) and for noisy gradients (phase B).
- Scaling check at N ~ 130k (28x28 px x 168 h): generator cost (~57 s at 14x14x48 grows fast with the padded grid),
  `maximin_order` is O(N^2), `predict_local` at m=200+ per field.
- Compare against the old per-snapshot code; tiling with overlap + partition-of-unity blending; HPC layer.
- If m needed grows with domain size, consider Vecchia on the latent (u, v, S) prior with sparse H instead of
  on the data covariance (better screening when noise is large).

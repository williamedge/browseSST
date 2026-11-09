"""
Experiment 18: multiple imputation with GRID-RESOLVED gradients.

Experiment 16 imputed the pointwise analytic derivatives of the T-GP; the draws were 2.6x over-dispersed for the spatial gradients (the
true T is a rough field, 1-D spectral slope about -2 to -2.8, and a Matern-5/2 mixture puts gradient variance at scales the 4 km grid
cannot resolve), and the pooled estimates were worse than the plug-in. Here the latent field is imputed through ('T', 'Tt') and the spatial
gradients are the 4th-order centred differences of the imputed T on the pixel grid; Tt stays analytic.

Stage-2 noise, three ways (the first two are NOT oracles):
  'eq'     equation-error variance induced by the derivative uncertainty:  var(dTt) + E[u^2] var(dTx) + E[v^2] var(dTy)
           + 2 E[uv] cov(dTx, dTy)  in scaled units, from the draw spread about the mean and the PRIOR velocity variances;
  'mle'    fitted by marginal likelihood on the plug-in products (experiment 16 found this too small);
  'oracle' empirical residual against the true u, v, S (diagnostic).
Estimators: plug-in (posterior-mean products); plug-in + propagated variance (W_plug + (1 + 1/K) B, B = between-draw variance of the
draw estimates; the draw estimates are used for the spread only); Rubin-pooled draw estimates.
Run: python experiments/18_resolved_gradient_mi.py <sigma_T> [K] [nnum_prior] [nnum_pred] [modes]
"""
import importlib.util, json, pathlib, sys, time, warnings
import numpy as np

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / (name + ".py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


e16 = _load("16_tgp_multiple_imputation")
e15, e13, G, K, ZS = e16.e15, e16.e13, e16.G, e16.K, e16.ZS
from inversion_sst_gp.spacetime.impute import DerivativeImputer, grid_gradients


def products(d, shape, dx, order=4):
    Tx, Ty = grid_gradients(d['T'], shape, dx, order=order)
    return Tx, Ty, d['Tt']


def prior_velocity_moments(p_vel):
    """E[u^2], E[v^2], E[uv] of the stage-2 prior at zero lag (physical units)."""
    r = lambda a, b: np.array([0.0, 0.0, 0.0, a, b, 0.0])
    cov = lambda r1, r2: float(K.transport_obs_covfunc(r1, r2, p_vel))
    return cov(r(1, 0), r(1, 0)), cov(r(0, 1), r(0, 1)), cov(r(1, 0), r(0, 1))


def equation_noise(mean_prod, draw_prods, moments, valid):
    """Mean over draws and valid points of (dTt + u dTx + v dTy)^2 with u, v averaged out against their prior moments."""
    Eu2, Ev2, Euv = moments
    v = valid
    dtt = np.array([(d[2] - mean_prod[2])[v] for d in draw_prods])
    dtx = np.array([(d[0] - mean_prod[0])[v] for d in draw_prods])
    dty = np.array([(d[1] - mean_prod[1])[v] for d in draw_prods])
    var = np.mean(dtt ** 2) + Eu2 * np.mean(dtx ** 2) + Ev2 * np.mean(dty ** 2) + 2 * Euv * np.mean(dtx * dty)
    return float(var) / ZS ** 2          # the stage-2 data are scaled by 1 / ZS


def propagated(plug, runs):
    """Plug-in estimate with total variance W_plug + (1 + 1/K) B."""
    out = {}
    Kn = len(runs)
    for f in 'uvS':
        ms = np.array([r[f][0] for r in runs])
        B = ms.var(0, ddof=1)
        out[f] = (plug[f][0], plug[f][1] + (1 + 1.0 / Kn) * B)
        out[f + '_ratio'] = float(np.mean(B) / np.mean(plug[f][1]))
    return out


def main(sigma_T, Kdraws=10, nnum_prior=100, nnum_pred=200, modes=('eq', 'oracle')):
    cfg, ds0, ds, p_vel, flat, ref4, common3 = e15.setup(sigma_T)
    common = common3.ravel()
    dx = float(ds0.attrs['dx'])
    shape = common3.shape
    fitfile = ROOT / "data" / f"exp15_fit_sigma{sigma_T}.json"
    r = json.loads(fitfile.read_text())
    print(f"\n=========== resolved-gradient multiple imputation, sigma_T = {sigma_T} K, K = {Kdraws} ===========", flush=True)
    print(f"stage-1 (exp15): eddy {r['sigma1']:.3f} K, {r['l1'] / 1e3:.1f} km, {r['tau1'] / 3600:.1f} h | fast {r['sigma2']:.3f} K, "
          f"{r['l2'] / 1e3:.1f} km, {r['tau2'] / 3600:.1f} h | noise sd {np.sqrt(r['noise']):.4f}", flush=True)
    pts = G.build_T_points(flat['x'], flat['y'], flat['t'])
    p0 = G.pack_params_T(r['sigma1'], r['l1'], r['tau1'], r['sigma2'], r['l2'], r['tau2'], nu_t=2.5)
    gp = G.TGP(pts, ds['T_obs'].values.ravel(), p0, r['noise'], lowrank=e13.T_terms(r['lr:mean'], r['lr:diurnal']),
               nnum=100, Lx=20e3, Lt=24 * 3600.0)
    imp = DerivativeImputer(gp, flat['x'], flat['y'], flat['t'], dx=dx, dt=3600.0, nnum_prior=nnum_prior, nnum_pred=nnum_pred,
                            seed=21, fields=('T', 'Tt'), targets=('T', 'Tt'))
    t0 = time.time()
    mean, draws = imp.posterior_samples(Kdraws)
    print(f"mean + {Kdraws} draws in {time.time() - t0:.0f}s", flush=True)
    mean_prod = products(mean, shape, dx)
    draw_prods = [products(d, shape, dx) for d in draws]
    valid = np.isfinite(mean_prod[0]) & np.isfinite(mean_prod[1])

    print("\ngradient calibration on the common interior (rms vs 4th-order FD of true T):")
    ref = (ref4['dTdx'].ravel(), ref4['dTdy'].ravel(), ref4['dTdt'].ravel())
    for j, name in enumerate(('Tx', 'Ty', 'Tt')):
        err = np.sqrt(np.mean((mean_prod[j] - ref[j])[common] ** 2))
        spread = np.mean([np.sqrt(np.mean((d[j] - mean_prod[j])[common] ** 2)) for d in draw_prods])
        single = np.mean([np.sqrt(np.mean((d[j] - ref[j])[common] ** 2)) for d in draw_prods])
        print(f"  {name}: error of mean {err:.2e} ({err / np.std(ref[j][common]):.2f} of std) | draw spread {spread:.2e} | "
              f"spread/error {spread / err:.2f} | single-draw error {single:.2e} (calibrated: ~{np.sqrt(2) * err:.2e})", flush=True)

    st = e16.Stage2(flat, common, p_vel, nnum=100, nnum_pred=nnum_pred, valid=valid)
    moments = prior_velocity_moments(p_vel)
    print(f"\nprior velocity moments E[u^2] {moments[0]:.4f}, E[v^2] {moments[1]:.4f}, E[uv] {moments[2]:.4f} (m/s)^2; "
          f"stage-2 points {valid.sum()} of {len(valid)}", flush=True)
    noises = {'eq': equation_noise(mean_prod, draw_prods, moments, valid),
              'oracle_mean': st.oracle_noise(*mean_prod),
              'oracle_draws': float(np.mean([st.oracle_noise(*d) for d in draw_prods]))}
    print(f"stage-2 noise variances (scaled): equation-error {noises['eq']:.2f} | oracle residual of mean products {noises['oracle_mean']:.2f} "
          f"| of the draws {noises['oracle_draws']:.2f}", flush=True)
    for mode in modes:
        print(f"\n -- stage-2 noise: {mode}", flush=True)
        t0 = time.time()
        if mode == 'mle':
            fixed = st.fit_noise(*mean_prod)
            print(f"  fitted noise variance {fixed:.2f}", flush=True)
            nz = lambda d: fixed
        elif mode == 'eq':
            nz = lambda d: noises['eq']
        else:
            nz = lambda d: st.oracle_noise(*d)
        plug = st.run(*mean_prod, nz(mean_prod))
        e16.score("plug-in (posterior-mean products)", plug, st.truth)
        runs = [st.run(*dp, nz(dp)) for dp in draw_prods]
        e16.score("one random draw (draw 1)", runs[0], st.truth)
        pooled = e16.rubin(runs)
        e16.score(f"Rubin-pooled over K = {len(runs)}", pooled, st.truth,
                  extra=f"  | between/within u {pooled['u_ratio']:.2f} v {pooled['v_ratio']:.2f} S {pooled['S_ratio']:.2f}")
        prop = propagated(plug, runs)
        e16.score("plug-in + propagated variance", prop, st.truth,
                  extra=f"  | B/W_plug u {prop['u_ratio']:.2f} v {prop['v_ratio']:.2f} S {prop['S_ratio']:.2f}")
        print(f"  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(float(a[0]), Kdraws=int(a[1]) if len(a) > 1 else 10, nnum_prior=int(a[2]) if len(a) > 2 else 100,
         nnum_pred=int(a[3]) if len(a) > 3 else 200, modes=tuple(a[4].split(',')) if len(a) > 4 else ('eq', 'oracle'))

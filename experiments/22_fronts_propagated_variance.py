"""
Experiment 22: the exp-18 recipe (latent T + Tt imputation, 4th-order resolved gradients, equation-error stage-2 noise, plug-in mean with the
between-draw variance added) on the FRONTS + FILAMENT field of experiment 21, using the 5/2 : 5/2 T-GP hyperparameters fitted there.

Reports gradient calibration (draw spread / error of the mean), then velocity skill and 95% coverage, all points and the steepest-gradient 20%, for the
plug-in, one draw, Rubin-pooled draws, and plug-in + propagated variance. True velocity hyperparameters, scored on the common interior.
Run: python experiments/22_fronts_propagated_variance.py <sigma_T> [K] [nnum_prior] [nnum_pred] [modes eq,oracle]
"""
import importlib.util, json, pathlib, sys, time, warnings
import numpy as np

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / (name + ".py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


e18 = _load("18_resolved_gradient_mi"); e21 = _load("21_fronts_filaments")
e16, e13, e15, G, ZS = e18.e16, e18.e13, e18.e15, e18.G, e18.ZS
from inversion_sst_gp.spacetime.impute import DerivativeImputer


def main(sigma_T, Kdraws=10, nnum_prior=100, nnum_pred=200, modes=('eq',)):
    cfg, ds0, ds, p_vel, flat, ref4, common3 = e21.setup(sigma_T)
    common = common3.ravel(); shape = common3.shape; dx = float(ds0.attrs['dx'])
    gmag = np.hypot(ref4['dTdx'], ref4['dTdy']).ravel()[common]
    hi = gmag >= np.percentile(gmag, 80)
    r = json.loads((ROOT / "data" / f"exp21_fit_sigma{sigma_T}_nu2.5_2.5.json").read_text())
    print(f"\n=========== fronts: propagated variance, sigma_T = {sigma_T} K, K = {Kdraws} ===========", flush=True)
    print(f"stage-1 (exp21 5/2:5/2): eddy {r['sigma1']:.3f} K, {r['l1'] / 1e3:.1f} km, {r['tau1'] / 3600:.1f} h | fast {r['sigma2']:.3f} K, {r['l2'] / 1e3:.2f} km, "
          f"{r['tau2'] / 3600:.1f} h | noise sd {np.sqrt(r['noise']):.4f}", flush=True)
    pts = G.build_T_points(flat['x'], flat['y'], flat['t'])
    p0 = G.pack_params_T_res(r['sigma1'], r['l1'], r['tau1'], r['sigma2'], r['l2'], r['tau2'], 2.5, 2.5, 2.5)
    gp = G.TGP(pts, ds['T_obs'].values.ravel(), p0, r['noise'], lowrank=e13.T_terms(r['lr:mean'], r['lr:diurnal']),
               nnum=100, Lx=20e3, Lt=24 * 3600.0, covfunc=G.T_covfunc_res)
    imp = DerivativeImputer(gp, flat['x'], flat['y'], flat['t'], dx=dx, dt=3600.0, nnum_prior=nnum_prior, nnum_pred=nnum_pred,
                            seed=31, fields=('T', 'Tt'), targets=('T', 'Tt'))
    t0 = time.time()
    mean, draws = imp.posterior_samples(Kdraws, verbose=False)
    print(f"mean + {Kdraws} draws in {time.time() - t0:.0f}s", flush=True)
    mean_prod = e18.products(mean, shape, dx)
    draw_prods = [e18.products(d, shape, dx) for d in draws]
    valid = np.isfinite(mean_prod[0]) & np.isfinite(mean_prod[1])

    print("\ngradient calibration on the common interior (rms vs 4th-order FD of true T; draw spread / error of the mean):", flush=True)
    ref = (ref4['dTdx'].ravel(), ref4['dTdy'].ravel(), ref4['dTdt'].ravel())
    for j, name in enumerate(('Tx', 'Ty', 'Tt')):
        for tag, sel in (('all', np.ones(common.sum(), bool)), ('steep', hi)):
            err = np.sqrt(np.mean((mean_prod[j] - ref[j])[common][sel] ** 2))
            spread = np.mean([np.sqrt(np.mean((d[j] - mean_prod[j])[common][sel] ** 2)) for d in draw_prods])
            print(f"  {name} {tag:<5s}: error of mean {err:.2e} ({err / np.std(ref[j][common][sel]):.2f} of std) | draw spread {spread:.2e} | spread/error {spread / err:.2f}", flush=True)

    st = e16.Stage2(flat, common, p_vel, nnum=100, nnum_pred=nnum_pred, valid=valid)
    moments = e18.prior_velocity_moments(p_vel)
    noises = {'eq': e18.equation_noise(mean_prod, draw_prods, moments, valid),
              'oracle_mean': st.oracle_noise(*mean_prod),
              'oracle_draws': float(np.mean([st.oracle_noise(*d) for d in draw_prods]))}
    print(f"\nstage-2 noise variances (scaled): equation-error {noises['eq']:.2f} | oracle residual of mean products {noises['oracle_mean']:.2f} "
          f"| of the draws {noises['oracle_draws']:.2f}", flush=True)
    for mode in modes:
        print(f"\n -- stage-2 noise: {mode}   (skill all / steep ; cover all / steep)", flush=True)
        t0 = time.time()
        nz = (lambda d: noises['eq']) if mode == 'eq' else (lambda d: st.oracle_noise(*d))
        plug = st.run(*mean_prod, nz(mean_prod))
        e21.score("plug-in (posterior-mean products)", plug, st.truth, hi)
        runs = [st.run(*dp, nz(dp)) for dp in draw_prods]
        e21.score("one random draw (draw 1)", runs[0], st.truth, hi)
        pooled = e16.rubin(runs)
        e21.score(f"Rubin-pooled over K = {len(runs)}", pooled, st.truth, hi, extra=f"  | B/W u {pooled['u_ratio']:.2f} v {pooled['v_ratio']:.2f}")
        prop = e18.propagated(plug, runs)
        e21.score("plug-in + propagated variance", prop, st.truth, hi, extra=f"  | B/W_plug u {prop['u_ratio']:.2f} v {prop['v_ratio']:.2f}")
        print(f"  ({time.time() - t0:.0f}s)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(float(a[0]), Kdraws=int(a[1]) if len(a) > 1 else 10, nnum_prior=int(a[2]) if len(a) > 2 else 100,
         nnum_pred=int(a[3]) if len(a) > 3 else 200, modes=tuple(a[4].split(',')) if len(a) > 4 else ('eq',))

"""
Experiment 20: does the latent-T route still win at LOW noise (sigma_T = 0.01 K, the original package's default)?
Experiments 13-19 used 0.03 and 0.1 K; at 0.01 K finite differences were fine (exp 12). Same synthetic truth re-observed at sigma_T; T-GP
(5/2 : 5/2, split bounds, warm start from the exp-15 0.03 K fit, noise started at 2 sigma_T^2) fitted on the noisy T; resolved gradients
(4th- and 2nd-order differences of the posterior-mean T, analytic Tt); against finite-difference products of the noisy T (FD2, FD4) and the
clean-gradient bounds, all with true velocity hyperparameters and oracle iid stage-2 noise, scored on the common interior.
Run: python experiments/20_low_noise_comparison.py [sigma_T ...]
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
from inversion_sst_gp.spacetime.fit import T_KERNEL_SLOTS, fit
from inversion_sst_gp.spacetime.impute import grid_gradients

H = 3600.0
FREE = ['sigma1', 'l1', 'tau1', 'sigma2', 'l2', 'tau2', 'noise', 'lr:mean', 'lr:diurnal']


def main(sigma_T, maxiter=60, nnum=100, nnum_pred=200):
    cfg, ds0, ds, p_vel, flat, ref4, common3 = e15.setup(sigma_T)
    common = common3.ravel(); shape = common3.shape; dx = float(ds0.attrs['dx'])
    ref = (ref4['dTdx'].ravel(), ref4['dTdy'].ravel(), ref4['dTdt'].ravel())
    w = json.loads((ROOT / "data" / "exp15_fit_sigma0.03.json").read_text())
    print(f"\n=========== low-noise comparison, sigma_T = {sigma_T} K ===========", flush=True)
    pts = G.build_T_points(flat['x'], flat['y'], flat['t'])
    p0 = G.pack_params_T_res(w['sigma1'], w['l1'], w['tau1'], w['sigma2'], w['l2'], w['tau2'], 2.5, 2.5, 2.5)
    gp = G.TGP(pts, ds['T_obs'].values.ravel(), p0, 2 * sigma_T ** 2, lowrank=e13.T_terms(w['lr:mean'], w['lr:diurnal']),
               nnum=nnum, Lx=20e3, Lt=24 * 3600.0, covfunc=G.T_covfunc_res)
    centre = dict(sigma1=0.5, l1=40e3, tau1=2 * 86400.0, sigma2=0.2, l2=6e3, tau2=6 * H, noise=2 * sigma_T ** 2, **{'lr:mean': 1.0, 'lr:diurnal': 0.1})
    start = {k: w[k] for k in FREE}; start['noise'] = 2 * sigma_T ** 2
    t0 = time.time()
    res = fit(gp, centre, FREE, bounds=e15.SPLIT_BOUNDS, prior_sd_log={n: 1.5 for n in FREE}, maxiter=maxiter, verbose=False,
              slots=T_KERNEL_SLOTS, start=start)
    r = res.params
    print(f"T-GP fit: {res.n_eval} evals, {time.time() - t0:.0f}s, success={res.success}, loglik {res.loglik:.1f}")
    print(f"  eddy sigma {r['sigma1']:.3f} K, l {r['l1'] / 1e3:.1f} km, tau {r['tau1'] / H:.1f} h | fast sigma {r['sigma2']:.3f} K, l {r['l2'] / 1e3:.2f} km, "
          f"tau {r['tau2'] / H:.1f} h | noise sd {np.sqrt(r['noise']):.4f} (true {sigma_T}) | lr mean {r['lr:mean']:.3f} diurnal {r['lr:diurnal']:.4f}", flush=True)
    (ROOT / "data" / f"exp20_fit_sigma{sigma_T}.json").write_text(json.dumps({k: float(v) for k, v in r.items()} | {'loglik': float(res.loglik)}, indent=1))

    nn = gp.pred_neighbours(flat['x'], flat['y'], flat['t'], nnum_pred)
    T_mean = gp.predict('T', flat['x'], flat['y'], flat['t'], nn_array=nn)[0]
    Tt_mean = gp.predict('Tt', flat['x'], flat['y'], flat['t'], nn_array=nn)[0]
    ds2, ds4 = e13.e12.with_products(ds, 2), e13.e12.with_products(ds, 4)
    print("\nderivative accuracy on the common interior (rms vs 4th-order FD of true T, fraction of std):")
    e13.deriv_rmse("FD2 (noisy T)", {k: ds2[k].values.ravel() for k in ('dTdx', 'dTdy', 'dTdt')}, {k: v.ravel() for k, v in ref4.items()}, common)
    e13.deriv_rmse("FD4 (noisy T)", {k: ds4[k].values.ravel() for k in ('dTdx', 'dTdy', 'dTdt')}, {k: v.ravel() for k, v in ref4.items()}, common)
    for order in (4, 2):
        Tx, Ty = grid_gradients(T_mean, shape, dx, order=order)
        e13.deriv_rmse(f"latent T, FD{order} + analytic Tt", {'dTdx': Tx, 'dTdy': Ty, 'dTdt': Tt_mean}, {k: v.ravel() for k, v in ref4.items()}, common)
    print("\nvelocity stage (true velocity hyperparameters, oracle iid noise), scored on the common interior:")
    e13.e12.evaluate("FD2 products, noisy gradients", ds2, 'obs', p_vel, common3, 'obs')
    e13.e12.evaluate("FD4 products, noisy gradients", ds4, 'obs', p_vel, common3, 'obs')
    e13.e12.evaluate("FD2 products, CLEAN gradients (bound)", ds2, 'obs', p_vel, common3, 'clean')
    e13.e12.evaluate("FD4 products, CLEAN gradients (bound)", ds4, 'obs', p_vel, common3, 'clean')
    for order in (4, 2):
        Tx, Ty = grid_gradients(T_mean, shape, dx, order=order)
        valid = np.isfinite(Tx) & np.isfinite(Ty)
        st = e16.Stage2(flat, common, p_vel, nnum=100, nnum_pred=nnum_pred, valid=valid)
        plug = st.run(Tx, Ty, Tt_mean, st.oracle_noise(Tx, Ty, Tt_mean))
        e16.score(f"latent T, FD{order} gradients (plug-in)", plug, st.truth)
    print("done", flush=True)


if __name__ == "__main__":
    for s in (sys.argv[1:] or ["0.01"]):
        main(float(s))

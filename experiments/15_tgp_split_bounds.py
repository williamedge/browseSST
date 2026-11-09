"""
Experiment 15: stage-1 T-GP fit with the two components SPLIT by bounds ("eddy" vs "fast" red noise, after the browseGP practice),
more iterations, warm-started from experiment 13's fits, and the plug-in re-run with nnum_pred = 200.

  component 1 (eddy):  l1 in [10 km, 500 km],  tau1 in [12 h, 1e7 s]
  component 2 (fast):  l2 in [3 km, 10 km],    tau2 in [1800 s, 12 h]

Saves the fitted parameters to data/exp15_fit_sigma<sigma_T>.json for the multiple-imputation experiment (16).
Run: python experiments/15_tgp_split_bounds.py <sigma_T> [maxiter] [nnum] [nnum_pred]
"""
import importlib.util, json, pathlib, sys, time, warnings
import numpy as np

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / (name + ".py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


e13 = _load("13_tgp_plugin")
G, S, K, ZS = e13.G, e13.S, e13.K, e13.ZS
from inversion_sst_gp.spacetime.fit import T_KERNEL_SLOTS, fit

H = 3600.0
SPLIT_BOUNDS = {'l1': (10e3, 500e3), 'tau1': (12 * H, 1e7), 'l2': (3e3, 10e3), 'tau2': (1800.0, 12 * H),
                'noise': (1e-7, 1.0), 'lr:mean': (1e-3, 1e2), 'lr:diurnal': (1e-4, 1e1)}
# experiment 13 fits (warm start); l2 nudged off its 3 km floor
WARM = {0.03: dict(sigma1=0.183, l1=12.4e3, tau1=26.8 * H, sigma2=0.067, l2=3.3e3, tau2=9.2 * H, noise=0.0292 ** 2, **{'lr:mean': 0.42, 'lr:diurnal': 0.017}),
        0.1: dict(sigma1=0.212, l1=13.6e3, tau1=29.6 * H, sigma2=0.066, l2=3.5e3, tau2=8.1 * H, noise=0.0988 ** 2, **{'lr:mean': 0.26, 'lr:diurnal': 0.005})}


def setup(sigma_T):
    cfg = S.SyntheticConfig(nx=14, ny=14, n_hours=48, n_modes=400)
    ds0 = S.load_or_generate(cfg, ROOT / "data" / "synth_14px_48h_seed1.nc")
    ds = S.reobserve(ds0, sigma_T, seed=cfg.seed + 7)
    tp = cfg.true_params(ZS)
    p_vel = K.pack_params(tp['sigma_psi'], tp['l_psi'], tp['sigma_phi'], tp['l_phi'], tp['tau0_psi'],
                          tp['tau0_phi'], tp['sigma_S'], tp['l_S'], tp['tau0_S'])
    tt, yy, xx = np.meshgrid(ds0['time'].values, ds0['y'].values, ds0['x'].values, indexing='ij')
    flat = dict(x=xx.ravel(), y=yy.ravel(), t=tt.ravel(), u=ds0['u_true'].values.ravel(), v=ds0['v_true'].values.ravel(),
                S=ds0['S_true'].values.ravel() / ZS)
    ref4 = dict(zip(('dTdt', 'dTdx', 'dTdy'), S.derivative_products(ds0['T_true'].values, float(ds0.attrs['dx']), order=4)))
    common3 = np.isfinite(ref4['dTdt']) & np.isfinite(ref4['dTdx']) & np.isfinite(ref4['dTdy'])
    return cfg, ds0, ds, p_vel, flat, ref4, common3


def main(sigma_T, maxiter=60, nnum=100, nnum_pred=200):
    cfg, ds0, ds, p_vel, flat, ref4, common3 = setup(sigma_T)
    common = common3.ravel()
    print(f"\n================ sigma_T = {sigma_T} K, split bounds, maxiter {maxiter}, nnum {nnum}, nnum_pred {nnum_pred} ================", flush=True)
    pts = G.build_T_points(flat['x'], flat['y'], flat['t'])
    free = ['sigma1', 'l1', 'tau1', 'sigma2', 'l2', 'tau2', 'noise', 'lr:mean', 'lr:diurnal']
    centre = dict(sigma1=0.5, l1=40e3, tau1=2 * 86400.0, sigma2=0.2, l2=6e3, tau2=6 * H,
                  noise=2 * sigma_T ** 2, **{'lr:mean': 1.0, 'lr:diurnal': 0.1})      # prior centres (l2, tau2 moved inside the new bounds)
    w = WARM[sigma_T]
    p0 = G.pack_params_T(w['sigma1'], w['l1'], w['tau1'], w['sigma2'], w['l2'], w['tau2'], nu_t=2.5)
    gp = G.TGP(pts, ds['T_obs'].values.ravel(), p0, w['noise'], lowrank=e13.T_terms(w['lr:mean'], w['lr:diurnal']),
               nnum=nnum, Lx=20e3, Lt=24 * 3600.0)
    t0 = time.time()
    res = fit(gp, centre, free, bounds=SPLIT_BOUNDS, prior_sd_log={n: 1.5 for n in free}, maxiter=maxiter, verbose=False,
              slots=T_KERNEL_SLOTS, start=w)
    r = res.params
    print(f"T-GP fit: {res.n_eval} evals, {time.time() - t0:.0f}s, success={res.success} ({res.message}), loglik {res.loglik:.1f}")
    print(f"  eddy: sigma {r['sigma1']:.3f} K, l {r['l1'] / 1e3:.1f} km, tau {r['tau1'] / H:.1f} h | "
          f"fast: sigma {r['sigma2']:.3f} K, l {r['l2'] / 1e3:.1f} km, tau {r['tau2'] / H:.1f} h")
    print(f"  noise sd {np.sqrt(r['noise']):.4f} K (true {sigma_T}); mean scale {r['lr:mean']:.3f}, diurnal scale {r['lr:diurnal']:.4f}", flush=True)
    out = ROOT / "data" / f"exp15_fit_sigma{sigma_T}.json"
    out.write_text(json.dumps({k: float(v) for k, v in r.items()} | {'loglik': float(res.loglik), 'success': bool(res.success)}, indent=1))

    t0 = time.time()
    est = {}
    for f, k in (('Tx', 'dTdx'), ('Ty', 'dTdy'), ('Tt', 'dTdt')):
        est[k], _ = gp.predict(f, flat['x'], flat['y'], flat['t'], nnum_pred=nnum_pred)
    print(f"derivative predictions at {len(flat['x'])} points (nnum_pred {nnum_pred}): {time.time() - t0:.0f}s", flush=True)
    print("\nderivative accuracy on the common interior:")
    e13.deriv_rmse("T-GP posterior mean (split)", est, {k: v.ravel() for k, v in ref4.items()}, common)
    print("\nvelocity stage (true velocity hyperparameters, oracle iid noise), scored on the common interior:")
    fd2 = e13.e12.with_products(ds, 2)
    fd2_valid = np.isfinite(fd2['dTdt'].values).ravel() & np.isfinite(fd2['dTdx'].values).ravel() & np.isfinite(fd2['dTdy'].values).ravel()
    e13.stage2("T-GP plug-in, FD2 point set", ds, flat, fd2_valid, est['dTdx'], est['dTdy'], est['dTdt'], common, p_vel)
    e13.stage2("T-GP plug-in, all points", ds, flat, np.ones_like(fd2_valid), est['dTdx'], est['dTdy'], est['dTdt'], common, p_vel)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(float(a[0]), maxiter=int(a[1]) if len(a) > 1 else 60, nnum=int(a[2]) if len(a) > 2 else 100,
         nnum_pred=int(a[3]) if len(a) > 3 else 200)

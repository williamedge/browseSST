"""
Experiment 37: does a STRUCTURED source term explain the real dT/dt better than white noise?

Real tile of exps 35 / 36 (28 x 28 px x 48 h, days 3-5, 99% observed; stage-1 hyperparameters from exp 35; no tide removed). The velocity GP is fitted by the Vecchia marginal likelihood with ALL
parameters free (velocity amplitude / scales, S amplitude / scales, diurnal scale, and the noise above a floor of 1 scaled unit) in three variants:
  whiteS     : the old structure: a tiny S prior (sigma_S 0.3, tau_S 1 d) and a fixed-cycle uniform diurnal term; the noise (start 709, the exp-36 ML value) has to carry the non-advective dT/dt
  fixedcycle : structured S: Matern space-time S started at sigma_S 15 (1.5e-5 K/s), l_S 8 km, tau_S 2 h, + the same uniform diurnal cycle every day, small noise (start 4)
  daily      : as fixedcycle but the diurnal harmonics' amplitude may change day by day (24 h linear hats)
Compared by the log marginal likelihood (the same data rows and Vecchia settings in every variant, so the numbers are comparable) and by what the fitted model says: where the noise ends up, the
size of S, the velocity amplitude, the posterior variance of u relative to its prior (information gained), and the agreement of the posterior-mean velocity between variants on a thinned set of points.
There is no velocity truth: this is a model-adequacy test, not a skill test.
Run: python experiments/37_real_structured_S.py products <cube>   |   fit <cube> <whiteS|fixedcycle|daily> [maxiter]   |   compare
"""
import json, pathlib, sys, time, warnings
import numpy as np
import xarray as xr

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = pathlib.Path.home() / "browsesst_data" / "real_test"
from inversion_sst_gp.spacetime import kernels as K, pipeline as P, tiling as TL
from inversion_sst_gp.spacetime.fit import fit

H0, J0, I0, NT, N = 72, 24, 24, 48, 28
START_VEL = dict(gamma=0.3, f_delta=0.2, l_psi=30e3, l_phi=20e3, tau0_psi=86400.0, tau0_phi=43200.0)
STARTS = {
    'whiteS': dict(START_VEL, sigma_S=0.3, l_S=20e3, tau0_S=86400.0, noise=709.0, **{'lr:diurnal': 0.5}),
    'fixedcycle': dict(START_VEL, sigma_S=15.0, l_S=8e3, tau0_S=7200.0, noise=4.0, **{'lr:diurnal': 20.0}),
    'daily': dict(START_VEL, sigma_S=15.0, l_S=8e3, tau0_S=7200.0, noise=4.0, **{'lr:diurnal': 20.0}),
}


def tile(cube):
    spec = TL.TileSpec("real48", H0, H0 + NT, J0, J0 + N, I0, I0 + N, H0, H0 + NT, J0, J0 + N, I0, I0 + N)
    return TL.extract_tile(cube, spec)


def products(path):
    cube = xr.open_dataset(path)
    td = tile(cube)
    r = json.loads((ROOT / "data" / f"exp35_real_stage1_h{H0}_j{J0}_i{I0}.json").read_text())['params']
    h1 = {k: float(r[k]) for k in P.STAGE1_KEYS}
    cfg = P.PipelineConfig()
    xo, yo, to, To = td.observed()
    t0 = time.time()
    gp1 = P.build_stage1(xo, yo, to, To, h1, cfg)
    Tm, Ttm, vTt = P.stage1_products(td, gp1, cfg)
    np.savez(DATA / "exp37_products.npz", Tm=Tm, Ttm=Ttm, vTt=vTt)
    print(f"stage-1 products {time.time() - t0:.0f}s saved", flush=True)


def prior_for(variant):
    return P.HelmholtzMaternPrior() if variant == 'whiteS' else P.StructuredSPrior(daily=(variant == 'daily'))


def run_fit(path, variant, maxiter=20):
    cube = xr.open_dataset(path)
    td = tile(cube)
    z_ = np.load(DATA / "exp37_products.npz")
    Tm, Ttm, vTt = z_['Tm'], z_['Ttm'], z_['vTt']
    base = P.PipelineConfig()
    prior = prior_for(variant)
    cfg = P.PipelineConfig(prior=prior)
    zero = np.zeros(td.x.size)
    idx, Tx, Ty, z = P.stage2_rows(td, Tm, Ttm, vTt, zero, zero, base)
    h2 = dict(STARTS[variant])
    gp2 = P.build_stage2(td, idx, Tx, Ty, z, h2, cfg)
    free = list(prior.free) if 'noise' in prior.free else list(prior.free) + ['noise']
    bounds = dict(prior.bounds); bounds.setdefault('noise', (1.0, 1e4))
    print(f"[{variant}] {idx.size} stage-2 rows, free {free}", flush=True)
    t0 = time.time()
    ll0 = gp2.loglik()
    res = fit(gp2, h2, free, bounds=bounds, prior_sd_log={n: 1.5 for n in free}, maxiter=maxiter, verbose=False)
    fitted = dict(h2); fitted.update({k: float(res.params[k]) for k in free})
    gp2.update(cov_params=prior.cov_params(fitted), noise=fitted['noise'], lowrank_scales={'diurnal': fitted['lr:diurnal']})
    ll = gp2.loglik()
    print(f"[{variant}] fit {res.n_eval} evals {time.time() - t0:.0f}s success={res.success}; loglik start {ll0:.1f} -> {ll:.1f}", flush=True)
    print("  fitted: " + ", ".join(f"{k} {fitted[k]:.3g}" for k in free), flush=True)
    # velocity on a thinned set: every 4th hour, every 3rd pixel
    sh = td.shape
    kk, jj, ii = np.meshgrid(np.arange(2, NT - 2, 4), np.arange(3, N - 3, 3), np.arange(3, N - 3, 3), indexing='ij')
    sel = np.ravel_multi_index((kk.ravel(), jj.ravel(), ii.ravel()), sh)
    nn = gp2.pred_neighbours(td.x[sel], td.y[sel], td.t[sel], 200)
    u, vu = gp2.predict('u', td.x[sel], td.y[sel], td.t[sel], nn_array=nn)
    v, vv = gp2.predict('v', td.x[sel], td.y[sel], td.t[sel], nn_array=nn)
    pv = prior.cov_params(fitted)
    r = lambda a, b: np.array([0.0, 0.0, 0.0, a, b, 0.0])
    prior_var_u = float(K.transport_obs_covfunc(r(1, 0), r(1, 0), pv)); prior_var_v = float(K.transport_obs_covfunc(r(0, 1), r(0, 1), pv))
    info = dict(variant=variant, loglik=float(ll), loglik_start=float(ll0), n_eval=int(res.n_eval), success=bool(res.success), fitted={k: float(fitted[k]) for k in free},
                prior_sd_u=float(np.sqrt(prior_var_u)), post_over_prior_var_u=float(np.mean(vu) / prior_var_u), post_over_prior_var_v=float(np.mean(vv) / prior_var_v),
                rms_u=float(np.sqrt(np.mean(u ** 2))), rms_v=float(np.sqrt(np.mean(v ** 2))), n_rows=int(idx.size))
    (DATA / f"exp37_{variant}.json").write_text(json.dumps(info, indent=1))
    np.savez(DATA / f"exp37_{variant}_vel.npz", u=u, v=v, vu=vu, vv=vv)
    print(f"  velocity: prior sd {info['prior_sd_u']:.3f} m/s, posterior / prior variance of u {info['post_over_prior_var_u']:.2f}, v {info['post_over_prior_var_v']:.2f}; rms posterior-mean u {info['rms_u']:.3f}, v {info['rms_v']:.3f} m/s", flush=True)


def compare():
    names = [v for v in ('whiteS', 'fixedcycle', 'daily') if (DATA / f"exp37_{v}.json").exists()]
    out = {v: json.loads((DATA / f"exp37_{v}.json").read_text()) for v in names}
    vel = {v: np.load(DATA / f"exp37_{v}_vel.npz") for v in names}
    print("variant      loglik      noise(sd K/s)  sigma_S(K/s)  l_S(km) tau_S(h)  lr:diurnal  gamma  post/prior var u  rms u")
    for v in names:
        f = out[v]['fitted']
        print(f"{v:11s} {out[v]['loglik']:10.1f}  {np.sqrt(f['noise']) * 1e-6:11.2e}  {f['sigma_S'] * 1e-6:11.2e}  {f['l_S'] / 1e3:6.1f}  {f['tau0_S'] / 3600:6.1f}  {f['lr:diurnal']:9.2f}  {f['gamma']:.3f}  {out[v]['post_over_prior_var_u']:8.2f}      {out[v]['rms_u']:.3f}")
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            A, B = vel[names[a]], vel[names[b]]
            cu = np.corrcoef(A['u'], B['u'])[0, 1]; cv = np.corrcoef(A['v'], B['v'])[0, 1]
            print(f"posterior-mean velocity, {names[a]} vs {names[b]}: correlation u {cu:.2f}, v {cv:.2f}; rms difference u {np.sqrt(np.mean((A['u'] - B['u']) ** 2)):.3f}, v {np.sqrt(np.mean((A['v'] - B['v']) ** 2)):.3f} m/s")


if __name__ == "__main__":
    a = sys.argv[1:]
    DATA.mkdir(parents=True, exist_ok=True)
    if a[0] == 'products':
        products(a[1])
    elif a[0] == 'fit':
        run_fit(a[1], a[2], int(a[3]) if len(a) > 3 else 20)
    else:
        compare()

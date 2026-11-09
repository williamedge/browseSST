"""
Experiment 23: replicate the key comparison over independent synthetic realisations (new velocity / source / initial-tracer draws; the fronts field of exp 21 or the
smooth field of exps 01-20).

Per seed, at one noise level: simulate, re-observe, fit the 5/2 : 5/2 T-GP (split bounds, warm start from the exp-21 / exp-15 fit at this noise level), then velocity skill and 95%
coverage (all points / steepest 20%) for: FD2 and FD4 on the noisy T, the FD4 clean-gradient bound, the latent-T plug-in (4th-order gradients) and the plug-in + propagated variance
(K Matheron draws, equation-error stage-2 noise). True velocity hyperparameters, oracle noise for the finite-difference rows. Saves data/exp23_<field>_seed<seed>_sigma<sigma>.json;
`aggregate` (run with "agg") prints mean +- sd over the seeds found.
Run: python experiments/23_multiseed.py <seed> <sigma_T> [fronts|smooth] [K]      |     python experiments/23_multiseed.py agg <sigma_T> [fronts|smooth]
"""
import importlib.util, json, pathlib, sys, time, warnings
import numpy as np

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / (name + ".py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


e18 = _load("18_resolved_gradient_mi"); e21 = _load("21_fronts_filaments")
e16, e13, e15, G, K, ZS = e18.e16, e18.e13, e18.e15, e18.G, e18.K, e18.ZS
S = e13.S
from inversion_sst_gp.spacetime.fit import T_KERNEL_SLOTS, fit
from inversion_sst_gp.spacetime.impute import DerivativeImputer

H = 3600.0
FREE = ['sigma1', 'l1', 'tau1', 'sigma2', 'l2', 'tau2', 'noise', 'lr:mean', 'lr:diurnal']
SMOOTH_CFG = dict(nx=14, ny=14, n_hours=48, n_modes=400)


def metrics(est, truth, hi):
    out = {}
    lo = ~hi
    for f in 'uvS':
        m, v = est[f]; e = m - truth[f]
        for tag, sel in (('all', np.ones_like(hi)), ('steep', hi)):
            out[f"{f}_skill_{tag}"] = float(1 - np.mean(e[sel] ** 2) / truth[f][sel].var())
            out[f"{f}_cover_{tag}"] = float(np.mean((np.abs(e) < 1.96 * np.sqrt(v))[sel]))
    return out


def pipeline(seed, sigma_T, field='fronts', Kdraws=10, maxiter=60, nnum_pred=200):
    kw = dict(e21.FRONT_CFG) if field == 'fronts' else dict(SMOOTH_CFG)
    cfg = S.SyntheticConfig(**kw, seed=seed)
    ds0 = S.load_or_generate(cfg, ROOT / "data" / f"synth_14px_48h_{field}_seed{seed}.nc")
    ds = S.reobserve(ds0, sigma_T, seed=cfg.seed + 7)
    tp = cfg.true_params(ZS)
    p_vel = K.pack_params(tp['sigma_psi'], tp['l_psi'], tp['sigma_phi'], tp['l_phi'], tp['tau0_psi'], tp['tau0_phi'], tp['sigma_S'], tp['l_S'], tp['tau0_S'])
    tt, yy, xx = np.meshgrid(ds0['time'].values, ds0['y'].values, ds0['x'].values, indexing='ij')
    flat = dict(x=xx.ravel(), y=yy.ravel(), t=tt.ravel(), u=ds0['u_true'].values.ravel(), v=ds0['v_true'].values.ravel(), S=ds0['S_true'].values.ravel() / ZS)
    dx = float(ds0.attrs['dx'])
    ref4 = dict(zip(('dTdt', 'dTdx', 'dTdy'), S.derivative_products(ds0['T_true'].values, dx, order=4)))
    common3 = np.isfinite(ref4['dTdt']) & np.isfinite(ref4['dTdx']) & np.isfinite(ref4['dTdy'])
    common = common3.ravel(); shape = common3.shape
    gmag = np.hypot(ref4['dTdx'], ref4['dTdy']).ravel()[common]
    hi = gmag >= np.percentile(gmag, 80)
    print(f"[seed {seed}, {field}, sigma_T {sigma_T}] velocity std u {flat['u'][common].std():.3f} v {flat['v'][common].std():.3f}; |grad T| rms {np.sqrt(np.mean(gmag ** 2)):.2e}", flush=True)
    res = {}

    def fd_run(order, clean):
        dsx = e13.e12.with_products(ds, order)
        sfx = '_clean' if clean else ''
        Tx, Ty, Tt = dsx['dTdx' + sfx].values.ravel(), dsx['dTdy' + sfx].values.ravel(), dsx['dTdt'].values.ravel()
        valid = np.isfinite(Tx) & np.isfinite(Ty) & np.isfinite(Tt)
        st = e16.Stage2(flat, common, p_vel, nnum=100, nnum_pred=nnum_pred, valid=valid)
        return metrics(st.run(Tx, Ty, Tt, st.oracle_noise(Tx, Ty, Tt)), st.truth, hi)
    res['FD2 noisy'] = fd_run(2, False); res['FD4 noisy'] = fd_run(4, False); res['FD4 clean-gradient bound'] = fd_run(4, True)
    print("  baselines done", flush=True)

    src = ROOT / "data" / (f"exp21_fit_sigma{sigma_T}_nu2.5_2.5.json" if field == 'fronts' else
                           (f"exp15_fit_sigma{sigma_T}.json" if sigma_T in (0.03, 0.1) else f"exp20_fit_sigma{sigma_T}.json"))
    w = json.loads(src.read_text())
    pts = G.build_T_points(flat['x'], flat['y'], flat['t'])
    p0 = G.pack_params_T_res(w['sigma1'], w['l1'], w['tau1'], w['sigma2'], w['l2'], w['tau2'], 2.5, 2.5, 2.5)
    gp = G.TGP(pts, ds['T_obs'].values.ravel(), p0, w['noise'], lowrank=e13.T_terms(w['lr:mean'], w['lr:diurnal']), nnum=100, Lx=20e3, Lt=24 * 3600.0, covfunc=G.T_covfunc_res)
    centre = dict(sigma1=0.5, l1=40e3, tau1=2 * 86400.0, sigma2=0.2, l2=6e3, tau2=6 * H, noise=2 * sigma_T ** 2, **{'lr:mean': 1.0, 'lr:diurnal': 0.1})
    t0 = time.time()
    fr = fit(gp, centre, FREE, bounds=e15.SPLIT_BOUNDS, prior_sd_log={n: 1.5 for n in FREE}, maxiter=maxiter, verbose=False, slots=T_KERNEL_SLOTS, start={k: w[k] for k in FREE})
    r = fr.params
    print(f"  T-GP fit {fr.n_eval} evals, {time.time() - t0:.0f}s, success={fr.success}, loglik {fr.loglik:.1f}; noise sd {np.sqrt(r['noise']):.4f} (true {sigma_T}); eddy {r['sigma1']:.3f} K {r['l1'] / 1e3:.1f} km {r['tau1'] / H:.0f} h; "
          f"fast {r['sigma2']:.3f} K {r['l2'] / 1e3:.2f} km {r['tau2'] / H:.1f} h", flush=True)
    res['fit'] = {k: float(v) for k, v in r.items()} | {'loglik': float(fr.loglik), 'success': bool(fr.success)}
    imp = DerivativeImputer(gp, flat['x'], flat['y'], flat['t'], dx=dx, dt=3600.0, nnum_prior=100, nnum_pred=nnum_pred, seed=100 + seed, fields=('T', 'Tt'), targets=('T', 'Tt'))
    mean, draws = imp.posterior_samples(Kdraws, verbose=False)
    mean_prod = e18.products(mean, shape, dx); draw_prods = [e18.products(d, shape, dx) for d in draws]
    valid = np.isfinite(mean_prod[0]) & np.isfinite(mean_prod[1])
    st = e16.Stage2(flat, common, p_vel, nnum=100, nnum_pred=nnum_pred, valid=valid)
    nz_eq = e18.equation_noise(mean_prod, draw_prods, e18.prior_velocity_moments(p_vel), valid)
    plug = st.run(*mean_prod, nz_eq)
    runs = [st.run(*dp, nz_eq) for dp in draw_prods]
    res['latent T FD4 plug-in'] = metrics(plug, st.truth, hi)
    res['latent T FD4 + propagated variance'] = metrics(e18.propagated(plug, runs), st.truth, hi)
    res['latent T FD4 plug-in, oracle noise'] = metrics(st.run(*mean_prod, st.oracle_noise(*mean_prod)), st.truth, hi)
    res['noise_eq'] = float(nz_eq)
    out = ROOT / "data" / f"exp23_{field}_seed{seed}_sigma{sigma_T}.json"
    out.write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        if isinstance(v, dict) and 'u_skill_all' in v:
            print(f"  {k:<40s} u {v['u_skill_all']:+.3f}/{v['u_skill_steep']:+.3f} v {v['v_skill_all']:+.3f}/{v['v_skill_steep']:+.3f} S {v['S_skill_all']:+.3f} | cover u {v['u_cover_all']:.2f} v {v['v_cover_all']:.2f} S {v['S_cover_all']:.2f}", flush=True)
    print("done", flush=True)


def aggregate(sigma_T, field='fronts'):
    files = sorted((ROOT / "data").glob(f"exp23_{field}_seed*_sigma{sigma_T}.json"))
    runs = [json.loads(f.read_text()) for f in files]
    print(f"{len(runs)} seeds: {[f.name.split('_')[2] for f in files]}  ({field}, sigma_T {sigma_T})")
    names = [k for k, v in runs[0].items() if isinstance(v, dict) and 'u_skill_all' in v]
    for k in names:
        def ms(key):
            a = np.array([r[k][key] for r in runs]); return f"{a.mean():+.3f}+-{a.std(ddof=1) if len(a) > 1 else 0:.3f}"
        def mc(key):
            a = np.array([r[k][key] for r in runs]); return f"{a.mean():.2f}+-{a.std(ddof=1) if len(a) > 1 else 0:.2f}"
        print(f"  {k:<40s} u {ms('u_skill_all')} (steep {ms('u_skill_steep')}) | v {ms('v_skill_all')} (steep {ms('v_skill_steep')}) | S {ms('S_skill_all')} | cover u {mc('u_cover_all')} v {mc('v_cover_all')} S {mc('S_cover_all')}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == 'agg':
        aggregate(float(a[1]), a[2] if len(a) > 2 else 'fronts')
    else:
        pipeline(int(a[0]), float(a[1]), a[2] if len(a) > 2 else 'fronts', Kdraws=int(a[3]) if len(a) > 3 else 10)

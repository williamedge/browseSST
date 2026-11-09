"""
Experiment 24: ESTIMATE the velocity-stage hyperparameters (every earlier stage-2 result used the true ones).

Per seed (fronts + filament field, sigma_T = 0.03 K, the five realisations of exp 23; stage-1 T-GP hyperparameters reloaded from `data/exp23_*.json`, no refit):
posterior-mean latent T / Tt and K Matern draws -> resolved (4th-order) gradients -> equation-error stage-2 noise (fixed, not fitted) -> the stage-2 Helmholtz + S kernel
hyperparameters (gamma, f_delta, l_psi, l_phi, tau_psi, tau_phi, sigma_S, l_S, tau_S and the diurnal low-rank S scale) fitted by the Vecchia marginal likelihood with
log-normal priors (sd 1.0 in log) centred on the START values:
  'true'   the true values (reference, as in exp 23);
  'fit A'  start = a generic, roughly right guess (gamma 0.10 m/s, f_delta 0.2, l_psi 30 km, l_phi 20 km, tau 2 d / 1 d, sigma_S 0.3, l_S 20 km, tau_S 1 d);
  'fit B'  start = deliberately poor, wrong-prior guess (gamma 0.05, f_delta 0.5, l_psi 60 km, l_phi 8 km, tau 1 d / 6 h, sigma_S 0.1, l_S 40 km, tau_S 3 h).
Reports u / v skill and 95% coverage (all / steepest 20%), plug-in and (for 'true' and 'fit A') plug-in + propagated variance, and fitted / true parameter ratios.
S is a nuisance here (its skill is stored but not reported).
Run: python experiments/24_estimated_velocity_hyperparams.py <seed> [sigma_T] [K] [maxiter]      |      ... agg [sigma_T]
"""
import importlib.util, json, pathlib, sys, time, warnings
import numpy as np

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / (name + ".py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


e23 = _load("23_multiseed"); e18, e21, e16, e15, e13 = e23.e18, e23.e21, e23.e16, e23.e15, e23.e13
G, K, ZS, S = e23.G, e23.K, e23.ZS, e23.S
from inversion_sst_gp.spacetime.fit import fit, to_amplitude_params, to_potential_params, params_to_model_args
from inversion_sst_gp.spacetime.impute import DerivativeImputer

DAY = 86400.0
FREE = ['gamma', 'f_delta', 'l_psi', 'l_phi', 'tau0_psi', 'tau0_phi', 'sigma_S', 'l_S', 'tau0_S', 'lr:diurnal']
BOUNDS = {'lr:diurnal': (1e-3, 1e2), 'sigma_S': (1e-2, 1e1), 'tau0_psi': (3600.0, 3e6), 'tau0_phi': (3600.0, 3e6), 'tau0_S': (3600.0, 3e6)}
STARTS = {
    'fit A': dict(gamma=0.10, f_delta=0.2, l_psi=30e3, l_phi=20e3, tau0_psi=2 * DAY, tau0_phi=1 * DAY, sigma_S=0.3, l_S=20e3, tau0_S=1 * DAY, **{'lr:diurnal': 0.5}),
    'fit B': dict(gamma=0.05, f_delta=0.5, l_psi=60e3, l_phi=8e3, tau0_psi=1 * DAY, tau0_phi=0.25 * DAY, sigma_S=0.1, l_S=40e3, tau0_S=3 * 3600.0, **{'lr:diurnal': 0.5}),
}
PARAMS = ['gamma', 'f_delta', 'l_psi', 'l_phi', 'tau0_psi', 'tau0_phi', 'sigma_S', 'l_S', 'tau0_S']


def cov_from_amp(a):
    p = to_potential_params(a)
    return K.pack_params(p['sigma_psi'], p['l_psi'], p['sigma_phi'], p['l_phi'], p['tau0_psi'], p['tau0_phi'], p['sigma_S'], p['l_S'], p['tau0_S'])


def pipeline(seed, sigma_T=0.03, Kdraws=10, maxiter=40, nnum_pred=200):
    cfg = S.SyntheticConfig(**e21.FRONT_CFG, seed=seed)
    ds0 = S.load_or_generate(cfg, ROOT / "data" / f"synth_14px_48h_fronts_seed{seed}.nc")
    ds = S.reobserve(ds0, sigma_T, seed=cfg.seed + 7)
    tp = cfg.true_params(ZS)
    p_true = K.pack_params(tp['sigma_psi'], tp['l_psi'], tp['sigma_phi'], tp['l_phi'], tp['tau0_psi'], tp['tau0_phi'], tp['sigma_S'], tp['l_S'], tp['tau0_S'])
    truth_amp = to_amplitude_params(dict(sigma_psi=tp['sigma_psi'], l_psi=tp['l_psi'], sigma_phi=tp['sigma_phi'], l_phi=tp['l_phi'], tau0_psi=tp['tau0_psi'], tau0_phi=tp['tau0_phi'],
                                         sigma_S=tp['sigma_S'], l_S=tp['l_S'], tau0_S=tp['tau0_S']))
    tt, yy, xx = np.meshgrid(ds0['time'].values, ds0['y'].values, ds0['x'].values, indexing='ij')
    flat = dict(x=xx.ravel(), y=yy.ravel(), t=tt.ravel(), u=ds0['u_true'].values.ravel(), v=ds0['v_true'].values.ravel(), S=ds0['S_true'].values.ravel() / ZS)
    dx = float(ds0.attrs['dx'])
    ref4 = dict(zip(('dTdt', 'dTdx', 'dTdy'), S.derivative_products(ds0['T_true'].values, dx, order=4)))
    common3 = np.isfinite(ref4['dTdt']) & np.isfinite(ref4['dTdx']) & np.isfinite(ref4['dTdy'])
    common = common3.ravel(); shape = common3.shape
    gmag = np.hypot(ref4['dTdx'], ref4['dTdy']).ravel()[common]
    hi = gmag >= np.percentile(gmag, 80)
    print(f"[seed {seed}] true amplitude parameters: " + ", ".join(f"{k} {truth_amp[k]:.3g}" for k in PARAMS), flush=True)

    # stage-1 products from the saved T-GP fit (no refit)
    prev = json.loads((ROOT / "data" / f"exp23_fronts_seed{seed}_sigma{sigma_T}.json").read_text())['fit']
    pts = G.build_T_points(flat['x'], flat['y'], flat['t'])
    p0 = G.pack_params_T_res(prev['sigma1'], prev['l1'], prev['tau1'], prev['sigma2'], prev['l2'], prev['tau2'], 2.5, 2.5, 2.5)
    gp = G.TGP(pts, ds['T_obs'].values.ravel(), p0, prev['noise'], lowrank=e13.T_terms(prev['lr:mean'], prev['lr:diurnal']), nnum=100, Lx=20e3, Lt=24 * 3600.0, covfunc=G.T_covfunc_res)
    imp = DerivativeImputer(gp, flat['x'], flat['y'], flat['t'], dx=dx, dt=3600.0, nnum_prior=100, nnum_pred=nnum_pred, seed=100 + seed, fields=('T', 'Tt'), targets=('T', 'Tt'))
    t0 = time.time()
    mean, draws = imp.posterior_samples(Kdraws, verbose=False)
    mean_prod = e18.products(mean, shape, dx); draw_prods = [e18.products(d, shape, dx) for d in draws]
    valid = np.isfinite(mean_prod[0]) & np.isfinite(mean_prod[1])
    st = e16.Stage2(flat, common, p_true, nnum=100, nnum_pred=nnum_pred, valid=valid)
    noise = e18.equation_noise(mean_prod, draw_prods, e18.prior_velocity_moments(p_true), valid)
    print(f"[seed {seed}] stage-1 products + {Kdraws} draws {time.time() - t0:.0f}s; equation-error noise {noise:.2f}", flush=True)

    out = {'truth': {k: float(truth_amp[k]) for k in PARAMS}, 'noise_eq': float(noise)}

    def evaluate(label, p, lr, with_draws):
        plug = st.run(*mean_prod, noise, p=p, lr=lr)
        out.setdefault(label, {})['plug-in'] = e23.metrics(plug, st.truth, hi)
        line = f"[seed {seed}] {label:<7s} plug-in u {out[label]['plug-in']['u_skill_all']:+.3f} (steep {out[label]['plug-in']['u_skill_steep']:+.3f}) v {out[label]['plug-in']['v_skill_all']:+.3f} (steep {out[label]['plug-in']['v_skill_steep']:+.3f}) cover {out[label]['plug-in']['u_cover_all']:.2f}/{out[label]['plug-in']['v_cover_all']:.2f}"
        if with_draws:
            runs = [st.run(*dp, noise, p=p, lr=lr) for dp in draw_prods]
            out[label]['propagated'] = e23.metrics(e18.propagated(plug, runs), st.truth, hi)
            line += f" | + propagated cover {out[label]['propagated']['u_cover_all']:.2f}/{out[label]['propagated']['v_cover_all']:.2f} (steep {out[label]['propagated']['u_cover_steep']:.2f}/{out[label]['propagated']['v_cover_steep']:.2f})"
        print(line, flush=True)

    evaluate('true', p_true, 0.5, True)
    for label, start in STARTS.items():
        t0 = time.time()
        params0 = dict(start); params0['noise'] = noise
        gp2, _, _ = st._gp(*mean_prod, noise, p=cov_from_amp(start), lr=start['lr:diurnal'])
        res = fit(gp2, params0, FREE, bounds=BOUNDS, prior_sd_log={n: 1.0 for n in FREE}, maxiter=maxiter, verbose=False)
        fitted = res.params
        cov = np.array(gp2.op.covparams, float)
        ratios = {k: float(fitted[k] / truth_amp[k]) for k in PARAMS}
        out[label] = {'fitted': {k: float(fitted[k]) for k in FREE}, 'ratio_to_truth': ratios, 'loglik': float(res.loglik), 'n_eval': int(res.n_eval), 'success': bool(res.success)}
        print(f"[seed {seed}] {label} fit {res.n_eval} evals {time.time() - t0:.0f}s success={res.success}; fitted/true: " + ", ".join(f"{k} {ratios[k]:.2f}" for k in PARAMS), flush=True)
        evaluate(label, cov, fitted['lr:diurnal'], label == 'fit A')
    (ROOT / "data" / f"exp24_fronts_seed{seed}_sigma{sigma_T}.json").write_text(json.dumps(out, indent=1))
    print("done", flush=True)


def aggregate(sigma_T=0.03):
    files = sorted((ROOT / "data").glob(f"exp24_fronts_seed*_sigma{sigma_T}.json"))
    runs = [json.loads(f.read_text()) for f in files]
    print(f"{len(runs)} seeds, sigma_T {sigma_T}")
    for label in ('true', 'fit A', 'fit B'):
        for kind in ('plug-in', 'propagated'):
            rs = [r[label][kind] for r in runs if kind in r[label]]
            if not rs:
                continue
            def ms(key):
                a = np.array([r[key] for r in rs]); return f"{a.mean():+.3f}+-{a.std(ddof=1) if len(a) > 1 else 0:.3f}"
            def mc(key):
                a = np.array([r[key] for r in rs]); return f"{a.mean():.2f}+-{a.std(ddof=1) if len(a) > 1 else 0:.2f}"
            print(f"  {label:<6s} {kind:<11s} u {ms('u_skill_all')} (steep {ms('u_skill_steep')}) | v {ms('v_skill_all')} (steep {ms('v_skill_steep')}) | cover u {mc('u_cover_all')} v {mc('v_cover_all')}")
    print("\nfitted / true (geometric mean; min - max over seeds):")
    for label in ('fit A', 'fit B'):
        print(f"  {label}: " + ", ".join(
            f"{k} {np.exp(np.mean(np.log([r[label]['ratio_to_truth'][k] for r in runs]))):.2f} ({min(r[label]['ratio_to_truth'][k] for r in runs):.2f}-{max(r[label]['ratio_to_truth'][k] for r in runs):.2f})" for k in PARAMS))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == 'agg':
        aggregate(float(a[1]) if len(a) > 1 else 0.03)
    else:
        pipeline(int(a[0]), float(a[1]) if len(a) > 1 else 0.03, Kdraws=int(a[2]) if len(a) > 2 else 10, maxiter=int(a[3]) if len(a) > 3 else 40)

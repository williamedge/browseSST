"""
Experiment 21: the latent-T route on a field with FRONTS and a FILAMENT.

Same velocity / source truth as every earlier experiment (the generator's seeds are independent of the initial tracer); the initial T gets two
meandering fronts (half-jump 0.5 K, tanh width 4 km; the second weaker, opposite sign) and a diagonal filament (0.3 K, sigma 3.5 km), which the flow then deforms.
|grad T| rms rises from 3.3e-5 to 6.6e-5 K/m, max 2.5e-4. The 4th-order gradient of the true T is itself unreliable across a front 2-3 pixels wide, so the
primary metric is velocity skill against the analytic u, v, S; skill is also reported on the 20% of scored points with the largest |grad T|.

Per sigma_T: finite-difference baselines on the noisy T (FD2, FD4) and clean-gradient bounds; then for each spatial smoothness combination
(eddy : fast Matern nu) a T-GP fit (split bounds, warm start from the smooth-case fit) and the resolved-gradient plug-in (FD4 / FD2 of the posterior-mean T).
True velocity hyperparameters, oracle iid stage-2 noise, scored on the common interior.
Run: python experiments/21_fronts_filaments.py <sigma_T> [combos "2.5:2.5,1.5:1.5,0.5:0.5"] [maxiter]
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
S = e13.S
from inversion_sst_gp.spacetime.fit import T_KERNEL_SLOTS, fit
from inversion_sst_gp.spacetime.impute import grid_gradients

H = 3600.0
FREE = ['sigma1', 'l1', 'tau1', 'sigma2', 'l2', 'tau2', 'noise', 'lr:mean', 'lr:diurnal']
NAMES = {0.5: '1/2', 1.5: '3/2', 2.5: '5/2'}
FRONT_CFG = dict(nx=14, ny=14, n_hours=48, n_modes=400, front_amp=0.5, front_width=4000.0, filament_amp=0.3, filament_width=3500.0)


def setup(sigma_T):
    cfg = S.SyntheticConfig(**FRONT_CFG)
    ds0 = S.load_or_generate(cfg, ROOT / "data" / "synth_14px_48h_fronts.nc")
    ds = S.reobserve(ds0, sigma_T, seed=cfg.seed + 7)
    tp = cfg.true_params(ZS)
    p_vel = K.pack_params(tp['sigma_psi'], tp['l_psi'], tp['sigma_phi'], tp['l_phi'], tp['tau0_psi'], tp['tau0_phi'], tp['sigma_S'], tp['l_S'], tp['tau0_S'])
    tt, yy, xx = np.meshgrid(ds0['time'].values, ds0['y'].values, ds0['x'].values, indexing='ij')
    flat = dict(x=xx.ravel(), y=yy.ravel(), t=tt.ravel(), u=ds0['u_true'].values.ravel(), v=ds0['v_true'].values.ravel(), S=ds0['S_true'].values.ravel() / ZS)
    ref4 = dict(zip(('dTdt', 'dTdx', 'dTdy'), S.derivative_products(ds0['T_true'].values, float(ds0.attrs['dx']), order=4)))
    common3 = np.isfinite(ref4['dTdt']) & np.isfinite(ref4['dTdx']) & np.isfinite(ref4['dTdy'])
    return cfg, ds0, ds, p_vel, flat, ref4, common3


def score(label, est, truth, hi, extra=""):
    lo = ~hi; al = np.ones_like(hi)
    parts = []
    for f in 'uv':
        m, v = est[f]; e = m - truth[f]
        sk = lambda s: 1 - np.mean(e[s] ** 2) / truth[f][s].var()
        parts.append(f"{f} skill all {sk(al):+.3f} hi-grad {sk(hi):+.3f} lo {sk(lo):+.3f} | cover all {np.mean(np.abs(e) < 1.96 * np.sqrt(v)):.2f} hi {np.mean((np.abs(e) < 1.96 * np.sqrt(v))[hi]):.2f}")
    m, v = est['S']; e = m - truth['S']
    parts.append(f"S skill {1 - np.mean(e ** 2) / truth['S'].var():+.3f}")
    print(f"  {label:<36s} " + " || ".join(parts) + extra, flush=True)


def run_products(flat, common, p_vel, Tx, Ty, Tt, hi, label, nnum_pred=200):
    valid = np.isfinite(Tx) & np.isfinite(Ty) & np.isfinite(Tt)
    st = Stage2_cache(flat, common, p_vel, valid, nnum_pred)
    est = st.run(Tx, Ty, Tt, st.oracle_noise(Tx, Ty, Tt))
    score(label, est, st.truth, hi)


def Stage2_cache(flat, common, p_vel, valid, nnum_pred):
    return e16.Stage2(flat, common, p_vel, nnum=100, nnum_pred=nnum_pred, valid=valid)


def main(sigma_T, combos, maxiter=60, nnum=100, nnum_pred=200):
    cfg, ds0, ds, p_vel, flat, ref4, common3 = setup(sigma_T)
    common = common3.ravel(); shape = common3.shape; dx = float(ds0.attrs['dx'])
    gmag = np.hypot(ref4['dTdx'], ref4['dTdy']).ravel()[common]
    hi = gmag >= np.percentile(gmag, 80)
    print(f"\n=========== fronts + filament, sigma_T = {sigma_T} K ===========", flush=True)
    print(f"scored points {common.sum()}, high-gradient subset {hi.sum()} (|grad T| >= {np.percentile(gmag, 80):.2e} K/m)", flush=True)
    ref = (ref4['dTdx'].ravel(), ref4['dTdy'].ravel(), ref4['dTdt'].ravel())
    ds2, ds4 = e13.e12.with_products(ds, 2), e13.e12.with_products(ds, 4)
    print("\nfinite-difference baselines (oracle iid stage-2 noise):", flush=True)
    for order, dsx in ((2, ds2), (4, ds4)):
        run_products(flat, common, p_vel, dsx['dTdx'].values.ravel(), dsx['dTdy'].values.ravel(), dsx['dTdt'].values.ravel(), hi, f"FD{order}, noisy T")
    for order, dsx in ((2, ds2), (4, ds4)):
        run_products(flat, common, p_vel, dsx['dTdx_clean'].values.ravel(), dsx['dTdy_clean'].values.ravel(), dsx['dTdt'].values.ravel(), hi, f"FD{order}, CLEAN gradients (bound)")

    src = ROOT / "data" / (f"exp15_fit_sigma{sigma_T}.json" if sigma_T in (0.03, 0.1) else f"exp20_fit_sigma{sigma_T}.json")
    w = json.loads(src.read_text())
    pts = G.build_T_points(flat['x'], flat['y'], flat['t'])
    T_obs = ds['T_obs'].values.ravel()
    centre = dict(sigma1=0.5, l1=40e3, tau1=2 * 86400.0, sigma2=0.2, l2=6e3, tau2=6 * H, noise=2 * sigma_T ** 2, **{'lr:mean': 1.0, 'lr:diurnal': 0.1})
    for (n1, n2) in combos:
        label = f"eddy {NAMES[n1]} / fast {NAMES[n2]}"
        p0 = G.pack_params_T_res(w['sigma1'], w['l1'], w['tau1'], w['sigma2'], w['l2'], w['tau2'], 2.5, n1, n2)
        gp = G.TGP(pts, T_obs, p0, w['noise'], lowrank=e13.T_terms(w['lr:mean'], w['lr:diurnal']), nnum=nnum, Lx=20e3, Lt=24 * 3600.0, covfunc=G.T_covfunc_res)
        t0 = time.time()
        res = fit(gp, centre, FREE, bounds=e15.SPLIT_BOUNDS, prior_sd_log={n: 1.5 for n in FREE}, maxiter=maxiter, verbose=False,
                  slots=T_KERNEL_SLOTS, start={k: w[k] for k in FREE})
        r = res.params
        print(f"\n[{label}] fit {res.n_eval} evals, {time.time() - t0:.0f}s, success={res.success}, loglik {res.loglik:.1f}", flush=True)
        print(f"   eddy sigma {r['sigma1']:.3f} K, l {r['l1'] / 1e3:.1f} km, tau {r['tau1'] / H:.1f} h | fast sigma {r['sigma2']:.3f} K, l {r['l2'] / 1e3:.2f} km, "
              f"tau {r['tau2'] / H:.1f} h | noise sd {np.sqrt(r['noise']):.4f} (true {sigma_T}) | lr mean {r['lr:mean']:.3f} diurnal {r['lr:diurnal']:.4f}", flush=True)
        (ROOT / "data" / f"exp21_fit_sigma{sigma_T}_nu{n1}_{n2}.json").write_text(json.dumps({k: float(v) for k, v in r.items()} | {'loglik': float(res.loglik)}, indent=1))
        nn = gp.pred_neighbours(flat['x'], flat['y'], flat['t'], nnum_pred)
        T_mean = gp.predict('T', flat['x'], flat['y'], flat['t'], nn_array=nn)[0]
        Tt_mean = gp.predict('Tt', flat['x'], flat['y'], flat['t'], nn_array=nn)[0]
        T_true = ds0['T_true'].values.ravel()
        print(f"   latent T error: rms {np.sqrt(np.mean((T_mean - T_true) ** 2)):.4f} K (noise sd {sigma_T}), at high-gradient points "
              f"{np.sqrt(np.mean(((T_mean - T_true)[common])[hi] ** 2)):.4f} K", flush=True)
        for order in (4, 2):
            Tx, Ty = grid_gradients(T_mean, shape, dx, order=order)
            errs = []
            for j, (name, a) in enumerate((('Tx', Tx), ('Ty', Ty), ('Tt', Tt_mean))):
                e = (a - ref[j])[common]
                errs.append(f"{name} {np.sqrt(np.mean(e ** 2)) / np.std(ref[j][common]):.2f}/{np.sqrt(np.mean(e[hi] ** 2)) / np.std(ref[j][common][hi]):.2f}")
            run_products(flat, common, p_vel, Tx, Ty, Tt_mean, hi, f"latent T, FD{order} ({label})")
            print(f"      derivative error / std (all / hi-grad) vs FD4 of true T: " + " | ".join(errs), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    spec = a[1] if len(a) > 1 else "2.5:2.5,1.5:1.5,0.5:0.5"
    main(float(a[0]), [tuple(float(v) for v in c.split(':')) for c in spec.split(',')], maxiter=int(a[2]) if len(a) > 2 else 60)

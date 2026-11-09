"""
Experiment 19: spatial smoothness of the two T-GP components (eddy / fast), resolved-gradient mode (fields T and dT/dt only).

For each (nu_s_eddy : nu_s_fast) in {5/2:5/2 (exp-15 reference), 3/2:3/2, 5/2:1/2, 3/2:1/2, 1/2:1/2}: fit the nine free T-GP parameters
(split bounds as exp 15, warm start from exp 15), then report the fitted marginal likelihood (the principled selector: same data and
neighbour structure for every combination), derivative error of the resolved gradients (4th-order centred difference of the posterior-mean T,
and 2nd-order) against the 4th-order gradient of the true T, and the plug-in velocity skill / coverage with oracle iid stage-2 noise.
Run: python experiments/19_smoothness_variants.py <sigma_T> [combos "2.5:2.5,1.5:1.5,..."] [maxiter]
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
NAMES = {0.5: '1/2', 1.5: '3/2', 2.5: '5/2'}


def main(sigma_T, combos, maxiter=60, nnum=100, nnum_pred=200):
    cfg, ds0, ds, p_vel, flat, ref4, common3 = e15.setup(sigma_T)
    common = common3.ravel(); shape = common3.shape; dx = float(ds0.attrs['dx'])
    r15 = json.loads((ROOT / "data" / f"exp15_fit_sigma{sigma_T}.json").read_text())
    pts = G.build_T_points(flat['x'], flat['y'], flat['t'])
    T_obs = ds['T_obs'].values.ravel()
    ref = (ref4['dTdx'].ravel(), ref4['dTdy'].ravel(), ref4['dTdt'].ravel())
    centre = dict(sigma1=0.5, l1=40e3, tau1=2 * 86400.0, sigma2=0.2, l2=6e3, tau2=6 * H,
                  noise=2 * sigma_T ** 2, **{'lr:mean': 1.0, 'lr:diurnal': 0.1})
    print(f"\n=========== smoothness variants, sigma_T = {sigma_T} K ===========", flush=True)
    base_ll = None
    for combo in combos:
        n1, n2 = combo
        label = f"eddy {NAMES[n1]} / fast {NAMES[n2]}"
        p0 = G.pack_params_T_res(r15['sigma1'], r15['l1'], r15['tau1'], r15['sigma2'], r15['l2'], r15['tau2'], 2.5, n1, n2)
        gp = G.TGP(pts, T_obs, p0, r15['noise'], lowrank=e13.T_terms(r15['lr:mean'], r15['lr:diurnal']),
                   nnum=nnum, Lx=20e3, Lt=24 * 3600.0, covfunc=G.T_covfunc_res)
        ll0 = gp.loglik()
        t0 = time.time()
        if (n1, n2) == (2.5, 2.5):
            res_params, ll, n_eval = dict(r15), ll0, 0
        else:
            res = fit(gp, centre, FREE, bounds=e15.SPLIT_BOUNDS, prior_sd_log={n: 1.5 for n in FREE}, maxiter=maxiter,
                      verbose=False, slots=T_KERNEL_SLOTS, start={k: r15[k] for k in FREE})
            res_params, ll, n_eval = res.params, res.loglik, res.n_eval
            print(f"[{label}] fit {n_eval} evals, {time.time() - t0:.0f}s, success={res.success}", flush=True)
        r = res_params
        if base_ll is None and (n1, n2) == (2.5, 2.5):
            base_ll = ll
        print(f"[{label}] loglik {ll:.1f}" + (f" (vs 5/2:5/2 {ll - base_ll:+.1f})" if base_ll is not None else "")
              + f" | start loglik {ll0:.1f}", flush=True)
        print(f"   eddy sigma {r['sigma1']:.3f} K, l {r['l1'] / 1e3:.1f} km, tau {r['tau1'] / H:.1f} h | fast sigma {r['sigma2']:.3f} K, "
              f"l {r['l2'] / 1e3:.2f} km, tau {r['tau2'] / H:.1f} h | noise sd {np.sqrt(r['noise']):.4f} (true {sigma_T}) | "
              f"lr mean {r['lr:mean']:.3f} diurnal {r['lr:diurnal']:.4f}", flush=True)
        out = ROOT / "data" / f"exp19_fit_sigma{sigma_T}_nu{n1}_{n2}.json"
        out.write_text(json.dumps({k: float(v) for k, v in r.items()} | {'loglik': float(ll)}, indent=1))
        # posterior means of T and Tt, resolved gradients
        nn = gp.pred_neighbours(flat['x'], flat['y'], flat['t'], nnum_pred)
        T_mean = gp.predict('T', flat['x'], flat['y'], flat['t'], nn_array=nn)[0]
        Tt_mean = gp.predict('Tt', flat['x'], flat['y'], flat['t'], nn_array=nn)[0]
        for order in (4, 2):
            Tx, Ty = grid_gradients(T_mean, shape, dx, order=order)
            errs = []
            for j, (name, a) in enumerate((('Tx', Tx), ('Ty', Ty), ('Tt', Tt_mean))):
                e = np.sqrt(np.mean((a - ref[j])[common] ** 2))
                errs.append(f"{name} {e:.2e} ({e / np.std(ref[j][common]):.2f})")
            valid = np.isfinite(Tx) & np.isfinite(Ty)
            st = e16.Stage2(flat, common, p_vel, nnum=100, nnum_pred=nnum_pred, valid=valid)
            plug = st.run(Tx, Ty, Tt_mean, st.oracle_noise(Tx, Ty, Tt_mean))
            e16.score(f"   plug-in, FD{order} of latent T", plug, st.truth, extra="   | gradient error " + " | ".join(errs))
    print("done", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    spec = a[1] if len(a) > 1 else "2.5:2.5,1.5:1.5,2.5:0.5,1.5:0.5,0.5:0.5"
    main(float(a[0]), [tuple(float(v) for v in c.split(':')) for c in spec.split(',')], maxiter=int(a[2]) if len(a) > 2 else 60)

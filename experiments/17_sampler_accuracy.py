"""
Experiment 17: accuracy of the Vecchia Matheron sampler (`impute.DerivativeImputer`) at realistic neighbour counts, against a dense GP,
on the 9 x 9 px x 24 h subset (1944 points) at the experiment-13 fitted hyperparameters (sigma_T = 0.03 K). For nnum_prior in the
list: draw K posterior samples of (Tx, Ty, Tt) at all 1944 locations and compare, at 40 random locations, the draw variance with the
dense posterior variance and the sample mean with the dense posterior mean. Run: python experiments/17_sampler_accuracy.py [K] [nnum_prior,...]
"""
import importlib.util, pathlib, sys, time, warnings
import numpy as np

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments" / "13_tgp_plugin.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
G, S = e13.G, e13.S
from inversion_sst_gp.spacetime.impute import DerivativeImputer

H = 3600.0


def main(Kdraws=40, ms=(100, 200)):
    cfg = S.SyntheticConfig(nx=14, ny=14, n_hours=48, n_modes=400)
    ds0 = S.load_or_generate(cfg, ROOT / "data" / "synth_14px_48h_seed1.nc")
    ds = S.reobserve(ds0, 0.03, seed=cfg.seed + 7)
    p = G.pack_params_T(0.183, 12.4e3, 26.8 * H, 0.067, 3.3e3, 9.2 * H, nu_t=2.5)
    noise, ms_, ds_ = 0.0292 ** 2, 0.42, 0.017
    sub = ds.isel(time=slice(0, 24), x=slice(0, 9), y=slice(0, 9))
    tt, yy, xx = np.meshgrid(sub['time'].values, sub['y'].values, sub['x'].values, indexing='ij')
    x, y, t = xx.ravel(), yy.ravel(), tt.ravel()
    pts = G.build_T_points(x, y, t); T = sub['T_obs'].values.ravel(); n = len(T)
    terms = e13.T_terms(ms_, ds_)
    U = np.hstack([tm.basis(pts) for tm in terms])
    C = e13._cross_T(pts, pts, p) + noise * np.eye(n) + U @ U.T
    sel = np.random.default_rng(0).choice(n, 40, replace=False)
    fields = ('Tx', 'Ty', 'Tt')
    D = np.vstack([G.build_T_pred_points(x[sel], y[sel], t[sel], f) for f in fields])
    Ud = np.hstack([tm.basis(D) for tm in terms])
    Kdc = e13._cross_T(D, pts, p) + Ud @ U.T
    Kdd = e13._cross_T(D, D, p) + Ud @ Ud.T
    pm = Kdc @ np.linalg.solve(C, T)
    pc = Kdd - Kdc @ np.linalg.solve(C, Kdc.T)
    sd = np.sqrt(np.diag(pc))
    for m in ms:
        gp = G.TGP(pts, T, p, noise, lowrank=e13.T_terms(ms_, ds_), nnum=100, Lx=20e3, Lt=24 * 3600.0)
        imp = DerivativeImputer(gp, x, y, t, dx=4000.0, dt=3600.0, nnum_prior=m, nnum_pred=200, seed=5)
        t0 = time.time()
        mean, draws = imp.posterior_samples(Kdraws, verbose=False)
        Dd = np.array([np.concatenate([d[f][sel] for f in fields]) for d in draws])
        vm = np.concatenate([mean[f][sel] for f in fields])
        ratio = Dd.var(axis=0, ddof=1) / np.diag(pc)
        print(f"nnum_prior {m:3d} ({time.time() - t0:.0f}s, {Kdraws} draws): Vecchia posterior mean vs dense {np.sqrt(np.mean(((vm - pm) / sd) ** 2)):.3f} sd | "
              f"sample mean vs dense {np.sqrt(np.mean(((Dd.mean(0) - pm) / sd) ** 2)):.3f} sd (sampling noise ~{1 / np.sqrt(Kdraws):.2f}) | "
              f"draw var / dense var: mean {ratio.mean():.2f}; Tx {ratio[:40].mean():.2f} Ty {ratio[40:80].mean():.2f} Tt {ratio[80:].mean():.2f}", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if a else 40, tuple(int(v) for v in a[1].split(',')) if len(a) > 1 else (100, 200))

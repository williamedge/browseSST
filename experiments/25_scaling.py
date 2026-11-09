"""
Experiment 25: cost of the Vecchia machinery versus the number of points N (resource sizing for tiles). One process per size so the peak memory is clean.

T-GP (5/2 : 5/2, the exp-21 sigma_T = 0.03 K fitted hyperparameters, 100 likelihood neighbours, 2 low-rank terms) on a regular (t, y, x) grid, random data (the cost does not depend
on the values). Reports: construction (maximin order + neighbour search + first factor build), one hyperparameter update (= the cost of one likelihood evaluation inside a fit,
the factor rebuild) + loglik, prediction time per 1000 points (Tt, 200 prediction neighbours) and the process peak RSS.
Run: python experiments/25_scaling.py <nx> <ny> <nt>
"""
import importlib.util, json, pathlib, resource, sys, time, warnings
import numpy as np

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / (name + ".py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


e13 = _load("13_tgp_plugin")
G = e13.G


def main(nx, ny, nt, nnum=100, nnum_pred=200, npred=2000):
    r = json.loads((ROOT / "data" / "exp21_fit_sigma0.03_nu2.5_2.5.json").read_text())
    dx = 4000.0
    tt, yy, xx = np.meshgrid(np.arange(nt) * 3600.0, np.arange(ny) * dx, np.arange(nx) * dx, indexing='ij')
    pts = G.build_T_points(xx.ravel(), yy.ravel(), tt.ravel())
    N = len(pts)
    rng = np.random.default_rng(0)
    T = 0.2 * rng.standard_normal(N)
    p0 = G.pack_params_T_res(r['sigma1'], r['l1'], r['tau1'], r['sigma2'], r['l2'], r['tau2'], 2.5, 2.5, 2.5)
    t0 = time.time()
    gp = G.TGP(pts, T, p0, r['noise'], lowrank=e13.T_terms(r['lr:mean'], r['lr:diurnal']), nnum=nnum, Lx=20e3, Lt=24 * 3600.0, covfunc=G.T_covfunc_res)
    t_build = time.time() - t0
    rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    t0 = time.time(); gp.update(cov_params=p0, noise=r['noise']); t_update = time.time() - t0
    t0 = time.time(); ll = gp.loglik(); t_ll = time.time() - t0
    sel = rng.choice(N, min(npred, N), replace=False)
    x, y, t = pts[sel, 0], pts[sel, 1], pts[sel, 2]
    t0 = time.time(); nn = gp.pred_neighbours(x, y, t, nnum_pred); t_nn = time.time() - t0
    t0 = time.time(); gp.predict('Tt', x, y, t, nn_array=nn); t_pred = time.time() - t0
    rss2 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"N = {N:7d} ({nx}x{ny} px x {nt} h): construct (order + neighbours + factor) {t_build:6.1f}s | update (factor rebuild) {t_update:5.1f}s + loglik {t_ll:4.1f}s "
          f"| prediction neighbours {t_nn:5.1f}s + predict {t_pred:5.1f}s for {len(sel)} points ({1000 * t_pred / len(sel):.1f}s per 1000) | peak RSS {rss2:.0f} MB (after build {rss1:.0f}) | loglik {ll:.0f}", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]), int(a[1]), int(a[2]))

"""
Experiment 27: first real-data pass of the loader + stage-1 latent-T GP on a Himawari tile (the 4 hourly pysatview test files, 2025-10-01 00-03 UTC,
111-114E, 25-20S, 0.02 deg). Tests the plumbing (cube -> tile -> metres / seconds / anomaly -> TGP) and asks the real-data questions that the synthetic
runs could not: what noise level and scales does the likelihood choose, and are the leave-out predictions calibrated?

  1. load the cube (stride 2 = 4 km, 'stride' or 'mean' reduction), pick the 28 x 28 px window with the most valid data;
  2. fit sigma1, l1, sigma2, l2, noise and the mean term of the two-component T-GP (5/2 : 5/2, the split bounds of exp 15); the time scales and the diurnal term cannot be learned from
     4 h and are fixed at the synthetic values (27 h, 9 h, small diurnal);
  3. leave-out test with the fitted hyperparameters fixed: (a) 5% of the observed pixels at random, (b) 4 x 4 px blocks removed for one hour at a time (a crude cloud hole);
     standardised residual z = (T - mean) / sqrt(var + noise): calibrated means sd(z) ~ 1 and Gaussian tails.
Run: python experiments/27_real_tile_stage1.py [stride|mean] [maxiter]
"""
import json, pathlib, sys, time, warnings
import numpy as np

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
REAL = ROOT.parent / "pysatview" / "test_data" / "himawari" / "nc"

from inversion_sst_gp.spacetime import himawari as H, tgp as G, tiling as TL
from inversion_sst_gp.spacetime.fit import T_KERNEL_SLOTS, fit

HR = 3600.0
BOUNDS = {'l1': (10e3, 500e3), 'l2': (3e3, 10e3), 'noise': (1e-7, 1.0), 'lr:mean': (1e-3, 1e2), 'sigma1': (1e-3, 1e1), 'sigma2': (1e-3, 1e1)}
TAU1, TAU2 = 27 * HR, 9 * HR                       # not identifiable from 4 h
LR_DIURNAL = 0.01


def terms(mean_scale):
    return [G.TLowRankTerm('mean', G.poly_mean(), mean_scale), G.TLowRankTerm('diurnal', G.diurnal_T(3), LR_DIURNAL)]


def best_window(cube, n=28):
    v = np.isfinite(cube['sst'].values)
    nt, ny, nx = v.shape
    best = (-1, 0, 0)
    for j in range(0, ny - n + 1, 2):
        for i in range(0, nx - n + 1, 2):
            f = v[:, j:j + n, i:i + n].mean(axis=(1, 2)).min()        # worst hour
            if f > best[0]:
                best = (f, j, i)
    return best


def make_gp(x, y, t, T, p, noise, mean_scale):
    p0 = G.pack_params_T_res(p['sigma1'], p['l1'], TAU1, p['sigma2'], p['l2'], TAU2, 2.5, 2.5, 2.5)
    return G.TGP(G.build_T_points(x, y, t), T, p0, noise, lowrank=terms(mean_scale), nnum=100, Lx=20e3, Lt=24 * HR, covfunc=G.T_covfunc_res)


def zscores(td, p, noise, mean_scale, hold):
    """Rebuild the GP without the points `hold` (boolean over the observed points), predict them; returns z."""
    x, y, t, T = td.observed()
    gp = make_gp(x[~hold], y[~hold], t[~hold], T[~hold], p, noise, mean_scale)
    m, v = gp.predict('T', x[hold], y[hold], t[hold], nnum_pred=200)
    return (T[hold] - m) / np.sqrt(v + noise), T[hold] - m, np.sqrt(v + noise)


def describe(name, z, err, sd):
    q = np.quantile(np.abs(z), [0.5, 0.9, 0.99])
    print(f"  {name}: n {z.size}, sd(z) {z.std():.2f}, mean(z) {z.mean():+.2f}, |z| median / 90% / 99% = {q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f} "
          f"(Gaussian 0.67 / 1.64 / 2.58), rmse {np.sqrt(np.mean(err ** 2)):.4f} K, mean predictive sd {sd.mean():.4f} K, kurtosis {np.mean(z ** 4) / np.mean(z ** 2) ** 2:.1f}", flush=True)


def main(reduce='stride', maxiter=60):
    cube = H.load_cube(REAL, t0="2025-10-01T00:00:00", t1="2025-10-01T03:00:00", stride=2, reduce=reduce)
    print(f"cube: {dict(cube.sizes)}, reduce = {reduce}, valid {np.isfinite(cube['sst'].values).mean():.3f}", flush=True)
    f, j, i = best_window(cube)
    spec = TL.TileSpec('real', 0, cube.sizes['time'], j, j + 28, i, i + 28, 0, cube.sizes['time'], j, j + 28, i, i + 28)
    td = TL.extract_tile(cube, spec)
    x, y, t, T = td.observed()
    print(f"tile: lon {td.lon[0]:.2f}..{td.lon[-1]:.2f}, lat {td.lat[0]:.2f}..{td.lat[-1]:.2f}, 28 x 28 px x {td.shape[0]} h, observed {td.n_obs} of {td.obs.size} "
          f"(worst-hour coverage {f:.3f}), spacing {td.hx.mean() / 1e3:.2f} x {td.hy.mean() / 1e3:.2f} km, T range {T.min() + td.Tref:.2f}..{T.max() + td.Tref:.2f} K, sd {T.std():.3f} K", flush=True)

    free = ['sigma1', 'l1', 'sigma2', 'l2', 'noise', 'lr:mean']
    centre = dict(sigma1=0.5, l1=40e3, sigma2=0.1, l2=6e3, noise=0.03 ** 2, **{'lr:mean': 1.0})
    start = dict(sigma1=0.3, l1=20e3, sigma2=0.05, l2=4e3, noise=0.02 ** 2, **{'lr:mean': 0.5})
    p0 = G.pack_params_T_res(centre['sigma1'], centre['l1'], TAU1, centre['sigma2'], centre['l2'], TAU2, 2.5, 2.5, 2.5)
    gp = G.TGP(G.build_T_points(x, y, t), T, p0, centre['noise'], lowrank=terms(centre['lr:mean']), nnum=100, Lx=20e3, Lt=24 * HR, covfunc=G.T_covfunc_res)
    t0 = time.time()
    res = fit(gp, centre, free, bounds=BOUNDS, prior_sd_log={n: 1.5 for n in free}, maxiter=maxiter, verbose=False, slots=T_KERNEL_SLOTS, start=start)
    r = res.params
    print(f"\nfit: {res.n_eval} evals, {time.time() - t0:.0f}s, success={res.success}, loglik {res.loglik:.1f}")
    print(f"  eddy: sigma {r['sigma1']:.3f} K, l {r['l1'] / 1e3:.1f} km | fast: sigma {r['sigma2']:.3f} K, l {r['l2'] / 1e3:.1f} km | noise sd {np.sqrt(r['noise']):.4f} K | mean scale {r['lr:mean']:.3f}", flush=True)
    out = ROOT / "data" / f"exp27_real_tile_{reduce}.json"
    out.write_text(json.dumps({k: float(v) for k, v in r.items()} | {'loglik': float(res.loglik), 'reduce': reduce, 'tile': dict(lon0=float(td.lon[0]), lat0=float(td.lat[0]))}, indent=1))

    rng = np.random.default_rng(0)
    n = x.size
    print("\nleave-out standardised residuals (hyperparameters fixed at the fit):")
    hold = rng.random(n) < 0.05
    describe("random 5% of pixels", *zscores(td, r, r['noise'], r['lr:mean'], hold))
    # 4 x 4 px blocks for one hour at a time
    kk, jj, ii = [a.reshape(td.shape)[td.obs.reshape(td.shape)] for a in np.meshgrid(np.arange(td.shape[0]), np.arange(td.shape[1]), np.arange(td.shape[2]), indexing='ij')]
    hold = np.zeros(n, bool)
    for _ in range(6):
        k0, j0, i0 = rng.integers(0, td.shape[0]), rng.integers(0, 24), rng.integers(0, 24)
        hold |= (kk == k0) & (jj >= j0) & (jj < j0 + 4) & (ii >= i0) & (ii < i0 + 4)
    describe("six 4 x 4 px, 1 h holes", *zscores(td, r, r['noise'], r['lr:mean'], hold))
    # whole hours removed: a time-interpolation test (hour 1 or 2 out)
    for k0 in (1, 2):
        hold = kk == k0
        describe(f"hour {k0} removed entirely", *zscores(td, r, r['noise'], r['lr:mean'], hold))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else 'stride', int(a[1]) if len(a) > 1 else 60)

"""
Experiment 13: Phase B, stage 1 + plug-in. Fit a space-time Vecchia GP to the noisy tracer T, predict dT/dx, dT/dy, dT/dt
at every pixel, and use those in place of finite-difference products in the velocity model (posterior means only: the
derivative uncertainty is NOT propagated yet).

Same synthetic truth as experiments 04-12 (14x14 px @ 4 km, 48 h), re-observed at each sigma_T with NOISY T. Velocity
stage uses the TRUE velocity hyperparameters and the oracle iid noise variance (so only the derivative products differ),
scored on the 4400-point common interior. Reports: derivative accuracy vs a 4th-order reference computed from the true T,
and u / v skill against FD2 / FD4 products from the same noisy T (noisy gradients).
Run: python experiments/13_tgp_plugin.py [sigma_T,...] [maxiter] [nnum] [nnum_pred]
"""
import importlib.util, pathlib, sys, time, warnings
import numpy as np
from numba import njit

from inversion_sst_gp.spacetime import kernels as K
from inversion_sst_gp.spacetime import synthetic as S
from inversion_sst_gp.spacetime import tgp as G
from inversion_sst_gp.spacetime.fit import T_KERNEL_SLOTS, fit
from inversion_sst_gp.spacetime.model import LowRankTerm, SpaceTimeGP, diurnal_harmonics

warnings.filterwarnings("ignore")
ZS = 1e-6
ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, pathlib.Path(__file__).with_name(name + ".py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


e12 = _load("12_stencil_order")


@njit(cache=True)
def _cross_T(X, Y, p):
    out = np.zeros((X.shape[0], Y.shape[0]))
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            out[i, j] = G.T_covfunc(X[i], Y[j], p)
    return out


def T_terms(mean_scale=1.0, diurnal_scale=0.1):
    return [G.TLowRankTerm('mean', G.poly_mean(), mean_scale), G.TLowRankTerm('diurnal', G.diurnal_T(3), diurnal_scale)]


def sanity_vs_dense(ds, p0, noise, nnum, nnum_pred):
    """Vecchia T-GP vs dense GP on a small subset: derivative predictions at 30 random interior points."""
    rng = np.random.default_rng(0)
    sub = ds.isel(time=slice(0, 24), x=slice(0, 9), y=slice(0, 9))
    tt, yy, xx = np.meshgrid(sub['time'].values, sub['y'].values, sub['x'].values, indexing='ij')
    pts = G.build_T_points(xx.ravel(), yy.ravel(), tt.ravel())
    T = sub['T_obs'].values.ravel()
    terms = T_terms()
    U = np.hstack([tm.basis(pts) for tm in terms])
    C = _cross_T(pts, pts, p0) + noise * np.eye(len(T)) + U @ U.T
    alpha = np.linalg.solve(C, T)
    gp = G.TGP(pts, T, p0, noise, lowrank=terms, nnum=nnum, Lx=20e3, Lt=24 * 3600.0)
    sel = rng.choice(len(T), 40, replace=False)
    out = []
    for f in ('Tx', 'Ty', 'Tt'):
        rows = G.build_T_pred_points(pts[sel, 0], pts[sel, 1], pts[sel, 2], f)
        ref = (_cross_T(rows, pts, p0) + np.hstack([tm.basis(rows) for tm in terms]) @ U.T) @ alpha
        m, _ = gp.predict(f, pts[sel, 0], pts[sel, 1], pts[sel, 2], nnum_pred=nnum_pred)
        out.append(f"{f}: rms diff {np.sqrt(np.mean((m - ref) ** 2)):.2e} (ref std {ref.std():.2e})")
    print("  Vecchia vs dense T-GP derivative predictions (subset, start hyperparameters): " + " | ".join(out), flush=True)


def deriv_rmse(label, est, ref, common):
    out = []
    for k, name in zip(('dTdx', 'dTdy', 'dTdt'), ('Tx', 'Ty', 'Tt')):
        e = (est[k] - ref[k])[common]
        out.append(f"{name} {np.sqrt(np.mean(e ** 2)):.2e} ({np.sqrt(np.mean(e ** 2)) / np.std(ref[k][common]):.2f} of std)")
    print(f"  {label:<26s} derivative rmse vs 4th-order(T_true): " + " | ".join(out), flush=True)


def stage2(label, ds, flat, sel, Tx, Ty, Tt, common, p_vel):
    """Velocity stage on derivative products given on the flat cube; `sel` picks the observation points."""
    idx = np.nonzero(sel)[0]
    x, y, t = flat['x'][idx], flat['y'][idx], flat['t'][idx]
    pts = K.build_points(x, y, t, Tx[idx] / ZS, Ty[idx] / ZS)
    z = Tt[idx] / ZS
    tr = {k: flat[k][idx] for k in 'uvS'}
    noise = float(np.var(z - (pts[:, 3] * tr['u'] + pts[:, 4] * tr['v'] + tr['S'])))
    gp = SpaceTimeGP(pts, z, p_vel, noise, lowrank=[LowRankTerm('diurnal', {'S': diurnal_harmonics(4)}, scale=0.5)], nnum=100)
    score = common[idx]
    out = []
    for f in 'uvS':
        m, v = gp.predict(f, x[score], y[score], t[score], nnum_pred=200)
        e = m - tr[f][score]; rmse = np.sqrt(np.mean(e ** 2))
        out.append(f"{f}: rmse {rmse:.4f} skill {1 - rmse**2 / tr[f][score].var():+.3f} cover {np.mean(np.abs(e) < 1.96 * np.sqrt(v)):.2f}")
    print(f"  {label:<34s} N={len(idx):5d} resid var {noise:7.2f} | " + " | ".join(out), flush=True)


def main(sigmas=(0.03, 0.1), maxiter=30, nnum=100, nnum_pred=100):
    cfg = S.SyntheticConfig(nx=14, ny=14, n_hours=48, n_modes=400)
    ds0 = S.load_or_generate(cfg, ROOT / "data" / "synth_14px_48h_seed1.nc")
    tp = cfg.true_params(ZS)
    p_vel = K.pack_params(tp['sigma_psi'], tp['l_psi'], tp['sigma_phi'], tp['l_phi'], tp['tau0_psi'],
                          tp['tau0_phi'], tp['sigma_S'], tp['l_S'], tp['tau0_S'])
    dx = float(ds0.attrs['dx'])
    tt, yy, xx = np.meshgrid(ds0['time'].values, ds0['y'].values, ds0['x'].values, indexing='ij')
    flat = dict(x=xx.ravel(), y=yy.ravel(), t=tt.ravel(), u=ds0['u_true'].values.ravel(), v=ds0['v_true'].values.ravel(),
                S=ds0['S_true'].values.ravel() / ZS)
    ref4 = dict(zip(('dTdt', 'dTdx', 'dTdy'), S.derivative_products(ds0['T_true'].values, dx, order=4)))
    common3 = np.isfinite(ref4['dTdt']) & np.isfinite(ref4['dTdx']) & np.isfinite(ref4['dTdy'])
    common = common3.ravel()

    for sigma_T in sigmas:
        print(f"\n================ sigma_T = {sigma_T} K (noisy T, noisy gradients) ================", flush=True)
        ds = S.reobserve(ds0, sigma_T, seed=cfg.seed + 7)
        T_flat = ds['T_obs'].values.ravel()
        pts = G.build_T_points(flat['x'], flat['y'], flat['t'])
        free = ['sigma1', 'l1', 'tau1', 'sigma2', 'l2', 'tau2', 'noise', 'lr:mean', 'lr:diurnal']
        start = dict(sigma1=0.5, l1=40e3, tau1=2 * 86400.0, sigma2=0.2, l2=15e3, tau2=86400.0,
                     noise=2 * sigma_T ** 2, **{'lr:mean': 1.0, 'lr:diurnal': 0.1})
        p0 = G.pack_params_T(start['sigma1'], start['l1'], start['tau1'], start['sigma2'], start['l2'], start['tau2'], nu_t=2.5)
        if sigma_T == sigmas[0]:
            sanity_vs_dense(ds, p0, start['noise'], nnum, nnum_pred)
        gp = G.TGP(pts, T_flat, p0, start['noise'], lowrank=T_terms(), nnum=nnum, Lx=20e3, Lt=24 * 3600.0)
        t0 = time.time()
        res = fit(gp, start, free, bounds={'noise': (1e-7, 1.0), 'lr:mean': (1e-3, 1e2), 'lr:diurnal': (1e-4, 1e1)},
                  prior_sd_log={n: 1.5 for n in free}, maxiter=maxiter, verbose=False, slots=T_KERNEL_SLOTS)
        r = res.params
        print(f"T-GP fit: {res.n_eval} evals, {time.time() - t0:.0f}s, success={res.success}, loglik {res.loglik:.1f}")
        print(f"  component 1: sigma {r['sigma1']:.3f} K, l {r['l1'] / 1e3:.1f} km, tau {r['tau1'] / 3600:.1f} h | "
              f"component 2: sigma {r['sigma2']:.3f} K, l {r['l2'] / 1e3:.1f} km, tau {r['tau2'] / 3600:.1f} h")
        print(f"  noise sd {np.sqrt(r['noise']):.4f} K (true sigma_T {sigma_T}); mean scale {r['lr:mean']:.2f}, diurnal scale {r['lr:diurnal']:.3f}", flush=True)

        t0 = time.time()
        est = {}
        for f, k in (('Tx', 'dTdx'), ('Ty', 'dTdy'), ('Tt', 'dTdt')):
            m, _ = gp.predict(f, flat['x'], flat['y'], flat['t'], nnum_pred=nnum_pred)
            est[k] = m
        print(f"derivative predictions at {len(flat['x'])} points: {time.time() - t0:.0f}s", flush=True)

        ds2, ds4 = e12.with_products(ds, 2), e12.with_products(ds, 4)
        print("\nderivative accuracy on the common interior:")
        deriv_rmse("FD2 (noisy T)", {k: ds2[k].values.ravel() for k in ('dTdx', 'dTdy', 'dTdt')}, {k: v.ravel() for k, v in ref4.items()}, common)
        deriv_rmse("FD4 (noisy T)", {k: ds4[k].values.ravel() for k in ('dTdx', 'dTdy', 'dTdt')}, {k: v.ravel() for k, v in ref4.items()}, common)
        deriv_rmse("T-GP posterior mean", est, {k: v.ravel() for k, v in ref4.items()}, common)

        print("\nvelocity stage (true velocity hyperparameters, oracle iid noise), scored on the common interior:")
        e12.evaluate("FD2 products, noisy gradients", ds2, 'obs', p_vel, common3, 'obs')
        e12.evaluate("FD4 products, noisy gradients", ds4, 'obs', p_vel, common3, 'obs')
        e12.evaluate("FD2 products, CLEAN gradients (bound)", ds2, 'obs', p_vel, common3, 'clean')
        fd2_valid = np.isfinite(ds2['dTdt'].values).ravel() & np.isfinite(ds2['dTdx'].values).ravel() & np.isfinite(ds2['dTdy'].values).ravel()
        stage2("T-GP plug-in, FD2 point set", ds, flat, fd2_valid, est['dTdx'], est['dTdy'], est['dTdt'], common, p_vel)
        stage2("T-GP plug-in, all points", ds, flat, np.ones_like(fd2_valid), est['dTdx'], est['dTdy'], est['dTdt'], common, p_vel)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(sigmas=tuple(float(v) for v in a[0].split(',')) if a else (0.03, 0.1),
         maxiter=int(a[1]) if len(a) > 1 else 30, nnum=int(a[2]) if len(a) > 2 else 100,
         nnum_pred=int(a[3]) if len(a) > 3 else 100)

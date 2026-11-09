"""
Experiment 30: regional least-squares tide fit (M2, S2, K1, O1; spatially varying amplitude / phase) on the 60-day synthetic of exp 29.

Rows: finite differences of the NOISY (sigma_T = 0.01 K) 4 km snapshots (4th order in time and space), block-averaged 5 x 5 to a 20 km grid (the proposal), or sub-sampled every 5th pixel;
rows shallower than `min_depth` (200 m) dropped. Scored against the analytic truth on the 4 km grid: rms error of the tidal (u, v) relative to the rms tide, per region (deep >= 200 m,
and the shallow shelf where the fit is an extrapolation / masked), amplitude and phase errors per constituent in the deep region, and the leakage / eddy: how much of the TOTAL velocity
variance the fitted tide explains.
Run: python experiments/30_tide_regional_ls.py <scenario> [n_days] [offset_days]
     scenarios: base (20 km blocks, mask 200 m, nuisance H=4), nomask, nonuis, stride, sp50, sp120, noblock (f=2), clean (noise-free T), pool / pool10 (moment-pooled equations in 20 / 40 km blocks), cleanpool
"""
import json, pathlib, sys, time, warnings
import numpy as np
import xarray as xr

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = pathlib.Path.home() / "browsesst_data"

from inversion_sst_gp.spacetime import synthetic as S, tides as TD, tiling as TL

HR = 3600.0
NAMES = ('M2', 'S2', 'K1', 'O1')


def load(n_days=60):
    ds = xr.load_dataset(DATA / f"synth_tide_84px_{n_days}d.nc")
    return ds


def products(ds, key='T_obs'):
    """(Tt, Tx, Ty) of the (noisy by default) snapshots at 4 km, 4th order."""
    T = ds[key].values.astype(np.float64)
    dx = float(ds.attrs['dx'])
    ny, nx = T.shape[1:]
    Tt = TD.time_derivative(T, HR, 4)
    Tx, Ty = TL.grid_gradients_metric(T, T.shape, np.full(ny, dx), np.full(ny, dx), 4)
    return Tt, Tx.reshape(T.shape), Ty.reshape(T.shape)


def truth_harmonics(cfg, x, y):
    """Analytic smooth amplitude / phase of the u component (m/s, rad) at positions, per constituent."""
    tf = S.TideField(cfg)
    h, parts = tf.smooth_parts(x, y)
    return h, parts


def score(fit, ds, cfg, k0, k1, step=4, min_depth=200.0, pexp=0.0):
    x1, y1 = ds['x'].values, ds['y'].values
    depth = ds['depth'].values
    ks = np.arange(k0, k1, step)
    tt = ks * HR
    ut, vt = ds['u_tide_true'].values[ks].astype(float), ds['v_tide_true'].values[ks].astype(float)
    utot, vtot = ds['u_true'].values[ks].astype(float), ds['v_true'].values[ks].astype(float)
    xx, yy = np.meshgrid(x1, y1)
    u = np.empty_like(ut); v = np.empty_like(vt)
    for i, t in enumerate(tt):
        u[i], v[i] = fit.velocity(xx, yy, np.full(xx.shape, t), mask_unsupported=False, scale=TD.depth_scale(depth, pexp))
    out = {}
    for nm, reg in (('deep', depth >= min_depth), ('shelf', depth < min_depth)):
        r = np.broadcast_to(reg, u.shape)
        err = np.sqrt(np.mean(((u - ut) ** 2 + (v - vt) ** 2)[r]))
        rms = np.sqrt(np.mean((ut ** 2 + vt ** 2)[r]))
        tot = np.sqrt(np.mean((utot ** 2 + vtot ** 2)[r]))
        # fraction of the total velocity variance in the region that the fitted tide accounts for, and the remaining (non-tidal + error) rms
        res = np.sqrt(np.mean(((utot - u) ** 2 + (vtot - v) ** 2)[r]))
        out[nm] = dict(tide_err_rms=float(err), tide_rms=float(rms), rel_err=float(err / rms), total_rms=float(tot), residual_rms_after_removal=float(res))
    # amplitude / phase per constituent at the 20 km cell centres of the deep region
    xs = x1[:80].reshape(16, 5).mean(1); ys = y1[:80].reshape(16, 5).mean(1)
    cx, cy = np.meshgrid(xs, ys)
    h, parts = truth_harmonics(cfg, cx.ravel(), cy.ravel())
    hf = fit.harmonics(cx.ravel(), cy.ravel(), scale=TD.depth_scale(h, pexp))
    deep = h >= min_depth
    for nm in NAMES:
        if nm not in hf:
            continue
        A, Phi = parts[nm]
        dA = hf[nm]['Au'] - A
        dphi = np.angle(np.exp(1j * (hf[nm]['phi_u'] - Phi)))
        out[nm] = dict(amp_true_rms=float(np.sqrt(np.mean(A[deep] ** 2))), amp_err_rms=float(np.sqrt(np.mean(dA[deep] ** 2))), phase_err_rms_deg=float(np.degrees(np.sqrt(np.mean(dphi[deep] ** 2)))),
                       phase_err_median_deg=float(np.degrees(np.median(np.abs(dphi[deep])))))
    return out


def main(scenario='base', n_days=30, offset_days=0, total_days=60):
    ds = load(total_days)
    cfg = S.SyntheticConfig(nx=84, ny=84, tide=True, tide_shelf_rough=0.5)
    t_all = time.time()
    Tt, Tx, Ty = products(ds, 'T_true' if scenario.startswith('clean') else 'T_obs')   # (clean: noise-free T)
    k0, k1 = int(offset_days * 24), int((offset_days + n_days) * 24)
    x1, y1, depth = ds['x'].values, ds['y'].values, ds['depth'].values
    pexp = 0.5 if scenario.endswith('D5') else (0.25 if scenario.endswith('D25') else 0.0)
    base_name = scenario[:-2] if scenario.endswith('D5') else (scenario[:-3] if scenario.endswith('D25') else scenario)
    kw = dict(spacing=80e3, min_depth=200.0, nuisance_harmonics=4, n_jobs=1, chunk=20000)
    f = 5
    if base_name == 'nomask':
        kw['min_depth'] = None
    elif base_name == 'nonuis':
        kw['nuisance_harmonics'] = -1
    elif base_name == 'sp50':
        kw['spacing'] = 50e3
    elif base_name == 'sp120':
        kw['spacing'] = 120e3
    elif base_name == 'noblock':
        f = 2
    pool = base_name.startswith('pool') or base_name.startswith('cleanpool')
    if base_name in ('pool10', 'cleanpool10'):
        f = 10
    sl = slice(k0, k1)
    if pexp:                                      # known-bathymetry scaling of the tide: scale the gradients per pixel, the splines then represent the remainder
        sc = TD.depth_scale(depth, pexp)[None]
        Tx, Ty = Tx * sc, Ty * sc
    t1 = np.arange(k0, k1) * HR                    # absolute record time: the truth phases refer to t = 0 at the start of the record
    if pool:
        rr = TD.pool_rows(Tt[sl], Tx[sl], Ty[sl], x1, y1, t1, f, depth=depth)
        kw.update(Cn=rr['Cn'], n_equiv=rr['n_equiv'])
    elif scenario == 'stride':
        rr = TD.rows_from_cube(Tt[sl, 2::5, 2::5], Tx[sl, 2::5, 2::5], Ty[sl, 2::5, 2::5], x1[2::5], y1[2::5], t1, f=1, depth=depth[2::5, 2::5])
    else:
        rr = TD.rows_from_cube(Tt[sl], Tx[sl], Ty[sl], x1, y1, t1, f=f, depth=depth)
    print(f"[{scenario}] window days {offset_days}-{offset_days + n_days}: {rr['x'].size} rows, spacing {kw['spacing'] / 1e3:.0f} km, min_depth {kw['min_depth']}, nuisance H {kw['nuisance_harmonics']}, block {f}", flush=True)
    t0 = time.time()
    fit = TD.fit_tides(rr['x'], rr['y'], rr['t'], rr['Tt'], rr['Tx'], rr['Ty'], depth=rr['depth'], bbox=(x1.min(), x1.max(), y1.min(), y1.max()), **kw)
    print(f"  fit {time.time() - t0:.0f}s: {fit.n_rows} rows used ({fit.info['n_dropped']} dropped), {fit.info['n_params']} parameters, residual rms {fit.resid_rms:.2e} K/s", flush=True)
    sc = score(fit, ds, cfg, k0, k1, pexp=pexp)
    import pickle
    pickle.dump(fit, open(DATA / f"exp30_{scenario}_{n_days}d_off{offset_days}_fit.pkl", "wb"))
    for reg in ('deep', 'shelf'):
        d = sc[reg]
        print(f"  {reg:5s}: tide rms {d['tide_rms']:.3f} m/s, error rms {d['tide_err_rms']:.3f} ({100 * d['rel_err']:.0f}% of the tide); total velocity rms {d['total_rms']:.3f} -> {d['residual_rms_after_removal']:.3f} m/s after removing the fitted tide")
    for nm in NAMES:
        d = sc[nm]
        print(f"  {nm}: amplitude {d['amp_true_rms']:.3f} m/s (rms), error {d['amp_err_rms']:.3f}; phase error rms {d['phase_err_rms_deg']:.1f} deg (median {d['phase_err_median_deg']:.1f})")
    out = ROOT / "data" / f"exp30_{scenario}_{n_days}d_off{offset_days}.json"
    out.write_text(json.dumps(dict(scenario=scenario, n_days=n_days, offset_days=offset_days, n_rows=int(fit.n_rows), n_params=int(fit.info['n_params']), resid_rms=float(fit.resid_rms), score=sc), indent=1))
    print(f"  total {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else 'base', int(a[1]) if len(a) > 1 else 30, int(a[2]) if len(a) > 2 else 0)

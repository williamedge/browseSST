"""
Experiment 14: Vecchia vs dense T-GP derivative predictions at the hyperparameters FITTED in experiment 13 (sigma_T = 0.03 and
0.1 K; values copied from that run's log). 9 x 9 px x 24 h subset (1944 points), dense reference, 40 random points; reports
rms(Vecchia - dense) / std(dense) for Tx, Ty, Tt over (nnum, nnum_pred). Caveat: with only 1944 points, m = 400 covers 20% of
the data, so the large-m rows are optimistic for the full 9408-point cube.
Run: python experiments/14_tgp_vecchia_check.py
"""
import importlib.util, pathlib, warnings, numpy as np
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("e13", ROOT / "experiments" / "13_tgp_plugin.py")
e13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e13)
G, S = e13.G, e13.S

cfg = S.SyntheticConfig(nx=14, ny=14, n_hours=48, n_modes=400)
ds0 = S.load_or_generate(cfg, ROOT / "data" / "synth_14px_48h_seed1.nc")
fitted = {
    0.03: dict(p=(0.183, 12.4e3, 26.8*3600, 0.067, 3.0e3, 9.2*3600), noise=0.0292**2, mean=0.42, diur=0.017),
    0.1:  dict(p=(0.212, 13.6e3, 29.6*3600, 0.066, 3.5e3, 8.1*3600), noise=0.0988**2, mean=0.26, diur=0.005),
}
for sigma_T, f in fitted.items():
    ds = S.reobserve(ds0, sigma_T, seed=cfg.seed + 7)
    p0 = G.pack_params_T(*f['p'], nu_t=2.5)
    terms_fn = lambda: e13.T_terms(f['mean'], f['diur'])
    sub = ds.isel(time=slice(0, 24), x=slice(0, 9), y=slice(0, 9))
    tt, yy, xx = np.meshgrid(sub['time'].values, sub['y'].values, sub['x'].values, indexing='ij')
    pts = G.build_T_points(xx.ravel(), yy.ravel(), tt.ravel()); T = sub['T_obs'].values.ravel()
    terms = terms_fn(); U = np.hstack([tm.basis(pts) for tm in terms])
    C = e13._cross_T(pts, pts, p0) + f['noise'] * np.eye(len(T)) + U @ U.T
    alpha = np.linalg.solve(C, T)
    sel = np.random.default_rng(0).choice(len(T), 40, replace=False)
    refs = {}
    for fld in ('Tx', 'Ty', 'Tt'):
        rows = G.build_T_pred_points(pts[sel, 0], pts[sel, 1], pts[sel, 2], fld)
        refs[fld] = (e13._cross_T(rows, pts, p0) + np.hstack([tm.basis(rows) for tm in terms]) @ U.T) @ alpha
    print(f"\nsigma_T={sigma_T}  FITTED hyperparameters, subset N={len(T)}", flush=True)
    for nnum, nnum_pred in ((100, 100), (100, 200), (200, 200), (400, 400)):
        gp = G.TGP(pts, T, p0, f['noise'], lowrank=terms_fn(), nnum=nnum, Lx=20e3, Lt=24*3600.0)
        out = []
        for fld in ('Tx', 'Ty', 'Tt'):
            m, _ = gp.predict(fld, pts[sel, 0], pts[sel, 1], pts[sel, 2], nnum_pred=nnum_pred)
            out.append(f"{fld} {np.sqrt(np.mean((m - refs[fld])**2)) / refs[fld].std():.3f}")
        print(f"  nnum {nnum:3d} nnum_pred {nnum_pred:3d}: rms diff / dense std: " + " | ".join(out), flush=True)

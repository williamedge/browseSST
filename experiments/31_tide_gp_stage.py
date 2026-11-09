"""
Experiment 31: the sequential pipeline  regional tide fit (exp 30) -> subtract u_bt . grad T -> per-tile GP, on a 28 x 28 px x 48 h tile of the 60-day tide synthetic (deep water, 200 m or deeper).

Stage 1 (latent T, 5/2 : 5/2 two-component GP, hyperparameters REFITTED on the tile because the tide changes the temporal structure of T) gives T and dT/dt at every grid point; gradients are
4th-order differences of the latent T. Stage 2 (Helmholtz GP with the TRUE eddy hyperparameters, oracle equation-error noise of the oracle-tide variant, diurnal S low-rank term) is run on three responses:
  none    : z = dT/dt                       (the tide is left to the GP: the naive baseline)
  fitted  : z = dT/dt + Tx u_fit + Ty v_fit (tide from the regional least-squares fit)
  oracle  : z = dT/dt + Tx u_tide + Ty v_tide (the true tide)
and the total velocity is u_est = u_GP + u_tide(variant). Scored against the TRUE TOTAL velocity (tide + eddies) on interior points; skill = 1 - mse / var(truth), reported against the variance
of the total velocity and of the eddy (non-tidal) part, so that the tide cannot flatter the score.
Run: python experiments/31_tide_gp_stage.py <fit_pickle> [maxiter] [day0] [j0]
"""
import importlib.util, json, pathlib, pickle, sys, time, warnings
import numpy as np
import xarray as xr

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = pathlib.Path.home() / "browsesst_data"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / (name + ".py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


e21 = _load("21_fronts_filaments"); e15 = _load("15_tgp_split_bounds"); e13, G, K, ZS, S = e21.e13, e21.G, e21.K, e21.ZS, e21.S
from inversion_sst_gp.spacetime.fit import T_KERNEL_SLOTS, fit
from inversion_sst_gp.spacetime.impute import grid_gradients
from inversion_sst_gp.spacetime.model import LowRankTerm, SpaceTimeGP, diurnal_harmonics

HR = 3600.0
NX = 28
NT = 48


def tile_data(ds, j0, i0, k0):
    sl = (slice(k0, k0 + NT), slice(j0, j0 + NX), slice(i0, i0 + NX))
    g = lambda name: ds[name].values[sl].astype(np.float64)
    xs, ys = ds['x'].values[i0:i0 + NX].astype(float), ds['y'].values[j0:j0 + NX].astype(float)
    t_abs = (k0 + np.arange(NT)) * HR
    tt, yy, xx = np.meshgrid(t_abs, ys, xs, indexing='ij')
    return dict(T=g('T_obs'), Tclean=g('T_true'), u=g('u_true'), v=g('v_true'), ut=g('u_tide_true'), vt=g('v_tide_true'), S=g('S_true'),
                x=xx.ravel(), y=yy.ravel(), t=tt.ravel(), depth=ds['depth'].values[j0:j0 + NX, i0:i0 + NX])


def stage1(d, maxiter, dx):
    r0 = json.loads((ROOT / "data" / "exp21_fit_sigma0.01_nu2.5_2.5.json").read_text())
    free = ['sigma1', 'l1', 'tau1', 'sigma2', 'l2', 'tau2', 'noise', 'lr:mean', 'lr:diurnal']
    p0 = G.pack_params_T_res(r0['sigma1'], r0['l1'], r0['tau1'], r0['sigma2'], r0['l2'], r0['tau2'], 2.5, 2.5, 2.5)
    pts = G.build_T_points(d['x'], d['y'], d['t'])
    gp = G.TGP(pts, d['T'].ravel(), p0, r0['noise'], lowrank=e13.T_terms(r0['lr:mean'], r0['lr:diurnal']), nnum=100, Lx=20e3, Lt=24 * HR, covfunc=G.T_covfunc_res)
    centre = dict(sigma1=0.5, l1=40e3, tau1=2 * 86400.0, sigma2=0.2, l2=6e3, tau2=6 * HR, noise=2 * 0.01 ** 2, **{'lr:mean': 1.0, 'lr:diurnal': 0.1})
    t0 = time.time()
    res = fit(gp, centre, free, bounds=e15.SPLIT_BOUNDS, prior_sd_log={n: 1.5 for n in free}, maxiter=maxiter, verbose=False, slots=T_KERNEL_SLOTS, start={k: r0[k] for k in free})
    r = res.params
    print(f"  stage-1 fit: {res.n_eval} evals, {time.time() - t0:.0f}s, loglik {res.loglik:.1f}: eddy sigma {r['sigma1']:.3f} K l {r['l1'] / 1e3:.1f} km tau {r['tau1'] / HR:.1f} h | fast sigma {r['sigma2']:.3f} K l {r['l2'] / 1e3:.1f} km tau {r['tau2'] / HR:.1f} h | "
          f"noise sd {np.sqrt(r['noise']):.4f} K (true 0.01)", flush=True)
    t0 = time.time()
    nn = gp.pred_neighbours(d['x'], d['y'], d['t'], 200)
    Tm = gp.predict('T', d['x'], d['y'], d['t'], nn_array=nn)[0]
    Ttm = gp.predict('Tt', d['x'], d['y'], d['t'], nn_array=nn)[0]
    print(f"  stage-1 predictions {time.time() - t0:.0f}s", flush=True)
    return Tm, Ttm, r


def stage2(d, Tx, Ty, z, noise, p_vel, nnum_pred=200):
    valid = np.isfinite(Tx) & np.isfinite(Ty)
    pts = K.build_points(d['x'][valid], d['y'][valid], d['t'][valid], Tx[valid] / ZS, Ty[valid] / ZS)
    gp2 = SpaceTimeGP(pts, z[valid] / ZS, p_vel, noise, lowrank=[LowRankTerm('diurnal', {'S': diurnal_harmonics(4)}, scale=0.5)], nnum=100)
    nn2 = gp2.pred_neighbours(d['x'], d['y'], d['t'], nnum_pred)
    u, vu = gp2.predict('u', d['x'], d['y'], d['t'], nn_array=nn2)
    v, vv = gp2.predict('v', d['x'], d['y'], d['t'], nn_array=nn2)
    return u, v, vu, vv


def main(fit_pickle, maxiter=60, day0=40, j0=44):
    ds = xr.load_dataset(DATA / "synth_tide_84px_60d.nc")
    tfit = pickle.load(open(fit_pickle, "rb"))
    cfg = S.SyntheticConfig(nx=84, ny=84, tide=True)
    dx = float(ds.attrs['dx'])
    # tile: rows j0..j0+27 (default 44: the deep plateau, y 176-284 km; 22: the shelf-break slope, y 88-196 km), columns 20..47; the window starts at day0
    i0, k0 = 20, int(day0 * 24)
    d = tile_data(ds, j0, i0, k0)
    print(f"tile: {NX} x {NX} px x {NT} h at day {day0}, depth {d['depth'].min():.0f}-{d['depth'].max():.0f} m, N = {d['x'].size}; tide rms {np.sqrt(np.mean(d['ut'] ** 2 + d['vt'] ** 2)):.3f} m/s, "
          f"eddy rms {np.sqrt(np.mean((d['u'] - d['ut']) ** 2 + (d['v'] - d['vt']) ** 2)):.3f} m/s", flush=True)
    cache = DATA / f"exp31_stage1_day{day0}_row{j0}.npz"          # stage 1 is the slow part: cache it for exp 32
    if cache.exists():
        z_ = np.load(cache, allow_pickle=True); Tm, Ttm, r1 = z_['Tm'], z_['Ttm'], json.loads(str(z_['params']))
        print("  stage 1 loaded from", cache.name, flush=True)
    else:
        Tm, Ttm, r1 = stage1(d, maxiter, dx)
        np.savez(cache, Tm=Tm, Ttm=Ttm, params=json.dumps({k: float(v) for k, v in r1.items()}))
    sh = (NT, NX, NX)
    Tx, Ty = grid_gradients(Tm, sh, dx, order=4)
    ufit, vfit = tfit.velocity(d['x'], d['y'], d['t'], mask_unsupported=False)
    tp = cfg.true_params(ZS)
    p_vel = K.pack_params(tp['sigma_psi'], tp['l_psi'], tp['sigma_phi'], tp['l_phi'], tp['tau0_psi'], tp['tau0_phi'], tp['sigma_S'], tp['l_S'], tp['tau0_S'])
    ut, vt = d['ut'].ravel(), d['vt'].ravel()
    ue, ve = d['u'].ravel() - ut, d['v'].ravel() - vt                         # eddy (non-tidal) truth
    # oracle equation-error noise of the oracle-tide response (scaled units), as in exps 22 / 24
    zor = Ttm + Tx * ut + Ty * vt
    ok = np.isfinite(zor)
    err = zor[ok] - (-Tx[ok] * ue[ok] - Ty[ok] * ve[ok] + d['S'].ravel()[ok])
    noise = float(np.var(err) / ZS ** 2)
    print(f"  stage-2 noise (oracle equation error of the oracle-tide response) {noise:.2f} (scaled units)", flush=True)
    rim = 3
    score = np.zeros(sh, bool); score[4:NT - 4, rim:NX - rim, rim:NX - rim] = True
    score = score.ravel() & ok
    var_tot = np.var(np.r_[d['u'].ravel()[score], d['v'].ravel()[score]]); var_eddy = np.var(np.r_[ue[score], ve[score]])
    out = dict(stage1=r1, noise=noise, var_total=float(var_tot), var_eddy=float(var_eddy))
    for name, (cu, cv) in dict(none=(0 * ut, 0 * vt), fitted=(ufit, vfit), oracle=(ut, vt)).items():
        t0 = time.time()
        z = Ttm + Tx * cu + Ty * cv
        u, v, vu, vv = stage2(d, Tx, Ty, z, noise, p_vel)
        ue_est, ve_est = u + cu, v + cv                                          # total velocity estimate
        mu = (ue_est - d['u'].ravel())[score]; mv = (ve_est - d['v'].ravel())[score]
        mse = np.mean(np.r_[mu, mv] ** 2)
        mse_u, mse_v = np.mean(mu ** 2), np.mean(mv ** 2)
        # eddy-part check: the GP output alone against the eddy truth when the tide is subtracted first
        res = dict(rmse=float(np.sqrt(mse)), skill_vs_total=float(1 - mse / var_tot), skill_vs_eddy=float(1 - mse / var_eddy),
                   skill_u_eddy=float(1 - mse_u / np.var(ue[score])), skill_v_eddy=float(1 - mse_v / np.var(ve[score])),
                   tide_term_err=float(np.sqrt(np.mean(((cu - ut) ** 2 + (cv - vt) ** 2)[score]))), seconds=time.time() - t0)
        out[name] = res
        print(f"  [{name:6s}] total-velocity rmse {res['rmse']:.4f} m/s (tide rms {np.sqrt(np.mean(ut[score] ** 2 + vt[score] ** 2)):.3f}, eddy rms {np.sqrt(var_eddy):.3f}); skill vs total {res['skill_vs_total']:.3f}, "
              f"vs eddy variance {res['skill_vs_eddy']:.3f} (u {res['skill_u_eddy']:.3f}, v {res['skill_v_eddy']:.3f}); tide-term error {res['tide_term_err']:.4f} m/s; {res['seconds']:.0f}s", flush=True)
    (ROOT / "data" / f"exp31_day{day0}_row{j0}.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], int(a[1]) if len(a) > 1 else 60, int(a[2]) if len(a) > 2 else 40, int(a[3]) if len(a) > 3 else 44)

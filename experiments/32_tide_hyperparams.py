"""
Experiment 32: does removing the fitted tide first protect the ESTIMATED velocity hyperparameters?  (exp 31 used the true eddy hyperparameters in every variant.)

Same tile and cached stage-1 products as exp 31 (28 x 28 px x 48 h, day 20). The stage-2 Helmholtz + S hyperparameters (gamma, f_delta, l_psi, l_phi, tau_psi, tau_phi, sigma_S, l_S, tau_S and the
diurnal scale) are FITTED by the Vecchia marginal likelihood from a generic start (as 'fit A' of exp 24), with the stage-2 noise fixed at the exp-31 value, on the response
  none   : z = dT/dt                         (tide left in the data)
  fitted : z = dT/dt + Tx u_fit + Ty v_fit   (regional tide fit removed first)
and the resulting velocity (+ the tide term of the variant) is scored against the true total velocity, as in exp 31.  The fitted time scales are compared with the truth
(tau_psi 3 d, tau_phi 0.5 d): the point of the test is whether an unmodelled 12 h tide pulls them short.
Run: python experiments/32_tide_hyperparams.py <fit_pickle> <none|fitted|oracle> [maxiter] [day0] [j0]
"""
import importlib.util, json, pathlib, pickle, sys, time, warnings
import numpy as np
import xarray as xr

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / (name + ".py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


e31 = _load("31_tide_gp_stage"); e24 = _load("24_estimated_velocity_hyperparams")
S, K, ZS, DATA = e31.S, e31.K, e31.ZS, e31.DATA
from inversion_sst_gp.spacetime.fit import fit, to_amplitude_params
from inversion_sst_gp.spacetime.impute import grid_gradients
from inversion_sst_gp.spacetime.model import LowRankTerm, SpaceTimeGP, diurnal_harmonics

NX, NT = e31.NX, e31.NT
HR = 3600.0


def main(fit_pickle, variant, maxiter=25, day0=20, j0=22):
    ds = xr.load_dataset(DATA / "synth_tide_84px_60d.nc")
    tfit = pickle.load(open(fit_pickle, "rb"))
    cfg = S.SyntheticConfig(nx=84, ny=84, tide=True)
    dx = float(ds.attrs['dx'])
    i0, k0 = 20, int(day0 * 24)
    d = e31.tile_data(ds, j0, i0, k0)
    z_ = np.load(DATA / f"exp31_stage1_day{day0}_row{j0}.npz", allow_pickle=True)
    Tm, Ttm = z_['Tm'], z_['Ttm']
    sh = (NT, NX, NX)
    Tx, Ty = grid_gradients(Tm, sh, dx, order=4)
    ufit, vfit = tfit.velocity(d['x'], d['y'], d['t'], mask_unsupported=False)
    ut, vt = d['ut'].ravel(), d['vt'].ravel()
    ue, ve = d['u'].ravel() - ut, d['v'].ravel() - vt
    tide = dict(none=(0 * ut, 0 * vt), fitted=(ufit, vfit), oracle=(ut, vt))[variant]
    z = Ttm + Tx * tide[0] + Ty * tide[1]
    ok = np.isfinite(Tx) & np.isfinite(Ty) & np.isfinite(z)
    # the exp-31 noise (oracle equation error of the oracle-tide response) so that the variants differ only in the response
    zor = Ttm + Tx * ut + Ty * vt
    err = zor[ok] - (-Tx[ok] * ue[ok] - Ty[ok] * ve[ok] + d['S'].ravel()[ok])
    noise = float(np.var(err) / ZS ** 2)
    tp = cfg.true_params(ZS)
    truth_amp = to_amplitude_params(dict(sigma_psi=tp['sigma_psi'], l_psi=tp['l_psi'], sigma_phi=tp['sigma_phi'], l_phi=tp['l_phi'], tau0_psi=tp['tau0_psi'], tau0_phi=tp['tau0_phi'],
                                         sigma_S=tp['sigma_S'], l_S=tp['l_S'], tau0_S=tp['tau0_S']))
    start = e24.STARTS['fit A']
    params0 = dict(start); params0['noise'] = noise
    pts = K.build_points(d['x'][ok], d['y'][ok], d['t'][ok], Tx[ok] / ZS, Ty[ok] / ZS)
    gp2 = SpaceTimeGP(pts, z[ok] / ZS, e24.cov_from_amp(start), noise, lowrank=[LowRankTerm('diurnal', {'S': diurnal_harmonics(4)}, scale=start['lr:diurnal'])], nnum=100)
    t0 = time.time()
    res = fit(gp2, params0, e24.FREE, bounds=e24.BOUNDS, prior_sd_log={n: 1.0 for n in e24.FREE}, maxiter=maxiter, verbose=False)
    fitted = res.params
    ratios = {k: float(fitted[k] / truth_amp[k]) for k in e24.PARAMS}
    print(f"[{variant}] fit {res.n_eval} evals {time.time() - t0:.0f}s success={res.success}, loglik {res.loglik:.1f}")
    print("  fitted: " + ", ".join(f"{k} {fitted[k]:.3g}" for k in e24.PARAMS))
    print("  truth : " + ", ".join(f"{k} {truth_amp[k]:.3g}" for k in e24.PARAMS))
    print("  ratio : " + ", ".join(f"{k} {ratios[k]:.2f}" for k in e24.PARAMS), flush=True)
    nn2 = gp2.pred_neighbours(d['x'], d['y'], d['t'], 200)
    u = gp2.predict('u', d['x'], d['y'], d['t'], nn_array=nn2)[0]
    v = gp2.predict('v', d['x'], d['y'], d['t'], nn_array=nn2)[0]
    rim = 3
    score = np.zeros(sh, bool); score[4:NT - 4, rim:NX - rim, rim:NX - rim] = True
    score = score.ravel() & ok
    var_eddy = np.var(np.r_[ue[score], ve[score]])
    mu = (u + tide[0] - d['u'].ravel())[score]; mv = (v + tide[1] - d['v'].ravel())[score]
    mse = np.mean(np.r_[mu, mv] ** 2)
    print(f"  velocity with the FITTED hyperparameters: total rmse {np.sqrt(mse):.4f} m/s, skill vs eddy variance {1 - mse / var_eddy:.3f} (u {1 - np.mean(mu ** 2) / np.var(ue[score]):.3f}, v {1 - np.mean(mv ** 2) / np.var(ve[score]):.3f})")
    (ROOT / "data" / f"exp32_{variant}_day{day0}_row{j0}.json").write_text(json.dumps(dict(fitted={k: float(fitted[k]) for k in e24.FREE}, truth={k: float(truth_amp[k]) for k in e24.PARAMS}, ratio=ratios,
                                                                                    loglik=float(res.loglik), rmse=float(np.sqrt(mse)), skill_vs_eddy=float(1 - mse / var_eddy), n_eval=int(res.n_eval)), indent=1))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], int(a[2]) if len(a) > 2 else 25, int(a[3]) if len(a) > 3 else 20, int(a[4]) if len(a) > 4 else 22)

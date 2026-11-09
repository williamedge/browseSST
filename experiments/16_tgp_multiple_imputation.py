"""
Experiment 16: multiple imputation of the T-GP derivative fields (Matheron-rule posterior draws, `impute.py`), then the velocity
stage on each draw and Rubin pooling. Uses the stage-1 hyperparameters fitted in experiment 15 (data/exp15_fit_sigma<s>.json).

Reports
  * derivative-field calibration: rms(posterior mean - truth) vs rms(draw - posterior mean) (should be ~1 if the sampler is calibrated);
  * velocity skill / 95% coverage for: the plug-in (posterior mean), a single random draw, and the Rubin-pooled estimate,
    for two stage-2 noise choices: 'mle' (noise fitted by marginal likelihood on the plug-in products, then fixed for every draw;
    NOT an oracle) and 'oracle' (empirical residual variance against the true u, v, S, per set of products; a diagnostic).
Rubin: mean = avg_k m_k;  total var = avg_k v_k + (1 + 1/K) var_k(m_k).
Run: python experiments/16_tgp_multiple_imputation.py <sigma_T> [K] [nnum_prior] [nnum_pred] [modes: mle,oracle]
"""
import importlib.util, json, pathlib, sys, time, warnings
import numpy as np
from scipy.optimize import minimize_scalar

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / (name + ".py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


e15 = _load("15_tgp_split_bounds")
e13, G, K, ZS = e15.e13, e15.G, e15.K, e15.ZS
from inversion_sst_gp.spacetime.impute import DerivativeImputer
from inversion_sst_gp.spacetime.model import LowRankTerm, SpaceTimeGP, diurnal_harmonics


class Stage2:
    """Velocity stage on arbitrary derivative products, sharing the ordering / neighbour arrays across calls."""

    def __init__(self, flat, score, p_vel, nnum=100, nnum_pred=200, valid=None):
        """`valid`: optional boolean mask over the flat points that carry products (e.g. the border of NaN gradients is dropped);
        every product array passed in is full length and is subset with it. `score` must lie inside `valid`."""
        self.flat, self.score, self.p_vel, self.nnum = flat, score, p_vel, nnum
        self.idx = np.arange(len(flat['x'])) if valid is None else np.nonzero(valid)[0]
        self.xs, self.ys, self.ts = flat['x'][score], flat['y'][score], flat['t'][score]
        self.truth = {f: flat[f][score] for f in 'uvS'}
        self.nnum_pred = nnum_pred
        self.order = self.nn = self.nn_s = None

    def _gp(self, Tx, Ty, Tt, noise, p=None, lr=0.5):
        """`p`: velocity-kernel parameter vector (default the stored `p_vel`); `lr`: scale of the diurnal low-rank S term."""
        fl, ix = self.flat, self.idx
        pts = K.build_points(fl['x'][ix], fl['y'][ix], fl['t'][ix], Tx[ix] / ZS, Ty[ix] / ZS)
        z = Tt[ix] / ZS
        gp = SpaceTimeGP(pts, z, self.p_vel if p is None else p, noise, lowrank=[LowRankTerm('diurnal', {'S': diurnal_harmonics(4)}, scale=lr)],
                         nnum=self.nnum, order_idx=self.order, nn_array=self.nn)
        if self.order is None:
            self.order, self.nn = gp.op.ord, gp.op.nn_array
            self.nn_s = gp.pred_neighbours(self.xs, self.ys, self.ts, self.nnum_pred)
        return gp, pts, z

    def oracle_noise(self, Tx, Ty, Tt):
        fl, ix = self.flat, self.idx
        a, b = -Tx[ix] / ZS, -Ty[ix] / ZS
        return float(np.var(Tt[ix] / ZS - (a * fl['u'][ix] + b * fl['v'][ix] + fl['S'][ix])))

    def fit_noise(self, Tx, Ty, Tt):
        gp, _, _ = self._gp(Tx, Ty, Tt, 1.0)

        def nll(ln):
            gp.update(noise=float(np.exp(ln)))
            return -gp.loglik()
        r = minimize_scalar(nll, bounds=(np.log(1e-2), np.log(1e3)), method='bounded', options=dict(xatol=0.05))
        return float(np.exp(r.x))

    def run(self, Tx, Ty, Tt, noise, p=None, lr=0.5):
        gp, _, _ = self._gp(Tx, Ty, Tt, noise, p=p, lr=lr)
        out = {}
        for f in 'uvS':
            out[f] = gp.predict(f, self.xs, self.ys, self.ts, nn_array=self.nn_s)
        return out


def score(label, est, truth, extra=""):
    parts = []
    for f in 'uvS':
        m, v = est[f]
        e = m - truth[f]; rmse = np.sqrt(np.mean(e ** 2))
        parts.append(f"{f}: skill {1 - rmse ** 2 / truth[f].var():+.3f} cover {np.mean(np.abs(e) < 1.96 * np.sqrt(v)):.2f}")
    print(f"  {label:<38s} " + " | ".join(parts) + extra, flush=True)


def rubin(runs):
    out = {}
    for f in 'uvS':
        ms = np.array([r[f][0] for r in runs]); vs = np.array([r[f][1] for r in runs])
        Kn = len(runs)
        W, B = vs.mean(0), ms.var(0, ddof=1)
        out[f] = (ms.mean(0), W + (1 + 1.0 / Kn) * B)
        out[f + '_ratio'] = float(np.mean(B) / np.mean(W))
    return out


def main(sigma_T, Kdraws=10, nnum_prior=100, nnum_pred=200, modes=('mle', 'oracle')):
    cfg, ds0, ds, p_vel, flat, ref4, common3 = e15.setup(sigma_T)
    common = common3.ravel()
    fitfile = ROOT / "data" / f"exp15_fit_sigma{sigma_T}.json"
    r = json.loads(fitfile.read_text()) if fitfile.exists() else e15.WARM[sigma_T]
    print(f"\n=========== multiple imputation, sigma_T = {sigma_T} K, K = {Kdraws}, nnum_prior {nnum_prior}, nnum_pred {nnum_pred} ===========", flush=True)
    print(f"stage-1 hyperparameters ({'exp15 fit' if fitfile.exists() else 'exp13 fit, exp15 file missing'}): "
          f"eddy {r['sigma1']:.3f} K, {r['l1'] / 1e3:.1f} km, {r['tau1'] / 3600:.1f} h | fast {r['sigma2']:.3f} K, {r['l2'] / 1e3:.1f} km, "
          f"{r['tau2'] / 3600:.1f} h | noise sd {np.sqrt(r['noise']):.4f}", flush=True)
    pts = G.build_T_points(flat['x'], flat['y'], flat['t'])
    p0 = G.pack_params_T(r['sigma1'], r['l1'], r['tau1'], r['sigma2'], r['l2'], r['tau2'], nu_t=2.5)
    gp = G.TGP(pts, ds['T_obs'].values.ravel(), p0, r['noise'], lowrank=e13.T_terms(r['lr:mean'], r['lr:diurnal']),
               nnum=100, Lx=20e3, Lt=24 * 3600.0)
    t0 = time.time()
    imp = DerivativeImputer(gp, flat['x'], flat['y'], flat['t'], dx=float(ds0.attrs['dx']), dt=3600.0,
                            nnum_prior=nnum_prior, nnum_pred=nnum_pred, seed=11)
    print(f"imputer built in {time.time() - t0:.0f}s", flush=True)
    t0 = time.time()
    mean, draws = imp.posterior_samples(Kdraws)
    print(f"{Kdraws} posterior draws in {time.time() - t0:.0f}s", flush=True)

    print("\nderivative calibration on the common interior (rms vs 4th-order FD of true T):")
    for f, k in (('Tx', 'dTdx'), ('Ty', 'dTdy'), ('Tt', 'dTdt')):
        err = np.sqrt(np.mean((mean[f] - ref4[k].ravel())[common] ** 2))
        spread = np.mean([np.sqrt(np.mean((d[f] - mean[f])[common] ** 2)) for d in draws])
        derr = np.mean([np.sqrt(np.mean((d[f] - ref4[k].ravel())[common] ** 2)) for d in draws])
        print(f"  {f}: error of mean {err:.2e} | draw spread about mean {spread:.2e} | ratio spread/error {spread / err:.2f} | "
              f"error of a single draw {derr:.2e} (expect ~sqrt(2) x error of mean if calibrated)", flush=True)

    st = Stage2(flat, common, p_vel, nnum=100, nnum_pred=nnum_pred)
    print("\nvelocity stage (true velocity hyperparameters), scored on the common interior:")
    fixed_noise = None
    for mode in modes:
        print(f"\n -- stage-2 noise: {mode}", flush=True)
        t0 = time.time()
        if mode == 'mle':
            fixed_noise = st.fit_noise(mean['Tx'], mean['Ty'], mean['Tt'])
            print(f"  fitted stage-2 noise variance {fixed_noise:.2f} (scaled units); oracle residual of the mean products "
                  f"{st.oracle_noise(mean['Tx'], mean['Ty'], mean['Tt']):.2f}; of the draws {np.mean([st.oracle_noise(d['Tx'], d['Ty'], d['Tt']) for d in draws]):.2f}", flush=True)
        nz = (lambda d: fixed_noise) if mode == 'mle' else (lambda d: st.oracle_noise(d['Tx'], d['Ty'], d['Tt']))
        plug = st.run(mean['Tx'], mean['Ty'], mean['Tt'], nz(mean))
        score("plug-in (posterior mean)", plug, st.truth)
        runs = []
        for k, d in enumerate(draws):
            runs.append(st.run(d['Tx'], d['Ty'], d['Tt'], nz(d)))
            if k == 0:
                score("one random draw (draw 1)", runs[0], st.truth)
        pooled = rubin(runs)
        score(f"Rubin-pooled over K = {len(runs)}", pooled, st.truth,
              extra=f"  | between/within var ratio u {pooled['u_ratio']:.2f} v {pooled['v_ratio']:.2f} S {pooled['S_ratio']:.2f}")
        print(f"  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(float(a[0]), Kdraws=int(a[1]) if len(a) > 1 else 10, nnum_prior=int(a[2]) if len(a) > 2 else 100,
         nnum_pred=int(a[3]) if len(a) > 3 else 200, modes=tuple(a[4].split(',')) if len(a) > 4 else ('mle', 'oracle'))

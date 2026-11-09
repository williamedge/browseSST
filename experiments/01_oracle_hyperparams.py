"""
Experiment 01: oracle hyperparameters on the small synthetic test bed.

Question: at the TRUE hyperparameters, how good is the u, v, S posterior from
  (a) dense space-time GP            (exact reference)
  (b) Vecchia space-time GP          (locked neighbour array)
  (c) snapshot-independent baseline  (same spatial kernel, no cross-time correlation)
under three data conditions: clean products / noisy dTdt + clean gradients /
noisy dTdt + noisy gradients (the last violates the 'gradients are exact' assumption).

Noise variance is the oracle residual variance of z - (a u + b v + c S) so every case
is on equal footing. Run: python experiments/01_oracle_hyperparams.py
"""
import time
import warnings
import numpy as np
from numba import njit

from inversion_sst_gp.spacetime import kernels as K
from inversion_sst_gp.spacetime import inference as I
from inversion_sst_gp.spacetime import synthetic as S

warnings.filterwarnings("ignore")
ZS = 1e-6


@njit(cache=True)
def cross_matrix(X, Y, p):
    out = np.zeros((X.shape[0], Y.shape[0]))
    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            out[i, j] = K.transport_obs_covfunc(X[i], Y[j], p)
    return out


def dense_posterior(pts, z, p, noise, Xp):
    C = cross_matrix(pts, pts, p) + noise * np.eye(len(z))
    L = np.linalg.cholesky(C)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, z))
    Kc = cross_matrix(Xp, pts, p)
    mean = Kc @ alpha
    var = np.diag(cross_matrix(Xp, Xp, p)) - np.einsum('ij,ij->i', Kc, np.linalg.solve(L.T, np.linalg.solve(L, Kc.T)).T)
    return mean, var


def score(name, mean, var, truth):
    err = mean - truth
    rmse = np.sqrt(np.mean(err**2))
    cover = np.mean(np.abs(err) < 1.96 * np.sqrt(np.maximum(var, 0)))
    print(f"    {name:<26s} rmse {rmse:.4f} (truth std {truth.std():.4f}, skill {1 - rmse**2 / truth.var():+.3f})  95% cover {cover:.2f}")
    return rmse


def main(nx=12, n_hours=24, nnum=30, Lx=20e3, Lt=6 * 3600.0):
    cfg = S.SyntheticConfig(nx=nx, ny=nx, n_hours=n_hours, n_modes=400, diurnal_shape='cosine', diurnal_amp=5e-7)
    ds = S.generate(cfg)
    print("equation residual of FD products:", {k: round(v, 3) for k, v in S.equation_residual(ds).items()})
    tp = cfg.true_params(ZS)
    p = K.pack_params(tp['sigma_psi'], tp['l_psi'], tp['sigma_phi'], tp['l_phi'], tp['tau0_psi'],
                      tp['tau0_phi'], tp['sigma_S'], tp['l_S'], tp['tau0_S'])
    p_snap = p.copy(); p_snap[[4, 5, 10]] = 1.0     # tau0 -> 1 s: no correlation between hourly snapshots
    # diurnal cycle is a uniform S term; with only the Matern S prior it is unmodelled (low-rank scale comes later)

    cases = [('clean products', 'clean', 'clean'), ('noisy dTdt, clean grads', 'clean', 'obs'),
             ('noisy dTdt, noisy grads', 'obs', 'obs')]
    for title, gsrc, dsrc in cases:
        obs = I.make_observations(ds, grad_source=gsrc, dTdt_source=dsrc, zscale=ZS)
        N = len(obs.z)
        idx = obs.index
        u_t, v_t, S_t = (ds[k].values[idx] for k in ('u_true', 'v_true', 'S_true'))
        resid = obs.z - (obs.pts[:, 3] * u_t + obs.pts[:, 4] * v_t + S_t / ZS)
        noise = float(np.var(resid))
        print(f"\n== {title}: N={N}, oracle noise var {noise:.3f} (scaled), z std {obs.z.std():.2f}")
        x, y, t = obs.pts[:, 0], obs.pts[:, 1], obs.pts[:, 2]
        truths = {'u': u_t, 'v': v_t, 'S': S_t / ZS}

        # (a) dense
        t0 = time.time()
        res = {}
        for f in 'uvS':
            Xp = K.build_pred_points(x, y, t, f)
            res[f] = dense_posterior(obs.pts, obs.z, p, noise, Xp)
        print(f"  dense space-time ({time.time()-t0:.1f}s)")
        for f in 'uvS':
            score(f"dense {f}", *res[f], truths[f])

        # (b) Vecchia space-time, (c) Vecchia snapshot-independent
        for label, pp, Lt_ in (("vecchia space-time", p, Lt), ("snapshot-independent", p_snap, 1.0)):
            t0 = time.time()
            op = I.build_operator(obs.pts, pp, noise, nnum=nnum, Lx=Lx, Lt=Lt_)
            tb = time.time() - t0
            print(f"  {label}: nnum={nnum} build {tb:.1f}s  loglik {I.loglik(op, obs.z):.2f}")
            for f in 'uvS':
                m, v = I.predict(op, obs.z, f, x, y, t)
                score(f"{label[:10]} {f}", m, v, truths[f])
            if label.startswith('vecchia'):
                for f in 'uv':
                    md, _ = res[f]
                    m, _ = I.predict(op, obs.z, f, x, y, t)
                    print(f"      vecchia vs dense {f}: rms diff {np.sqrt(np.mean((m - md)**2)):.4f}")


if __name__ == "__main__":
    main()

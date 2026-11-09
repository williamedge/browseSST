"""
Experiment 02: how many neighbours does the Vecchia approximation need here?

Case: noisy dTdt + clean gradients (the phase-A assumption), true hyperparameters, oracle noise.
Compares to the exact dense GP: log-likelihood difference and posterior-mean agreement for u, v,
and adds a DENSE snapshot-independent baseline (so 'does time help' is dense vs dense).
Run: python experiments/02_nnum_sweep.py
"""
import time
import warnings
import numpy as np

from inversion_sst_gp.spacetime import kernels as K
from inversion_sst_gp.spacetime import inference as I
from inversion_sst_gp.spacetime import synthetic as S

import importlib.util, pathlib
spec = importlib.util.spec_from_file_location("e01", pathlib.Path(__file__).with_name("01_oracle_hyperparams.py"))
e01 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e01)

warnings.filterwarnings("ignore")
ZS = 1e-6


def dense_loglik(pts, z, p, noise):
    C = e01.cross_matrix(pts, pts, p) + noise * np.eye(len(z))
    L = np.linalg.cholesky(C)
    w = np.linalg.solve(L, z)
    return -0.5 * (w @ w) - np.sum(np.log(np.diag(L)))


def main(nx=12, n_hours=24):
    cfg = S.SyntheticConfig(nx=nx, ny=nx, n_hours=n_hours, n_modes=400, diurnal_shape='cosine', diurnal_amp=5e-7)
    ds = S.generate(cfg)
    tp = cfg.true_params(ZS)
    p = K.pack_params(tp['sigma_psi'], tp['l_psi'], tp['sigma_phi'], tp['l_phi'], tp['tau0_psi'],
                      tp['tau0_phi'], tp['sigma_S'], tp['l_S'], tp['tau0_S'])
    p_snap = p.copy(); p_snap[[4, 5, 10]] = 1.0
    obs = I.make_observations(ds, grad_source='clean', dTdt_source='obs', zscale=ZS)
    idx = obs.index
    u_t, v_t, S_t = (ds[k].values[idx] for k in ('u_true', 'v_true', 'S_true'))
    noise = float(np.var(obs.z - (obs.pts[:, 3] * u_t + obs.pts[:, 4] * v_t + S_t / ZS)))
    x, y, t = obs.pts[:, 0], obs.pts[:, 1], obs.pts[:, 2]
    N = len(obs.z)
    print(f"N={N} noise var {noise:.2f}")

    ref = {}
    for name, pp in (("dense space-time", p), ("dense snapshot-indep", p_snap)):
        t0 = time.time()
        ll = dense_loglik(obs.pts, obs.z, pp, noise)
        ref[name] = dict(ll=ll)
        for f, tr in (('u', u_t), ('v', v_t)):
            m, v = e01.dense_posterior(obs.pts, obs.z, pp, noise, K.build_pred_points(x, y, t, f))
            ref[name][f] = m
            rmse = np.sqrt(np.mean((m - tr) ** 2))
            print(f"{name:<22s} {f}: rmse {rmse:.4f} skill {1 - rmse**2/tr.var():+.3f}  cover {np.mean(np.abs(m-tr) < 1.96*np.sqrt(v)):.2f}")
        print(f"{name:<22s} loglik {ll:.2f}  ({time.time()-t0:.0f}s)")

    print("\nVecchia space-time vs dense space-time (Lx=20km, Lt=6h)")
    print(" nnum  nnum_pred   dLL(vec-dense)   rms diff u   rms diff v   rmse u   rmse v   build+pred s")
    for nnum, nnum_pred in ((30, 30), (30, 200), (100, 100), (100, 400), (200, 200), (200, 600)):
        t0 = time.time()
        op = I.build_operator(obs.pts, p, noise, nnum=nnum, Lx=20e3, Lt=6 * 3600.0)
        dll = I.loglik(op, obs.z) - ref["dense space-time"]['ll']
        row = []
        for f, tr in (('u', u_t), ('v', v_t)):
            m, _ = I.predict(op, obs.z, f, x, y, t, nnum_pred=nnum_pred)
            row.append((np.sqrt(np.mean((m - ref["dense space-time"][f]) ** 2)), np.sqrt(np.mean((m - tr) ** 2))))
        print(f" {nnum:4d}  {nnum_pred:8d}   {dll:12.2f}   {row[0][0]:.4f}      {row[1][0]:.4f}      {row[0][1]:.4f}   {row[1][1]:.4f}   {time.time()-t0:.0f}")


if __name__ == "__main__":
    main()

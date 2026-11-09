"""
Experiment 05: does time help once hyperparameters are FITTED? (snapshot-independent baseline)

Same synthetic data, start point, priors and optimiser as experiment 04, but the temporal scales are
fixed at 1 s (no correlation between hourly snapshots; neighbours stay within a snapshot), so the model is the same
Helmholtz + Matern spatial prior applied independently to each snapshot. The low-rank diurnal term is kept (it is
shared across snapshots by construction), so any difference in u, v skill vs experiment 04 is due to temporal correlation.
Run: python experiments/05_fit_snapshot_independent.py [nnum] [maxiter]
"""

import sys, time, warnings
import numpy as np

from inversion_sst_gp.spacetime import kernels as K
from inversion_sst_gp.spacetime import inference as I
from inversion_sst_gp.spacetime import synthetic as S
from inversion_sst_gp.spacetime.fit import fit, params_to_model_args
from inversion_sst_gp.spacetime.model import LowRankTerm, SpaceTimeGP, diurnal_harmonics

warnings.filterwarnings("ignore")
ZS = 1e-6


def report(title, gp, obs, truths, nnum_pred):
    x, y, t = obs.pts[:, 0], obs.pts[:, 1], obs.pts[:, 2]
    out = []
    for f in 'uvS':
        m, v = gp.predict(f, x, y, t, nnum_pred=nnum_pred)
        err = m - truths[f]
        rmse = np.sqrt(np.mean(err ** 2))
        out.append(f"{f}: rmse {rmse:.4f} skill {1 - rmse**2 / truths[f].var():+.3f} cover {np.mean(np.abs(err) < 1.96*np.sqrt(v)):.2f}")
    print(f"  {title:<34s} loglik {gp.loglik():9.1f} | " + " | ".join(out), flush=True)


def main(nnum=100, maxiter=40, nx=14, n_hours=48, nnum_pred=200):
    cfg = S.SyntheticConfig(nx=nx, ny=nx, n_hours=n_hours, n_modes=400)
    t0 = time.time(); ds = S.generate(cfg); print(f"generated in {time.time()-t0:.0f}s", flush=True)
    tp = cfg.true_params(ZS)
    obs = I.make_observations(ds, grad_source='clean', dTdt_source='obs', zscale=ZS)
    idx = obs.index
    truths = {'u': ds['u_true'].values[idx], 'v': ds['v_true'].values[idx], 'S': ds['S_true'].values[idx] / ZS}
    noise = float(np.var(obs.z - (obs.pts[:, 3] * truths['u'] + obs.pts[:, 4] * truths['v'] + truths['S'])))
    print(f"N={len(obs.z)} z std {obs.z.std():.2f} oracle noise var {noise:.2f}; S truth std {truths['S'].std():.3f}", flush=True)

    p_true = K.pack_params(tp['sigma_psi'], tp['l_psi'], tp['sigma_phi'], tp['l_phi'], 1.0, 1.0, tp['sigma_S'], tp['l_S'], 1.0)
    diurnal = lambda scale: LowRankTerm('diurnal', {'S': diurnal_harmonics(4)}, scale=scale)

    print("\nOracle hyperparameters", flush=True)
    t0 = time.time()
    gp0 = SpaceTimeGP(obs.pts, obs.z, p_true, noise, lowrank=[], nnum=nnum, Lt=1.0)
    print(f"  build {time.time()-t0:.1f}s", flush=True)
    t0 = time.time(); gp0.update(cov_params=p_true, noise=noise); print(f"  one update+rebuild {time.time()-t0:.1f}s", flush=True)
    report("M0 Matern S only", gp0, obs, truths, nnum_pred)
    gp1 = SpaceTimeGP(obs.pts, obs.z, p_true, noise, lowrank=[diurnal(0.5)], nnum=nnum, Lt=1.0,
                      order_idx=gp0.op.ord, nn_array=gp0.op.nn_array)
    report("M1 + diurnal low-rank (scale 0.5)", gp1, obs, truths, nnum_pred)

    # fit M2 from a perturbed start
    factors = dict(sigma_psi=0.6, l_psi=1.5, sigma_phi=2.0, l_phi=0.6, tau0_psi=0.5, tau0_phi=2.0,
                   sigma_S=2.0, l_S=0.6, tau0_S=0.5)
    start = {n: tp[n] * f for n, f in factors.items()}
    start['noise'] = noise * 0.5
    start['lr:diurnal'] = 1.5
    for n in ('tau0_psi', 'tau0_phi', 'tau0_S'):
        start[n] = 1.0
    free = [n for n in start if not n.startswith('tau0')]
    gp2 = SpaceTimeGP(obs.pts, obs.z, p_true, noise, lowrank=[diurnal(1.5)], nnum=nnum, Lt=1.0,
                      order_idx=gp0.op.ord, nn_array=gp0.op.nn_array)
    cov, nz, lr = params_to_model_args(start, p_true)
    gp2.update(cov_params=cov, noise=nz, lowrank_scales=lr)
    print("\nFit from perturbed start", flush=True)
    report("M2 start (perturbed)", gp2, obs, truths, nnum_pred)
    res = fit(gp2, start, free, prior_sd_log={n: 1.0 for n in free}, maxiter=maxiter)
    print(f"fit: {res.n_eval} evals, {res.seconds:.0f}s, success={res.success} ({res.message})")
    truth_all = dict(tp); truth_all['noise'] = noise; truth_all['lr:diurnal'] = 0.5
    print(f"  {'param':<12s}{'start':>12s}{'fitted':>12s}{'truth':>12s}{'fit/truth':>10s}")
    for n in free:
        print(f"  {n:<12s}{start[n]:12.4g}{res.params[n]:12.4g}{truth_all[n]:12.4g}{res.params[n]/truth_all[n]:10.2f}")
    report("M2 fitted", gp2, obs, truths, nnum_pred)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(nnum=int(a[0]) if a else 100, maxiter=int(a[1]) if len(a) > 1 else 40)

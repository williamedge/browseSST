"""
Experiment 06: is the space-time hyperparameter drift (exp 04) a sigma-ell ridge?

Same synthetic data, perturbed start (in function space), neighbours and priors as experiment 04. Compares
  A  fit in the potential parameterisation (sigma_psi, l_psi, sigma_phi, l_phi, ...)   [as exp 04]
  B  fit in browseGP's amplitude parameterisation (gamma, f_delta, l_psi, l_phi, ...)  [same prior family:
     log-normal sd 1 on each coordinate, centred on the same start point]
then a finite-difference Laplace covariance of the log-likelihood (flat prior) at B's optimum and at the truth, to show
which parameters (and which combinations) the data actually constrain.
Run: python experiments/06_amplitude_param_and_laplace.py [nnum] [maxiter]
"""
import sys, time, warnings
import numpy as np

from inversion_sst_gp.spacetime import kernels as K
from inversion_sst_gp.spacetime import inference as I
from inversion_sst_gp.spacetime import synthetic as S
from inversion_sst_gp.spacetime.fit import (amplitude_uncertainty, fit, laplace_covariance, params_to_model_args,
                                            to_amplitude_params, to_potential_params)
from inversion_sst_gp.spacetime.model import LowRankTerm, SpaceTimeGP, diurnal_harmonics

warnings.filterwarnings("ignore")
ZS = 1e-6


def skill_line(gp, obs, truths, nnum_pred):
    x, y, t = obs.pts[:, 0], obs.pts[:, 1], obs.pts[:, 2]
    out = []
    for f in 'uvS':
        m, v = gp.predict(f, x, y, t, nnum_pred=nnum_pred)
        rmse = np.sqrt(np.mean((m - truths[f]) ** 2))
        out.append(f"{f} skill {1 - rmse**2 / truths[f].var():+.3f}")
    return f"loglik {gp.loglik():9.1f} | " + " | ".join(out)


def table(title, fitted, truth_pot, free_names):
    print(f"\n{title}")
    f_pot = to_potential_params(fitted)
    f_amp = to_amplitude_params(f_pot); t_amp = to_amplitude_params(truth_pot)
    rows = ['gamma', 'f_delta', 'l_psi', 'l_phi', 'tau0_psi', 'tau0_phi', 'sigma_S', 'l_S', 'tau0_S', 'noise', 'lr:diurnal']
    print(f"  {'param':<12s}{'fitted':>12s}{'truth':>12s}{'fit/truth':>10s}")
    for n in rows:
        print(f"  {n:<12s}{f_amp[n]:12.4g}{t_amp[n]:12.4g}{f_amp[n] / t_amp[n]:10.2f}")
    print(f"  (potential form: sigma_psi {f_pot['sigma_psi']:.4g} [{f_pot['sigma_psi']/truth_pot['sigma_psi']:.2f}x], "
          f"sigma_phi {f_pot['sigma_phi']:.4g} [{f_pot['sigma_phi']/truth_pot['sigma_phi']:.2f}x])")


def laplace_report(label, gp, params, nnum_pred=None):
    free = ['sigma_psi', 'l_psi', 'sigma_phi', 'l_phi', 'tau0_psi', 'tau0_phi', 'sigma_S', 'l_S', 'tau0_S', 'noise', 'lr:diurnal']
    pot = to_potential_params(params)
    t0 = time.time()
    H, cov, evals, evecs = laplace_covariance(gp, pot, free)
    sd = np.sqrt(np.diag(cov))
    print(f"\nLaplace covariance at {label} ({time.time()-t0:.0f}s); min Hessian eigenvalue {evals.min():.3g}")
    print(f"  {'param':<12s}{'sd(log)':>9s}   (factor 1-sigma)")
    for n, s in zip(free, sd):
        print(f"  {n:<12s}{s:9.2f}   x/{np.exp(s):.2f}..x{np.exp(s):.2f}")
    g, sg, f, sf = amplitude_uncertainty(cov, free, pot)
    print(f"  derived: gamma {g:.4f} (sd log {sg:.2f}),  f_delta {f:.3f} (sd {sf:.3f})")
    i, j = free.index('sigma_psi'), free.index('l_psi')
    print(f"  corr(log sigma_psi, log l_psi) = {cov[i, j] / (sd[i] * sd[j]):+.2f}")
    k = free.index('tau0_psi')
    print(f"  corr(log sigma_psi, log tau0_psi) = {cov[i, k] / (sd[i] * sd[k]):+.2f},  corr(log l_psi, log tau0_psi) = {cov[j, k] / (sd[j] * sd[k]):+.2f}")
    w = evecs[:, 0]
    print("  weakest direction (smallest eigenvalue) loadings: " +
          ", ".join(f"{n} {c:+.2f}" for n, c in zip(free, w) if abs(c) > 0.2))


def main(nnum=100, maxiter=40, nx=14, n_hours=48, nnum_pred=200):
    cfg = S.SyntheticConfig(nx=nx, ny=nx, n_hours=n_hours, n_modes=400)
    ds = S.generate(cfg)
    tp = cfg.true_params(ZS)
    obs = I.make_observations(ds, grad_source='clean', dTdt_source='obs', zscale=ZS)
    idx = obs.index
    truths = {'u': ds['u_true'].values[idx], 'v': ds['v_true'].values[idx], 'S': ds['S_true'].values[idx] / ZS}
    noise = float(np.var(obs.z - (obs.pts[:, 3] * truths['u'] + obs.pts[:, 4] * truths['v'] + truths['S'])))
    print(f"N={len(obs.z)}; oracle noise var {noise:.2f}")
    p_true = K.pack_params(tp['sigma_psi'], tp['l_psi'], tp['sigma_phi'], tp['l_phi'], tp['tau0_psi'],
                           tp['tau0_phi'], tp['sigma_S'], tp['l_S'], tp['tau0_S'])
    truth_pot = {**tp, 'noise': noise, 'lr:diurnal': 0.5}
    ta = to_amplitude_params(truth_pot)
    print(f"truth: gamma {ta['gamma']:.4f}, f_delta {ta['f_delta']:.3f}")

    factors = dict(sigma_psi=0.6, l_psi=1.5, sigma_phi=2.0, l_phi=0.6, tau0_psi=0.5, tau0_phi=2.0,
                   sigma_S=2.0, l_S=0.6, tau0_S=0.5)
    start_pot = {n: tp[n] * f for n, f in factors.items()}
    start_pot.update(noise=noise * 0.5, **{'lr:diurnal': 1.5})
    diurnal = lambda scale: LowRankTerm('diurnal', {'S': diurnal_harmonics(4)}, scale=scale)

    gp0 = SpaceTimeGP(obs.pts, obs.z, p_true, noise, lowrank=[], nnum=nnum)
    results = {}
    for label, start in (("A potential (sigma_psi, l_psi, sigma_phi, l_phi)", start_pot),
                         ("B amplitude (gamma, f_delta, l_psi, l_phi)", to_amplitude_params(start_pot))):
        gp = SpaceTimeGP(obs.pts, obs.z, p_true, noise, lowrank=[diurnal(1.5)], nnum=nnum,
                         order_idx=gp0.op.ord, nn_array=gp0.op.nn_array)
        cov, nz, lr = params_to_model_args(start, p_true)
        gp.update(cov_params=cov, noise=nz, lowrank_scales=lr)
        free = list(start)
        t0 = time.time()
        res = fit(gp, start, free, prior_sd_log={n: 1.0 for n in free}, maxiter=maxiter, verbose=False)
        print(f"\n=== {label}: {res.n_eval} evals, {time.time()-t0:.0f}s, success={res.success}")
        table(label, res.params, truth_pot, free)
        print("  " + skill_line(gp, obs, truths, nnum_pred), flush=True)
        results[label[0]] = (gp, res)

    gpB, resB = results['B']
    laplace_report("B optimum", gpB, resB.params)
    # at the truth
    cov, nz, lr = params_to_model_args(truth_pot, p_true)
    gpB.update(cov_params=cov, noise=nz, lowrank_scales=lr)
    laplace_report("TRUTH", gpB, truth_pot)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(nnum=int(a[0]) if a else 100, maxiter=int(a[1]) if len(a) > 1 else 40)

"""
Experiment 11: observation-error models for the finite-difference products (phase A: noisy dTdt, clean gradients).

Same synthetic data as experiment 04 (14x14 px @ 4 km, 48 h, N=6624), same perturbed start, priors (log-normal sd 1)
and locked neighbours. Variants:
  V0  iid noise only                                           (experiment 04)
  V1  + exact hourly-difference noise: lag-2 covariance -noise/2 at the same pixel  (no extra parameters)
  V2  V1 + FD residual nuisance: white in space x Matern-3/2 in time, sigma_R and tau0_R fitted
Reports hyperparameter recovery and u / v / S skill and 95% coverage at the observation points.
Run: python experiments/11_nuisance_models.py [maxiter] [variants e.g. 012]
"""
import sys, time, warnings
import numpy as np

from inversion_sst_gp.spacetime import kernels as K
from inversion_sst_gp.spacetime import inference as I
from inversion_sst_gp.spacetime import synthetic as S
from inversion_sst_gp.spacetime.fit import fit, params_to_model_args, to_amplitude_params, to_potential_params
from inversion_sst_gp.spacetime.model import LowRankTerm, SpaceTimeGP, diurnal_harmonics

warnings.filterwarnings("ignore")
ZS = 1e-6
H = 3600.0


def skills(gp, obs, truths, nnum_pred=200):
    x, y, t = obs.pts[:, 0], obs.pts[:, 1], obs.pts[:, 2]
    out = {}
    for f in 'uvS':
        m, v = gp.predict(f, x, y, t, nnum_pred=nnum_pred)
        err = m - truths[f]
        rmse = np.sqrt(np.mean(err ** 2))
        out[f] = (1 - rmse ** 2 / truths[f].var(), np.mean(np.abs(err) < 1.96 * np.sqrt(v)))
    return out


def fmt(sk):
    return " | ".join(f"{f}: skill {s:+.3f} cover {c:.2f}" for f, (s, c) in sk.items())


def main(maxiter=40, variants='012', nx=14, n_hours=48):
    cfg = S.SyntheticConfig(nx=nx, ny=nx, n_hours=n_hours, n_modes=400)
    ds = S.generate(cfg)
    tp = cfg.true_params(ZS)
    noise_T = (cfg.sigma_T / (np.sqrt(2) * H)) ** 2 / ZS ** 2        # exact variance of the differenced snapshot noise
    obs0 = I.make_observations(ds, grad_source='clean', dTdt_source='obs', zscale=ZS)
    idx = obs0.index
    truths = {'u': ds['u_true'].values[idx], 'v': ds['v_true'].values[idx], 'S': ds['S_true'].values[idx] / ZS}
    oracle_total = float(np.var(obs0.z - (obs0.pts[:, 3] * truths['u'] + obs0.pts[:, 4] * truths['v'] + truths['S'])))
    print(f"N={len(obs0.z)}; iid-equivalent oracle residual var {oracle_total:.2f}; exact differenced-noise var {noise_T:.2f}", flush=True)

    base = K.pack_params(tp['sigma_psi'], tp['l_psi'], tp['sigma_phi'], tp['l_phi'], tp['tau0_psi'],
                         tp['tau0_phi'], tp['sigma_S'], tp['l_S'], tp['tau0_S'])
    factors = dict(sigma_psi=0.6, l_psi=1.5, sigma_phi=2.0, l_phi=0.6, tau0_psi=0.5, tau0_phi=2.0,
                   sigma_S=2.0, l_S=0.6, tau0_S=0.5)
    diurnal = lambda: LowRankTerm('diurnal', {'S': diurnal_harmonics(4)}, scale=1.5)
    gp_ref = SpaceTimeGP(obs0.pts, obs0.z, base, oracle_total, lowrank=[], nnum=100)      # ordering / neighbours shared by all
    results = {}

    for v in variants:
        start = {n: tp[n] * f for n, f in factors.items()}
        start['noise'] = (oracle_total if v == '0' else noise_T) * 0.5
        start['lr:diurnal'] = 1.5
        if v == '0':
            obs, p0, covfunc = obs0, base, None
        else:
            obs = I.make_observations(ds, grad_source='clean', dTdt_source='obs', zscale=ZS, resid=True)
            sigR0 = 0.3 if v == '2' else 0.0
            p0 = K.pack_params_resid(base, sigma_R=sigR0, tau0_R=6 * H, noise_lag2=-0.5 * start['noise'])
            covfunc = K.transport_obs_covfunc_resid
            if v == '2':
                start['sigma_R'], start['tau0_R'] = 0.3, 6 * H
        gp = SpaceTimeGP(obs.pts, obs.z, p0, start['noise'], lowrank=[diurnal()], nnum=100,
                         order_idx=gp_ref.op.ord, nn_array=gp_ref.op.nn_array, covfunc=covfunc)
        cov, nz, lr = params_to_model_args(start, p0)
        gp.update(cov_params=cov, noise=nz, lowrank_scales=lr)
        sk0 = skills(gp, obs, truths)
        t0 = time.time()
        res = fit(gp, start, list(start), prior_sd_log={n: 1.0 for n in start}, maxiter=maxiter, verbose=False)
        sk = skills(gp, obs, truths)
        results[v] = (res, sk)
        name = {'0': 'V0 iid noise', '1': 'V1 + lag-2 differencing noise', '2': 'V2 + lag-2 + FD-residual nuisance'}[v]
        print(f"\n=== {name}: {res.n_eval} evals, {time.time()-t0:.0f}s, success={res.success}, loglik {res.loglik:.1f}", flush=True)
        pf = to_amplitude_params(to_potential_params(res.params))
        pt = to_amplitude_params({**tp, 'noise': 1.0, 'lr:diurnal': 0.5})
        row = lambda n, tv: f"{n} {pf[n]:.4g} ({pf[n] / tv:.2f}x)"
        print("  fit vs truth: " + "; ".join(row(n, pt[n]) for n in ('gamma', 'f_delta', 'l_psi', 'l_phi', 'tau0_psi', 'tau0_phi', 'sigma_S', 'l_S', 'tau0_S')))
        nz_true = oracle_total if v == '0' else noise_T
        extra = f"noise {res.params['noise']:.2f} (exact differenced-noise var {noise_T:.2f})"
        if v == '2':
            extra += f"; sigma_R {res.params['sigma_R']:.3f} tau0_R {res.params['tau0_R'] / H:.1f} h"
        print("  " + extra + f"; diurnal scale {res.params['lr:diurnal']:.3f} (0.5 truth-ish)")
        print("  start:  " + fmt(sk0)); print("  fitted: " + fmt(sk), flush=True)

    print("\nSummary (fitted): u skill / v skill / S skill | cover u v S")
    for v, (res, sk) in results.items():
        print(f"  V{v}: " + " / ".join(f"{sk[f][0]:+.3f}" for f in 'uvS') + " | " + " ".join(f"{sk[f][1]:.2f}" for f in 'uvS') + f" | loglik {res.loglik:.1f}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(maxiter=int(a[0]) if a else 40, variants=a[1] if len(a) > 1 else '012')

"""
Hyperparameter fitting for `SpaceTimeGP`: maximise the Vecchia marginal likelihood (plus
optional log-normal priors) over log-parameters with bounded L-BFGS-B and finite-difference
gradients. The ordering / neighbour array is locked at construction, so each evaluation only
rebuilds the sparse factor.

Parameter names: the nine kernel parameters of `kernels.pack_params`
(sigma_psi, l_psi, sigma_phi, l_phi, tau0_psi, tau0_phi, sigma_S, l_S, tau0_S), the data noise
variance `noise` (scaled units), and `lr:<term name>` for each low-rank term's prior scale.
`nu_t_*` (temporal smoothness) are structural and not fitted.

Amplitude parameterisation (browseGP's, better conditioned than the potentials' sigmas): pass
`gamma` and `f_delta` INSTEAD of `sigma_psi` and `sigma_phi`. With the per-component velocity
variances `e = 5 sigma^2 / (3 l^2)`,  gamma^2 = e_psi + e_phi  and  f_delta = e_phi / gamma^2, so

    sigma_psi = gamma l_psi sqrt(3 (1 - f_delta) / 5),   sigma_phi = gamma l_phi sqrt(3 f_delta / 5).

This removes the sigma-l ridge (velocity variance fixed => sigma proportional to l).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from browseGP.covariance.helmholtz import amplitude_from_sigmas, sigmas_from_amplitude

KERNEL_NAMES = ['sigma_psi', 'l_psi', 'sigma_phi', 'l_phi', 'tau0_psi', 'tau0_phi', 'sigma_S', 'l_S', 'tau0_S']
_KERNEL_SLOTS = dict(zip(KERNEL_NAMES, [0, 1, 2, 3, 4, 5, 8, 9, 10]))
_KERNEL_SLOTS.update(sigma_R=12, tau0_R=13)      # observation-error extension (kernels.pack_params_resid)
_NOISE_LAG2_SLOT = 15

DEFAULT_BOUNDS = {
    'sigma_psi': (1e1, 1e5), 'l_psi': (3e3, 3e5), 'sigma_phi': (1e0, 1e5), 'l_phi': (3e3, 3e5),
    'gamma': (1e-3, 2.0), 'f_delta': (1e-3, 0.95),
    'tau0_psi': (1800.0, 3e6), 'tau0_phi': (1800.0, 3e6),
    'sigma_S': (1e-3, 1e2), 'l_S': (3e3, 3e5), 'tau0_S': (1800.0, 3e6),
    'sigma_R': (1e-3, 1e2), 'tau0_R': (1800.0, 3e6),
    'sigma1': (1e-3, 1e1), 'l1': (3e3, 5e5), 'tau1': (1800.0, 1e7),                  # T-GP (T_KERNEL_SLOTS)
    'sigma2': (1e-3, 1e1), 'l2': (3e3, 5e5), 'tau2': (1800.0, 1e7),
    'noise': (1e-3, 1e3),
}


@dataclass
class FitResult:
    params: dict
    loglik: float
    n_eval: int
    seconds: float
    success: bool
    message: str
    history: list = field(default_factory=list)


def to_amplitude_params(params):
    """Replace `sigma_psi`, `sigma_phi` by `gamma`, `f_delta` (same function, different coordinates)."""
    p = {k: v for k, v in params.items() if k not in ('sigma_psi', 'sigma_phi')}
    p['gamma'], p['f_delta'] = amplitude_from_sigmas(params['sigma_psi'], params['l_psi'],
                                                     params['sigma_phi'], params['l_phi'])
    return {k: float(v) for k, v in p.items()}


def to_potential_params(params):
    """Inverse of `to_amplitude_params`."""
    if 'gamma' not in params:
        return dict(params)
    p = {k: v for k, v in params.items() if k not in ('gamma', 'f_delta')}
    p['sigma_psi'], p['sigma_phi'] = sigmas_from_amplitude(params['gamma'], params['f_delta'],
                                                           params['l_psi'], params['l_phi'])
    return {k: float(v) for k, v in p.items()}


T_KERNEL_SLOTS = dict(sigma1=0, l1=1, tau1=2, sigma2=3, l2=4, tau2=5)      # `tgp.pack_params_T` layout


def params_to_model_args(params, base_cov, slots=None):
    """Merge named params into a `cov_params` vector (keeping structural `nu_t_*` from `base_cov`).

    `slots`: name -> index map of the covariance parameter vector (default: the velocity kernel; `T_KERNEL_SLOTS` for the T-GP).
    """
    slots = _KERNEL_SLOTS if slots is None else slots
    params = to_potential_params(params) if slots is _KERNEL_SLOTS else params
    cov = np.array(base_cov, float)
    for name, slot in slots.items():
        if name in params:
            cov[slot] = params[name]
    # hourly-difference noise: the lag-2 covariance is -noise/2, kept tied to the fitted noise if it started non-zero
    if slots is _KERNEL_SLOTS and len(cov) > _NOISE_LAG2_SLOT and base_cov[_NOISE_LAG2_SLOT] != 0.0 and params.get('noise') is not None:
        cov[_NOISE_LAG2_SLOT] = -0.5 * params['noise']
    if hasattr(slots, 'post'):          # slot maps with derived entries (browseGP amplitudes gamma -> potential sigma = gamma ell sqrt(3/5))
        cov = slots.post(cov, params, np.asarray(base_cov, float))
    lr = {k[3:]: v for k, v in params.items() if k.startswith('lr:')}
    return cov, params.get('noise'), (lr or None)


class _Objective:
    """Negative log marginal likelihood (+ optional log-normal prior) as a function of log-params."""

    def __init__(self, gp, params0, free, prior_sd_log=None, slots=None):
        self.gp, self.params0, self.free, self.slots = gp, dict(params0), list(free), slots
        self.base_cov = np.array(gp.op.covparams, float)
        self.prior_sd_log = prior_sd_log or {}
        self.centre = {n: np.log(params0[n]) for n in free}
        self.n_eval, self.best = 0, np.inf

    def unpack(self, x):
        p = dict(self.params0)
        p.update({n: float(np.exp(v)) for n, v in zip(self.free, x)})
        return p

    def __call__(self, x):
        cov, noise, lr = params_to_model_args(self.unpack(x), self.base_cov, self.slots)
        try:
            self.gp.update(cov_params=cov, noise=noise, lowrank_scales=lr)
            ll = self.gp.loglik()
        except Exception:      # non-PD local system at an extreme trial point
            return 1e12
        pen = sum(0.5 * ((x[i] - self.centre[n]) / self.prior_sd_log[n]) ** 2
                  for i, n in enumerate(self.free) if n in self.prior_sd_log)
        self.n_eval += 1
        self.best = min(self.best, -ll + pen)
        return -ll + pen


def fit(gp, params0, free, bounds=None, prior_sd_log=None, maxiter=40, fd_eps=1e-3, verbose=True, slots=None, start=None):
    """Fit `free` parameters of `gp` starting from `params0` (dict of all parameter values).

    prior_sd_log : dict name -> sd of a log-normal prior centred on `params0[name]` (omit = flat).
    start : optional dict name -> starting value overriding `params0` for the optimiser only (warm start); the priors stay
        centred on `params0`.
    """
    bounds = {**DEFAULT_BOUNDS, **(bounds or {})}
    obj = _Objective(gp, params0, free, prior_sd_log, slots)
    x0 = np.log([(start or {}).get(n, params0[n]) for n in free])
    lb = [np.log(bounds[n][0]) if n in bounds else None for n in free]
    ub = [np.log(bounds[n][1]) if n in bounds else None for n in free]
    hist = []

    def callback(xk):
        hist.append((obj.n_eval, obj.best))
        if verbose:
            print(f"  iter {len(hist):3d}  evals {obj.n_eval:4d}  best objective {obj.best:.3f}", flush=True)

    t0 = time.time()
    res = minimize(obj, x0, method='L-BFGS-B', bounds=list(zip(lb, ub)), callback=callback,
                   options=dict(maxiter=maxiter, eps=fd_eps, ftol=1e-9))
    p = obj.unpack(res.x)
    cov, noise, lr = params_to_model_args(p, obj.base_cov, slots)
    gp.update(cov_params=cov, noise=noise, lowrank_scales=lr)
    return FitResult(params=p, loglik=gp.loglik(), n_eval=obj.n_eval, seconds=time.time() - t0,
                     success=bool(res.success), message=str(res.message), history=hist)


def laplace_covariance(gp, params, free, h=0.05, slots=None):
    """Finite-difference Hessian of the NEGATIVE LOG-LIKELIHOOD (flat prior) in log-parameters at `params`.

    Returns `(H, cov, eigvals, eigvecs)`; `cov = inv(H)` (pseudo-inverse if H is not positive definite), so
    `sqrt(diag(cov))` is the Laplace-approximation sd of each log-parameter: how well the data pin it down,
    independent of the optimiser. Eigenvectors with the smallest eigenvalues are the ridge directions.
    Cost: about 2 n^2 likelihood evaluations.
    """
    obj = _Objective(gp, params, free, slots=slots)
    x0 = np.log([params[n] for n in free])
    n = len(free)
    f0 = obj(x0)
    fp = np.zeros(n); fm = np.zeros(n)
    for i in range(n):
        e = np.zeros(n); e[i] = h
        fp[i], fm[i] = obj(x0 + e), obj(x0 - e)
    H = np.zeros((n, n))
    for i in range(n):
        H[i, i] = (fp[i] - 2 * f0 + fm[i]) / h ** 2
    for i in range(n):
        for j in range(i + 1, n):
            ei = np.zeros(n); ei[i] = h
            ej = np.zeros(n); ej[j] = h
            H[i, j] = H[j, i] = (obj(x0 + ei + ej) - obj(x0 + ei - ej) - obj(x0 - ei + ej) + obj(x0 - ei - ej)) / (4 * h ** 2)
    evals, evecs = np.linalg.eigh(H)
    cov = np.linalg.pinv(H) if evals.min() <= 0 else np.linalg.inv(H)
    return H, cov, evals, evecs


def amplitude_uncertainty(cov, free, params):
    """Delta-method sd of `log gamma` and of `f_delta` from a log-parameter covariance over potential-parameters.

    `free` must contain sigma_psi, l_psi, sigma_phi, l_phi. Returns `(gamma, sd_log_gamma, f_delta, sd_f_delta)`.
    """
    idx = {n: free.index(n) for n in ('sigma_psi', 'l_psi', 'sigma_phi', 'l_phi')}
    gamma, f = amplitude_from_sigmas(params['sigma_psi'], params['l_psi'], params['sigma_phi'], params['l_phi'])
    g_log_e_psi = np.zeros(len(free)); g_log_e_phi = np.zeros(len(free))
    g_log_e_psi[idx['sigma_psi']], g_log_e_psi[idx['l_psi']] = 2.0, -2.0     # e = (5/3) sigma^2 / l^2
    g_log_e_phi[idx['sigma_phi']], g_log_e_phi[idx['l_phi']] = 2.0, -2.0
    g_log_gamma = 0.5 * ((1 - f) * g_log_e_psi + f * g_log_e_phi)           # gamma^2 = e_psi + e_phi
    g_f = f * (1 - f) * (g_log_e_phi - g_log_e_psi)
    return float(gamma), float(np.sqrt(g_log_gamma @ cov @ g_log_gamma)), float(f), float(np.sqrt(g_f @ cov @ g_f))

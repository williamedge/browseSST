"""
Pointwise covariance for the transport-equation observations, in the form
gpvecchia's `VecchiaOperator` expects (`covfunc(x1, x2, cov_params) -> float`, njit).

Each observation of `dT/dt` is a pointwise linear combination of the latent
fields,

    z_i = a_i u(x_i, t_i) + b_i v(x_i, t_i) + c_i S(x_i, t_i) + noise,

with `a = -dTdx`, `b = -dTdy`, `c = 1`. The latent fields are independent
blocks: (u, v) follow the browseGP Helmholtz (rotational + divergent) space-time
velocity tensor, and S is a separable space-time Matern. So

    Cov(z_i, z_j) = a_i a_j Cuu + b_i b_j Cvv + (a_i b_j + b_i a_j) Cuv + c_i c_j K_S.

Augmented point layout: `(x, y, t, a, b, c)`. Units are metres and seconds.
Prediction of a latent quantity uses the same operator with that quantity's
coefficients on the prediction point: u = (1, 0, 0), v = (0, 1, 0), S = (0, 0, 1).
Keep the Vecchia neighbour search on `nn_cols=[0, 1, 2]` (see `COORD_COLS`).

cov_params layout (see `pack_params`):
    (sigma_psi, l_psi, sigma_phi, l_phi, tau0_psi, tau0_phi, nu_t_psi, nu_t_phi,
     sigma_S, l_S, tau0_S, nu_t_S)

Observation-error extension (`transport_obs_covfunc_resid`, 7th point column `r` = 1 on data rows, 0 on prediction
rows, 17 parameters, see `pack_params_resid`). Two terms that act only between DATA rows at the SAME pixel
(identical x and y), because both are white in space:

  * hourly-difference noise: `dTdt = (T[t+dt] - T[t-dt]) / (2 dt)` of independent snapshot noise has covariance
    `noise` at lag 0 and `-noise / 2` at lag `2 dt` (the two differences share a snapshot): `noise_lag2` at lag `lag_s`;
  * finite-difference residual: smooth in time, nearly white in space (synthetic experiments 08/09):
    `sigma_R**2 * Matern_nu(tau / tau0_R)`.
"""
import numpy as np
from numba import njit

# Building blocks are reused from browseGP (njit, inline) rather than re-derived.
from browseGP.covariance.helmholtz_numba import (
    _matern52_AB_njit,
    _scalar_matern_njit,
    _rot_components_njit,
    _div_components_njit,
    temporal_matern_njit,
    temporal_wave_njit,
)

COORD_COLS = np.arange(3)  # x, y, t -- columns used for ordering / neighbour search
N_PARAMS = 12


def pack_params(sigma_psi, l_psi, sigma_phi, l_phi, tau0_psi, tau0_phi,
                sigma_S, l_S, tau0_S, nu_t_psi=1.5, nu_t_phi=0.5, nu_t_S=1.5):
    """Flat `cov_params` array for `transport_obs_covfunc`."""
    return np.array([sigma_psi, l_psi, sigma_phi, l_phi,
                     tau0_psi, tau0_phi, nu_t_psi, nu_t_phi,
                     sigma_S, l_S, tau0_S, nu_t_S], dtype=float)


def build_points(x, y, t, dTdx, dTdy, resid=False):
    """Augmented observation rows `(x, y, t, a, b, c)` for the data vector `dTdt` (+ `r = 1` if `resid`)."""
    x, y, t, dTdx, dTdy = (np.asarray(v, dtype=float).ravel() for v in (x, y, t, dTdx, dTdy))
    cols = [x, y, t, -dTdx, -dTdy, np.ones_like(x)]
    if resid:
        cols.append(np.ones_like(x))
    return np.column_stack(cols)


def build_pred_points(x, y, t, field, resid=False):
    """Prediction rows for 'u', 'v' or 'S' at `(x, y, t)` (+ `r = 0` if `resid`: no observation-error terms)."""
    coeff = {'u': (1.0, 0.0, 0.0), 'v': (0.0, 1.0, 0.0), 'S': (0.0, 0.0, 1.0)}[field]
    x, y, t = (np.asarray(v, dtype=float).ravel() for v in (x, y, t))
    n = x.size
    cols = [x, y, t, np.full(n, coeff[0]), np.full(n, coeff[1]), np.full(n, coeff[2])]
    if resid:
        cols.append(np.zeros(n))
    return np.column_stack(cols)


N_PARAMS_RESID = 17


def pack_params_resid(base, sigma_R=0.0, tau0_R=6 * 3600.0, nu_t_R=1.5, noise_lag2=0.0, lag_s=7200.0):
    """Extend a 12-vector from `pack_params` with the observation-error terms.

    `noise_lag2` is the covariance at lag `lag_s` (for hourly differences of independent snapshots:
    `-0.5 * noise`; `fit.params_to_model_args` keeps it tied to the fitted noise when it starts non-zero).
    """
    return np.concatenate([np.asarray(base, float), [sigma_R, tau0_R, nu_t_R, noise_lag2, lag_s]])


@njit(cache=True)
def transport_obs_covfunc(x1, x2, p):
    """Covariance between two `(x, y, t, a, b, c)` rows. See module docstring."""
    dx = x1[0] - x2[0]
    dy = x1[1] - x2[1]
    tau = x1[2] - x2[2]
    r = np.hypot(dx, dy)

    # (u, v): independent rotational (psi) and divergent (phi) potentials
    _, B_psi, c_psi = _matern52_AB_njit(r, p[0], p[1])
    A_phi, _, c_phi = _matern52_AB_njit(r, p[2], p[3])
    t_psi = temporal_matern_njit(tau, p[4], p[6])
    t_phi = temporal_matern_njit(tau, p[5], p[7])
    Cuu = t_psi * (B_psi + c_psi * dx * dx) + t_phi * (A_phi - c_phi * dx * dx)
    Cvv = t_psi * (B_psi + c_psi * dy * dy) + t_phi * (A_phi - c_phi * dy * dy)
    Cuv = (t_psi * c_psi - t_phi * c_phi) * dx * dy

    # S: separable space-time Matern, independent of (u, v)
    KS = _scalar_matern_njit(r, p[8], p[9]) * temporal_matern_njit(tau, p[10], p[11])

    a1, b1, c1 = x1[3], x1[4], x1[5]
    a2, b2, c2 = x2[3], x2[4], x2[5]
    return a1 * a2 * Cuu + b1 * b2 * Cvv + (a1 * b2 + b1 * a2) * Cuv + c1 * c2 * KS


@njit(cache=True)
def transport_obs_covfunc_resid(x1, x2, p):
    """`transport_obs_covfunc` plus same-pixel observation-error terms. Rows are `(x, y, t, a, b, c, r)`."""
    k = transport_obs_covfunc(x1, x2, p)
    if x1[6] != 0.0 and x2[6] != 0.0 and x1[0] == x2[0] and x1[1] == x2[1]:
        tau = x1[2] - x2[2]
        k += p[12] ** 2 * temporal_matern_njit(tau, p[13], p[14])
        if abs(abs(tau) - p[16]) < 1.0:
            k += p[15]
    return k


# ----------------------------------------------------------------------------------------------------------------------------------
# browseGP's in-situ velocity model as the (u, v) prior of the transport equation (exp 40)
# ----------------------------------------------------------------------------------------------------------------------------------
# cov_params layout of `browsegp_transport_covfunc` (28 entries; a block with amplitude 0 is skipped):
#   [0:4]   rotational Helmholtz block   (sigma_psi, l, tau, nu_t)          Matern-5/2 space x Matern-nu_t time, sigma_psi = gamma l sqrt(3/5)
#   [4:10]  divergent M2 wave            (sigma_phi, l, tau_d, omega0, gamma_env, nu_t)   Matern-5/2 space x decaying cosine, omega0 = M2
#   [10:16] divergent M4 wave            (same, omega0 = 2 x M2)
#   [16:19] fast block                   (gamma_f, l, tau)                 independent u / v, Matern-5/2 space x exp(-|t|/tau)
#   [19:24] inertial block               (gamma_in, l, tau, omega, pol)    circularly polarised decaying oscillation (pol = +1 anticlockwise)
#   [24:28] S                            (sigma_S, l_S, tau0_S, nu_t_S)
# These are the covariance tensors of browseGP/scripts/fit_multiscale_real.py (`covfunc_inertial_fast_m4`), with the transport-equation observation z = a u + b v + c S + noise.
BG_N_PARAMS = 28


class BrowseSlots(dict):
    """`fit` slot map of `browsegp_transport_covfunc`: name -> index of the lengths / times / S (set directly), plus the velocity AMPLITUDES in browseGP's own units (gamma, m/s), which `post` turns into the
    covariance vector's entries AFTER the lengths are set (potential sigma = gamma * ell * sqrt(3/5); the fast block's gamma is its entry itself). Freeing gamma instead of sigma removes the sigma-ell ridge
    (a fixed velocity variance means sigma proportional to ell). A block that was switched off in the starting vector (amplitude 0) stays off whatever the parameter dict holds."""
    AMP = dict(gamma_rot=(0, 1), gamma_div=(4, 5), gamma_m4=(10, 11), gamma_f=(16, None), gamma_in=(19, None))

    def post(self, cov, params, base_cov):
        s35 = np.sqrt(3.0 / 5.0)
        for name, (i, j) in self.AMP.items():
            if base_cov[i] == 0.0:
                cov[i] = 0.0
            elif name in params:
                cov[i] = params[name] * (cov[j] * s35 if j is not None else 1.0)
        return cov


BG_SLOTS = BrowseSlots(ell_rot=1, tau_rot=2, ell_div=5, tau_div=6, ell_m4=11, tau_m4=12, ell_f=17, tau_f=18, ell_in=20, tau_in=21, sigma_S=24, l_S=25, tau0_S=26)


def pack_browsegp(gamma_rot, ell_rot, tau_rot, gamma_div, ell_div, tau_div, omega_m2, gamma_m4=0.0, ell_m4=1e4, tau_m4=1e5, gamma_f=0.0, ell_f=2e3, tau_f=4.6e3,
                  gamma_in=0.0, ell_in=2e4, tau_in=2e5, omega_in=0.0, pol=1.0, sigma_S=1.0, l_S=2e4, tau0_S=3600.0, nu_t_rot=1.5, gamma_env=2.0, nu_t_S=1.5):
    """`cov_params` of `browsegp_transport_covfunc` from browseGP's own parameters (velocity amplitudes gamma in m/s; the potentials' sigma = gamma l sqrt(3/5))."""
    s35 = np.sqrt(3.0 / 5.0)
    return np.array([gamma_rot * ell_rot * s35, ell_rot, tau_rot, nu_t_rot,
                     gamma_div * ell_div * s35, ell_div, tau_div, omega_m2, gamma_env, 0.0,
                     gamma_m4 * ell_m4 * s35, ell_m4, tau_m4, 2.0 * omega_m2, gamma_env, 0.0,
                     gamma_f, ell_f, tau_f,
                     gamma_in, ell_in, tau_in, omega_in, pol,
                     sigma_S, l_S, tau0_S, nu_t_S], dtype=float)


@njit(cache=True)
def browsegp_transport_covfunc(x1, x2, p):
    """Covariance between two `(x, y, t, a, b, c)` rows with the browseGP velocity tensor (rot + M2 wave + M4 wave + fast + inertial) for (u, v) and a Matern space-time S. A 33-entry parameter vector adds
    block weights (see below) for block-by-block posterior-mean predictions."""
    dx = x1[0] - x2[0]
    dy = x1[1] - x2[1]
    tau = x1[2] - x2[2]
    r = np.hypot(dx, dy)
    # optional block weights p[28:33] = (rot, M2 wave, M4 wave, fast, inertial), applied ONLY to the covariance between a prediction row (c = 0) and a data row (c = 1): the posterior-mean
    # contribution of one block to the predicted velocity (data-data covariances are always the full model)
    w0 = 1.0
    w1 = 1.0
    w2 = 1.0
    w3 = 1.0
    w4 = 1.0
    if p.size > 28 and ((x1[5] == 0.0) != (x2[5] == 0.0)):
        w0 = p[28]
        w1 = p[29]
        w2 = p[30]
        w3 = p[31]
        w4 = p[32]
    Cuu = 0.0
    Cvv = 0.0
    Cuv = 0.0          # Cov(u(x1), v(x2))
    Cvu = 0.0          # Cov(v(x1), u(x2))
    if p[0] > 0.0:
        cuu, cvv, cuv = _rot_components_njit(dx, dy, p[0], p[1])
        t = temporal_matern_njit(tau, p[2], p[3]) * w0
        Cuu += cuu * t
        Cvv += cvv * t
        Cuv += cuv * t
        Cvu += cuv * t
    for k in (4, 10):
        if p[k] > 0.0:
            cuu, cvv, cuv = _div_components_njit(dx, dy, p[k], p[k + 1])
            t = temporal_wave_njit(tau, p[k + 2], p[k + 3], p[k + 4], p[k + 5]) * (w1 if k == 4 else w2)
            Cuu += cuu * t
            Cvv += cvv * t
            Cuv += cuv * t
            Cvu += cuv * t
    if p[16] > 0.0:
        z = np.sqrt(5.0) * r / p[17]
        g = w3 * p[16] * p[16] * (1.0 + z + z * z / 3.0) * np.exp(-z) * np.exp(-abs(tau) / p[18])
        Cuu += g
        Cvv += g
    if p[19] > 0.0:
        z = np.sqrt(5.0) * r / p[20]
        g = w4 * p[19] * p[19] * (1.0 + z + z * z / 3.0) * np.exp(-z) * np.exp(-abs(tau) / p[21])
        Cuu += g * np.cos(p[22] * tau)
        Cvv += g * np.cos(p[22] * tau)
        Cuv += -p[23] * g * np.sin(p[22] * tau)
        Cvu += p[23] * g * np.sin(p[22] * tau)
    KS = _scalar_matern_njit(r, p[24], p[25]) * temporal_matern_njit(tau, p[26], p[27])
    a1, b1, c1 = x1[3], x1[4], x1[5]
    a2, b2, c2 = x2[3], x2[4], x2[5]
    return a1 * a2 * Cuu + b1 * b2 * Cvv + a1 * b2 * Cuv + b1 * a2 * Cvu + c1 * c2 * KS

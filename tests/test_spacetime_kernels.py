"""
Checks the projected transport-equation covariance against dense references.

Run with `python tests/test_spacetime_kernels.py` (pytest-compatible, but pytest
is not required). Units are rescaled so that z, a, b, c are O(1): gpvecchia's
default jitter is an absolute 1e-8, so raw K/s units (~1e-6) would be swamped.
"""
import numpy as np

from browseGP.covariance import helmholtz as H
from gpvecchia.vecchia_operator import VecchiaOperator
from inversion_sst_gp.spacetime import kernels as K

DAY = 86400.0
N = 120
SP, LP = 0.1 * 30e3 * np.sqrt(0.6), 30e3
SF, LF = 0.05 * 20e3 * np.sqrt(0.6), 20e3
TAU_PSI, TAU_PHI = 3 * DAY, 0.5 * DAY
SIG_S, L_S, TAU_S = 1.0, 20e3, DAY
NOISE = 0.05 ** 2


def _setup(seed=0):
    rng = np.random.default_rng(seed)
    x, y = rng.uniform(0, 60e3, N), rng.uniform(0, 60e3, N)
    t = rng.uniform(0, 2 * DAY, N)
    a, b = 10 * rng.standard_normal(N), 10 * rng.standard_normal(N)
    pts = K.build_points(x, y, t, -a, -b)
    p = K.pack_params(SP, LP, SF, LF, TAU_PSI, TAU_PHI, SIG_S, L_S, TAU_S)
    X = np.column_stack([x, y, t])
    Bk = H.velocity_covariance(X, X.T, (SP, LP, SF, LF, TAU_PSI, TAU_PHI), nu_t_psi=1.5, nu_t_phi=0.5)
    KS = H.scalar_covariance(X, X.T, (SIG_S, L_S, TAU_S), nu_t=1.5)
    Ha = np.hstack([np.diag(a), np.diag(b)])
    Kref = Ha @ Bk @ Ha.T + KS
    return rng, X, a, b, pts, p, Kref


def test_covfunc_matches_dense():
    _, _, _, _, pts, p, Kref = _setup()
    Kmine = np.array([[K.transport_obs_covfunc(pts[i], pts[j], p) for j in range(N)] for i in range(N)])
    assert np.abs(Kmine - Kref).max() / np.abs(Kref).max() < 1e-12
    assert np.linalg.eigvalsh(Kref).min() > -1e-10


def test_vecchia_loglik_matches_dense_at_full_neighbours():
    rng, _, _, _, pts, p, Kref = _setup()
    C = Kref + NOISE * np.eye(N)
    z = rng.multivariate_normal(np.zeros(N), C)
    ll_dense = -0.5 * (z @ np.linalg.solve(C, z) + np.linalg.slogdet(C)[1])  # no 2*pi constant
    op = VecchiaOperator(pts, K.transport_obs_covfunc, p, noise=NOISE, nnum=N,
                         nn_cols=K.COORD_COLS, jitter=1e-10)
    assert abs(float(np.squeeze(op.loglik(z))) - ll_dense) < 1e-8


def test_predict_u_v_S_matches_dense_posterior():
    rng, X, a, b, pts, p, Kref = _setup()
    C = Kref + NOISE * np.eye(N)
    z = rng.multivariate_normal(np.zeros(N), C)
    alpha = np.linalg.solve(C, z)
    op = VecchiaOperator(pts, K.transport_obs_covfunc, p, noise=NOISE, nnum=N,
                         nn_cols=K.COORD_COLS, jitter=1e-10)

    Np = 15
    xs, ys = rng.uniform(10e3, 50e3, Np), rng.uniform(10e3, 50e3, Np)
    ts = rng.uniform(0.3 * DAY, 1.7 * DAY, Np)
    Xs = np.column_stack([xs, ys, ts])
    Bc = H.velocity_covariance(Xs, X.T, (SP, LP, SF, LF, TAU_PSI, TAU_PHI), nu_t_psi=1.5, nu_t_phi=0.5)
    cross = {
        'u': Bc[:Np, :N] * a[None, :] + Bc[:Np, N:] * b[None, :],
        'v': Bc[Np:, :N] * a[None, :] + Bc[Np:, N:] * b[None, :],
        'S': H.scalar_covariance(Xs, X.T, (SIG_S, L_S, TAU_S), nu_t=1.5),
    }
    for name, Kc in cross.items():
        mean, var = op.predict_local(z, K.build_pred_points(xs, ys, ts, name))
        assert np.abs(mean - Kc @ alpha).max() < 1e-8, name
        assert var.min() > 0, name


if __name__ == "__main__":
    for fn in (test_covfunc_matches_dense,
               test_vecchia_loglik_matches_dense_at_full_neighbours,
               test_predict_u_v_S_matches_dense_posterior):
        fn()
        print("ok", fn.__name__)

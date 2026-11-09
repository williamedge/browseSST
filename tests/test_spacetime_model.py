"""
SpaceTimeGP (Vecchia block + low-rank diurnal term) against a dense GP with the same prior,
at full neighbours where the Vecchia approximation is exact. Run: python tests/test_spacetime_model.py
"""
import numpy as np

from inversion_sst_gp.spacetime import kernels as K
from inversion_sst_gp.spacetime.model import LowRankTerm, SpaceTimeGP, diurnal_harmonics

DAY = 86400.0
N = 100


def _dense_cross(X, Y, p):
    return np.array([[K.transport_obs_covfunc(a, b, p) for b in Y] for a in X])


def test_lowrank_matches_dense():
    rng = np.random.default_rng(3)
    x, y = rng.uniform(0, 50e3, N), rng.uniform(0, 50e3, N)
    t = rng.uniform(0, 2 * DAY, N)
    a, b = 10 * rng.standard_normal(N), 10 * rng.standard_normal(N)
    pts = K.build_points(x, y, t, -a, -b)
    p = K.pack_params(0.1 * 30e3 * 0.77, 30e3, 0.03 * 15e3 * 0.77, 15e3, 3 * DAY, 0.5 * DAY, 0.3, 20e3, DAY)
    noise = 0.1
    term = LowRankTerm('diurnal', {'S': diurnal_harmonics(3)}, scale=0.7)

    C = _dense_cross(pts, pts, p) + noise * np.eye(N)
    U = term.basis(pts)
    Cfull = C + U @ U.T
    z = rng.multivariate_normal(np.zeros(N), Cfull)

    gp = SpaceTimeGP(pts, z, p, noise, lowrank=[term], nnum=N, Lx=20e3, Lt=DAY, jitter=1e-10)
    ll_dense = -0.5 * (z @ np.linalg.solve(Cfull, z) + np.linalg.slogdet(Cfull)[1] + N * np.log(2 * np.pi))
    assert abs(gp.loglik() - ll_dense) < 1e-6, (gp.loglik(), ll_dense)

    Np = 12
    xs, ys, ts = rng.uniform(10e3, 40e3, Np), rng.uniform(10e3, 40e3, Np), rng.uniform(0.3 * DAY, 1.7 * DAY, Np)
    alpha = np.linalg.solve(Cfull, z)
    for fld in 'uvS':
        rows = K.build_pred_points(xs, ys, ts, fld)
        cross = _dense_cross(rows, pts, p) + term.basis(rows) @ U.T
        mean, var = gp.predict(fld, xs, ys, ts)
        assert np.abs(mean - cross @ alpha).max() < 1e-6, fld
        assert var.min() > 0

    # hyperparameter update reuses the locked neighbours and matches a fresh build
    p2 = p.copy(); p2[1] *= 1.3
    gp.update(cov_params=p2, noise=0.2, lowrank_scales={'diurnal': 1.1})
    C2 = _dense_cross(pts, pts, p2) + 0.2 * np.eye(N)
    U2 = term.basis(pts)
    C2f = C2 + U2 @ U2.T
    ll2 = -0.5 * (z @ np.linalg.solve(C2f, z) + np.linalg.slogdet(C2f)[1] + N * np.log(2 * np.pi))
    assert abs(gp.loglik() - ll2) < 1e-6


if __name__ == "__main__":
    test_lowrank_matches_dense()
    print("ok test_lowrank_matches_dense")

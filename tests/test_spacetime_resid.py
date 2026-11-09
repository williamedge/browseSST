"""
Observation-error terms (same-pixel lag-2 differencing noise + white-in-space/Matern-in-time FD residual).
Run: python tests/test_spacetime_resid.py
"""
import numpy as np

from inversion_sst_gp.spacetime import fit as F
from inversion_sst_gp.spacetime import kernels as K
from inversion_sst_gp.spacetime.model import LowRankTerm, SpaceTimeGP, diurnal_harmonics

H = 3600.0
BASE = K.pack_params(2000, 30e3, 350, 15e3, 2.6e5, 4.3e4, 0.3, 2e4, 8.6e4)


def _grid(nx=4, nt=8, seed=0):
    """Small regular space-time grid with dropouts (like cloud) so some pixels miss some hours."""
    rng = np.random.default_rng(seed)
    xs = np.arange(nx) * 4000.0
    X, Y, T = np.meshgrid(xs, xs, np.arange(nt) * H, indexing='ij')
    keep = rng.uniform(size=X.shape) > 0.2
    x, y, t = X[keep], Y[keep], T[keep]
    n = x.size
    return x, y, t, 5 * rng.standard_normal(n), 5 * rng.standard_normal(n), rng


def _dense(pts, p, covfunc):
    return np.array([[covfunc(a, b, p) for b in pts] for a in pts])


def test_reduces_to_base_when_terms_off():
    x, y, t, a, b, _ = _grid()
    p = K.pack_params_resid(BASE, sigma_R=0.0, noise_lag2=0.0)
    pts7 = K.build_points(x, y, t, a, b, resid=True)
    pts6 = K.build_points(x, y, t, a, b)
    K7 = _dense(pts7, p, K.transport_obs_covfunc_resid)
    K6 = _dense(pts6, BASE, K.transport_obs_covfunc)
    assert np.abs(K7 - K6).max() < 1e-12


def test_terms_structure_and_psd():
    x, y, t, a, b, _ = _grid()
    noise = 4.0
    p = K.pack_params_resid(BASE, sigma_R=0.5, tau0_R=6 * H, noise_lag2=-0.5 * noise)
    pts = K.build_points(x, y, t, a, b, resid=True)
    Kfull = _dense(pts, p, K.transport_obs_covfunc_resid)
    Kbase = _dense(pts, K.pack_params_resid(BASE), K.transport_obs_covfunc_resid)
    D = Kfull - Kbase
    same_pix = (x[:, None] == x[None, :]) & (y[:, None] == y[None, :])
    dt = np.abs(t[:, None] - t[None, :])
    assert np.abs(D[~same_pix]).max() == 0.0                      # white in space
    tau = dt / (6 * H); z = np.sqrt(3) * tau
    expect = np.where(same_pix, 0.25 * (1 + z) * np.exp(-z) + np.where(np.abs(dt - 2 * H) < 1, -0.5 * noise, 0.0), 0.0)
    assert np.abs(D - expect).max() < 1e-12
    # with the iid diagonal noise added (as the operator does), the whole thing is a valid covariance
    assert np.linalg.eigvalsh(Kfull + noise * np.eye(len(x))).min() > 0
    # lag-2 structure = covariance of hourly differences of independent noise: the diagonal-noise + lag-2 part alone is PSD
    Dn = np.where(same_pix & (np.abs(dt - 2 * H) < 1), -0.5 * noise, 0.0) + noise * np.eye(len(x))
    assert np.linalg.eigvalsh(Dn).min() > -1e-12


def test_prediction_rows_see_no_observation_error():
    x, y, t, a, b, rng = _grid()
    p = K.pack_params_resid(BASE, sigma_R=0.5, tau0_R=6 * H, noise_lag2=-2.0)
    pts = K.build_points(x, y, t, a, b, resid=True)
    rows = K.build_pred_points(x[:5], y[:5], t[:5], 'S', resid=True)
    assert rows.shape[1] == 7 and np.all(rows[:, 6] == 0)
    c_with = np.array([[K.transport_obs_covfunc_resid(r, q, p) for q in pts] for r in rows])
    c_base = np.array([[K.transport_obs_covfunc(r[:6], q[:6], BASE) for q in pts] for r in rows])
    assert np.abs(c_with - c_base).max() < 1e-12


def test_gp_matches_dense_with_resid_and_lowrank():
    x, y, t, a, b, rng = _grid(nx=3, nt=6, seed=2)
    N = x.size
    noise = 3.0
    p = K.pack_params_resid(BASE, sigma_R=0.6, tau0_R=5 * H, noise_lag2=-0.5 * noise)
    pts = K.build_points(x, y, t, a, b, resid=True)
    term = LowRankTerm('diurnal', {'S': diurnal_harmonics(2)}, scale=0.6)
    U = term.basis(pts)
    C = _dense(pts, p, K.transport_obs_covfunc_resid) + noise * np.eye(N) + U @ U.T
    z = rng.multivariate_normal(np.zeros(N), C)
    gp = SpaceTimeGP(pts, z, p, noise, lowrank=[term], nnum=N, Lx=20e3, Lt=86400.0, jitter=1e-10,
                     covfunc=K.transport_obs_covfunc_resid)
    ll = -0.5 * (z @ np.linalg.solve(C, z) + np.linalg.slogdet(C)[1] + N * np.log(2 * np.pi))
    assert abs(gp.loglik() - ll) < 1e-6
    alpha = np.linalg.solve(C, z)
    xs, ys, ts = x[:6] + 1000.0, y[:6] + 500.0, t[:6] + 1800.0
    for fld in 'uvS':
        rows = K.build_pred_points(xs, ys, ts, fld, resid=True)
        cross = np.array([[K.transport_obs_covfunc_resid(r, q, p) for q in pts] for r in rows]) + term.basis(rows) @ U.T
        mean, var = gp.predict(fld, xs, ys, ts)
        assert np.abs(mean - cross @ alpha).max() < 1e-6, fld


def test_noise_lag2_stays_tied_in_fit_params():
    p0 = K.pack_params_resid(BASE, sigma_R=0.3, noise_lag2=-2.0)
    params = dict(sigma_psi=2000., noise=5.0, sigma_R=0.4, tau0_R=3 * H)
    cov, noise, lr = F.params_to_model_args(params, p0)
    assert cov[15] == -2.5 and cov[12] == 0.4 and cov[13] == 3 * H and noise == 5.0
    p_off = K.pack_params_resid(BASE, sigma_R=0.3, noise_lag2=0.0)       # lag-2 switched off stays off
    cov, _, _ = F.params_to_model_args(params, p_off)
    assert cov[15] == 0.0


if __name__ == "__main__":
    for fn in (test_reduces_to_base_when_terms_off, test_terms_structure_and_psd,
               test_prediction_rows_see_no_observation_error, test_gp_matches_dense_with_resid_and_lowrank,
               test_noise_lag2_stays_tied_in_fit_params):
        fn(); print("ok", fn.__name__)

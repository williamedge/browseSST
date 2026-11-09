"""
Derivative-capable T kernel and TGP model against finite differences / dense references.
Run: python tests/test_tgp.py
"""
import numpy as np

from inversion_sst_gp.spacetime import tgp as G

H = 3600.0
P = G.pack_params_T(0.6, 25e3, 2 * 86400.0, 0.25, 12e3, 0.5 * 86400.0, nu_t=2.5)
STEP = np.array([40.0, 40.0, 60.0])           # FD steps in x, y, t (m, m, s): small against ell ~ 12 km, tau ~ 12 h


def _value(p1, p2, p):
    r1 = np.array([p1[0], p1[1], p1[2], 1.0, 0, 0, 0])
    r2 = np.array([p2[0], p2[1], p2[2], 1.0, 0, 0, 0])
    return G.T_covfunc(r1, r2, p)


def _fd_cov(i, j, p1, p2, p):
    """Cov(D_i T(p1), D_j T(p2)) by finite differences of the value kernel; i, j in {0:T, 1:x, 2:y, 3:t}."""
    def shifted(pt, op, sgn):
        q = np.array(pt, float)
        if op: q[op - 1] += sgn * STEP[op - 1]
        return q
    if i == 0 and j == 0:
        return _value(p1, p2, p)
    if j == 0:
        h = STEP[i - 1]
        return (_value(shifted(p1, i, 1), p2, p) - _value(shifted(p1, i, -1), p2, p)) / (2 * h)
    if i == 0:
        h = STEP[j - 1]
        return (_value(p1, shifted(p2, j, 1), p) - _value(p1, shifted(p2, j, -1), p)) / (2 * h)
    hi, hj = STEP[i - 1], STEP[j - 1]
    f = lambda si, sj: _value(shifted(p1, i, si), shifted(p2, j, sj), p)
    return (f(1, 1) - f(1, -1) - f(-1, 1) + f(-1, -1)) / (4 * hi * hj)


def test_all_derivative_pairs_match_finite_differences():
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(6):
        p1 = np.array([0.0, 0.0, 0.0]) + rng.uniform(-1, 1, 3) * [8e3, 8e3, 6e3]
        p2 = rng.uniform(-1, 1, 3) * [8e3, 8e3, 6e3] + [3e3, -2e3, 9e3]
        if _ == 0:                      # include a coincident-point case
            p2 = p1.copy()
        for i in range(4):
            for j in range(4):
                w1 = np.zeros(4); w1[i] = 1
                w2 = np.zeros(4); w2[j] = 1
                r1 = np.concatenate([p1, w1]); r2 = np.concatenate([p2, w2])
                got = G.T_covfunc(r1, r2, P)
                ref = _fd_cov(i, j, p1, p2, P)
                scale = max(abs(got), abs(ref), 1e-30)
                worst = max(worst, abs(got - ref) / scale if scale > 1e-14 else 0.0)
                assert abs(got - ref) <= 2e-4 * scale + 1e-16, (i, j, got, ref)
    print(f"  worst relative mismatch over all pairs: {worst:.2e}")


def test_symmetry_and_psd_with_derivatives():
    rng = np.random.default_rng(1)
    n = 40
    x, y, t = rng.uniform(0, 30e3, n), rng.uniform(0, 30e3, n), rng.uniform(0, 3 * 86400, n)
    rows = []
    for k in range(n):
        w = np.zeros(4); w[k % 4] = 1
        rows.append(np.array([x[k], y[k], t[k], *w]))
    Kmat = np.array([[G.T_covfunc(a, b, P) for b in rows] for a in rows])
    assert np.abs(Kmat - Kmat.T).max() < 1e-12 * np.abs(Kmat).max()
    # scale rows so derivative entries are O(1) before the eigenvalue test
    d = np.sqrt(np.abs(np.diag(Kmat)))
    assert np.linalg.eigvalsh(Kmat / np.outer(d, d)).min() > -1e-9


def test_lowrank_basis_derivatives_match_finite_differences():
    for phi in (G.poly_mean(), G.diurnal_T(3)):
        x, y, t = np.array([1e3, 2e4]), np.array([3e3, -5e3]), np.array([4e4, 9e4])
        f, fx, fy, ft = phi(x, y, t)
        for arr, (dx, dy, dt) in ((fx, (1.0, 0, 0)), (fy, (0, 1.0, 0)), (ft, (0, 0, 1.0))):
            h = 2.0
            up = phi(x + h * dx, y + h * dy, t + h * dt)[0]
            dn = phi(x - h * dx, y - h * dy, t - h * dt)[0]
            assert np.abs((up - dn) / (2 * h) - arr).max() < 1e-9


def test_tgp_matches_dense_with_derivatives_and_lowrank():
    rng = np.random.default_rng(2)
    N = 90
    x, y, t = rng.uniform(0, 40e3, N), rng.uniform(0, 40e3, N), rng.uniform(0, 2 * 86400, N)
    pts = G.build_T_points(x, y, t)
    noise = 1e-3
    terms = [G.TLowRankTerm('mean', G.poly_mean(), scale=0.5), G.TLowRankTerm('diurnal', G.diurnal_T(2), scale=0.2)]
    U = np.hstack([tm.basis(pts) for tm in terms])
    C = np.array([[G.T_covfunc(a, b, P) for b in pts] for a in pts]) + noise * np.eye(N) + U @ U.T
    T = rng.multivariate_normal(np.zeros(N), C)
    gp = G.TGP(pts, T, P, noise, lowrank=terms, nnum=N, Lx=20e3, Lt=86400.0, jitter=1e-12)
    ll = -0.5 * (T @ np.linalg.solve(C, T) + np.linalg.slogdet(C)[1] + N * np.log(2 * np.pi))
    assert abs(gp.loglik() - ll) < 1e-6, (gp.loglik(), ll)
    alpha = np.linalg.solve(C, T)
    xs, ys, ts = rng.uniform(8e3, 32e3, 8), rng.uniform(8e3, 32e3, 8), rng.uniform(0.3 * 86400, 1.7 * 86400, 8)
    for fld in ('T', 'Tx', 'Ty', 'Tt'):
        rows = G.build_T_pred_points(xs, ys, ts, fld)
        cross = np.array([[G.T_covfunc(r, q, P) for q in pts] for r in rows]) + np.hstack([tm.basis(rows) for tm in terms]) @ U.T
        mean, var = gp.predict(fld, xs, ys, ts)
        assert np.abs(mean - cross @ alpha).max() < 1e-7 * max(1.0, np.abs(cross @ alpha).max()), fld
        assert (var > 0).all(), fld


def test_rough_kernel_matches_base_kernel_for_nu52_and_finite_differences_for_all_nu():
    """T_covfunc_res (fields T and dT/dt only): equals T_covfunc at nu_s = 5/2; time-derivative pairs match FD for nu_s in {1/2, 3/2};
    PSD for the rough kernels."""
    rng = np.random.default_rng(5)
    p52 = G.pack_params_T_res(0.6, 25e3, 2 * 86400.0, 0.25, 12e3, 0.5 * 86400.0, 2.5, 2.5, 2.5)
    pts = rng.uniform(-1, 1, (8, 3)) * [8e3, 8e3, 6e3]
    for i in range(8):
        for j in range(8):
            for wi in ((1, 0), (0, 1)):
                for wj in ((1, 0), (0, 1)):
                    r1 = np.array([*pts[i], wi[0], 0, 0, wi[1]], float); r2 = np.array([*pts[j], wj[0], 0, 0, wj[1]], float)
                    a, b = G.T_covfunc_res(r1, r2, p52), G.T_covfunc(r1, r2, P)
                    assert abs(a - b) <= 1e-12 * max(abs(b), 1e-30) + 1e-18, (a, b)
    for nus in (0.5, 1.5):
        p = G.pack_params_T_res(0.6, 25e3, 2 * 86400.0, 0.25, 12e3, 0.5 * 86400.0, 2.5, nus, nus)
        h = 60.0
        for _ in range(5):
            p1 = rng.uniform(-1, 1, 3) * [8e3, 8e3, 6e3]; p2 = rng.uniform(-1, 1, 3) * [8e3, 8e3, 6e3] + [3e3, -2e3, 9e3]
            val = lambda q1, q2: G.T_covfunc_res(np.array([*q1, 1.0, 0, 0, 0]), np.array([*q2, 1.0, 0, 0, 0]), p)
            sh = lambda q, d: q + np.array([0, 0, d])
            got_t0 = G.T_covfunc_res(np.array([*p1, 0, 0, 0, 1.0]), np.array([*p2, 1.0, 0, 0, 0]), p)
            ref_t0 = (val(sh(p1, h), p2) - val(sh(p1, -h), p2)) / (2 * h)
            got_tt = G.T_covfunc_res(np.array([*p1, 0, 0, 0, 1.0]), np.array([*p2, 0, 0, 0, 1.0]), p)
            ref_tt = (val(sh(p1, h), sh(p2, h)) - val(sh(p1, h), sh(p2, -h)) - val(sh(p1, -h), sh(p2, h)) + val(sh(p1, -h), sh(p2, -h))) / (4 * h * h)
            assert abs(got_t0 - ref_t0) <= 2e-4 * max(abs(ref_t0), 1e-30) + 1e-16, (nus, got_t0, ref_t0)
            assert abs(got_tt - ref_tt) <= 2e-4 * max(abs(ref_tt), 1e-30) + 1e-16, (nus, got_tt, ref_tt)
        rows = np.array([[*q, w0, 0, 0, wt] for q in rng.uniform(-1, 1, (25, 3)) * [8e3, 8e3, 6e3] for (w0, wt) in ((1, 0), (0, 1))], float)
        C = np.array([[G.T_covfunc_res(a, b, p) for b in rows] for a in rows])
        assert np.abs(C - C.T).max() <= 1e-12 * np.abs(C).max()
        assert np.linalg.eigvalsh(C + 1e-12 * np.abs(C).max() * np.eye(len(C))).min() > 0


if __name__ == "__main__":
    for fn in (test_all_derivative_pairs_match_finite_differences, test_symmetry_and_psd_with_derivatives,
               test_lowrank_basis_derivatives_match_finite_differences, test_tgp_matches_dense_with_derivatives_and_lowrank,
               test_rough_kernel_matches_base_kernel_for_nu52_and_finite_differences_for_all_nu):
        fn(); print("ok", fn.__name__)

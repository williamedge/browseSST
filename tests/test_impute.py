"""
Matheron-rule derivative sampler against a dense GP in the exact regime (every point a neighbour, so Vecchia is exact).
Run: python tests/test_impute.py
"""
import numpy as np

from inversion_sst_gp.spacetime import tgp as G
from inversion_sst_gp.spacetime.impute import DerivativeImputer

H = 3600.0
DX, DT = 4000.0, H
P = G.pack_params_T(0.2, 12e3, 27 * H, 0.07, 4e3, 9 * H, nu_t=2.5)
NOISE = 2e-3


def _dense(rows1, rows2, p):
    out = np.zeros((len(rows1), len(rows2)))
    for i in range(len(rows1)):
        for j in range(len(rows2)):
            out[i, j] = G.T_covfunc(rows1[i], rows2[j], p)
    return out


def _problem(seed=0):
    xs, ts = np.arange(4) * DX, np.arange(6) * H
    tt, yy, xx = np.meshgrid(ts, xs, xs, indexing='ij')
    x, y, t = xx.ravel(), yy.ravel(), tt.ravel()
    pts = G.build_T_points(x, y, t)
    terms = [G.TLowRankTerm('mean', G.poly_mean(), 0.3), G.TLowRankTerm('diurnal', G.diurnal_T(2), 0.05)]
    # truth for the data: any draw from the prior + noise (the posterior covariance does not depend on the values)
    rng = np.random.default_rng(seed)
    z = 0.1 * rng.standard_normal(len(x))
    n = len(x)
    gp = G.TGP(pts, z, P, NOISE, lowrank=terms, nnum=n - 1, Lx=20e3, Lt=24 * H)
    return x, y, t, pts, terms, z, gp


def test_prior_operator_is_exact_when_every_point_is_a_neighbour():
    x, y, t, pts, terms, z, gp = _problem()
    imp = DerivativeImputer(gp, x, y, t, dx=DX, dt=DT, nnum_prior=4 * len(x) - 1, nnum_pred=len(x), jitter=1e-10)
    S = _dense(imp.rows, imp.rows, P) + 1e-10 * np.eye(4 * len(x))
    v = np.random.default_rng(1).standard_normal(4 * len(x))
    got, ref = imp.prior_op.matvec(v), S @ v
    err = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    print(f"  prior operator vs dense Sigma v: relative error {err:.2e}")
    assert err < 1e-5


def test_matheron_draws_match_dense_posterior():
    x, y, t, pts, terms, z, gp = _problem()
    n = len(x)
    imp = DerivativeImputer(gp, x, y, t, dx=DX, dt=DT, nnum_prior=4 * n - 1, nnum_pred=n, jitter=1e-10, seed=3)
    mean, draws = imp.posterior_samples(500, verbose=False)
    # dense posterior of the 3 derivative fields at all N points (physical units)
    w = {'Tx': (0, 1, 0, 0), 'Ty': (0, 0, 1, 0), 'Tt': (0, 0, 0, 1)}
    Drows = np.vstack([G.build_T_pred_points(x, y, t, f) for f in ('Tx', 'Ty', 'Tt')])
    U = np.hstack([tm.basis(pts) for tm in terms]); Ud = np.hstack([tm.basis(Drows) for tm in terms])
    C = _dense(pts, pts, P) + NOISE * np.eye(n) + U @ U.T
    Kdc = _dense(Drows, pts, P) + Ud @ U.T
    Kdd = _dense(Drows, Drows, P) + Ud @ Ud.T
    post_mean = Kdc @ np.linalg.solve(C, z)
    post_cov = Kdd - Kdc @ np.linalg.solve(C, Kdc.T)
    vec_mean = np.concatenate([mean[f] for f in ('Tx', 'Ty', 'Tt')])
    D = np.array([np.concatenate([d[f] for f in ('Tx', 'Ty', 'Tt')]) for d in draws])
    sd = np.sqrt(np.diag(post_cov))
    mean_err = np.abs(vec_mean - post_mean) / sd
    var_ratio = D.var(axis=0, ddof=1) / np.diag(post_cov)
    mean_of_draws_err = np.abs(D.mean(axis=0) - post_mean) / sd
    print(f"  Vecchia mean vs dense (units of posterior sd): max {mean_err.max():.2e}")
    print(f"  sample mean vs dense (units of sd; ~1/sqrt(500)=0.045 expected): rms {np.sqrt(np.mean(mean_of_draws_err**2)):.3f}")
    print(f"  draw variance / dense posterior variance: mean {var_ratio.mean():.3f}, min {var_ratio.min():.2f}, max {var_ratio.max():.2f}")
    assert mean_err.max() < 1e-4
    assert np.sqrt(np.mean(mean_of_draws_err ** 2)) < 0.07
    assert abs(var_ratio.mean() - 1) < 0.06 and var_ratio.min() > 0.7 and var_ratio.max() < 1.4


def test_resolved_mode_fields_T_and_Tt_match_dense_posterior():
    """fields = ('T', 'Tt'), targets = ('T', 'Tt'): the other imputation mode, also exact when every point is a neighbour."""
    x, y, t, pts, terms, z, gp = _problem(seed=1)
    n = len(x)
    imp = DerivativeImputer(gp, x, y, t, dx=DX, dt=DT, nnum_prior=2 * n - 1, nnum_pred=n, jitter=1e-10, seed=4,
                            fields=('T', 'Tt'), targets=('T', 'Tt'))
    mean, draws = imp.posterior_samples(400, verbose=False)
    D = np.vstack([G.build_T_pred_points(x, y, t, f) for f in ('T', 'Tt')])
    U = np.hstack([tm.basis(pts) for tm in terms]); Ud = np.hstack([tm.basis(D) for tm in terms])
    C = _dense(pts, pts, P) + NOISE * np.eye(n) + U @ U.T
    Kdc = _dense(D, pts, P) + Ud @ U.T
    Kdd = _dense(D, D, P) + Ud @ Ud.T
    post_mean = Kdc @ np.linalg.solve(C, z)
    post_cov = Kdd - Kdc @ np.linalg.solve(C, Kdc.T)
    sd = np.sqrt(np.diag(post_cov))
    vec_mean = np.concatenate([mean['T'], mean['Tt']])
    Dd = np.array([np.concatenate([d['T'], d['Tt']]) for d in draws])
    ratio = Dd.var(axis=0, ddof=1) / np.diag(post_cov)
    print(f"  resolved mode: Vecchia mean vs dense (sd) max {np.abs(vec_mean - post_mean).max() / sd.min():.1e}; "
          f"draw var / dense var mean {ratio.mean():.3f} (min {ratio.min():.2f}, max {ratio.max():.2f})")
    assert np.abs((vec_mean - post_mean) / sd).max() < 1e-4
    assert abs(ratio.mean() - 1) < 0.06 and ratio.min() > 0.7 and ratio.max() < 1.4


def test_grid_gradients_exact_on_cubic_fields_and_nan_borders():
    from inversion_sst_gp.spacetime.impute import grid_gradients
    nt, ny, nx, dx = 3, 9, 8, 4000.0
    tt, yy, xx = np.meshgrid(np.arange(nt), np.arange(ny) * dx, np.arange(nx) * dx, indexing='ij')
    Tfield = 1e-9 * xx ** 2 * yy / dx + 2e-10 * xx ** 3 - 3e-10 * yy ** 3 + 1e-3 * tt
    gx_ref = 1e-9 * 2 * xx * yy / dx + 6e-10 * xx ** 2
    gy_ref = 1e-9 * xx ** 2 / dx - 9e-10 * yy ** 2
    for order, b in ((4, 2), (2, 1)):
        gx, gy = grid_gradients(Tfield.ravel(), (nt, ny, nx), dx, order=order)
        gx, gy = gx.reshape(nt, ny, nx), gy.reshape(nt, ny, nx)
        inner_x = (slice(None), slice(None), slice(b, nx - b)); inner_y = (slice(None), slice(b, ny - b), slice(None))
        assert np.isnan(gx[:, :, :b]).all() and np.isnan(gy[:, :b, :]).all() and np.isnan(gx[:, :, nx - b:]).all()
        tol = 1e-9 if order == 4 else None
        if order == 4:                                  # the 5-point stencil is exact for cubics
            assert np.allclose(gx[inner_x], gx_ref[inner_x], rtol=1e-9, atol=1e-18)
            assert np.allclose(gy[inner_y], gy_ref[inner_y], rtol=1e-9, atol=1e-18)
        else:                                           # the 3-point stencil is exact for quadratics: check a quadratic field
            q = 1e-9 * xx ** 2 + 2e-9 * yy ** 2
            gq, _ = grid_gradients(q.ravel(), (nt, ny, nx), dx, order=2)
            assert np.allclose(gq.reshape(nt, ny, nx)[inner_x], (2e-9 * xx)[inner_x], rtol=1e-9, atol=1e-18)


if __name__ == "__main__":
    import time
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            t0 = time.time(); fn(); print(f"PASS {name} ({time.time() - t0:.1f}s)")

"""
Regional tidal regression (spacetime/tides.py): algebraic tests on exactly consistent rows (no simulation): given gradient fields, a tide with SPATIALLY VARYING amplitude and phase
and a diurnal heating source, dT/dt is built from the equation and the coefficient fields must be recovered; shallow rows must be droppable.
Run: python tests/test_tides.py   (plain runner, no pytest)
"""
import numpy as np

from inversion_sst_gp.spacetime import tides as TD

DAY = 86400.0
NAMES = ('M2', 'S2', 'K1', 'O1')
L = 300e3


def true_tide(x, y, t, rough=None):
    """Smooth tide whose amplitude and phase vary across the box (a deeper-water-weaker, phase-lagging pattern); optional rough extra term where `rough` (bool / weight) is set."""
    u = np.zeros_like(x); v = np.zeros_like(x)
    for k, nm in enumerate(NAMES):
        w = TD.omega_of(nm)
        A = (0.35 - 0.2 * y / L) * (1 + 0.1 * k)
        ph = 0.3 + k + 0.8 * y / L + 0.4 * x / L
        u += A * np.cos(w * t - ph)
        v += 0.6 * A * (1 + 0.2 * x / L) * np.cos(w * t - ph - 1.1)
    if rough is not None:
        w = TD.omega_of('M2')
        u = u + rough * 0.5 * np.cos(w * t - 2 * np.pi * (x + y) / 30e3)
        v = v + rough * 0.4 * np.sin(w * t - 2 * np.pi * (x - y) / 30e3)
    return u, v


def rows(n_days=40, n_per_hour=60, seed=0, shallow_rough=False, day_vary=False):
    rng = np.random.default_rng(seed)
    nt = 24 * n_days
    t = np.repeat(np.arange(nt) * 3600.0, n_per_hour)
    x = rng.uniform(0, L, t.size); y = rng.uniform(0, L, t.size)
    # gradients: smooth in space, slowly varying in time and not commensurate with the tide
    ph = rng.uniform(0, 2 * np.pi, 6)
    Tx = 3e-5 * (np.sin(2 * np.pi * x / 120e3 + 2 * np.pi * t / (5.3 * DAY) + ph[0]) + 0.5 * np.cos(2 * np.pi * y / 90e3 - 2 * np.pi * t / (3.1 * DAY) + ph[1]))
    Ty = 3e-5 * (np.cos(2 * np.pi * y / 100e3 + 2 * np.pi * t / (4.3 * DAY) + ph[2]) + 0.5 * np.sin(2 * np.pi * x / 80e3 - 2 * np.pi * t / (6.7 * DAY) + ph[3]))
    depth = np.minimum(3000.0, 100.0 * 2.0 ** (y / 35e3))          # shallow (< 200 m) for y below 35 km, 3000 m beyond 170 km
    u, v = true_tide(x, y, t, rough=(depth < 200.0) if shallow_rough else None)
    if day_vary:                      # the heating cycle differs from day to day (amplitude 0.2-1.8 of the mean, onset shifted by up to 1.5 h)
        d = np.floor(t / DAY).astype(int)
        amp = rng.uniform(0.2, 1.8, d.max() + 2)[d]; shift = rng.uniform(-1.5, 1.5, d.max() + 2)[d] * 3600.0
        S = 8e-7 * amp * np.maximum(0.0, np.sin(2 * np.pi * (t - 6 * 3600.0 - shift) / DAY)) * (1 + 0.3 * x / L)
    else:
        S = 8e-7 * np.maximum(0.0, np.sin(2 * np.pi * (t - 6 * 3600.0) / DAY)) * (1 + 0.3 * x / L)
    Tt = -Tx * u - Ty * v + S
    return dict(x=x, y=y, t=t, Tt=Tt, Tx=Tx, Ty=Ty, depth=depth)


def _score(fit, mask_deep=True, seed=5):
    rng = np.random.default_rng(seed)
    n = 4000
    x, y, t = rng.uniform(0, L, n), rng.uniform(0, L, n), rng.uniform(0, 30 * DAY, n)
    keep = (y > 80e3) if mask_deep else np.ones(n, bool)
    u, v = fit.velocity(x, y, t, mask_unsupported=False)
    ut, vt = true_tide(x, y, t)
    err = np.hypot(u - ut, v - vt)[keep]
    return float(np.sqrt(np.mean(err ** 2))), float(np.sqrt(np.mean(ut[keep] ** 2 + vt[keep] ** 2)))


def test_rayleigh_table():
    assert abs(TD.rayleigh_days('M2', 'S2') - 14.77) < 0.02
    assert abs(TD.rayleigh_days('K1', 'O1') - 13.66) < 0.02
    assert abs(TD.rayleigh_days('M2', 'N2') - 27.55) < 0.03
    assert 175 < TD.rayleigh_days('K1', 'P1') < 190 and 175 < TD.rayleigh_days('S2', 'K2') < 190      # not separable in a few months


def test_spline_basis_is_a_partition_of_unity_and_penalty_ignores_planes():
    b = TD.SplineBasis2D((0, 300e3, 0, 200e3), 50e3)
    rng = np.random.default_rng(0)
    x, y = rng.uniform(0, 300e3, 500), rng.uniform(0, 200e3, 500)
    cols, vals = b.design(x, y)
    assert cols.shape == (500, 16) and np.allclose(vals.sum(1), 1.0) and (vals >= -1e-12).all()
    # a plane c0 + c1 x + c2 y is exactly representable: coefficients from Greville abscissae, penalty must vanish
    gx = np.array([np.mean(b.knots[0][i + 1:i + 4]) for i in range(b.Mx)]); gy = np.array([np.mean(b.knots[1][i + 1:i + 4]) for i in range(b.My)])
    coef = (2.0 + 3e-6 * gx[None, :] - 1e-6 * gy[:, None]).ravel()
    assert np.allclose(coef[cols] @ np.ones(1) if False else np.einsum('ij,ij->i', coef[cols], vals), 2.0 + 3e-6 * x - 1e-6 * y, atol=1e-9)
    assert abs(coef @ (b.penalty() @ coef)) < 1e-12 * max(1.0, coef @ coef)


def test_recovers_spatially_varying_amplitude_and_phase():
    r = rows()
    fit = TD.fit_tides(r['x'], r['y'], r['t'], r['Tt'], r['Tx'], r['Ty'], depth=None, spacing=75e3, bbox=(0, L, 0, L))
    err, rms = _score(fit, mask_deep=False)
    assert err < 0.02 * rms, (err, rms)
    # amplitude / phase maps at a few points
    xs = np.array([50e3, 150e3, 250e3]); ys = np.array([60e3, 150e3, 240e3])
    h = fit.harmonics(xs, ys)
    for k, nm in enumerate(NAMES):
        A = (0.35 - 0.2 * ys / L) * (1 + 0.1 * k)
        ph = 0.3 + k + 0.8 * ys / L + 0.4 * xs / L
        assert np.allclose(h[nm]['Au'], A, rtol=0.05, atol=0.01), (nm, h[nm]['Au'], A)
        d = np.angle(np.exp(1j * (h[nm]['phi_u'] - ph)))
        assert np.all(np.abs(d) < 0.05), (nm, d)
    assert fit.resid_rms < 5e-8 and fit.info['n_dropped'] == 0      # the residual is the truncated heating harmonics (0.2% of rms dT/dt)


def test_without_the_nuisance_term_the_diurnal_heating_contaminates_k1_and_s2():
    r = rows()
    good = TD.fit_tides(r['x'], r['y'], r['t'], r['Tt'], r['Tx'], r['Ty'], spacing=75e3, bbox=(0, L, 0, L))
    bad = TD.fit_tides(r['x'], r['y'], r['t'], r['Tt'], r['Tx'], r['Ty'], spacing=75e3, bbox=(0, L, 0, L), nuisance_harmonics=-1)
    eg, _ = _score(good, mask_deep=False); eb, _ = _score(bad, mask_deep=False)
    assert eb > 3 * eg, (eg, eb)


def test_shallow_rows_are_dropped_and_protect_the_deep_estimate():
    r = rows(shallow_rough=True)
    kw = dict(spacing=75e3, bbox=(0, L, 0, L))
    masked = TD.fit_tides(r['x'], r['y'], r['t'], r['Tt'], r['Tx'], r['Ty'], depth=r['depth'], min_depth=200.0, support_radius=20e3, **kw)
    unmasked = TD.fit_tides(r['x'], r['y'], r['t'], r['Tt'], r['Tx'], r['Ty'], depth=r['depth'], min_depth=None, **kw)
    assert masked.info['n_dropped'] > 0 and unmasked.info['n_dropped'] == 0
    em, rms = _score(masked); eu, _ = _score(unmasked)       # scored in the deep part (y > 80 km)
    assert em < 0.05 * rms and eu > 2 * em, (em, eu, rms)
    # positions farther than the support radius (20 km) from every retained row (the shelf at y = 0 is 35 km from the first kept row) are NaN, deep ones are not
    far = masked.velocity(np.array([150e3]), np.array([0.0]), np.array([1e5]))[0]
    near = masked.velocity(np.array([150e3]), np.array([150e3]), np.array([1e5]))[0]
    assert np.isnan(far).all() and np.isfinite(near).all()


def test_time_derivative_and_block_mean_and_rows():
    t = np.arange(40) * 3600.0
    w = TD.omega_of('M2')
    T = np.sin(w * t)[:, None, None] * np.ones((1, 4, 6))
    d4 = TD.time_derivative(T, 3600.0, 4); d2 = TD.time_derivative(T, 3600.0, 2)
    ref = w * np.cos(w * t)
    assert np.isnan(d4[:2]).all() and np.isnan(d4[-2:]).all()
    assert np.max(np.abs(d4[2:-2, 0, 0] - ref[2:-2])) < 3e-3 * w and np.max(np.abs(d2[1:-1, 0, 0] - ref[1:-1])) > 3e-2 * w      # FD4 error 0.2% of w, FD2 loses 4% at M2
    a = np.arange(2 * 4 * 6, dtype=float).reshape(2, 4, 6)
    m = TD.block_mean(a, 2)
    assert m.shape == (2, 2, 3) and m[0, 0, 0] == a[0, :2, :2].mean()
    a2 = a.copy(); a2[0, 0, 0] = np.nan; a2[0, 0, 1] = np.nan
    assert np.isnan(TD.block_mean(a2, 2, min_frac=0.75)[0, 0, 0]) and np.isfinite(TD.block_mean(a2, 2, min_frac=0.5)[0, 0, 0])
    Tt = np.random.default_rng(0).standard_normal((2, 4, 6)); Tx = Tt * 2; Ty = Tt * 3
    out = TD.rows_from_cube(Tt, Tx, Ty, np.arange(6) * 4e3, np.arange(4) * 4e3, [0.0, 3600.0], f=2, depth=np.full((4, 6), 500.0))
    assert out['x'].size == 12 and np.allclose(out['x'][:3], [2e3, 10e3, 18e3]) and np.allclose(out['depth'], 500.0)


def test_pooled_moments_reproduce_the_pixel_normal_equations_exactly():
    """With every pixel of a block sharing the block-centre position, the pooled pseudo-rows must give the same fit (and the same residual) as the pixel rows."""
    rng = np.random.default_rng(3)
    f, nbl = 3, 4
    nt, ny = 24 * 8, f * nbl
    cell = 40e3 / nbl * nbl / ny * 0 + 25e3                                        # pixel spacing 25 km -> blocks of 75 km
    x1 = (np.arange(ny) + 0.5) * cell; y1 = (np.arange(ny) + 0.5) * cell
    t1 = np.arange(nt) * 3600.0
    tt, yy, xx = np.meshgrid(t1, y1, x1, indexing='ij')
    Tx = 3e-5 * np.sin(2 * np.pi * xx / 120e3 + 2 * np.pi * tt / (5.3 * DAY)) * (1 + 0.5 * rng.standard_normal(xx.shape))
    Ty = 3e-5 * np.cos(2 * np.pi * yy / 100e3 - 2 * np.pi * tt / (4.3 * DAY)) * (1 + 0.5 * rng.standard_normal(xx.shape))
    xb = np.repeat((np.arange(nbl) + 0.5) * f * cell, f)[None, :] * np.ones((ny, 1)); yb = np.repeat((np.arange(nbl) + 0.5) * f * cell, f)[:, None] * np.ones((1, ny))
    u, v = true_tide(np.broadcast_to(xb, xx.shape), np.broadcast_to(yb, xx.shape), tt)
    S = 8e-7 * np.maximum(0.0, np.sin(2 * np.pi * (tt - 6 * 3600.0) / DAY))
    Tt = -Tx * u - Ty * v + S + 2e-7 * rng.standard_normal(xx.shape)
    Tt[:, 1, 2] = np.nan; Tx[5, 7, 7] = np.nan                                  # gaps are excluded from both
    L_ = nbl * f * cell
    kw = dict(spacing=75e3, bbox=(0, L_, 0, L_), smooth=1e-6, nuisance_harmonics=2, chunk=7000, compute_cov=False)
    ok = np.isfinite(Tt) & np.isfinite(Tx) & np.isfinite(Ty)
    pix = TD.fit_tides(np.broadcast_to(xb, xx.shape)[ok], np.broadcast_to(yb, xx.shape)[ok], tt[ok], Tt[ok], Tx[ok], Ty[ok], **kw)
    pool = TD.pool_rows(Tt, Tx, Ty, x1, y1, t1, f, min_frac=0.0)
    assert abs(pool['x'].min() - 0.5 * f * cell) < 1e-6 and pool['n_equiv'] == int(ok.sum())
    pl = TD.fit_tides(pool['x'], pool['y'], pool['t'], pool['Tt'], pool['Tx'], pool['Ty'], Cn=pool['Cn'], n_equiv=pool['n_equiv'], **kw)
    # compare the identifiable quantity (the tidal velocity at the block centres), not raw spline coefficients (poorly determined where there is little data)
    cx, cy = np.meshgrid((np.arange(nbl) + 0.5) * f * cell, (np.arange(nbl) + 0.5) * f * cell)
    for tq in (1e5, 3e5, 5e5):
        up, vp = pix.velocity(cx, cy, np.full(cx.shape, tq), mask_unsupported=False); ul, vl = pl.velocity(cx, cy, np.full(cx.shape, tq), mask_unsupported=False)
        assert np.allclose(up, ul, atol=2e-5) and np.allclose(vp, vl, atol=2e-5), (np.abs(up - ul).max(), np.abs(vp - vl).max())
    assert abs(pix.resid_rms / pl.resid_rms - 1) < 1e-3, (pix.resid_rms, pl.resid_rms)


def test_day_by_day_nuisance_handles_inconsistent_heating():
    r = rows(day_vary=True)
    kw = dict(spacing=75e3, bbox=(0, L, 0, L))
    fixed = TD.fit_tides(r['x'], r['y'], r['t'], r['Tt'], r['Tx'], r['Ty'], **kw)
    daily = TD.fit_tides(r['x'], r['y'], r['t'], r['Tt'], r['Tx'], r['Ty'], nuisance_t_spacing=DAY, **kw)
    ef, rms = _score(fixed, mask_deep=False); ed, _ = _score(daily, mask_deep=False)
    print(f"   inconsistent heating: tide error fixed cycle {ef:.4f}, day-by-day {ed:.4f} m/s (tide rms {rms:.3f})")
    assert ed < 0.5 * ef, (ed, ef, rms)
    assert daily.gamma.shape[:2] == (9, int(np.ceil(r['t'].max() / DAY)) + 1) and daily.nuis_t_spacing == DAY
    # with consistent heating the extra freedom costs little
    r0 = rows()
    d0 = TD.fit_tides(r0['x'], r0['y'], r0['t'], r0['Tt'], r0['Tx'], r0['Ty'], nuisance_t_spacing=DAY, **kw)
    e0, rms0 = _score(d0, mask_deep=False)
    print(f"   consistent heating, day-by-day nuisance: error {e0:.4f} m/s (tide rms {rms0:.3f})")
    assert e0 < 0.05 * rms0, (e0, rms0)


def test_schur_solver_matches_the_dense_solution():
    r = rows(n_days=12, n_per_hour=25, day_vary=True)
    kw = dict(spacing=100e3, bbox=(0, L, 0, L), nuisance_harmonics=3, nuisance_t_spacing=DAY, nuisance_t_smooth=0.05, constituents=('M2', 'S2'))
    d = TD.fit_tides(r['x'], r['y'], r['t'], r['Tt'], r['Tx'], r['Ty'], solver='dense', **kw)
    s_ = TD.fit_tides(r['x'], r['y'], r['t'], r['Tt'], r['Tx'], r['Ty'], solver='schur', **kw)
    assert d.info['solver'] == 'dense' and s_.info['solver'] == 'schur'
    assert np.allclose(s_.beta, d.beta, rtol=1e-6, atol=1e-9 * np.abs(d.beta).max()), np.abs(s_.beta - d.beta).max()
    assert np.allclose(s_.gamma, d.gamma, rtol=1e-6, atol=1e-9 * np.abs(d.gamma).max()), np.abs(s_.gamma - d.gamma).max()
    assert abs(s_.resid_rms - d.resid_rms) < 1e-8 * d.resid_rms
    nt = d.beta.size
    assert np.allclose(s_.cov, d.cov[:nt, :nt], rtol=1e-6, atol=1e-9 * np.abs(d.cov).max())
    print(f"   dense vs schur: max |dbeta| {np.abs(s_.beta - d.beta).max():.1e} (beta rms {np.sqrt(np.mean(d.beta ** 2)):.3f}), {d.info['n_params']} parameters")


if __name__ == "__main__":
    import time
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            t0 = time.time(); fn(); print(f"PASS {name} ({time.time() - t0:.1f}s)")

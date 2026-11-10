"""Tests of the real-data pipeline pieces: bathymetry loader, 2-D pooling positions, tile evaluation of the regional tide, hyperparameter files, tile files + merge, a tiny run_tile.

Plain runner (no pytest in the env):  python tests/test_pipeline.py
"""
import json
import pathlib
import sys
import tempfile
import time

import numpy as np
import xarray as xr

from inversion_sst_gp.spacetime import bathymetry as B
from inversion_sst_gp.spacetime import pipeline as P
from inversion_sst_gp.spacetime import regional as R
from inversion_sst_gp.spacetime import tides as TD
from inversion_sst_gp.spacetime import tiling as TL


def depth_fn(lo, la):
    """Depth (m, positive down; negative = land): deepens to the north, a land strip in the south-west corner."""
    d = 50.0 + 4000.0 * (la - (-16.0)) / 4.0
    return np.where((lo < 121.5) & (la < -15.5), -20.0, d)


def test_bathymetry_cell_mean_and_conventions():
    tmp = pathlib.Path(tempfile.mkdtemp())
    lon = 121.0 + 0.04 * np.arange(20)
    lat = -15.8 + 0.04 * np.arange(25)                         # ascending, the cube's convention
    ref = None
    for kw in (dict(), dict(descending_lat=True), dict(lon360=True), dict(name="z"), dict(descending_lat=True, lon360=True)):
        p = tmp / ("g%d.nc" % len(list(tmp.iterdir())))
        B.write_synthetic_gebco(p, lon, lat, depth_fn, dlon=1 / 120.0, dlat=1 / 120.0, **kw)
        d = B.load_depth(p, lon, lat)
        assert d.shape == (25, 20)
        ref = d if ref is None else ref
        assert np.allclose(d, ref, atol=1e-3), "descending latitude / 0..360 longitude / variable name must not change the result"
    # cell mean of a linear field is the value at the cell centre (within the source resolution); skip the land cells
    dd = depth_fn(*np.meshgrid(lon, lat))
    wet = dd > 0
    assert np.abs(ref[wet] - dd[wet]).max() < 4000.0 / 4.0 * (1 / 120.0) * 1.0 + 1.0, np.abs(ref[wet] - dd[wet]).max()
    assert ref[~wet].max() == 0.0, "land is depth 0"
    # target axes in descending order give the flipped array
    d2 = B.load_depth(tmp / "g0.nc", lon[::-1], lat[::-1])
    assert np.allclose(d2, ref[::-1, ::-1], atol=1e-3)
    # 'depth' variable (positive down)
    ds = xr.Dataset({"depth": (("lat", "lon"), np.full((30, 30), 500.0))}, coords={"lat": np.linspace(-17, -14, 30), "lon": np.linspace(120, 123, 30)})
    ds.to_netcdf(tmp / "pd.nc")
    assert np.allclose(B.load_depth(tmp / "pd.nc", lon, lat), 500.0)


def test_pool_rows_2d_positions_match_1d():
    rng = np.random.default_rng(0)
    nt, ny, nx = 6, 20, 25
    Tt, Tx, Ty = (rng.standard_normal((nt, ny, nx)) for _ in range(3))
    x1 = 4000.0 * np.arange(nx); y1 = 4000.0 * np.arange(ny)
    X, Y = np.meshgrid(x1, y1)
    a = TD.pool_rows(Tt, Tx, Ty, x1, y1, np.arange(nt) * 3600.0, 5)
    b = TD.pool_rows(Tt, Tx, Ty, X, Y, np.arange(nt) * 3600.0, 5)
    for k in ("x", "y", "t", "Tt", "Tx", "Ty", "Cn"):
        assert np.allclose(a[k], b[k]), k
    # a sheared grid: block centres are the means of the pixel positions
    Xs = X + 0.1 * Y
    c = TD.pool_rows(Tt, Tx, Ty, Xs, Y, np.arange(nt) * 3600.0, 5)
    assert np.allclose(c["x"] - a["x"], 0.1 * (c["y"]), atol=1e-6)


def _fake_cube(nt=12, ny=16, nx=20, seed=0):
    rng = np.random.default_rng(seed)
    lon = 121.0 + 0.037 * np.arange(nx); lat = -15.0 + 0.036 * np.arange(ny)
    t = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(nt) * np.timedelta64(3600, "s")
    LO, LA = np.meshgrid(lon, lat)
    sst = 300.0 + 0.8 * (LA + 15.0) + 0.2 * np.sin(2 * np.pi * (LO - 121.0) / 0.5)[None] + 0.05 * np.sin(2 * np.pi * np.arange(nt) / 24)[:, None, None]
    sst = sst + 0.002 * rng.standard_normal((nt, ny, nx))
    return xr.Dataset({"sst": (("time", "lat", "lon"), sst.astype(np.float32))}, coords={"time": t, "lat": lat, "lon": lon})


def _fake_tide_cube():
    """T advected exactly by an elliptical M2 current (u = 0.2 cos wt, v = 0.1 sin wt along the frame axes): T(x, y, t) = f(x - xi(t), y - eta(t)) with f two plane waves of
    different directions (so that u and v are separately identifiable)."""
    w = TD.omega_of("M2")
    cube = _fake_cube(nt=24, ny=40, nx=40)
    lon = cube["lon"].values; lat = cube["lat"].values
    frame = R.regional_frame(lon, lat)
    x2, y2 = frame.forward(lon[None, :], lat[:, None])
    nt = 24 * 14
    t = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(nt) * np.timedelta64(3600, "s")
    ts = np.arange(nt) * 3600.0
    xi = (0.2 / w) * np.sin(w * ts)[:, None, None]
    eta = -(0.1 / w) * np.cos(w * ts)[:, None, None]
    X, Y = x2[None] - xi, y2[None] - eta
    k1 = 2 * np.pi / 90e3; k2 = 2 * np.pi / 70e3
    T = 290.0 + 0.5 * np.sin(k1 * (X * np.cos(0.3) + Y * np.sin(0.3))) + 0.4 * np.sin(k2 * (X * np.cos(1.9) + Y * np.sin(1.9)) + 1.0)
    return xr.Dataset({"sst": (("time", "lat", "lon"), T.astype(np.float64))}, coords={"time": t, "lat": lat, "lon": lon}), frame


def test_fit_region_and_tile_velocity():
    big, frame = _fake_tide_cube()
    cfg = R.TideConfig(constituents=("M2",), block=5, spacing=80e3, nuisance_harmonics=-1, min_depth=200.0, chunk_hours=100)
    depth = np.full((big.sizes["lat"], big.sizes["lon"]), 3000.0)
    m = R.fit_region(big, depth, cfg, log=None)
    # M2 u amplitude 0.2 m/s (cos), v 0 at the cube centre
    lon0, lat0 = m.lon0, m.lat0
    w = TD.omega_of("M2")
    for hrs in (0, 3):
        u, v = m.velocity(np.array([lon0]), np.array([lat0]), np.datetime64("2026-01-01T00:00:00", "ns") + np.timedelta64(hrs * 3600, "s"))
        assert abs(u[0] - 0.2 * np.cos(w * hrs * 3600)) < 0.03 and abs(v[0] - 0.1 * np.sin(w * hrs * 3600)) < 0.03, (hrs, u, v)
    # tile evaluation equals the point evaluation at the tile's own pixel positions / times, and time is measured from the fit's origin, not the tile's
    sub = big.isel(time=slice(100, 140), lat=slice(5, 25), lon=slice(8, 30))
    spec = TL.TileSpec("t", 0, 40, 0, 20, 0, 22, 0, 40, 0, 20, 0, 22)
    td = TL.extract_tile(sub, spec)
    ub, vb, sup = m.tile_velocity(td, mask="none")
    uu, vv = m.velocity(td.lon[None, :], td.lat[:, None], td.time[:, None, None], mask="none")
    assert np.allclose(ub, uu.ravel()) and np.allclose(vb, vv.ravel())
    assert sup.all()
    # outputs: constituent maps
    ds = m.constituents(td.lon, td.lat)
    assert abs(float(ds["M2_Au"].mean()) - 0.2) < 0.02
    # pickling round trip
    p = pathlib.Path(tempfile.mkdtemp()) / "tide.pkl"
    m.save(p)
    m2 = R.TideModel.load(p)
    u2, _ = m2.velocity(np.array([lon0]), np.array([lat0]), np.datetime64("2026-01-01T03:00:00", "ns"))
    u1, _ = m.velocity(np.array([lon0]), np.array([lat0]), np.datetime64("2026-01-01T03:00:00", "ns"))
    assert np.allclose(u1, u2)


def test_shallow_blocks_have_no_support():
    big, _ = _fake_tide_cube()
    cfg = R.TideConfig(constituents=("M2",), block=5, spacing=80e3, nuisance_harmonics=-1, min_depth=200.0, chunk_hours=100)   # default support radius = one block width
    depth = np.full((big.sizes["lat"], big.sizes["lon"]), 3000.0)
    depth[:20] = 50.0                                                       # southern half shelf
    m = R.fit_region(big, depth, cfg, log=None)
    lo = big["lon"].values[20]
    s = m.supported(np.array([lo, lo]), np.array([big["lat"].values[5], big["lat"].values[35]]))
    assert (not s[0]) and s[1], s


def test_hyper_roundtrip_and_pool():
    h = P.Hyper.default()
    p = pathlib.Path(tempfile.mkdtemp()) / "h.json"
    h.save(p)
    h2 = P.Hyper.load(p)
    assert h2.stage1 == h.stage1 and h2.stage2 == h.stage2
    a = P.Hyper(dict(h.stage1), dict(h.stage2))
    b = P.Hyper({k: 2 * v for k, v in h.stage1.items()}, {k: 2 * v for k, v in h.stage2.items()})
    c = P.Hyper({k: 4 * v for k, v in h.stage1.items()}, {k: 4 * v for k, v in h.stage2.items()})
    pooled = P.pool_hypers([a, b, c])
    assert np.isclose(pooled.stage1["l1"], 2 * h.stage1["l1"]) and np.isclose(pooled.stage2["gamma"], 2 * h.stage2["gamma"])
    assert np.isclose(pooled.meta["spread_max_over_min"]["l1"], 4.0)


def test_tile_files_and_merge_reproduce_a_smooth_field():
    cube = _fake_cube(nt=12, ny=30, nx=34)
    specs = TL.make_tiles((12, 30, 34), (6, 15, 17), (2, 3, 3))
    assert len(specs) == 8
    d = tempfile.mkdtemp()
    LO, LA = np.meshgrid(cube["lon"].values, cube["lat"].values)
    for s in specs:
        td = TL.extract_tile(cube, s)
        nt, ny, nx = td.shape
        field = (3.0 * (td.lon[None, None, :] - 121.0) - 2.0 * (td.lat[None, :, None] + 15.0) + 0.01 * (s.k0 + np.arange(nt))[:, None, None]) * np.ones(td.shape)
        out = {k: field.copy() for k in P.TILE_FIELDS}
        out["tide_supported"] = np.ones(td.shape, np.int8)
        out["info"] = dict(tile=s.id)
        P.write_tile(pathlib.Path(d) / f"tile_{s.id}.nc", td, out)
    ds = P.merge_tiles((dict(), specs), d, cube, None, fields=("u", "v", "T"), log=None)
    ref = (3.0 * (LO - 121.0)[None] - 2.0 * (LA + 15.0)[None] + 0.01 * np.arange(12)[:, None, None])
    assert np.allclose(ds["u"].values, ref, atol=1e-5), np.abs(ds["u"].values - ref).max()
    # lazy extraction touches only the block
    td = TL.extract_tile(cube, specs[3])
    assert td.shape == specs[3].shape


def test_array_tide_reads_the_cube_block_and_run_tile_adds_it_back():
    cube = _fake_cube(nt=12, ny=12, nx=12, seed=5)
    uu = 0.1 * np.ones((12, 12, 12)); vv = -0.05 * np.ones((12, 12, 12))
    uu[:, :3] = np.nan                                                     # an unsupported strip
    cube["u_tide"] = (("time", "lat", "lon"), uu); cube["v_tide"] = (("time", "lat", "lon"), vv)
    spec = TL.TileSpec("a", 2, 10, 1, 11, 2, 12, 2, 10, 1, 11, 2, 12)
    td = TL.extract_tile(cube, spec)
    at = R.ArrayTide(cube)
    u, v, sup = at.tile_velocity(td)
    assert u.shape == td.x.shape and np.isfinite(u).all()
    s3 = sup.reshape(td.shape)
    assert (not s3[:, :2].any()) and s3[:, 3:].all(), "rows 1-2 of the block are the NaN strip (cube rows 1, 2)"
    assert np.allclose(u.reshape(td.shape)[:, 3:], 0.1) and np.allclose(u.reshape(td.shape)[:, :2], 0.0)
    cfg = P.PipelineConfig(nnum=30, nnum_pred=40, min_obs=100)
    out = P.run_tile(td, P.Hyper.default(), cfg, tide=at, log=None)
    assert np.allclose(out["u"], out["u_eddy"] + out["u_tide"]) and abs(out["u_tide"].reshape(td.shape)[:, 5:].mean() - 0.1) < 1e-9


def test_daily_diurnal_basis_and_structured_prior():
    from inversion_sst_gp.spacetime.model import diurnal_harmonics, diurnal_harmonics_daily
    knots = np.array([0.0, 86400.0, 172800.0])
    phi = diurnal_harmonics_daily(knots, 2)
    t = np.linspace(0, 172800.0, 97); z = np.zeros_like(t)
    B = phi(z, z, t); base = diurnal_harmonics(2)(z, z, t)
    assert B.shape == (97, 3 * 5)
    # hats are a partition of unity: summing the columns of the same harmonic over the knots gives the plain harmonic
    assert np.allclose(B.reshape(97, 3, 5).sum(1), base)
    # a coefficient on the first hat only changes the first day
    only_first = B[:, :5] @ np.ones(5)
    assert np.allclose(only_first[t >= 86400.0], 0.0) and np.abs(only_first[t < 43200.0]).max() > 0.1
    cube = _fake_cube(nt=12, ny=12, nx=12, seed=4)
    td = TL.extract_tile(cube, TL.TileSpec("a", 0, 12, 0, 12, 0, 12, 0, 12, 0, 12, 0, 12))
    pr = P.StructuredSPrior(daily=True)
    terms = pr.lowrank(dict(P.DEFAULT_STAGE2), td)
    assert terms[0].phi['S'](td.x, td.y, td.t).shape[1] % 9 == 0 and "noise" in pr.free and "noise" in pr.bounds
    assert P.StructuredSPrior(daily=False).lowrank(dict(P.DEFAULT_STAGE2), td)[0].phi['S'](td.x, td.y, td.t).shape[1] == 9


def test_run_tile_smoke():
    """A tiny tile with the default (placeholder) hyperparameters: runs end to end, finite outputs, tide hook adds the tide back."""
    cube = _fake_cube(nt=12, ny=12, nx=12, seed=3)
    spec = TL.TileSpec("s", 0, 12, 0, 12, 0, 12, 0, 12, 0, 12, 0, 12)
    td = TL.extract_tile(cube, spec)
    cfg = P.PipelineConfig(nnum=30, nnum_pred=40, min_obs=100)
    t0 = time.time()
    out = P.run_tile(td, P.Hyper.default(), cfg, tide=None, log=None)
    assert out is not None
    for k in ("u", "v", "S", "T", "var_u"):
        assert out[k].shape == (12, 12, 12) and np.isfinite(out[k]).all(), k
    assert np.allclose(out["u"], out["u_eddy"])                       # no tide: total = eddy
    assert (out["var_u"] > 0).all()
    # too few observations -> skipped
    cube2 = cube.copy(deep=True); cube2["sst"][:] = np.nan
    assert P.run_tile(TL.extract_tile(cube2, spec), P.Hyper.default(), cfg, log=None) is None
    print(f"    run_tile smoke {time.time() - t0:.0f}s")


def _rand_rows(n, rng):
    return np.column_stack([rng.uniform(0, 40e3, n), rng.uniform(0, 40e3, n), rng.uniform(0, 3 * 86400.0, n), rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)])


def test_browsegp_covfunc_matches_the_helmholtz_covfunc_and_browsegp_tensors():
    from inversion_sst_gp.spacetime import kernels as K
    import browseGP.covariance.helmholtz_numba as hn
    rng = np.random.default_rng(0)
    X = _rand_rows(60, rng)
    # (1) rot + static-envelope divergent block == the existing transport covfunc (omega = 0, exponential envelope = Matern-1/2)
    p_old = K.pack_params(5e3, 30e3, 4e3, 20e3, 2 * 86400.0, 86400.0, 3.0, 15e3, 7200.0, nu_t_psi=1.5, nu_t_phi=0.5, nu_t_S=1.5)
    p_new = np.zeros(K.BG_N_PARAMS)
    p_new[0:4] = [5e3, 30e3, 2 * 86400.0, 1.5]
    p_new[4:10] = [4e3, 20e3, 86400.0, 0.0, 1.0, 0.5]
    p_new[24:28] = [3.0, 15e3, 7200.0, 1.5]
    for i in range(0, 60, 3):
        for j in range(0, 60, 4):
            a, b = K.transport_obs_covfunc(X[i], X[j], p_old), K.browsegp_transport_covfunc(X[i], X[j], p_new)
            assert abs(a - b) < 1e-9 * max(1.0, abs(a)), (a, b)
    # (2) the full model against browseGP's own tensors (rot, M2 wave, M4 wave) + the fast / inertial blocks written out
    om = TD.omega_of("M2"); f = 2 * 7.2921159e-5 * np.sin(np.deg2rad(14.0))
    p = K.pack_browsegp(0.07, 54e3, 3.8e5, 0.12, 33e3, 6.1e5, om, gamma_m4=0.025, ell_m4=12e3, tau_m4=3.4e5, gamma_f=0.068, ell_f=2e3, tau_f=4600.0,
                        gamma_in=0.024, ell_in=18e3, tau_in=2.3e5, omega_in=f, pol=1.0, sigma_S=2.0, l_S=15e3, tau0_S=3600.0)
    m52 = lambda r, s_, l_: s_ ** 2 * (1 + np.sqrt(5) * r / l_ + 5 * r ** 2 / (3 * l_ ** 2)) * np.exp(-np.sqrt(5) * r / l_)

    def tensor(x1, x2, a, b):
        aug1 = np.array([x1[0], x1[1], x1[2], a]); aug2 = np.array([x2[0], x2[1], x2[2], b])
        v = hn.rotational_velocity_tensor_spacetime(aug1, aug2, p[0:4]) + hn.divergent_velocity_tensor_wave_spacetime(aug1, aug2, p[4:10]) + hn.divergent_velocity_tensor_wave_spacetime(aug1, aug2, p[10:16])
        dx, dy, tau = x1[0] - x2[0], x1[1] - x2[1], x1[2] - x2[2]
        r = np.hypot(dx, dy)
        if a == b:
            v += m52(r, p[16], p[17]) * np.exp(-abs(tau) / p[18]) + m52(r, p[19], p[20]) * np.exp(-abs(tau) / p[21]) * np.cos(p[22] * tau)
        else:
            sg = -1.0 if a == 0 else 1.0
            v += sg * p[23] * m52(r, p[19], p[20]) * np.exp(-abs(tau) / p[21]) * np.sin(p[22] * tau)
        return v
    for i in range(0, 60, 5):
        for j in range(0, 60, 6):
            x1, x2 = X[i], X[j]
            ks = m52(np.hypot(x1[0] - x2[0], x1[1] - x2[1]), 2.0, 15e3) * (1 + np.sqrt(3) * abs(x1[2] - x2[2]) / 3600.0) * np.exp(-np.sqrt(3) * abs(x1[2] - x2[2]) / 3600.0)
            ref = x1[3] * x2[3] * tensor(x1, x2, 0, 0) + x1[4] * x2[4] * tensor(x1, x2, 1, 1) + x1[3] * x2[4] * tensor(x1, x2, 0, 1) + x1[4] * x2[3] * tensor(x1, x2, 1, 0) + x1[5] * x2[5] * ks
            got = K.browsegp_transport_covfunc(x1, x2, p)
            assert abs(ref - got) < 1e-9 * max(1e-6, abs(ref)), (ref, got)
    # (3) symmetric positive semi-definite (the polarised inertial block has an antisymmetric u-v cross term)
    Xs = _rand_rows(120, rng)
    G = np.array([[K.browsegp_transport_covfunc(Xs[i], Xs[j], p) for j in range(120)] for i in range(120)])
    assert np.allclose(G, G.T, rtol=1e-9, atol=1e-12) and np.linalg.eigvalsh(G).min() > -1e-9 * np.linalg.eigvalsh(G).max()


def test_browsegp_block_weights_apply_to_the_cross_covariance_only():
    from inversion_sst_gp.spacetime import kernels as K
    pr = P.BrowseGPPrior(dict(gamma_rot=0.07, ell_rot=54e3, tau_rot=3.8e5, gamma_div=0.12, ell_div=33e3, tau_div=6.1e5, gamma_m4=0.025, ell_m4=12e3, tau_m4=3.4e5, gamma_f=0.068, ell_f=2e3, tau_f=4600.0,
                                gamma_in=0.024, ell_in=18e3, tau_in=2.3e5))
    h2 = dict(P.DEFAULT_STAGE2)
    rng = np.random.default_rng(1)
    X = _rand_rows(30, rng); X[:, 5] = 1.0                                  # data rows: c = 1
    pu = np.array([[1e3, 2e3, 5e4, 1.0, 0.0, 0.0], [3e4, 1e4, 1e5, 0.0, 1.0, 0.0]])    # prediction rows (c = 0)
    full = pr.cov_params(h2)
    for blocks in (("rot",), ("div", "m4"), ("fast", "inertial")):
        w = pr.cov_params_blocks(h2, blocks)
        for i in range(30):
            assert K.browsegp_transport_covfunc(X[i], X[(i + 7) % 30], w) == K.browsegp_transport_covfunc(X[i], X[(i + 7) % 30], full)       # data-data: untouched
    for a in pu:
        for x in X[:10]:
            tot = K.browsegp_transport_covfunc(a, x, full)
            parts = [K.browsegp_transport_covfunc(a, x, pr.cov_params_blocks(h2, (b,))) for b in pr.BLOCK_ORDER]
            assert abs(sum(parts) - tot) < 1e-12 * max(1.0, abs(tot)), (sum(parts), tot)
            assert abs(K.browsegp_transport_covfunc(x, a, pr.cov_params_blocks(h2, ("rot",))) - K.browsegp_transport_covfunc(a, x, pr.cov_params_blocks(h2, ("rot",)))) < 1e-12   # symmetric


def test_browsegp_prior_and_run_tile():
    res = pathlib.Path("/mnt/c/Users/00099894/OneDrive - The University of Western Australia/pl/wedge/browseGP/results/mcmc_M4_v0.01/mcmc_results.json")
    vel = dict(gamma_rot=0.0718, ell_rot=54.4e3, tau_rot=3.8e5, gamma_div=0.1197, ell_div=32.8e3, tau_div=6.1e5, gamma_m4=0.0248, ell_m4=11.8e3, tau_m4=3.4e5, gamma_f=0.068, ell_f=1987.0, tau_f=4617.0,
               gamma_in=0.024, ell_in=18.2e3, tau_in=2.3e5)
    pr = P.BrowseGPPrior.from_results(res) if res.exists() else P.BrowseGPPrior(vel)
    h2 = dict(P.DEFAULT_STAGE2)
    cp = pr.cov_params(h2)
    assert cp.size == 28 and abs(cp[0] - pr.vel["gamma_rot"] * pr.vel["ell_rot"] * np.sqrt(0.6)) < 1e-9 and cp[24] == h2["sigma_S"]
    assert pr.free == ("sigma_S", "l_S", "tau0_S", "lr:diurnal", "noise") and {"sigma_S", "l_S", "tau0_S", "ell_rot", "tau_f"} <= set(pr.slots)      # velocity parameters FIXED by default
    pr2 = P.BrowseGPPrior(pr.vel, blocks=("rot", "div"))
    assert pr2.cov_params(h2)[16] == 0.0 and pr2.cov_params(h2)[19] == 0.0 and pr2.cov_params(h2)[10] == 0.0
    cube = _fake_cube(nt=12, ny=12, nx=12, seed=3)
    spec = TL.TileSpec("s", 0, 12, 0, 12, 0, 12, 0, 12, 0, 12, 0, 12)
    td = TL.extract_tile(cube, spec)
    cfg = P.PipelineConfig(nnum=30, nnum_pred=40, min_obs=100, prior=pr)
    hyper = P.Hyper.default(); hyper.stage2["noise"] = 5.0
    out = P.run_tile(td, hyper, cfg, tide=None, log=None)
    assert out is not None and np.isfinite(out["u"]).all() and (out["var_u"] > 0).all()
    h = P.fit_hyper(td, hyper, cfg, maxiter1=2, maxiter2=2, log=None)         # the fit varies only S / diurnal / noise through the kernel slots
    assert set(h.stage2) >= {"sigma_S", "l_S", "tau0_S", "noise"} and h.meta["n_eval2"] > 0


_BG_VEL = dict(gamma_rot=0.0718, ell_rot=54.4e3, tau_rot=3.8e5, gamma_div=0.1197, ell_div=32.8e3, tau_div=6.1e5, gamma_m4=0.0248, ell_m4=11.8e3, tau_m4=3.4e5, gamma_f=0.068, ell_f=1987.0, tau_f=4617.0,
               gamma_in=0.024, ell_in=18.2e3, tau_in=2.3e5)


def test_browsegp_slots_set_amplitudes_from_gamma_and_keep_off_blocks_off():
    from inversion_sst_gp.spacetime.fit import params_to_model_args
    pr = P.BrowseGPPrior(_BG_VEL, blocks=("rot", "div", "fast"))
    h2 = pr.complete(dict(P.DEFAULT_STAGE2))
    base = pr.cov_params(h2)
    s35 = np.sqrt(0.6)
    p = dict(h2); p.update(gamma_rot=0.05, ell_rot=40e3, tau_rot=2e5, gamma_div=0.2, ell_div=25e3, tau_div=3e5, gamma_f=0.03, ell_f=3e3, tau_f=1e4, gamma_m4=0.9, gamma_in=0.9, sigma_S=2.0)
    cov, noise, lr = params_to_model_args(p, base, pr.slots)
    assert abs(cov[0] - 0.05 * 40e3 * s35) < 1e-9 and cov[1] == 40e3 and cov[2] == 2e5          # sigma_psi = gamma ell sqrt(3/5) with the NEW ell
    assert abs(cov[4] - 0.2 * 25e3 * s35) < 1e-9 and cov[5] == 25e3 and cov[6] == 3e5
    assert cov[16] == 0.03 and cov[17] == 3e3 and cov[18] == 1e4 and cov[24] == 2.0               # fast gamma is its own entry
    assert cov[10] == 0.0 and cov[19] == 0.0                                                      # M4 / inertial were switched off: they stay off
    for i in (3, 7, 8, 9, 12, 13, 25, 26, 27):                                                    # structural entries (nu_t, omega, envelope) untouched unless a slot
        if i not in (25, 26):
            assert cov[i] == base[i]
    cov2, _, _ = params_to_model_args(dict(sigma_S=3.0, noise=1.0), base, pr.slots)             # no velocity names in the dict: the base vector is kept
    assert np.array_equal(cov2[:24], base[:24]) and cov2[24] == 3.0


def test_browsegp_prior_free_velocity_parameters_and_fit_hyper():
    pr0 = P.BrowseGPPrior(_BG_VEL, blocks=("rot", "div", "fast"))
    assert pr0.free == P.BrowseGPPrior.S_FREE                                                    # default: velocity fixed
    pr = P.BrowseGPPrior(_BG_VEL, blocks=("rot", "div", "fast"), free_vel=True)
    vel_names = ("gamma_rot", "ell_rot", "tau_rot", "gamma_div", "ell_div", "tau_div", "gamma_f", "ell_f", "tau_f")
    assert pr.free == P.BrowseGPPrior.S_FREE + vel_names
    h2 = pr.complete(dict(P.DEFAULT_STAGE2))
    for n in vel_names:
        lo, hi = pr.bounds[n]
        assert lo < h2[n] < hi, n
    assert "gamma_m4" not in h2 and pr.complete(h2) == h2                                          # blocks that are off are not added; idempotent
    h3 = dict(h2); h3["gamma_rot"] = 0.01
    assert abs(pr.cov_params(h3)[0] - 0.01 * _BG_VEL["ell_rot"] * np.sqrt(0.6)) < 1e-9             # tile jobs read the fitted value from Hyper.stage2
    assert P.BrowseGPPrior(_BG_VEL, blocks=("rot", "div"), free_vel=("rot",)).free[-3:] == ("gamma_rot", "ell_rot", "tau_rot")
    try:
        P.BrowseGPPrior(_BG_VEL, blocks=("rot",), free_vel=("fast",))
        raise AssertionError("freeing a block that is off must raise")
    except ValueError:
        pass
    cube = _fake_cube(nt=12, ny=12, nx=12, seed=3)
    spec = TL.TileSpec("s", 0, 12, 0, 12, 0, 12, 0, 12, 0, 12, 0, 12)
    td = TL.extract_tile(cube, spec)
    cfg = P.PipelineConfig(nnum=30, nnum_pred=40, min_obs=100, prior=pr)
    hyper = P.Hyper.default(); hyper.stage2["noise"] = 5.0
    h = P.fit_hyper(td, hyper, cfg, maxiter1=2, maxiter2=2, log=None)
    assert set(vel_names) <= set(h.stage2) and all(np.isfinite(h.stage2[n]) and pr.bounds[n][0] <= h.stage2[n] <= pr.bounds[n][1] for n in vel_names)
    assert h.meta["prior"] == "browsegp" and set(vel_names) <= set(h.meta["free2"]) and h.meta["n_eval2"] > 0
    pooled = P.pool_hypers([h, h])                                                                # no f_delta key needed
    assert all(abs(pooled.stage2[n] - h.stage2[n]) < 1e-9 * abs(h.stage2[n]) for n in vel_names)
    out = P.run_tile(td, pooled, cfg, tide=None, log=None)                                         # the tile job uses the fitted velocity parameters
    assert out is not None and np.isfinite(out["u"]).all()


def test_tiles_with_low_coverage_inside_the_depth_range_are_skipped():
    cube = _fake_cube(nt=12, ny=12, nx=12, seed=3)
    spec = TL.TileSpec("s", 0, 12, 0, 12, 0, 12, 0, 12, 0, 12, 0, 12)
    depth = np.full((12, 12), 500.0); depth[:, :6] = np.nan                  # the west half is shallow (masked)
    cube = cube.assign(depth=(("lat", "lon"), depth))
    sst = cube["sst"].values.copy()
    sst[:, :, 6:] = np.where(np.arange(12)[:, None, None] < 4, sst[:, :, 6:], np.nan)      # the deep half is observed only 4 of 12 hours (1/3)
    sst[:, :, :6] = 290.0                                                        # the shallow half fully observed: must not count
    cube["sst"] = (("time", "lat", "lon"), sst)
    td = TL.extract_tile(cube, spec)
    assert abs(td.cover_deep() - 1 / 3) < 1e-9 and td.deep.sum() == 72
    cfg = P.PipelineConfig(nnum=30, nnum_pred=40, min_obs=100)
    assert cfg.min_cover == 0.5 and P.run_tile(td, P.Hyper.default(), cfg, tide=None, log=None) is None
    assert td.cover_deep() < 0.5
    td2 = TL.extract_tile(_fake_cube(nt=12, ny=12, nx=12, seed=3), spec)      # no depth variable: all pixels count
    assert td2.deep is None and td2.cover_deep() > 0.5


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            t0 = time.time()
            try:
                fn()
                print(f"PASS {name} ({time.time() - t0:.1f}s)")
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                print(f"FAIL {name}: {e!r}")
                fails += 1
    sys.exit(1 if fails else 0)

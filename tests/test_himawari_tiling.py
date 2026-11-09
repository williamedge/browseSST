"""
Himawari loader, local frame and tiling. Synthetic files mimic the pysatview output (int16 SST scaled by 0.01 + 273.15, int8 quality_level, one `time`
step per file); the real pysatview test files are used too when they are present.
Run: python tests/test_himawari_tiling.py   (plain runner, no pytest)
"""
import pathlib
import tempfile

import numpy as np
import xarray as xr

from inversion_sst_gp.spacetime import geo, himawari as H, tiling as TL

REAL = pathlib.Path(__file__).resolve().parents[2] / "pysatview" / "test_data" / "himawari" / "nc"


def _write(dirname, stamp, sst_k, qual, lat, lon, descending=False, name=None):
    """One file in the pysatview layout. sst_k (ny, nx) in K with NaN for fill; qual (ny, nx) int."""
    raw = np.where(np.isfinite(sst_k), np.round((sst_k - 273.15) / 0.01), -32768).astype(np.int16)
    qa = np.where(np.isfinite(sst_k), qual, -128).astype(np.int8)
    la = lat[::-1] if descending else lat
    if descending:
        raw, qa = raw[::-1], qa[::-1]
    ds = xr.Dataset({"sea_surface_temperature": (("time", "lat", "lon"), raw[None], {"_FillValue": np.int16(-32768), "scale_factor": np.float32(0.01), "add_offset": np.float32(273.15)}),
                     "quality_level": (("time", "lat", "lon"), qa[None], {"_FillValue": np.int8(-128)})},
                    coords={"time": [np.datetime64(f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}T{stamp[8:10]}:{stamp[10:12]}:{stamp[12:14]}", "ns")], "lat": la, "lon": lon})
    ds["sea_surface_temperature"].encoding.update(dtype="int16")
    ds.to_netcdf(pathlib.Path(dirname) / (name or f"{stamp}.nc"))


def _grid(ny=24, nx=20):
    return 112.0 + 0.02 * np.arange(nx), -23.0 + 0.02 * np.arange(ny)


def _field(lon, lat, k):
    return 295.0 + 0.5 * np.sin(2 * np.pi * (lon[None, :] - 112) / 0.3) + 0.3 * (lat[:, None] + 23) + 0.01 * k


def test_local_frame_distances_and_round_trip():
    fr = geo.LocalFrame(112.5, -22.0)
    x, y = fr.forward(113.5, -21.5)
    import pyproj
    d = pyproj.Geod(ellps="WGS84").inv(112.5, -22.0, 113.5, -21.5)[2]
    assert abs(np.hypot(x, y) - d) < 1e-3
    lon, lat = fr.inverse(x, y)
    assert abs(lon - 113.5) < 1e-9 and abs(lat + 21.5) < 1e-9
    # local east at the centre is the x axis; off-centre it rotates by the (small) convergence angle
    assert abs(fr.east_angle(112.5, -22.0)) < 1e-7
    assert 0 < abs(np.degrees(fr.east_angle(113.3, -22.0))) < 0.5


def test_metric_spacing_matches_the_projection():
    fr = geo.LocalFrame(112.5, -22.0)
    lat = np.array([-22.5, -22.0, -21.5])
    hx, hy = geo.metric_spacing(lat, 0.04, 0.04)
    for j, la in enumerate(lat):
        x0, y0 = fr.forward(112.5, la)
        x1, y1 = fr.forward(112.54, la)
        x2, y2 = fr.forward(112.5, la + 0.04)
        assert abs(np.hypot(x1 - x0, y1 - y0) - hx[j]) / hx[j] < 5e-4
        assert abs(np.hypot(x2 - x0, y2 - y0) - hy[j]) / hy[j] < 5e-4


def test_loader_masks_decodes_and_fills_missing_hours():
    lon, lat = _grid()
    with tempfile.TemporaryDirectory() as d:
        for k, st in enumerate(["20251001000000", "20251001010000", "20251001030000"]):   # 02:00 missing
            T = _field(lon, lat, k)
            q = np.full(T.shape, 5); q[:3] = 4                              # first 3 rows low quality
            T = T.copy(); T[:, :2] = np.nan                                    # two columns: land / no data
            _write(d, st, T, q, lat, lon, descending=(k == 1))
        c = H.load_cube(d, qmin=5)
        assert c.sizes["time"] == 4 and c.attrs["n_missing_hours"] == 1
        s = c["sst"].values
        assert np.isnan(s[2]).all()                                             # missing hour is an empty slab
        assert np.isnan(s[0, :3]).all() and np.isnan(s[0, :, :2]).all() and np.isfinite(s[0, 3:, 2:]).all()
        assert np.all(np.diff(c["lat"].values) > 0)                             # descending file was re-sorted
        np.testing.assert_allclose(s[1, 3:, 2:], _field(lon, lat, 1)[3:, 2:], atol=0.0051)   # 0.01 K quantisation
        assert np.isfinite(H.load_cube(d, qmin=4)["sst"].values[0, :3, 2:]).all()          # quality threshold is respected
        # windows and time selection
        assert H.load_cube(d, t0="2025-10-01T01:00:00").sizes["time"] == 3
        cov = H.coverage(c)
        assert cov.shape == (4,) and cov[2] == 0 and cov[0] > 0.7


def test_loader_crop_and_grid_mismatch():
    lon, lat = _grid()
    with tempfile.TemporaryDirectory() as d:
        _write(d, "20251001000000", _field(lon, lat, 0), np.full((24, 20), 5), lat, lon)
        c = H.load_cube(d, lonlim=(112.1, 112.25), latlim=(-22.9, -22.7))
        assert c["lon"].values.min() >= 112.1 - 1e-6 and c["lon"].values.max() <= 112.25 + 1e-6
        assert c["lat"].values.min() >= -22.9 - 1e-6 and c["lat"].values.max() <= -22.7 + 1e-6
        _write(d, "20251001010000", _field(lon + 0.01, lat, 0), np.full((24, 20), 5), lat, lon + 0.01)
        try:
            H.load_cube(d)
            raise AssertionError("a file on a different grid must be rejected")
        except ValueError:
            pass


def test_reduce_modes():
    s = np.arange(2 * 6 * 8, dtype=np.float32).reshape(2, 6, 8)
    lat, lon = np.arange(6.0), np.arange(8.0)
    a, la, lo = H.reduce_grid(s, 2, "stride", lat=lat, lon=lon)
    assert a.shape == (2, 3, 4) and la.tolist() == [0, 2, 4] and np.array_equal(a, s[:, ::2, ::2])
    s2 = s.copy(); s2[0, 0, 0] = np.nan
    m, la, lo = H.reduce_grid(s2, 2, "mean", min_frac=1.0, lat=lat, lon=lon)
    assert np.isnan(m[0, 0, 0]) and m[1, 0, 0] == s[1, :2, :2].mean() and la.tolist() == [0.5, 2.5, 4.5]
    m3, _, _ = H.reduce_grid(s2, 2, "mean", min_frac=0.75)
    assert abs(m3[0, 0, 0] - np.nanmean(s2[0, :2, :2])) < 1e-4


def test_find_windows():
    cov = np.array([1, 1, 1, 0.2, 1, 1, 1, 1, 1, 0.9])
    assert H.find_windows(cov, 4, min_mean=0.95, min_each=0.5).tolist() == [4, 5, 6]   # window 6 = (1, 1, 1, 0.9), mean 0.975
    assert H.find_windows(cov, 11).size == 0


def test_tile_layout_is_a_partition_and_weights_are_a_partition_of_unity():
    shape = (30, 61, 47)
    specs = TL.make_tiles(shape, core=(12, 28, 20), buffer=(3, 4, 4))
    cnt = np.zeros(shape, int)
    for s in specs:
        cnt[s.ck0:s.ck1, s.cj0:s.cj1, s.ci0:s.ci1] += 1
        assert s.k0 <= s.ck0 < s.ck1 <= s.k1 and s.j0 <= s.cj0 < s.cj1 <= s.j1 and s.i0 <= s.ci0 < s.ci1 <= s.i1
    assert (cnt == 1).all()                                                     # cores tile the cube exactly once
    # the remainder rule: 61 = 28 + 28 + 5 -> the last core absorbs the 5
    assert sorted({(s.cj0, s.cj1) for s in specs}) == [(0, 28), (28, 61)]
    # blending a constant field returns the constant everywhere; blending a smooth field returns it unchanged where all tiles agree
    B = TL.Blender(shape)
    for s in specs:
        B.add(s, np.full(s.shape, 3.0))
    np.testing.assert_allclose(B.result()[0], 3.0, atol=1e-12)
    f = np.random.default_rng(0).standard_normal(shape)
    B = TL.Blender(shape)
    for s in specs:
        B.add(s, f[s.slices])
    np.testing.assert_allclose(B.result()[0], f, atol=1e-12)                      # identical tile outputs blend to themselves


def test_blender_ignores_nan_per_field_and_matches_exp26_taper():
    spec = TL.make_tiles((1, 1, 20), core=(1, 1, 10), buffer=(0, 0, 3))[0]
    assert spec.buffers == (0, 0, 0, 0, 0, 3)
    w = TL.taper_1d(13, 0, 3)
    np.testing.assert_allclose(w[10:], np.sin(0.5 * np.pi * (np.arange(3)[::-1] + 0.5) / 3) ** 2)
    assert (w[:10] == 1).all()
    a = np.ones(spec.shape); a[0, 0, 4] = np.nan
    B = TL.Blender((1, 1, 20), 2).add(spec, a, np.full(spec.shape, 2.0))
    r = B.result()
    assert np.isnan(r[0, 0, 0, 4]) and r[1, 0, 0, 4] == 2.0 and np.isnan(r[0, 0, 0, 15])      # outside the tile: nothing


def test_manifest_round_trip():
    specs = TL.make_tiles((20, 40, 40), core=(10, 28, 28), buffer=(2, 4, 4))
    with tempfile.TemporaryDirectory() as d:
        TL.write_manifest(pathlib.Path(d) / "m.json", specs, {"stride": 2})
        meta, back = TL.read_manifest(pathlib.Path(d) / "m.json")
    assert meta == {"stride": 2} and back == specs


def test_gradients_match_the_impute_grid_gradients_on_a_uniform_grid():
    from inversion_sst_gp.spacetime.impute import grid_gradients
    rng = np.random.default_rng(1)
    shape = (3, 12, 14)
    T = rng.standard_normal(shape).ravel()
    for order in (2, 4):
        a = TL.grid_gradients_metric(T, shape, np.full(12, 4000.0), np.full(12, 4000.0), order)
        b = grid_gradients(T, shape, 4000.0, order)
        for u, v in zip(a, b):
            np.testing.assert_allclose(u, v, atol=1e-15, equal_nan=True)


def test_extract_tile_geometry_gradient_of_a_plane_and_scatter():
    lon, lat = np.arange(30) * 0.04 + 112.0, np.arange(26) * 0.04 - 23.0
    fr = geo.LocalFrame(112.58, -22.5)
    # a plane in TRUE east / north metres: T = a * E + b * N (E, N from the geodesic offsets of the centre via the aeqd frame, rotated to local east / north)
    a_, b_ = 2.0e-5, -1.5e-5
    ds = xr.Dataset({"sst": (("time", "lat", "lon"), np.zeros((4, 26, 30), np.float32))}, coords={"time": np.arange(4) * np.timedelta64(3600, "s") + np.datetime64("2025-10-01T00:00:00"), "lat": lat, "lon": lon})
    x, y = fr.forward(lon[None, :], lat[:, None])
    sst = 295.0 + a_ * x + b_ * y                                              # plane in the FRAME axes; its gradient in the frame is (a, b)
    ds["sst"].values[:] = sst[None]
    ds["sst"].values[:, 5, 5] = np.nan
    spec = TL.make_tiles(ds["sst"].shape, core=(4, 26, 30), buffer=(0, 0, 0))[0]
    td = TL.extract_tile(ds, spec)
    assert td.shape == (4, 26, 30) and td.n_obs == 4 * 26 * 30 - 4
    assert abs(td.T[td.obs].mean()) < 1e-3 and abs(td.Tref - 295.0) < 1.0
    assert td.t.reshape(td.shape)[:, 0, 0].tolist() == [0.0, 3600.0, 7200.0, 10800.0]
    # frame-axis gradients of the plane recover (a, b) up to the convergence angle (<0.5 deg) and the 1e-4-level metric error
    full = (sst[None] - td.Tref).repeat(4, axis=0).ravel()                     # the same plane without the hole
    Tx, Ty = td.gradients(full)
    g = np.hypot(Tx, Ty)
    assert abs(np.nanmedian(g) / np.hypot(a_, b_) - 1) < 0.01
    th = fr.east_angle(td.lon[None, :], td.lat[:, None])                        # east / north components of (a, b) in the frame axes
    Te = np.tile((a_ * np.cos(th) + b_ * np.sin(th)).ravel(), 4); Tn = np.tile((-a_ * np.sin(th) + b_ * np.cos(th)).ravel(), 4)
    ok = np.isfinite(Tx)
    assert np.nanmedian(np.abs(Tx[ok] - Te[ok])) < 0.01 * np.hypot(a_, b_) and np.nanmedian(np.abs(Ty[ok] - Tn[ok])) < 0.01 * np.hypot(a_, b_)
    # scatter puts values back at the observed points
    v = td.scatter(np.arange(td.n_obs, dtype=float))
    assert np.isnan(v[~td.obs]).all() and v[td.obs][3] == 3
    assert td.weights.shape == td.shape and td.coverage().shape == (4,)


def test_real_pysatview_files_if_present():
    if not (REAL / "20251001000000.nc").exists():
        print("  (pysatview test data not found: skipped)")
        return
    c = H.load_cube(REAL, t0="2025-10-01T00:00:00", t1="2025-10-01T03:00:00")
    assert c.sizes["time"] == 4 and c.sizes["lat"] == 250 and c.sizes["lon"] == 150 and c.attrs["n_missing_hours"] == 0
    s = c["sst"].values
    assert 285 < np.nanmin(s) and np.nanmax(s) < 305
    assert 0.5 < np.isfinite(s).mean() < 1
    c2 = H.load_cube(REAL, t0="2025-10-01T00:00:00", t1="2025-10-01T03:00:00", stride=2)
    assert c2.sizes["lat"] == 125 and c2.sizes["lon"] == 75
    # the very large raw granule next to the cropped files is a different time and is not touched here
    assert len(H.scan_files(REAL)) >= 4


if __name__ == "__main__":
    import time
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            t0 = time.time(); fn(); print(f"PASS {name} ({time.time() - t0:.1f}s)")

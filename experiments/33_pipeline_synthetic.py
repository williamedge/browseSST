"""
Experiment 33: the HPC chain end to end on the tide synthetic (84 x 84 px x 60 d), through the same CLI scripts as the real run.

`make DIR [offset_days] [n_days]`: writes DIR/cube.nc (the synthetic T_obs mapped onto a lon / lat grid with 4 km spacing at the centre latitude, hourly time axis from 2026-01-01),
  DIR/gebco.nc (GEBCO-style elevation file from the synthetic depth, 1 arc-minute), DIR/hyper_truth.json (stage 1 = the exp-31 refit on the slope tile, stage 2 = the TRUE Helmholtz
  hyperparameters with the exp-31 noise: this test is about the plumbing and the tide hook, not about estimation), DIR/truth.nc (true total / tide velocity on the cube grid).
`score DIR TILEDIR_A [TILEDIR_B ...]`: merges nothing itself; reads DIR/merged_<name>.nc for each variant name given and scores total velocity against the truth over the cells that
  the run tiles cover, split deep (>= 200 m) / shelf; skill is relative to the variance of the eddy (non-tidal) velocity.
The geometry is approximate (the synthetic is Cartesian, the cube lon / lat: the metric factors vary by 0.7% over the box), which is far below the 25% effects scored here.
"""
import json, pathlib, sys
import numpy as np
import xarray as xr

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = pathlib.Path.home() / "browsesst_data"
LON0, LAT0 = 123.0, -14.0
ZS = 1e-6


def axes(ds):
    from inversion_sst_gp.spacetime.geo import radii_of_curvature
    M, N = radii_of_curvature(LAT0)
    x, y = ds["x"].values.astype(float), ds["y"].values.astype(float)
    lon = LON0 + np.degrees((x - x.mean()) / (N * np.cos(np.radians(LAT0))))
    lat = LAT0 + np.degrees((y - y.mean()) / M)
    return lon, lat


def make(out, offset_days=10, n_days=30):
    from scipy.interpolate import RegularGridInterpolator
    from inversion_sst_gp.spacetime import synthetic as S, bathymetry as B
    from inversion_sst_gp.spacetime.fit import to_amplitude_params
    out = pathlib.Path(out); out.mkdir(parents=True, exist_ok=True)
    ds = xr.load_dataset(DATA / "synth_tide_84px_60d.nc")
    lon, lat = axes(ds)
    k0, k1 = int(offset_days * 24), int((offset_days + n_days) * 24)
    t = np.datetime64("2026-01-01T00:00:00", "ns") + np.arange(k0, k1) * np.timedelta64(3600, "s")
    T = ds["T_obs"].values[k0:k1].astype(np.float64) + 290.0
    xr.Dataset({"sst": (("time", "lat", "lon"), T.astype(np.float32), {"units": "K"})}, coords={"time": t, "lat": lat, "lon": lon}).to_netcdf(
        out / "cube.nc", encoding={"sst": dict(zlib=True, complevel=4)})
    f = RegularGridInterpolator((lat, lon), ds["depth"].values.astype(float), bounds_error=False, fill_value=None)
    B.write_synthetic_gebco(out / "gebco.nc", lon, lat, lambda lo, la: f(np.column_stack([la.ravel(), lo.ravel()])).reshape(lo.shape), dlon=1 / 60.0, dlat=1 / 60.0)
    xr.Dataset({k: (("time", "lat", "lon"), ds[k].values[k0:k1].astype(np.float32)) for k in ("u_true", "v_true", "u_tide_true", "v_tide_true")}
               | {"depth": (("lat", "lon"), ds["depth"].values.astype(np.float32))}, coords={"time": t, "lat": lat, "lon": lon}).to_netcdf(out / "truth.nc")
    # shared hyperparameters for the plumbing test
    cfg = S.SyntheticConfig(nx=84, ny=84, tide=True)
    tp = cfg.true_params(ZS)
    amp = to_amplitude_params(dict(sigma_psi=tp['sigma_psi'], l_psi=tp['l_psi'], sigma_phi=tp['sigma_phi'], l_phi=tp['l_phi'], tau0_psi=tp['tau0_psi'], tau0_phi=tp['tau0_phi'],
                                   sigma_S=tp['sigma_S'], l_S=tp['l_S'], tau0_S=tp['tau0_S']))
    z = np.load(DATA / "exp31_stage1_day20_row22.npz", allow_pickle=True)
    s1 = json.loads(str(z['params']))
    e31 = json.loads((ROOT / "data" / "exp31_day20_row22.json").read_text())
    s2 = {k: float(amp[k]) for k in ('gamma', 'f_delta', 'l_psi', 'l_phi', 'tau0_psi', 'tau0_phi', 'sigma_S', 'l_S', 'tau0_S')}
    s2.update(noise=float(e31['noise']), **{'lr:diurnal': 0.5})
    (out / "hyper_truth.json").write_text(json.dumps(dict(stage1=s1, stage2=s2, meta=dict(source="exp31 stage-1 refit + true stage-2 hyperparameters")), indent=1))
    print("wrote", out, "cube", T.shape, "depth range", float(ds['depth'].min()), float(ds['depth'].max()))


def score(out, names):
    from inversion_sst_gp.spacetime import tiling as TL
    out = pathlib.Path(out)
    tr = xr.open_dataset(out / "truth.nc")
    dep = tr["depth"].values
    meta, specs = TL.read_manifest(out / "manifest.json")
    for nm in names:
        m = xr.open_dataset(out / f"merged_{nm}.nc")
        have = np.isfinite(m["u"].values)
        # cells in the CORES of the tiles that were run
        core = np.zeros(have.shape, bool)
        for s in specs:
            if (out / f"tiles_{nm}" / f"tile_{s.id}.nc").exists():
                core[s.ck0:s.ck1, s.cj0:s.cj1, s.ci0:s.ci1] = True
        core &= have
        ut, vt = tr["u_true"].values, tr["v_true"].values
        uT, vT = tr["u_tide_true"].values, tr["v_tide_true"].values
        ue, ve = ut - uT, vt - vT
        for reg, mask2 in (("deep >= 200 m", dep >= 200), ("shelf < 200 m", dep < 200)):
            sel = core & mask2[None]
            if sel.sum() == 0:
                continue
            e = np.r_[(m["u"].values - ut)[sel], (m["v"].values - vt)[sel]]
            ev = np.var(np.r_[ue[sel], ve[sel]])
            etide = np.sqrt(np.mean(np.r_[(m["u_tide"].values - uT)[sel], (m["v_tide"].values - vT)[sel]] ** 2))
            print(f"[{nm}] {reg}: {sel.sum()} cells, total-velocity rmse {np.sqrt(np.mean(e ** 2)):.4f} m/s, skill vs eddy variance {1 - np.mean(e ** 2) / ev:.3f}; tide-term error {etide:.4f}, tide rms {np.sqrt(np.mean(np.r_[uT[sel], vT[sel]] ** 2)):.3f}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "make":
        make(a[1], int(a[2]) if len(a) > 2 else 10, int(a[3]) if len(a) > 3 else 30)
    else:
        score(a[1], a[2:])

#!/usr/bin/env python
"""Step 2: the regional barotropic tide (M2, S2, K1, O1 by default) fitted on the whole cube.

  python hpc/run_tide.py --cube cube.nc --bathy GEBCO.nc --out-dir tide/ [--min-depth 200 --block 5 --spacing 80e3 --time-stride 2 --constituents M2,S2]
  python hpc/run_tide.py --cube test_3x3.nc --out-dir tide/ --min-depth 100            # depth taken from the cube file's `depth` variable (NaN = masked out)

Writes tide/tide_model.pkl (read by run_tile.py / fit_hyper.py), tide/depth.nc (the depth used), tide/tide_constituents.nc (amplitude and phase maps on the block grid with the support
mask) and tide/summary.json. Rows shallower than --min-depth are not used; the fit is only valid in deep water (support mask).
"""
import argparse
import json
import pathlib

import numpy as np
import xarray as xr

import _util


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cube", required=True)
    ap.add_argument("--bathy", help="GEBCO-style NetCDF (elevation, negative below sea level); default: the `depth` variable of the cube file")
    ap.add_argument("--depth-var", default="depth")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-depth", type=float, default=200.0); ap.add_argument("--block", type=int, default=5)
    ap.add_argument("--spacing", type=float, default=80e3); ap.add_argument("--smooth", type=float, default=1e-3)
    ap.add_argument("--constituents", default="M2,S2"); ap.add_argument("--nuisance-harmonics", type=int, default=4)
    ap.add_argument("--nuisance-t-days", type=float, default=None, help="day-by-day heating amplitudes: knot spacing (days) of the piecewise-linear nuisance amplitudes (default: one fixed cycle)")
    ap.add_argument("--nuisance-spacing", type=float, default=None, help="spatial spline spacing of the nuisance term (m; default 3 x --spacing)")
    ap.add_argument("--time-stride", type=int, default=1); ap.add_argument("--chunk-hours", type=int, default=120)
    ap.add_argument("--n-jobs", type=int, default=1); ap.add_argument("--chunk-rows", type=int, default=20000)
    a = ap.parse_args()
    _util.set_threads()
    from inversion_sst_gp.spacetime import bathymetry as BA, regional as RG
    out = pathlib.Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    cube = xr.open_dataset(a.cube)
    lon, lat = cube["lon"].values, cube["lat"].values
    if a.bathy:
        depth = BA.load_depth(a.bathy, lon, lat)
    else:
        depth = np.asarray(cube[a.depth_var].values, float)            # NaN = masked out
    xr.Dataset({"depth": (("lat", "lon"), depth.astype(np.float32), {"units": "m"})}, coords={"lat": lat, "lon": lon}).to_netcdf(out / "depth.nc")
    _util.log(f"depth on the cube grid: {np.mean(depth >= a.min_depth):.2f} of the pixels at least {a.min_depth:.0f} m deep, {np.mean(~np.isfinite(depth)):.2f} masked (NaN); range {np.nanmin(depth):.0f}-{np.nanmax(depth):.0f} m")
    cfg = RG.TideConfig(constituents=tuple(a.constituents.split(",")), min_depth=a.min_depth, block=a.block, spacing=a.spacing, smooth=a.smooth, nuisance_harmonics=a.nuisance_harmonics,
                        nuisance_t_spacing=None if a.nuisance_t_days is None else a.nuisance_t_days * 86400.0, nuisance_spacing=a.nuisance_spacing,
                        time_stride=a.time_stride, chunk_hours=a.chunk_hours, n_jobs=a.n_jobs, chunk_rows=a.chunk_rows)
    model = RG.fit_region(cube, depth, cfg, log=_util.log)
    model.save(out / "tide_model.pkl")
    b = a.block
    ds = model.constituents(lon[b // 2::b], lat[b // 2::b])
    ds.to_netcdf(out / "tide_constituents.nc")
    summ = dict(info=model.info, config={k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(cfg).items()})
    for nm in model.fit.names:
        sup = ds["supported"].values.astype(bool)
        summ[nm] = dict(Au_median=float(np.median(ds[f"{nm}_Au"].values[sup])), Av_median=float(np.median(ds[f"{nm}_Av"].values[sup])))
    (out / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
    _util.log("tide stage done: " + ", ".join(f"{nm} {summ[nm]['Au_median']:.3f}/{summ[nm]['Av_median']:.3f} m/s (u/v median amplitude)" for nm in model.fit.names))


if __name__ == "__main__":
    main()

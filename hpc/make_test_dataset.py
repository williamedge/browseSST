#!/usr/bin/env python
"""Combine real Himawari SST and SUNTANS bathymetry into ONE test file for the HPC (cropped box, 4 km grid, depths shallower than a threshold set to NaN).

  python hpc/make_test_dataset.py \\
      --nc-dir /mnt/c/Users/00099894/Large_Data/Other/browseSST/himawari/nc \\
      --bathy  /mnt/c/Users/00099894/Large_Data/Papers/drifterOI_paper1/data/processed/SUNTANS_bathy_unstructured.nc \\
      --lon 121 124 --lat -14 -11 --stride 2 --reduce stride --mask-depth 100 --out test_3x3_may2026.nc

Output (the cube format every hpc/ script reads):
  sst(time, lat, lon)   K, float32, NaN where the pixel is not valid (quality_level >= 5, SST in range); hourly, regular time axis (missing hours are all-NaN)
  depth(lat, lon)       m, positive down: cell mean of the bathymetry points in each 4 km cell, NaN where shallower than --mask-depth or where the bathymetry has no point (this is the
                        variable the tide stage reads: NaN pixels are excluded from the tide equations)
  depth_raw(lat, lon)   the same without the shallow mask (NaN only where the bathymetry has no point), kept so that another threshold can be chosen later
SUNTANS file layout: 1-D unstructured cell centres `xv` (lon), `yv` (lat), `dv` (depth, m); override the names with --bathy-vars.
Also writes <out>_quicklook.png (mean SST, depth with the mask, coverage per day) and prints the coverage per day.
"""
import argparse
import pathlib

import numpy as np
import xarray as xr

import _util


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nc-dir", required=True); ap.add_argument("--bathy", required=True)
    ap.add_argument("--bathy-vars", default="xv,yv,dv", help="names of the lon, lat, depth variables of the scattered bathymetry file")
    ap.add_argument("--lon", nargs=2, type=float, default=[121.0, 124.0]); ap.add_argument("--lat", nargs=2, type=float, default=[-14.0, -11.0])
    ap.add_argument("--t0"); ap.add_argument("--t1")
    ap.add_argument("--stride", type=int, default=2); ap.add_argument("--reduce", default="stride", choices=("stride", "mean"))
    ap.add_argument("--min-frac", type=float, default=0.5, help="with --reduce mean: fraction of the block that must be valid")
    ap.add_argument("--qmin", type=int, default=5)
    ap.add_argument("--mask-depth", type=float, default=100.0, help="depths shallower than this (m) are set to NaN in `depth`")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    _util.set_threads()
    from inversion_sst_gp.spacetime import bathymetry as BA, himawari as HW

    t0 = np.datetime64(a.t0) if a.t0 else None
    t1 = np.datetime64(a.t1) if a.t1 else None
    cube = HW.load_cube(a.nc_dir, t0, t1, 3600, lonlim=a.lon, latlim=a.lat, qmin=a.qmin, stride=a.stride, reduce=a.reduce, min_frac=a.min_frac)
    lon, lat = cube["lon"].values, cube["lat"].values
    _util.log(f"cube {dict(cube.sizes)} from {cube.attrs['n_files']} files ({cube.attrs['n_missing_hours']} missing hours); lon {lon[0]:.3f}..{lon[-1]:.3f}, lat {lat[0]:.3f}..{lat[-1]:.3f}")

    vx, vy, vd = a.bathy_vars.split(",")
    with xr.open_dataset(a.bathy) as b:
        px, py, pd_ = (np.asarray(b[v].values, float) for v in (vx, vy, vd))
    depth_raw = BA.grid_scattered_depth(px, py, pd_, lon, lat)
    depth = np.where(depth_raw >= a.mask_depth, depth_raw, np.nan)
    ntot = depth.size
    _util.log(f"bathymetry: {np.isfinite(depth_raw).sum()} of {ntot} cells have mesh points ({np.isfinite(depth_raw).mean():.2f}); depth {np.nanmin(depth_raw):.0f}-{np.nanmax(depth_raw):.0f} m; "
              f"{np.isfinite(depth).mean():.2f} of the cells are at least {a.mask_depth:.0f} m deep and kept")

    sst = cube["sst"]
    ocean = HW.ocean_mask(cube)
    cov = HW.coverage(cube, ocean)
    cov_deep = HW.coverage(cube, ocean & np.isfinite(depth))
    _util.log(f"SST valid fraction of the ocean pixels: mean {cov.mean():.2f} (deep kept pixels: {cov_deep.mean():.2f})")
    for d in range(0, cov.size, 24):
        _util.log(f"  day {d // 24:2d} {str(cube['time'].values[d])[:10]}  coverage {cov[d:d + 24].mean():.2f} (hourly min {cov[d:d + 24].min():.2f}, max {cov[d:d + 24].max():.2f})")

    ds = xr.Dataset(
        {"sst": sst,
         "depth": (("lat", "lon"), depth.astype(np.float32), {"units": "m", "long_name": f"water depth, NaN where shallower than {a.mask_depth:g} m or no bathymetry"}),
         "depth_raw": (("lat", "lon"), depth_raw.astype(np.float32), {"units": "m", "long_name": "water depth, cell mean of the bathymetry points; NaN where none"})},
        coords={"time": cube["time"], "lat": cube["lat"], "lon": cube["lon"]})
    ds.attrs.update(cube.attrs)
    ds.attrs.update(source_sst=str(a.nc_dir), source_bathy=str(a.bathy), box_lon=list(a.lon), box_lat=list(a.lat), mask_depth_m=float(a.mask_depth),
                    description="Himawari-9 L3C SST (quality>=5) + gridded SUNTANS depth; made by hpc/make_test_dataset.py")
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    enc = {"sst": dict(zlib=True, complevel=4, dtype="float32", _FillValue=np.float32(np.nan), chunksizes=(min(24, cube.sizes["time"]), cube.sizes["lat"], cube.sizes["lon"])),
           "depth": dict(zlib=True, _FillValue=np.float32(np.nan)), "depth_raw": dict(zlib=True, _FillValue=np.float32(np.nan))}
    ds.to_netcdf(out, encoding=enc)
    _util.log(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))
        m = ax[0, 0].pcolormesh(lon, lat, np.nanmean(sst.values, axis=0), shading="auto"); fig.colorbar(m, ax=ax[0, 0], label="mean SST (K)"); ax[0, 0].set_title("mean SST")
        m = ax[0, 1].pcolormesh(lon, lat, depth_raw, shading="auto", cmap="viridis_r", vmin=0, vmax=np.nanpercentile(depth_raw, 99)); fig.colorbar(m, ax=ax[0, 1], label="depth (m)")
        ax[0, 1].contour(lon, lat, np.where(np.isfinite(depth_raw), depth_raw, 0), [a.mask_depth], colors="r", linewidths=1); ax[0, 1].set_title(f"depth; red = {a.mask_depth:g} m (shallower: NaN in `depth`)")
        ax[1, 0].plot(cube["time"].values, cov, label="all ocean pixels"); ax[1, 0].plot(cube["time"].values, cov_deep, label="kept (deep) pixels"); ax[1, 0].set_ylim(0, 1.02)
        ax[1, 0].legend(); ax[1, 0].set_title("valid fraction per hour"); ax[1, 0].tick_params(axis="x", rotation=30)
        k = int(np.argmax(cov)); m = ax[1, 1].pcolormesh(lon, lat, sst.values[k], shading="auto"); fig.colorbar(m, ax=ax[1, 1], label="SST (K)"); ax[1, 1].set_title(f"best-coverage hour {str(cube['time'].values[k])[:13]}")
        for x in ax.ravel()[[0, 1, 3]]:
            x.set_xlabel("lon"); x.set_ylabel("lat")
        fig.tight_layout(); png = out.with_name(out.stem + "_quicklook.png"); fig.savefig(png, dpi=110)
        _util.log(f"wrote {png}")
    except Exception as e:  # noqa: BLE001
        _util.log(f"quick-look figure skipped: {e!r}")


if __name__ == "__main__":
    main()

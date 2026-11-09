#!/usr/bin/env python
"""Step 0: per-hour Himawari NetCDFs (pysatview output) -> ONE cube file (time, lat, lon) that every later job opens lazily.

  python hpc/make_cube.py --nc-dir DIR --t0 2026-01-01T00 --t1 2026-01-31T23 --lon 120 123 --lat -17 -14 --stride 2 --reduce mean --out cube.nc

Reduction to the working grid (the plan: 4 km from the 2 km product): `--stride 2 --reduce mean` averages 2 x 2 blocks (lower noise), `--reduce stride` subsamples.
Prints the coverage per day so a window of good coverage can be chosen (the cloud-free periods; clouds are not modelled).
"""
import argparse

import numpy as np

import _util


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nc-dir", required=True)
    ap.add_argument("--t0"); ap.add_argument("--t1")
    ap.add_argument("--lon", nargs=2, type=float); ap.add_argument("--lat", nargs=2, type=float)
    ap.add_argument("--stride", type=int, default=2); ap.add_argument("--reduce", default="mean", choices=("mean", "stride"))
    ap.add_argument("--min-frac", type=float, default=0.5, help="fraction of valid pixels a block needs when --reduce mean")
    ap.add_argument("--qmin", type=int, default=5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    _util.set_threads()
    from inversion_sst_gp.spacetime import himawari as HW
    t0 = np.datetime64(a.t0) if a.t0 else None
    t1 = np.datetime64(a.t1) if a.t1 else None
    cube = HW.load_cube(a.nc_dir, t0, t1, 3600, lonlim=a.lon, latlim=a.lat, qmin=a.qmin, stride=a.stride, reduce=a.reduce, min_frac=a.min_frac)
    _util.log(f"cube {dict(cube.sizes)}, {cube.attrs['n_files']} files, {cube.attrs['n_missing_hours']} missing hours")
    cov = HW.coverage(cube, HW.ocean_mask(cube))
    for d in range(0, cov.size, 24):
        _util.log(f"  day {d // 24:3d}  coverage {cov[d:d + 24].mean():.2f}  (min hour {cov[d:d + 24].min():.2f})")
    cube.to_netcdf(a.out, encoding={"sst": dict(zlib=True, complevel=4, dtype="float32", _FillValue=np.float32(np.nan), chunksizes=(min(24, cube.sizes["time"]), cube.sizes["lat"], cube.sizes["lon"]))})
    _util.log(f"wrote {a.out}")


if __name__ == "__main__":
    main()

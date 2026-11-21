#!/usr/bin/env python
"""Step 5: blend the tile files into the cube's grid (partition of unity: weight 1 on a core, sin^2 taper over the buffers).

  python hpc/merge.py --cube cube.nc --manifest manifest.json --tiles-dir tiles/ --out merged.nc

Sparse-observation flag (from the cube, not the tiles): `obs_frac_local` = fraction of valid SST in a (+-`--sparse-hours` h) x (`--sparse-px` x `--sparse-px` px) neighbourhood of every grid point, and `sparse` = 1 where it is
below `--sparse-thresh` (default 0.5). The velocity there rests on little data (the extremes are 5x more frequent: first full 2023 run). `--mask-sparse` also sets the velocity / S fields to NaN there.

The blended variances are the convex combination of the tile variances (conservative where tile errors are correlated); they omit the stage-1 and tide uncertainty.
"""
import argparse

import _util


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cube", required=True); ap.add_argument("--manifest", required=True); ap.add_argument("--tiles-dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--fields", default="auto", help="comma list, or auto = every field of the tile files except Tt, var_S, tide_supported (includes the block outputs u_m2 ... if present)")
    ap.add_argument("--sparse-px", type=int, default=5); ap.add_argument("--sparse-hours", type=int, default=3); ap.add_argument("--sparse-thresh", type=float, default=0.5)
    ap.add_argument("--mask-sparse", action="store_true", help="set the velocity / S / variance fields to NaN where `sparse` = 1")
    a = ap.parse_args()
    import xarray as xr
    from inversion_sst_gp.spacetime import pipeline as P, tiling as TL
    cube = xr.open_dataset(a.cube)
    ds = P.merge_tiles(TL.read_manifest(a.manifest), a.tiles_dir, cube, None, fields="auto" if a.fields == "auto" else tuple(a.fields.split(",")), log=_util.log)
    import numpy as np
    from scipy.ndimage import uniform_filter
    frac = uniform_filter(np.isfinite(cube["sst"].values).astype(np.float32), size=(2 * a.sparse_hours + 1, a.sparse_px, a.sparse_px), mode="nearest")
    ds["obs_frac_local"] = (("time", "lat", "lon"), frac.astype(np.float32), {"long_name": f"fraction of valid SST within +-{a.sparse_hours} h and {a.sparse_px} x {a.sparse_px} px"})
    ds["sparse"] = (("time", "lat", "lon"), (frac < a.sparse_thresh).astype(np.int8), {"long_name": f"1 where obs_frac_local < {a.sparse_thresh}"})
    if a.mask_sparse:
        for f in ds.data_vars:
            if f not in ("T", "obs_frac_local", "sparse"):
                ds[f] = ds[f].where(~(frac < a.sparse_thresh))
    ds.attrs.update(sparse_px=a.sparse_px, sparse_hours=a.sparse_hours, sparse_thresh=a.sparse_thresh, mask_sparse=int(a.mask_sparse))
    ds.to_netcdf(a.out, encoding={k: dict(zlib=True, complevel=4) for k in ds.data_vars})
    _util.log(f"merged {ds.attrs['n_tiles']} of {ds.attrs['n_tiles_expected']} tiles -> {a.out}")


if __name__ == "__main__":
    main()

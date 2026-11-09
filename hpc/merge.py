#!/usr/bin/env python
"""Step 5: blend the tile files into the cube's grid (partition of unity: weight 1 on a core, sin^2 taper over the buffers).

  python hpc/merge.py --cube cube.nc --manifest manifest.json --tiles-dir tiles/ --out merged.nc

The blended variances are the convex combination of the tile variances (conservative where tile errors are correlated); they omit the stage-1 and tide uncertainty.
"""
import argparse

import _util


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cube", required=True); ap.add_argument("--manifest", required=True); ap.add_argument("--tiles-dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--fields", default="T,u,v,u_eddy,v_eddy,u_tide,v_tide,var_u,var_v,S")
    a = ap.parse_args()
    import xarray as xr
    from inversion_sst_gp.spacetime import pipeline as P, tiling as TL
    cube = xr.open_dataset(a.cube)
    ds = P.merge_tiles(TL.read_manifest(a.manifest), a.tiles_dir, cube, a.out, fields=tuple(a.fields.split(",")), log=_util.log)
    _util.log(f"merged {ds.attrs['n_tiles']} of {ds.attrs['n_tiles_expected']} tiles -> {a.out}")


if __name__ == "__main__":
    main()

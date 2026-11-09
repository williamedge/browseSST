#!/usr/bin/env python
"""Step 1: tile layout for a cube -> manifest.json (+ tiles.csv with the size and observed fraction of each tile, + sample_tiles.txt for the hyperparameter fits).

  python hpc/make_manifest.py --cube cube.nc --core 168 28 28 --buffer 12 4 4 --out manifest.json --n-sample 4

--core / --buffer are (time h, lat cells, lon cells). A buffer exists only on the sides that border another tile; weights are 1 on the core and a sin^2 taper over the buffer.
The SLURM array for run_tile.py is 0..N-1 over the tile list printed at the end.
"""
import argparse
import csv

import numpy as np
import xarray as xr

import _util


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cube", required=True)
    ap.add_argument("--core", nargs=3, type=int, default=[168, 28, 28]); ap.add_argument("--buffer", nargs=3, type=int, default=[12, 4, 4])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-sample", type=int, default=4, help="number of sample tiles (best coverage, away from the domain edge) for the hyperparameter fits")
    a = ap.parse_args()
    from inversion_sst_gp.spacetime import tiling as TL
    cube = xr.open_dataset(a.cube)
    shape = (cube.sizes["time"], cube.sizes["lat"], cube.sizes["lon"])
    specs = TL.make_tiles(shape, tuple(a.core), tuple(a.buffer))
    sst = cube["sst"]
    rows = []
    for s in specs:
        block = np.isfinite(sst.isel(time=slice(s.k0, s.k1), lat=slice(s.j0, s.j1), lon=slice(s.i0, s.i1)).values)
        ocean = block.any(axis=0)
        frac = float(block[:, ocean].mean()) if ocean.any() else 0.0
        edge = int(s.cj0 == 0) + int(s.cj1 == shape[1]) + int(s.ci0 == 0) + int(s.ci1 == shape[2])
        rows.append(dict(id=s.id, n_grid=s.n_points, shape="x".join(map(str, s.shape)), n_obs=int(block.sum()), obs_frac=round(frac, 3), ocean_frac=round(float(ocean.mean()), 3), edge_sides=edge))
    meta = dict(cube=str(a.cube), shape=list(shape), core=a.core, buffer=a.buffer, lon0=float(cube["lon"][0]), lat0=float(cube["lat"][0]), t0=str(cube["time"].values[0]))
    TL.write_manifest(a.out, specs, meta)
    csv_path = str(a.out).rsplit(".", 1)[0] + "_tiles.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    # sample tiles: best coverage and mostly ocean, preferring tiles away from the domain edge and in distinct places (spread over space / time)
    cand = sorted(rows, key=lambda r: (r["edge_sides"], -r["obs_frac"] * r["ocean_frac"]))
    ids = [r["id"] for r in cand[:a.n_sample]]
    open(str(a.out).rsplit(".", 1)[0] + "_sample_tiles.txt", "w").write(",".join(ids) + "\n")
    n = [r["n_grid"] for r in rows]
    _util.log(f"{len(specs)} tiles of {min(n)}-{max(n)} grid points (cube {shape}); SLURM array 0-{len(specs) - 1}; manifest {a.out}, table {csv_path}")
    _util.log(f"sample tiles for the hyperparameter fits: {','.join(ids)}")


if __name__ == "__main__":
    main()

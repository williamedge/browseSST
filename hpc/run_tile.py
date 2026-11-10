#!/usr/bin/env python
"""Step 4: one tile = one job (a SLURM array task).

  python hpc/run_tile.py --cube cube.nc --manifest manifest.json --hyper hyper.json --tide tide/tide_model.pkl --out-dir tiles/ --array-index $SLURM_ARRAY_TASK_ID

Writes tiles/tile_<id>.nc (the whole buffered block: latent T, eddy / tide / total velocity, S, stage-2 variances, tide support mask) unless it exists (use --force to redo).
Without --tide the tide is not removed (the GP absorbs it). A tile with too few observations is skipped (no file; the merge leaves a NaN hole where it alone covers).
"""
import argparse
import pathlib
import sys

import _util


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cube", required=True); ap.add_argument("--manifest", required=True); ap.add_argument("--hyper", required=True)
    ap.add_argument("--tide"); ap.add_argument("--tide-from-cube", action="store_true", help="take the tide from the cube file's u_tide / v_tide variables (external product, e.g. FES2022)"); ap.add_argument("--tile"); ap.add_argument("--tiles"); ap.add_argument("--array-index", type=int)
    ap.add_argument("--out-dir", required=True); ap.add_argument("--force", action="store_true")
    ap.add_argument("--nnum", type=int, default=100); ap.add_argument("--nnum-pred", type=int, default=200)
    ap.add_argument("--noise-mode", default="fixed", choices=("fixed", "mle")); ap.add_argument("--min-obs", type=int, default=2000)
    ap.add_argument("--min-cover", type=float, default=0.5, help="skip a tile if fewer than this fraction of its pixels inside the depth range (finite cube depth) are observed; 0 = off")
    ap.add_argument("--tide-mask", default="supported", choices=("supported", "none"))
    _util.add_prior_args(ap)
    a = ap.parse_args()
    _util.set_threads()
    import xarray as xr
    from inversion_sst_gp.spacetime import pipeline as P, regional as RG, tiling as TL
    meta, specs = TL.read_manifest(a.manifest)
    tid = _util.tile_id_from_args(a, [s.id for s in specs])
    out_path = pathlib.Path(a.out_dir) / f"tile_{tid}.nc"
    if out_path.exists() and not a.force:
        _util.log(f"{out_path} exists; nothing to do")
        return
    spec = next(s for s in specs if s.id == tid)
    cube = xr.open_dataset(a.cube)
    td = TL.extract_tile(cube, spec)
    hyper = P.Hyper.load(a.hyper)
    tide = RG.ArrayTide(cube) if a.tide_from_cube else (RG.TideModel.load(a.tide) if a.tide else None)
    prior = _util.make_prior(a)
    cfg = P.PipelineConfig(nnum=a.nnum, nnum_pred=a.nnum_pred, noise_mode=a.noise_mode, min_obs=a.min_obs, min_cover=a.min_cover, tide_mask=a.tide_mask, **({} if prior is None else dict(prior=prior)))
    _util.log(f"tile {tid}: block {td.shape} ({td.x.size} points), {td.n_obs} observed, tide {'yes' if tide else 'no'}")
    out = P.run_tile(td, hyper, cfg, tide=tide, log=_util.log)
    if out is None:
        sys.exit(0)
    P.write_tile(out_path, td, out)
    _util.log(f"wrote {out_path}")


if __name__ == "__main__":
    main()

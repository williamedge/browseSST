#!/usr/bin/env python
"""Step 3: estimate the shared hyperparameters on sample tiles (one SLURM array task per tile), then pool them with pool_hyper.py.

  python hpc/fit_hyper.py --cube cube.nc --manifest manifest.json --tiles id1,id2,id3,id4 --array-index $SLURM_ARRAY_TASK_ID --tide tide/tide_model.pkl --out-dir hyper/

Stage 1 (latent-T GP) is fitted by its marginal likelihood, then the stage-2 velocity-prior parameters from the tide-removed response; the start / prior centre is --hyper-in (default:
the placeholder defaults). With --prior browsegp the stage-2 velocity parameters can be fitted too (--free-vel rot,div,fast), centred on browseGP's values; use hpc/hyper_browsegp_start.json as --hyper-in. A fit is expensive (hundreds of likelihood evaluations on ~10^5 points): use only a few tiles, run with --maxiter1 / --maxiter2 of 20-40 and check `converged`.
"""
import argparse
import pathlib

import _util


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cube", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--tile"); ap.add_argument("--tiles"); ap.add_argument("--array-index", type=int)
    ap.add_argument("--tide"); ap.add_argument("--hyper-in"); ap.add_argument("--tide-from-cube", action="store_true", help="take the tide from the cube file's u_tide / v_tide variables")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--maxiter1", type=int, default=25); ap.add_argument("--maxiter2", type=int, default=25)
    ap.add_argument("--nnum", type=int, default=100); ap.add_argument("--noise-mode", default="fixed", choices=("fixed", "mle"))
    ap.add_argument("--fit-hours", type=int, default=0, help="fit on the central N hours of the tile only (cost is linear in the number of points; 0 = the whole buffered tile)")
    _util.add_prior_args(ap)
    a = ap.parse_args()
    _util.set_threads()
    import xarray as xr
    from inversion_sst_gp.spacetime import pipeline as P, regional as RG, tiling as TL
    meta, specs = TL.read_manifest(a.manifest)
    tid = _util.tile_id_from_args(a, [s.id for s in specs])
    spec = next(s for s in specs if s.id == tid)
    cube = xr.open_dataset(a.cube)
    if a.fit_hours and a.fit_hours < spec.k1 - spec.k0:
        import dataclasses
        k0 = spec.k0 + ((spec.k1 - spec.k0) - a.fit_hours) // 2
        spec = dataclasses.replace(spec, k0=k0, k1=k0 + a.fit_hours, ck0=k0, ck1=k0 + a.fit_hours)
    td = TL.extract_tile(cube, spec)
    tide = RG.ArrayTide(cube) if a.tide_from_cube else (RG.TideModel.load(a.tide) if a.tide else None)
    h0 = P.Hyper.load(a.hyper_in) if a.hyper_in else P.Hyper.default()
    prior = _util.make_prior(a)
    cfg = P.PipelineConfig(nnum=a.nnum, nnum_pred=2 * a.nnum, noise_mode=a.noise_mode, **({} if prior is None else dict(prior=prior)))
    _util.log(f"tile {tid}: {td.shape}, {td.n_obs} observed of {td.x.size}; tide {'yes' if tide else 'no'}")
    h = P.fit_hyper(td, h0, cfg, tide=tide, maxiter1=a.maxiter1, maxiter2=a.maxiter2, log=_util.log)
    out = pathlib.Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    h.save(out / f"hyper_{tid}.json")
    _util.log(f"wrote {out / f'hyper_{tid}.json'}")


if __name__ == "__main__":
    main()

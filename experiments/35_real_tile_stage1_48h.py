"""
Experiment 35: stage-1 (latent-T GP) fit on a REAL 48 h tile, to measure the temporal noise level.

Exp 27 fitted sigma_T = 0.008-0.013 K from four hours (spatial structure only). Exp 34's diagnostics on the 28 d test cube suggest the hour-to-hour pixel noise is larger (box-mean-removed hourly
differences have rms 0.068 K, so white-noise sigma_T <= 0.048 K). Here the two-component 5/2:5/2 T-GP is fitted to 48 hourly snapshots of a 28 x 28 px tile (days 3-5, coverage 0.95-0.99) with the
noise variance's prior loosened (log-normal sd 3 about (0.03 K)^2) so that the likelihood, not the prior, decides; everything else as in the pipeline (split bounds, prior sd 1.5).
Run: python experiments/35_real_tile_stage1_48h.py <cube_file> [hour0] [j0] [i0] [maxiter] [stride_pixels]
"""
import json, pathlib, sys, time, warnings
import numpy as np
import xarray as xr

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
from inversion_sst_gp.spacetime import pipeline as P, tiling as TL
from inversion_sst_gp.spacetime.fit import T_KERNEL_SLOTS, fit


def main(path, h0=72, j0=24, i0=24, maxiter=30, nt=48, n=28):
    cube = xr.open_dataset(path)
    spec = TL.TileSpec("real48", h0, h0 + nt, j0, j0 + n, i0, i0 + n, h0, h0 + nt, j0, j0 + n, i0, i0 + n)
    td = TL.extract_tile(cube, spec)
    print(f"tile {td.shape}: {td.n_obs} observed of {td.x.size} ({td.n_obs / td.x.size:.2f}); T anomaly sd {np.nanstd(td.T):.3f} K", flush=True)
    xo, yo, to, To = td.observed()
    cfg = P.PipelineConfig()
    h1 = dict(P.DEFAULT_STAGE1); h1['noise'] = 0.03 ** 2
    gp = P.build_stage1(xo, yo, to, To, h1, cfg)
    centre = dict(P.STAGE1_CENTRE); centre['noise'] = 0.03 ** 2
    free = list(P.STAGE1_KEYS)
    sd = {k: 1.5 for k in free}; sd['noise'] = 3.0
    t0 = time.time()
    res = fit(gp, centre, free, bounds=P.STAGE1_BOUNDS, prior_sd_log=sd, maxiter=maxiter, verbose=False, slots=T_KERNEL_SLOTS, start={k: h1[k] for k in free})
    r = res.params
    print(f"fit {res.n_eval} evals {time.time() - t0:.0f}s success={res.success} loglik {res.loglik:.1f}", flush=True)
    print(f"  eddy: sigma {r['sigma1']:.3f} K, l {r['l1'] / 1e3:.1f} km, tau {r['tau1'] / 3600:.1f} h | fast: sigma {r['sigma2']:.3f} K, l {r['l2'] / 1e3:.1f} km, tau {r['tau2'] / 3600:.1f} h", flush=True)
    print(f"  noise sd {np.sqrt(r['noise']):.4f} K; mean-term scale {r['lr:mean']:.3f}, diurnal scale {r['lr:diurnal']:.4f}", flush=True)
    out = ROOT / "data" / f"exp35_real_stage1_h{h0}_j{j0}_i{i0}.json"
    out.write_text(json.dumps(dict(params={k: float(r[k]) for k in free}, loglik=float(res.loglik), n_eval=int(res.n_eval), success=bool(res.success), tile=[h0, j0, i0, nt, n], n_obs=int(td.n_obs)), indent=1))
    print("wrote", out)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], int(a[1]) if len(a) > 1 else 72, int(a[2]) if len(a) > 2 else 24, int(a[3]) if len(a) > 3 else 24, int(a[4]) if len(a) > 4 else 30)

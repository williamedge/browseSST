"""
Experiment 36: what stage-2 (equation) noise do the real products imply?

Takes the stage-1 hyperparameters fitted on the real 48 h tile of exp 35, computes the latent T / dT/dt / gradients on that tile, builds the velocity GP (default Helmholtz-Matern prior at the
placeholder 'fit A' hyperparameters, no tide) and fits the scalar noise variance by maximum likelihood. Reports it in scaled units (data are dT/dt / 1e-6) and in K/s, next to the rms of dT/dt and
of u.grad T for a 0.1 m/s flow, so the size of the equation error is visible. The result is written into hpc/hyper_real_start.json as the stage-1 values and the stage-2 noise.
Run: python experiments/36_real_stage2_noise.py <cube_file> [hour0] [j0] [i0]
"""
import json, pathlib, sys, time, warnings
import numpy as np
import xarray as xr

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
from inversion_sst_gp.spacetime import pipeline as P, tiling as TL


def main(path, h0=72, j0=24, i0=24, nt=48, n=28):
    cube = xr.open_dataset(path)
    spec = TL.TileSpec("real48", h0, h0 + nt, j0, j0 + n, i0, i0 + n, h0, h0 + nt, j0, j0 + n, i0, i0 + n)
    td = TL.extract_tile(cube, spec)
    r = json.loads((ROOT / "data" / f"exp35_real_stage1_h{h0}_j{j0}_i{i0}.json").read_text())['params']
    hyper = P.Hyper.default()
    hyper.stage1 = {k: float(r[k]) for k in P.STAGE1_KEYS}
    cfg = P.PipelineConfig()
    xo, yo, to, To = td.observed()
    t0 = time.time()
    gp1 = P.build_stage1(xo, yo, to, To, hyper.stage1, cfg)
    Tm, Ttm, vTt = P.stage1_products(td, gp1, cfg)
    print(f"stage 1 products {time.time() - t0:.0f}s", flush=True)
    zero = np.zeros(td.x.size)
    idx, Tx, Ty, z = P.stage2_rows(td, Tm, Ttm, vTt, zero, zero, cfg)
    print(f"stage-2 rows {idx.size}; rms dT/dt {np.sqrt(np.mean(Ttm[idx] ** 2)):.2e} K/s; rms |grad T| {np.sqrt(np.mean(Tx[idx] ** 2 + Ty[idx] ** 2)):.2e} K/m; "
          f"rms u.gradT for |u| = 0.1 m/s: {0.1 * np.sqrt(np.mean(Tx[idx] ** 2 + Ty[idx] ** 2)) / np.sqrt(2):.2e} K/s", flush=True)
    t0 = time.time()
    gp2 = P.build_stage2(td, idx, Tx, Ty, z, hyper.stage2, cfg)
    noise = P.fit_noise_mle(gp2)
    ll = gp2.loglik()
    print(f"stage-2 noise by ML {noise:.1f} (scaled units) = rms {np.sqrt(noise) * P.ZS:.2e} K/s ({time.time() - t0:.0f}s), loglik {ll:.1f}; the placeholder was {hyper.stage2['noise']}", flush=True)
    hyper.stage2['noise'] = float(noise)
    hyper.meta = dict(source=f"exp 35 stage-1 fit on the real tile (hours {h0}-{h0 + nt}, rows {j0}-{j0 + n}, cols {i0}-{i0 + n}; not converged, eddy tau at its 12 h bound) + exp 36 ML stage-2 noise; "
                             "stage-2 velocity parameters are the placeholder 'fit A'")
    out = ROOT / "hpc" / "hyper_real_start.json"
    hyper.save(out)
    print("wrote", out)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], int(a[1]) if len(a) > 1 else 72, int(a[2]) if len(a) > 2 else 24, int(a[3]) if len(a) > 3 else 24)

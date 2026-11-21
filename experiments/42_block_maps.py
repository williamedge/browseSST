"""
Experiment 42: maps of the velocity blocks (rotational, M2 divergent wave, fast) at ONE time step, from the full 2023 HPC run (`run_bg_free_full`).

The tile jobs only store the total eddy velocity, so the block-by-block posterior means are recomputed here: the same chain as the HPC tile job (stage-1 latent T with the pooled `hyper.json`, 4th-order gradients, regional
tide hook, stage-2 velocity GP with the pooled browseGP parameters, 250 m depth mask) on a 72 h x 36 x 36 block (the lon / lat box of tile t06_01_01) around the chosen time, with the stage-2 prediction made only at that
time step. Block contributions use `BrowseGPPrior.cov_params_blocks` (block-weighted prediction-data cross covariance; the blocks sum to the total).

Run: python experiments/42_block_maps.py [YYYY-MM-DDTHH] [run dir]      (default 2023-05-16T12, run_bg_free_full)
"""
import json, pathlib, sys, time, warnings
import numpy as np
import xarray as xr

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pkg"))
from inversion_sst_gp.spacetime import pipeline as P, regional as RG, tiling as TL

DATA = pathlib.Path.home() / "browsesst_data" / "real_test"
BASE = pathlib.Path("/mnt/c/Users/00099894/Large_Data/Other/browseSST")
REPLOT = len(sys.argv) > 1 and sys.argv[1] == "replot"          # python 42_block_maps.py replot : redraw the figure from the saved arrays
when = np.datetime64(sys.argv[1] if len(sys.argv) > 1 and not REPLOT else "2023-05-16T12")
RUN = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else BASE / "run_bg_free_full"
MIN_DEPTH = 250.0
BLOCKS = ("rot", "div", "fast")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(out, lon, lat, label):
    fig, ax = plt.subplots(2, 3, figsize=(17, 10), sharex=True, sharey=True, constrained_layout=True)
    panels = [("rot", "rotational block"), ("div", "M2 divergent-wave block"), ("fast", "fast block"), ("tot", "total eddy"), ("tide", "barotropic tide")]
    st = 2
    for a, (tag, title) in zip(ax.ravel(), panels):
        u, v = out[f"u_{tag}"], out[f"v_{tag}"]
        sp = np.hypot(u, v)
        im = a.pcolormesh(lon, lat, sp, cmap="viridis", vmin=0, vmax=0.5, shading="auto")
        a.quiver(lon[::st], lat[::st], u[::st, ::st], v[::st, ::st], color="w", scale=6, width=0.003)
        a.set_title(f"{title}\nrms {np.sqrt(np.nanmean(sp ** 2)):.2f}, max {np.nanmax(sp):.2f} m/s", fontsize=11)
        a.contour(lon, lat, out["deep"].astype(float), [0.5], colors="r", linewidths=0.8)
    fig.colorbar(im, ax=ax[:, :2], shrink=0.7, location="bottom", pad=0.02, label="speed (m/s)")
    a = ax[1, 2]
    m = a.pcolormesh(lon, lat, np.where(out["obs"], out["T"], np.nan), cmap="turbo", shading="auto")
    a.contourf(lon, lat, np.where(out["obs_local"] < 0.5, 1.0, np.nan), levels=[0.5, 1.5], colors="none", hatches=["///"])
    a.set_title("SST at this hour (K)\nhatched: < 50% observed within +-3 h", fontsize=11)
    fig.colorbar(m, ax=a, shrink=0.8, location="bottom", pad=0.02, label="K")
    fig.suptitle(f"Velocity blocks at {label} UTC (posterior means; red line: 250 m depth limit)")
    fig.savefig(DATA / "exp42_block_maps.png", dpi=110, bbox_inches="tight")
    print("wrote", DATA / "exp42_block_maps.png")



if REPLOT:
    z = np.load(DATA / "exp42_block_maps.npz")
    plot({k: z[k] for k in z.files}, z["lon"], z["lat"], str(z["time"]))
    sys.exit(0)

cube = xr.open_dataset(BASE / "test_bb23_2023.nc")
kt = int(np.argmin(np.abs(cube.time.values - when)))
k0, k1 = kt - 36, kt + 36
spec = TL.TileSpec("map", k0, k1, 24, 60, 24, 60, k0, k1, 24, 60, 24, 60)           # the lon / lat box of tile t06_01_01
td = TL.extract_tile(cube, spec, min_depth=MIN_DEPTH)
hyper = P.Hyper.load(RUN / "hyper.json")
pr = P.BrowseGPPrior.from_results(ROOT / "hpc" / "browsegp_vel.json", blocks=BLOCKS)
cfg = P.PipelineConfig(prior=pr, depth_mask=True, min_cover=0.0)
tide = RG.TideModel.load(RUN / "tide" / "tide_model.pkl")
print(f"time step {np.datetime_as_string(cube.time.values[kt], unit='h')} (index {kt}); block {td.shape}, {td.n_obs} observed, {int(td.deep.sum())} deep pixels", flush=True)

t0 = time.time()
xo, yo, to, To = td.observed()
gp1 = P.build_stage1(xo, yo, to, To, hyper.stage1, cfg)
Tm, Ttm, vTt = P.stage1_products(td, gp1, cfg)
ub, vb, sup = P._tide_for_tile(td, tide, cfg)
idx, Tx, Ty, z = P.stage2_rows(td, Tm, Ttm, vTt, ub, vb, cfg)
h2 = dict(hyper.stage2)
gp2 = P.build_stage2(td, idx, Tx, Ty, z, h2, cfg)
print(f"stage 1 + stage-2 build {time.time() - t0:.0f}s, {idx.size} stage-2 rows", flush=True)

nt, ny, nx = td.shape
sel = np.zeros(td.x.size, bool)
sel.reshape(td.shape)[kt - k0] = True
q = sel & P.deep_points(td, cfg)
qx, qy, qt = td.x[q], td.y[q], td.t[q]
nn = gp2.pred_neighbours(qx, qy, qt, cfg.nnum_pred)
out = {}
for tag, only in (("tot", None),) + tuple((b, (b,)) for b in BLOCKS):
    gp2.update(cov_params=pr.cov_params(h2) if only is None else pr.cov_params_blocks(h2, only))
    for c in "uv":
        a, va = gp2.predict(c, qx, qy, qt, nn_array=nn)
        full = np.full((ny, nx), np.nan); full.reshape(-1)[np.flatnonzero(q) - (kt - k0) * ny * nx] = a
        out[f"{c}_{tag}"] = full
        if tag == "tot":
            fv = np.full((ny, nx), np.nan); fv.reshape(-1)[np.flatnonzero(q) - (kt - k0) * ny * nx] = va; out[f"var_{c}"] = fv
k = kt - k0
out["u_tide"] = np.where(td.deep, ub.reshape(td.shape)[k], np.nan); out["v_tide"] = np.where(td.deep, vb.reshape(td.shape)[k], np.nan)
out["T"] = (Tm + td.Tref).reshape(td.shape)[k]; out["obs"] = td.obs.reshape(td.shape)[k]
out["obs_local"] = td.obs.reshape(td.shape)[max(k - 3, 0):k + 4].mean(axis=0)
out["deep"] = td.deep
np.savez(DATA / "exp42_block_maps.npz", lon=td.lon, lat=td.lat, time=np.datetime_as_string(cube.time.values[kt], unit="h"), **out)

d = max(np.nanmax(np.abs(sum(out[f"{c}_{b}"] for b in BLOCKS) - out[f"{c}_tot"])) for c in "uv")
print(f"max |sum of blocks - total| = {d:.1e} m/s")
print("speed (m/s) over the deep pixels of this time step:      rms     p99     max    posterior sd")
for tag in ("tot",) + BLOCKS:
    sp = np.hypot(out[f"u_{tag}"], out[f"v_{tag}"])
    print(f"  {tag:5s}  {np.sqrt(np.nanmean(sp ** 2)):7.3f} {np.nanpercentile(sp, 99):7.3f} {np.nanmax(sp):7.3f}")
print(f"  tide   {np.sqrt(np.nanmean(np.hypot(out['u_tide'], out['v_tide']) ** 2)):7.3f}")
print(f"  posterior sd of u: median {np.sqrt(np.nanmedian(out['var_u'])):.3f}")

plot(out, td.lon, td.lat, np.datetime_as_string(cube.time.values[kt], unit="h"))

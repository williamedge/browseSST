"""
Experiment 50: does the cloud-correlated amplitude of the velocity blocks come from stage-2 rows inside / next to cloud gaps?

One 36 x 36 px block covering the exp 47 transect, 72 h over the 5 May 2023 burst, run three times with `PipelineConfig.row_mask` = 'all' (current behaviour), 'observed' (rows only at observed pixels) and 'stencil'
(cloud-margin mask: SST observed at t-1, t, t+1 and over the +-2 px cross of the gradient). Everything else is the v0.08 control: its hyper.json (S and noise NOT refitted), regional tide, ring kernel, 250 m depth mask.
Stage 1 (latent T) uses all SST in all three. Predictions are made at every deep grid point in all three.

Run:  python experiments/50_cloud_mask.py run all|observed|stencil      (one process per mask: memory)
      python experiments/50_cloud_mask.py report                        (table + figure)
Env:  BSST_HALF (36), BSST_NNUM (80), BSST_CENTRE (2023-05-05T00)
Writes ~/browsesst_data/real_test/exp50_<centre>_<mask>.npz and exp50_<centre>_cloud_mask.png
"""
import os, pathlib, sys, warnings
import numpy as np, xarray as xr
from scipy.ndimage import uniform_filter

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "pkg"))
BASE = pathlib.Path("/mnt/c/Users/00099894/Large_Data/Other/browseSST"); RUN = BASE / "run_v008_control"
OUT = pathlib.Path.home() / "browsesst_data" / "real_test"; OUT.mkdir(parents=True, exist_ok=True)
HALF = int(os.environ.get("BSST_HALF", 36)); NNUM = int(os.environ.get("BSST_NNUM", 80))
CENTRE = os.environ.get("BSST_CENTRE", "2023-05-05T00")
J0, I0, SIZE = 28, 4, 36                    # block: contains the transect row (13.59 S) and 121.8 - 123.2 E
ROWLAT = -13.6
TRIM_T, TRIM_X = 12, 4                      # scored interior: the usual tile buffers
MASKS = ("all", "observed", "stencil")
FIELDS = ("u_m2", "v_m2", "u_m2w", "v_m2w", "u_eddy", "v_eddy", "u", "v")
tag = CENTRE.replace(":", "")
npz = lambda m: OUT / f"exp50_{tag}_{m}.npz"


def run(mask):
    from inversion_sst_gp.spacetime import pipeline as P, regional as RG, tiling as TL
    cube = xr.open_dataset(BASE / "test_bb23_2023.nc")
    kt = int(np.argmin(np.abs(cube.time.values - np.datetime64(CENTRE)))); k0, k1 = kt - HALF, kt + HALF
    spec = TL.TileSpec("cm", k0, k1, J0, J0 + SIZE, I0, I0 + SIZE, k0, k1, J0, J0 + SIZE, I0, I0 + SIZE)
    td = TL.extract_tile(cube, spec, min_depth=250.0)
    prior = P.BrowseGPPrior.from_results(ROOT / "hpc" / "browsegp_vel_v008.json", blocks=("rot", "div", "d2"), div_kernel="ring")
    cfg = P.PipelineConfig(nnum=NNUM, nnum_pred=2 * NNUM, prior=prior, depth_mask=True, min_cover=0.0, block_outputs=True, row_mask=mask)
    out = P.run_tile(td, P.Hyper.load(RUN / "hyper.json"), cfg, tide=RG.TideModel.load(RUN / "tide" / "tide_model.pkl"))
    np.savez_compressed(npz(mask), **{f: out[f].astype(np.float32) for f in FIELDS}, obs=td.obs.reshape(td.shape), rows=P.cloud_row_mask(td, mask).reshape(td.shape),
                        n_stage2=out["info"]["n_stage2"], lon=td.lon, lat=td.lat, time=td.time)


def report():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt, matplotlib.dates as mdates
    R = {m: dict(np.load(npz(m))) for m in MASKS if npz(m).exists()}
    obs = R["all"]["obs"]; lon, lat, t = R["all"]["lon"], R["all"]["lat"], R["all"]["time"]
    frac = uniform_filter(obs.astype(float), size=(7, 5, 5), mode="nearest")          # as merged.nc obs_frac_local: +-3 h, 5 x 5 px
    inner = np.zeros(obs.shape, bool); inner[TRIM_T:-TRIM_T, TRIM_X:-TRIM_X, TRIM_X:-TRIM_X] = True
    inner &= np.isfinite(R["all"]["u"])
    bins = [("0.0-0.2", 0.0, 0.2), ("0.2-0.4", 0.2, 0.4), ("0.4-0.6", 0.4, 0.6), ("0.6-0.8", 0.6, 0.8), ("0.8-0.9", 0.8, 0.9), ("0.9-1.0", 0.9, 1.01), ("all", 0.0, 1.01)]
    for m, r in R.items():
        print(f"{m:9s}: {int(r['n_stage2'])} stage-2 rows")
    ctl = xr.open_dataset(RUN / "merged.nc").u_m2.sel(time=t, lat=lat, lon=lon, method="nearest").values
    ok = inner & np.isfinite(ctl)
    print(f"check, 'all' vs the merged HPC control (u_m2, interior): corr {np.corrcoef(ctl[ok], R['all']['u_m2'][ok])[0, 1]:.3f}, rms {np.sqrt(np.mean(ctl[ok] ** 2)):.3f} vs {np.sqrt(np.mean(R['all']['u_m2'][ok] ** 2)):.3f}")
    for name, a, b in (("ring M2 (prior rms 0.117)", "u_m2", "v_m2"), ("short-scale d2 (prior rms 0.042)", "u_m2w", "v_m2w"), ("rotational (prior rms 0.087)", "u_eddy", "v_eddy")):
        print(f"\n{name}: rms per component (m/s) by local observed fraction")
        print("  obs frac |       n | " + " | ".join(f"{m:>8s}" for m in R))
        for lab, lo, hi in bins:
            s = inner & (frac >= lo) & (frac < hi)
            if s.sum() < 100:
                continue
            print(f"  {lab:8s} | {s.sum():7d} | " + " | ".join(f"{np.sqrt(np.nanmean(0.5 * (r[a][s] ** 2 + r[b][s] ** 2))):8.3f}" for r in R.values()))
    jr = int(np.argmin(np.abs(lat - ROWLAT)))
    vm = np.nanpercentile(np.abs(R["all"]["u_m2"][:, jr]), 98)
    fig, ax = plt.subplots(2, len(R) + 1, figsize=(4.2 * (len(R) + 1), 10), sharey=True, sharex=True, constrained_layout=True)
    for row, c in enumerate("uv"):
        im0 = ax[row, 0].pcolormesh(lon, t, frac[:, jr], cmap="Greys_r", vmin=0, vmax=1, shading="auto")
        ax[row, 0].set_title("local observed fraction (white = clear)", fontsize=10)
        for a, (m, r) in zip(ax[row, 1:], R.items()):
            im = a.pcolormesh(lon, t, r[f"{c}_m2"][:, jr], cmap="RdBu_r", vmin=-vm, vmax=vm, shading="auto")
            a.set_title(f"{c}_m2, rows: {m} ({int(r['n_stage2'])})", fontsize=10)
        fig.colorbar(im, ax=ax[row, 1:], shrink=0.8, label=f"{c} (m/s)")
        ax[row, 0].yaxis.set_major_locator(mdates.DayLocator()); ax[row, 0].yaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.colorbar(im0, ax=ax[:, 0], shrink=0.4, location="bottom", label="observed fraction")
    for a in ax[1]:
        a.set_xlabel("longitude (°E)")
    for a in ax.ravel():                                     # dotted lines: the scored interior in time
        for k in (TRIM_T, len(t) - TRIM_T):
            a.axhline(t[k], color="k", lw=0.7, ls=":")
    fig.suptitle(f"Cloud mask test: ring M2 block along {lat[jr]:.2f}°N (one block, {2 * HALF} h, {NNUM} neighbours)")
    p = OUT / f"exp50_{tag}_cloud_mask.png"
    fig.savefig(p, dpi=100); print("\nwrote", p)


if __name__ == "__main__":
    run(sys.argv[2]) if sys.argv[1] == "run" else report()

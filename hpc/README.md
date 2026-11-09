# HPC workflow (browseSST, space-time Vecchia)

One cube file -> (regional tide) -> shared hyperparameters -> independent tiles -> blend. Every script reads the same cube file lazily and `manifest.json`.

## The test file
`make_test_dataset.py` builds ONE file from the per-hour pysatview Himawari NetCDFs and a SUNTANS bathymetry (unstructured cell centres `xv, yv, dv`): box lon 121.6-124.6, lat -15.4 to -12.4 (centred on the BB23 moorings: `python hpc/make_test_dataset.py --lon 121.6 124.6 --lat -15.4 -12.4 ...`; the first test box was lon 121-124, lat -14 to -11, which missed five of the six moorings), 4 km
(stride 2 of the 2 km product), 670 hourly steps 2026-05-01 00:00 to 05-28 21:00 -> 75 x 75 x 670. Variables: `sst(time, lat, lon)` (K, NaN = not valid, quality >= 5), `depth(lat, lon)` (cell mean of the
bathymetry points; **NaN where shallower than 100 m or no bathymetry point** - the tide stage excludes those pixels), `depth_raw` (no shallow mask). Also writes a quick-look PNG. The local copy is
`/mnt/c/Users/00099894/Large_Data/Other/browseSST/test_bb23_may2026.nc` (4.3 MB): copy it to the HPC data directory set in `run_test.py` (`TEST_FILE`).

Coverage of the test month: good (0.8-0.99 per day) on days 0-14 (May 1-15) and 22-27 (May 23-28); a cloud gap on days 15-21 with about zero valid pixels on May 17-19. Weekly tile windows 3 (days 14-20)
is therefore mostly empty (the tile job skips a block with fewer than `--min-obs` observed pixels).

## Running the test on the HPC (browseGP style)
`run_test.py <stage> [pilot] [dry]` is the driver (like `browseGP/scripts/run_*.py`: subprocess, log file in `RUN/logs/`); `sbat_test_<stage>.sbatch` submit one stage; `submit_test.sh` chains them.
Paths (EDIT the constants at the top of `run_test.py`, or set `BSST_ROOT`, `BSST_TEST_FILE`, `BSST_RUN`): defaults assume the repository at `/group/oi019/wedge/pl/pckgs/browseSST`, the test file at
`/group/oi019/wedge/data/browseSST/test_bb23_may2026.nc`, outputs in `/group/oi019/wedge/data/browseSST/run_test_bb23`. The `drift` env needs `inversion_sst_gp` (`pip install -e browseSST`), gpvecchia, datide.

| stage | what | pilot |
|---|---|---|
| `manifest` | tile layout (`--core 168 28 28 --buffer 12 4 4`: 1 week x 1 deg, buffers 12 h / 4 px), `manifest_tiles.csv`, 4 sample tiles (best coverage) | - |
| `tide` | regional barotropic tide M2 S2 on the whole cube, `--min-depth 100`, 150 km splines (`--spacing`), day-by-day heating nuisance (`--nuisance-t-days 1`; K1 / O1 dropped: the BB23 moorings have < 0.05 m/s there) | time stride 6 |
| `hyper` | hyperparameters fitted on one sample tile per array task (stage 1 latent-T, then the velocity prior, stage-2 noise fitted by ML) | 3 + 3 iterations |
| `pool` | log-median of the per-tile fits -> `hyper.json` | - |
| `tiles` | one tile per array task | ONLY the first sample tile: use it to time a real tile and read its memory |
| `merge` | blend -> `merged.nc` | - |

1. `bash submit_test.sh pilot` (manifest, a pilot tide fit, one pilot tile); then `sacct -j <jobid> --format=JobID,Elapsed,MaxRSS,State` and size `--mem` / `--time` in `sbat_test_tiles.sbatch`
   (a 1 deg x 1 week tile with buffers is ~250k points; the maximin ordering is O(N^2), 474 s at 132k points in exp 25; expect an hour or more per tile).
2. `bash submit_test.sh` for the full chain (the tile array has one task per tile, 8 at a time).
`BSST_TIDE_SOURCE=cube` takes the tide from `u_tide` / `v_tide` variables in the test file (an external product such as FES2022 predicted on the grid) instead of the SST regression; `none` skips it.

## Scripts (all take `--help`)
`make_cube.py` (per-hour files -> cube), `make_test_dataset.py` (cube + bathymetry -> the single test file), `make_manifest.py`, `run_tide.py` (`--bathy FILE`, or the cube's own `depth`),
`fit_hyper.py`, `pool_hyper.py`, `run_tile.py` (`--tide FILE | --tide-from-cube`), `merge.py`. `hyper_real_start.json` is the start / prior centre of the real-data hyperparameter fits. `hyper_browsegp_start.json` is the same for the browseGP prior (below).

## The browseGP velocity prior, with its parameters fitted on the tiles
`--prior browsegp` (fit_hyper.py and run_tile.py) uses browseGP's in-situ velocity covariance as the (u, v) prior of the transport equation: rotational (`rot`), M2 divergent wave (`div`), fast red noise (`fast`),
optionally `m4` and `inertial` (`--blocks rot,div,fast` is the default: M4 and inertial dropped). Starting values = browseGP's posterior medians (`browsegp_vel.json`, from `results/mcmc_M4_v0.01`), S and noise from the exp-41 fit.
- `--free-vel rot,div,fast` (or `all`) FITS the (gamma, ell, tau) of those blocks on the sample tile together with S, the diurnal term and the noise: 9 velocity + 5 other parameters, finite-difference L-BFGS, so one
  iteration costs ~15 likelihood evaluations; gamma is the velocity amplitude (m/s), not the potential sigma, so the sigma-ell ridge is avoided. Log-normal prior of sd `--vel-prior-sd` (default 1 = a factor e) about
  the start values. Without `--free-vel` the velocity parameters stay at browseGP's values (only S, diurnal, noise fitted).
- `--fit-hours N` fits on the central N hours of the (buffered) tile: a likelihood evaluation is linear in the number of points (about 20 s for 36 x 28 x 28 points, 4 threads), so a 192 h tile x 25 iterations is
  many hours; 48-96 h is enough to see the parameters move. The fitted values go into `Hyper.stage2` (`hyper_<tile>.json`), `pool_hyper.py` takes their log-median, and `run_tile.py --prior browsegp --hyper hyper.json`
  reads them. `run_tile.py` needs the same `--prior` / `--blocks` as the fit.
- `run_test.py`: `BSST_PRIOR=browsegp` (+ `BSST_BLOCKS`, `BSST_FREE_VEL` default `rot,div,fast`, `BSST_FIT_HOURS`) switches the `hyper` and `tiles` stages; the hyper stage then starts from `hyper_browsegp_start.json`.
- Check `converged2` and the per-tile spread printed by `pool_hyper.py`: a parameter that sits on its bound or differs by a large factor between tiles is not identified by SST on that tile.

## Not in this chain (see notes/00_spacetime_vecchia_plan.md)
- The default covariance model is the placeholder Helmholtz-Matern prior; the browseGP model is the option above (`pipeline.BrowseGPPrior`; other models plug in through `pipeline.VelocityPrior`).
- Propagated (multiple-imputation) variance: the returned variances are the stage-2 block variances only.
- The SST-regression tide could not be validated on the real test month (notes: exp 34); the external-tide option exists for that reason.
- Cloud gaps: a guard drops stage-2 rows where the latent dT/dt is very uncertain (`tt_sd_factor`); the temporal seam between weekly windows was tested only on synthetic data with a 6 h buffer.

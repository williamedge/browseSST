"""
Experiment 29 (data): long synthetic record WITH a barotropic-like tide for the regional least-squares test (exp 30).

3 deg x 3 deg (84 x 84 px at 4 km), 60 days hourly, M2 / S2 / K1 / O1 with amplitude and phase varying smoothly with a synthetic depth field and a rough component on the shelf
(depth < ~200 m, about 22% of the area), plus the usual eddy field (Helmholtz), fronts + filament in the initial tracer, Matern S and the diurnal heating, sigma_T = 0.01 K.
Coarser simulation settings than the earlier experiments (dt_sim 1 h, 200 modes, padding 45 cells instead of the worst-case drift) so that a 60-day run takes about an hour.
The file is written OUTSIDE the repository tree (it is about 1 GB; the repository sits in OneDrive): ~/browsesst_data/synth_tide_84px_60d.nc, float32.
Run: python experiments/29_tide_generate.py [n_days]
"""
import pathlib, sys, time
import numpy as np

from inversion_sst_gp.spacetime import synthetic as S

OUT = pathlib.Path.home() / "browsesst_data"


def config(n_days=60):
    return S.SyntheticConfig(nx=84, ny=84, n_hours=24 * n_days, dt_sim=3600.0, n_modes=200, seed=11, pad_cells=45, tide=True, sigma_T=0.01,
                             front_amp=0.5, front_width=4000.0, filament_amp=0.3, filament_width=3500.0, products=False)


if __name__ == "__main__":
    n_days = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    cfg = config(n_days)
    OUT.mkdir(exist_ok=True)
    path = OUT / f"synth_tide_84px_{n_days}d.nc"
    t0 = time.time()
    ds = S.generate(cfg)
    print(f"generated {cfg.n_hours} h in {time.time() - t0:.0f} s", flush=True)
    enc = {v: {"dtype": "float32"} for v in ds.data_vars}
    ds.to_netcdf(path, encoding=enc)
    print("saved", path, f"{path.stat().st_size / 1e6:.0f} MB", flush=True)

"""
Experiment 10: generate and cache one large synthetic domain for tile-size experiments.

~3 deg x 3 deg (84 x 84 px @ 4 km = 336 km) x 7 days hourly, same prior family as experiments 04-07. The tracer is
integrated once on a padded grid (slow, ~tens of minutes, single thread); tiles of any size/duration can then be cut
from the cached file with `inference.make_observations(..., ) ` on a sliced dataset.
Output: data/synth_84px_168h.nc (ignored by git: *.nc). Run: python experiments/10_make_big_domain.py [nx] [n_hours]
"""
import pathlib, sys, time, warnings
from inversion_sst_gp.spacetime import synthetic as S

warnings.filterwarnings("ignore")


def main(nx=84, n_hours=168, n_modes=400):
    cfg = S.SyntheticConfig(nx=nx, ny=nx, n_hours=n_hours, n_modes=n_modes, seed=3)
    t0 = time.time()
    ds = S.generate(cfg)
    out = pathlib.Path(__file__).resolve().parents[1] / "data" / f"synth_{nx}px_{n_hours}h.nc"
    out.parent.mkdir(exist_ok=True)
    ds.attrs.update(seed=cfg.seed, n_modes=n_modes, true_params_note="see SyntheticConfig defaults")
    ds.to_netcdf(out)
    print(f"generated in {time.time() - t0:.0f}s, pad={ds.attrs['pad_cells']}, vmax={ds.attrs['vmax']:.2f} -> {out}")
    print("equation residual:", S.equation_residual(ds))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(nx=int(a[0]) if a else 84, n_hours=int(a[1]) if len(a) > 1 else 168)

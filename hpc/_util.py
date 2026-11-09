"""Small helpers shared by the HPC drivers (logging with timestamps, thread settings)."""
import os
import sys
import time

_T0 = time.time()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - _T0:6.0f}s] {msg}", flush=True)


def set_threads():
    """Respect SLURM's allocation for numba / BLAS before they are imported by the heavy modules."""
    n = os.environ.get("SLURM_CPUS_PER_TASK")
    if n:
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
            os.environ.setdefault(k, n)


def tile_id_from_args(args, ids):
    """Tile id from --tile, or from --array-index (position in `ids`, or in the comma list given by --tiles)."""
    if getattr(args, "tile", None):
        return args.tile
    k = args.array_index if args.array_index is not None else os.environ.get("SLURM_ARRAY_TASK_ID")
    if k is None:
        sys.exit("give --tile ID or --array-index K (or run as a SLURM array)")
    if getattr(args, "tiles", None):
        ids = args.tiles.split(",")
    return ids[int(k)]


def add_prior_args(ap):
    """The stage-2 velocity prior options shared by fit_hyper.py and run_tile.py."""
    ap.add_argument("--prior", default="helmholtz", choices=("helmholtz", "browsegp"), help="stage-2 velocity covariance: the Matern Helmholtz default, or browseGP's in-situ model")
    ap.add_argument("--vel-json", help="browsegp: browseGP velocity parameters (mcmc_results.json or a plain {name: value} json); default hpc/browsegp_vel.json")
    ap.add_argument("--blocks", default="rot,div,fast", help="browsegp: velocity blocks that are on (rot, div = M2 wave, m4, fast, inertial)")
    ap.add_argument("--free-vel", default="", help="browsegp, fit_hyper only: blocks whose (gamma, ell, tau) are FITTED on the sample tile ('all' = every block that is on); default none = fixed at browseGP's values")
    ap.add_argument("--vel-prior-sd", type=float, default=1.0, help="browsegp: sd (log units) of the log-normal prior on the fitted velocity parameters, centred on the start values")
    ap.add_argument("--lat", type=float, default=-14.0, help="browsegp: latitude for the inertial frequency (only if the inertial block is on)")


def make_prior(a):
    """The `VelocityPrior` for the options of `add_prior_args` (None = the library default)."""
    if a.prior == "helmholtz":
        return None
    import pathlib
    from inversion_sst_gp.spacetime import pipeline as P
    path = a.vel_json or str(pathlib.Path(__file__).resolve().parent / "browsegp_vel.json")
    blocks = tuple(b for b in a.blocks.split(",") if b)
    free = getattr(a, "free_vel", "")
    free_vel = True if free == "all" else tuple(b for b in free.split(",") if b)
    return P.BrowseGPPrior.from_results(path, blocks=blocks, lat=a.lat, free_vel=free_vel, prior_sd_log={n: a.vel_prior_sd for names in P.BrowseGPPrior.VEL_NAMES.values() for n in names})

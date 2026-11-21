"""Small helpers shared by the HPC drivers (logging with timestamps, thread settings)."""
import os
import sys
import time

_T0 = time.time()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - _T0:6.0f}s] {msg}", flush=True)


def set_threads():
    """Respect SLURM's allocation for numba / BLAS before they are imported by the heavy modules, and give each SLURM job its OWN numba cache directory (node-local temp dir, removed at exit): the jitted
    functions are `cache=True`, and concurrent array tasks sharing one cache index corrupted it (KeyError / 'underlying object has vanished' in numba.core.caching; 24 of 27 tiles of the first full run).
    An NUMBA_CACHE_DIR set by the caller is kept."""
    job = os.environ.get("SLURM_JOB_ID")
    if job and "NUMBA_CACHE_DIR" not in os.environ:
        import atexit
        import shutil
        import tempfile
        d = os.path.join(tempfile.gettempdir(), f"numba_cache_{job}_{os.environ.get('SLURM_ARRAY_TASK_ID', '0')}_{os.getpid()}")
        os.makedirs(d, exist_ok=True)
        os.environ["NUMBA_CACHE_DIR"] = d
        atexit.register(shutil.rmtree, d, True)
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
    ap.add_argument("--vel-json", help="browsegp: browseGP velocity parameters (results.json / mcmc_results.json or a plain {name: value} json); default hpc/browsegp_vel_v008.json (v0.08)")
    ap.add_argument("--blocks", default="rot,div,d2", help="browsegp: velocity blocks that are on: rot (= the 'eddy'), div (M2 wave), d2 (short-scale / mode-2 M2 block), m4 (instead of d2), fast, inertial")
    ap.add_argument("--div-kernel", default="ring", choices=("ring", "matern"), help="browsegp: spatial kernel of the M2 wave block: browseGP's ring (Matern x J0(k1 r), parameter c1) or the plain Matern wave")
    ap.add_argument("--free-vel", default="", help="browsegp, fit_hyper only: blocks whose (gamma, ell, tau) [+ c1 for a ring] are FITTED on the sample tile ('all' = every block that is on); default none = fixed at browseGP's values")
    ap.add_argument("--fix-vel", default="ell_div,tau_div", help="browsegp, fit_hyper only: parameter names kept fixed even in a free block (default: the M2 wave's length and time scale, which are much larger than a tile and not identifiable on it)")
    ap.add_argument("--vel-prior-sd", type=float, default=1.0, help="browsegp: sd (log units) of the log-normal prior on the fitted velocity parameters, centred on the start values")
    ap.add_argument("--lat", type=float, default=-14.0, help="browsegp: latitude for the Coriolis parameter (ring wavenumber, inertial block)")
    ap.add_argument("--no-block-outputs", action="store_true", help="browsegp, run_tile only: do not write the block-by-block fields (u_eddy = rotational only, u_m2, u_m2w, ...); u_eddy is then the total")


def make_prior(a):
    """The `VelocityPrior` for the options of `add_prior_args` (None = the library default)."""
    if a.prior == "helmholtz":
        return None
    import pathlib
    from inversion_sst_gp.spacetime import pipeline as P
    path = a.vel_json or str(pathlib.Path(__file__).resolve().parent / "browsegp_vel_v008.json")
    blocks = tuple(b for b in a.blocks.split(",") if b)
    free = getattr(a, "free_vel", "")
    free_vel = True if free == "all" else tuple(b for b in free.split(",") if b)
    fixed = tuple(n for n in getattr(a, "fix_vel", "").split(",") if n)
    sd = {n: a.vel_prior_sd for names in P.BrowseGPPrior.VEL_NAMES.values() for n in names}
    sd["c1"] = a.vel_prior_sd
    return P.BrowseGPPrior.from_results(path, blocks=blocks, lat=a.lat, free_vel=free_vel, prior_sd_log=sd, div_kernel=a.div_kernel, fixed=fixed)

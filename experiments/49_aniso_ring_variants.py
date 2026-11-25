"""
Experiment 49: stage-2 marginal log-likelihood with an ANISOTROPIC ring for the M2 divergent block (browseGP's anisotropic ring), same 72 h x 36 x 36 blocks as exp 44, every other parameter held at the control run's values
(stage 1, S, diurnal, noise from `run_v008_control/hyper.json`; velocity values from the JSON of each variant, NOT from the hyper file).
Variants: control (isotropic ring, ell 135 km); user's L_x = 60 km / L_y = 180 km (crests along y, wave travelling east-west); the swapped orientation (L_x = 180, L_y = 60); browseGP's own fitted anisotropic result
(results/dense_simple_aniso_d2m_v0.08: ell 90 km, s 1.19, theta 0.43 rad, its own c1 and amplitudes); and the isotropic ring with ell 60 km / 180 km for reference.
Run: python experiments/49_aniso_ring_variants.py [YYYY-MM-DDTHH ...]
"""
import gc, json, os, pathlib, sys, time, warnings
import numpy as np, xarray as xr
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "pkg"))
from inversion_sst_gp.spacetime import pipeline as P, regional as RG, tiling as TL
BASE = pathlib.Path("/mnt/c/Users/00099894/Large_Data/Other/browseSST"); RUN = BASE / "run_v008_control"
BG = pathlib.Path("/mnt/c/Users/00099894/OneDrive - The University of Western Australia/pl/wedge/browseGP/results/dense_simple_aniso_d2m_v0.08/results.json")
HALF = int(os.environ.get('BSST_HALF', 36)); NNUM = int(os.environ.get('BSST_NNUM', 100))          # half-window (h) and Vecchia neighbours: lower both on a small machine (the 72 h / 100 default peaks at ~3 GB)
whens = sys.argv[1:] or ["2023-05-05T12", "2023-05-16T12"]
V0 = json.loads((ROOT / "hpc" / "browsegp_vel_v008.json").read_text()); V0 = {k: v for k, v in V0.items() if not k.startswith("_")}
def var(**kw): return {**V0, **kw}
PI2 = np.pi / 2
VARIANTS = [("isotropic ring ell 135 km (control)", var(), "ring"),
            ("aniso L_x 60 / L_y 180 (crests along y)", var(ell_div=60e3, aniso_div=3.0, theta_div=PI2), "ring_aniso"),
            ("aniso swapped L_x 180 / L_y 60 (crests along x)", var(ell_div=60e3, aniso_div=3.0, theta_div=0.0), "ring_aniso"),
            ("browseGP fitted aniso (90 km, s 1.19, 24.6 deg)", json.loads(BG.read_text())["params"], "ring_aniso"),
            ("isotropic ring ell 60 km", var(ell_div=60e3), "ring"),
            ("isotropic ring ell 180 km", var(ell_div=180e3), "ring")]
cube = xr.open_dataset(BASE / "test_bb23_2023.nc"); hyper = P.Hyper.load(RUN / "hyper.json"); tide = RG.TideModel.load(RUN / "tide" / "tide_model.pkl")
KEEP = ("sigma_S", "l_S", "tau0_S", "lr:diurnal", "noise")
h2 = {k: hyper.stage2[k] for k in KEEP}
for w in whens:
    kt = int(np.argmin(np.abs(cube.time.values - np.datetime64(w)))); k0, k1 = kt - HALF, kt + HALF
    spec = TL.TileSpec("map", k0, k1, 24, 60, 24, 60, k0, k1, 24, 60, 24, 60); td = TL.extract_tile(cube, spec, min_depth=250.0)
    cfg0 = P.PipelineConfig(nnum=NNUM, nnum_pred=2 * NNUM, prior=P.BrowseGPPrior(V0, blocks=("rot", "div", "d2"), div_kernel="ring"), depth_mask=True, min_cover=0.0)
    xo, yo, to, To = td.observed(); gp1 = P.build_stage1(xo, yo, to, To, hyper.stage1, cfg0); Tm, Ttm, vTt = P.stage1_products(td, gp1, cfg0); del gp1; gc.collect()          # (memory: the machine has 7.5 GB)
    ub, vb, sup = P._tide_for_tile(td, tide, cfg0); idx, Tx, Ty, z = P.stage2_rows(td, Tm, Ttm, vTt, ub, vb, cfg0)
    print(f"\n== window centred {w}: {idx.size} stage-2 rows", flush=True)
    base = None
    for name, vel, dk in VARIANTS:
        t0 = time.time()
        pr = P.BrowseGPPrior(vel, blocks=("rot", "div", "d2"), div_kernel=dk)
        cfg = P.PipelineConfig(nnum=NNUM, nnum_pred=2 * NNUM, prior=pr, depth_mask=True, min_cover=0.0)
        gp2 = P.build_stage2(td, idx, Tx, Ty, z, h2, cfg)
        ll = gp2.loglik(); base = ll if base is None else base; del gp2; gc.collect()
        print(f"  {name:50s} loglik {ll:12.1f}   vs control {ll - base:+9.1f}   ({time.time() - t0:.0f}s)", flush=True)

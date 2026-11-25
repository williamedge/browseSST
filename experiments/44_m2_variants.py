"""
Experiment 44: does the M2 block help, and does the ring help?  Stage-2 marginal log-likelihood on one 72 h x 36 x 36 block (the lon / lat box of tile t06_01_01) for several velocity priors, ALL other parameters held at the
control run's pooled values (stage 1, S, diurnal, noise, velocity parameters). Blocks: rot = rotational ('eddy'), div = M2 divergent wave (ring or plain Matern), d2 = short-scale M2 block.
Run: python experiments/44_m2_variants.py [YYYY-MM-DDTHH ...]   (default two windows: 2023-05-05T12 spring-tide, 2023-05-16T12)
"""
import pathlib, sys, time, warnings
import numpy as np, xarray as xr
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "pkg"))
from inversion_sst_gp.spacetime import pipeline as P, regional as RG, tiling as TL
BASE = pathlib.Path("/mnt/c/Users/00099894/Large_Data/Other/browseSST"); RUN = BASE / "run_v008_control"
whens = sys.argv[1:] or ["2023-05-05T12", "2023-05-16T12"]
cube = xr.open_dataset(BASE / "test_bb23_2023.nc")
hyper = P.Hyper.load(RUN / "hyper.json"); tide = RG.TideModel.load(RUN / "tide" / "tide_model.pkl")
VARIANTS = [("rot + ring M2 + d2 (control)", ("rot", "div", "d2"), "ring"), ("rot + plain M2 + d2", ("rot", "div", "d2"), "matern"), ("rot + ring M2", ("rot", "div"), "ring"),
            ("rot + plain M2", ("rot", "div"), "matern"), ("rot + d2", ("rot", "d2"), "ring"), ("rot only", ("rot",), "ring")]
for w in whens:
    kt = int(np.argmin(np.abs(cube.time.values - np.datetime64(w)))); k0, k1 = kt - 36, kt + 36
    spec = TL.TileSpec("map", k0, k1, 24, 60, 24, 60, k0, k1, 24, 60, 24, 60)
    td = TL.extract_tile(cube, spec, min_depth=250.0)
    cfg0 = P.PipelineConfig(prior=P.BrowseGPPrior.from_results(ROOT / "hpc" / "browsegp_vel_v008.json", blocks=("rot", "div", "d2"), div_kernel="ring"), depth_mask=True, min_cover=0.0)
    xo, yo, to, To = td.observed()
    gp1 = P.build_stage1(xo, yo, to, To, hyper.stage1, cfg0); Tm, Ttm, vTt = P.stage1_products(td, gp1, cfg0)
    ub, vb, sup = P._tide_for_tile(td, tide, cfg0); idx, Tx, Ty, z = P.stage2_rows(td, Tm, Ttm, vTt, ub, vb, cfg0)
    print(f"\n== window centred {w}: {idx.size} stage-2 rows", flush=True)
    base = None
    for name, blocks, dk in VARIANTS:
        t0 = time.time()
        pr = P.BrowseGPPrior.from_results(ROOT / "hpc" / "browsegp_vel_v008.json", blocks=blocks, div_kernel=dk)
        cfg = P.PipelineConfig(prior=pr, depth_mask=True, min_cover=0.0)
        h2 = dict(hyper.stage2)
        gp2 = P.build_stage2(td, idx, Tx, Ty, z, h2, cfg)
        ll = gp2.loglik()
        base = ll if base is None else base
        print(f"  {name:32s} loglik {ll:12.1f}   vs control {ll - base:+9.1f}   ({time.time() - t0:.0f}s)", flush=True)

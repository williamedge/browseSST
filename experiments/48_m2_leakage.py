"""
Experiment 48: does the rotational ('eddy') block leak into / change the M2 blocks?  72 h x 36 x 36 block (lon / lat box of tile t06_01_01), control parameters.  Compare the posterior-mean M2 part (div + d2) predicted
(a) from the full prior (rot + div + d2, block-attributed) and (b) from a prior that has ONLY the two M2 blocks (div + d2), at three hours around the window centre.
Run: python experiments/48_m2_leakage.py [YYYY-MM-DDTHH ...]
"""
import pathlib, sys, time, warnings
import numpy as np, xarray as xr
warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "pkg"))
from inversion_sst_gp.spacetime import pipeline as P, regional as RG, tiling as TL
BASE = pathlib.Path("/mnt/c/Users/00099894/Large_Data/Other/browseSST"); RUN = BASE / "run_v008_control"
whens = sys.argv[1:] or ["2023-05-05T12", "2023-05-16T12"]
cube = xr.open_dataset(BASE / "test_bb23_2023.nc"); hyper = P.Hyper.load(RUN / "hyper.json"); tide = RG.TideModel.load(RUN / "tide" / "tide_model.pkl")
V = ROOT / "hpc" / "browsegp_vel_v008.json"
for w in whens:
    kt = int(np.argmin(np.abs(cube.time.values - np.datetime64(w)))); k0, k1 = kt - 36, kt + 36
    spec = TL.TileSpec("map", k0, k1, 24, 60, 24, 60, k0, k1, 24, 60, 24, 60); td = TL.extract_tile(cube, spec, min_depth=250.0)
    full = P.BrowseGPPrior.from_results(V, blocks=("rot", "div", "d2"), div_kernel="ring"); red = P.BrowseGPPrior.from_results(V, blocks=("div", "d2"), div_kernel="ring")
    cfg = P.PipelineConfig(prior=full, depth_mask=True, min_cover=0.0); cfgr = P.PipelineConfig(prior=red, depth_mask=True, min_cover=0.0)
    xo, yo, to, To = td.observed(); gp1 = P.build_stage1(xo, yo, to, To, hyper.stage1, cfg); Tm, Ttm, vTt = P.stage1_products(td, gp1, cfg)
    ub, vb, sup = P._tide_for_tile(td, tide, cfg); idx, Tx, Ty, z = P.stage2_rows(td, Tm, Ttm, vTt, ub, vb, cfg); h2 = dict(hyper.stage2)
    gpf = P.build_stage2(td, idx, Tx, Ty, z, h2, cfg); gpr = P.build_stage2(td, idx, Tx, Ty, z, h2, cfgr)
    nt, ny, nx = td.shape
    print(f"\n== window {w}: {idx.size} rows", flush=True)
    for dk in (-6, 0, 6):
        sel = np.zeros(td.x.size, bool); sel.reshape(td.shape)[36 + dk] = True; q = sel & P.deep_points(td, cfg)
        qx, qy, qt = td.x[q], td.y[q], td.t[q]; nn = gpf.pred_neighbours(qx, qy, qt, cfg.nnum_pred)
        gpf.update(cov_params=full.cov_params_blocks(h2, ("div", "d2"))); gpr.update(cov_params=red.cov_params(h2))
        a = {c: gpf.predict(c, qx, qy, qt, nn_array=nn)[0] for c in "uv"}; b = {c: gpr.predict(c, qx, qy, qt, nn_array=nn)[0] for c in "uv"}
        gpf.update(cov_params=full.cov_params_blocks(h2, ("rot",))); r = {c: gpf.predict(c, qx, qy, qt, nn_array=nn)[0] for c in "uv"}
        ca = np.concatenate([a["u"], a["v"]]); cb = np.concatenate([b["u"], b["v"]]); cr = np.concatenate([r["u"], r["v"]])
        print(f"  hour {dk:+d}: M2 part (full prior) rms {np.sqrt(np.mean(ca**2)):.3f}, M2-only prior rms {np.sqrt(np.mean(cb**2)):.3f}, corr {np.corrcoef(ca, cb)[0,1]:.3f}, rms diff {np.sqrt(np.mean((ca-cb)**2)):.3f} | rot block rms {np.sqrt(np.mean(cr**2)):.3f}, corr(rot, M2 diff) {np.corrcoef(cr, ca-cb)[0,1]:.2f}", flush=True)

"""
Experiment 40: browseGP's fitted velocity kernel as the stage-2 prior, scored against the BB23 in-situ velocities (2023 Himawari cube).

Tile: 28 x 28 px (1 deg) x 48 h of the 2023 cube `test_bb23_2023.nc` centred on the BB23 moorings (rows 18-46, cols 24-52: lon 122.57-123.65, lat -14.67 to -13.55); two windows, A = Apr 24 00:00 -
Apr 26 00:00 and B = May 4 - May 6 (SST coverage 0.95-0.97; drifters, moorings and ship ADCP inside). Stage 1 (latent T GP) uses the shared stage-1 hyperparameters of `hpc/hyper_real_start.json`;
the barotropic tide is the regional M2 / S2 fit of exp 38 (full 77-day record, 150 km splines, day-by-day nuisance) and is subtracted from dT/dt and added back to the velocity.
Stage-2 priors:
  browsegp  : `pipeline.BrowseGPPrior`: the velocity covariance of browseGP (rotational + M2 wave + M4 wave + fast + inertial) with the values of `mcmc_M4_v0.01` (posterior medians) FIXED; the source
              term S (Matern amplitude / length / time, day-by-day diurnal scale) and the equation noise are fitted by the Vecchia marginal likelihood
  helmholtz : our placeholder Helmholtz-Matern prior with ALL parameters free (the exp 37 'daily' structured-S model), started at the exp 37 values
Scoring: the posterior-mean EDDY velocity (u, v) is predicted at the position and time of every in-situ observation inside the tile (3 px / 3 h margins) and added to the tide; compared with the observed velocity.
Platforms: drifters, mooring ADCP (hourly means per site) and ship ADCP, from the `_long` file (as browseGP loads it) and from the `_corrected` file (Ekman-balanced currents removed, drifter Stokes subtracted).
Metrics per platform and component: rms of the observed velocity, rms of obs - tide, rmse of tide + eddy, skill = 1 - MSE(tide + eddy) / MSE(tide) (positive = the SST eddy velocity helps), and the
correlation of (obs - tide) with the predicted eddy velocity. Note the in-situ velocities contain the baroclinic tide, inertial motion and (moorings) depth-dependent flow that SST cannot see.
Run: python experiments/40_browsegp_prior_2023.py products <A|B> | fit <A|B> <browsegp|helmholtz> [maxiter] | score <A|B> <variant> ...
"""
import json, pathlib, sys, time, warnings
import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = pathlib.Path.home() / "browsesst_data" / "real_test"
from inversion_sst_gp.spacetime import pipeline as P, tiling as TL
from inversion_sst_gp.spacetime.fit import fit
from inversion_sst_gp.spacetime.regional import TideModel

CUBE = "/mnt/c/Users/00099894/Large_Data/Other/browseSST/test_bb23_2023.nc"
TIDE = DATA / "exp38bb23y_full_ms_daily150_tide.pkl"
BG = pathlib.Path("/mnt/c/Users/00099894/OneDrive - The University of Western Australia/pl/wedge/browseGP/results/mcmc_M4_v0.01/mcmc_results.json")
OBS = {"long": "/mnt/c/Users/00099894/Large_Data/Papers/drifterOI_paper2/observations/BB23_Obs_vector_150m_long.nc",
       "corrected": "/mnt/c/Users/00099894/Large_Data/Papers/drifterOI_paper2/observations/BB23_Obs_vector_150m_corrected.nc"}
WINDOWS = {"A": 23 * 24, "B": 33 * 24}                 # first hour (from 2023-04-01 00:00) of the 48 h window
NT, J0, I0, N = 48, 18, 24, 28
EXP37 = json.loads((DATA / "exp37_daily.json").read_text())["fitted"] if (DATA / "exp37_daily.json").exists() else {}


def tile(cube, win):
    h0 = WINDOWS[win] if win in WINDOWS else int(win[1:]) * 24            # 'W<d>': the 48 h window starting d days after 2023-04-01 00:00
    return TL.extract_tile(cube, TL.TileSpec(f"bb23{win}", h0, h0 + NT, J0, J0 + N, I0, I0 + N, h0, h0 + NT, J0, J0 + N, I0, I0 + N))


def products(win):
    cube = xr.open_dataset(CUBE)
    td = tile(cube, win)
    h1 = P.Hyper.load(ROOT / "hpc" / "hyper_real_start.json").stage1
    cfg = P.PipelineConfig()
    xo, yo, to, To = td.observed()
    t0 = time.time()
    gp1 = P.build_stage1(xo, yo, to, To, h1, cfg)
    Tm, Ttm, vTt = P.stage1_products(td, gp1, cfg)
    np.savez(DATA / f"exp40_{win}_products.npz", Tm=Tm, Ttm=Ttm, vTt=vTt)
    print(f"[{win}] {td.n_obs} observed of {td.x.size} grid points; stage-1 products {time.time() - t0:.0f}s saved", flush=True)


def prior_and_start(variant):
    if variant == "browsegp":
        s = dict(P.DEFAULT_STAGE2, sigma_S=EXP37.get("sigma_S", 17.0), l_S=EXP37.get("l_S", 15e3), tau0_S=EXP37.get("tau0_S", 3600.0), noise=4.0, **{"lr:diurnal": EXP37.get("lr:diurnal", 90.0)})
        return P.BrowseGPPrior.from_results(BG, lat=-14.1), s
    s = dict(P.DEFAULT_STAGE2)
    s.update({k: EXP37[k] for k in ("gamma", "f_delta", "l_psi", "l_phi", "tau0_psi", "tau0_phi", "sigma_S", "l_S", "tau0_S", "lr:diurnal") if k in EXP37}); s["noise"] = 4.0
    return P.StructuredSPrior(daily=True), s


def insitu(path, td):
    """In-situ observations inside the tile (3 px / 3 h margins) as a DataFrame: lon, lat, time, u, v, platform (moorings: hourly means per site)."""
    ds = xr.load_dataset(path)
    df = pd.DataFrame({"time": ds["time"].values, "lon": ds["longitude"].values, "lat": ds["latitude"].values, "u": ds["east_vel"].values, "v": ds["north_vel"].values, "platform": ds["dtype"].values.astype(str)})
    df = df.drop_duplicates(subset=["time", "lon", "lat", "u", "v"])
    t0, t1 = td.time[3], td.time[-4]
    df = df[(df.time >= t0) & (df.time <= t1) & (df.lon >= td.lon[3]) & (df.lon <= td.lon[-4]) & (df.lat >= td.lat[3]) & (df.lat <= td.lat[-4])]
    m = df.platform == "mADCP"
    mo = df[m].copy()
    mo["site"] = mo.groupby([mo.lon.round(5), mo.lat.round(5)]).ngroup()
    mo["hour"] = mo.time.dt.floor("1h")
    mo = mo.groupby(["site", "hour"]).agg(time=("hour", "first"), lon=("lon", "mean"), lat=("lat", "mean"), u=("u", "mean"), v=("v", "mean"), platform=("platform", "first")).reset_index(drop=True)
    return pd.concat([df[~m], mo], ignore_index=True).dropna(subset=["u", "v"])


def run_fit(win, variant, maxiter=20, fixed_from=None):
    """`fixed_from` (window name): do not optimise, use the parameters fitted in that window (predict-only runs on further windows)."""
    cube = xr.open_dataset(CUBE)
    td = tile(cube, win)
    z_ = np.load(DATA / f"exp40_{win}_products.npz")
    tide = TideModel.load(TIDE)
    ub, vb, sup = tide.tile_velocity(td, mask="supported")
    base = P.PipelineConfig()
    idx, Tx, Ty, z = P.stage2_rows(td, z_["Tm"], z_["Ttm"], z_["vTt"], ub, vb, base)
    prior, h2 = prior_and_start(variant)
    cfg = P.PipelineConfig(prior=prior)
    gp2 = P.build_stage2(td, idx, Tx, Ty, z, h2, cfg)
    free = list(prior.free)
    bounds = dict(prior.bounds); bounds.setdefault("noise", (1.0, 1e4))
    print(f"[{win}/{variant}] {idx.size} stage-2 rows of {td.x.size}, tide supported on {sup.mean():.2f} of the grid, free {free}", flush=True)
    ll0 = gp2.loglik(); t0 = time.time()
    if fixed_from:
        fitted = dict(json.loads((DATA / f"exp40_{fixed_from}_{variant}.json").read_text())["fitted"])
        res = type("R", (), dict(n_eval=0, success=True))()
    else:
        res = fit(gp2, h2, free, bounds=bounds, prior_sd_log={n: 1.5 for n in free}, maxiter=maxiter, verbose=False, slots=getattr(prior, "slots", None))
        fitted = dict(h2); fitted.update({k: float(res.params[k]) for k in free})
    gp2.update(cov_params=prior.cov_params(fitted), noise=fitted["noise"], lowrank_scales={"diurnal": fitted["lr:diurnal"]})
    ll = gp2.loglik()
    print(f"[{win}/{variant}] fit {res.n_eval} evals {time.time() - t0:.0f}s success={res.success}; loglik start {ll0:.1f} -> {ll:.1f}", flush=True)
    print("  fitted: " + ", ".join(f"{k} {fitted[k]:.3g}" for k in free), flush=True)
    out = dict(win=win, variant=variant, loglik=float(ll), loglik_start=float(ll0), n_eval=int(res.n_eval), success=bool(res.success), fitted={k: float(fitted[k]) for k in fitted}, n_rows=int(idx.size))
    for lab, path in OBS.items():
        o = insitu(path, td)
        xk, yk = td.frame.forward(o.lon.values, o.lat.values)
        tk = (o.time.values - td.time[0]).astype("timedelta64[s]").astype(np.float64)
        nn = gp2.pred_neighbours(xk, yk, tk, 200)
        ue, vue = gp2.predict("u", xk, yk, tk, nn_array=nn)
        ve, vve = gp2.predict("v", xk, yk, tk, nn_array=nn)
        utd, vtd = tide.velocity(o.lon.values, o.lat.values, o.time.values, mask="supported")
        utd, vtd = np.nan_to_num(utd), np.nan_to_num(vtd)
        np.savez(DATA / f"exp40_{win}_{variant}_{lab}.npz", platform=np.array(list(o.platform), dtype="U8"), lon=o.lon.values, lat=o.lat.values, time=o.time.values.astype("datetime64[s]").astype(np.int64), u=o.u.values, v=o.v.values,
                 ue=ue, ve=ve, vue=vue, vve=vve, ut=utd, vt=vtd)
        print(f"  [{lab}] predicted at {len(o)} in-situ points ({dict(o.platform.value_counts())})", flush=True)
    (DATA / f"exp40_{win}_{variant}.json").write_text(json.dumps(out, indent=1))


def metrics(d, sel):
    rows = {}
    for c, (obs, ed, tid) in {"u": (d["u"], d["ue"], d["ut"]), "v": (d["v"], d["ve"], d["vt"])}.items():
        o, e, t = obs[sel], ed[sel], tid[sel]
        mt, mtot = np.mean((o - t) ** 2), np.mean((o - t - e) ** 2)
        rows[c] = dict(n=int(sel.sum()), rms_obs=float(np.sqrt(np.mean(o ** 2))), rms_obs_minus_tide=float(np.sqrt(mt)), rms_eddy=float(np.sqrt(np.mean(e ** 2))), rmse=float(np.sqrt(mtot)),
                       skill=float(1 - mtot / mt), corr=float(np.corrcoef(o - t, e)[0, 1]) if sel.sum() > 3 else float("nan"))
    return rows


def score(win, variants):
    for lab in OBS:
        print(f"\n== window {win}, in-situ file '{lab}'")
        print(f"{'variant':<11}{'platform':<9}{'comp':>5}{'n':>5}{'rms obs':>9}{'obs-tide':>9}{'rms eddy':>9}{'rmse':>8}{'skill':>7}{'corr':>7}")
        for v in variants:
            d = np.load(DATA / f"exp40_{win}_{v}_{lab}.npz", allow_pickle=True)
            d = {k: (np.asarray(d[k]).astype(str) if k == "platform" else d[k]) for k in d.files}
            for plat in ("drifter", "mADCP", "sADCP", "all"):
                sel = np.ones(d["u"].size, bool) if plat == "all" else d["platform"] == plat
                if sel.sum() < 5:
                    continue
                for c, r in metrics(d, sel).items():
                    print(f"{v:<11}{plat:<9}{c:>5}{r['n']:>5}{r['rms_obs']:>9.3f}{r['rms_obs_minus_tide']:>9.3f}{r['rms_eddy']:>9.3f}{r['rmse']:>8.3f}{r['skill']:>7.2f}{r['corr']:>7.2f}")
    print("\nfits:")
    for v in variants:
        j = json.loads((DATA / f"exp40_{win}_{v}.json").read_text())
        print(f"  {v}: loglik {j['loglik']:.1f} (start {j['loglik_start']:.1f}), {j['n_eval']} evals, success={j['success']}; " + ", ".join(f"{k} {x:.3g}" for k, x in j["fitted"].items() if k in ("gamma", "f_delta", "l_psi", "l_phi", "tau0_psi", "tau0_phi", "sigma_S", "l_S", "tau0_S", "lr:diurnal", "noise")))


def _load(win, v, lab):
    d = np.load(DATA / f"exp40_{win}_{v}_{lab}.npz", allow_pickle=True)
    d = {k: (np.asarray(d[k]).astype(str) if k == "platform" else d[k]) for k in d.files}
    d["win"] = np.full(d["u"].size, win)
    return d


def pooled(wins, variants):
    """Scores pooled over windows: hourly-level (as `score`) and the WINDOW-MEAN comparison at the moorings (mean of obs - tide and of the predicted eddy over each 48 h window and site, which removes the
    baroclinic tide / inertial motion that SST cannot see and that dominates the hourly scores)."""
    for lab in OBS:
        print(f"\n== windows {','.join(wins)}, in-situ file '{lab}'")
        print(f"{'variant':<11}{'platform':<9}{'comp':>5}{'n':>5}{'rms obs':>9}{'obs-tide':>9}{'rms eddy':>9}{'rmse':>8}{'skill':>7}{'corr':>7}")
        for v in variants:
            parts = [_load(w, v, lab) for w in wins if (DATA / f"exp40_{w}_{v}_{lab}.npz").exists()]
            d = {k: np.concatenate([p_[k] for p_ in parts]) for k in parts[0]}
            for plat in ("drifter", "mADCP", "sADCP", "all"):
                sel = np.ones(d["u"].size, bool) if plat == "all" else d["platform"] == plat
                if sel.sum() < 5:
                    continue
                for c, r in metrics(d, sel).items():
                    print(f"{v:<11}{plat:<9}{c:>5}{r['n']:>5}{r['rms_obs']:>9.3f}{r['rms_obs_minus_tide']:>9.3f}{r['rms_eddy']:>9.3f}{r['rmse']:>8.3f}{r['skill']:>7.2f}{r['corr']:>7.2f}")
            # window means at the moorings
            m = d["platform"] == "mADCP"
            key = np.array([f"{w}_{lo:.4f}_{la:.4f}" for w, lo, la in zip(d["win"][m], d["lon"][m], d["lat"][m])])
            uk = np.unique(key)
            for c, (o, e, t) in {"u": (d["u"], d["ue"], d["ut"]), "v": (d["v"], d["ve"], d["vt"])}.items():
                om = np.array([np.mean((o - t)[m][key == k]) for k in uk]); em = np.array([np.mean(e[m][key == k]) for k in uk])
                sk = 1 - np.mean((om - em) ** 2) / np.mean(om ** 2)
                print(f"{v:<11}{'moor 48h':<9}{c:>5}{len(uk):>5}{'':>9}{np.sqrt(np.mean(om ** 2)):>9.3f}{np.sqrt(np.mean(em ** 2)):>9.3f}{np.sqrt(np.mean((om - em) ** 2)):>8.3f}{sk:>7.2f}{np.corrcoef(om, em)[0, 1]:>7.2f}")


if __name__ == "__main__":
    a = sys.argv[1:]
    DATA.mkdir(parents=True, exist_ok=True)
    if a[0] == "products":
        products(a[1])
    elif a[0] == "fit":
        run_fit(a[1], a[2], int(a[3]) if len(a) > 3 else 20)
    elif a[0] == "batch":                                    # batch <fixed_from window> <variants comma list> <windows...>: products + predict-only with the parameters fitted in `fixed_from`
        for w in a[3:]:
            if not (DATA / f"exp40_{w}_products.npz").exists():
                products(w)
            for v in a[2].split(","):
                run_fit(w, v, fixed_from=a[1])
    elif a[0] == "pooled":                                   # pooled <windows comma list> <variants...>
        pooled(a[1].split(","), a[2:])
    else:
        score(a[1], a[2:])

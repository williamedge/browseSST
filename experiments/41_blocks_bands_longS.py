"""
Experiment 41: where does the SST-derived velocity have skill? Block decomposition, frequency bands, longer windows, and a longer / smoother source term S.  (follow-up to exp 40)

Prior: browseGP's velocity kernel (mcmc_M4_v0.01 posterior medians) WITHOUT the M4 and inertial blocks (user decision): rotational + M2 divergent wave + fast. Velocity parameters fixed; S, the diurnal scale and the
equation noise fitted. Variants of the S model:
  bg3        S free (Matern amplitude / length / time + day-by-day diurnal + noise), as in exp 40 (the short, ~1 h, 17 km S that exp 37 / 40 found)
  bg3_longS  S forced to be smooth and slow: l_S = 60 km, tau_S = 6 h fixed; amplitude, diurnal and noise free (the short-scale misfit has to go into the white noise)
  bg3_noS    no Matern S at all (sigma_S = 0.01 scaled, i.e. none): the diurnal term and the white equation noise carry all the non-advective dT/dt
Tiles: 'A', 'B', 'W<d>' = 28 x 28 px x 48 h windows of exp 40 (day d after 2023-04-01); 'L<d>' = 22 x 22 px x 96 h windows (rows 21-43, cols 25-47) for the longer-window test.
Predictions at every in-situ observation (3 px / 3 h margins) come with the posterior-mean contribution of each velocity block (`BrowseGPPrior.cov_params_blocks`: block-weighted prediction-data cross
covariance, the data covariance stays the full model; the blocks sum to the total prediction), so the skill can be attributed to the rotational, M2-wave or fast block.
Band analysis (`bands`): at the moorings, per site and window, obs - tide and each predicted series are regressed on [1, cos w_M2 t, sin w_M2 t] (the window mean, the semidiurnal band, and the rest, which holds
the inertial motion and high frequencies); the semidiurnal band is compared as a complex amplitude (the baroclinic M2 / S2 tide that remains after the barotropic fit: is it seen?).
Run: python experiments/41_blocks_bands_longS.py products <tile> | fit <tile> <variant> [maxiter] | predict <tile> <variant> <fitted-in tile> | bands <variant> <tiles comma list> | score <variant> <tiles>
"""
import json, pathlib, sys, time, warnings
import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = pathlib.Path.home() / "browsesst_data" / "real_test"
from inversion_sst_gp.spacetime import pipeline as P, tiling as TL, tides as TD
from inversion_sst_gp.spacetime.fit import fit
from inversion_sst_gp.spacetime.regional import TideModel

CUBE = "/mnt/c/Users/00099894/Large_Data/Other/browseSST/test_bb23_2023.nc"
TIDE = DATA / "exp38bb23y_full_ms_daily150_tide.pkl"
BG = pathlib.Path("/mnt/c/Users/00099894/OneDrive - The University of Western Australia/pl/wedge/browseGP/results/mcmc_M4_v0.01/mcmc_results.json")
OBS = {"long": "/mnt/c/Users/00099894/Large_Data/Papers/drifterOI_paper2/observations/BB23_Obs_vector_150m_long.nc",
       "corrected": "/mnt/c/Users/00099894/Large_Data/Papers/drifterOI_paper2/observations/BB23_Obs_vector_150m_corrected.nc"}
BLOCKS = ("rot", "div", "fast")
VARIANTS = {
    "bg3": dict(free=("sigma_S", "l_S", "tau0_S", "lr:diurnal", "noise"), fixed={}, start=dict(sigma_S=19.9, l_S=17.5e3, tau0_S=2270.0, noise=4.0, **{"lr:diurnal": 55.0})),
    "bg3_longS": dict(free=("sigma_S", "lr:diurnal", "noise"), fixed=dict(l_S=60e3, tau0_S=6 * 3600.0), start=dict(sigma_S=8.0, noise=20.0, **{"lr:diurnal": 55.0})),
    "bg3_noS": dict(free=("lr:diurnal", "noise"), fixed=dict(sigma_S=0.01, l_S=60e3, tau0_S=6 * 3600.0), start=dict(noise=30.0, **{"lr:diurnal": 55.0})),
}
WIN48 = {"A": 23, "B": 33}


def tile_def(name):
    """(h0, nt, j0, i0, n) of a tile name."""
    if name in WIN48:
        return WIN48[name] * 24, 48, 18, 24, 28
    if name[0] == "W":
        return int(name[1:]) * 24, 48, 18, 24, 28
    if name[0] == "L":
        return int(name[1:]) * 24, 96, 21, 25, 22
    raise KeyError(name)


def tile(cube, name):
    h0, nt, j0, i0, n = tile_def(name)
    return TL.extract_tile(cube, TL.TileSpec(f"bb23{name}", h0, h0 + nt, j0, j0 + n, i0, i0 + n, h0, h0 + nt, j0, j0 + n, i0, i0 + n))


def products_path(name):
    p40 = DATA / f"exp40_{name}_products.npz"
    return p40 if (name[0] in "ABW" and p40.exists()) else DATA / f"exp41_{name}_products.npz"


def products(name):
    cube = xr.open_dataset(CUBE)
    td = tile(cube, name)
    h1 = P.Hyper.load(ROOT / "hpc" / "hyper_real_start.json").stage1
    cfg = P.PipelineConfig()
    xo, yo, to, To = td.observed()
    t0 = time.time()
    gp1 = P.build_stage1(xo, yo, to, To, h1, cfg)
    Tm, Ttm, vTt = P.stage1_products(td, gp1, cfg)
    np.savez(DATA / f"exp41_{name}_products.npz", Tm=Tm, Ttm=Ttm, vTt=vTt)
    print(f"[{name}] {td.n_obs} observed of {td.x.size} grid points; stage-1 products {time.time() - t0:.0f}s saved", flush=True)


def insitu(path, td):
    ds = xr.load_dataset(path)
    df = pd.DataFrame({"time": ds["time"].values, "lon": ds["longitude"].values, "lat": ds["latitude"].values, "u": ds["east_vel"].values, "v": ds["north_vel"].values, "platform": ds["dtype"].values.astype(str)})
    df = df.drop_duplicates(subset=["time", "lon", "lat", "u", "v"])
    df = df[(df.time >= td.time[3]) & (df.time <= td.time[-4]) & (df.lon >= td.lon[3]) & (df.lon <= td.lon[-4]) & (df.lat >= td.lat[3]) & (df.lat <= td.lat[-4])]
    m = df.platform == "mADCP"
    mo = df[m].copy()
    mo["site"] = mo.groupby([mo.lon.round(5), mo.lat.round(5)]).ngroup()
    mo["hour"] = mo.time.dt.floor("1h")
    mo = mo.groupby(["site", "hour"]).agg(time=("hour", "first"), lon=("lon", "mean"), lat=("lat", "mean"), u=("u", "mean"), v=("v", "mean"), platform=("platform", "first")).reset_index(drop=True)
    return pd.concat([df[~m], mo], ignore_index=True).dropna(subset=["u", "v"])


def run(name, variant, maxiter=8, fixed_from=None):
    V = VARIANTS[variant]
    cube = xr.open_dataset(CUBE)
    td = tile(cube, name)
    z_ = np.load(products_path(name))
    tide = TideModel.load(TIDE)
    ub, vb, sup = tide.tile_velocity(td, mask="supported")
    base = P.PipelineConfig()
    idx, Tx, Ty, z = P.stage2_rows(td, z_["Tm"], z_["Ttm"], z_["vTt"], ub, vb, base)
    pr = P.BrowseGPPrior.from_results(BG, lat=-14.1, blocks=BLOCKS)
    pr.free = tuple(V["free"])
    h2 = dict(P.DEFAULT_STAGE2); h2.update(V["start"]); h2.update(V["fixed"])
    cfg = P.PipelineConfig(prior=pr)
    gp2 = P.build_stage2(td, idx, Tx, Ty, z, h2, cfg)
    bounds = dict(pr.bounds)
    print(f"[{name}/{variant}] {idx.size} stage-2 rows of {td.x.size}, blocks {pr.blocks}, free {list(pr.free)}, fixed {V['fixed']}", flush=True)
    ll0 = gp2.loglik(); t0 = time.time()
    if fixed_from:
        fitted = dict(json.loads((DATA / f"exp41_{fixed_from}_{variant}.json").read_text())["fitted"]); nev, ok = 0, True
    else:
        res = fit(gp2, h2, list(pr.free), bounds=bounds, prior_sd_log={n: 1.5 for n in pr.free}, maxiter=maxiter, verbose=False, slots=pr.slots)
        fitted = dict(h2); fitted.update({k: float(res.params[k]) for k in pr.free}); nev, ok = res.n_eval, bool(res.success)
    gp2.update(cov_params=pr.cov_params(fitted), noise=fitted["noise"], lowrank_scales={"diurnal": fitted["lr:diurnal"]})
    ll = gp2.loglik()
    print(f"[{name}/{variant}] fit {nev} evals {time.time() - t0:.0f}s success={ok}; loglik start {ll0:.1f} -> {ll:.1f}", flush=True)
    print("  fitted: " + ", ".join(f"{k} {fitted[k]:.3g}" for k in ("sigma_S", "l_S", "tau0_S", "lr:diurnal", "noise")), flush=True)
    for lab, path in OBS.items():
        o = insitu(path, td)
        xk, yk = td.frame.forward(o.lon.values, o.lat.values)
        tk = (o.time.values - td.time[0]).astype("timedelta64[s]").astype(np.float64)
        nn = gp2.pred_neighbours(xk, yk, tk, 200)
        out = {}
        for tag, only in (("tot", None),) + tuple((b, (b,)) for b in BLOCKS):
            gp2.update(cov_params=pr.cov_params(fitted) if only is None else pr.cov_params_blocks(fitted, only))
            out["u_" + tag], _ = gp2.predict("u", xk, yk, tk, nn_array=nn)
            out["v_" + tag], _ = gp2.predict("v", xk, yk, tk, nn_array=nn)
        gp2.update(cov_params=pr.cov_params(fitted))
        dsum = max(np.abs(sum(out["u_" + b] for b in BLOCKS) - out["u_tot"]).max(), np.abs(sum(out["v_" + b] for b in BLOCKS) - out["v_tot"]).max())
        utd, vtd = tide.velocity(o.lon.values, o.lat.values, o.time.values, mask="supported")
        np.savez(DATA / f"exp41_{name}_{variant}_{lab}.npz", platform=np.array(list(o.platform), dtype="U8"), lon=o.lon.values, lat=o.lat.values, tsec=tk, u=o.u.values, v=o.v.values, ut=np.nan_to_num(utd), vt=np.nan_to_num(vtd), **out)
        print(f"  [{lab}] predicted at {len(o)} in-situ points ({dict(o.platform.value_counts())}); max |sum of blocks - total| = {dsum:.1e} m/s", flush=True)
    (DATA / f"exp41_{name}_{variant}.json").write_text(json.dumps(dict(tile=name, variant=variant, loglik=float(ll), loglik_start=float(ll0), n_eval=int(nev), success=ok, fitted={k: float(v) for k, v in fitted.items()}, n_rows=int(idx.size)), indent=1))


def load(names, variant, lab):
    parts = []
    for nme in names:
        p = DATA / f"exp41_{nme}_{variant}_{lab}.npz"
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=True)
        d = {k: (np.asarray(d[k]).astype(str) if k == "platform" else d[k]) for k in d.files}
        d["win"] = np.full(d["u"].size, nme)
        parts.append(d)
    return {k: np.concatenate([p_[k] for p_ in parts]) for k in parts[0]}, len(parts)


def score(variant, names):
    print(f"{'file':<10}{'platform':<9}{'comp':>5}{'n':>6}{'rms obs-tide':>13}{'rms eddy':>9}{'skill':>7}{'corr':>7}{'slope':>7}  | block rms / corr (rot, div, fast)")
    for lab in OBS:
        d, nw = load(names, variant, lab)
        for plat in ("drifter", "mADCP", "sADCP", "all"):
            sel = np.ones(d["u"].size, bool) if plat == "all" else d["platform"] == plat
            if sel.sum() < 5:
                continue
            for c in ("u", "v"):
                o = d[c][sel] - d[c + "t"][sel]; e = d[c + "_tot"][sel]
                sk = 1 - np.mean((o - e) ** 2) / np.mean(o ** 2)
                bl = "  ".join(f"{np.sqrt(np.mean(d[c + '_' + b][sel] ** 2)):.3f}/{np.corrcoef(o, d[c + '_' + b][sel])[0, 1]:+.2f}" for b in BLOCKS)
                print(f"{lab:<10}{plat:<9}{c:>5}{sel.sum():>6}{np.sqrt(np.mean(o ** 2)):>13.3f}{np.sqrt(np.mean(e ** 2)):>9.3f}{sk:>7.2f}{np.corrcoef(o, e)[0, 1]:>7.2f}{np.sum(o * e) / np.sum(e * e):>7.2f}  | {bl}")


def bands(variant, names):
    """Mooring band analysis (window mean / semidiurnal / rest) of obs - tide and of the predicted velocity (total and per block)."""
    w = TD.omega_of("M2")
    for lab in OBS:
        d, nw = load(names, variant, lab)
        m = d["platform"] == "mADCP"
        key = np.array([f"{a}_{lo:.4f}_{la:.4f}" for a, lo, la in zip(d["win"][m], d["lon"][m], d["lat"][m])])
        series = {}
        for c in ("u", "v"):
            series[c] = dict(obs=(d[c] - d[c + "t"])[m], **{b: d[c + "_" + b][m] for b in ("tot",) + BLOCKS})
        t = d["tsec"][m]
        keys = [k for k in np.unique(key) if (key == k).sum() >= 24]
        print(f"\n== {variant}, in-situ file '{lab}': {len(keys)} site-windows from tiles {','.join(names)}")
        for c in ("u", "v"):
            rows = {b: dict(mean=[], z=[], res=[]) for b in ("obs", "tot") + BLOCKS}
            for k in keys:
                s = key == k
                tt = t[s]; X = np.column_stack([np.ones(tt.size), np.cos(w * tt), np.sin(w * tt)])
                for b in rows:
                    y = series[c]["obs" if b == "obs" else b][s]
                    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
                    rows[b]["mean"].append(coef[0]); rows[b]["z"].append(coef[1] + 1j * coef[2]); rows[b]["res"].append(y - X @ coef)
            mo, zo = np.array(rows["obs"]["mean"]), np.array(rows["obs"]["z"]); ro = np.concatenate(rows["obs"]["res"])
            print(f"  {c}: window mean: rms obs {np.sqrt(np.mean(mo ** 2)):.3f}; semidiurnal (M2-frequency) amplitude: rms obs {np.sqrt(np.mean(np.abs(zo) ** 2)):.3f}; rest: rms obs {np.sqrt(np.mean(ro ** 2)):.3f} m/s")
            print(f"     {'series':<6}{'mean: rms':>10}{'corr':>6}{'slope':>6} |{'M2 band: rms|z|':>16}{'coherence':>10}{'|slope|':>8}{'arg deg':>8} |{'rest: rms':>10}{'corr':>6}{'slope':>6}")
            for b in ("tot",) + BLOCKS:
                mp, zp, rp = np.array(rows[b]["mean"]), np.array(rows[b]["z"]), np.concatenate(rows[b]["res"])
                cs = np.vdot(zp, zo); coh = abs(cs) / np.sqrt(np.vdot(zp, zp).real * np.vdot(zo, zo).real); sl = cs / np.vdot(zp, zp).real
                print(f"     {b:<6}{np.sqrt(np.mean(mp ** 2)):>10.3f}{np.corrcoef(mo, mp)[0, 1]:>6.2f}{np.sum(mo * mp) / np.sum(mp * mp):>6.2f} |{np.sqrt(np.mean(np.abs(zp) ** 2)):>16.3f}{coh:>10.2f}{abs(sl):>8.2f}{np.degrees(np.angle(sl)):>8.0f} |"
                      f"{np.sqrt(np.mean(rp ** 2)):>10.3f}{np.corrcoef(ro, rp)[0, 1]:>6.2f}{np.sum(ro * rp) / np.sum(rp * rp):>6.2f}")


def _m2_amplitudes(d, names):
    """Epoch-referenced complex M2-frequency amplitude (y = Re[conj(z) e^{i w t}]-style regression coefficients a + i b, t from 2023-04-01 00:00) of obs - tide and of each predicted series for every
    (window, mooring site) with >= 24 hourly samples: {(win, site): {'u'|'v': {'obs': z, 'tot': z, 'rot': z, 'div': z, 'fast': z}}}."""
    w = TD.omega_of("M2")
    m = d["platform"] == "mADCP"
    out = {}
    for win in names:
        sw = m & (d["win"] == win)
        if not sw.any():
            continue
        t0 = tile_def(win)[0] * 3600.0
        key = np.array([f"{lo:.4f}_{la:.4f}" for lo, la in zip(d["lon"][sw], d["lat"][sw])])
        t = d["tsec"][sw]
        for k in np.unique(key):
            s = key == k
            if s.sum() < 24:
                continue
            X = np.column_stack([np.ones(s.sum()), np.cos(w * t[s]), np.sin(w * t[s])])
            rec = {}
            for c in ("u", "v"):
                rec[c] = {}
                for b, y in dict(obs=(d[c] - d[c + "t"])[sw][s], **{b: d[c + "_" + b][sw][s] for b in ("tot",) + BLOCKS}).items():
                    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
                    rec[c][b] = (coef[1] + 1j * coef[2]) * np.exp(1j * w * t0)
            out[(win, k)] = rec
    return out


def _coh(zo, zp):
    cs = np.vdot(zp, zo)
    return abs(cs) / np.sqrt(np.vdot(zo, zo).real * np.vdot(zp, zp).real), cs / np.vdot(zp, zp).real


def bandsig(variant, names, nperm=4000, seed=0):
    """Is the semidiurnal (M2-frequency) amplitude of the predicted velocity coherent with the observed one beyond chance? Statistic: complex coherence of (u, v stacked) over mooring site-windows. Null: the
    prediction of window w is replaced by that of another window w' (a random derangement of the windows, the same site; epoch-referenced phases), which keeps each prediction's amplitude and spatial structure
    but destroys its timing. Also a bootstrap over windows for the coherence / complex slope, and the phase concentration of the observed and predicted M2 across windows at each site."""
    rng = np.random.default_rng(seed)
    for lab in OBS:
        d, nw = load(names, variant, lab)
        rec = _m2_amplitudes(d, names)
        wins = sorted({k[0] for k in rec}); sites = sorted({k[1] for k in rec})
        print(f"\n== {variant}, in-situ file '{lab}': {len(rec)} site-windows, {len(wins)} windows ({','.join(wins)}), {len(sites)} sites; {nperm} permutations")
        zo = {k: np.array([r["u"]["obs"], r["v"]["obs"]]) for k, r in rec.items()}
        for b in ("tot", "div", "rot"):
            zp = {k: np.array([r["u"][b], r["v"][b]]) for k, r in rec.items()}
            ks = list(rec)
            c0, sl0 = _coh(np.concatenate([zo[k] for k in ks]), np.concatenate([zp[k] for k in ks]))
            null = []
            for _ in range(nperm):
                perm = rng.permutation(len(wins))
                while np.any(perm == np.arange(len(wins))) and len(wins) > 1:
                    perm = rng.permutation(len(wins))
                mp = {w: wins[perm[i]] for i, w in enumerate(wins)}
                pairs = [(k, (mp[k[0]], k[1])) for k in ks if (mp[k[0]], k[1]) in rec]
                if len(pairs) < 4:
                    continue
                null.append(_coh(np.concatenate([zo[a] for a, _ in pairs]), np.concatenate([zp[b_] for _, b_ in pairs]))[0])
            null = np.array(null)
            pval = (1 + np.sum(null >= c0)) / (1 + null.size)
            boot = []
            for _ in range(2000):
                wb = rng.choice(len(wins), len(wins), replace=True)
                kb = [k for i in wb for k in ks if k[0] == wins[i]]
                cb, sb = _coh(np.concatenate([zo[k] for k in kb]), np.concatenate([zp[k] for k in kb]))
                boot.append((cb, abs(sb)))
            boot = np.array(boot)
            print(f"  {b:<4} coherence {c0:.2f} (null mean {null.mean():.2f}, 95th pct {np.percentile(null, 95):.2f}, p = {pval:.3f}); complex slope |b| {abs(sl0):.2f} at {np.degrees(np.angle(sl0)):+.0f} deg; "
                  f"bootstrap 95% CI: coherence {np.percentile(boot[:, 0], 2.5):.2f}-{np.percentile(boot[:, 0], 97.5):.2f}, |b| {np.percentile(boot[:, 1], 2.5):.2f}-{np.percentile(boot[:, 1], 97.5):.2f}")
        # phase concentration across windows per site (observed baroclinic M2 phase-locked over the record?)
        print("  phase concentration across windows R (1 = same epoch-referenced phase every window; random ~ 0.9/sqrt(N)):")
        for tag in ("obs", "div"):
            Rs = []
            for k in sites:
                for c in ("u", "v"):
                    z = np.array([rec[(w, k)][c][tag] for w in wins if (w, k) in rec])
                    if z.size >= 3:
                        Rs.append((abs(np.mean(z / np.abs(z))), z.size))
            print(f"    {tag}: mean R {np.mean([r for r, _ in Rs]):.2f} over {len(Rs)} site x component series (mean N = {np.mean([n for _, n in Rs]):.1f}, random expectation ~ {np.mean([0.89 / np.sqrt(n) for _, n in Rs]):.2f})")


if __name__ == "__main__":
    a = sys.argv[1:]
    DATA.mkdir(parents=True, exist_ok=True)
    if a[0] == "products":
        products(a[1])
    elif a[0] == "fit":
        run(a[1], a[2], int(a[3]) if len(a) > 3 else 8)
    elif a[0] == "predict":
        run(a[1], a[2], fixed_from=a[3])
    elif a[0] == "bands":
        bands(a[1], a[2].split(","))
    elif a[0] == "bandsig":
        bandsig(a[1], a[2].split(","))
    else:
        score(a[1], a[2].split(","))

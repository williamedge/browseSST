"""
Experiment 43: snapshots and time series of a merged run (default: the v0.08 control, `run_v008_control`) with the BB23 in-situ observations overlaid.

Observations (browseGP's files): moorings (mADCP, hourly means per site) from `BB23_Obs_vector_150m_long.nc` (longer records; Ekman / Stokes NOT removed, but the ADCP is below the surface layer), drifters and ship ADCP from
`BB23_Obs_vector_150m_corrected.nc` (Ekman-balanced currents removed, Stokes subtracted). The in-situ velocity contains the barotropic + baroclinic tide, inertial motion and depth-dependent flow that SST cannot see; the model
fields are the posterior means: u = tide + rotational ('eddy') + M2 ring wave (m2) + short-scale M2 block (m2w).

Figures (written to ~/browsesst_data/real_test/ as exp43_<tag>_*.png):
  snapshots      : maps at four hours (model total velocity, eddy block, M2 blocks) with drifters (+-1.5 h), ship ADCP (+-1.5 h) and moorings overlaid
  moorings_5d    : 5-day hourly series at each mooring site on a model deep pixel: observed, tide only, model total, eddy
  moorings_full  : the whole mooring record, 25 h running mean (subtidal): observed vs model eddy and vs model total
Prints hourly and subtidal scores (correlation, rmse) of observed vs tide / model total, and of observed-minus-tide vs the model non-tide part.

Run: python experiments/43_control_plots.py [run dir] [tag]
"""
import pathlib, sys, warnings
import numpy as np, pandas as pd, xarray as xr

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = pathlib.Path("/mnt/c/Users/00099894/Large_Data/Other/browseSST")
OBSD = pathlib.Path("/mnt/c/Users/00099894/Large_Data/Papers/drifterOI_paper2/observations")
OUT = pathlib.Path.home() / "browsesst_data" / "real_test"
RUN = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "run_v008_control"
TAG = sys.argv[2] if len(sys.argv) > 2 else RUN.name
SNAP_TIMES = ["2023-05-05T12", "2023-05-12T00", "2023-05-16T12", "2023-05-24T06"]
WIN5 = ("2023-05-14T00", "2023-05-19T00")
OUT.mkdir(parents=True, exist_ok=True)


def load_obs(name):
    d = xr.load_dataset(OBSD / f"BB23_Obs_vector_150m_{name}.nc")
    df = pd.DataFrame({"time": d.time.values, "lon": d.longitude.values, "lat": d.latitude.values, "u": d.east_vel.values, "v": d.north_vel.values, "p": d.dtype.values.astype(str)})
    return df.drop_duplicates(subset=["time", "lon", "lat", "u", "v"]).dropna(subset=["u", "v"])


def moorings(df, t0, t1):
    mo = df[(df.p == "mADCP") & (df.time >= t0) & (df.time <= t1)].copy()
    mo["site"] = mo.groupby([mo.lon.round(2), mo.lat.round(2)]).ngroup()
    mo["hour"] = mo.time.dt.round("1h")
    g = mo.groupby(["site", "hour"]).agg(lon=("lon", "mean"), lat=("lat", "mean"), u=("u", "mean"), v=("v", "mean"), n=("u", "size")).reset_index()
    med = g.groupby("site").n.transform("median")
    return g[g.n >= 0.5 * med]                              # sampling differs by site (2-minute to hourly): drop partial hours


def at_site(da, lon, lat):
    """Bilinear in (lon, lat); nearest pixel where a neighbour is NaN."""
    a = da.interp(lon=lon, lat=lat)
    if np.isfinite(a.values).any():
        return a.values
    return da.sel(lon=lon, lat=lat, method="nearest").values


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 10 else np.nan


def rmse(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


M = xr.open_dataset(RUN / "merged.nc").load()
t0, t1 = M.time.values[0], M.time.values[-1]
long, corrd = load_obs("long"), load_obs("corrected")
mo = moorings(long, t0, t1)
print(f"run {RUN.name}: {M.time.size} hours, {int(M.attrs.get('n_tiles', 0))} tiles; mooring site-hours {len(mo)}")

# ---- model at the mooring sites -----------------------------------------------------------------------------------
sites = {}
for s, g in mo.groupby("site"):
    lon, lat = float(g.lon.mean()), float(g.lat.mean())
    series = {}
    for c in "uv":
        for n in ("", "_eddy", "_m2", "_m2w", "_tide"):
            da = M[f"{c}{n}"]
            a = da.interp(lon=lon, lat=lat).values
            if not np.isfinite(a).any():
                a = da.sel(lon=lon, lat=lat, method="nearest").values
            series[f"{c}{n}"] = a
    ser = pd.DataFrame(series, index=pd.DatetimeIndex(M.time.values))
    obs = g.set_index("hour")[["u", "v"]].reindex(ser.index)
    ok = int(np.isfinite(ser["u"]).sum())
    print(f"  site {s}: ({lon:.3f}, {lat:.3f}) obs hours {int(obs.u.notna().sum())}, model finite hours {ok}")
    if ok > 24 * 5 and obs.u.notna().sum() > 24 * 5:
        sites[s] = dict(lon=lon, lat=lat, ser=ser, obs=obs)
print(f"{len(sites)} mooring sites usable (on a model deep pixel and >5 days of data)")

# ---- scores --------------------------------------------------------------------------------------------------------
def lowpass(x):
    return pd.Series(x).rolling(25, center=True, min_periods=18).mean().values


print("\nscores of OBSERVED velocity at the moorings (u and v pooled)           corr   rmse (m/s)")
rows = {k: ([], []) for k in ("hourly: obs vs tide", "hourly: obs vs model total", "hourly: obs-tide vs model non-tide", "subtidal: obs vs tide", "subtidal: obs vs model total", "subtidal: obs-tide vs eddy")}
for s, d in sites.items():
    for c in "uv":
        o, ser = d["obs"][c].values, d["ser"]
        tide, tot, eddy = ser[f"{c}_tide"].values, ser[c].values, ser[f"{c}_eddy"].values
        nt = tot - tide
        for k, (a, b) in {"hourly: obs vs tide": (o, tide), "hourly: obs vs model total": (o, tot), "hourly: obs-tide vs model non-tide": (o - tide, nt),
                          "subtidal: obs vs tide": (lowpass(o), lowpass(tide)), "subtidal: obs vs model total": (lowpass(o), lowpass(tot)), "subtidal: obs-tide vs eddy": (lowpass(o) - lowpass(tide), lowpass(eddy))}.items():
            rows[k][0].append(a); rows[k][1].append(b)
for k, (a, b) in rows.items():
    a, b = np.concatenate(a), np.concatenate(b)
    print(f"  {k:40s} {corr(a, b):6.2f} {rmse(a, b):7.3f}")

# ---- figure 1: snapshots ------------------------------------------------------------------------------------------
fig, ax = plt.subplots(len(SNAP_TIMES), 4, figsize=(21, 4.6 * len(SNAP_TIMES)), sharex=True, sharey=True, constrained_layout=True)
lon, lat, st = M.lon.values, M.lat.values, 3
for r, ts in enumerate(SNAP_TIMES):
    t = np.datetime64(ts)
    k = int(np.argmin(np.abs(M.time.values - t)))
    tk = M.time.values[k]
    for a, (title, comp) in zip(ax[r], [("model total (tide + all blocks)", ("", None)), ("eddy (rotational)", ("_eddy", None)), ("M2 ring + short-scale M2 blocks", ("_m2", "_m2w")), ("tide only", ("_tide", None))]):
        u = M[f"u{comp[0]}"].values[k] + (M[f"u{comp[1]}"].values[k] if comp[1] else 0)
        v = M[f"v{comp[0]}"].values[k] + (M[f"v{comp[1]}"].values[k] if comp[1] else 0)
        sp = np.hypot(u, v)
        im = a.pcolormesh(lon, lat, sp, cmap="viridis", vmin=0, vmax=0.6, shading="auto")
        a.quiver(lon[::st], lat[::st], u[::st, ::st], v[::st, ::st], color="w", scale=8, width=0.003)
        a.set_title(f"{str(tk)[:13]} UTC  {title}\nrms {np.sqrt(np.nanmean(sp ** 2)):.2f} m/s", fontsize=10)
        if title.startswith("model total") or title.startswith("tide"):
            w = pd.Timedelta("90min")
            for lab, df, col, mk in (("drifter", corrd[corrd.p == "drifter"], "orange", "o"), ("ship ADCP", corrd[corrd.p == "sADCP"], "magenta", "^")):
                s = df[(df.time >= tk - w) & (df.time <= tk + w)]
                if len(s):
                    a.quiver(s.lon, s.lat, s.u, s.v, color=col, scale=8, width=0.005, edgecolor="k", linewidth=0.4, label=f"{lab} ({len(s)})")
            g = mo[mo.hour == pd.Timestamp(tk).round("1h")]
            if len(g):
                a.quiver(g.lon, g.lat, g.u, g.v, color="red", scale=8, width=0.005, edgecolor="k", linewidth=0.4, label=f"mooring ({len(g)})")
            a.legend(loc="lower right", fontsize=8)
    for s, d in sites.items():
        ax[r, 0].plot(d["lon"], d["lat"], "k+", ms=7)
    ax[r, 0].set_ylabel("lat")
fig.colorbar(im, ax=ax, shrink=0.4, label="speed (m/s)", location="right")
fig.suptitle(f"{RUN.name}: velocity snapshots with in-situ observations (arrows: model white, observed coloured; + = mooring sites on model deep pixels)")
fig.savefig(OUT / f"exp43_{TAG}_snapshots.png", dpi=80)
print("wrote", OUT / f"exp43_{TAG}_snapshots.png")

# ---- figure 2: 5-day series ---------------------------------------------------------------------------------------
ns = len(sites)
fig, ax = plt.subplots(ns, 2, figsize=(17, 3.0 * ns), sharex=True, constrained_layout=True, squeeze=False)
for r, (s, d) in enumerate(sites.items()):
    ser, obs = d["ser"].loc[WIN5[0]:WIN5[1]], d["obs"].loc[WIN5[0]:WIN5[1]]
    for j, c in enumerate("uv"):
        a = ax[r, j]
        a.plot(obs.index, obs[c], "k", lw=1.6, label="observed")
        a.plot(ser.index, ser[f"{c}_tide"], color="0.6", lw=1.2, label="tide only")
        a.plot(ser.index, ser[c], "C3", lw=1.4, label="model total")
        a.plot(ser.index, ser[f"{c}_tide"] + ser[f"{c}_eddy"], "C0", lw=1.0, ls="--", label="tide + eddy")
        a.axhline(0, color="k", lw=0.3)
        a.set_ylabel(f"{c} (m/s)")
        a.set_title(f"site {s} ({d['lon']:.3f}, {d['lat']:.3f})  {c}   corr obs/total {corr(obs[c].values, ser[c].values):.2f}, obs/tide {corr(obs[c].values, ser[f'{c}_tide'].values):.2f}", fontsize=10)
        if r == 0 and j == 0:
            a.legend(ncol=4, fontsize=8)
fig.suptitle(f"{RUN.name}: hourly velocity at the moorings, {WIN5[0][:10]} to {WIN5[1][:10]}")
fig.savefig(OUT / f"exp43_{TAG}_moorings_5d.png", dpi=80)
print("wrote", OUT / f"exp43_{TAG}_moorings_5d.png")

# ---- figure 3: whole record, subtidal -----------------------------------------------------------------------------
fig, ax = plt.subplots(ns, 2, figsize=(17, 3.0 * ns), sharex=True, constrained_layout=True, squeeze=False)
for r, (s, d) in enumerate(sites.items()):
    ser, obs = d["ser"], d["obs"]
    for j, c in enumerate("uv"):
        a = ax[r, j]
        a.plot(ser.index, lowpass(obs[c].values), "k", lw=1.6, label="observed (25 h mean)")
        a.plot(ser.index, lowpass(ser[c].values), "C3", lw=1.2, label="model total (25 h mean)")
        a.plot(ser.index, lowpass(ser[f"{c}_eddy"].values), "C0", lw=1.2, label="model eddy (25 h mean)")
        a.plot(ser.index, lowpass(ser[f"{c}_tide"].values), color="0.6", lw=1.0, label="tide (25 h mean)")
        a.axhline(0, color="k", lw=0.3)
        a.set_ylabel(f"{c} (m/s)")
        a.set_title(f"site {s}  {c}   subtidal corr obs/total {corr(lowpass(obs[c].values), lowpass(ser[c].values)):.2f}, obs-tide/eddy {corr(lowpass(obs[c].values) - lowpass(ser[f'{c}_tide'].values), lowpass(ser[f'{c}_eddy'].values)):.2f}", fontsize=10)
        if r == 0 and j == 0:
            a.legend(ncol=4, fontsize=8)
fig.suptitle(f"{RUN.name}: subtidal (25 h running mean) velocity at the moorings, whole record")
fig.savefig(OUT / f"exp43_{TAG}_moorings_full.png", dpi=80)
print("wrote", OUT / f"exp43_{TAG}_moorings_full.png")

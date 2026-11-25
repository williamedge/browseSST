"""
Experiment 46: animation of the M2 divergent-wave block (the ring kernel's posterior mean) of a merged run (default: the v0.08 control).

Three panels per frame: (left) the M2 block velocity (arrows, speed shaded) with the mooring observations minus the barotropic tide as red arrows; (middle) its horizontal divergence (1e-5 s-1, shaded; a propagating wave
shows as moving bands); (right) a time-distance diagram of u_m2 along one latitude row with a cursor at the current time (a ring wave of ~67 km wavelength would show as slanted stripes with a 12.4 h period).
Writes ~/browsesst_data/real_test/exp46_<tag>_m2.mp4.

Run: python experiments/46_m2_animation.py [run dir] [start YYYY-MM-DDTHH] [hours] [row latitude]      (default: run_v008_control, 2023-05-03T00, 120 h, -13.6)
"""
import pathlib, sys, warnings
import numpy as np, pandas as pd, xarray as xr

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

BASE = pathlib.Path("/mnt/c/Users/00099894/Large_Data/Other/browseSST")
OBSD = pathlib.Path("/mnt/c/Users/00099894/Large_Data/Papers/drifterOI_paper2/observations")
OUT = pathlib.Path.home() / "browsesst_data" / "real_test"
RUN = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "run_v008_control"
START = np.datetime64(sys.argv[2] if len(sys.argv) > 2 else "2023-05-03T00")
NH = int(sys.argv[3]) if len(sys.argv) > 3 else 120
ROWLAT = float(sys.argv[4]) if len(sys.argv) > 4 else -13.6
OUT.mkdir(parents=True, exist_ok=True)

M = xr.open_dataset(RUN / "merged.nc").load()
k0 = int(np.argmin(np.abs(M.time.values - START))); ks = np.arange(k0, k0 + NH)
lon, lat = M.lon.values, M.lat.values
dx = np.diff(lon).mean() * 111.32e3 * np.cos(np.deg2rad(lat.mean())); dy = np.diff(lat).mean() * 111.2e3
U, V = M.u_m2.values[ks], M.v_m2.values[ks]
SP = np.hypot(U, V)
DIV = np.stack([np.gradient(U[i], dx, axis=1) + np.gradient(V[i], dy, axis=0) for i in range(NH)]) * 1e5
vd = np.nanpercentile(np.abs(DIV), 98)
deep = np.isfinite(M.u.values[k0])
jr = int(np.argmin(np.abs(lat - ROWLAT)))
hov = U[:, jr, :]

# observed minus barotropic tide at the mooring sites (hourly means)
d = xr.load_dataset(OBSD / "BB23_Obs_vector_150m_long.nc")
df = pd.DataFrame({"time": d.time.values, "lon": d.longitude.values, "lat": d.latitude.values, "u": d.east_vel.values, "v": d.north_vel.values, "p": d.dtype.values.astype(str)})
t0, t1 = M.time.values[ks[0]], M.time.values[ks[-1]]
mo = df[(df.p == "mADCP") & (df.time >= t0 - np.timedelta64(1, "h")) & (df.time <= t1 + np.timedelta64(1, "h"))].dropna(subset=["u", "v"]).copy()
mo["site"] = mo.groupby([mo.lon.round(2), mo.lat.round(2)]).ngroup(); mo["hour"] = mo.time.dt.round("1h")
g = mo.groupby(["site", "hour"]).agg(lon=("lon", "mean"), lat=("lat", "mean"), u=("u", "mean"), v=("v", "mean"), n=("u", "size")).reset_index()
g = g[g.n >= 0.5 * g.groupby("site").n.transform("median")]
obs = {}
for s, x in g.groupby("site"):
    lo, la = float(x.lon.mean()), float(x.lat.mean())
    ut, vt = M.u_tide.interp(lon=lo, lat=la).values[ks], M.v_tide.interp(lon=lo, lat=la).values[ks]
    if not np.isfinite(ut).any():
        continue
    ser = x.set_index("hour")[["u", "v"]].reindex(pd.DatetimeIndex(M.time.values[ks]))
    obs[s] = (lo, la, ser.u.values - ut, ser.v.values - vt)

fig = plt.figure(figsize=(17, 6.2), constrained_layout=True)
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.8])
a1, a2, a3 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1], sharex=None), fig.add_subplot(gs[2])
st = 3
m1 = a1.pcolormesh(lon, lat, SP[0], cmap="viridis", vmin=0, vmax=0.4, shading="auto")
q1 = a1.quiver(lon[::st], lat[::st], U[0][::st, ::st], V[0][::st, ::st], color="w", scale=5, width=0.003)
m2 = a2.pcolormesh(lon, lat, DIV[0], cmap="RdBu_r", vmin=-vd, vmax=vd, shading="auto")
q2 = a2.quiver(lon[::st], lat[::st], U[0][::st, ::st], V[0][::st, ::st], color="k", scale=5, width=0.002, alpha=0.5)
for a in (a1, a2):
    a.contour(lon, lat, deep.astype(float), [0.5], colors="r", linewidths=0.6)
    a.axhline(lat[jr], color="w" if a is a1 else "k", lw=0.5, ls=":")
    for s, (lo, la, _, _) in obs.items():
        a.plot(lo, la, "k+" if a is a1 else "m+", ms=8)
qo = a1.quiver([o[0] for o in obs.values()], [o[1] for o in obs.values()], [o[2][0] for o in obs.values()], [o[3][0] for o in obs.values()], color="red", scale=5, width=0.006, edgecolor="k", linewidth=0.5, zorder=5)
a1.set_title("M2 divergent block (arrows, speed m/s)\nred: observed minus barotropic tide at moorings", fontsize=10)
a2.set_title("horizontal divergence of the M2 block (1e-5 /s)", fontsize=10)
fig.colorbar(m1, ax=a1, shrink=0.7, location="bottom", pad=0.02, label="speed (m/s)")
fig.colorbar(m2, ax=a2, shrink=0.7, location="bottom", pad=0.02, label="divergence (1e-5 /s)")
th = (M.time.values[ks] - M.time.values[ks[0]]).astype("timedelta64[m]").astype(float) / 60.0
vm = np.nanpercentile(np.abs(hov), 98)
a3.pcolormesh(lon, th, hov, cmap="RdBu_r", vmin=-vm, vmax=vm, shading="auto")
cur = a3.axhline(0, color="k", lw=1.5)
a3.set_xlabel("lon"); a3.set_ylabel("hours since start"); a3.set_title(f"u_m2 along {lat[jr]:.2f}°N  (dotted row; blue/red = west/east)", fontsize=10)
ttl = fig.suptitle("")

w = FFMpegWriter(fps=8, bitrate=2500)
path = OUT / f"exp46_{RUN.name}_m2.mp4"
with w.saving(fig, str(path), dpi=80):
    for i in range(NH):
        m1.set_array(SP[i].ravel()); q1.set_UVC(U[i][::st, ::st], V[i][::st, ::st])
        m2.set_array(DIV[i].ravel()); q2.set_UVC(U[i][::st, ::st], V[i][::st, ::st])
        qo.set_UVC(np.array([o[2][i] for o in obs.values()]), np.array([o[3][i] for o in obs.values()]))
        cur.set_ydata([th[i], th[i]])
        ttl.set_text(f"{RUN.name}: M2 divergent block, {str(M.time.values[ks[i]])[:13]} UTC")
        w.grab_frame()
print("wrote", path, f"({NH} frames); divergence colour limit +-{vd:.2f}e-5 /s; M2 block speed rms over the window {np.sqrt(np.nanmean(SP ** 2)):.3f} m/s")

"""
Experiment 47: time-longitude (Hovmoller) diagrams of the M2 divergent block, u and v, along one east-west row of a merged run (the row of the exp 46 animation).

Run: python experiments/47_m2_hovmoller.py [run dir] [start YYYY-MM-DDTHH] [hours] [row latitude] [field]
Defaults: run_v008_control, 2023-05-03T00, 120 h, -13.6, m2 (field = m2 | eddy | m2w | total ...: the `u_<field>` / `v_<field>` variables of merged.nc; 'total' = u / v).
Writes ~/browsesst_data/real_test/exp47_<tag>_<field>_hovmoller.png (map of the region + the two Hovmoller panels) and exp47_<tag>_map.png (the map alone).
The map shows the cube's depth (the gridded SUNTANS bathymetry, `depth_raw`), the 250 m and 500 m contours, the transect row, and an unlabelled mooring marker at (123.28E, 13.81S).
"""
import pathlib, sys, warnings
import numpy as np, xarray as xr
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

BASE = pathlib.Path("/mnt/c/Users/00099894/Large_Data/Other/browseSST")
OUT = pathlib.Path.home() / "browsesst_data" / "real_test"
RUN = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "run_v008_control"
START = np.datetime64(sys.argv[2] if len(sys.argv) > 2 else "2023-05-03T00")
NH = int(sys.argv[3]) if len(sys.argv) > 3 else 120
ROWLAT = float(sys.argv[4]) if len(sys.argv) > 4 else -13.6
FIELD = sys.argv[5] if len(sys.argv) > 5 else "m2"
CUBE = BASE / "test_bb23_2023.nc"
MOOR = (123.28, -13.81)
OUT.mkdir(parents=True, exist_ok=True)

M = xr.open_dataset(RUN / "merged.nc").load()
k0 = int(np.argmin(np.abs(M.time.values - START))); ks = np.arange(k0, k0 + NH)
jr = int(np.argmin(np.abs(M.lat.values - ROWLAT)))
lon, t = M.lon.values, M.time.values[ks]
names = ("u", "v") if FIELD == "total" else (f"u_{FIELD}", f"v_{FIELD}")
data = [M[n].values[ks, jr, :] for n in names]
vm = max(np.nanpercentile(np.abs(a), 98) for a in data)
fin = np.isfinite(data[0]).any(0) | np.isfinite(data[1]).any(0)
xlim = (lon[fin].min() - 0.02, lon[fin].max() + 0.02)                 # only the longitudes with output on this row

KM = 111.32 * np.cos(np.deg2rad(M.lat.values[jr]))
with xr.open_dataset(CUBE) as C:
    dep = C.depth_raw.load()


def draw_map(a, label_km=True):
    """Bathymetry shading, 250 / 500 m contours, the transect (with distance ticks matching the Hovmoller top axis) and the mooring."""
    pm = a.pcolormesh(dep.lon, dep.lat, dep.values, cmap="Blues", norm=matplotlib.colors.PowerNorm(0.6, vmin=0, vmax=2800), shading="auto")
    c250 = a.contour(dep.lon, dep.lat, dep.values, [250], colors="k", linewidths=1.3)
    c500 = a.contour(dep.lon, dep.lat, dep.values, [500], colors="k", linewidths=0.9, linestyles="--")
    a.clabel(c250, fmt="250 m", fontsize=7, inline=True); a.clabel(c500, fmt="500 m", fontsize=7, inline=True)
    a.plot(lon[fin], np.full(fin.sum(), M.lat.values[jr]), color="tab:red", lw=2.2, solid_capstyle="butt", label=f"transect ({M.lat.values[jr]:.2f}°N)", zorder=4)
    if label_km:
        for k in np.arange(0, (lon[-1] - lon[0]) * KM, 50.0):
            lo = lon[0] + k / KM
            if xlim[0] <= lo <= xlim[1]:
                a.plot(lo, M.lat.values[jr], "|", color="tab:red", ms=9, mew=1.5, zorder=4)
                a.text(lo, M.lat.values[jr] + 0.05, f"{k:.0f}", color="tab:red", ha="left" if k == 0 else "center", va="bottom", fontsize=7)
    a.scatter(*MOOR, marker="o", s=55, facecolor="gold", edgecolor="k", linewidths=1.0, zorder=5, label="mooring")
    a.set_xlim(lon[0], lon[-1]); a.set_ylim(M.lat.values[0], M.lat.values[-1])
    a.set_aspect(1 / np.cos(np.deg2rad(M.lat.values.mean())))
    a.set_xlabel("longitude (°E)"); a.set_ylabel("latitude (°N)")
    a.set_title("region, 250 / 500 m isobaths and transect\n(red ticks: km from the western edge of the cube, as on the Hovmoller top axes)", fontsize=9)
    a.legend(loc="lower left", fontsize=8, framealpha=0.9)
    return pm


# the map alone
fm, am = plt.subplots(figsize=(6.5, 6.2), constrained_layout=True)
pm = draw_map(am)
fm.colorbar(pm, ax=am, shrink=0.8, label="depth (m)")
pmap = OUT / f"exp47_{RUN.name}_map.png"
fm.savefig(pmap, dpi=110); plt.close(fm)

# map + Hovmoller panels
fig = plt.figure(figsize=(19, 7.5), constrained_layout=True)
gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1, 1])
am = fig.add_subplot(gs[0]); a1 = fig.add_subplot(gs[1]); a2 = fig.add_subplot(gs[2], sharey=a1)
pm = draw_map(am)
fig.colorbar(pm, ax=am, shrink=0.6, location="bottom", pad=0.03, label="depth (m)")
ax = [a1, a2]
for a, d, c in zip(ax, data, "uv"):
    im = a.pcolormesh(lon, t, d, cmap="RdBu_r", vmin=-vm, vmax=vm, shading="auto")
    a.set_title(f"{c} ({FIELD})", fontsize=12)
    a.set_xlabel("longitude (°E)"); a.set_xlim(*xlim)
    sec = a.secondary_xaxis("top", functions=(lambda x: (x - lon[0]) * KM, lambda k: lon[0] + k / KM))
    sec.set_xlabel("distance from the western edge (km)")
    a.yaxis.set_major_locator(mdates.DayLocator()); a.yaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    a.yaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
    a.grid(axis="y", which="major", color="k", lw=0.3, alpha=0.4)
ax[0].set_ylabel("time (UTC)"); ax[1].tick_params(labelleft=False)
fig.colorbar(im, ax=ax, shrink=0.6, location="right", label="velocity (m/s)")
fig.suptitle(f"{RUN.name}: {FIELD} block along {M.lat.values[jr]:.2f}°N, {str(t[0])[:13]} to {str(t[-1])[:13]} UTC (blue = westward / southward)")
p = OUT / f"exp47_{RUN.name}_{FIELD}_hovmoller.png"
fig.savefig(p, dpi=100)
print("wrote", p, "and", pmap, f"| colour limit ±{vm:.3f} m/s | row {M.lat.values[jr]:.3f}°N")

"""
Experiment 39: independent check of the SST-regression tide against the BB23 mooring velocities (the in-situ data of browseGP).

BB23 observations (browseGP `scripts/fit_multiscale_real.py` loads them): `BB23_Obs_vector_150m_long.nc` (Feb 1 - Jul 31 2023, mADCP = 6 moorings at lon 122.84-123.30, lat -14.24 to -13.94, ~1 min sampling;
drifters and ship ADCP are not used: moorings give a fixed point for the harmonic analysis). The `_corrected` file (Ekman-balanced currents removed, Stokes drift subtracted from the drifters) does not
change the tidal harmonics of the moorings.

The user's finding: the site-to-site differences among the moorings are BAROCLINIC; a PLANE fitted through all the moorings for each constituent reproduces the depth-averaged currents' harmonics almost
perfectly, so the barotropic tide = that plane. The SST regression (smooth splines, 150 km) is therefore compared with the PLANE through the mooring harmonics, not with each mooring.

Mode `moor <t0> <t1> <tag>`: per-site harmonic regression of the mooring u, v (hourly means in [t0, t1]; constituents M2 S2 N2 K1 O1 + a cubic-spline mean with 7-day knots for the sub-tidal flow),
u = A cos(w t - phi) with t in seconds from EPOCH (2023-04-01T00:00), saved to ~/browsesst_data/real_test/exp39_moorings_<tag>.json.
Mode `compare <tag> <tide pickle prefix> ...`: for each SST tide model (~/browsesst_data/real_test/<prefix>_tide.pkl; its phases are referred to the same epoch) tabulates amplitude and phase of M2 and S2
at each site against (a) the individual mooring and (b) the plane through all moorings (least squares in the complex amplitude, local coordinates in km).
Run: python experiments/39_bb23_mooring_tides.py moor 2023-04-21 2023-05-29 win | compare win exp38bb23y_win_ms_daily150
"""
import json, os, sys, warnings
import numpy as np
import xarray as xr
import pandas as pd
from scipy.interpolate import BSpline

warnings.filterwarnings("ignore")
from inversion_sst_gp.spacetime import tides as TD

OBS = "/mnt/c/Users/00099894/Large_Data/Papers/drifterOI_paper2/observations/BB23_Obs_vector_150m_long.nc"
OUT = os.path.expanduser('~/browsesst_data/real_test')
EPOCH = np.datetime64('2023-04-01T00:00:00')
NAMES = ('M2', 'S2', 'N2', 'K1', 'O1')
SHOW = ('M2', 'S2')


def moor(t0, t1, tag):
    ds = xr.load_dataset(OBS)
    pl = ds['dtype'].values.astype(str)
    m = pl == 'mADCP'
    df = pd.DataFrame({'t': ds['time'].values[m], 'lon': ds['longitude'].values[m], 'lat': ds['latitude'].values[m], 'u': ds['east_vel'].values[m], 'v': ds['north_vel'].values[m]})
    df = df.drop_duplicates()
    df['site'] = df.groupby([df.lon.round(5), df.lat.round(5)]).ngroup()
    out = []
    for s, g in df.groupby('site'):
        g = g.set_index('t').sort_index()
        g = g[(g.index >= np.datetime64(t0)) & (g.index < np.datetime64(t1))]
        if len(g) < 100:
            continue
        h = g[['u', 'v']].resample('1h').mean().dropna()
        tsec = ((h.index.values - EPOCH) / np.timedelta64(1, 's')).astype(float)
        k = 3; a, b = tsec.min(), tsec.max()
        n = max(4, int((b - a) / (7 * 86400))); inner = np.linspace(a, b, n + 1)
        kn = np.r_[[a] * k, inner, [b] * k]
        Bm = BSpline.design_matrix(np.clip(tsec, a, b - 1e-6), kn, k).toarray()
        cols = [Bm]
        for nm in NAMES:
            w = TD.omega_of(nm); cols.append(np.column_stack([np.cos(w * tsec), np.sin(w * tsec)]))
        X = np.hstack(cols)
        rec = dict(site=int(s), lon=float(g.lon.mean()), lat=float(g.lat.mean()), n_hours=int(len(h)), days=float((b - a) / 86400), const={})
        for comp in ('u', 'v'):
            y = h[comp].values
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            rec[f'{comp}_resid_rms'] = float((y - X @ beta).std()); rec[f'{comp}_rms'] = float(y.std())
            for i, nm in enumerate(NAMES):
                c, sn = beta[Bm.shape[1] + 2 * i: Bm.shape[1] + 2 * i + 2]
                rec['const'].setdefault(nm, {})[comp] = dict(A=float(np.hypot(c, sn)), phi_deg=float(np.degrees(np.arctan2(sn, c))))
        out.append(rec)
        print(f"site {rec['site']}: {rec['lon']:.3f} {rec['lat']:.3f}, {rec['n_hours']} h ({rec['days']:.0f} d); "
              + "; ".join(f"{nm} A_u {rec['const'][nm]['u']['A']:.3f} A_v {rec['const'][nm]['v']['A']:.3f}" for nm in SHOW), flush=True)
    os.makedirs(OUT, exist_ok=True)
    json.dump(dict(epoch=str(EPOCH), window=[t0, t1], sites=out), open(os.path.join(OUT, f'exp39_moorings_{tag}.json'), 'w'), indent=1)


def plane_fit(xk, yk, z):
    """Least-squares plane z = a + b x + c y (complex z) through the sites; returns a function (x, y) -> z."""
    A = np.column_stack([np.ones_like(xk), xk, yk])
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    return lambda x, y: coef[0] + coef[1] * x + coef[2] * y


def compare(tag, prefixes):
    from inversion_sst_gp.spacetime.regional import TideModel
    mo = json.load(open(os.path.join(OUT, f'exp39_moorings_{tag}.json')))
    sites = mo['sites']
    lon = np.array([s['lon'] for s in sites]); lat = np.array([s['lat'] for s in sites])
    lon0, lat0 = lon.mean(), lat.mean()
    xk = (lon - lon0) * 111.32 * np.cos(np.deg2rad(lat0)); yk = (lat - lat0) * 110.57
    print(f"{len(sites)} mooring sites, window {mo['window']}, hours per site {[s['n_hours'] for s in sites]}")

    def insitu(nm, comp):
        return np.array([s['const'][nm][comp]['A'] * np.exp(-1j * np.radians(s['const'][nm][comp]['phi_deg'])) for s in sites])
    print("\nin-situ barotropic proxy: plane through the moorings; spread of the individual moorings about it (rms |z - plane| / rms |plane|, i.e. the 'baroclinic' part):")
    plane = {}
    for nm in SHOW:
        for comp in ('u', 'v'):
            z = insitu(nm, comp); f = plane_fit(xk, yk, z); plane[(nm, comp)] = f
            zp = f(xk, yk)
            print(f"  {nm} {comp}: plane at the centroid {np.abs(f(0.0, 0.0)):.3f} m/s @ {-np.degrees(np.angle(f(0.0, 0.0))):.0f} deg; plane gradient {1e3 * np.abs(np.polyfit(xk, np.abs(zp), 1)[0]):.2f} mm/s per km; "
                  f"spread about the plane {np.sqrt(np.mean(np.abs(z - zp) ** 2)) / np.sqrt(np.mean(np.abs(zp) ** 2)):.2f}")
    for pre in prefixes:
        m = TideModel.load(os.path.join(OUT, f'{pre}_tide.pkl'))
        dsec = float((np.datetime64(m.t_ref, 'ns') - np.datetime64(EPOCH, 'ns')) / np.timedelta64(1, 's'))
        x, y = m.frame.forward(lon, lat)
        sup = m.fit.supported(x, y)
        h = m.fit.harmonics(x, y)
        print(f"\n[{pre}] t_ref {np.datetime64(m.t_ref, 's')}, supported at the sites {sup.astype(int)}; phases shifted to the epoch {EPOCH}")
        tot = {}
        for nm in [n for n in SHOW if n in h]:
            w = TD.omega_of(nm)
            for comp, (A, ph) in (('u', ('Au', 'phi_u')), ('v', ('Av', 'phi_v'))):
                zs = h[nm][A] * np.exp(-1j * (h[nm][ph] + w * dsec))
                zp = plane[(nm, comp)](xk, yk); zo = insitu(nm, comp)
                ok = sup
                dphi = np.degrees(np.angle(zs[ok] * np.conj(zp[ok])))
                e_plane = np.sqrt(np.mean(np.abs(zs[ok] - zp[ok]) ** 2)) / np.sqrt(np.mean(np.abs(zp[ok]) ** 2))
                e_site = np.sqrt(np.mean(np.abs(zs[ok] - zo[ok]) ** 2)) / np.sqrt(np.mean(np.abs(zo[ok]) ** 2))
                print(f"   {nm} {comp}: |plane| {np.mean(np.abs(zp[ok])):.3f}  |SST| {np.mean(np.abs(zs[ok])):.3f}  amplitude ratio {np.mean(np.abs(zs[ok])) / np.mean(np.abs(zp[ok])):.2f}  "
                      f"phase diff to the plane {np.mean(dphi):+.0f} deg (sites {np.round(dphi).astype(int).tolist()})  rel. error vs plane {e_plane:.2f}, vs sites {e_site:.2f}")
                tot.setdefault(nm, [0.0, 0.0]); tot[nm][0] += np.sum(np.abs(zs[ok] - zp[ok]) ** 2); tot[nm][1] += np.sum(np.abs(zp[ok]) ** 2)
        print("   u and v together, error / plane rms: " + ", ".join(f"{nm} {np.sqrt(a / b):.2f}" for nm, (a, b) in tot.items()) + "  (1.0 = no better than predicting zero)")


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == 'moor':
        moor(a[1], a[2], a[3])
    else:
        compare(a[1], a[2:])

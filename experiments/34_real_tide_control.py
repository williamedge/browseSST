"""
Experiment 34: is the real-data tide fit above its noise floor?  Control frequencies.

The regional tide fit on the real test cube (hpc/make_test_dataset.py output) gave M2 / S2 / K1 / O1 amplitudes all about 0.2 m/s (median), which is not the physical ratio (S2 should be well below M2).
Control: run the SAME code on the SAME data with four fictitious constituents at frequencies that are neither tidal nor solar harmonics (deg / hour: 18.0, 22.0, 26.5, 32.0 = periods 20, 16.4, 13.6,
11.25 h; each > 2 deg/h from any tidal or solar line, > 0.5 deg/h from each other, i.e. resolved by a 28 d record). A real tide stands out above the control amplitudes; if the controls are as
large, the fit is absorbing broadband variability (eddy advection, diurnal-heating residuals) and its amplitudes are a noise floor.
Split-half test (mode `split`): the same fit on the EVEN days only and on the ODD days only (same time origin), then the complex coherence of the fitted amplitude fields (u and v stacked,
over the supported 20 km cells) and the median phase difference between the halves, per constituent. A real, spatially coherent tide gives coherence near 1 and small phase differences; broadband
residual gives coherence near 0 and a uniform phase difference (median |dphi| ~ 90 deg). The control constituents are the null.
Run (control): python experiments/34_real_tide_control.py <cube_file> [min_depth] [time_stride] [spacing_km] [nuisance_harmonics] [nuisance_spacing_km] [days0 days1]
"""
import os, sys, time, warnings
import numpy as np
import xarray as xr

warnings.filterwarnings("ignore")
from inversion_sst_gp.spacetime import regional as RG, tides as TD

CONTROLS = {'X1': 18.0, 'X2': 22.0, 'X3': 26.5, 'X4': 32.0}
TIDAL = ('M2', 'S2', 'K1', 'O1')


def amp_table(model, cube):
    lon, lat = cube['lon'].values, cube['lat'].values
    b = 5
    ds = model.constituents(lon[b // 2::b], lat[b // 2::b])
    sup = ds['supported'].values.astype(bool)
    out = {}
    for nm in model.fit.names:
        au, av = ds[f'{nm}_Au'].values[sup], ds[f'{nm}_Av'].values[sup]
        out[nm] = dict(u_median=float(np.median(au)), v_median=float(np.median(av)), rms=float(np.sqrt(np.mean(au ** 2 + av ** 2))))
    return out


def split_test(path, min_depth=100.0, stride=2, spacing_km=80.0, nuis=4):
    TD.CONSTITUENT_DEG_PER_HOUR.update(CONTROLS)
    cube = xr.open_dataset(path)
    depth = np.asarray(cube['depth'].values, float)
    nt = cube.sizes['time']
    day = np.arange(nt) // 24
    lon, lat = cube['lon'].values, cube['lat'].values
    for label, names in (('tidal', TIDAL), ('control', tuple(CONTROLS))):
        fields = []
        for par in (0, 1):
            cfg = RG.TideConfig(constituents=names, min_depth=min_depth, time_stride=stride, hourly_detrend=os.environ.get('DETREND') or None, spacing=spacing_km * 1e3, nuisance_harmonics=nuis, keep_hours=(day % 2 == par))
            m = RG.fit_region(cube, depth, cfg, log=None)
            ds = m.constituents(lon[2::5], lat[2::5])
            fields.append(ds)
        sup = fields[0]['supported'].values.astype(bool) & fields[1]['supported'].values.astype(bool)
        print(f"[{label}] split-half (even / odd days), {sup.sum()} supported cells", flush=True)
        for nm in names:
            z = []
            for ds in fields:
                a = np.r_[ds[f'{nm}_Au'].values[sup] * np.exp(-1j * np.radians(ds[f'{nm}_phi_u'].values[sup])), ds[f'{nm}_Av'].values[sup] * np.exp(-1j * np.radians(ds[f'{nm}_phi_v'].values[sup]))]
                z.append(a)
            a, b = z
            coh = np.abs(np.vdot(a, b)) / np.sqrt(np.vdot(a, a).real * np.vdot(b, b).real)
            dphi = np.degrees(np.abs(np.angle(a * np.conj(b))))
            print(f"   {nm}: coherence {coh:.2f}; median |dphase| {np.median(dphi):.0f} deg; rms amplitude even {np.sqrt(np.mean(np.abs(a) ** 2)):.3f} odd {np.sqrt(np.mean(np.abs(b) ** 2)):.3f} m/s", flush=True)


def main(path, min_depth=100.0, stride=2, spacing_km=80.0, nuis=4, nsp_km=None, d0=None, d1=None):
    TD.CONSTITUENT_DEG_PER_HOUR.update(CONTROLS)
    cube = xr.open_dataset(path)
    if d0 is not None:
        cube = cube.isel(time=slice(int(d0 * 24), int(d1 * 24)))
    depth = np.asarray(cube['depth'].values, float)
    print(f"cube {dict(cube.sizes)}, {np.isfinite(depth).mean():.2f} of the pixels kept; spacing {spacing_km} km, nuisance harmonics {nuis}, nuisance spacing {nsp_km} km", flush=True)
    res = {}
    for label, names in (('tidal', TIDAL), ('control', tuple(CONTROLS))):
        t0 = time.time()
        cfg = RG.TideConfig(constituents=names, min_depth=min_depth, time_stride=stride, hourly_detrend=os.environ.get('DETREND') or None, spacing=spacing_km * 1e3, nuisance_harmonics=nuis, nuisance_spacing=None if nsp_km is None else nsp_km * 1e3)
        m = RG.fit_region(cube, depth, cfg, log=None)
        res[label] = amp_table(m, cube)
        print(f"[{label}] fit {time.time() - t0:.0f}s, residual rms {m.fit.resid_rms:.2e} K/s, rows {m.fit.n_rows}", flush=True)
        for nm, d in res[label].items():
            print(f"   {nm}: median amplitude u {d['u_median']:.3f} v {d['v_median']:.3f}; rms over the supported cells {d['rms']:.3f} m/s", flush=True)
    t = np.mean([d['rms'] for d in res['tidal'].values()]); c = np.mean([d['rms'] for d in res['control'].values()])
    print(f"mean rms amplitude: tidal constituents {t:.3f} m/s, controls {c:.3f} m/s (ratio {t / c:.2f})")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == 'split':
        split_test(a[1], float(a[2]) if len(a) > 2 else 100.0, int(a[3]) if len(a) > 3 else 2, float(a[4]) if len(a) > 4 else 80.0, int(a[5]) if len(a) > 5 else 4)
        sys.exit(0)
    main(a[0], float(a[1]) if len(a) > 1 else 100.0, int(a[2]) if len(a) > 2 else 2, float(a[3]) if len(a) > 3 else 80.0, int(a[4]) if len(a) > 4 else 4,
         float(a[5]) if len(a) > 5 else None, float(a[6]) if len(a) > 6 else None, float(a[7]) if len(a) > 7 else None)

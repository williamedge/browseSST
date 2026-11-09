"""
Experiment 38: does a day-by-day diurnal-heating nuisance lift the real-data tide estimate above its noise floor?

Exp 34 found that the SST-regression tide on the real 3 deg x 3 deg month (hpc/make_test_dataset.py output) is not distinguishable from the same fit at fictitious control frequencies (mean rms amplitude
ratio tidal / control 1.1-1.2 in every variant). Exp 37 found that the heating-like part of dT/dt is not a fixed cycle (day-by-day diurnal amplitudes beat one fixed cycle by +174 log-lik), and the
user confirmed that the day-to-day heat flux is inconsistent. Here the nuisance of the tide regression is made day-by-day (`TideConfig.nuisance_t_spacing`): each diurnal harmonic's amplitude
field is piecewise linear in time with one knot per day (a fixed cycle is a special case).
Variants (all: M2 S2 K1 O1, min depth 100 m, 80 km splines, 20 km pooled blocks, time stride 2):
  base        fixed cycle, 4 harmonics (the exp 34 default)
  daily       day-by-day amplitudes, 4 harmonics
  daily_h2    day-by-day amplitudes, 2 harmonics (diurnal + semidiurnal only)
  daily_mean  daily + the per-hour spatial mean of dT/dt removed first (hourly_detrend='mean')
  mean        base + hourly_detrend='mean'  (exp 34 reference)
  daily_sm    daily + a second-difference penalty on the day-to-day amplitudes (nuisance_t_smooth = 30 x spatial smooth)
Run:  python experiments/38_real_tide_daily_nuisance.py fit <cube> <variant> [split]   (writes ~/browsesst_data/real_test/exp38_<variant>*.json / .pkl)
      python experiments/38_real_tide_daily_nuisance.py compare
Each fit runs the tidal constituents and the four control frequencies (18.0, 22.0, 26.5, 32.0 deg/h, exp 34) and with `split` also the even / odd day split-half coherence of the tidal fit.
"""
import json, os, sys, time, warnings
import numpy as np
import xarray as xr

warnings.filterwarnings("ignore")
from inversion_sst_gp.spacetime import regional as RG, tides as TD

DAY = 86400.0
CONTROLS = {'X1': 18.0, 'X2': 22.0, 'X3': 26.5, 'X4': 32.0}
TIDAL = ('M2', 'S2', 'K1', 'O1')
OUT = os.path.expanduser('~/browsesst_data/real_test')
TAG = os.environ.get('EXP38_TAG', 'exp38')                      # output prefix (a different cube gets a different tag)
VARIANTS = {
    'base': dict(),
    'daily': dict(nuisance_t_spacing=DAY),
    'daily_h2': dict(nuisance_t_spacing=DAY, nuisance_harmonics=2),
    'daily_mean': dict(nuisance_t_spacing=DAY, hourly_detrend='mean'),
    'mean': dict(hourly_detrend='mean'),
    'daily_sm': dict(nuisance_t_spacing=DAY, nuisance_t_smooth=3e-2),
    'ms_daily': dict(_names=('M2', 'S2'), nuisance_t_spacing=DAY),                           # M2 and S2 only (K1 / O1 dropped), 80 km splines
    'ms_daily150': dict(_names=('M2', 'S2'), nuisance_t_spacing=DAY, spacing=150e3),         # the same with 150 km splines
    'ms_daily150_d180': dict(_names=('M2', 'S2'), nuisance_t_spacing=DAY, spacing=150e3, min_depth=180.0),   # 150 km splines, blocks shallower than 180 m dropped (instead of 100 m)
}


def config(names, variant, **extra):
    kw = dict(constituents=names, min_depth=100.0, time_stride=2, spacing=80e3, nuisance_harmonics=4)
    kw.update({k: v for k, v in VARIANTS[variant].items() if not k.startswith('_')}); kw.update(extra)
    return RG.TideConfig(**kw)


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


def split_half(cube, depth, variant):
    nt = cube.sizes['time']
    day = np.arange(nt) // 24
    lon, lat = cube['lon'].values, cube['lat'].values
    res = {}
    for label, names in (('tidal', VARIANTS[variant].get('_names', TIDAL)), ('control', tuple(CONTROLS))):
        fields = [RG.fit_region(cube, depth, config(names, variant, keep_hours=(day % 2 == par)), log=None).constituents(lon[2::5], lat[2::5]) for par in (0, 1)]
        sup = fields[0]['supported'].values.astype(bool) & fields[1]['supported'].values.astype(bool)
        res[label] = {}
        for nm in names:
            z = [np.r_[ds[f'{nm}_Au'].values[sup] * np.exp(-1j * np.radians(ds[f'{nm}_phi_u'].values[sup])), ds[f'{nm}_Av'].values[sup] * np.exp(-1j * np.radians(ds[f'{nm}_phi_v'].values[sup]))] for ds in fields]
            a, b = z
            res[label][nm] = dict(coherence=float(np.abs(np.vdot(a, b)) / np.sqrt(np.vdot(a, a).real * np.vdot(b, b).real)), dphi_median=float(np.median(np.degrees(np.abs(np.angle(a * np.conj(b)))))))
    return res


def fit(path, variant, do_split=False):
    TD.CONSTITUENT_DEG_PER_HOUR.update(CONTROLS)
    cube = xr.open_dataset(path)
    if os.environ.get('EXP38_DAYS'):                                  # 'd0,d1': keep days d0 <= day < d1 of the cube (e.g. the in-situ overlap window)
        d0, d1 = (int(v) for v in os.environ['EXP38_DAYS'].split(','))
        cube = cube.isel(time=slice(24 * d0, 24 * d1))
    depth = np.asarray(cube['depth'].values, float)
    os.makedirs(OUT, exist_ok=True)
    TIDAL_ = VARIANTS[variant].get('_names', TIDAL)
    res = dict(variant=variant, config={k: v for k, v in VARIANTS[variant].items()})
    for label, names in (('tidal', TIDAL_), ('control', tuple(CONTROLS))):
        t0 = time.time()
        m = RG.fit_region(cube, depth, config(names, variant), log=None)
        res[label] = amp_table(m, cube)
        res[label + '_meta'] = dict(resid_rms=float(m.fit.resid_rms), n_params=int(m.fit.info['n_params']), seconds=time.time() - t0)
        if label == 'tidal':
            m.save(os.path.join(OUT, f'{TAG}_{variant}_tide.pkl'))
        print(f"[{variant}/{label}] fit {time.time() - t0:.0f}s, residual rms {m.fit.resid_rms:.3e} K/s, {m.fit.info['n_params']} parameters, "
              + ", ".join(f"{nm} {d['rms']:.3f}" for nm, d in res[label].items()), flush=True)
    res['ratio'] = float(np.mean([d['rms'] for d in res['tidal'].values()]) / np.mean([d['rms'] for d in res['control'].values()]))
    print(f"[{variant}] mean rms amplitude tidal {np.mean([d['rms'] for d in res['tidal'].values()]):.3f}, control {np.mean([d['rms'] for d in res['control'].values()]):.3f} m/s, ratio {res['ratio']:.2f}", flush=True)
    if do_split:
        res['split'] = split_half(cube, depth, variant)
        for label, d in res['split'].items():
            print(f"[{variant}/split/{label}] " + ", ".join(f"{nm} coh {v['coherence']:.2f} dphi {v['dphi_median']:.0f}" for nm, v in d.items()), flush=True)
    json.dump(res, open(os.path.join(OUT, f'{TAG}_{variant}.json'), 'w'), indent=1)


def compare():
    print(f"{'variant':<11}{'resid K/s':>10}{'tidal':>7}{'control':>8}{'ratio':>7}   split-half coherence tidal (M2 S2 K1 O1) | control (mean)")
    for v in VARIANTS:
        p = os.path.join(OUT, f'{TAG}_{v}.json')
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        t = np.mean([d['rms'] for d in r['tidal'].values()]); c = np.mean([d['rms'] for d in r['control'].values()])
        sp = ''
        if 'split' in r:
            sp = ' '.join(f"{x['coherence']:.2f}" for x in r['split']['tidal'].values()) + ' | ' + f"{np.mean([x['coherence'] for x in r['split']['control'].values()]):.2f}"
        print(f"{v:<11}{r['tidal_meta']['resid_rms']:>10.2e}{t:>7.3f}{c:>8.3f}{r['ratio']:>7.2f}   {sp}")


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == 'fit':
        fit(a[1], a[2], len(a) > 3 and a[3] == 'split')
    else:
        compare()

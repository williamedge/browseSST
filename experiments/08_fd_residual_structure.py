"""
Experiment 08: what is the finite-difference residual made of?

Residual r = dTdt_FD2 - (S - u dTdx_FD2 - v dTdy_FD2) on the exp-04 dataset (14x14 px @ 4 km, 48 h). Splits it with
4th-order reference stencils into the time-truncation part, the space-truncation part (times u, v) and what is left, and
checks (i) whether its size scales with |grad T| and (ii) its space/time correlation, to decide how to model it.
Run: python experiments/08_fd_residual_structure.py [dt_sim_s [dx_m [nx]]]
"""
import warnings
import numpy as np
from inversion_sst_gp.spacetime import synthetic as S

warnings.filterwarnings("ignore")
H = 3600.0


def d4_axis(T, h, axis):
    """4th-order centred first derivative along `axis`; NaN within 2 cells of the border."""
    out = np.full_like(T, np.nan)
    sl = lambda a, b: tuple(slice(a, b) if ax == axis else slice(None) for ax in range(T.ndim))
    n = T.shape[axis]
    out[sl(2, n - 2)] = (-T[sl(4, n)] + 8 * T[sl(3, n - 1)] - 8 * T[sl(1, n - 3)] + T[sl(0, n - 4)]) / (12 * h)
    return out


def acf(r, axis, lag):
    a = np.moveaxis(r, axis, 0)
    x, y = a[:-lag].ravel(), a[lag:].ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[ok], y[ok])[0, 1])


def main(dt_sim=1800.0, dx=4000.0, nx=14):
    cfg = S.SyntheticConfig(nx=nx, ny=nx, dx=dx, n_hours=48, n_modes=400, dt_sim=dt_sim)
    ds = S.generate(cfg)
    T = ds['T_true'].values; dx = cfg.dx
    u, v, Sg = ds['u_true'].values, ds['v_true'].values, ds['S_true'].values
    dTdt2, dTdx2, dTdy2 = ds['dTdt_clean'].values, ds['dTdx_clean'].values, ds['dTdy_clean'].values
    dTdt4, dTdx4, dTdy4 = d4_axis(T, H, 0), d4_axis(T, dx, 2), d4_axis(T, dx, 1)

    rhs2 = Sg - u * dTdx2 - v * dTdy2
    r_total = dTdt2 - rhs2
    r_time = dTdt2 - dTdt4
    r_space = -(u * (dTdx2 - dTdx4) + v * (dTdy2 - dTdy4))
    r_ref = dTdt4 - (Sg - u * dTdx4 - v * dTdy4)           # what remains with 4th-order stencils everywhere
    ok = np.isfinite(r_total) & np.isfinite(r_ref)
    rms = lambda a: float(np.sqrt(np.mean(a[ok] ** 2)))
    print(f"rms dTdt {rms(dTdt2):.3e} K/s   (noise from sigma_T=0.01 K: {cfg.sigma_T / (np.sqrt(2) * H):.3e})")
    print(f"residual rms: total {rms(r_total):.3e} | time truncation {rms(r_time):.3e} | space truncation (x u,v) {rms(r_space):.3e}"
          f" | remaining with 4th-order stencils {rms(r_ref):.3e}")
    print(f"  (time + space parts explain: corr(total, time+space) = {np.corrcoef(r_total[ok], (r_time + r_space)[ok])[0, 1]:+.3f})")
    print(f"  fraction of total variance: time {np.var(r_time[ok]) / np.var(r_total[ok]):.2f}, space {np.var(r_space[ok]) / np.var(r_total[ok]):.2f}")

    g = np.hypot(dTdx2, dTdy2)
    ok2 = ok & np.isfinite(g)
    qs = np.quantile(g[ok2], [0, .25, .5, .75, 1.0])
    print("\nresidual vs |grad T| (does its size scale with the gradient?)")
    print("  |gradT| bin (K/m)         mean|g|     rms(res)     rms(res)/mean|g|   [rms(res)/(mean|g|*|u|) in m/s -> equivalent velocity error]")
    speed = np.hypot(u, v)
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = ok2 & (g >= lo) & (g <= hi)
        mg = g[m].mean()
        print(f"  {lo:.2e}..{hi:.2e}  {mg:.2e}  {np.sqrt(np.mean(r_total[m]**2)):.2e}   {np.sqrt(np.mean(r_total[m]**2)) / mg:.2e}     (|u| rms {np.sqrt(np.mean(speed[m]**2)):.3f})")
    print(f"  corr(|res|, |gradT|) = {np.corrcoef(np.abs(r_total[ok2]), g[ok2])[0, 1]:+.2f}")

    print("\nautocorrelation of the residual: lag 1/2/4 steps")
    print(f"  time (hours): " + "  ".join(f"{acf(r_total, 0, l):+.2f}" for l in (1, 2, 4)))
    print(f"  x     (4 km): " + "  ".join(f"{acf(r_total, 2, l):+.2f}" for l in (1, 2, 4)))
    print(f"  y     (4 km): " + "  ".join(f"{acf(r_total, 1, l):+.2f}" for l in (1, 2, 4)))
    # compare to the white-noise level a real pipeline would add: iid noise would have lag-1 autocorrelation ~ 0 in space,
    # and ~+0.0 / -0.5 pattern in time for centred differences of independent snapshots (dTdt(t), dTdt(t+2) share a snapshot).
    print("\n(for reference, dTdt noise from independent T noise has time autocorrelation ~0 at lag 1 and -0.5 at lag 2)")


if __name__ == "__main__":
    import sys
    a = [float(v) for v in sys.argv[1:]]      # [dt_sim [dx [nx]]]
    main(*([a[0]] if a else []), *([a[1]] if len(a) > 1 else []), *([int(a[2])] if len(a) > 2 else []))

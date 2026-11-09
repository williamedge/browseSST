"""
Experiment 09: how accurate is the semi-Lagrangian tracer simulator, independent of finite differencing?

Uniform flow U = (0.30, 0.10) m/s, S = 0 has the exact solution T(x, y, t) = T0(x - U t, y - V t) with the same
analytic initial field T0 the synthetic generator uses. Compares the simulated T with it, then computes the
transport-equation residual of the hourly FD products for BOTH the simulated and the exact T. If the two residuals
differ, simulator numerics (not differencing) are contributing to the 'finite-difference floor' seen in exp 04-08.
Run: python experiments/09_simulator_accuracy.py
"""
import warnings
import numpy as np
from inversion_sst_gp.spacetime import synthetic as S

warnings.filterwarnings("ignore")
U0, V0 = 0.30, 0.10


def main(dx=4000.0, nx=14, n_hours=48, dt_sim=1800.0, n_modes=400):
    # patch the analytic fields: constant flow, no source
    S._Fields.uv = lambda self, x, y, t: (np.full_like(x, U0), np.full_like(x, V0))
    S._Fields.S = lambda self, x, y, t: np.zeros_like(x)
    cfg = S.SyntheticConfig(nx=nx, ny=nx, dx=dx, n_hours=n_hours, dt_sim=dt_sim, n_modes=n_modes, sigma_T=0.0)
    ds = S.generate(cfg)
    f = S._Fields(cfg)
    X, Y = np.meshgrid(ds['x'].values, ds['y'].values)
    t = ds['time'].values
    T_exact = np.stack([f.T0((X - U0 * tk).ravel(), (Y - V0 * tk).ravel()).reshape(X.shape) for tk in t])
    T_sim = ds['T_true'].values
    err = T_sim - T_exact
    print(f"dx={dx/1e3:.0f} km, dt_sim={dt_sim:.0f}s: T std {T_exact.std():.3f} K; simulator error rms at t=1h/12h/24h/48h: " +
          " / ".join(f"{np.sqrt(np.mean(err[k]**2)):.2e}" for k in (1, 12, 24, n_hours - 1)) + " K")

    for label, T in (("exact solution", T_exact), ("simulated T", T_sim)):
        dTdt, dTdx, dTdy = S.derivative_products(T, dx)
        res = dTdt + U0 * dTdx + V0 * dTdy           # S = 0
        ok = np.isfinite(res)
        print(f"  FD-product residual of the transport equation, {label:<15s}: rms {np.sqrt(np.mean(res[ok]**2)):.3e} K/s "
              f"= {np.sqrt(np.mean(res[ok]**2)) / np.sqrt(np.mean(dTdt[ok]**2)):.3f} of rms dTdt")


if __name__ == "__main__":
    for dx, nx in ((4000.0, 14), (2000.0, 28)):
        main(dx=dx, nx=nx)

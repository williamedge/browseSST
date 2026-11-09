"""
Synthetic test bed: a tracer T obeying dT/dt + u dT/dx + v dT/dy = S, advected by
analytic ocean_syndata Helmholtz velocity fields, with the derivative products the
real Himawari pipeline would produce (centred differences of hourly snapshots).

Truth is drawn from the same prior family the inference uses (Helmholtz rotational +
divergent (u, v); Matern space-time S), so any misfit is the code/approximation,
not model mismatch. Units are metres, seconds, Kelvin.

The tracer is integrated with a semi-Lagrangian scheme (midpoint departure points,
cubic interpolation) on a grid padded far enough that the departure points of the
core region never leave it, so no boundary inflow is needed. Velocities and the
source term are analytic (random plane-wave sums), evaluated at any (x, y, t).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr
from scipy.ndimage import map_coordinates

from ocean_syndata.fields.helmholtz import (
    DivergentWaveField, DivergentWaveParams, RotationalMaternField, RotationalMaternParams,
)

DAY = 86400.0


def _sigma_from_vel(vel_std, ell):
    """Potential std giving velocity std `vel_std` (Var(u) = 5 sigma^2 / (3 ell^2))."""
    return vel_std * ell * np.sqrt(3.0 / 5.0)


@dataclass
class SyntheticConfig:
    nx: int = 16                    # core grid cells (x)
    ny: int = 16                    # core grid cells (y)
    dx: float = 4000.0              # grid spacing, m (4 km: keeps the hourly Courant number < 1)
    n_hours: int = 36               # number of hourly snapshots
    dt_sim: float = 1800.0          # internal advection step, s
    n_modes: int = 600
    seed: int = 1
    # (u, v): rotational (psi) and divergent (phi) potentials
    vel_psi: float = 0.09           # rms velocity of the rotational part, m/s
    ell_psi: float = 30e3
    tau_psi: float = 3 * DAY        # Matern-3/2 in time
    vel_phi: float = 0.03
    ell_phi: float = 15e3
    tau_phi: float = 0.5 * DAY      # exp(-|tau|/tau): Matern-1/2 in time
    # source term S: independent Matern-5/2 x Matern-3/2 field plus a uniform diurnal cycle
    sigma_S: float = 3e-7           # K/s
    ell_S: float = 20e3
    tau_S: float = 1 * DAY
    diurnal_amp: float = 8e-7       # K/s peak
    diurnal_shape: str = 'halfsine'  # 'halfsine': max(0, sin) (heating by day, flat at night); 'cosine': legacy
    diurnal_sunrise: float = 6 * 3600.0  # s, start of the positive half-sine ('halfsine')
    diurnal_phase: float = 0.0      # rad, only for 'cosine'
    # initial tracer: two Matern-5/2 scales, K
    sigma_T0: tuple = (0.6, 0.25)
    ell_T0: tuple = (60e3, 20e3)
    # observation noise on T (the existing package's default sigma_tau is 1e-2 K)
    sigma_T: float = 0.01
    pad_safety: float = 1.3
    # optional sharp structure added to the initial tracer (all zero = the smooth Matern test bed of experiments 01-20):
    # two meandering fronts (tanh profiles, E-W and N-S, the second weaker, opposite sign) and one diagonal filament (Gaussian ridge)
    front_amp: float = 0.0          # K, half-jump of front A (jump across the front = 2 * front_amp)
    front_width: float = 5000.0     # m, tanh half-width
    filament_amp: float = 0.0       # K, peak of the filament
    filament_width: float = 4000.0  # m, Gaussian sigma across the filament
    # optional barotropic tide (off by default; see TideField): M2, S2, K1, O1 with amplitude and phase varying smoothly with a synthetic depth field,
    # plus a rough short-scale component on the shallow shelf (depth < ~200 m) that a smooth regional fit cannot represent
    tide: bool = False
    tide_scale: float = 1.0         # multiplies all tidal amplitudes
    tide_shelf_rough: float = 0.5   # amplitude of the rough shelf component relative to the smooth tide there (0 = none)
    pad_cells: int = -1             # >= 0: override the worst-case-drift padding (needed for long runs, where it would be absurd)
    n_threads: int = 1              # worker processes for the plane-wave sums (compute-bound); results are identical
    products: bool = True           # False: skip the finite-difference products (long runs; compute them from T_obs)

    @property
    def T_total(self):
        return (self.n_hours - 1) * 3600.0

    def true_params(self, zscale=1.0):
        """Kernel hyperparameters in `kernels.pack_params` order, S in units of `zscale`."""
        return dict(
            sigma_psi=_sigma_from_vel(self.vel_psi, self.ell_psi), l_psi=self.ell_psi,
            sigma_phi=_sigma_from_vel(self.vel_phi, self.ell_phi), l_phi=self.ell_phi,
            tau0_psi=self.tau_psi, tau0_phi=self.tau_phi,
            sigma_S=self.sigma_S / zscale, l_S=self.ell_S, tau0_S=self.tau_S,
        )


def front_field(x, y, c: SyntheticConfig):
    """Sharp structure added to the initial tracer: meandering fronts and a diagonal filament (zero unless enabled in `c`).

    Defined on the whole plane (the integration grid is padded), in metres; the core domain starts at (0, 0).
    """
    out = np.zeros_like(x, dtype=float)
    if c.front_amp:
        k = 2 * np.pi / 90e3                       # front A: E-W, y = 18 km + 7 km sin(kx + 0.6)
        yc = 18e3 + 7e3 * np.sin(k * x + 0.6)
        slope = 7e3 * k * np.cos(k * x + 0.6)
        out += c.front_amp * np.tanh((y - yc) / np.sqrt(1 + slope ** 2) / c.front_width)
        k = 2 * np.pi / 70e3                       # front B: N-S, x = 38 km + 6 km sin(ky + 1.9), weaker, opposite sign
        xc = 38e3 + 6e3 * np.sin(k * y + 1.9)
        slope = 6e3 * k * np.cos(k * y + 1.9)
        out += -0.7 * c.front_amp * np.tanh((x - xc) / np.sqrt(1 + slope ** 2) / (1.2 * c.front_width))
    if c.filament_amp:
        th = np.deg2rad(60.0)                      # diagonal ridge through (8 km, 6 km), gentle meander
        along = (x - 8e3) * np.cos(th) + (y - 6e3) * np.sin(th)
        across = -(x - 8e3) * np.sin(th) + (y - 6e3) * np.cos(th) + 4e3 * np.sin(2 * np.pi * along / 60e3)
        out += c.filament_amp * np.exp(-0.5 * (across / c.filament_width) ** 2)
    return out


G_ACC = 9.81
# name: (omega rad/s, amplitude at 200 m depth m/s, propagation direction deg from +x, v/u amplitude ratio, v phase lag rad, phase offset rad)
TIDES = {
    'M2': (1.405189e-4, 0.30, 70.0, 0.60, -np.pi / 2, 0.3),
    'S2': (1.454441e-4, 0.12, 70.0, 0.60, -np.pi / 2, 0.9),
    'K1': (7.292117e-5, 0.10, 20.0, 0.80, np.pi / 3, 2.0),
    'O1': (6.759774e-5, 0.07, 20.0, 0.80, np.pi / 3, 4.1),
}


def synthetic_depth(x, y):
    """Smooth synthetic bathymetry (m): a meandering shelf break (shelf ~60 m in the south, ~3000 m in the north), depth 200 m about 37 km south of it."""
    s = y - (110e3 + 15e3 * np.sin(2 * np.pi * np.asarray(x) / 180e3))
    return 60.0 + 2940.0 * 0.5 * (1.0 + np.tanh(s / 25e3)), s


class TideField:
    """Analytic barotropic-like tide u_k = A(x, y) cos(w t - Phi(x, y)), v_k = r A cos(w t - Phi + lag), summed over M2, S2, K1, O1.

    A = A200 * sqrt(200 / depth) (flux conservation), Phi = phi0 + w [cos(th) * int ds / c(s) + sin(th) x / c_mean] with c = sqrt(g depth) (a progressive wave
    crossing the shelf break), so amplitude and phase vary slowly and with depth. Not dynamically consistent (no continuity / Coriolis); a test bed for the regression.
    Plus a rough component on the shelf, weight 1 / (1 + (depth / 150)^4), with a 40 km wavelength, amplitude `tide_shelf_rough` times the local smooth amplitude.
    """

    def __init__(self, c: SyntheticConfig):
        self.c = c
        s = np.linspace(-1500e3, 1500e3, 6001)
        h = 60.0 + 2940.0 * 0.5 * (1.0 + np.tanh(s / 25e3))
        self.s_grid = s
        self.I = np.concatenate([[0.0], np.cumsum(0.5 * (1 / np.sqrt(G_ACC * h[1:]) + 1 / np.sqrt(G_ACC * h[:-1])) * np.diff(s))])
        self.c_mean = np.sqrt(G_ACC * 1000.0)

    def smooth_parts(self, x, y):
        """(depth, {name: (A, Phi)}) at positions."""
        h, s = synthetic_depth(x, y)
        I = np.interp(s, self.s_grid, self.I)
        out = {}
        for name, (w, a200, th, r, lag, ph0) in TIDES.items():
            th = np.deg2rad(th)
            A = np.clip(self.c.tide_scale * a200 * np.sqrt(200.0 / h), 0.0, 2.0)
            out[name] = (A, ph0 + w * (np.cos(th) * I + np.sin(th) * x / self.c_mean))
        return h, out

    def uv(self, x, y, t, smooth_only=False):
        x, y, t = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float), np.asarray(t, float))
        h, parts = self.smooth_parts(x, y)
        u = np.zeros_like(x); v = np.zeros_like(x)
        wsh = 1.0 / (1.0 + (h / 150.0) ** 4)
        psi = 2 * np.pi * (x * np.cos(np.deg2rad(30.0)) + y * np.sin(np.deg2rad(30.0))) / 40e3
        for name, (w, a200, th, r, lag, ph0) in TIDES.items():
            A, Phi = parts[name]
            u += A * np.cos(w * t - Phi)
            v += r * A * np.cos(w * t - Phi + lag)
            if self.c.tide_shelf_rough and not smooth_only:
                Ar = self.c.tide_shelf_rough * A * wsh
                u += Ar * np.cos(w * t - Phi - psi + 1.0)
                v += 0.5 * Ar * np.cos(w * t - Phi - psi + 1.0 + lag)
        return u, v


class _Fields:
    """Analytic u, v, S, T0 for one seed."""

    def __init__(self, c: SyntheticConfig):
        self.c = c
        rng = np.random.default_rng(c.seed)
        seeds = rng.integers(0, 2**31 - 1, size=8)
        self.rot = RotationalMaternField(RotationalMaternParams(
            sigma=_sigma_from_vel(c.vel_psi, c.ell_psi), ell=c.ell_psi, tau0=c.tau_psi,
            n_modes=c.n_modes, seed=int(seeds[0]), nu_t=1.5))
        self.div = DivergentWaveField(DivergentWaveParams(
            sigma=_sigma_from_vel(c.vel_phi, c.ell_phi), ell=c.ell_phi, tau_d=c.tau_phi,
            omega0=0.0, n_modes=c.n_modes, seed=int(seeds[1]), gamma=1.0))
        # scalar Matern fields reuse the rotational mode sum's `.potential()`
        self.S_field = RotationalMaternField(RotationalMaternParams(
            sigma=c.sigma_S, ell=c.ell_S, tau0=c.tau_S, n_modes=c.n_modes,
            seed=int(seeds[2]), nu_t=1.5))
        self.T0_fields = [RotationalMaternField(RotationalMaternParams(
            sigma=s, ell=l, tau0=1e12, n_modes=c.n_modes, seed=int(seeds[3 + i]), nu_t=1.5))
            for i, (s, l) in enumerate(zip(c.sigma_T0, c.ell_T0))]
        self.tide = TideField(c) if c.tide else None

    def uv(self, x, y, t):
        _, ur, vr = self.rot.sample(x, y, t)
        _, ud, vd = self.div.sample(x, y, t)
        if self.tide is not None:
            ut, vt = self.tide.uv(x, y, t)
            return ur + ud + ut, vr + vd + vt
        return ur + ud, vr + vd

    def S(self, x, y, t):
        return self.S_field.potential(x, y, t) + self.diurnal(t)

    def diurnal(self, t):
        c = self.c
        if c.diurnal_shape == 'halfsine':
            return c.diurnal_amp * np.maximum(0.0, np.sin(2 * np.pi * (t - c.diurnal_sunrise) / DAY))
        return c.diurnal_amp * np.cos(2 * np.pi * t / DAY - c.diurnal_phase)

    def T0(self, x, y):
        t0 = np.zeros_like(x)
        return sum(f.potential(x, y, t0) for f in self.T0_fields) + front_field(x, y, self.c)


def _estimate_max_speed(f: _Fields, c: SyntheticConfig):
    rng = np.random.default_rng(c.seed + 1000)
    n = 4000
    ext = 400e3
    x, y = rng.uniform(-ext, ext, n), rng.uniform(-ext, ext, n)
    vmax = 0.0
    for t in np.linspace(0, c.T_total, 5):
        u, v = f.uv(x, y, np.full(n, t))
        vmax = max(vmax, np.hypot(u, v).max())
    return vmax


_WORKER_F = None


def _init_worker(f):
    global _WORKER_F
    _WORKER_F = f


def _work(name, args):
    return getattr(_WORKER_F, name)(*args)


class _Par:
    """Pointwise calls of `_Fields.uv` / `.S` split over forked worker processes (the plane-wave sums are compute-bound; results are identical to the serial call)."""

    def __init__(self, f, n):
        self.f, self.n = f, n
        self.ex = None
        if n > 1:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor
            self.ex = ProcessPoolExecutor(n, mp_context=multiprocessing.get_context('fork'), initializer=_init_worker, initargs=(f,))

    def call(self, name, *arrs):
        if self.ex is None:
            return getattr(self.f, name)(*arrs)
        edges = np.linspace(0, arrs[0].size, self.n + 1).astype(int)
        parts = list(self.ex.map(_work, [name] * self.n, [tuple(a[i:j] for a in arrs) for i, j in zip(edges[:-1], edges[1:])]))
        if isinstance(parts[0], tuple):
            return tuple(np.concatenate([q[i] for q in parts]) for i in range(len(parts[0])))
        return np.concatenate(parts)

    def close(self):
        if self.ex is not None:
            self.ex.shutdown()


def generate(c: SyntheticConfig = SyntheticConfig()) -> xr.Dataset:
    """Simulate the tracer and return truth + noisy observations + derivative products."""
    f = _Fields(c)
    vmax = _estimate_max_speed(f, c)
    pad = c.pad_cells if c.pad_cells >= 0 else int(np.ceil(c.pad_safety * vmax * c.T_total / c.dx)) + 4

    nxp, nyp = c.nx + 2 * pad, c.ny + 2 * pad
    xs = (np.arange(nxp) - pad) * c.dx
    ys = (np.arange(nyp) - pad) * c.dx
    XX, YY = np.meshgrid(xs, ys)            # (nyp, nxp)
    xf, yf = XX.ravel(), YY.ravel()
    core = (slice(pad, pad + c.ny), slice(pad, pad + c.nx))

    def to_idx(x, y):
        return [(y / c.dx + pad).reshape(nyp, nxp), (x / c.dx + pad).reshape(nyp, nxp)]

    T = f.T0(xf, yf).reshape(nyp, nxp)
    n_steps_per_hour = int(round(3600.0 / c.dt_sim))
    assert abs(n_steps_per_hour * c.dt_sim - 3600.0) < 1e-9, "dt_sim must divide 1 hour"

    Ts = np.zeros((c.n_hours, c.ny, c.nx))
    Us = np.zeros_like(Ts); Vs = np.zeros_like(Ts); Ss = np.zeros_like(Ts)
    Ds = np.zeros(c.n_hours)
    Ut = np.zeros_like(Ts); Vt = np.zeros_like(Ts)        # tide only (zeros when no tide)
    t = 0.0

    def record(k, t):
        Ts[k] = T[core]
        xc, yc = XX[core].ravel(), YY[core].ravel()
        u, v = f.uv(xc, yc, np.full(xc.size, t))
        Us[k], Vs[k] = u.reshape(c.ny, c.nx), v.reshape(c.ny, c.nx)
        Ss[k] = f.S(xc, yc, np.full(xc.size, t)).reshape(c.ny, c.nx)
        Ds[k] = f.diurnal(t)
        if f.tide is not None:
            ut, vt = f.tide.uv(xc, yc, np.full(xc.size, t))
            Ut[k], Vt[k] = ut.reshape(c.ny, c.nx), vt.reshape(c.ny, c.nx)

    par = _Par(f, c.n_threads)
    record(0, t)
    for k in range(1, c.n_hours):
        for _ in range(n_steps_per_hour):
            tm = t + 0.5 * c.dt_sim
            tfull = np.full(xf.size, tm)
            um, vm = par.call('uv', xf, yf, tfull)
            xstar, ystar = xf - 0.5 * c.dt_sim * um, yf - 0.5 * c.dt_sim * vm
            us, vs = par.call('uv', xstar, ystar, tfull)
            xd, yd = xf - c.dt_sim * us, yf - c.dt_sim * vs
            Td = map_coordinates(T, to_idx(xd, yd), order=3, mode='nearest')
            # source integrated along the trajectory (midpoint rule)
            Sm = par.call('S', 0.5 * (xf + xd), 0.5 * (yf + yd), tfull)
            T = Td + c.dt_sim * Sm.reshape(nyp, nxp)
            t += c.dt_sim
        record(k, t)

    par.close()
    x = xs[pad:pad + c.nx]; y = ys[pad:pad + c.ny]
    time = np.arange(c.n_hours) * 3600.0
    rng = np.random.default_rng(c.seed + 7)
    T_obs = Ts + c.sigma_T * rng.standard_normal(Ts.shape)

    ds = xr.Dataset(
        coords=dict(time=time, y=y, x=x),
        attrs=dict(pad_cells=pad, vmax=vmax, dx=c.dx, sigma_T=c.sigma_T),
    )
    dims = ('time', 'y', 'x')
    ds['T_true'] = (dims, Ts); ds['T_obs'] = (dims, T_obs)
    ds['u_true'] = (dims, Us); ds['v_true'] = (dims, Vs); ds['S_true'] = (dims, Ss)
    ds['S_diurnal_true'] = (('time',), Ds)
    if c.tide:
        ds['u_tide_true'] = (dims, Ut); ds['v_tide_true'] = (dims, Vt)
        ds['depth'] = (('y', 'x'), synthetic_depth(*np.meshgrid(x, y))[0])
    for suffix, Tfield in ((('', T_obs), ('_clean', Ts)) if c.products else ()):
        dTdt, dTdx, dTdy = derivative_products(Tfield, c.dx)
        ds['dTdt' + suffix] = (dims, dTdt)
        ds['dTdx' + suffix] = (dims, dTdx)
        ds['dTdy' + suffix] = (dims, dTdy)
    return ds


def load_or_generate(cfg: SyntheticConfig, path) -> xr.Dataset:
    """Cache `generate(cfg)` as NetCDF at `path` (the simulation takes minutes; *.nc is git-ignored)."""
    import pathlib
    path = pathlib.Path(path)
    if path.exists():
        return xr.load_dataset(path)
    ds = generate(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)
    return ds


def reobserve(ds: xr.Dataset, sigma_T: float, seed: int = 7) -> xr.Dataset:
    """Same simulated truth, new observation noise: redraw `T_obs` and its derivative products (no re-simulation).

    `generate` draws its noise with seed `cfg.seed + 7`; pass that seed (and the same sigma_T) to reproduce it exactly.
    """
    out = ds.copy(deep=True)
    rng = np.random.default_rng(seed)
    T_obs = ds['T_true'].values + sigma_T * rng.standard_normal(ds['T_true'].shape)
    dTdt, dTdx, dTdy = derivative_products(T_obs, float(ds.attrs['dx']))
    dims = ('time', 'y', 'x')
    out['T_obs'] = (dims, T_obs)
    out['dTdt'], out['dTdx'], out['dTdy'] = (dims, dTdt), (dims, dTdx), (dims, dTdy)
    out.attrs['sigma_T'] = sigma_T
    return out


def derivative_products(T, dx, dt=3600.0, order=2):
    """Centred differences of (time, y, x) snapshots, NaN on the border.

    order=2 mirrors the real pipeline: `dTdt = (T[t+1] - T[t-1]) / (2 dt)` and spatial central differences of the
    snapshot at t. order=4 uses the 5-point stencil (-1, 8, -8, 1) / 12h in all three directions (border of 2 cells,
    about 1.8x the noise variance of the 2nd-order time difference for independent snapshot noise).
    """
    dTdt = np.full_like(T, np.nan); dTdx = np.full_like(T, np.nan); dTdy = np.full_like(T, np.nan)
    if order == 2:
        dTdt[1:-1] = (T[2:] - T[:-2]) / (2 * dt)
        dTdx[:, :, 1:-1] = (T[:, :, 2:] - T[:, :, :-2]) / (2 * dx)
        dTdy[:, 1:-1, :] = (T[:, 2:, :] - T[:, :-2, :]) / (2 * dx)
    elif order == 4:
        dTdt[2:-2] = (-T[4:] + 8 * T[3:-1] - 8 * T[1:-3] + T[:-4]) / (12 * dt)
        dTdx[:, :, 2:-2] = (-T[:, :, 4:] + 8 * T[:, :, 3:-1] - 8 * T[:, :, 1:-3] + T[:, :, :-4]) / (12 * dx)
        dTdy[:, 2:-2, :] = (-T[:, 4:, :] + 8 * T[:, 3:-1, :] - 8 * T[:, 1:-3, :] + T[:, :-4, :]) / (12 * dx)
    else:
        raise ValueError("order must be 2 or 4")
    return dTdt, dTdx, dTdy


def equation_residual(ds: xr.Dataset) -> dict:
    """How well do the finite-difference products satisfy the transport equation?

    Compares clean `dTdt` against `S - u dTdx - v dTdy` (clean, analytic u, v, S) on the
    interior. The relative RMS is the discretisation error floor of the hourly,
    centred-difference pipeline at this grid spacing (the Courant-number issue).
    """
    lhs = ds['dTdt_clean']
    rhs = ds['S_true'] - ds['u_true'] * ds['dTdx_clean'] - ds['v_true'] * ds['dTdy_clean']
    ok = np.isfinite(lhs.values) & np.isfinite(rhs.values)
    res = (lhs.values - rhs.values)[ok]
    return dict(rms_residual=float(np.sqrt(np.mean(res**2))),
                rms_dTdt=float(np.sqrt(np.mean(lhs.values[ok]**2))),
                relative=float(np.sqrt(np.mean(res**2)) / np.sqrt(np.mean(lhs.values[ok]**2))))

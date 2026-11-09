"""Stage 0 on a real cube: the barotropic tide fitted over the whole region, then evaluated on any tile.

`fit_region(cube, depth, cfg)` takes an SST cube from `himawari.load_cube` (time, lat, lon; NaN where not observed) and a depth grid on the same lon / lat axes
(`bathymetry.load_depth`), forms 4th-order finite differences of the (noisy) SST in time and along local east / north (ellipsoid metric factors), pools the equations in
`cfg.block` x `cfg.block` pixel blocks by exact second-moment pooling (`tides.pool_rows`), drops blocks shallower than `cfg.min_depth`, and fits M2 and S2 by default (configurable; K1 / O1 dropped, see `TideConfig`)
with smoothly varying amplitude and phase (`tides.fit_tides`). The result is a `TideModel`: the fit plus what is needed to evaluate it anywhere (the regional local frame
and the time origin of the phases), picklable for the HPC tile jobs.

The time axis is processed in chunks (with a 2-step margin for the time derivative), so memory is a few chunk-sized arrays, not the whole cube's products.

Applying it: `TideModel.tile_velocity(td)` returns the tidal (u, v) at every grid point of a `tiling.TileData` block, in the tile's own time axis. Where the fit has no support (nearer
than `support_radius` to no retained block: the shelf shallower than `min_depth`) the tide is set to zero by default (`mask='supported'`) and the tile's GP absorbs whatever is there;
the boolean `supported` mask is returned so the tile product can say where the tide was removed. The step at the support edge is an open item (see notes).
"""
import dataclasses
import pickle
import time

import numpy as np
import xarray as xr

from . import tides as TD
from .geo import LocalFrame, metric_spacing
from .tiling import grid_gradients_metric

HR = 3600.0


@dataclasses.dataclass
class TideConfig:
    constituents: tuple = ("M2", "S2")      # K1 / O1 dropped for now (exp 38-39: the BB23 moorings show < 0.05 m/s there, the SST fit gives 0.25 m/s of noise); any of K1 O1 N2 K2 P1 Q1 can be added
    min_depth: float = 200.0           # m; blocks shallower than this are dropped (None keeps everything)
    block: int = 5                     # equations pooled in block x block pixels (5 x 4 km = 20 km)
    spacing: float = 80e3              # spline breakpoint spacing (m): the smoothness scale of the amplitude / phase fields
    smooth: float = 1e-3
    nuisance_harmonics: int = 4        # diurnal-heating harmonics 0..H in the nuisance term (-1 = none)
    nuisance_spacing: float = None     # m; spatial spline spacing of the nuisance term (default 3 x spacing)
    nuisance_t_spacing: float = None   # s; None = one fixed heating cycle for the whole record; 86400 = day-by-day amplitudes of the nuisance harmonics (piecewise linear, exp 38)
    nuisance_t_smooth: float = 0.0     # relative weight of the second-difference penalty on the day-by-day amplitudes (0 = free)
    min_frac: float = 0.5              # a block needs this fraction of valid pixels (per time step)
    time_stride: int = 1               # keep every k-th hour of rows (hourly rows are strongly correlated; 2-3 cuts the cost)
    chunk_hours: int = 120             # time chunk for the finite differences
    n_jobs: int = 1
    chunk_rows: int = 20000            # rows per block of the normal-equation assembly
    support_radius: float = None       # m; a point is 'supported' if a retained block lies within this distance. Default: one block width (block x pixel spacing), NOT the spline spacing
    order: int = 4
    hourly_detrend: str = None         # None | 'mean' | 'plane': subtract, hour by hour, the spatial mean (or best-fit plane) of dT/dt over the retained pixels (removes the spatially uniform
                                       # part of the residual, 34% of its variance in the real test month, together with the tide's response to the mean gradient)
    keep_hours: object = None          # optional (nt,) bool array: only the equations of these hours are used (split-half tests); the products still use every hour


@dataclasses.dataclass
class TideModel:
    fit: TD.TideFit
    lon0: float                        # regional local frame centre
    lat0: float
    t_ref: np.datetime64               # time origin of the fitted phases
    cfg: TideConfig
    info: dict = dataclasses.field(default_factory=dict)

    @property
    def frame(self):
        return LocalFrame(self.lon0, self.lat0)

    def seconds(self, time):
        return (np.asarray(time, dtype="datetime64[ns]") - np.datetime64(self.t_ref, "ns")).astype("timedelta64[ns]").astype(np.float64) / 1e9

    def velocity(self, lon, lat, time, mask="supported"):
        """Tidal (east, north) velocity (m/s) at lon / lat (broadcast with `time`, datetime64). `mask='supported'` sets NaN where the fit has no support, 'none' keeps the extrapolation."""
        x, y = self.frame.forward(np.asarray(lon, float), np.asarray(lat, float))
        t = self.seconds(time)
        return self.fit.velocity(x, y, t, mask_unsupported=(mask == "supported"))

    def supported(self, lon, lat):
        x, y = self.frame.forward(np.asarray(lon, float), np.asarray(lat, float))
        return self.fit.supported(x, y).reshape(np.shape(x))

    def tile_velocity(self, td, mask="supported"):
        """(u_bt, v_bt, supported) on all grid points of a TileData block, flat in (time, y, x) order; the tide is 0 (not NaN) where unsupported if mask == 'supported'."""
        nt, ny, nx = td.shape
        lon2, lat2 = np.meshgrid(td.lon, td.lat)
        x2, y2 = self.frame.forward(lon2, lat2)
        sup = self.fit.supported(x2.ravel(), y2.ravel()).reshape(ny, nx)
        ts = self.seconds(td.time)                                                # (nt,)
        u = np.empty((nt, ny, nx)); v = np.empty((nt, ny, nx))
        for k in range(nt):
            u[k], v[k] = self.fit.velocity(x2, y2, np.full((ny, nx), ts[k]), mask_unsupported=False)
        if mask == "supported":
            u = np.where(sup[None], u, 0.0); v = np.where(sup[None], v, 0.0)
        return u.ravel(), v.ravel(), np.broadcast_to(sup[None], (nt, ny, nx)).ravel().copy()

    def constituents(self, lon, lat):
        """Amplitude (m/s) and phase (deg; u = A cos(w t - phi), t in seconds from `t_ref`) of each constituent on a lon / lat grid, as a Dataset, with the support mask."""
        lon2, lat2 = np.meshgrid(lon, lat)
        x2, y2 = self.frame.forward(lon2, lat2)
        h = self.fit.harmonics(x2.ravel(), y2.ravel())
        sh = lon2.shape
        out = {}
        for nm, d in h.items():
            for key, unit in (("Au", "m/s"), ("Av", "m/s")):
                out[f"{nm}_{key}"] = (("lat", "lon"), d[key].reshape(sh), {"units": unit})
            for key in ("phi_u", "phi_v"):
                out[f"{nm}_{key}"] = (("lat", "lon"), np.degrees(d[key]).reshape(sh), {"units": "degrees"})
        out["supported"] = (("lat", "lon"), self.fit.supported(x2.ravel(), y2.ravel()).reshape(sh).astype(np.int8))
        ds = xr.Dataset(out, coords={"lat": np.asarray(lat), "lon": np.asarray(lon)})
        ds.attrs.update(t_ref=str(np.datetime64(self.t_ref, "s")), constituents=",".join(self.fit.names), min_depth=-1.0 if self.cfg.min_depth is None else float(self.cfg.min_depth),
                        spacing_m=float(self.cfg.spacing))
        return ds

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return pickle.load(f)


class ArrayTide:
    """An EXTERNAL barotropic tide stored in the cube file (variables `u_tide`, `v_tide`: east / north m/s on the cube's time / lat / lon grid, NaN where the product is not valid),
    for example FES2022 currents predicted on the cube grid, instead of the SST regression. Same `tile_velocity` interface as `TideModel`, so `pipeline.run_tile` takes either.
    `cube` should be opened lazily (xr.open_dataset): only the tile's block is read."""

    def __init__(self, cube, u="u_tide", v="v_tide"):
        if u not in cube or v not in cube:
            raise KeyError(f"the cube has no {u!r} / {v!r} variables")
        self.cube, self.u, self.v = cube, u, v

    def tile_velocity(self, td, mask="supported"):
        sp = td.spec
        sel = dict(time=slice(sp.k0, sp.k1), lat=slice(sp.j0, sp.j1), lon=slice(sp.i0, sp.i1))
        u = np.asarray(self.cube[self.u].isel(**sel).values, float)
        v = np.asarray(self.cube[self.v].isel(**sel).values, float)
        sup = np.isfinite(u) & np.isfinite(v)
        if mask == "supported":
            u = np.where(sup, u, 0.0); v = np.where(sup, v, 0.0)
        else:
            u = np.nan_to_num(u); v = np.nan_to_num(v)
        return u.ravel(), v.ravel(), sup.ravel()


def regional_frame(lon, lat):
    return LocalFrame(0.5 * (float(lon[0]) + float(lon[-1])), 0.5 * (float(lat[0]) + float(lat[-1])))


def fit_region(cube, depth, cfg=None, log=print):
    """Fit the tide on a cube (see the module docstring). `depth`: (ny, nx) metres on the cube's axes; NaN = masked out (pixels excluded from the equations, e.g. the depths already set to NaN
    shallower than a threshold), and `cfg.min_depth` additionally drops blocks whose mean depth is below it."""
    cfg = TideConfig() if cfg is None else cfg
    t_wall = time.time()
    sst = cube["sst"]
    nt, ny, nx = sst.shape
    lon = np.asarray(cube["lon"].values, float); lat = np.asarray(cube["lat"].values, float)
    tax = cube["time"].values
    step = float(np.median(np.diff(tax)).astype("timedelta64[s]").astype(np.float64)) if nt > 1 else HR
    frame = regional_frame(lon, lat)
    x2, y2 = frame.forward(lon[None, :], lat[:, None])
    dlon = float(np.median(np.diff(lon))); dlat = float(np.median(np.diff(lat)))
    hx, hy = metric_spacing(lat, dlon, dlat)
    depth = np.asarray(depth, float)
    if depth.shape != (ny, nx):
        raise ValueError(f"depth has shape {depth.shape}, the cube grid is {(ny, nx)}")
    if not (dlon > 0 and dlat > 0):
        raise ValueError("the cube must have ascending lon and lat axes")
    masked = ~np.isfinite(depth)                       # NaN depth = masked out (shallow, land, outside the bathymetry): those pixels never enter the tide equations
    t_ref = tax[0]
    tsec = (tax - t_ref).astype("timedelta64[s]").astype(np.float64)
    m = 2                                                                       # margin of the 4th-order time derivative
    rows = []
    n_equiv = 0
    for k0 in range(0, nt, cfg.chunk_hours):
        k1 = min(nt, k0 + cfg.chunk_hours)
        a0, a1 = max(0, k0 - m), min(nt, k1 + m)
        T = np.asarray(sst.isel(time=slice(a0, a1)).values, np.float64)
        Tt = TD.time_derivative(T, step, cfg.order)
        Tx, Ty = grid_gradients_metric(T, T.shape, hx, hy, cfg.order)
        Tx = Tx.reshape(T.shape); Ty = Ty.reshape(T.shape)
        if masked.any():
            Tt[:, masked] = np.nan; Tx[:, masked] = np.nan; Ty[:, masked] = np.nan
        if cfg.hourly_detrend:
            for k in range(Tt.shape[0]):
                g = np.isfinite(Tt[k]) & np.isfinite(Tx[k]) & np.isfinite(Ty[k])
                if g.sum() < 200:
                    continue
                if cfg.hourly_detrend == "mean":
                    Tt[k][g] -= Tt[k][g].mean()
                else:
                    Ad = np.column_stack([np.ones(g.sum()), x2[g], y2[g]])
                    c, *_ = np.linalg.lstsq(Ad, Tt[k][g], rcond=None)
                    Tt[k][g] -= Ad @ c
        if cfg.keep_hours is not None:
            drop = ~np.asarray(cfg.keep_hours, bool)[a0:a1]
            Tt[drop] = np.nan; Tx[drop] = np.nan; Ty[drop] = np.nan
        sl = slice(k0 - a0, k0 - a0 + (k1 - k0))
        sub = np.arange(k0, k1)[:: cfg.time_stride] - k0
        if sub.size == 0:
            continue
        r = TD.pool_rows(Tt[sl][sub], Tx[sl][sub], Ty[sl][sub], x2, y2, tsec[k0:k1][sub], cfg.block, depth=depth, min_frac=cfg.min_frac)
        rows.append(r)
        n_equiv += r["n_equiv"]
        if log:
            log(f"  stage 0: hours {k0}-{k1} of {nt}: {r['x'].size} pooled rows ({r['n_equiv']} pixel equations)")
    cat = lambda key: np.concatenate([r[key] for r in rows])
    R = {key: cat(key) for key in ("x", "y", "t", "Tt", "Tx", "Ty", "Cn", "depth")}
    bbox = (float(np.nanmin(x2)), float(np.nanmax(x2)), float(np.nanmin(y2)), float(np.nanmax(y2)))
    kw = dict(constituents=cfg.constituents, min_depth=cfg.min_depth, spacing=cfg.spacing, smooth=cfg.smooth, nuisance_harmonics=cfg.nuisance_harmonics, nuisance_spacing=cfg.nuisance_spacing, nuisance_t_spacing=cfg.nuisance_t_spacing, nuisance_t_smooth=cfg.nuisance_t_smooth, bbox=bbox,
              n_jobs=cfg.n_jobs, chunk=cfg.chunk_rows, Cn=R["Cn"], n_equiv=n_equiv)
    kw["support_radius"] = cfg.support_radius if cfg.support_radius is not None else cfg.block * float(np.mean(hx))
    depth_rows = np.where(np.isfinite(R["depth"]), R["depth"], np.inf)
    t0 = time.time()
    fit = TD.fit_tides(R["x"], R["y"], R["t"], R["Tt"], R["Tx"], R["Ty"], depth=depth_rows, **kw)
    if log:
        log(f"  stage 0: fit {time.time() - t0:.0f}s, {fit.n_rows} rows used ({fit.info['n_dropped']} dropped), {fit.info['n_params']} parameters, residual rms {fit.resid_rms:.2e} K/s")
    info = dict(n_rows=int(fit.n_rows), n_equiv=int(n_equiv), seconds=time.time() - t_wall, window=(str(np.datetime64(tax[0], "s")), str(np.datetime64(tax[-1], "s"))),
                step_s=step, n_hours=int(nt), grid=(int(ny), int(nx)), frac_deep=float(np.mean(depth >= (cfg.min_depth or 0.0))))
    return TideModel(fit=fit, lon0=frame.lon0, lat0=frame.lat0, t_ref=np.datetime64(t_ref, "ns"), cfg=cfg, info=info)

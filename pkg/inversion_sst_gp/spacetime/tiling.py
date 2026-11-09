"""Independent overlapping tiles with partition-of-unity blending, and tile extraction from a Himawari cube (see himawari.py).

Tiles are defined on the INDEX grid of the (already reduced) cube, so cores partition the domain exactly (no float edges) and the seam behaviour is the one tested in
experiments/26_tiling.py: each tile is the core plus a buffer on the sides that border another tile; its weight is 1 on the core and a sin^2 taper across the buffer
(0 at the outer edge); the blend is sum_k w_k f_k / sum_k w_k (a partition of unity). The same construction is used in time (a long period in windows with a
temporal buffer); the TEMPORAL seam has not been tested yet (only the spatial one has).

`extract_tile` returns everything a job needs in its own local frame: positions in metres from the tile centre (`geo.LocalFrame`), times in seconds from the
tile start, the SST anomaly (tile mean removed), the observed mask, and east / north gradients of a gridded field with the ellipsoid's metric factors (`TileData.gradients`).
"""
import dataclasses
import json
import pathlib

import numpy as np

from .geo import LocalFrame, metric_spacing


# ---------------------------------------------------------------------------------------------------------------- layout
def core_edges(n, size):
    """[(lo, hi)] cores of `size` cells along an axis of n cells; a remainder smaller than size / 2 is merged into the last core."""
    if size >= n:
        return [(0, n)]
    starts = list(range(0, n, size))
    if n - starts[-1] < size / 2 and len(starts) > 1:
        starts.pop()
    ends = starts[1:] + [n]
    return list(zip(starts, ends))


@dataclasses.dataclass
class TileSpec:
    """Index extents into the cube (time k, lat j, lon i): the buffered block [k0:k1, j0:j1, i0:i1] and its core [ck0:ck1, cj0:cj1, ci0:ci1]."""
    id: str
    k0: int; k1: int; j0: int; j1: int; i0: int; i1: int
    ck0: int; ck1: int; cj0: int; cj1: int; ci0: int; ci1: int

    @property
    def shape(self):
        return (self.k1 - self.k0, self.j1 - self.j0, self.i1 - self.i0)

    @property
    def core_shape(self):
        return (self.ck1 - self.ck0, self.cj1 - self.cj0, self.ci1 - self.ci0)

    @property
    def n_points(self):
        a, b, c = self.shape
        return a * b * c

    @property
    def buffers(self):
        """(lo_t, hi_t, lo_y, hi_y, lo_x, hi_x) buffer widths in cells."""
        return (self.ck0 - self.k0, self.k1 - self.ck1, self.cj0 - self.j0, self.j1 - self.cj1, self.ci0 - self.i0, self.i1 - self.ci1)

    @property
    def slices(self):
        return (slice(self.k0, self.k1), slice(self.j0, self.j1), slice(self.i0, self.i1))

    @property
    def core_slices_local(self):
        """Core inside the buffered block (for cutting a tile's output down to its core)."""
        return (slice(self.ck0 - self.k0, self.ck1 - self.k0), slice(self.cj0 - self.j0, self.cj1 - self.j0), slice(self.ci0 - self.i0, self.ci1 - self.i0))


def make_tiles(shape, core, buffer):
    """Tiles of an (nt, ny, nx) cube. `core` = (ct, cy, cx) and `buffer` = (bt, by, bx) in cells; a buffer exists only on sides that border another tile."""
    nt, ny, nx = shape
    out = []
    for a, (ck0, ck1) in enumerate(core_edges(nt, core[0])):
        for b, (cj0, cj1) in enumerate(core_edges(ny, core[1])):
            for c, (ci0, ci1) in enumerate(core_edges(nx, core[2])):
                out.append(TileSpec(f"t{a:02d}_{b:02d}_{c:02d}", max(0, ck0 - buffer[0]), min(nt, ck1 + buffer[0]), max(0, cj0 - buffer[1]), min(ny, cj1 + buffer[1]),
                                    max(0, ci0 - buffer[2]), min(nx, ci1 + buffer[2]), ck0, ck1, cj0, cj1, ci0, ci1))
    return out


def taper_1d(n, lo_buf, hi_buf):
    """Weights along one axis of a tile of n cells with lo_buf / hi_buf buffer cells at the low / high end (0 = no taper there): sin^2 rising across the buffer."""
    w = np.ones(n)
    for k in range(lo_buf):
        w[k] = np.sin(0.5 * np.pi * (k + 0.5) / lo_buf) ** 2
    for k in range(hi_buf):
        w[n - 1 - k] = np.sin(0.5 * np.pi * (k + 0.5) / hi_buf) ** 2
    return w


def tile_weights(spec):
    """(nt, ny, nx) blending weights of the buffered block of `spec`."""
    bt0, bt1, by0, by1, bx0, bx1 = spec.buffers
    nt, ny, nx = spec.shape
    return taper_1d(nt, bt0, bt1)[:, None, None] * taper_1d(ny, by0, by1)[None, :, None] * taper_1d(nx, bx0, bx1)[None, None, :]


class Blender:
    """Accumulates tile outputs into one field: result = sum_k w_k f_k / sum_k w_k. A convex combination, so a mean blends exactly; for a variance this is
    the conservative convex combination (the tile errors are correlated in the overlap, so the true variance of the blend is smaller than this only if they cancel).
    NaN cells of a field get zero weight for that field only (the normaliser is kept per field)."""

    def __init__(self, shape, n_fields=1):
        self.num = np.zeros((n_fields,) + tuple(shape))
        self.den = np.zeros((n_fields,) + tuple(shape))

    def add(self, spec, *fields):
        """`fields`: arrays of shape spec.shape, one per field."""
        w = tile_weights(spec)
        sl = spec.slices
        for f, a in enumerate(fields):
            a = np.asarray(a, float).reshape(spec.shape)
            ok = np.isfinite(a)
            self.num[(f,) + sl] += np.where(ok, w * a, 0.0)
            self.den[(f,) + sl] += np.where(ok, w, 0.0)
        return self

    def result(self):
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.num / np.where(self.den > 0, self.den, np.nan)


# ---------------------------------------------------------------------------------------------------------------- manifest
def write_manifest(path, specs, meta=None):
    pathlib.Path(path).write_text(json.dumps({"meta": meta or {}, "tiles": [dataclasses.asdict(s) for s in specs]}, indent=1))


def read_manifest(path):
    d = json.loads(pathlib.Path(path).read_text())
    return d["meta"], [TileSpec(**t) for t in d["tiles"]]


# ---------------------------------------------------------------------------------------------------------------- extraction
def grid_gradients_metric(T, shape, hx, hy, order=4):
    """Centred differences of a flat (time, y, x) field with the per-row metric spacings hx[j], hy[j] (metres per cell): returns flat (T_east, T_north), NaN on the border.
    Equals `impute.grid_gradients` on a uniform grid."""
    nt, ny, nx = shape
    T = np.asarray(T, float).reshape(shape)
    Tx = np.full_like(T, np.nan); Ty = np.full_like(T, np.nan)
    if order == 2:
        Tx[:, :, 1:-1] = (T[:, :, 2:] - T[:, :, :-2]) / 2
        Ty[:, 1:-1, :] = (T[:, 2:, :] - T[:, :-2, :]) / 2
    elif order == 4:
        Tx[:, :, 2:-2] = (-T[:, :, 4:] + 8 * T[:, :, 3:-1] - 8 * T[:, :, 1:-3] + T[:, :, :-4]) / 12
        Ty[:, 2:-2, :] = (-T[:, 4:, :] + 8 * T[:, 3:-1, :] - 8 * T[:, 1:-3, :] + T[:, :-4, :]) / 12
    else:
        raise ValueError("order must be 2 or 4")
    return (Tx / np.asarray(hx)[None, :, None]).ravel(), (Ty / np.asarray(hy)[None, :, None]).ravel()


@dataclasses.dataclass
class TileData:
    spec: TileSpec
    frame: LocalFrame
    shape: tuple            # (nt, ny, nx) of the buffered block
    x: np.ndarray           # flat (time, y, x) order, ALL grid points of the block: metres from the block centre
    y: np.ndarray
    t: np.ndarray           # seconds from the first time of the block
    T: np.ndarray           # SST anomaly (K, tile mean removed), NaN where not observed
    obs: np.ndarray         # bool, True where observed
    Tref: float             # the removed mean (K)
    lon: np.ndarray         # tile lon / lat axes (degrees)
    lat: np.ndarray
    hx: np.ndarray          # metres per cell east / north, per row
    hy: np.ndarray
    time: np.ndarray        # datetime64 axis
    weights: np.ndarray     # (nt, ny, nx) blending weights

    @property
    def n_obs(self):
        return int(self.obs.sum())

    def observed(self):
        """(x, y, t, T) at the observed points only."""
        o = self.obs
        return self.x[o], self.y[o], self.t[o], self.T[o]

    def gradients(self, field, order=4):
        """East / north gradients (per metre) of a flat field on the block grid, flat, NaN on the border."""
        return grid_gradients_metric(field, self.shape, self.hx, self.hy, order)

    def scatter(self, values_obs, fill=np.nan):
        """Place values defined at the observed points onto the full block grid (flat)."""
        out = np.full(self.obs.shape, fill, float)
        out[self.obs] = values_obs
        return out

    def coverage(self):
        """Observed fraction of the block's ocean pixels (pixels observed at least once), per time step."""
        o = self.obs.reshape(self.shape)
        ocean = o.any(axis=0)
        return o[:, ocean].mean(axis=1) if ocean.any() else np.zeros(self.shape[0])


def extract_tile(cube, spec, step_s=None):
    """Cut `spec` out of a cube from `himawari.load_cube` (or any Dataset with `sst(time, lat, lon)`), project to the block's local frame, remove the mean."""
    sst = np.asarray(cube["sst"].isel(time=slice(spec.k0, spec.k1), lat=slice(spec.j0, spec.j1), lon=slice(spec.i0, spec.i1)).values, np.float64)   # lazy: reads only the block
    nt, ny, nx = sst.shape
    lon = np.asarray(cube["lon"].values[spec.i0:spec.i1], float)
    lat = np.asarray(cube["lat"].values[spec.j0:spec.j1], float)
    time = cube["time"].values[spec.k0:spec.k1]
    frame = LocalFrame(0.5 * (lon[0] + lon[-1]), 0.5 * (lat[0] + lat[-1]))
    x2, y2 = frame.forward(lon[None, :], lat[:, None])
    sec = (time - time[0]).astype("timedelta64[s]").astype(np.float64)
    obs = np.isfinite(sst)
    Tref = float(np.nanmean(sst)) if obs.any() else float("nan")
    dlon = float(np.median(np.diff(lon))) if nx > 1 else 0.0
    dlat = float(np.median(np.diff(lat))) if ny > 1 else 0.0
    hx, hy = metric_spacing(lat, dlon, dlat)
    return TileData(spec=spec, frame=frame, shape=(nt, ny, nx),
                    x=np.broadcast_to(x2, (nt, ny, nx)).ravel().copy(), y=np.broadcast_to(y2, (nt, ny, nx)).ravel().copy(),
                    t=np.broadcast_to(sec[:, None, None], (nt, ny, nx)).ravel().copy(),
                    T=(sst - Tref).ravel(), obs=obs.ravel(), Tref=Tref, lon=lon, lat=lat, hx=hx, hy=hy, time=time, weights=tile_weights(spec))

"""Water depth on the cube's lon / lat grid, from a gridded bathymetry file (GEBCO layout; ETOPO / any lon-lat NetCDF with an elevation or depth variable works).

`load_depth(path, lon, lat)` returns the depth in metres (positive down, 0 on land) averaged over each target cell (the source is usually much finer than the 4 km cube: GEBCO
is 15 arc-seconds, so about 9 x 9 source cells per target cell), which is what the 200 m mask of the tide stage needs. Conventions handled:

  * variable: `elevation` / `z` / `topo` / `Band1` (positive up, ocean negative) or `depth` / `bathymetry` (positive down); override with `var=` and `positive='up'|'down'`.
  * coordinates: `lat` / `latitude` / `y` and `lon` / `longitude` / `x`, ascending or descending; a 0..360 source is shifted to the target's longitude convention.
  * the average is taken over the elevation (so a cell that is half land is shallow, not deep), then converted to depth.

Nothing here downloads data: on the HPC point it at the local GEBCO NetCDF.
"""
import numpy as np
import xarray as xr

_VAR_UP = ("elevation", "z", "topo", "Band1", "bed_elevation")
_VAR_DOWN = ("depth", "bathymetry", "bathy", "water_depth")
_LAT = ("lat", "latitude", "y")
_LON = ("lon", "longitude", "x")


def _find(names, candidates, what):
    low = {n.lower(): n for n in names}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    raise KeyError(f"no {what} found among {sorted(names)}; tried {candidates}")


def _edges(c):
    """Cell edges of a monotonic increasing centre axis."""
    c = np.asarray(c, float)
    d = np.diff(c)
    return np.r_[c[0] - d[0] / 2, c[:-1] + d / 2, c[-1] + d[-1] / 2]


def _bin_mean_1d(axis_src, edges, a, axis):
    """Sum and count of `a` along `axis` binned by the target `edges` (separable cell average, NaN-aware). Source cells outside the edges are dropped."""
    idx = np.searchsorted(edges, axis_src, side="right") - 1
    keep = (idx >= 0) & (idx < len(edges) - 1)
    a = np.moveaxis(a, axis, 0)[keep]
    idx = idx[keep]
    ok = np.isfinite(a)
    n_t = len(edges) - 1
    s = np.zeros((n_t,) + a.shape[1:]); n = np.zeros((n_t,) + a.shape[1:])
    np.add.at(s, idx, np.where(ok, a, 0.0))
    np.add.at(n, idx, ok.astype(float))
    return np.moveaxis(s, 0, axis), np.moveaxis(n, 0, axis)


def load_depth(path, lon, lat, var=None, positive=None, max_depth=None):
    """Depth (m, positive down, 0 on land) averaged over the cells of the target axes `lon` (nx,) and `lat` (ny,) (degrees, either order). Returns (ny, nx) in the target's order.

    A target cell with no source data inside it (a source coarser than the target, or a hole) is filled from the nearest source cell instead.
    """
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)
    with xr.open_dataset(path) as ds:
        names = list(ds.variables)
        vname = var or _find([n for n in names if ds[n].ndim >= 2], _VAR_UP + _VAR_DOWN, "bathymetry variable")
        pos = positive or ("down" if vname.lower() in _VAR_DOWN else "up")
        latn = _find(names, _LAT, "latitude"); lonn = _find(names, _LON, "longitude")
        slat = np.asarray(ds[latn].values, float); slon = np.asarray(ds[lonn].values, float)
        # a 0..360 source against a -180..180 target (or the reverse)
        if slon.max() > 180.0 and lon.min() < 0.0:
            slon = np.where(slon > 180.0, slon - 360.0, slon)
        elif slon.min() < 0.0 and lon.max() > 180.0:
            slon = np.where(slon < 0.0, slon + 360.0, slon)
        # window with a margin of one target cell
        dlon = float(np.median(np.abs(np.diff(lon)))) if lon.size > 1 else 0.1
        dlat = float(np.median(np.abs(np.diff(lat)))) if lat.size > 1 else 0.1
        jm = (slat >= lat.min() - dlat) & (slat <= lat.max() + dlat)
        im = (slon >= lon.min() - dlon) & (slon <= lon.max() + dlon)
        if not jm.any() or not im.any():
            raise ValueError("the bathymetry file does not cover the requested box")
        jj, ii = np.nonzero(jm)[0], np.nonzero(im)[0]
        da = ds[vname]
        da = da.isel({latn: slice(jj.min(), jj.max() + 1), lonn: slice(ii.min(), ii.max() + 1)})
        a = np.asarray(da.transpose(latn, lonn).values, float)
        slat_w = slat[jj.min():jj.max() + 1]; slon_w = slon[ii.min():ii.max() + 1]
    fill = a.copy()
    if not np.isfinite(a).any():
        raise ValueError("the bathymetry window contains no finite values")
    elev = a if pos == "up" else -a                       # elevation, positive up
    # ascending axes for the binning
    if slat_w[0] > slat_w[-1]:
        slat_w = slat_w[::-1]; elev = elev[::-1]
    if slon_w[0] > slon_w[-1]:
        slon_w = slon_w[::-1]; elev = elev[:, ::-1]
    lat_a = np.sort(lat); lon_a = np.sort(lon)
    sy, ny_ = _bin_mean_1d(slat_w, _edges(lat_a), elev, 0)
    sxy, nxy = _bin_mean_1d(slon_w, _edges(lon_a), sy, 1)
    cnt, _ = _bin_mean_1d(slon_w, _edges(lon_a), ny_, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_elev = sxy / cnt
    empty = ~(cnt > 0)
    if empty.any():                                       # nearest source cell for cells that caught nothing
        jn = np.abs(slat_w[None, :] - lat_a[:, None]).argmin(1)
        in_ = np.abs(slon_w[None, :] - lon_a[:, None]).argmin(1)
        near = elev[np.ix_(jn, in_)]
        mean_elev = np.where(empty, near, mean_elev)
    depth = np.clip(-mean_elev, 0.0, None)
    depth = np.nan_to_num(depth, nan=0.0)
    if max_depth is not None:
        depth = np.minimum(depth, max_depth)
    # back to the target's order
    if lat[0] > lat[-1]:
        depth = depth[::-1]
    if lon[0] > lon[-1]:
        depth = depth[:, ::-1]
    return depth


def grid_scattered_depth(px, py, pdepth, lon, lat, min_count=1):
    """Cell-mean depth (m, positive down) on the lon / lat axes from SCATTERED points (e.g. the cell centres of an unstructured SUNTANS mesh): the mean of `pdepth` over the points inside
    each target cell (edges half-way between the axis values), NaN where a cell holds fewer than `min_count` points (land, or outside the mesh). Returns (ny, nx) in the target's order."""
    px, py, pd_ = (np.asarray(a, float).ravel() for a in (px, py, pdepth))
    lon = np.asarray(lon, float); lat = np.asarray(lat, float)
    lo, la = np.sort(lon), np.sort(lat)
    ex, ey = _edges(lo), _edges(la)
    ix = np.searchsorted(ex, px, side="right") - 1
    iy = np.searchsorted(ey, py, side="right") - 1
    ok = (ix >= 0) & (ix < lo.size) & (iy >= 0) & (iy < la.size) & np.isfinite(pd_)
    flat = iy[ok] * lo.size + ix[ok]
    s = np.bincount(flat, weights=pd_[ok], minlength=la.size * lo.size).reshape(la.size, lo.size)
    n = np.bincount(flat, minlength=la.size * lo.size).reshape(la.size, lo.size)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(n >= min_count, s / n, np.nan)
    if lat[0] > lat[-1]:
        out = out[::-1]
    if lon[0] > lon[-1]:
        out = out[:, ::-1]
    return out


def write_synthetic_gebco(path, lon, lat, depth_fn, dlon=1 / 240.0, dlat=1 / 240.0, name="elevation", descending_lat=False, lon360=False):
    """Write a GEBCO-style NetCDF (15 arc-second grid by default) from `depth_fn(lon, lat) -> depth (m, positive down)`; for tests. Land is negative depth."""
    lo = np.arange(lon.min() - 0.1, lon.max() + 0.1, dlon)
    la = np.arange(lat.min() - 0.1, lat.max() + 0.1, dlat)
    if descending_lat:
        la = la[::-1]
    LO, LA = np.meshgrid(lo, la)
    z = -depth_fn(LO, LA)
    lo_out = np.where(lo < 0, lo + 360.0, lo) if lon360 else lo
    ds = xr.Dataset({name: (("lat", "lon"), z.astype(np.float32))}, coords={"lat": la, "lon": lo_out})
    ds.to_netcdf(path)
    return path

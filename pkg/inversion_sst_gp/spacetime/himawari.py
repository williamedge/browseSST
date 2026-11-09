"""Himawari-9 L3C SST loader: per-timestamp NetCDF files -> a masked (time, lat, lon) cube.

Reads the files written by `pysatview` (`HimawariDataProcessor.process_time_series`, one `YYYYmmddHHMMSS.nc` per hour holding `sea_surface_temperature` and
`quality_level` for the downloaded box), and also accepts raw granules (`YYYYmmddHHMMSS-STAR-L3C_...nc`, global, lat descending), which are cropped lazily. This module
needs only xarray / netCDF4: the download (earthaccess, an Earthdata .netrc) stays in pysatview, so loading works on the HPC without network access.

Masking is the quality flag only (the user's scope: choose periods of good coverage; clouds are not modelled): a pixel is valid when `quality_level >= qmin`
(default 5 = ACSPO "best quality / clear sky") and its SST is finite and inside `sst_range`. Missing hours become all-NaN slabs so the time axis is regular.
Optional reduction to a coarser grid (the plan is 4 km from the 0.02 deg = ~2 km product): `reduce='stride'` keeps every `stride`-th pixel, `'mean'` averages
`stride x stride` blocks that have at least `min_frac` of their pixels valid (this lowers the noise by up to `stride` and changes what sigma_T means).
"""
import pathlib
import re
import warnings

import numpy as np
import xarray as xr

_NAME_RE = re.compile(r"^(\d{14})(?:-STAR.*)?\.nc$")
_SST_NAMES = ("sea_surface_temperature", "sst", "sea_surface_temperature_skin")
_QUAL_NAMES = ("quality_level", "quality_level_sst")


def scan_files(nc_dir):
    """{timestamp (numpy datetime64[s]): path} for the Himawari files in `nc_dir`; the short pysatview name wins over a raw granule of the same time."""
    out = {}
    for p in sorted(pathlib.Path(nc_dir).glob("*.nc")):
        m = _NAME_RE.match(p.name)
        if not m:
            continue
        ts = np.datetime64(f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:8]}T{m.group(1)[8:10]}:{m.group(1)[10:12]}:{m.group(1)[12:14]}", 's')
        if ts not in out or p.name == f"{m.group(1)}.nc":
            out[ts] = p
    return out


def _pick(names, candidates, what):
    for c in candidates:
        if c in names:
            return c
    raise KeyError(f"no {what} variable among {sorted(names)}")


def _read_one(path, lonlim, latlim, qmin, sst_range):
    """(lat, lon, sst[ny, nx] with NaN for invalid) of one file, cropped, lat ascending."""
    with xr.open_dataset(path, engine="netcdf4", mask_and_scale=True) as ds:
        lon_n = _pick(ds.coords, ("lon", "longitude"), "longitude")
        lat_n = _pick(ds.coords, ("lat", "latitude"), "latitude")
        sst_n = _pick(ds.data_vars, _SST_NAMES, "SST")
        q_n = next((c for c in _QUAL_NAMES if c in ds.data_vars), None)
        if float(ds[lat_n].values[0]) > float(ds[lat_n].values[-1]):
            ds = ds.sortby(lat_n)
        sel = {}
        if lonlim is not None:
            sel[lon_n] = slice(min(lonlim), max(lonlim))
        if latlim is not None:
            sel[lat_n] = slice(min(latlim), max(latlim))
        sub = ds[[sst_n] + ([q_n] if q_n else [])].sel(sel)
        sst = np.asarray(sub[sst_n].values, np.float32)
        sst = sst.reshape(sst.shape[-2:])
        valid = np.isfinite(sst) & (sst >= sst_range[0]) & (sst <= sst_range[1])
        if q_n is not None and qmin is not None:
            q = np.asarray(sub[q_n].values, np.float32).reshape(sst.shape)
            valid &= np.isfinite(q) & (q >= qmin)
        sst = np.where(valid, sst, np.nan).astype(np.float32)
        return np.asarray(sub[lat_n].values, float), np.asarray(sub[lon_n].values, float), sst


def reduce_grid(sst, stride, mode="stride", min_frac=1.0, lat=None, lon=None):
    """Reduce a (time, ny, nx) cube by `stride`; returns (sst, lat, lon) (lat / lon None if not given)."""
    s = int(stride)
    if s == 1:
        return sst, lat, lon
    if mode == "stride":
        return sst[:, ::s, ::s], None if lat is None else lat[::s], None if lon is None else lon[::s]
    if mode != "mean":
        raise ValueError("mode must be 'stride' or 'mean'")
    nt, ny, nx = sst.shape
    ny2, nx2 = ny // s, nx // s
    blk = sst[:, :ny2 * s, :nx2 * s].reshape(nt, ny2, s, nx2, s)
    cnt = np.isfinite(blk).sum(axis=(2, 4))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = np.nanmean(blk, axis=(2, 4))
    m = np.where(cnt >= min_frac * s * s, m, np.nan).astype(np.float32)
    lat2 = None if lat is None else lat[:ny2 * s].reshape(ny2, s).mean(1)
    lon2 = None if lon is None else lon[:nx2 * s].reshape(nx2, s).mean(1)
    return m, lat2, lon2


def load_cube(nc_dir, t0=None, t1=None, step=3600, lonlim=None, latlim=None, qmin=5, sst_range=(268.0, 313.0), stride=1, reduce="stride", min_frac=1.0):
    """Stack the files of `nc_dir` with `t0 <= time <= t1` (datetime64 / ISO strings; default all) into an `xarray.Dataset` with `sst` (K, NaN = invalid)
    on a regular `step`-second time axis, ascending lat. Raises if the files do not share one lon / lat grid."""
    files = scan_files(nc_dir)
    if not files:
        raise FileNotFoundError(f"no Himawari files (YYYYmmddHHMMSS[-STAR...].nc) in {nc_dir}")
    times = np.array(sorted(files), dtype="datetime64[s]")
    lo = times[0] if t0 is None else np.datetime64(t0, 's')
    hi = times[-1] if t1 is None else np.datetime64(t1, 's')
    times = times[(times >= lo) & (times <= hi)]
    if times.size == 0:
        raise FileNotFoundError(f"no files between {lo} and {hi} in {nc_dir}")
    taxis = np.arange(times[0], times[-1] + np.timedelta64(1, 's'), np.timedelta64(int(step), 's'))
    off = (times - times[0]).astype('timedelta64[s]').astype(int)
    if np.any(off % int(step)):
        raise ValueError(f"file times are not a multiple of step = {step} s apart")
    lat = lon = None
    slabs = {}
    for ts in times:
        la, lo_, s = _read_one(files[ts], lonlim, latlim, qmin, sst_range)
        if lat is None:
            lat, lon = la, lo_
        elif la.shape != lat.shape or lo_.shape != lon.shape or not (np.allclose(la, lat, atol=1e-4) and np.allclose(lo_, lon, atol=1e-4)):
            raise ValueError(f"{files[ts].name} is on a different lon / lat grid from the first file")
        slabs[ts] = s
    if lat.size == 0 or lon.size == 0:
        raise ValueError("the requested box contains no grid points")
    cube = np.full((taxis.size,) + lat.shape + lon.shape, np.nan, np.float32)
    for ts, s in slabs.items():
        cube[int((ts - taxis[0]) / np.timedelta64(int(step), 's'))] = s
    cube, lat, lon = reduce_grid(cube, stride, reduce, min_frac, lat, lon)
    ds = xr.Dataset({"sst": (("time", "lat", "lon"), cube, {"units": "K"})}, coords={"time": taxis.astype("datetime64[ns]"), "lat": lat, "lon": lon})
    ds.attrs.update(step_s=int(step), qmin=-1 if qmin is None else int(qmin), stride=int(stride), reduce=reduce, n_files=len(times), n_missing_hours=int(taxis.size - len(times)))
    return ds


def ocean_mask(cube, min_frac=0.05):
    """(ny, nx) bool: pixels valid at least `min_frac` of the time. Pixels never valid are land / always-flagged, not 'cloud'; with a
    short cube this cannot tell a persistently clouded ocean pixel from land, so give it at least a few days."""
    return np.isfinite(cube["sst"].values).mean(axis=0) >= min_frac


def coverage(cube, ocean=None):
    """Fraction of the ocean pixels with a valid SST, per time step (numpy array)."""
    v = np.isfinite(cube["sst"].values)
    if ocean is None:
        ocean = v.any(axis=0)
    return v[:, ocean].mean(axis=1) if ocean.any() else np.zeros(v.shape[0])


def find_windows(cov, length, min_mean=0.8, min_each=0.5):
    """Start indices of all `length`-step windows whose mean coverage >= `min_mean` and whose every step is >= `min_each` (cov = `coverage`)."""
    cov = np.asarray(cov, float)
    if cov.size < length:
        return np.array([], int)
    k = np.ones(length)
    mean = np.convolve(cov, k / length, mode="valid")
    worst = np.array([cov[i:i + length].min() for i in range(cov.size - length + 1)])
    return np.flatnonzero((mean >= min_mean) & (worst >= min_each))

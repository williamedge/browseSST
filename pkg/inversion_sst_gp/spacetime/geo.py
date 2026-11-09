"""Local map frame and metric factors for lon / lat grids (vectorised; replaces the per-pixel geodesic loop of `utils.geo_to_car`).

`LocalFrame(lon0, lat0)` is the WGS84 azimuthal-equidistant projection about a tile centre (pyproj): distances from the centre are exact and the
distortion is O((d / R)^2) elsewhere (about 1e-4 at 100 km). The kernels only need metres, so the Gaussian-process stage works in these (x, y).
The advection equation needs gradients along LOCAL EAST / NORTH, which are obtained exactly from the lon / lat grid with the ellipsoid's radii of
curvature (`metric_spacing`); the resulting (u, v) are then east / north components. The tile x axis differs from local east by the convergence angle
(`east_angle`), a fraction of a degree for tiles of 1-2 degrees, which is ignored in the kernels (and can be removed with `to_east_north`).
"""
import numpy as np

A_WGS84 = 6378137.0
E2_WGS84 = 0.00669437999014131


def radii_of_curvature(lat):
    """Meridional M and prime-vertical N radii of curvature (m) at latitude `lat` (degrees)."""
    s2 = np.sin(np.radians(lat)) ** 2
    w = np.sqrt(1.0 - E2_WGS84 * s2)
    return A_WGS84 * (1.0 - E2_WGS84) / w ** 3, A_WGS84 / w


def metric_spacing(lat, dlon, dlat):
    """Grid spacing in metres of a regular lon / lat grid with steps `dlon`, `dlat` (degrees): returns (hx, hy) at the latitudes `lat`
    (east: N cos(lat) dlon, north: M dlat)."""
    M, N = radii_of_curvature(np.asarray(lat, float))
    return N * np.cos(np.radians(lat)) * np.radians(abs(dlon)), M * np.radians(abs(dlat))


class LocalFrame:
    def __init__(self, lon0, lat0):
        import pyproj
        self.lon0, self.lat0 = float(lon0), float(lat0)
        self._p = pyproj.Proj(proj='aeqd', lat_0=self.lat0, lon_0=self.lon0, ellps='WGS84', units='m')

    def forward(self, lon, lat):
        lon, lat = np.broadcast_arrays(np.asarray(lon, float), np.asarray(lat, float))
        x, y = self._p(lon, lat)
        return np.asarray(x), np.asarray(y)

    def inverse(self, x, y):
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        lon, lat = self._p(x, y, inverse=True)
        return np.asarray(lon), np.asarray(lat)

    def east_angle(self, lon, lat, eps=1e-4):
        """Angle (radians, counter-clockwise) of local east relative to the frame x axis at (lon, lat)."""
        x0, y0 = self.forward(np.asarray(lon, float) - eps, lat)
        x1, y1 = self.forward(np.asarray(lon, float) + eps, lat)
        return np.arctan2(y1 - y0, x1 - x0)

    def to_east_north(self, u, v, lon, lat):
        """Rotate frame-axis components (u_x, u_y) to (east, north)."""
        th = self.east_angle(lon, lat)
        c, s = np.cos(th), np.sin(th)
        return u * c + v * s, -u * s + v * c

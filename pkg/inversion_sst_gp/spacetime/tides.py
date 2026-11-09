"""Regional barotropic-tide regression from SST (stage 0, run before the per-tile GP pipeline).

Model. The tracer equation dT/dt + u.grad T = S with u = u_bt + u'. The tidal part is a sum of harmonics of FIXED frequencies (M2, S2, K1, O1 by default) whose coefficients vary
SLOWLY IN SPACE (amplitude and phase change with depth):

    u_bt(x, y, t) = sum_k [ a_k(x, y) cos(w_k t) + b_k(x, y) sin(w_k t) ]   (same for v with c_k, d_k)
    a_k(x, y) = sum_m beta_{k,m} B_m(x, y),     B_m a tensor-product cubic B-spline basis with knot spacing `spacing`.

The equation is linear in the unknown coefficients once grad T and dT/dt are taken from the data:

    dT/dt = - Tx u_bt - Ty v_bt + S_nuis(x, y, t) + (noise = the eddy / non-tidal terms u'.grad T and the part of S that is not modelled).

S_nuis = sum_h [ p_h(x, y) cos(h Omega t) + q_h(x, y) sin(h Omega t) ], Omega the solar day (h = 0..H), with a coarser spatial basis: the diurnal heating shares frequencies with K1 / S1 and (its
second harmonic) with S2 exactly, so it MUST be fitted jointly; what separates it from the tide is that the tide multiplies the (time-varying) gradient. K1 also contains P1 and S2 contains K2
(they are not separable from K1 / S2 in a window shorter than ~6 months).

Fitting is penalised least squares (second-difference P-spline penalty), normal equations accumulated in row chunks with sparse design matrices (memory is set by `chunk`, not by the
number of rows) and optionally in forked worker processes (`n_jobs`). Cost: about 1 s per 1000 rows serial on the test machine (a 30-day, 3 deg x 3 deg fit with moment pooling, 0.73 M
pseudo-rows and 928 coefficients, took 11 min while the machine was loaded), independent of the number of coefficients; a 15 deg x 15 deg x 3 month fit would be of order 75 times that
(not benchmarked), so use `n_jobs` or the HPC. Rows shallower than `min_depth` (default 200 m) are dropped: shelf tides are rough in space and strongly non-linear.
Inputs are flat arrays of rows (x, y metres; t seconds; Tt K/s; Tx, Ty K/m along the SAME axes as the velocity components). `rows_from_cube` builds them from a gridded cube
(block means) and `pool_rows` pools the equations of coarse blocks through their second moments, which keeps the within-block gradient structure (plain block means of dT/dt and
grad T lose most of the information: tested in experiments/30). `depth_scale` applies a known-bathymetry factor (depth / ref) ** -p to the gradients so that the splines only represent the
remainder. Workflow: (1) products of the data (4th-order differences, `time_derivative`, `tiling.grid_gradients_metric`), (2) rows, (3) `fit_tides`, (4) in each tile subtract
u_bt . grad T from dT/dt before the GP velocity stage and add `TideFit.velocity` back to the GP velocity.
"""
import dataclasses

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
from scipy.interpolate import BSpline
from scipy.spatial import cKDTree

# angular speed in degrees per hour (standard tidal-constituent table)
CONSTITUENT_DEG_PER_HOUR = {'M2': 28.9841042, 'S2': 30.0, 'N2': 28.4397295, 'K2': 30.0821373, 'K1': 15.0410686, 'O1': 13.9430356, 'P1': 14.9589314, 'Q1': 13.3986609}
SOLAR_DAY = 86400.0


def omega_of(name):
    """Angular frequency (rad / s) of a constituent."""
    return np.deg2rad(CONSTITUENT_DEG_PER_HOUR[name]) / 3600.0


def rayleigh_days(a, b):
    """Record length (days) needed to separate two constituents (Rayleigh criterion)."""
    fa, fb = CONSTITUENT_DEG_PER_HOUR[a] / 360.0, CONSTITUENT_DEG_PER_HOUR[b] / 360.0     # cycles per hour
    return 1.0 / abs(fa - fb) / 24.0


def depth_scale(depth, exponent=0.5, ref=1000.0):
    """Known-bathymetry scaling (depth / ref) ** (-exponent) of the barotropic current (Green's law: flux conservation gives exponent 1/2). Use it to scale the gradients BEFORE
    the regression (Tx_s = Tx * scale(depth), per pixel) so that the splines only have to represent the slow remainder, and multiply `TideFit.velocity(..., scale=...)` /
    `harmonics(..., scale=...)` by the same factor at the query positions. exponent = 0 gives 1 (no scaling)."""
    return (np.asarray(depth, float) / ref) ** (-exponent)


# ---------------------------------------------------------------------------------------------------------------- spatial basis
class SplineBasis2D:
    """Tensor-product cubic B-splines on uniform knots covering [x0, x1] x [y0, y1], `spacing` between breakpoints (coordinates outside are clipped to the box)."""

    def __init__(self, bbox, spacing):
        x0, x1, y0, y1 = bbox
        self.bbox = (float(x0), float(x1), float(y0), float(y1))
        self.knots = []
        self.n = []
        for a, b in ((x0, x1), (y0, y1)):
            ni = max(1, int(round((b - a) / spacing)))
            step = (b - a) / ni
            self.knots.append(a + step * np.arange(-3, ni + 4))          # uniform knots (P-splines): the second-difference penalty is exactly zero on planes
            self.n.append(ni + 3)
        self.Mx, self.My = self.n
        self.size = self.Mx * self.My

    def design(self, x, y):
        """(cols, vals): (n, 16) arrays; row i of the sparse design matrix has `vals[i, j]` in column `cols[i, j]` (column = iy * Mx + ix)."""
        x = np.clip(np.asarray(x, float), self.bbox[0], self.bbox[1])
        y = np.clip(np.asarray(y, float), self.bbox[2], self.bbox[3])
        n = x.size
        Bx = BSpline.design_matrix(x, self.knots[0], 3).tocsr()
        By = BSpline.design_matrix(y, self.knots[1], 3).tocsr()
        ix, vx = Bx.indices.reshape(n, 4), Bx.data.reshape(n, 4)
        iy, vy = By.indices.reshape(n, 4), By.data.reshape(n, 4)
        cols = (iy[:, :, None] * self.Mx + ix[:, None, :]).reshape(n, 16)
        vals = (vy[:, :, None] * vx[:, None, :]).reshape(n, 16)
        return cols, vals

    def penalty(self):
        """Second-difference (P-spline) penalty on the coefficient grid, sparse (size, size)."""
        def d2(m):
            return sp.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(m - 2, m)).tocsr() if m > 2 else sp.csr_matrix((0, m))
        Dx, Dy = d2(self.Mx), d2(self.My)
        Ix, Iy = sp.identity(self.Mx, format="csr"), sp.identity(self.My, format="csr")
        return (sp.kron(Iy, Dx.T @ Dx) + sp.kron(Dy.T @ Dy, Ix)).tocsr()


# ---------------------------------------------------------------------------------------------------------------- fit object
@dataclasses.dataclass
class TideFit:
    names: tuple
    omegas: np.ndarray
    basis: SplineBasis2D
    beta: np.ndarray            # (K, 2, 2, size): constituent, component (u, v), (cos, sin), spline coefficients; m/s
    nuis_basis: SplineBasis2D
    nuis_harmonics: int         # -1 = no nuisance term
    gamma: np.ndarray           # (2 * H + 1, nuis size) nuisance coefficients, K / s; (2 * H + 1, n_days, nuis size) if the nuisance amplitudes are day-by-day (`nuis_t_spacing`)
    support_xy: np.ndarray      # unique (x, y) of the rows used (for the support mask)
    support_radius: float
    n_rows: int
    resid_rms: float            # K / s, rms of the weighted residual
    cov: np.ndarray = None      # covariance of the coefficients (beta first, then the nuisance), optimistic: residuals are not white; the beta block only (n_tide x n_tide) when solver='schur'
    nuis_t_spacing: float = None  # s; knot spacing of the piecewise-linear (hat) day-by-day amplitudes of the nuisance harmonics (None = one fixed cycle for the whole record)
    nuis_t0: float = 0.0        # s; first hat knot
    info: dict = dataclasses.field(default_factory=dict)

    def _tide_basis_values(self, x, y):
        cols, vals = self.basis.design(x, y)
        return cols, vals

    def coefficient_fields(self, x, y):
        """{name: (a_u, b_u, a_v, b_v)} (m/s) at positions: u = a cos(w t) + b sin(w t) per constituent."""
        cols, vals = self._tide_basis_values(x, y)
        out = {}
        for k, nm in enumerate(self.names):
            out[nm] = tuple(np.einsum('ij,ij->i', self.beta[k, c, p][cols], vals) for c in (0, 1) for p in (0, 1))
        return out

    def harmonics(self, x, y, scale=None):
        """{name: dict(Au, phi_u, Av, phi_v)}: amplitude (m/s) and phase (rad) with u = A cos(w t - phi), t as given to the fit. `scale`: the `depth_scale` factor at the positions
        if the fit used scaled gradients."""
        out = {}
        sc = 1.0 if scale is None else np.asarray(scale, float)
        for nm, (au, bu, av, bv) in self.coefficient_fields(x, y).items():
            out[nm] = dict(Au=np.hypot(au, bu) * sc, phi_u=np.arctan2(bu, au), Av=np.hypot(av, bv) * sc, phi_v=np.arctan2(bv, av))
        return out

    def velocity(self, x, y, t, mask_unsupported=True, scale=None):
        """Tidal (u, v) at positions and times (broadcast together); NaN farther than `support_radius` from any retained data point if `mask_unsupported`.
        `scale` (broadcastable to x): multiplies the result, the `depth_scale` factor if the fit used depth-scaled gradients."""
        x, y, t = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float), np.asarray(t, float))
        sh = x.shape
        xf, yf, tf = x.ravel(), y.ravel(), t.ravel()
        cols, vals = self._tide_basis_values(xf, yf)
        u = np.zeros(xf.size); v = np.zeros(xf.size)
        for k in range(len(self.names)):
            ct, st = np.cos(self.omegas[k] * tf), np.sin(self.omegas[k] * tf)
            f = lambda c, p: np.einsum('ij,ij->i', self.beta[k, c, p][cols], vals)
            u += f(0, 0) * ct + f(0, 1) * st
            v += f(1, 0) * ct + f(1, 1) * st
        if scale is not None:
            sc = np.broadcast_to(np.asarray(scale, float), sh).ravel()
            u *= sc; v *= sc
        if mask_unsupported:
            ok = self.supported(xf, yf)
            u[~ok] = np.nan; v[~ok] = np.nan
        return u.reshape(sh), v.reshape(sh)

    def supported(self, x, y):
        d, _ = cKDTree(self.support_xy).query(np.column_stack([np.ravel(x), np.ravel(y)]))
        return d <= self.support_radius


_ASSEMBLE = None


def _run_assemble(a):
    return _ASSEMBLE(a)


def _time_terms(t, omegas):
    return [(np.cos(w * t), np.sin(w * t)) for w in omegas]


def fit_tides(x, y, t, Tt, Tx, Ty, depth=None, min_depth=200.0, constituents=('M2', 'S2', 'K1', 'O1'), spacing=75e3, smooth=1e-3,
              nuisance_harmonics=4, nuisance_spacing=None, nuisance_t_spacing=None, nuisance_t_smooth=0.0, bbox=None, weights=None, chunk=100000, compute_cov=True, support_radius=None, n_jobs=1, Cn=None, n_equiv=None, solver='auto'):
    """Penalised least squares for the tidal coefficient fields (see the module docstring).

    x, y, t, Tt, Tx, Ty : flat arrays of rows (finite). `depth` (same length, m) with `min_depth` drops shallow rows (`min_depth=None` keeps everything).
    spacing : spline breakpoint spacing (m); the tidal fields can vary on scales longer than this. smooth : relative weight of the second-difference penalty
    (`lam = smooth * trace(X'X) / trace(P)`). nuisance_harmonics : harmonics H of the solar day in the diurnal-heating nuisance (-1 = none) on a spatial basis of spacing
    `nuisance_spacing` (default 3 x spacing). nuisance_t_spacing : if given (s, e.g. 86400), the nuisance amplitudes are DAY-BY-DAY: each harmonic's coefficient field is a piecewise-linear
    function of time with knots every `nuisance_t_spacing` (a partition of unity, so a constant amplitude is a special case), i.e. the heating cycle may differ from one day to the next;
    nuisance_t_smooth (relative weight, 0 = free) penalises the second difference of the knot amplitudes in time. weights : optional row weights (e.g. IRLS). n_jobs : worker processes (forked) for the normal-equation assembly.
    Cn : optional multiplier of the nuisance term per row (default 1); with `n_equiv` (the number of ORIGINAL equations, for the residual degrees of freedom) it lets `pool_moments` pass
    pseudo-rows that reproduce the normal equations of every pixel in a block exactly (see there).
    solver : 'dense' (one dense Cholesky of all P_tot coefficients; covariance of all coefficients), 'schur' (the day-by-day nuisance block is banded in time, so it is eliminated by a banded
    Cholesky and only the small tide block is solved densely: memory and time grow linearly with the record length; `cov` is then the tide block only), or 'auto' (schur for P_tot > 3000).
    """
    x, y, t, Tt, Tx, Ty = (np.asarray(a, float).ravel() for a in (x, y, t, Tt, Tx, Ty))
    Cn = np.ones(x.size) if Cn is None else np.asarray(Cn, float).ravel()
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(t) & np.isfinite(Tt) & np.isfinite(Tx) & np.isfinite(Ty) & np.isfinite(Cn)
    if depth is not None and min_depth is not None:
        keep &= np.asarray(depth, float).ravel() >= min_depth
    w = np.ones(x.size) if weights is None else np.asarray(weights, float).ravel()
    keep &= np.isfinite(w) & (w > 0)
    n_drop_shallow = int((~keep).sum())
    x, y, t, Tt, Tx, Ty, w, Cn = x[keep], y[keep], t[keep], Tt[keep], Tx[keep], Ty[keep], w[keep], Cn[keep]
    N = x.size
    if N == 0:
        raise ValueError("no usable rows")
    names = tuple(constituents)
    omegas = np.array([omega_of(n) for n in names])
    K = len(names)
    if bbox is None:
        bbox = (x.min(), x.max(), y.min(), y.max())
    basis = SplineBasis2D(bbox, spacing)
    nuis_basis = SplineBasis2D(bbox, nuisance_spacing or 3 * spacing)
    H = nuisance_harmonics
    n_tide = K * 4 * basis.size
    daily = nuisance_t_spacing is not None and H >= 0
    if daily:
        nuis_t0 = float(t.min()); nd = int(np.ceil((t.max() - nuis_t0) / nuisance_t_spacing)) + 1
        nd = max(nd, 2)
    else:
        nuis_t0, nd = 0.0, 1
    n_nuis = (2 * H + 1) * nd * nuis_basis.size if H >= 0 else 0
    P_tot = n_tide + n_nuis
    # column scaling: response R0, gradients G0 (so that all columns are O(1))
    R0 = float(np.sqrt(np.mean(Tt ** 2))) or 1.0
    G0 = float(np.sqrt(np.mean(np.r_[Tx, Ty] ** 2))) or 1.0
    comps = (Tx / G0, Ty / G0)
    H_ = H

    def assemble(a):
        """Normal-equation contribution of rows a:a+chunk: (sparse X'X, X'r, r'r)."""
        sl = slice(a, min(N, a + chunk))
        n = sl.stop - sl.start
        sw = np.sqrt(w[sl])
        cols_b, vals_b = basis.design(x[sl], y[sl])
        tt = _time_terms(t[sl], omegas)
        C, V = [], []
        off = 0
        for k in range(K):
            for c in (0, 1):
                g = -comps[c][sl]
                for p in (0, 1):
                    C.append(cols_b + off); V.append(vals_b * (g * tt[k][p] * sw)[:, None]); off += basis.size
        if H_ >= 0:
            cols_n, vals_n = nuis_basis.design(x[sl], y[sl])
            if daily:
                u_ = (t[sl] - nuis_t0) / nuisance_t_spacing
                k0 = np.clip(np.floor(u_).astype(int), 0, nd - 2); fr = np.clip(u_ - k0, 0.0, 1.0)
            for h in range(H_ + 1):
                ang = h * 2 * np.pi / SOLAR_DAY * t[sl]
                for tr in ((np.cos(ang),) if h == 0 else (np.cos(ang), np.sin(ang))):
                    if daily:
                        C.append(cols_n + (off + k0 * nuis_basis.size)[:, None]); V.append(vals_n * (tr * Cn[sl] * sw * (1 - fr))[:, None])
                        C.append(cols_n + (off + (k0 + 1) * nuis_basis.size)[:, None]); V.append(vals_n * (tr * Cn[sl] * sw * fr)[:, None])
                        off += nd * nuis_basis.size
                    else:
                        C.append(cols_n + off); V.append(vals_n * (tr * Cn[sl] * sw)[:, None]); off += nuis_basis.size
        assert off == P_tot
        Xc = sp.csr_array((np.concatenate(V, axis=1).ravel(), np.concatenate(C, axis=1).ravel(), np.arange(n + 1) * (16 * len(C))), shape=(n, P_tot))
        rc = Tt[sl] / R0 * sw
        return (Xc.T @ Xc).tocsr(), np.asarray(Xc.T @ rc), float(rc @ rc)

    starts = list(range(0, N, chunk))
    if n_jobs > 1 and len(starts) > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor
        global _ASSEMBLE
        _ASSEMBLE = assemble                                              # inherited by the forked workers
        with ProcessPoolExecutor(min(n_jobs, len(starts)), mp_context=multiprocessing.get_context('fork')) as ex:
            parts = list(ex.map(_run_assemble, starts))
    else:
        parts = map(assemble, starts)
    XtX_s = sp.csr_matrix((P_tot, P_tot)); Xtr = np.zeros(P_tot); rss0 = 0.0
    for A_, b_, r_ in parts:
        XtX_s = XtX_s + A_; Xtr += b_; rss0 += r_
    if solver not in ('auto', 'dense', 'schur'):
        raise ValueError("solver must be 'auto', 'dense' or 'schur'")
    use_schur = (solver == 'schur' or (solver == 'auto' and P_tot > 3000)) and daily
    dg = XtX_s.diagonal()
    # penalty (same lam for every block, relative to the mean diagonal of that block family)
    Pt = sp.block_diag([basis.penalty()] * (K * 4) + ([nuis_basis.penalty()] * ((2 * H + 1) * nd) if H >= 0 else []), format="csr")
    pdg = Pt.diagonal()
    lam_t = smooth * dg[:n_tide].sum() / max(pdg[:n_tide].sum(), 1e-30)
    pen = np.zeros(P_tot); pen[:n_tide] = lam_t
    if H >= 0:
        lam_n = smooth * dg[n_tide:].sum() / max(pdg[n_tide:].sum(), 1e-30)
        pen[n_tide:] = lam_n
    if daily and nuisance_t_smooth > 0:
        D2t = sp.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(nd - 2, nd)).tocsr() if nd > 2 else sp.csr_matrix((0, nd))
        Pnt = sp.block_diag([sp.kron(D2t.T @ D2t, sp.identity(nuis_basis.size, format="csr"))] * (2 * H + 1), format="csr")
        Pt = Pt + sp.block_diag([sp.csr_matrix((n_tide, n_tide)), Pnt * (nuisance_t_smooth / smooth)], format="csr")
    ridge = 1e-10 * dg.mean()
    cov = None
    if not use_schur:
        XtX = XtX_s.toarray()
        A = XtX + (Pt.multiply(pen[:, None])).toarray() + ridge * np.eye(P_tot)
        cf = sla.cho_factor(A, lower=True)
        sol = sla.cho_solve(cf, Xtr)
        rss = rss0 - 2 * sol @ Xtr + sol @ XtX @ sol
        sigma2 = max(rss, 0.0) / max((n_equiv or N) - P_tot, 1)
        if compute_cov and P_tot <= 6000:
            cov = sla.cho_solve(cf, np.eye(P_tot)) * sigma2
            scale = np.ones(P_tot); scale[:n_tide] = R0 / G0; scale[n_tide:] = R0
            cov = cov * np.outer(scale, scale)
    else:
        # Schur complement: eliminate the nuisance block (banded in time once ordered time-major) with a banded Cholesky, solve the small tide block densely
        nq = 2 * H + 1; sz = nuis_basis.size
        perm = (np.arange(nq)[None, :, None] * nd * sz + np.arange(nd)[:, None, None] * sz + np.arange(sz)[None, None, :]).ravel()    # time-major order (k, q, s) -> q-major column index
        A = (XtX_s + sp.diags(pen) @ Pt + ridge * sp.identity(P_tot, format="csr")).tocsr()
        At = A[:n_tide, :n_tide].toarray()
        Ann = A[n_tide:, n_tide:][perm][:, perm].tocoo()
        low = Ann.row >= Ann.col
        ii, jj, vv = Ann.row[low], Ann.col[low], Ann.data[low]
        bw = int((ii - jj).max())
        ab = np.zeros((bw + 1, nq * nd * sz)); ab[ii - jj, jj] = vv
        cb = sla.cholesky_banded(ab, lower=True)
        Atn = A[:n_tide, n_tide:][:, perm].toarray()                                   # (n_tide, n_nuis) time-major
        Z = sla.cho_solve_banded((cb, True), Atn.T)                                    # A_nn^-1 A_nt
        wv = sla.cho_solve_banded((cb, True), Xtr[n_tide:][perm])
        S = At - Atn @ Z
        cfS = sla.cho_factor(S, lower=True)
        sol_t = sla.cho_solve(cfS, Xtr[:n_tide] - Atn @ wv)
        sol_n = np.empty(n_nuis); sol_n[perm] = wv - Z @ sol_t
        sol = np.r_[sol_t, sol_n]
        rss = rss0 - 2 * sol @ Xtr + sol @ (XtX_s @ sol)
        sigma2 = max(rss, 0.0) / max((n_equiv or N) - P_tot, 1)
        if compute_cov:
            cov = sla.cho_solve(cfS, np.eye(n_tide)) * sigma2 * (R0 / G0) ** 2           # tide block only
    beta = (sol[:n_tide] * (R0 / G0)).reshape(K, 2, 2, basis.size)
    gamma = (sol[n_tide:] * R0).reshape((2 * H + 1, nd, nuis_basis.size) if daily else (2 * H + 1, nuis_basis.size)) if H >= 0 else np.zeros((0, 0))
    uxy = np.unique(np.round(np.column_stack([x, y]), 3), axis=0)
    return TideFit(names=names, omegas=omegas, basis=basis, beta=beta, nuis_basis=nuis_basis, nuis_harmonics=H, gamma=gamma, support_xy=uxy,
                   support_radius=float(support_radius or 1.5 * spacing), n_rows=int(n_equiv or N), resid_rms=float(np.sqrt(sigma2)) * R0, cov=cov,
                   nuis_t_spacing=float(nuisance_t_spacing) if daily else None, nuis_t0=nuis_t0,
                   info=dict(R0=R0, G0=G0, lam_tide=float(lam_t), n_dropped=n_drop_shallow, n_params=P_tot, solver='schur' if use_schur else 'dense'))


# ---------------------------------------------------------------------------------------------------------------- rows from a cube
def time_derivative(T, dt, order=4):
    """dT/dt of a (time, ...) array by centred differences, NaN on the first / last 1 (order 2) or 2 (order 4) steps."""
    out = np.full(T.shape, np.nan)
    if order == 2:
        out[1:-1] = (T[2:] - T[:-2]) / (2 * dt)
    elif order == 4:
        out[2:-2] = (-T[4:] + 8 * T[3:-1] - 8 * T[1:-3] + T[:-4]) / (12 * dt)
    else:
        raise ValueError("order must be 2 or 4")
    return out


def block_mean(a, f, min_frac=0.5):
    """Mean over f x f spatial blocks of a (time, ny, nx) array, ignoring NaN; blocks with fewer than `min_frac` valid pixels are NaN. Trailing cells that do not fill a block are dropped."""
    nt, ny, nx = a.shape
    ny2, nx2 = ny // f, nx // f
    blk = a[:, :ny2 * f, :nx2 * f].reshape(nt, ny2, f, nx2, f)
    cnt = np.isfinite(blk).sum(axis=(2, 4))
    s = np.nansum(blk, axis=(2, 4))
    with np.errstate(invalid="ignore", divide="ignore"):
        m = s / cnt
    return np.where(cnt >= min_frac * f * f, m, np.nan)


def pool_rows(Tt, Tx, Ty, x1, y1, t1, f, depth=None, min_frac=0.5):
    """Block-pool the EQUATIONS without losing the within-block gradient structure (use instead of `rows_from_cube(..., f)` for coarse blocks).

    Averaging dT/dt and grad T over a block and regressing the means throws away most of the information (opposite-signed gradients cancel in the mean, while the tide acts on every
    pixel). The normal equations of all pixels of a block only need the moment matrix A = sum_i [Tx, Ty, 1, Tt]_i [Tx, Ty, 1, Tt]_i^T (the spline basis is constant over the block),
    so each block and time keeps A, factored by an eigen-decomposition as A = sum_j q_j q_j^T; the pseudo-rows q_j = [Cx, Cy, Cn, Ct] stand in for the pixels exactly (same normal
    equations, same residual sum of squares). Returns the arrays of `rows_from_cube` (x, y, t, Tt, Tx, Ty, depth) with Tx, Ty, Tt = Cx, Cy, Ct, plus `Cn` and `n_equiv` (number of
    pixel equations); call `fit_tides(..., Cn=out['Cn'], n_equiv=out['n_equiv'])`. Blocks with fewer than `min_frac` valid pixels are skipped.
    """
    nt, ny, nx = Tt.shape
    ny2, nx2 = ny // f, nx // f

    def blk(a):
        return a[:, :ny2 * f, :nx2 * f].reshape(nt, ny2, f, nx2, f)
    vx, vy, vt = blk(Tx), blk(Ty), blk(Tt)
    ok = np.isfinite(vx) & np.isfinite(vy) & np.isfinite(vt)
    v = [np.where(ok, vx, 0.0), np.where(ok, vy, 0.0), ok.astype(float), np.where(ok, vt, 0.0)]
    n = ok.sum(axis=(2, 4))
    A = np.empty(n.shape + (4, 4))
    for i in range(4):
        for j in range(i, 4):
            A[..., i, j] = A[..., j, i] = (v[i] * v[j]).sum(axis=(2, 4))
    good = n >= min_frac * f * f
    lam, V = np.linalg.eigh(A[good])                                           # (nb, 4), (nb, 4, 4)
    q = np.sqrt(np.maximum(lam, 0.0))[:, :, None] * np.transpose(V, (0, 2, 1))    # q[b, j, :] = sqrt(lam_j) V[b, :, j]
    keep = np.abs(q).max(axis=2) > 1e-9 * np.abs(q).max()
    ti, yi, xi = np.nonzero(good)
    x1, y1 = np.asarray(x1, float), np.asarray(y1, float)
    if x1.ndim == 2:                                  # 2-D pixel positions (ny, nx), e.g. a lon / lat grid in a local azimuthal frame
        xc = x1[:ny2 * f, :nx2 * f].reshape(ny2, f, nx2, f).mean(axis=(1, 3))[yi, xi]
        yc = y1[:ny2 * f, :nx2 * f].reshape(ny2, f, nx2, f).mean(axis=(1, 3))[yi, xi]
    else:
        xc = x1[:nx2 * f].reshape(nx2, f).mean(1)[xi]
        yc = y1[:ny2 * f].reshape(ny2, f).mean(1)[yi]
    dd = None
    if depth is not None:
        dd = block_mean(np.asarray(depth, float)[None], f, 0.0)[0][yi, xi]
    jj = np.nonzero(keep)
    bidx = jj[0]
    out = dict(x=xc[bidx], y=yc[bidx], t=np.asarray(t1, float)[ti][bidx], Tx=q[jj[0], jj[1], 0], Ty=q[jj[0], jj[1], 1], Cn=q[jj[0], jj[1], 2], Tt=q[jj[0], jj[1], 3],
               depth=None if dd is None else dd[bidx], n_equiv=int(n[good].sum()))
    return out


def rows_from_cube(Tt, Tx, Ty, x1, y1, t1, f=1, depth=None, min_frac=0.5):
    """Rows (x, y, t, Tt, Tx, Ty, depth) from (time, y, x) products, optionally block-averaged by `f` (positions = block centres). The products must
    already be centred differences of the data (e.g. `tiling.grid_gradients_metric` and `time_derivative`)."""
    if f > 1:
        Tt, Tx, Ty = (block_mean(a, f, min_frac) for a in (Tt, Tx, Ty))
        ny2, nx2 = Tt.shape[1:]
        x1 = np.asarray(x1)[:nx2 * f].reshape(nx2, f).mean(1)
        y1 = np.asarray(y1)[:ny2 * f].reshape(ny2, f).mean(1)
        if depth is not None:
            depth = block_mean(np.asarray(depth)[None], f, 0.0)[0]
    nt, ny, nx = Tt.shape
    tt, yy, xx = np.meshgrid(np.asarray(t1, float), np.asarray(y1, float), np.asarray(x1, float), indexing='ij')
    dd = None if depth is None else np.broadcast_to(np.asarray(depth, float), (nt, ny, nx)).ravel()
    ok = np.isfinite(Tt) & np.isfinite(Tx) & np.isfinite(Ty)
    o = ok.ravel()
    return dict(x=xx.ravel()[o], y=yy.ravel()[o], t=tt.ravel()[o], Tt=Tt.ravel()[o], Tx=Tx.ravel()[o], Ty=Ty.ravel()[o], depth=None if dd is None else dd[o])

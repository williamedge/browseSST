"""The per-tile chain, as run by one HPC job, and its hyperparameter fit.

    cube + TileSpec --extract_tile--> TileData
      stage 1: latent-T Vecchia GP (two components) on the observed pixels      -> T, dT/dt at EVERY grid point of the block
      gradients: 4th-order differences of the latent T along local east / north (ellipsoid metric factors)
      tide hook: z = dT/dt + u_bt . grad T with the regional tide of `regional.TideModel` (optional; zero where the tide fit has no support)
      stage 2: velocity GP (default prior: Helmholtz, Matern, + Matern S + diurnal low-rank S) on z with design rows (-Tx, -Ty, 1) -> eddy u, v, S (mean, variance)
      total velocity = eddy + tide

Hyperparameters are SHARED and held fixed across tiles (exp 26: estimated once and applied everywhere); `Hyper` is their JSON container. `fit_hyper` estimates them on one
sample tile (stage 1 by the T-GP marginal likelihood, stage 2 by the Vecchia likelihood of the velocity model after removing the tide, as exp 32), and `pool_hypers` combines
the fits of several sample tiles (median in log space).

The velocity prior is a small interface (`VelocityPrior`) so that a different covariance (the simplified browseGP model: semidiurnal divergent wave etc.) can replace the default:
it supplies the packed covariance vector, the covariance function for the Vecchia operator, the low-rank terms and the free parameters / bounds that `fit_hyper` varies.

Stage-2 noise: `Hyper.stage2['noise']` (scaled units, the equation-error variance; 2.17 in the 0.03 K synthetic of exp 22/26) is used as is (`noise_mode='fixed'`), or fitted by
maximum likelihood per tile (`'mle'`). The multiple-imputation (Matheron) propagated variance of exps 18/22 is NOT in this chain: the variances returned are those of the stage-2
Vecchia block only (they neglect the uncertainty of the stage-1 products and of the tide).
"""
import dataclasses
import json
import pathlib
import time

import numpy as np
import xarray as xr

from . import kernels as K
from . import tgp as G
from .fit import T_KERNEL_SLOTS, fit, to_potential_params
from .model import LowRankTerm, SpaceTimeGP, diurnal_harmonics, diurnal_harmonics_daily
from .tiling import TileData, Blender, read_manifest, tile_weights

H = 3600.0
DAY = 86400.0
ZS = 1e-6                                   # data scaling of the stage-2 equation (z = dT/dt / ZS, design = gradT / ZS)

STAGE1_KEYS = ("sigma1", "l1", "tau1", "sigma2", "l2", "tau2", "noise", "lr:mean", "lr:diurnal")
STAGE1_BOUNDS = {"l1": (10e3, 500e3), "tau1": (12 * H, 1e7), "l2": (3e3, 10e3), "tau2": (1800.0, 12 * H), "noise": (1e-7, 1.0), "lr:mean": (1e-3, 1e2), "lr:diurnal": (1e-4, 1e1)}
STAGE1_CENTRE = dict(sigma1=0.5, l1=40e3, tau1=2 * DAY, sigma2=0.2, l2=6e3, tau2=6 * H, noise=2 * 0.01 ** 2, **{"lr:mean": 1.0, "lr:diurnal": 0.1})

# Placeholders: the exp-21 5/2:5/2 fit to the 0.01 K synthetic fronts field (stage 1) and the generic start 'fit A' of exp 24 (stage 2). Replace by fits on real sample tiles.
DEFAULT_STAGE1 = dict(sigma1=0.2788, l1=11205.0, tau1=134178.0, sigma2=0.1293, l2=3000.0, tau2=43200.0, noise=1.0e-4, **{"lr:mean": 0.5223, "lr:diurnal": 0.002})
DEFAULT_STAGE2 = dict(gamma=0.10, f_delta=0.2, l_psi=30e3, l_phi=20e3, tau0_psi=2 * DAY, tau0_phi=1 * DAY, sigma_S=0.3, l_S=20e3, tau0_S=1 * DAY, noise=2.17, **{"lr:diurnal": 0.5})


# ------------------------------------------------------------------------------------------------------------------------------ hyperparameters
@dataclasses.dataclass
class Hyper:
    stage1: dict
    stage2: dict
    meta: dict = dataclasses.field(default_factory=dict)

    @staticmethod
    def default():
        return Hyper(dict(DEFAULT_STAGE1), dict(DEFAULT_STAGE2), dict(source="placeholder: exp-21 fronts fit (stage 1) and exp-24 'fit A' (stage 2); replace by fits on real tiles"))

    def save(self, path):
        pathlib.Path(path).write_text(json.dumps(dataclasses.asdict(self), indent=1))

    @staticmethod
    def load(path):
        d = json.loads(pathlib.Path(path).read_text())
        return Hyper(d["stage1"], d["stage2"], d.get("meta", {}))


def pool_hypers(hypers, keys1=STAGE1_KEYS, keys2=None):
    """Median (log space) of several fitted `Hyper`s, key by key; the pooled meta lists the inputs' spread (max / min ratio)."""
    keys2 = keys2 or [k for k in hypers[0].stage2 if k != "f_delta"] + (["f_delta"] if "f_delta" in hypers[0].stage2 else [])
    med = lambda vals: float(np.exp(np.median(np.log(np.asarray(vals, float)))))
    s1 = {k: med([h.stage1[k] for h in hypers]) for k in keys1}
    s2 = {k: med([h.stage2[k] for h in hypers]) for k in keys2}
    spread = {k: float(max(h.stage1[k] for h in hypers) / min(h.stage1[k] for h in hypers)) for k in keys1}
    spread.update({k: float(max(h.stage2[k] for h in hypers) / min(h.stage2[k] for h in hypers)) for k in keys2})
    return Hyper(s1, s2, dict(source="pooled", n=len(hypers), spread_max_over_min=spread))


# ------------------------------------------------------------------------------------------------------------------------------ velocity prior
class VelocityPrior:
    """Interface of the stage-2 covariance model (default: `HelmholtzMaternPrior`). A replacement must provide the same four members."""
    name = "abstract"
    free = ()                                   # names (keys of Hyper.stage2) varied by `fit_hyper`
    bounds = {}
    covfunc = None                              # covariance function of the Vecchia operator (None = `kernels.transport_obs_covfunc`)

    def cov_params(self, h2):                   # packed parameter vector for `covfunc`
        raise NotImplementedError

    def lowrank(self, h2, td=None):             # list of LowRankTerm (`td`: the TileData, for priors that depend on the tile's time span)
        raise NotImplementedError


class HelmholtzMaternPrior(VelocityPrior):
    """Matern Helmholtz velocity (rotational psi + divergent phi potentials, space-time) + Matern S, in the (gamma, f_delta) amplitude parameterisation, + diurnal low-rank S."""
    name = "helmholtz_matern"
    free = ("gamma", "f_delta", "l_psi", "l_phi", "tau0_psi", "tau0_phi", "sigma_S", "l_S", "tau0_S", "lr:diurnal")
    bounds = {"lr:diurnal": (1e-3, 1e3), "sigma_S": (1e-2, 1e2), "tau0_psi": (3600.0, 3e6), "tau0_phi": (3600.0, 3e6), "tau0_S": (1800.0, 3e6)}   # S bounds widened for real data (exp 36)

    def cov_params(self, h2):
        p = to_potential_params(h2)
        return K.pack_params(p["sigma_psi"], p["l_psi"], p["sigma_phi"], p["l_phi"], p["tau0_psi"], p["tau0_phi"], p["sigma_S"], p["l_S"], p["tau0_S"])

    def lowrank(self, h2, td=None):
        return [LowRankTerm("diurnal", {"S": diurnal_harmonics(4)}, scale=float(h2["lr:diurnal"]))]


class StructuredSPrior(HelmholtzMaternPrior):
    """The same velocity prior with a STRUCTURED source term for real data (exp 37): S = (Matern space-time field with free amplitude / scales, not a tiny nuisance) + a spatially uniform diurnal
    low-rank term whose harmonics may change amplitude from day to day (`daily=True`, linear hats of 24 h spacing over the tile's time span) or repeat the same cycle every day (`daily=False`),
    with the observation noise FREE above a small floor (the S terms, not the noise, are meant to carry the non-advective dT/dt)."""
    name = "helmholtz_matern_structured_S"
    free = HelmholtzMaternPrior.free + ("noise",)
    bounds = dict(HelmholtzMaternPrior.bounds, noise=(1.0, 1e4))

    def __init__(self, daily=True, n_harm=4, spacing=DAY):
        self.daily, self.n_harm, self.spacing = daily, n_harm, spacing

    def lowrank(self, h2, td=None):
        if self.daily and td is not None:
            t = np.asarray(td.t, float)
            k0 = np.floor(t.min() / self.spacing) * self.spacing; k1 = np.ceil(t.max() / self.spacing) * self.spacing
            knots = np.arange(k0, k1 + 0.5 * self.spacing, self.spacing)
            phi = diurnal_harmonics_daily(knots, self.n_harm, spacing=self.spacing)
        else:
            phi = diurnal_harmonics(self.n_harm)
        return [LowRankTerm("diurnal", {"S": phi}, scale=float(h2["lr:diurnal"]))]


class BrowseGPPrior(StructuredSPrior):
    """The velocity covariance of browseGP's in-situ model (browseGP/scripts/fit_multiscale_real.py: rotational + M2 divergent wave + M4 wave + fast + inertial blocks) as the (u, v) prior of the
    transport equation. `vel` holds browseGP's names (gamma_rot, ell_rot, tau_rot, gamma_div, ell_div, tau_div, gamma_m4, ell_m4, tau_m4, gamma_f, ell_f, tau_f, gamma_in, ell_in, tau_in; m/s, m, s);
    `from_results` reads the posterior medians of a browseGP `mcmc_results.json`. `blocks` switches blocks on / off ('rot', 'div', 'm4', 'fast', 'inertial'); `lat` sets the inertial frequency.

    Velocity parameters: FIXED at `vel` by default (exp 40 / 41: only the source term S and the equation noise are fitted). With `free_vel` (True = every block that is on, or a tuple of block names) the
    (gamma, ell, tau) of those blocks are fitted too, on the SST of the sample tiles (`fit_hyper`), with a log-normal prior of sd `prior_sd_log` (default 1 in log, i.e. a factor e) centred on the
    start values in `Hyper.stage2` (browseGP's posterior medians via `complete`); the fitted values travel in `Hyper.stage2` to the tile jobs, which read them in `cov_params`."""
    name = "browsegp"
    covfunc = K.browsegp_transport_covfunc
    S_FREE = ("sigma_S", "l_S", "tau0_S", "lr:diurnal", "noise")
    free = S_FREE
    bounds = {"lr:diurnal": (1e-3, 1e3), "sigma_S": (1e-2, 1e2), "tau0_S": (1800.0, 3e6), "noise": (1.0, 1e4)}
    slots = K.BG_SLOTS
    BLOCK_ORDER = ("rot", "div", "m4", "fast", "inertial")
    VEL_NAMES = dict(rot=("gamma_rot", "ell_rot", "tau_rot"), div=("gamma_div", "ell_div", "tau_div"), m4=("gamma_m4", "ell_m4", "tau_m4"), fast=("gamma_f", "ell_f", "tau_f"),
                     inertial=("gamma_in", "ell_in", "tau_in"))
    VEL_BOUNDS = dict(gamma_rot=(1e-3, 1.0), ell_rot=(5e3, 3e5), tau_rot=(3600.0, 3e6), gamma_div=(1e-3, 1.0), ell_div=(5e3, 3e5), tau_div=(3600.0, 3e6),
                      gamma_m4=(1e-3, 1.0), ell_m4=(3e3, 1e5), tau_m4=(3600.0, 3e6), gamma_f=(1e-3, 1.0), ell_f=(1e3, 6e4), tau_f=(600.0, 2e5), gamma_in=(1e-3, 1.0), ell_in=(3e3, 1e5),
                     tau_in=(3600.0, 3e6))
    VEL_DEFAULTS = dict(ell_m4=1e4, tau_m4=1e5, ell_f=2e3, tau_f=4.6e3, ell_in=2e4, tau_in=2e5)

    def __init__(self, vel, blocks=("rot", "div", "m4", "fast", "inertial"), lat=-14.0, daily=True, n_harm=4, spacing=DAY, free_vel=False, prior_sd_log=None):
        super().__init__(daily=daily, n_harm=n_harm, spacing=spacing)
        self.vel, self.blocks, self.lat = {**self.VEL_DEFAULTS, **dict(vel)}, tuple(blocks), float(lat)
        self.free_vel = tuple(self.blocks) if free_vel is True else (tuple(free_vel) if free_vel else ())
        bad = [b for b in self.free_vel if b not in self.blocks]
        if bad:
            raise ValueError(f"free_vel blocks {bad} are not switched on in blocks={self.blocks}")
        self.free = self.S_FREE + tuple(n for b in self.BLOCK_ORDER if b in self.free_vel for n in self.VEL_NAMES[b])
        self.bounds = {**BrowseGPPrior.bounds, **{n: self.VEL_BOUNDS[n] for n in self.free if n in self.VEL_BOUNDS}}
        self.prior_sd_log = dict(prior_sd_log or {})

    def complete(self, h2):
        """`h2` (a `Hyper.stage2` dict) with the start values of the velocity parameters that are to be fitted added (existing entries are kept)."""
        h2 = dict(h2)
        for n in self.free:
            if n in self.vel and n not in h2:
                h2[n] = float(self.vel[n])
        return h2

    @staticmethod
    def from_results(path, **kw):
        """From a browseGP `mcmc_results.json` (posterior medians) or a plain `{name: value}` json such as `hpc/browsegp_vel.json` (keys starting with '_' are comments)."""
        d = json.loads(pathlib.Path(path).read_text())
        if "posterior_summary" in d:
            vel = {k: float(v["median"]) for k, v in d["posterior_summary"].items()}
        else:
            vel = {k: float(v) for k, v in d.items() if not k.startswith("_")}
        return BrowseGPPrior(vel, **kw)

    def cov_params_blocks(self, h2, only):
        """`cov_params` with block weights (1 for the blocks in `only`, 0 for the others) applied to the prediction-data cross covariance: a prediction made with it is the posterior-mean contribution of
        those blocks to the predicted velocity (the data covariance stays the full model), and the contributions of a partition of the blocks sum to the full prediction."""
        return np.r_[self.cov_params(h2), [1.0 if b in only else 0.0 for b in self.BLOCK_ORDER]]

    def cov_params(self, h2):
        from .tides import omega_of
        on = self.blocks
        v = {**self.vel, **{k: h2[k] for k in self.vel if k in h2}}      # fitted values (Hyper.stage2) override browseGP's medians
        f_rad = 2 * 7.2921159e-5 * abs(np.sin(np.deg2rad(self.lat)))
        g = lambda key, blk: v[key] if blk in on else 0.0
        return K.pack_browsegp(g("gamma_rot", "rot"), v["ell_rot"], v["tau_rot"], g("gamma_div", "div"), v["ell_div"], v["tau_div"], omega_of("M2"),
                               gamma_m4=g("gamma_m4", "m4"), ell_m4=v.get("ell_m4", 1e4), tau_m4=v.get("tau_m4", 1e5),
                               gamma_f=g("gamma_f", "fast"), ell_f=v.get("ell_f", 2e3), tau_f=v.get("tau_f", 4.6e3),
                               gamma_in=g("gamma_in", "inertial"), ell_in=v.get("ell_in", 2e4), tau_in=v.get("tau_in", 2e5), omega_in=f_rad, pol=1.0,
                               sigma_S=h2["sigma_S"], l_S=h2["l_S"], tau0_S=h2["tau0_S"])


@dataclasses.dataclass
class PipelineConfig:
    nnum: int = 100                    # Vecchia conditioning set, both stages
    nnum_pred: int = 200               # prediction neighbours
    Lx: float = 20e3                   # ordering / neighbour-search scales (m, s)
    Lt: float = 24 * H
    order: int = 4                     # finite-difference order of the gradients
    noise_mode: str = "fixed"          # stage-2 noise: 'fixed' (Hyper.stage2['noise']) or 'mle' (scalar maximum likelihood on the tile)
    tide_mask: str = "supported"       # 'supported': tide 0 where the regional fit has no support; 'none': always apply the (extrapolated) tide
    min_obs: int = 2000                # skip a tile with fewer observed pixels
    tt_sd_factor: float = 3.0          # drop stage-2 rows whose posterior sd of dT/dt exceeds this multiple of the median (large gaps); 0 = keep all
    stage1_maxiter: int = 0            # >0: refit stage 1 on the tile (iterations); the shared values are the start / prior centre
    prior: VelocityPrior = dataclasses.field(default_factory=HelmholtzMaternPrior)


# ------------------------------------------------------------------------------------------------------------------------------ stages
def t_terms(mean_scale, diurnal_scale):
    return [G.TLowRankTerm("mean", G.poly_mean(), mean_scale), G.TLowRankTerm("diurnal", G.diurnal_T(3), diurnal_scale)]


def build_stage1(xo, yo, to, To, h1, cfg):
    p0 = G.pack_params_T_res(h1["sigma1"], h1["l1"], h1["tau1"], h1["sigma2"], h1["l2"], h1["tau2"], 2.5, 2.5, 2.5)
    return G.TGP(G.build_T_points(xo, yo, to), np.asarray(To, float), p0, h1["noise"], lowrank=t_terms(h1["lr:mean"], h1["lr:diurnal"]), nnum=cfg.nnum, Lx=cfg.Lx, Lt=cfg.Lt,
                 covfunc=G.T_covfunc_res)


def fit_stage1(gp, h1, maxiter):
    """L-BFGS on the T-GP marginal likelihood (log-normal priors, sd 1.5 about `STAGE1_CENTRE`), started at h1. Returns the new stage-1 dict."""
    free = list(STAGE1_KEYS)
    res = fit(gp, STAGE1_CENTRE, free, bounds=STAGE1_BOUNDS, prior_sd_log={n: 1.5 for n in free}, maxiter=maxiter, verbose=False, slots=T_KERNEL_SLOTS, start={k: h1[k] for k in free})
    return {k: float(res.params[k]) for k in free}, res


def stage1_products(td, gp, cfg, nn=None):
    """Latent T and dT/dt posterior means (and the dT/dt variance) at all grid points of the block."""
    nn = gp.pred_neighbours(td.x, td.y, td.t, cfg.nnum_pred) if nn is None else nn
    Tm, _ = gp.predict("T", td.x, td.y, td.t, nn_array=nn)
    Ttm, vTt = gp.predict("Tt", td.x, td.y, td.t, nn_array=nn)
    return Tm, Ttm, vTt


def stage2_rows(td, Tm, Ttm, vTt, ub, vb, cfg):
    """z and the design rows (-Tx, -Ty, 1) at the grid points that carry usable products. Returns (idx, Tx, Ty, z) with Tx, Ty, z full length."""
    Tx, Ty = td.gradients(Tm, cfg.order)
    z = Ttm + Tx * ub + Ty * vb
    ok = np.isfinite(Tx) & np.isfinite(Ty) & np.isfinite(z)
    if cfg.tt_sd_factor and cfg.tt_sd_factor > 0:
        sd = np.sqrt(np.maximum(vTt, 0.0))
        ref = np.median(sd[ok & td.obs]) if (ok & td.obs).any() else np.median(sd[ok])
        ok &= sd <= cfg.tt_sd_factor * ref
    return np.flatnonzero(ok), Tx, Ty, z


def build_stage2(td, idx, Tx, Ty, z, h2, cfg, noise=None):
    pts = K.build_points(td.x[idx], td.y[idx], td.t[idx], Tx[idx] / ZS, Ty[idx] / ZS)
    prior = cfg.prior
    return SpaceTimeGP(pts, z[idx] / ZS, prior.cov_params(h2), h2["noise"] if noise is None else noise, lowrank=prior.lowrank(h2, td), nnum=cfg.nnum, Lx=cfg.Lx, Lt=cfg.Lt,
                       covfunc=prior.covfunc)


def fit_noise_mle(gp, lo=1e-2, hi=1e3):
    from scipy.optimize import minimize_scalar

    def nll(ln):
        gp.update(noise=float(np.exp(ln)))
        return -gp.loglik()
    r = minimize_scalar(nll, bounds=(np.log(lo), np.log(hi)), method="bounded", options=dict(xatol=0.05))
    gp.update(noise=float(np.exp(r.x)))
    return float(np.exp(r.x))


def _tide_for_tile(td, tide, cfg):
    n = td.x.size
    if tide is None:
        return np.zeros(n), np.zeros(n), np.zeros(n, bool)
    return tide.tile_velocity(td, mask=cfg.tide_mask)


# ------------------------------------------------------------------------------------------------------------------------------ the tile job
def run_tile(td, hyper, cfg=None, tide=None, log=print):
    """Run the chain on one block. Returns a dict of (nt, ny, nx) arrays plus 'info' (timings, counts, the stage-2 noise used), or None if the block has too few observations.

    Fields: T (latent SST, K, absolute), Tt (K/s), u_eddy, v_eddy (m/s), var_u, var_v (stage-2 block variance), S (K/day), var_S, u_tide, v_tide, u, v (= eddy + tide), tide_supported.
    """
    cfg = PipelineConfig() if cfg is None else cfg
    t_all = time.time(); tim = {}
    if td.n_obs < cfg.min_obs:
        if log:
            log(f"  tile {td.spec.id}: only {td.n_obs} observed pixels (< {cfg.min_obs}); skipped")
        return None
    sh = td.shape
    xo, yo, to, To = td.observed()
    t0 = time.time()
    gp1 = build_stage1(xo, yo, to, To, hyper.stage1, cfg)
    h1 = dict(hyper.stage1)
    if cfg.stage1_maxiter > 0:
        h1, _ = fit_stage1(gp1, h1, cfg.stage1_maxiter)
    tim["stage1_build_fit"] = time.time() - t0
    t0 = time.time()
    Tm, Ttm, vTt = stage1_products(td, gp1, cfg)
    tim["stage1_predict"] = time.time() - t0
    ub, vb, sup = _tide_for_tile(td, tide, cfg)
    idx, Tx, Ty, z = stage2_rows(td, Tm, Ttm, vTt, ub, vb, cfg)
    t0 = time.time()
    h2 = dict(hyper.stage2)
    gp2 = build_stage2(td, idx, Tx, Ty, z, h2, cfg)
    noise = h2["noise"]
    if cfg.noise_mode == "mle":
        noise = fit_noise_mle(gp2)
    nn2 = gp2.pred_neighbours(td.x, td.y, td.t, cfg.nnum_pred)
    tim["stage2_build"] = time.time() - t0
    t0 = time.time()
    u, vu = gp2.predict("u", td.x, td.y, td.t, nn_array=nn2)
    v, vv = gp2.predict("v", td.x, td.y, td.t, nn_array=nn2)
    S, vS = gp2.predict("S", td.x, td.y, td.t, nn_array=nn2)
    tim["stage2_predict"] = time.time() - t0
    r = lambda a: np.asarray(a, float).reshape(sh)
    out = dict(T=r(Tm + td.Tref), Tt=r(Ttm), u_eddy=r(u), v_eddy=r(v), var_u=r(vu), var_v=r(vv), S=r(S * ZS * DAY), var_S=r(vS * (ZS * DAY) ** 2),
               u_tide=r(ub), v_tide=r(vb), u=r(u + ub), v=r(v + vb), tide_supported=r(sup).astype(np.int8))
    out["info"] = dict(tile=td.spec.id, n_grid=int(td.x.size), n_obs=int(td.n_obs), n_stage2=int(idx.size), noise_stage2=float(noise), tide=tide is not None,
                       seconds=time.time() - t_all, timings=tim, stage1_used=h1, mean_coverage=float(td.coverage().mean()))
    if log:
        log(f"  tile {td.spec.id}: {td.n_obs} obs of {td.x.size} grid points, {idx.size} stage-2 rows, noise {noise:.2f}, {out['info']['seconds']:.0f}s " + ", ".join(f"{k} {v_:.0f}s" for k, v_ in tim.items()))
    return out


# ------------------------------------------------------------------------------------------------------------------------------ hyperparameter fit on a sample tile
def fit_hyper(td, hyper, cfg=None, tide=None, maxiter1=25, maxiter2=25, log=print):
    """Estimate the shared hyperparameters on one tile: stage 1 by the T-GP likelihood from `hyper.stage1`, then the stage-2 parameters (`cfg.prior.free`) by the Vecchia likelihood of the
    tide-removed response from `hyper.stage2` (the noise is fixed at its value, or fitted by ML first if `cfg.noise_mode == 'mle'`). Returns a new `Hyper` (log of the optimisation in meta)."""
    cfg = PipelineConfig() if cfg is None else cfg
    xo, yo, to, To = td.observed()
    t0 = time.time()
    gp1 = build_stage1(xo, yo, to, To, hyper.stage1, cfg)
    h1, res1 = fit_stage1(gp1, hyper.stage1, maxiter1)
    if log:
        log(f"  stage 1 fit: {res1.n_eval} evals, {time.time() - t0:.0f}s, loglik {res1.loglik:.1f}; eddy {h1['sigma1']:.3f} K {h1['l1'] / 1e3:.1f} km {h1['tau1'] / H:.1f} h | "
            f"fast {h1['sigma2']:.3f} K {h1['l2'] / 1e3:.1f} km {h1['tau2'] / H:.1f} h | noise sd {np.sqrt(h1['noise']):.4f} K")
    Tm, Ttm, vTt = stage1_products(td, gp1, cfg)
    ub, vb, _ = _tide_for_tile(td, tide, cfg)
    idx, Tx, Ty, z = stage2_rows(td, Tm, Ttm, vTt, ub, vb, cfg)
    h2 = dict(hyper.stage2)
    if hasattr(cfg.prior, "complete"):
        h2 = cfg.prior.complete(h2)            # priors with fitted velocity parameters start them from their own defaults (browseGP's medians)
    gp2 = build_stage2(td, idx, Tx, Ty, z, h2, cfg)
    if cfg.noise_mode == "mle":
        h2["noise"] = fit_noise_mle(gp2)
        gp2 = build_stage2(td, idx, Tx, Ty, z, h2, cfg)
    t0 = time.time()
    free = list(cfg.prior.free)
    sd = {n: 1.0 for n in free}
    sd.update(getattr(cfg.prior, "prior_sd_log", {}))
    res2 = fit(gp2, h2, free, bounds=cfg.prior.bounds, prior_sd_log=sd, maxiter=maxiter2, verbose=False, slots=getattr(cfg.prior, "slots", None))
    new2 = dict(h2)
    new2.update({k: float(res2.params[k]) for k in free})
    if log:
        log(f"  stage 2 fit: {res2.n_eval} evals, {time.time() - t0:.0f}s, success={res2.success}, loglik {res2.loglik:.1f}: " + ", ".join(f"{k} {new2[k]:.3g}" for k in free))
    meta = dict(source="fit", prior=getattr(cfg.prior, "name", ""), free2=free, tile=td.spec.id, n_obs=int(td.n_obs), n_stage2=int(idx.size), tide=tide is not None, converged1=bool(res1.success), converged2=bool(res2.success),
                n_eval1=int(res1.n_eval), n_eval2=int(res2.n_eval), loglik1=float(res1.loglik), loglik2=float(res2.loglik))
    return Hyper(h1, new2, meta)


# ------------------------------------------------------------------------------------------------------------------------------ tile files and merge
TILE_FIELDS = ("T", "Tt", "u_eddy", "v_eddy", "var_u", "var_v", "S", "var_S", "u_tide", "v_tide", "u", "v", "tide_supported")
_UNITS = dict(T="K", Tt="K s-1", u_eddy="m s-1", v_eddy="m s-1", var_u="m2 s-2", var_v="m2 s-2", S="K day-1", var_S="K2 day-2", u_tide="m s-1", v_tide="m s-1", u="m s-1", v="m s-1")


def write_tile(path, td, out):
    """One NetCDF per tile: the whole buffered block (the merge needs the buffer for the blend), with the tile spec in the attributes."""
    data = {k: (("time", "lat", "lon"), out[k].astype(np.float32 if k != "tide_supported" else np.int8), {"units": _UNITS.get(k, "1")}) for k in TILE_FIELDS}
    ds = xr.Dataset(data, coords={"time": td.time, "lat": td.lat, "lon": td.lon})
    ds.attrs.update({k: int(v) for k, v in dataclasses.asdict(td.spec).items() if k != "id"}, tile_id=td.spec.id, Tref=float(td.Tref), info=json.dumps(out["info"], default=float))
    enc = {k: dict(zlib=True, complevel=4) for k in ds.data_vars}
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path, encoding=enc)


def merge_tiles(manifest, tiles_dir, cube, out_path, fields=("T", "u", "v", "u_eddy", "v_eddy", "u_tide", "v_tide", "var_u", "var_v", "S"), log=print):
    """Blend the tile files in `tiles_dir` (names tile_<id>.nc) into the cube's grid with the partition-of-unity weights; tiles that are missing (skipped) leave NaN holes where they alone cover.
    Returns the merged Dataset (also written to `out_path` if given)."""
    meta, specs = read_manifest(manifest) if not isinstance(manifest, tuple) else manifest
    nt, ny, nx = (cube.sizes["time"], cube.sizes["lat"], cube.sizes["lon"])
    bl = Blender((nt, ny, nx), n_fields=len(fields))
    n_used = 0
    for s in specs:
        p = pathlib.Path(tiles_dir) / f"tile_{s.id}.nc"
        if not p.exists():
            if log:
                log(f"  merge: tile {s.id} missing")
            continue
        with xr.open_dataset(p) as d:
            bl.add(s, *[d[f].values for f in fields])
        n_used += 1
    res = bl.result()
    ds = xr.Dataset({f: (("time", "lat", "lon"), res[i].astype(np.float32), {"units": _UNITS.get(f, "1")}) for i, f in enumerate(fields)},
                    coords={"time": cube["time"].values, "lat": cube["lat"].values, "lon": cube["lon"].values})
    ds.attrs.update(n_tiles=n_used, n_tiles_expected=len(specs))
    if out_path:
        ds.to_netcdf(out_path, encoding={k: dict(zlib=True, complevel=4) for k in ds.data_vars})
    return ds

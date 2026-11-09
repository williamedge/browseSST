"""
Experiment 12: do higher-order finite-difference products recover the skill lost to differencing error?

At the TRUE hyperparameters (oracle, so only the data differ), compares the exp-04 dataset processed with
  (a) 2nd-order products (the current pipeline) and (b) 4th-order products (5-point stencils in x, y and t),
against (c) data that satisfy the transport equation exactly (the ceiling, as in exp 07). Noise: sigma_T = 0.01 K on T,
so the 4th-order time difference has ~1.8x the noise variance. Noisy dTdt, clean gradients (phase A).
Run: python experiments/12_stencil_order.py [sigma_T,sigma_T,...] [clean,obs]   (gradient source: clean = from true T, obs = from noisy T)
"""
import sys, warnings
import numpy as np

from inversion_sst_gp.spacetime import kernels as K
from inversion_sst_gp.spacetime import inference as I
from inversion_sst_gp.spacetime import synthetic as S
from inversion_sst_gp.spacetime.model import LowRankTerm, SpaceTimeGP, diurnal_harmonics

warnings.filterwarnings("ignore")
ZS = 1e-6


def with_products(ds, order):
    out = ds.copy(deep=True)
    dims = ('time', 'y', 'x')
    for suffix, T in (('', ds['T_obs'].values), ('_clean', ds['T_true'].values)):
        a, b, c = S.derivative_products(T, float(ds.attrs['dx']), order=order)
        out['dTdt' + suffix], out['dTdx' + suffix], out['dTdy' + suffix] = (dims, a), (dims, b), (dims, c)
    return out


def evaluate(label, ds, dsrc, p, common=None, grad='clean'):
    obs = I.make_observations(ds, grad_source=grad, dTdt_source=dsrc, zscale=ZS)
    idx = obs.index
    tr = {'u': ds['u_true'].values[idx], 'v': ds['v_true'].values[idx], 'S': ds['S_true'].values[idx] / ZS}
    noise = float(np.var(obs.z - (obs.pts[:, 3] * tr['u'] + obs.pts[:, 4] * tr['v'] + tr['S'])))
    gp = SpaceTimeGP(obs.pts, obs.z, p, noise, lowrank=[LowRankTerm('diurnal', {'S': diurnal_harmonics(4)}, scale=0.5)], nnum=100)
    x, y, t = obs.pts[:, 0], obs.pts[:, 1], obs.pts[:, 2]
    sel = np.ones(len(obs.z), bool) if common is None else common[idx]       # score on the common interior only
    out = []
    for f in 'uvS':
        m, v = gp.predict(f, x, y, t, nnum_pred=200)
        e = (m - tr[f])[sel]; rmse = np.sqrt(np.mean(e ** 2))
        out.append(f"{f}: rmse {rmse:.4f} skill {1 - rmse**2 / tr[f][sel].var():+.3f} cover {np.mean(np.abs(e) < 1.96*np.sqrt(v[sel])):.2f}")
    print(f"  {label:<34s} N={len(obs.z):5d} scored {int(sel.sum()):5d} resid var {noise:5.2f} | " + " | ".join(out), flush=True)


def main(sigmas=(0.01,), grads=('clean',)):
    cfg = S.SyntheticConfig(nx=14, ny=14, n_hours=48, n_modes=400)
    ds0 = S.generate(cfg)
    tp = cfg.true_params(ZS)
    p = K.pack_params(tp['sigma_psi'], tp['l_psi'], tp['sigma_phi'], tp['l_phi'], tp['tau0_psi'],
                      tp['tau0_phi'], tp['sigma_S'], tp['l_S'], tp['tau0_S'])
    for sigma_T in sigmas:
        ds = S.reobserve(ds0, sigma_T, seed=cfg.seed + 7) if sigma_T != cfg.sigma_T else ds0
        ds2, ds4 = with_products(ds, 2), with_products(ds, 4)
        # score every variant on the points where the 4th-order products exist (the common interior)
        common = np.isfinite(ds4['dTdt'].values) & np.isfinite(ds4['dTdx'].values) & np.isfinite(ds4['dTdy'].values)
        for grad in grads:
            print(f"\nsigma_T = {sigma_T} K, {grad} gradients; true hyperparameters, oracle iid noise; scored on {int(common.sum())} common interior points", flush=True)
            evaluate("(a) 2nd-order products", ds2, 'obs', p, common, grad)
            evaluate("(b) 4th-order products", ds4, 'obs', p, common, grad)
            evaluate("(c) exact-equation ceiling, 2nd-order grads", ds2, 'equation', p, common, grad)
            evaluate("(c') exact-equation ceiling, 4th-order grads", ds4, 'equation', p, common, grad)


if __name__ == "__main__":
    a = sys.argv[1:]
    # usage: 12_stencil_order.py [sigma_T,...] [grad,...]   e.g.  0.01,0.03,0.1 clean,obs
    main(sigmas=tuple(float(v) for v in a[0].split(',')) if a else (0.01,),
         grads=tuple(a[1].split(',')) if len(a) > 1 else ('clean',))

"""
Experiment 03: does the guessed neighbour metric (Lx, Lt) change how many neighbours are needed?

Same case as experiment 02 (noisy dTdt, clean gradients, true hyperparameters, oracle noise).
Fixed nnum = nnum_pred = 100; reports log-likelihood deficit and posterior-mean agreement with dense.
Run: python experiments/03_neighbour_metric.py
"""
import time, warnings, importlib.util, pathlib
import numpy as np

from inversion_sst_gp.spacetime import kernels as K
from inversion_sst_gp.spacetime import inference as I
from inversion_sst_gp.spacetime import synthetic as S

def _load(name):
    spec = importlib.util.spec_from_file_location(name, pathlib.Path(__file__).with_name(name + ".py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
e01 = _load("01_oracle_hyperparams"); e02 = _load("02_nnum_sweep")

warnings.filterwarnings("ignore")
ZS = 1e-6
H = 3600.0


def main(nx=12, n_hours=24, nnum=100, metrics=None):
    cfg = S.SyntheticConfig(nx=nx, ny=nx, n_hours=n_hours, n_modes=400, diurnal_shape='cosine', diurnal_amp=5e-7)
    ds = S.generate(cfg)
    tp = cfg.true_params(ZS)
    p = K.pack_params(tp['sigma_psi'], tp['l_psi'], tp['sigma_phi'], tp['l_phi'], tp['tau0_psi'],
                      tp['tau0_phi'], tp['sigma_S'], tp['l_S'], tp['tau0_S'])
    obs = I.make_observations(ds, grad_source='clean', dTdt_source='obs', zscale=ZS)
    idx = obs.index
    u_t, v_t, S_t = (ds[k].values[idx] for k in ('u_true', 'v_true', 'S_true'))
    noise = float(np.var(obs.z - (obs.pts[:, 3] * u_t + obs.pts[:, 4] * v_t + S_t / ZS)))
    x, y, t = obs.pts[:, 0], obs.pts[:, 1], obs.pts[:, 2]
    ll_dense = e02.dense_loglik(obs.pts, obs.z, p, noise)
    mu = {f: e01.dense_posterior(obs.pts, obs.z, p, noise, K.build_pred_points(x, y, t, f))[0] for f in 'uv'}
    print(f"dense loglik {ll_dense:.2f}; nnum=nnum_pred={nnum}")
    print("   Lx     Lt     dLL     rms diff u  rms diff v   rmse u  rmse v")
    if metrics is None:
        metrics = ((10e3, 3 * H), (20e3, 1 * H), (20e3, 3 * H), (20e3, 6 * H), (20e3, 12 * H),
                   (20e3, 24 * H), (40e3, 6 * H), (40e3, 24 * H), (80e3, 24 * H))
    for Lx, Lt in metrics:
        op = I.build_operator(obs.pts, p, noise, nnum=nnum, Lx=Lx, Lt=Lt)
        dll = I.loglik(op, obs.z) - ll_dense
        r = []
        for f, tr in (('u', u_t), ('v', v_t)):
            m, _ = I.predict(op, obs.z, f, x, y, t, nnum_pred=nnum)
            r += [np.sqrt(np.mean((m - mu[f]) ** 2)), np.sqrt(np.mean((m - tr) ** 2))]
        print(f"{Lx/1e3:5.0f}km {Lt/H:4.0f}h  {dll:8.2f}   {r[0]:.4f}     {r[2]:.4f}     {r[1]:.4f}  {r[3]:.4f}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:   # e.g. 30 20e3:24 20e3:48 -> nnum, then Lx[m]:Lt[h] pairs
        main(nnum=int(sys.argv[1]), metrics=[(float(a.split(':')[0]), float(a.split(':')[1]) * H) for a in sys.argv[2:]])
    else:
        main()

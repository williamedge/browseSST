"""FastVecchiaOperator gives the same factor / logdet / solves as gpvecchia's VecchiaOperator.
Run: python tests/test_fastop.py"""
import time
import numpy as np
from gpvecchia.vecchia_operator import VecchiaOperator
from inversion_sst_gp.spacetime import kernels as K
from inversion_sst_gp.spacetime import inference as I
from inversion_sst_gp.spacetime.fastop import FastVecchiaOperator

DAY = 86400.0


def test_fastop_matches_base():
    rng = np.random.default_rng(5)
    N = 400
    x, y, t = rng.uniform(0, 60e3, N), rng.uniform(0, 60e3, N), rng.uniform(0, 2 * DAY, N)
    pts = K.build_points(x, y, t, 5 * rng.standard_normal(N), 5 * rng.standard_normal(N))
    p = K.pack_params(2000, 30e3, 300, 15e3, 3 * DAY, 0.5 * DAY, 0.3, 20e3, DAY)
    s = I.search_coords(pts, 20e3, DAY)
    from gpvecchia.vecchia_utils import find_nn, maximin_order
    order = maximin_order(s)
    nn = find_nn(s[order], 40, verbose=False)
    kw = dict(noise=0.5, nnum=40, nn_cols=K.COORD_COLS, order_idx=order, nn_array=nn)
    base = VecchiaOperator(pts, K.transport_obs_covfunc, p, **kw)
    fast = FastVecchiaOperator(pts, K.transport_obs_covfunc, p, **kw)
    assert abs(base.logdet() - fast.logdet()) < 1e-7 * abs(base.logdet())
    assert abs(base._Lsparse - fast._Lsparse).max() < 1e-12
    v = rng.standard_normal(N)
    assert np.abs(base.solve(v) - fast.solve(v)).max() < 1e-10
    assert abs(float(np.squeeze(base.loglik(v))) - float(np.squeeze(fast.loglik(v)))) < 1e-8
    p2 = p.copy(); p2[1] *= 1.2
    base.update_hyperparams(p2, 0.7); fast.update_hyperparams(p2, 0.7)
    assert abs(base.logdet() - fast.logdet()) < 1e-7 * abs(base.logdet())
    t0 = time.time(); base.update_hyperparams(p2, 0.7); tb = time.time() - t0
    t0 = time.time(); fast.update_hyperparams(p2, 0.7); tf = time.time() - t0
    print(f"  update_hyperparams N={N}: base {tb:.2f}s  fast {tf:.2f}s")


if __name__ == "__main__":
    test_fastop_matches_base()
    print("ok test_fastop_matches_base")

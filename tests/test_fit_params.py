"""Amplitude (gamma, f_delta) parameterisation round-trips; Laplace Hessian recovers a known quadratic.
Run: python tests/test_fit_params.py"""
import numpy as np

from inversion_sst_gp.spacetime import kernels as K
from inversion_sst_gp.spacetime import fit as F

BASE = dict(sigma_psi=2000.0, l_psi=30e3, sigma_phi=350.0, l_phi=15e3, tau0_psi=2.6e5, tau0_phi=4.3e4,
            sigma_S=0.3, l_S=2e4, tau0_S=8.6e4, noise=4.0, **{'lr:diurnal': 0.5})


def test_roundtrip_and_same_kernel_args():
    amp = F.to_amplitude_params(BASE)
    assert 'sigma_psi' not in amp and 'gamma' in amp and 0 < amp['f_delta'] < 1
    back = F.to_potential_params(amp)
    for k, v in BASE.items():
        assert abs(back[k] - v) < 1e-9 * abs(v), k
    base_cov = K.pack_params(1, 1, 1, 1, 1, 1, 1, 1, 1)
    c1, n1, l1 = F.params_to_model_args(BASE, base_cov)
    c2, n2, l2 = F.params_to_model_args(amp, base_cov)
    assert np.allclose(c1, c2) and n1 == n2 and l1 == l2
    # known case: velocity std 0.09 (psi) and 0.03 (phi) -> gamma = sqrt(0.0081 + 0.0009), f = 0.1
    sp, sf = 0.09 * 30e3 * np.sqrt(0.6), 0.03 * 15e3 * np.sqrt(0.6)
    a = F.to_amplitude_params({**BASE, 'sigma_psi': sp, 'sigma_phi': sf})
    assert abs(a['gamma'] - np.sqrt(0.009)) < 1e-9 and abs(a['f_delta'] - 0.1) < 1e-9


class _QuadGP:
    """Stand-in GP: loglik = -0.5 (x - m)' A (x - m) in log(noise), log(lr scale)."""
    class op:
        covparams = K.pack_params(1, 1, 1, 1, 1, 1, 1, 1, 1)

    A = np.array([[4.0, 1.5], [1.5, 2.0]])
    m = np.log([3.0, 0.7])

    def update(self, cov_params=None, noise=None, lowrank_scales=None):
        self.x = np.log([noise, lowrank_scales['t']])

    def loglik(self):
        d = self.x - self.m
        return -0.5 * d @ self.A @ d


def test_laplace_recovers_quadratic():
    gp = _QuadGP()
    params = {'noise': 3.0, 'lr:t': 0.7}
    H, cov, evals, evecs = F.laplace_covariance(gp, params, ['noise', 'lr:t'], h=0.05)
    assert np.allclose(H, _QuadGP.A, atol=1e-6), H
    assert np.allclose(cov, np.linalg.inv(_QuadGP.A), atol=1e-6)


def test_amplitude_uncertainty_delta_method():
    free = ['sigma_psi', 'l_psi', 'sigma_phi', 'l_phi']
    # perfect sigma-l correlation along constant e_psi: gamma uncertainty must vanish for psi-only variation
    p = {**BASE}
    cov = np.zeros((4, 4))
    d = np.array([1.0, 1.0, 0.0, 0.0])          # move sigma_psi and l_psi together: e_psi unchanged
    cov += 0.3 ** 2 * np.outer(d, d)
    g, sg, f, sf = F.amplitude_uncertainty(cov, free, p)
    assert sg < 1e-12 and sf < 1e-12
    cov2 = np.zeros((4, 4)); cov2[0, 0] = 0.3 ** 2    # sigma_psi alone: log e_psi moves by 2*0.3
    g, sg, f, sf = F.amplitude_uncertainty(cov2, free, p)
    assert abs(sg - 0.5 * (1 - f) * 2 * 0.3) < 1e-9


if __name__ == "__main__":
    for fn in (test_roundtrip_and_same_kernel_args, test_laplace_recovers_quadratic, test_amplitude_uncertainty_delta_method):
        fn(); print("ok", fn.__name__)

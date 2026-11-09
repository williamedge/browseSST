"""
`VecchiaOperator` with a cheaper `update_hyperparams` (the inner loop of hyperparameter fitting).

gpvecchia's `_build_factor` computes the factor with the parallel `L_matrix` and then, serially in
Python, (i) recomputes every local Cholesky just to get the log-determinant and (ii) assembles the
sparse factor entry by entry. Both are redundant: row i of `L` already holds `1/sigma_i` in column 0
(`sigma_i^2` = conditional variance of point i), so

    log|C| = sum_i log sigma_i^2 = -2 sum_i log L[i, 0],

and the sparse factor is just `L` scattered onto `nn_array`. Results are identical to the base
class (see tests/test_fastop.py); only the cost changes. Candidate to upstream into gpvecchia.
"""
import numpy as np
from gpvecchia.vecchia_operator import VecchiaOperator, _get_scipy_sparse_ops


class FastVecchiaOperator(VecchiaOperator):
    def _compute_logdet(self):
        # Relies on the nn_array convention from `find_nn`: column 0 of row i is point i itself, so L[i, 0] is the
        # diagonal entry 1/sigma_i of the regression-Cholesky factor. Fail loudly if a custom nn_array breaks it.
        if not np.array_equal(self.nn_array[:, 0], np.arange(self.nn_array.shape[0])):
            raise ValueError("FastVecchiaOperator needs nn_array[:, 0] == arange(n) (each point first in its own "
                             "neighbour set, as find_nn returns); use VecchiaOperator for other layouts")
        return float(-2.0 * np.sum(np.log(self.L[:, 0])))

    def _build_sparse_factor(self):
        csr_matrix, _, _, _ = _get_scipy_sparse_ops()
        n = self.Xo.shape[0]
        valid = self.nn_array >= 0
        rows, cols = np.nonzero(valid)
        data = self.L[:, :self.nn_array.shape[1]][valid]
        if csr_matrix is None:
            Ld = np.zeros((n, n))
            Ld[rows, self.nn_array[valid]] = data
            return Ld
        return csr_matrix((data, (rows, self.nn_array[valid])), shape=(n, n))

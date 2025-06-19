from __future__ import division, print_function

import numpy as np

class R_pca:
    """Robust PCA via principal component pursuit"""

    def __init__(self, D, mu=None, lmbda=None):
        self.D = D
        self.S = np.zeros(self.D.shape)
        self.Y = np.zeros(self.D.shape)

        if mu:
            self.mu = mu
        else:
            self.mu = np.prod(self.D.shape) / (4 * np.linalg.norm(self.D, ord=1))

        self.mu_inv = 1 / self.mu

        if lmbda:
            self.lmbda = lmbda
        else:
            self.lmbda = 1 / np.sqrt(np.max(self.D.shape))

    @staticmethod
    def frobenius_norm(M):
        return np.linalg.norm(M, ord='fro')

    @staticmethod
    def shrink(M, tau):
        return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros(M.shape))

    def svd_threshold(self, M, tau):
        U, S, V = np.linalg.svd(M, full_matrices=False)
        return U @ np.diag(self.shrink(S, tau)) @ V

    def fit(self, tol=None, max_iter=1000, iter_print=100, verbose=True):
        """Run Robust PCA via principal component pursuit.

        Args:
            tol (float, optional): Stopping tolerance.
            max_iter (int): Maximum number of iterations.
            iter_print (int): Iteration interval for logging.
            verbose (bool): If ``True``, print progress information.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Low-rank and sparse matrices.
        """
        iter = 0
        err = np.Inf
        Sk = self.S
        Yk = self.Y
        Lk = np.zeros(self.D.shape)

        if tol:
            _tol = tol
        else:
            _tol = 1e-7 * self.frobenius_norm(self.D)

        while (err > _tol) and iter < max_iter:
            Lk = self.svd_threshold(self.D - Sk + self.mu_inv * Yk, self.mu_inv)
            Sk = self.shrink(self.D - Lk + self.mu_inv * Yk, self.mu_inv * self.lmbda)
            Yk = Yk + self.mu * (self.D - Lk - Sk)
            err = self.frobenius_norm(self.D - Lk - Sk)
            iter += 1
            if verbose and ((iter % iter_print) == 0 or iter == 1 or iter > max_iter or err <= _tol):
                print(f"iteration: {iter}, error: {err}")

        self.L = Lk
        self.S = Sk
        return Lk, Sk


def robust_pca(D, mu=None, lmbda=None, **kwargs):
    """Convenience wrapper for performing robust PCA.

    Args:
        D (np.ndarray): Input data matrix.
        mu (float, optional): Parameter controlling shrinkage.
        lmbda (float, optional): Parameter for sparse term.
        **kwargs: Additional arguments passed to ``R_pca.fit``. Common options
            include ``tol``, ``max_iter``, ``iter_print``, and ``verbose``.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Low-rank (L) and sparse (S) matrices.
    """
    rpca = R_pca(D, mu=mu, lmbda=lmbda)
    L, S = rpca.fit(**kwargs)
    return L, S

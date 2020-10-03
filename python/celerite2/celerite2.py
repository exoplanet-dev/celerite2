# -*- coding: utf-8 -*-

__all__ = ["GaussianProcess"]

import warnings

import numpy as np

from . import driver
from .driver import LinAlgError


class ConstantMean:
    def __init__(self, value=0.0):
        self.value = value

    def __call__(self, x):
        return self.value


class GaussianProcess:
    """The main interface to the celerite2 Gaussian Process (GP) solver

    Args:
        kernel: An instance of a subclass of :class:`terms.Term`.
        mean (optional): The mean function of the process. This can either
            be a callable (it will be evaluated with a single argument, a
            vector of ``x`` values) or a scalar. (default: ``0.0``)
        **kwargs: Other arguments will be passed directly to
            :func:`GaussianProcess.compute` if the argument ``t`` is specified.
    """

    def __init__(self, kernel, t=None, *, mean=0.0, **kwargs):
        self.kernel = kernel
        self.mean = mean

        # Placeholders for storing data
        self._t = None
        self._mean_value = None
        self._diag = None
        self._log_det = -np.inf
        self._norm = np.inf

        # Placeholders to celerite matrices
        self._U = np.empty((0, 0), dtype=np.float64)
        self._V = np.empty((0, 0), dtype=np.float64)
        self._P = np.empty((0, 0), dtype=np.float64)
        self._d = np.empty(0, dtype=np.float64)

        if t is not None:
            self.compute(t, **kwargs)

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, mean):
        if callable(mean):
            self._mean = mean
        else:
            self._mean = ConstantMean(mean)

    @property
    def mean_value(self):
        if self._mean_value is None:
            raise AttributeError(
                "'compute' must be executed before accessing mean_value"
            )
        return self._mean_value

    def compute(
        self, t, *, yerr=None, diag=None, check_sorted=True, quiet=False
    ):
        """Compute the Cholesky factorization of the GP covariance matrix

        Args:
            t (shape[N]): The independent coordinates of the observations.
                This must be sorted in increasing order.
            yerr (shape[N], optional): If provided, the diagonal standard
                deviation of the observation model.
            diag (shape[N], optional): If provided, the diagonal variance of
                the observation model.
            check_sorted (bool, optional): If ``True``, a check is performed to
                make sure that ``t`` is correctly sorted. A ``ValueError`` will
                be thrown when this check fails.
            quiet (bool, optional): If ``True``, when the matrix cannot be
                factorized (because of numerics or otherwise) the solver's
                ``LinAlgError`` will be silenced and the determiniant will be
                set to zero. Otherwise, the exception will be propagated.

        Raises:
            ValueError: When the inputs are not valid (shape, number, etc.).
            LinAlgError: When the matrix is not numerically positive definite.
        """
        # Check the input coordinates
        t = np.atleast_1d(t)
        if check_sorted and np.any(np.diff(t) < 0.0):
            raise ValueError("the input coordinates must be sorted")
        if len(t.shape) != 1:
            raise ValueError("the input coordinates must be one dimensional")

        # Save the diagonal
        self._t = np.ascontiguousarray(t, dtype=np.float64)
        self._mean_value = self._mean(self._t)
        self._diag = np.empty_like(self._t)
        if yerr is None and diag is None:
            self._diag[:] = 0.0

        elif yerr is not None:
            if diag is not None:
                raise ValueError(
                    "only one of 'diag' and 'yerr' can be provided"
                )
            self._diag[:] = np.atleast_1d(yerr) ** 2

        else:
            self._diag[:] = np.atleast_1d(diag)

        # Fill the celerite matrices
        (
            self._d,
            self._U,
            self._V,
            self._P,
        ) = self.kernel.get_celerite_matrices(
            self._t, self._diag, a=self._d, U=self._U, V=self._V, P=self._P
        )

        # Compute the Cholesky factorization
        try:
            self._d, self._W = driver.factor(
                self._U, self._P, self._d, np.copy(self._V)
            )
        except LinAlgError:
            if not quiet:
                raise
            self._log_det = -np.inf
            self._norm = np.inf
        else:
            self._log_det = np.sum(np.log(self._d))
            self._norm = -0.5 * (
                self._log_det + len(self._t) * np.log(2 * np.pi)
            )

    def recompute(self, *, quiet=False):
        """Re-compute the factorization given a previous call to compute

        Args:
            quiet (bool, optional): If ``True``, when the matrix cannot be
                factorized (because of numerics or otherwise) the solver's
                ``LinAlgError`` will be silenced and the determiniant will be
                set to zero. Otherwise, the exception will be propagated.

        Raises:
            RuntimeError: If :func:`GaussianProcess.compute` is not called
                first.
            ValueError: When the inputs are not valid (shape, number, etc.).
        """
        if self._t is None:
            raise RuntimeError(
                "you must call 'compute' directly  at least once"
            )
        return self.compute(
            self._t, diag=self._diag, check_sorted=False, quiet=quiet
        )

    def _process_input(self, y, *, inplace=False, require_vector=False):
        y = np.atleast_1d(y)
        if self._t is None:
            raise RuntimeError("you must call 'compute' first")
        if self._t.shape[0] != y.shape[0]:
            raise ValueError("dimension mismatch")
        if require_vector and self._t.shape != y.shape:
            raise ValueError("'y' must be one dimensional")
        if inplace:
            if (
                y.dtype != "float64"
                or not y.flags.c_contiguous
                or not y.flags.writeable
            ):
                warnings.warn(
                    "Inplace operations can only be made on C-contiguous, "
                    "writable, float64 arrays; a copy will be made"
                )
            y = np.ascontiguousarray(y, dtype=np.float64)
        else:
            y = np.array(y, dtype=np.float64, copy=True, order="C")
        return y

    def apply_inverse(self, y, *, inplace=False):
        """Apply the inverse of the covariance matrix to a vector or matrix

        Solve ``K.x = y`` for ``x`` where ``K`` is the covariance matrix of
        the GP.

        .. note:: The mean function is not applied in this method.

        Args:
            y (shape[N] or shape[N, M]): The vector or matrix ``y`` described
                above.
            inplace (bool, optional): If ``True``, ``y`` will be overwritten
                with the result ``x``.

        Raises:
            RuntimeError: If :func:`GaussianProcess.compute` is not called
                first.
            ValueError: When the inputs are not valid (shape, number, etc.).
        """
        y = self._process_input(y, inplace=inplace)
        return driver.solve(self._U, self._P, self._d, self._W, y)

    def dot_tril(self, y, *, inplace=False):
        """Dot the Cholesky factor of the GP system into a vector or matrix

        Compute ``x = L.y`` where ``K = L.L^T`` and ``K`` is the covariance
        matrix of the GP.

        .. note:: The mean function is not applied in this method.

        Args:
            y (shape[N] or shape[N, M]): The vector or matrix ``y`` described
                above.
            inplace (bool, optional): If ``True``, ``y`` will be overwritten
                with the result ``x``.

        Raises:
            RuntimeError: If :func:`GaussianProcess.compute` is not called
                first.
            ValueError: When the inputs are not valid (shape, number, etc.).
        """
        y = self._process_input(y, inplace=inplace)
        return driver.dot_tril(self._U, self._P, self._d, self._W, y)

    def log_likelihood(self, y, *, inplace=False):
        """Compute the marginalized likelihood of the GP model

        The factorized matrix from the previous call to
        :func:`GaussianProcess.compute` is used so that method must be called
        first.

        Args:
            y (shape[N]): The observations at coordinates ``t`` as defined by
                :func:`GaussianProcess.compute`.
            inplace (bool, optional): If ``True``, ``y`` will be overwritten
                in the process of the calculation. This will reduce the memory
                footprint, but should be used with care since this will
                overwrite the data.

        Raises:
            RuntimeError: If :func:`GaussianProcess.compute` is not called
                first.
            ValueError: When the inputs are not valid (shape, number, etc.).
        """
        y = self._process_input(y, inplace=inplace, require_vector=True)
        if not np.isfinite(self._log_det):
            return -np.inf
        loglike = self._norm - 0.5 * driver.norm(
            self._U, self._P, self._d, self._W, y - self._mean(self._t)
        )
        if not np.isfinite(loglike):
            return -np.inf
        return loglike

    def predict(
        self,
        y,
        t=None,
        *,
        return_cov=False,
        return_var=False,
        include_mean=True,
        kernel=None,
    ):
        """Compute the conditional distribution

        The factorized matrix from the previous call to
        :func:`GaussianProcess.compute` is used so that method must be called
        first.

        Args:
            y (shape[N]): The observations at coordinates ``t`` as defined by
                :func:`GaussianProcess.compute`.
            t (shape[M], optional): The independent coordinates where the
                prediction should be evaluated. If not provided, this will be
                evaluated at the observations ``t`` from
                :func:`GaussianProcess.compute`.
            return_var (bool, optional): Return the variance of the conditional
                distribution.
            return_cov (bool, optional): Return the full covariance matrix of
                the conditional distribution.
            include_mean (bool, optional): Include the mean function in the
                prediction.
            kernel (optional): If provided, compute the conditional
                distribution using a different kernel. This is generally used
                to separate the contributions from different model components.
                Note that the computational cost and scaling will be worse
                when using this parameter.

        Raises:
            RuntimeError: If :func:`GaussianProcess.compute` is not called
                first.
            ValueError: When the inputs are not valid (shape, number, etc.).
        """
        y = self._process_input(y, inplace=True, require_vector=True)
        mean_value = self._mean(self._t)
        alpha = driver.solve(
            self._U, self._P, self._d, self._W, y - mean_value
        )

        if t is None:
            xs = self._t

        else:
            xs = np.ascontiguousarray(t, dtype=np.float64)
            if xs.ndim != 1:
                raise ValueError("dimension mismatch")

        KxsT = None
        if kernel is None:
            kernel = self.kernel

            if t is None:
                mu = y - self._diag * alpha
                if not include_mean:
                    mu -= mean_value
            else:

                (
                    U_star,
                    V_star,
                    inds,
                ) = self.kernel.get_conditional_mean_matrices(self._t, xs)
                mu = np.empty_like(xs)
                mu = driver.conditional_mean(
                    self._U, self._V, self._P, alpha, U_star, V_star, inds, mu
                )

                if include_mean:
                    mu += self._mean(xs)

        else:
            KxsT = kernel.get_value(xs[None, :] - self._t[:, None])
            mu = np.dot(KxsT.T, alpha)
            if include_mean:
                mu += self._mean(xs)

        if not (return_var or return_cov):
            return mu

        # Predictive variance.
        if KxsT is None:
            KxsT = kernel.get_value(xs[None, :] - self._t[:, None])
        if return_var:
            var = kernel.get_value(0.0) - np.einsum(
                "ij,ij->j", KxsT, self.apply_inverse(KxsT, inplace=False)
            )
            return mu, var

        # Predictive covariance
        cov = kernel.get_value(xs[:, None] - xs[None, :])
        cov -= np.tensordot(
            KxsT, self.apply_inverse(KxsT, inplace=False), axes=(0, 0)
        )
        return mu, cov

    def sample(self, *, size=None, include_mean=True):
        """Generate random samples from the prior implied by the GP system

        The factorized matrix from the previous call to
        :func:`GaussianProcess.compute` is used so that method must be called
        first.

        Args:
            size (int, optional): The number of samples to generate. If not
                provided, only one sample will be produced.
            include_mean (bool, optional): Include the mean function in the
                prediction.

        Raises:
            RuntimeError: If :func:`GaussianProcess.compute` is not called
                first.
            ValueError: When the inputs are not valid (shape, number, etc.).
        """

        if self._t is None:
            raise RuntimeError("you must call 'compute' first")
        if size is None:
            n = np.random.randn(len(self._t))
        else:
            n = np.random.randn(len(self._t), size)
        result = self.dot_tril(n, inplace=True).T
        if include_mean:
            result += self._mean(self._t)
        return result

    def sample_conditional(
        self,
        y,
        t=None,
        *,
        size=None,
        regularize=None,
        include_mean=True,
        kernel=None,
    ):
        """Sample from the conditional (predictive) distribution

        .. note:: this method scales as ``O(M^3)`` for large ``M``, where
            ``M == len(t)``.

        Args:
            y (shape[N]): The observations at coordinates ``x`` from
                :func:`GausianProcess.compute`.
            t (shape[M], optional): The independent coordinates where the
                prediction should be made. If this is omitted the coordinates
                will be assumed to be ``x`` from
                :func:`GaussianProcess.compute` and an efficient method will
                be used to compute the mean prediction.
            size (int, optional): The number of samples to generate. If not
                provided, only one sample will be produced.
            regularize (float, optional): For poorly conditioned systems, you
                can provide a small number here to regularize the predictive
                covariance. This number will be added to the diagonal.
            include_mean (bool, optional): Include the mean function in the
                prediction.
            kernel (optional): If provided, compute the conditional
                distribution using a different kernel. This is generally used
                to separate the contributions from different model components.
                Note that the computational cost and scaling will be worse
                when using this parameter.
        """
        mu, cov = self.predict(
            y, t, return_cov=True, include_mean=include_mean, kernel=kernel
        )
        if regularize is not None:
            cov[np.diag_indices_from(cov)] += regularize
        return np.random.multivariate_normal(mu, cov, size=size)

# -*- coding: utf-8 -*-

__all__ = ["KronTerm", "LowRankKronTerm", "KronTermSum"]

import numpy as np

from .terms import Term, TermSumGeneral


class KronTerm(Term):
    """A general multivariate celerite term

    This term supports rectangular data with the shape ``N, M`` where ``N`` is
    the usual "time" axis or similar, but ``M`` is the dimension of the output
    space. The model is that the multivariate data are correlated samples from
    an underlying process descirbed by a standard *celerite* kernel. The
    covariance between output dimensions is given by the general ``M x M``
    matrix ``R``. More details about this model can be found in `Gordon et al.
    (2020) <https://arxiv.org/abs/2007.05799>`_.

    Args:
        term (celerite2.terms.Term): The celerite term describing the
            underlying process.
        R (shape[M, M]): The covariance matrix between output dimensions.
    """

    __requires_general_addition__ = True

    @property
    def dimension(self):
        return self.M

    def __init__(self, term, *, R):
        self.term = term
        self.R = np.ascontiguousarray(np.atleast_2d(R), dtype=np.float64)
        self.M = len(self.R)
        if self.M < 1:
            raise ValueError("At least one 'band' is required")
        if self.R.ndim != 2 or self.R.shape != (self.M, self.M):
            raise ValueError(
                "R must be a square matrix; "
                "use a LowRankKronTerm for the low rank model"
            )
        self.alpha2 = np.diag(self.R)

    def __len__(self):
        return len(self.term) * self.M

    def __add__(self, b):
        if self.dimension != b.dimension:
            raise TypeError("Incompatible term dimensions")
        return KronTermSum(self, b)

    def __radd__(self, b):
        if self.dimension != b.dimension:
            raise TypeError("Incompatible term dimensions")
        return KronTermSum(b, self)

    def __mul__(self, b):
        raise NotImplementedError(
            "multiplication is not implemented for KronTerm"
        )

    def __rmul__(self, b):
        raise NotImplementedError(
            "multiplication is not implemented for KronTerm"
        )

    def get_coefficients(self):
        raise ValueError(
            "KronTerm objects don't provide coefficients and they can't be "
            "used in operations with other celerite2 terms"
        )

    def get_value(self, tau):
        """Compute the value of the kernel as a function of lag

        Args:
            tau (shape[...]): The lags where the kernel should be evaluated.
        """
        return np.kron(self.term.get_value(tau), self.R)

    def to_dense(self, x, diag):
        """Evaluate the dense covariance matrix for this term

        Args:
            x (shape[N]): The independent coordinates of the data.
            diag (shape[N, M]): The diagonal variance of the system.
        """
        K = self.get_value(x[:, None] - x[None, :])
        K[np.diag_indices_from(K)] += np.ascontiguousarray(diag).flatten()
        return K

    def get_psd(self, omega):
        """Compute the value of the power spectral density for this process

        For Kronecker terms, the PSD computed is actually for the base term.

        Args:
            omega (shape[...]): The (angular) frequencies where the power
                should be evaluated.
        """
        return self.term.get_psd(omega)

    def get_celerite_matrices(
        self, x, diag, *, a=None, U=None, V=None, P=None
    ):
        """Get the matrices needed to solve the celerite system

        Pre-allocated arrays can be provided to the Python interface to be
        re-used for multiple evaluations.

        .. note:: In-place operations are not supported by the modeling
            extensions.

        Args:
            x (shape[N]): The independent coordinates of the data.
            diag (shape[N, M]): The diagonal variance of the system.
            a (shape[N*M], optional): The diagonal of the A matrix.
            U (shape[N*M, J*M], optional): The first low-rank matrix.
            V (shape[N*M, J*M], optional): The second low-rank matrix.
            P (shape[N*M-1, J*M], optional): The regularization matrix used for
                numerical stability.

        Raises:
            ValueError: When the inputs are not valid.
        """

        # Check the input dimensions
        x = np.ascontiguousarray(np.atleast_1d(x))
        diag = np.ascontiguousarray(np.atleast_2d(diag))
        N0, M = diag.shape
        if len(x) != N0 or M != self.M:
            raise ValueError("'diag' must have the shape (N, M)")
        N = N0 * M

        # Compute the coefficients and kernel for the celerite subproblem
        ar, cr, ac, _, cc, _ = self.term.get_coefficients()
        _, U_sub, V_sub, _ = self.term.get_celerite_matrices(
            x, np.zeros_like(x)
        )
        J0 = U_sub.shape[1]
        J = self._get_J(J0)

        # Allocate memory as requested
        if a is None:
            a = np.empty(N)
        else:
            a.resize(N, refcheck=False)
        if U is None:
            U = np.empty((N, J))
        else:
            U.resize((N, J), refcheck=False)
        if V is None:
            V = np.empty((N, J))
        else:
            V.resize((N, J), refcheck=False)
        if P is None:
            P = np.empty((N - 1, J))
        else:
            P.resize((N - 1, J), refcheck=False)

        # Expand the times appropriately
        x_full = np.tile(x[:, None], (1, self.M)).flatten()
        dx = x_full[1:] - x_full[:-1]

        a[:] = (
            diag + self.alpha2[None, :] * (np.sum(ar) + np.sum(ac))
        ).flatten()
        U[:] = np.kron(U_sub, self._get_U_kron())
        V[:] = np.kron(V_sub, self._get_V_kron())

        c = np.concatenate((cr, cc, cc))
        P0 = np.exp(-c[None, :] * dx[:, None], out=P[:, :J0])
        self._copy_P(P0, P)

        return a, U, V, P

    def get_conditional_mean_matrices(self, x, t):
        """Get the matrices needed to compute the conditional mean function

        Args:
            x (shape[N]): The independent coordinates of the data.
            t (shape[M]): The independent coordinates where the predictions
                will be made.
        """
        raise NotImplementedError(
            "Conditional mean matrices have not (yet!) been implemented"
        )

    # The following should be implemented by subclasses
    def _get_J(self, J0):
        return self.M * J0

    def _get_U_kron(self):
        return self.R

    def _get_V_kron(self):
        return np.eye(self.M)

    def _copy_P(self, P0, P):
        P[:] = np.tile(P0[:, :, None], (1, 1, self.M)).reshape(P.shape)


class LowRankKronTerm(KronTerm):
    """A low rank multivariate celerite term

    A low rank version of :class:`celerite2.kron.KronTerm` where the covariance
    between outputs is ``R = alpha * alpha^T``, where ``alpha`` is a column
    vector of size ``M``. More details about this model can be found in `Gordon
    et al. (2020) <https://arxiv.org/abs/2007.05799>`_.

    Args:
        term (celerite2.terms.Term): The celerite term describing the
            underlying process.
        alpha (shape[M]): The vector of amplitudes for each output.
    """

    def __init__(self, term, *, alpha):
        self.term = term
        self.alpha = np.ascontiguousarray(
            np.atleast_1d(alpha), dtype=np.float64
        )
        if self.alpha.ndim != 1:
            raise ValueError(
                "alpha must be a vector; "
                "use a general KronTerm for a full rank model"
            )
        self.M = len(self.alpha)
        if self.M < 1:
            raise ValueError("At least one 'band' is required")
        self.R = np.outer(self.alpha, self.alpha)
        self.alpha2 = self.alpha ** 2

    def __len__(self):
        return len(self.term)

    def _get_J(self, J0):
        return J0

    def _get_U_kron(self):
        return self.alpha[:, None]

    def _get_V_kron(self):
        return self.alpha[:, None]

    def _copy_P(self, P0, P):
        pass


class KronTermSum(TermSumGeneral):
    @property
    def dimension(self):
        return self.terms[0].M

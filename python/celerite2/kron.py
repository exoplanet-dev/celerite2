# -*- coding: utf-8 -*-

__all__ = ["KronTerm", "KronTermSum"]

import numpy as np

from . import driver
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

    def __init__(self, term, *, R=None, L=None):
        self.term = term

        if R is not None:
            if L is not None:
                raise ValueError("Only one of 'R' and 'L' can be defined")
            self.R = np.ascontiguousarray(np.atleast_2d(R), dtype=np.float64)
            try:
                self.L = np.linalg.cholesky(self.R)
            except np.linalg.LinAlgError:
                M = np.copy(self.R)
                M[np.diag_indices_from(M)] += 1e-10
                self.L = np.linalg.cholesky(M)
        elif L is not None:
            self.L = np.ascontiguousarray(L, dtype=np.float64)
            if self.L.ndim == 1:
                self.L = np.reshape(self.L, (-1, 1))
            self.R = np.dot(self.L, self.L.T)
        else:
            raise ValueError("One of 'R' and 'L' must be defined")

        self.M, self.K = self.L.shape
        self.alpha2 = np.diag(self.R)

    def __len__(self):
        return len(self.term) * self.K

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
        self,
        x,
        diag,
        *,
        a=None,
        U=None,
        V=None,
        P=None,
        mask=None,
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

        if mask is None:
            N = N0 * M
        else:
            mask = np.ascontiguousarray(mask, dtype=bool)
            if mask.shape != (N0, M):
                raise ValueError("'mask' must have the shape (N, M)")
            N = mask.sum()

        # Compute the coefficients and kernel for the celerite subproblem
        ar, cr, ac, _, cc, _ = self.term.get_coefficients()
        _, U_sub, V_sub, _ = self.term.get_celerite_matrices(
            x, np.zeros_like(x)
        )
        J0 = U_sub.shape[1]
        J = self.K * J0

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

        if mask is None:
            x_full = np.tile(x[:, None], (1, self.M)).flatten()
            a[:] = (
                diag + self.alpha2[None, :] * (np.sum(ar) + np.sum(ac))
            ).flatten()
            U[:] = np.kron(U_sub, self.L)
            V[:] = np.kron(V_sub, self.L)
        else:
            x_full = np.tile(x[:, None], (1, self.M))[mask].flatten()
            a[:] = (diag + self.alpha2[None, :] * (np.sum(ar) + np.sum(ac)))[
                mask
            ].flatten()
            U[:] = np.kron(U_sub, self.L)[mask.flatten()]
            V[:] = np.kron(V_sub, self.L)[mask.flatten()]

        dx = x_full[1:] - x_full[:-1]
        c = np.concatenate((cr, cc, cc))
        P0 = np.exp(-c[None, :] * dx[:, None], out=P[:, :J0])
        if self.K > 1:
            P[:] = np.tile(P0[:, :, None], (1, 1, self.K)).reshape(P.shape)

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


class KronTermSum(TermSumGeneral):
    @property
    def dimension(self):
        return self.terms[0].M

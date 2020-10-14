# -*- coding: utf-8 -*-

__all__ = ["KronTerm"]

import numpy as np

from .terms import Term


class KronTerm(Term):
    def __init__(self, term, *, R):
        self.term = term
        self.R = np.ascontiguousarray(np.atleast_2d(R), dtype=np.float64)
        self.M = len(self.R)
        if self.R.ndim != 2 or self.R.shape != (self.M, self.M):
            raise ValueError(
                "R must be a square matrix; "
                "use a LowRankKronTerm for the low rank model"
            )

    def __add__(self, b):
        raise NotImplementedError("addition is not implemented for KronTerm")

    def __mul__(self, b):
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
            P (shape[N*M-1, J], optional): The regularization matrix used for
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
        J = J0 * self.M

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
            diag + np.diag(self.R)[None, :] * (np.sum(ar) + np.sum(ac))
        ).flatten()
        U[:] = np.kron(U_sub, self.R)
        V[:] = np.kron(V_sub, np.eye(self.M))

        c = np.concatenate((cr, cc, cc))
        P0 = np.exp(-c[None, :] * dx[:, None])
        P[:] = np.tile(P0[:, :, None], (1, 1, self.M)).reshape((-1, J))

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


class LowRankKronTerm(KronTerm):
    pass


class KronTermSum(Term):
    pass

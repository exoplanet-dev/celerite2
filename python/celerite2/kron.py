# -*- coding: utf-8 -*-
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
        return KronTermSum(self, b)

    def __mul__(self, b):
        raise NotImplementedError(
            "multiplication is not implemented for KronTerm"
        )

    def get_coefficients(self):
        raise ValueError("KronTerm objects don't provide coefficients")

    def get_celerite_matrices(
        self, x, diag, *, a=None, U=None, V=None, P=None
    ):
        # Check the input dimensions
        x = np.atleast_1d(x)
        diag = np.atleast_2d(diag)
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
        x_full = np.repeat(x, self.M)
        dx = x_full[1:] - x_full[:-1]

        a[:] = (
            diag + np.diag(self.R)[None, :] * (np.sum(ar) + np.sum(ac))
        ).flatten()
        U[:] = np.kron(U_sub, self.R)
        V[:] = np.kron(V_sub, np.eye(self.M))

        c = np.concatenate((cr, cc, cc))
        P[:, :J0] = np.exp(-c[None, :] * dx[:, None])
        P[:, J0:] = np.repeat(P[:, :J0], self.M - 1, axis=1)

        return a, U, V, P

    def get_value(self, tau):
        return np.kron(self.term.get_value(tau), self.R)


class LowRankKronTerm(KronTerm):
    pass


class KronTermSum(Term):
    pass

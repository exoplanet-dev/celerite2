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
        diag = np.atleast_1d(diag)
        N = len(diag)
        if N != len(x) * self.M:
            raise ValueError("'diag' must have the shape 'N * M'")

        # Compute the kernel for the celerite subproblem
        a_sub, U_sub, V_sub, _ = self.term.get_celerite_matrices(
            x, np.zeros_like(x)
        )
        J = U_sub.shape[1]

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
        a[:] = diag

        a[:] = diag + tt.diag(self.R)[:, None] * (tt.sum(ar) + tt.sum(ac))

        # a = tt.reshape(a.T, (1, a.size))[0]
        # U = tt.slinalg.kron(U, self.R)
        # V = tt.slinalg.kron(V, tt.eye(self.R.shape[0]))
        # c = tt.concatenate((cr, cc, cc))
        # P = tt.exp(-c[None, :] * dx[:, None])
        # P = tt.tile(P, (1, self.R.shape[0]))


class LowRankKronTerm(KronTerm):
    pass


class KronTermSum(Term):
    pass

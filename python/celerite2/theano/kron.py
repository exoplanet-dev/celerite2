# -*- coding: utf-8 -*-

__all__ = ["KronTerm", "LowRankKronTerm", "KronTermSum"]

import theano.tensor as tt
from theano.tensor.slinalg import kron

from .terms import Term, TermSumGeneral
from .. import kron as base_kron


class KronTerm(Term):
    __requires_general_addition__ = True
    __doc__ = base_kron.KronTerm.__doc__

    @property
    def dimension(self):
        return self.M

    def __init__(self, term, *, R):
        self.term = term
        self.R = tt.as_tensor_variable(R)
        self.M = self.R.shape[0]
        if self.R.ndim != 2:
            raise ValueError(
                "R must be a square matrix; "
                "use a LowRankKronTerm for the low rank model"
            )
        self.alpha2 = tt.diag(self.R)

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
        return kron(self.term.get_value(tau), self.R)

    def to_dense(self, x, diag):
        K = self.get_value(x[:, None] - x[None, :])
        K += tt.diag(tt.reshape(diag, (-1,)))
        return K

    def get_psd(self, omega):
        return self.term.get_psd(omega)

    def get_celerite_matrices(self, x, diag):
        # Check the input dimensions
        x = tt.as_tensor_variable(x)
        diag = tt.as_tensor_variable(diag)
        if diag.ndim != 2:
            raise ValueError("'diag' must have the shape (N, M)")

        # Compute the coefficients and kernel for the celerite subproblem
        ar, cr, ac, _, cc, _ = self.term.get_coefficients()
        _, U_sub, V_sub, _ = self.term.get_celerite_matrices(
            x, tt.zeros_like(x)
        )

        # Expand the times appropriately
        x_full = tt.reshape(tt.tile(x[:, None], (1, self.M)), (-1,))
        dx = x_full[1:] - x_full[:-1]

        a = tt.reshape(
            diag + self.alpha2[None, :] * (tt.sum(ar) + tt.sum(ac)), (-1,)
        )
        U = kron(U_sub, self._get_U_kron())
        V = kron(V_sub, self._get_V_kron())

        c = tt.concatenate((cr, cc, cc))
        P0 = tt.exp(-c[None, :] * dx[:, None])
        P = self._copy_P(P0)

        return a, U, V, P

    def get_conditional_mean_matrices(self, x, t):
        raise NotImplementedError(
            "Conditional mean matrices have not (yet!) been implemented"
        )

    # The following should be implemented by subclasses
    def _get_U_kron(self):
        return self.R

    def _get_V_kron(self):
        return tt.eye(self.M)

    def _copy_P(self, P0):
        return tt.tile(P0[:, :, None], (1, 1, self.M)).reshape(
            (P0.shape[0], -1)
        )


class LowRankKronTerm(KronTerm):
    def __init__(self, term, *, alpha):
        self.term = term
        self.alpha = tt.as_tensor_variable(alpha)
        if self.alpha.ndim != 1:
            raise ValueError(
                "alpha must be a vector; "
                "use a general KronTerm for a full rank model"
            )
        self.M = self.alpha.shape[0]
        self.R = self.alpha[None, :] * self.alpha[:, None]
        self.alpha2 = self.alpha ** 2

    def _get_U_kron(self):
        return self.alpha[:, None]

    def _get_V_kron(self):
        return self.alpha[:, None]

    def _copy_P(self, P0):
        return P0


class KronTermSum(TermSumGeneral):
    @property
    def dimension(self):
        return self.terms[0].M

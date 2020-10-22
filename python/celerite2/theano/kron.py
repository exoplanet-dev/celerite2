# -*- coding: utf-8 -*-

__all__ = ["KronTerm", "KronTermSum"]
from theano import tensor as tt
from theano.tensor.slinalg import Cholesky, kron

from .. import kron as base_kron
from .terms import Term, TermSumGeneral

CITATIONS = (
    ("celerite2:gordon20",),
    r"""
@article{celerite2:gordon20,
       author = {{Gordon}, Tyler and {Agol}, Eric and
                 {Foreman-Mackey}, Daniel},
        title = "{A Fast, 2D Gaussian Process Method Based on Celerite:
                  Applications to Transiting Exoplanet Discovery and
                  Characterization}",
      journal = {arXiv e-prints},
         year = 2020,
        month = jul,
          eid = {arXiv:2007.05799},
        pages = {arXiv:2007.05799},
archivePrefix = {arXiv},
       eprint = {2007.05799},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020arXiv200705799G},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
""",
)


cholesky = Cholesky(lower=True)


class KronTerm(Term):
    __requires_general_addition__ = True
    __doc__ = base_kron.KronTerm.__doc__
    __citations__ = CITATIONS

    @property
    def dimension(self):
        return self.M

    def __init__(self, term, *, R=None, L=None):
        self.term = term

        if R is not None:
            if L is not None:
                raise ValueError("Only one of 'R' and 'L' can be defined")
            self.R = tt.as_tensor_variable(R)
            self.L = cholesky(self.R + 1e-10 * tt.eye(R.shape[0]))
            self.M = self.R.shape[0]
            self.K = self.R.shape[0]

        elif L is not None:
            self.L = tt.as_tensor_variable(L)
            self.M = self.L.shape[0]
            if self.L.ndim == 1:
                self.L = tt.reshape(self.L, (-1, 1))
                self.K = 1
            else:
                self.K = self.L.shape[1]
            self.R = tt.dot(self.L, self.L.T)
        else:
            raise ValueError("One of 'R' and 'L' must be defined")

        self.alpha2 = tt.diag(self.R)

    def __add__(self, b):
        return KronTermSum(self, b)

    def __radd__(self, b):
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
        U = kron(U_sub, self.L)
        V = kron(V_sub, self.L)

        c = tt.concatenate((cr, cc, cc))
        P0 = tt.exp(-c[None, :] * dx[:, None])
        P = tt.tile(P0[:, :, None], (1, 1, self.K)).reshape((P0.shape[0], -1))

        return a, U, V, P

    def get_conditional_mean_matrices(self, x, t):
        raise NotImplementedError(
            "Conditional mean matrices have not (yet!) been implemented"
        )


class KronTermSum(TermSumGeneral):
    __citations__ = CITATIONS
    __doc__ = base_kron.KronTermSum.__doc__

    @property
    def dimension(self):
        return self.terms[0].M

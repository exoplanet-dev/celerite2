# -*- coding: utf-8 -*-

__all__ = [
    "Term",
    "TermSum",
    "TermProduct",
    "TermDiff",
    "TermConvolution",
    "RealTerm",
    "ComplexTerm",
    "SHOTerm",
    "Matern32Term",
    "RotationTerm",
]

import numpy as np
import pytensor
import pytensor.tensor as pt
from pytensor.ifelse import ifelse

import celerite2.terms as base_terms
from celerite2.pymc import ops


class Term(base_terms.Term):
    __doc__ = base_terms.Term.__doc__

    def __init__(self, *, dtype="float64"):
        self.dtype = dtype
        self.coefficients = self.get_coefficients()

    def __add__(self, b):
        dtype = pytensor.scalar.upcast(self.dtype, b.dtype)
        return TermSum(self, b, dtype=dtype)

    def __mul__(self, b):
        dtype = pytensor.scalar.upcast(self.dtype, b.dtype)
        return TermProduct(self, b, dtype=dtype)

    @property
    def terms(self):
        return [self]

    def get_coefficients(self):
        raise NotImplementedError("subclasses must implement this method")

    def get_value(self, tau):
        r = self._get_value_real(self.coefficients[:2], tau)
        c = self._get_value_complex(self.coefficients[2:], tau)
        return r + c

    def _get_value_real(self, coefficients, tau):
        ar, cr = coefficients
        tau = pt.abs(tau)
        tau = pt.shape_padright(tau)
        return pt.sum(ar * pt.exp(-cr * tau), axis=-1)

    def _get_value_complex(self, coefficients, tau):
        ac, bc, cc, dc = coefficients
        tau = pt.abs(tau)
        tau = pt.shape_padright(tau)
        factor = pt.exp(-cc * tau)
        K = pt.sum(ac * factor * pt.cos(dc * tau), axis=-1)
        K += pt.sum(bc * factor * pt.sin(dc * tau), axis=-1)
        return K

    def get_psd(self, omega):
        r = self._get_psd_real(self.coefficients[:2], omega)
        c = self._get_psd_complex(self.coefficients[2:], omega)
        return r + c

    def _get_psd_real(self, coefficients, omega):
        ar, cr = coefficients
        omega = pt.shape_padright(omega)
        w2 = omega**2
        power = pt.sum(ar * cr / (cr**2 + w2), axis=-1)
        return np.sqrt(2.0 / np.pi) * power

    def _get_psd_complex(self, coefficients, omega):
        ac, bc, cc, dc = coefficients
        omega = pt.shape_padright(omega)
        w2 = omega**2
        w02 = cc**2 + dc**2
        power = pt.sum(
            ((ac * cc + bc * dc) * w02 + (ac * cc - bc * dc) * w2)
            / (w2 * w2 + 2.0 * (cc**2 - dc**2) * w2 + w02 * w02),
            axis=-1,
        )
        return np.sqrt(2.0 / np.pi) * power

    def to_dense(self, x, diag):
        K = self.get_value(x[:, None] - x[None, :])
        K += pt.diag(diag)
        return K

    def get_celerite_matrices(self, x, diag, **kwargs):
        x = pt.as_tensor_variable(x)
        diag = pt.as_tensor_variable(diag)
        cr, ar, Ur, Vr = self._get_celerite_matrices_real(
            self.coefficients[:2], x, **kwargs
        )
        cc, ac, Uc, Vc = self._get_celerite_matrices_complex(
            self.coefficients[2:], x, **kwargs
        )
        c = pt.concatenate((cr, cc))
        a = diag + ar + ac
        U = pt.concatenate((Ur, Uc), axis=1)
        V = pt.concatenate((Vr, Vc), axis=1)
        return c, a, U, V

    def _get_celerite_matrices_real(self, coefficients, x, **kwargs):
        ar, cr = coefficients
        z = pt.zeros_like(x)
        return (
            cr,
            pt.sum(ar),
            ar[None, :] + z[:, None],
            pt.ones_like(ar)[None, :] + z[:, None],
        )

    def _get_celerite_matrices_complex(self, coefficients, x, **kwargs):
        ac, bc, cc, dc = coefficients

        arg = dc[None, :] * x[:, None]
        cos = pt.cos(arg)
        sin = pt.sin(arg)
        U = pt.concatenate(
            (
                ac[None, :] * cos + bc[None, :] * sin,
                ac[None, :] * sin - bc[None, :] * cos,
            ),
            axis=1,
        )
        V = pt.concatenate((cos, sin), axis=1)
        c = pt.concatenate((cc, cc))

        return c, pt.sum(ac), U, V

    def dot(self, x, diag, y):
        x = pt.as_tensor_variable(x)
        y = pt.as_tensor_variable(y)

        is_vector = False
        if y.ndim == 1:
            is_vector = True
            y = y[:, None]
        if y.ndim != 2:
            raise ValueError("'y' can only be a vector or matrix")

        c, a, U, V = self.get_celerite_matrices(x, diag)
        z = y * a[:, None]
        z += ops.matmul_lower(x, c, U, V, y)[0]
        z += ops.matmul_upper(x, c, U, V, y)[0]

        if is_vector:
            return z[:, 0]
        return z


class TermSum(Term):
    __doc__ = base_terms.TermSum.__doc__

    def __init__(self, *terms, **kwargs):
        self._terms = terms
        self.dtype = kwargs.get("dtype", "float64")

    @property
    def terms(self):
        return self._terms

    def get_value(self, tau):
        tau = pt.as_tensor_variable(tau)
        return sum(term.get_value(tau) for term in self._terms)

    def get_psd(self, omega):
        omega = pt.as_tensor_variable(omega)
        return sum(term.get_psd(omega) for term in self._terms)

    def get_celerite_matrices(self, x, diag, **kwargs):
        matrices = (
            term.get_celerite_matrices(x, pt.zeros_like(diag), **kwargs)
            for term in self._terms
        )
        c, a, U, V = zip(*matrices)
        return (
            pt.concatenate(c, axis=-1),
            sum(a) + diag,
            pt.concatenate(U, axis=-1),
            pt.concatenate(V, axis=-1),
        )


class TermProduct(Term):
    __doc__ = base_terms.TermProduct.__doc__

    def __init__(self, term1, term2, **kwargs):
        self.term1 = term1
        self.term2 = term2
        self.dtype = kwargs.get("dtype", "float64")

    def get_value(self, tau):
        tau = pt.as_tensor_variable(tau)
        return self.term1.get_value(tau) * self.term2.get_value(tau)

    def get_psd(self, omega):
        raise NotImplementedError(
            "The PSD function is not implemented for general Term products"
        )

    def get_celerite_matrices(self, x, diag, **kwargs):
        z = pt.zeros_like(diag)
        c1, a1, U1, V1 = self.term1.get_celerite_matrices(x, z, **kwargs)
        c2, a2, U2, V2 = self.term2.get_celerite_matrices(x, z, **kwargs)

        mg = pt.mgrid[: c1.shape[0], : c2.shape[0]]
        i = pt.flatten(mg[0])
        j = pt.flatten(mg[1])

        c = c1[i] + c2[j]
        a = a1 * a2 + diag
        U = U1[:, i] * U2[:, j]
        V = V1[:, i] * V2[:, j]
        return c, a, U, V


class TermDiff(Term):
    __doc__ = base_terms.TermDiff.__doc__

    def __init__(self, term, **kwargs):
        if not hasattr(term, "coefficients"):
            raise TypeError(
                "Term operations can only be performed on terms that provide "
                "coefficients"
            )
        self.term = term
        super().__init__(**kwargs)

    def get_coefficients(self):
        coeffs = self.term.coefficients
        a, b, c, d = coeffs[2:]
        final_coeffs = [
            -coeffs[0] * coeffs[1] ** 2,
            coeffs[1],
            a * (d**2 - c**2) + 2 * b * c * d,
            b * (d**2 - c**2) - 2 * a * c * d,
            c,
            d,
        ]
        return final_coeffs


class TermConvolution(Term):
    __doc__ = base_terms.TermConvolution.__doc__

    def __init__(self, term, delta, **kwargs):
        if not hasattr(term, "coefficients"):
            raise TypeError(
                "Term operations can only be performed on terms that provide "
                "coefficients"
            )
        self.term = term
        self.delta = pt.as_tensor_variable(delta).astype("float64")
        super().__init__(**kwargs)

    def get_celerite_matrices(self, x, diag, **kwargs):
        dt = self.delta
        ar, cr, a, b, c, d = self.term.coefficients

        # Real part
        cd = cr * dt
        delta_diag = 2 * pt.sum(ar * (cd - pt.sinh(cd)) / cd**2)

        # Complex part
        cd = c * dt
        dd = d * dt
        c2 = c**2
        d2 = d**2
        c2pd2 = c2 + d2
        C1 = a * (c2 - d2) + 2 * b * c * d
        C2 = b * (c2 - d2) - 2 * a * c * d
        norm = (dt * c2pd2) ** 2
        sinh = pt.sinh(cd)
        cosh = pt.cosh(cd)
        delta_diag += 2 * pt.sum(
            (
                C2 * cosh * pt.sin(dd)
                - C1 * sinh * pt.cos(dd)
                + (a * c + b * d) * dt * c2pd2
            )
            / norm
        )

        new_diag = diag + delta_diag
        return super().get_celerite_matrices(x, new_diag)

    def get_coefficients(self):
        ar, cr, a, b, c, d = self.term.coefficients

        # Real componenets
        crd = cr * self.delta
        coeffs = [2 * ar * (pt.cosh(crd) - 1) / crd**2, cr]

        # Imaginary coefficients
        cd = c * self.delta
        dd = d * self.delta
        c2 = c**2
        d2 = d**2
        factor = 2.0 / (self.delta * (c2 + d2)) ** 2
        cos_term = pt.cosh(cd) * pt.cos(dd) - 1
        sin_term = pt.sinh(cd) * pt.sin(dd)

        C1 = a * (c2 - d2) + 2 * b * c * d
        C2 = b * (c2 - d2) - 2 * a * c * d

        coeffs += [
            factor * (C1 * cos_term - C2 * sin_term),
            factor * (C2 * cos_term + C1 * sin_term),
            c,
            d,
        ]

        return coeffs

    def get_psd(self, omega):
        psd0 = self.term.get_psd(omega)
        arg = 0.5 * self.delta * omega
        sinc = pt.switch(pt.neq(arg, 0), pt.sin(arg) / arg, pt.ones_like(arg))
        return psd0 * sinc**2

    def get_value(self, tau0):
        dt = self.delta
        ar, cr, a, b, c, d = self.term.coefficients

        # Format the lags correctly
        tau0 = pt.abs(tau0)
        tau = tau0[..., None]

        # Precompute some factors
        dpt = dt + tau
        dmt = dt - tau

        # Real parts:
        # tau > Delta
        crd = cr * dt
        cosh = pt.cosh(crd)
        norm = 2 * ar / crd**2
        K_large = pt.sum(norm * (cosh - 1) * pt.exp(-cr * tau), axis=-1)

        # tau < Delta
        crdmt = cr * dmt
        K_small = K_large + pt.sum(norm * (crdmt - pt.sinh(crdmt)), axis=-1)

        # Complex part
        cd = c * dt
        dd = d * dt
        c2 = c**2
        d2 = d**2
        c2pd2 = c2 + d2
        C1 = a * (c2 - d2) + 2 * b * c * d
        C2 = b * (c2 - d2) - 2 * a * c * d
        norm = 1.0 / (dt * c2pd2) ** 2
        k0 = pt.exp(-c * tau)
        cdt = pt.cos(d * tau)
        sdt = pt.sin(d * tau)

        # For tau > Delta
        cos_term = 2 * (pt.cosh(cd) * pt.cos(dd) - 1)
        sin_term = 2 * (pt.sinh(cd) * pt.sin(dd))
        factor = k0 * norm
        K_large += pt.sum(
            (C1 * cos_term - C2 * sin_term) * factor * cdt, axis=-1
        )
        K_large += pt.sum(
            (C2 * cos_term + C1 * sin_term) * factor * sdt, axis=-1
        )

        # tau < Delta
        edmt = pt.exp(-c * dmt)
        edpt = pt.exp(-c * dpt)
        cos_term = (
            edmt * pt.cos(d * dmt) + edpt * pt.cos(d * dpt) - 2 * k0 * cdt
        )
        sin_term = (
            edmt * pt.sin(d * dmt) + edpt * pt.sin(d * dpt) - 2 * k0 * sdt
        )
        K_small += pt.sum(2 * (a * c + b * d) * c2pd2 * dmt * norm, axis=-1)
        K_small += pt.sum((C1 * cos_term + C2 * sin_term) * norm, axis=-1)

        return pt.switch(pt.le(tau0, dt), K_small, K_large)


class RealTerm(Term):
    __doc__ = base_terms.RealTerm.__doc__

    def __init__(self, *, a, c, **kwargs):
        self.a = pt.as_tensor_variable(a).astype("float64")
        self.c = pt.as_tensor_variable(c).astype("float64")
        super().__init__(**kwargs)

    def get_coefficients(self):
        empty = pt.zeros(0, dtype=self.dtype)
        if self.a.ndim == 0:
            return (
                self.a[None],
                self.c[None],
                empty,
                empty,
                empty,
                empty,
            )
        return (self.a, self.c, empty, empty, empty, empty)


class ComplexTerm(Term):
    __doc__ = base_terms.ComplexTerm.__doc__

    def __init__(self, *, a, b, c, d, **kwargs):
        self.a = pt.as_tensor_variable(a).astype("float64")
        self.b = pt.as_tensor_variable(b).astype("float64")
        self.c = pt.as_tensor_variable(c).astype("float64")
        self.d = pt.as_tensor_variable(d).astype("float64")
        super().__init__(**kwargs)

    def get_coefficients(self):
        empty = pt.zeros(0, dtype=self.dtype)
        if self.a.ndim == 0:
            return (
                empty,
                empty,
                self.a[None],
                self.b[None],
                self.c[None],
                self.d[None],
            )
        return (empty, empty, self.a, self.b, self.c, self.d)


class SHOTerm(Term):
    __doc__ = base_terms.SHOTerm.__doc__
    __parameter_spec__ = base_terms.SHOTerm.__parameter_spec__

    @base_terms.handle_parameter_spec(
        lambda x: pt.as_tensor_variable(x).astype("float64")
    )
    def __init__(self, *, eps=1e-5, **kwargs):
        self.eps = pt.as_tensor_variable(eps).astype("float64")
        self.dtype = kwargs.get("dtype", "float64")
        self.overdamped = self.get_overdamped_coefficients()
        self.underdamped = self.get_underdamped_coefficients()
        self.cond = pt.lt(self.Q, 0.5)

    def get_value(self, tau):
        return ifelse(
            self.cond,
            super()._get_value_real(self.overdamped, tau),
            super()._get_value_complex(self.underdamped, tau),
        )

    def get_psd(self, omega):
        return ifelse(
            self.cond,
            super()._get_psd_real(self.overdamped, omega),
            super()._get_psd_complex(self.underdamped, omega),
        )

    def get_celerite_matrices(self, x, diag, **kwargs):
        x = pt.as_tensor_variable(x)
        diag = pt.as_tensor_variable(diag)
        cr, ar, Ur, Vr = super()._get_celerite_matrices_real(
            self.overdamped, x, **kwargs
        )
        cc, ac, Uc, Vc = super()._get_celerite_matrices_complex(
            self.underdamped, x, **kwargs
        )
        ar = ar + diag
        ac = ac + diag

        cr, cc = pt.broadcast_arrays(cr, cc)
        ar, ac = pt.broadcast_arrays(ar, ac)
        Ur, Uc = pt.broadcast_arrays(Ur, Uc)
        Vr, Vc = pt.broadcast_arrays(Vr, Vc)

        return [
            pt.switch(self.cond, a, b)
            for a, b in zip((cr, ar, Ur, Vr), (cc, ac, Uc, Vc))
        ]

    def get_overdamped_coefficients(self):
        Q = self.Q
        f = pt.sqrt(pt.maximum(1.0 - 4.0 * Q**2, self.eps))
        return (
            0.5
            * self.S0
            * self.w0
            * Q
            * pt.stack([1.0 + 1.0 / f, 1.0 - 1.0 / f]),
            0.5 * self.w0 / Q * pt.stack([1.0 - f, 1.0 + f]),
        )

    def get_underdamped_coefficients(self):
        Q = self.Q
        f = pt.sqrt(pt.maximum(4.0 * Q**2 - 1.0, self.eps))
        a = self.S0 * self.w0 * Q
        c = 0.5 * self.w0 / Q
        return (
            pt.stack([a]),
            pt.stack([a / f]),
            pt.stack([c]),
            pt.stack([c * f]),
        )


class Matern32Term(Term):
    __doc__ = base_terms.Matern32Term.__doc__

    def __init__(self, *, sigma, rho, eps=0.01, **kwargs):
        self.sigma = pt.as_tensor_variable(sigma).astype("float64")
        self.rho = pt.as_tensor_variable(rho).astype("float64")
        self.eps = pt.as_tensor_variable(eps).astype("float64")
        super().__init__(**kwargs)

    def get_coefficients(self):
        w0 = np.sqrt(3.0) / self.rho
        S0 = self.sigma**2 / w0
        empty = pt.zeros(0, dtype=self.dtype)
        return (
            empty,
            empty,
            (w0 * S0)[None],
            (w0 * w0 * S0 / self.eps)[None],
            w0[None],
            self.eps[None],
        )


class RotationTerm(TermSum):
    __doc__ = base_terms.RotationTerm.__doc__

    def __init__(self, *, sigma, period, Q0, dQ, f, **kwargs):
        self.sigma = pt.as_tensor_variable(sigma).astype("float64")
        self.period = pt.as_tensor_variable(period).astype("float64")
        self.Q0 = pt.as_tensor_variable(Q0).astype("float64")
        self.dQ = pt.as_tensor_variable(dQ).astype("float64")
        self.f = pt.as_tensor_variable(f).astype("float64")

        self.amp = self.sigma**2 / (1 + self.f)

        # One term with a period of period
        Q1 = 0.5 + self.Q0 + self.dQ
        w1 = 4 * np.pi * Q1 / (self.period * pt.sqrt(4 * Q1**2 - 1))
        S1 = self.amp / (w1 * Q1)

        # Another term at half the period
        Q2 = 0.5 + self.Q0
        w2 = 8 * np.pi * Q2 / (self.period * pt.sqrt(4 * Q2**2 - 1))
        S2 = self.f * self.amp / (w2 * Q2)

        super().__init__(
            SHOTerm(S0=S1, w0=w1, Q=Q1), SHOTerm(S0=S2, w0=w2, Q=Q2), **kwargs
        )

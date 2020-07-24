# -*- coding: utf-8 -*-

__all__ = [
    "Term",
    "TermSum",
    "TermProduct",
    "TermDiff",
    "IntegratedTerm",
    "RealTerm",
    "ComplexTerm",
    "SHOTerm",
    "Matern32Term",
    "RotationTerm",
]
import numpy as np
import theano
from theano import tensor as tt
from theano.ifelse import ifelse

from . import ops


class Term:
    """The abstract base "term" that is the superclass of all other terms

    Subclasses should overload the :func:`terms.Term.get_real_coefficients`
    and :func:`terms.Term.get_complex_coefficients` methods.

    """

    def __init__(self, *, dtype="float64"):
        self.dtype = dtype
        self.coefficients = self.get_coefficients()

    def __add__(self, b):
        dtype = theano.scalar.upcast(self.dtype, b.dtype)
        return TermSum(self, b, dtype=dtype)

    def __radd__(self, b):
        dtype = theano.scalar.upcast(self.dtype, b.dtype)
        return TermSum(b, self, dtype=dtype)

    def __mul__(self, b):
        dtype = theano.scalar.upcast(self.dtype, b.dtype)
        return TermProduct(self, b, dtype=dtype)

    def __rmul__(self, b):
        dtype = theano.scalar.upcast(self.dtype, b.dtype)
        return TermProduct(b, self, dtype=dtype)

    @property
    def terms(self):
        return [self]

    def get_coefficients(self):
        raise NotImplementedError("subclasses must implement this method")

    def get_value(self, tau):
        ar, cr, ac, bc, cc, dc = self.coefficients
        tau = tt.abs_(tau)
        tau = tt.reshape(
            tau, tt.concatenate([tau.shape, [1]]), ndim=tau.ndim + 1
        )
        K = tt.sum(ar * tt.exp(-cr * tau), axis=-1)
        factor = tt.exp(-cc * tau)
        K += tt.sum(ac * factor * tt.cos(dc * tau), axis=-1)
        K += tt.sum(bc * factor * tt.sin(dc * tau), axis=-1)
        return K

    def get_psd(self, omega):
        ar, cr, ac, bc, cc, dc = self.coefficients
        omega = tt.reshape(
            omega, tt.concatenate([omega.shape, [1]]), ndim=omega.ndim + 1
        )
        w2 = omega ** 2
        w02 = cc ** 2 + dc ** 2
        power = tt.sum(ar * cr / (cr ** 2 + w2), axis=-1)
        power += tt.sum(
            ((ac * cc + bc * dc) * w02 + (ac * cc - bc * dc) * w2)
            / (w2 * w2 + 2.0 * (cc ** 2 - dc ** 2) * w2 + w02 * w02),
            axis=-1,
        )
        return np.sqrt(2.0 / np.pi) * power

    def to_dense(self, x, diag):
        K = self.get_value(x[:, None] - x[None, :])
        K += tt.diag(diag)
        return K

    def get_celerite_matrices(self, x, diag):
        x = tt.as_tensor_variable(x)
        diag = tt.as_tensor_variable(diag)
        ar, cr, ac, bc, cc, dc = self.coefficients
        a = diag + tt.sum(ar) + tt.sum(ac)

        arg = dc[None, :] * x[:, None]
        cos = tt.cos(arg)
        sin = tt.sin(arg)
        z = tt.zeros_like(x)

        U = tt.concatenate(
            (
                ar[None, :] + z[:, None],
                ac[None, :] * cos + bc[None, :] * sin,
                ac[None, :] * sin - bc[None, :] * cos,
            ),
            axis=1,
        )

        V = tt.concatenate(
            (tt.ones_like(ar)[None, :] + z[:, None], cos, sin), axis=1,
        )

        dx = x[1:] - x[:-1]
        c = tt.concatenate((cr, cc, cc))
        P = tt.exp(-c[None, :] * dx[:, None])

        return a, U, V, P

    def get_conditional_mean_matrices(self, x, t):
        ar, cr, ac, bc, cc, dc = self.coefficients
        x = tt.as_tensor_variable(x)

        inds = tt.extra_ops.searchsorted(x, t)
        _, U_star, V_star, _ = self.get_celerite_matrices(t, t)

        c = tt.concatenate((cr, cc, cc))
        dx = t - x[tt.minimum(inds, x.size - 1)]
        U_star *= tt.exp(-c[None, :] * dx[:, None])

        dx = x[tt.maximum(inds - 1, 0)] - t
        V_star *= tt.exp(-c[None, :] * dx[:, None])

        return U_star, V_star, inds

    def dot(self, x, diag, y):
        a, U, V, P = self.get_celerite_matrices(x, diag)
        return ops.matmul(a, U, V, P, tt.as_tensor_variable(y))[0]


class TermSum(Term):
    def __init__(self, *terms, **kwargs):
        if any(isinstance(term, IntegratedTerm) for term in terms):
            raise TypeError(
                "You cannot perform operations on an "
                "IntegratedTerm, it must be the outer term in "
                "the kernel"
            )
        self._terms = terms
        super().__init__(**kwargs)

    @property
    def terms(self):
        return self._terms

    def get_coefficients(self):
        coeffs = (t.coefficients for t in self.terms)
        return tuple(tt.concatenate(a, axis=0) for a in zip(*coeffs))


class TermProduct(Term):
    def __init__(self, term1, term2, **kwargs):
        int1 = isinstance(term1, IntegratedTerm)
        int2 = isinstance(term2, IntegratedTerm)
        if int1 or int2:
            raise TypeError(
                "You cannot perform operations on an "
                "IntegratedTerm, it must be the outer term in "
                "the kernel"
            )
        self.term1 = term1
        self.term2 = term2
        super().__init__(**kwargs)

    def get_coefficients(self):
        c1 = self.term1.coefficients
        c2 = self.term2.coefficients

        # First compute real terms
        ar = []
        cr = []
        ar.append(tt.flatten(c1[0][:, None] * c2[0][None, :]))
        cr.append(tt.flatten(c1[1][:, None] + c2[1][None, :]))

        # Then the complex terms
        ac = []
        bc = []
        cc = []
        dc = []

        # real * complex
        ac.append(tt.flatten(c1[0][:, None] * c2[2][None, :]))
        bc.append(tt.flatten(c1[0][:, None] * c2[3][None, :]))
        cc.append(tt.flatten(c1[1][:, None] + c2[4][None, :]))
        dc.append(tt.flatten(tt.zeros_like(c1[1])[:, None] + c2[5][None, :]))

        ac.append(tt.flatten(c2[0][:, None] * c1[2][None, :]))
        bc.append(tt.flatten(c2[0][:, None] * c1[3][None, :]))
        cc.append(tt.flatten(c2[1][:, None] + c1[4][None, :]))
        dc.append(tt.flatten(tt.zeros_like(c2[1])[:, None] + c1[5][None, :]))

        # complex * complex
        aj, bj, cj, dj = c1[2:]
        ak, bk, ck, dk = c2[2:]

        ac.append(
            tt.flatten(
                0.5 * (aj[:, None] * ak[None, :] + bj[:, None] * bk[None, :])
            )
        )
        bc.append(
            tt.flatten(
                0.5 * (bj[:, None] * ak[None, :] - aj[:, None] * bk[None, :])
            )
        )
        cc.append(tt.flatten(cj[:, None] + ck[None, :]))
        dc.append(tt.flatten(dj[:, None] - dk[None, :]))

        ac.append(
            tt.flatten(
                0.5 * (aj[:, None] * ak[None, :] - bj[:, None] * bk[None, :])
            )
        )
        bc.append(
            tt.flatten(
                0.5 * (bj[:, None] * ak[None, :] + aj[:, None] * bk[None, :])
            )
        )
        cc.append(tt.flatten(cj[:, None] + ck[None, :]))
        dc.append(tt.flatten(dj[:, None] + dk[None, :]))

        return [
            tt.concatenate(vals, axis=0)
            if len(vals)
            else tt.zeros(0, dtype=self.dtype)
            for vals in (ar, cr, ac, bc, cc, dc)
        ]


class TermDiff(Term):
    def __init__(self, term, **kwargs):
        if isinstance(term, IntegratedTerm):
            raise TypeError(
                "You cannot perform operations on an "
                "IntegratedTerm, it must be the outer term in "
                "the kernel"
            )
        self.term = term
        super().__init__(**kwargs)

    def get_coefficients(self):
        coeffs = self.term.coefficients
        a, b, c, d = coeffs[2:]
        final_coeffs = [
            -coeffs[0] * coeffs[1] ** 2,
            coeffs[1],
            a * (d ** 2 - c ** 2) + 2 * b * c * d,
            b * (d ** 2 - c ** 2) - 2 * a * c * d,
            c,
            d,
        ]
        return final_coeffs


class IntegratedTerm(Term):
    def __init__(self, term, delta, **kwargs):
        self.term = term
        self.delta = tt.as_tensor_variable(delta).astype("float64")
        super().__init__(**kwargs)

    def get_celerite_matrices(self, x, diag):
        dt = self.delta
        ar, cr, a, b, c, d = self.term.coefficients

        # Real part
        cd = cr * dt
        delta_diag = 2 * tt.sum(ar * (cd - tt.sinh(cd)) / cd ** 2)

        # Complex part
        cd = c * dt
        dd = d * dt
        c2 = c ** 2
        d2 = d ** 2
        c2pd2 = c2 + d2
        C1 = a * (c2 - d2) + 2 * b * c * d
        C2 = b * (c2 - d2) - 2 * a * c * d
        norm = (dt * c2pd2) ** 2
        sinh = tt.sinh(cd)
        cosh = tt.cosh(cd)
        delta_diag += 2 * tt.sum(
            (
                C2 * cosh * tt.sin(dd)
                - C1 * sinh * tt.cos(dd)
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
        coeffs = [2 * ar * (tt.cosh(crd) - 1) / crd ** 2, cr]

        # Imaginary coefficients
        cd = c * self.delta
        dd = d * self.delta
        c2 = c ** 2
        d2 = d ** 2
        factor = 2.0 / (self.delta * (c2 + d2)) ** 2
        cos_term = tt.cosh(cd) * tt.cos(dd) - 1
        sin_term = tt.sinh(cd) * tt.sin(dd)

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
        sinc = tt.switch(tt.neq(arg, 0), tt.sin(arg) / arg, tt.ones_like(arg))
        return psd0 * sinc ** 2

    def get_value(self, tau0):
        dt = self.delta
        ar, cr, a, b, c, d = self.term.coefficients

        # Format the lags correctly
        tau0 = tt.abs_(tau0)
        tau = tau0[..., None]

        # Precompute some factors
        dpt = dt + tau
        dmt = dt - tau

        # Real parts:
        # tau > Delta
        crd = cr * dt
        cosh = tt.cosh(crd)
        norm = 2 * ar / crd ** 2
        K_large = tt.sum(norm * (cosh - 1) * tt.exp(-cr * tau), axis=-1)

        # tau < Delta
        crdmt = cr * dmt
        K_small = K_large + tt.sum(norm * (crdmt - tt.sinh(crdmt)), axis=-1)

        # Complex part
        cd = c * dt
        dd = d * dt
        c2 = c ** 2
        d2 = d ** 2
        c2pd2 = c2 + d2
        C1 = a * (c2 - d2) + 2 * b * c * d
        C2 = b * (c2 - d2) - 2 * a * c * d
        norm = 1.0 / (dt * c2pd2) ** 2
        k0 = tt.exp(-c * tau)
        cdt = tt.cos(d * tau)
        sdt = tt.sin(d * tau)

        # For tau > Delta
        cos_term = 2 * (tt.cosh(cd) * tt.cos(dd) - 1)
        sin_term = 2 * (tt.sinh(cd) * tt.sin(dd))
        factor = k0 * norm
        K_large += tt.sum(
            (C1 * cos_term - C2 * sin_term) * factor * cdt, axis=-1
        )
        K_large += tt.sum(
            (C2 * cos_term + C1 * sin_term) * factor * sdt, axis=-1
        )

        # tau < Delta
        edmt = tt.exp(-c * dmt)
        edpt = tt.exp(-c * dpt)
        cos_term = (
            edmt * tt.cos(d * dmt) + edpt * tt.cos(d * dpt) - 2 * k0 * cdt
        )
        sin_term = (
            edmt * tt.sin(d * dmt) + edpt * tt.sin(d * dpt) - 2 * k0 * sdt
        )
        K_small += tt.sum(2 * (a * c + b * d) * c2pd2 * dmt * norm, axis=-1)
        K_small += tt.sum((C1 * cos_term + C2 * sin_term) * norm, axis=-1)

        return tt.switch(tt.le(tau0, dt), K_small, K_large)


class RealTerm(Term):
    r"""The simplest celerite term

    This term has the form

    .. math::

        k(\tau) = a_j\,e^{-c_j\,\tau}

    with the parameters ``a`` and ``c``.

    Strictly speaking, for a sum of terms, the parameter ``a`` could be
    allowed to go negative but since it is somewhat subtle to ensure positive
    definiteness, we recommend keeping both parameters strictly positive.
    Advanced users can build a custom term that has negative coefficients but
    care should be taken to ensure positivity.

    Args:
        tensor a: The amplitude of the term.
        tensor c: The exponent of the term.

    """

    def __init__(self, *, a, c, **kwargs):
        self.a = tt.as_tensor_variable(a).astype("float64")
        self.c = tt.as_tensor_variable(c).astype("float64")
        super().__init__(**kwargs)

    def get_coefficients(self):
        empty = tt.zeros(0, dtype=self.dtype)
        return (
            tt.reshape(self.a, (self.a.size,)),
            tt.reshape(self.c, (self.c.size,)),
            empty,
            empty,
            empty,
            empty,
        )


class ComplexTerm(Term):
    r"""A general celerite term

    This term has the form

    .. math::

        k(\tau) = \frac{1}{2}\,\left[(a_j + b_j)\,e^{-(c_j+d_j)\,\tau}
         + (a_j - b_j)\,e^{-(c_j-d_j)\,\tau}\right]

    with the parameters ``a``, ``b``, ``c``, and ``d``.

    This term will only correspond to a positive definite kernel (on its own)
    if :math:`a_j\,c_j \ge b_j\,d_j`.

    Args:
        tensor a: The real part of amplitude.
        tensor b: The imaginary part of amplitude.
        tensor c: The real part of the exponent.
        tensor d: The imaginary part of exponent.

    """

    def __init__(self, *, a, b, c, d, **kwargs):
        self.a = tt.as_tensor_variable(a).astype("float64")
        self.b = tt.as_tensor_variable(b).astype("float64")
        self.c = tt.as_tensor_variable(c).astype("float64")
        self.d = tt.as_tensor_variable(d).astype("float64")
        super().__init__(**kwargs)

    def get_coefficients(self):
        empty = tt.zeros(0, dtype=self.dtype)
        return (
            empty,
            empty,
            tt.reshape(self.a, (self.a.size,)),
            tt.reshape(self.b, (self.b.size,)),
            tt.reshape(self.c, (self.c.size,)),
            tt.reshape(self.d, (self.d.size,)),
        )


class SHOTerm(Term):
    r"""A term representing a stochastically-driven, damped harmonic oscillator

    The PSD of this term is

    .. math::

        S(\omega) = \sqrt{\frac{2}{\pi}} \frac{S_0\,\omega_0^4}
        {(\omega^2-{\omega_0}^2)^2 + {\omega_0}^2\,\omega^2/Q^2}

    with the parameters ``S0``, ``Q``, and ``w0``.

    Args:
        tensor S0: The parameter :math:`S_0`.
        tensor Q: The parameter :math:`Q`.
        tensor w0: The parameter :math:`\omega_0`.
        tensor Sw4: It can sometimes be more efficient to parameterize the
            amplitude of a SHO kernel using :math:`S_0\,{\omega_0}^4` instead
            of :math:`S_0` directly since :math:`S_0` and :math:`\omega_0` are
            strongly correlated. If provided, ``S0`` will be computed from
            ``Sw4`` and ``w0``.
        tensor S_tot: Another useful parameterization is :math:`S_tot =
            S_0\,\omega_0\,Q`.

    """

    def __init__(
        self, *, w0, Q, S0=None, Sw4=None, S_tot=None, eps=1e-5, **kwargs
    ):
        self.eps = tt.as_tensor_variable(eps).astype("float64")
        self.w0 = tt.as_tensor_variable(w0).astype("float64")
        self.Q = tt.as_tensor_variable(Q).astype("float64")

        if S0 is not None:
            if Sw4 is not None or S_tot is not None:
                raise ValueError("only one of S0, Sw4, and S_tot can be given")
            self.S0 = tt.as_tensor_variable(S0).astype("float64")
        elif Sw4 is not None:
            if S_tot is not None:
                raise ValueError("only one of S0, Sw4, and S_tot can be given")
            self.S0 = (
                tt.as_tensor_variable(Sw4).astype("float64") / self.w0 ** 4
            )
        elif S_tot is not None:
            self.S0 = tt.as_tensor_variable(S_tot).astype("float64") / (
                self.w0 * self.Q
            )
        else:
            raise ValueError("one of S0, Sw4, and S_tot must be given")

        super().__init__(**kwargs)

    def overdamped(self):
        Q = self.Q
        f = tt.sqrt(tt.maximum(1.0 - 4.0 * Q ** 2, self.eps))
        empty = tt.zeros(0, dtype=self.dtype)
        return (
            0.5
            * self.S0
            * self.w0
            * Q
            * tt.stack([1.0 + 1.0 / f, 1.0 - 1.0 / f]),
            0.5 * self.w0 / Q * tt.stack([1.0 - f, 1.0 + f]),
            empty,
            empty,
            empty,
            empty,
        )

    def underdamped(self):
        Q = self.Q
        f = tt.sqrt(tt.maximum(4.0 * Q ** 2 - 1.0, self.eps))
        a = self.S0 * self.w0 * Q
        c = 0.5 * self.w0 / Q
        empty = tt.zeros(0, dtype=self.dtype)
        return (
            empty,
            empty,
            tt.reshape(a, (a.size,)),
            tt.reshape(a / f, (a.size,)),
            tt.reshape(c, (c.size,)),
            tt.reshape(c * f, (c.size,)),
        )

    def get_coefficients(self):
        m = self.Q < 0.5
        return [
            ifelse(m, a, b)
            for a, b in zip(self.overdamped(), self.underdamped())
        ]


class Matern32Term(Term):
    r"""A term that approximates a Matern-3/2 function

    This term is defined as

    .. math::

        k(\tau) = \sigma^2\,\left[
            \left(1+1/\epsilon\right)\,e^{-(1-\epsilon)\sqrt{3}\,\tau/\rho}
            \left(1-1/\epsilon\right)\,e^{-(1+\epsilon)\sqrt{3}\,\tau/\rho}
        \right]

    with the parameters ``sigma`` and ``rho``. The parameter ``eps``
    controls the quality of the approximation since, in the limit
    :math:`\epsilon \to 0` this becomes the Matern-3/2 function

    .. math::

        \lim_{\epsilon \to 0} k(\tau) = \sigma^2\,\left(1+
        \frac{\sqrt{3}\,\tau}{\rho}\right)\,
        \exp\left(-\frac{\sqrt{3}\,\tau}{\rho}\right)

    Args:
        tensor sigma: The parameter :math:`\sigma`.
        tensor rho: The parameter :math:`\rho`.
        eps (Optional[float]): The value of the parameter :math:`\epsilon`.
            (default: `0.01`)

    """

    def __init__(self, *, sigma, rho, eps=0.01, **kwargs):
        self.sigma = tt.as_tensor_variable(sigma).astype("float64")
        self.rho = tt.as_tensor_variable(rho).astype("float64")
        self.eps = tt.as_tensor_variable(eps).astype("float64")
        super().__init__(**kwargs)

    def get_coefficients(self):
        w0 = np.sqrt(3.0) / self.rho
        S0 = self.sigma ** 2 / w0
        empty = tt.zeros(0, dtype=self.dtype)
        return (
            empty,
            empty,
            tt.reshape(w0 * S0, (w0.size,)),
            tt.reshape(w0 * w0 * S0 / self.eps, (w0.size,)),
            tt.reshape(w0, (w0.size,)),
            tt.reshape(self.eps, (w0.size,)),
        )


class RotationTerm(TermSum):
    r"""A mixture of two SHO terms that can be used to model stellar rotation

    This term has two modes in Fourier space: one at ``period`` and one at
    ``0.5 * period``. This can be a good descriptive model for a wide range of
    stochastic variability in stellar time series from rotation to pulsations.

    Args:
        tensor amp: The amplitude of the variability.
        tensor period: The primary period of variability.
        tensor Q0: The quality factor (or really the quality factor minus one
            half) for the secondary oscillation.
        tensor deltaQ: The difference between the quality factors of the first
            and the second modes. This parameterization (if ``deltaQ > 0``)
            ensures that the primary mode alway has higher quality.
        mix: The fractional amplitude of the secondary mode compared to the
            primary. This should probably always be ``0 < mix < 1``.

    """

    def __init__(self, *, amp, Q0, deltaQ, period, mix, **kwargs):
        self.amp = tt.as_tensor_variable(amp).astype("float64")
        self.Q0 = tt.as_tensor_variable(Q0).astype("float64")
        self.deltaQ = tt.as_tensor_variable(deltaQ).astype("float64")
        self.period = tt.as_tensor_variable(period).astype("float64")
        self.mix = tt.as_tensor_variable(mix).astype("float64")

        # One term with a period of period
        Q1 = 0.5 + self.Q0 + self.deltaQ
        w1 = 4 * np.pi * Q1 / (self.period * tt.sqrt(4 * Q1 ** 2 - 1))
        S1 = self.amp / (w1 * Q1)

        # Another term at half the period
        Q2 = 0.5 + self.Q0
        w2 = 8 * np.pi * Q2 / (self.period * tt.sqrt(4 * Q2 ** 2 - 1))
        S2 = self.mix * self.amp / (w2 * Q2)

        super().__init__(
            SHOTerm(S0=S1, w0=w1, Q=Q1), SHOTerm(S0=S2, w0=w2, Q=Q2), **kwargs
        )

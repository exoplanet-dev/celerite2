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
    "OriginalCeleriteTerm",
]

from functools import wraps
from itertools import chain, product

import numpy as np

from . import driver


class Term:
    """The abstract base "term" that is the superclass of all other terms

    Subclasses should overload at least the :func:`Term.get_coefficients`
    method.

    """

    def __add__(self, b):
        return TermSum(self, b)

    def __mul__(self, b):
        return TermProduct(self, b)

    @property
    def terms(self):
        return [self]

    def get_coefficients(self):
        """Compute and return the coefficients for the celerite model

        This should return a 6 element tuple with the following entries:

        .. code-block:: python

            (ar, cr, ac, bc, cc, dc)

        .. note:: All of the returned objects must be arrays, even if they only
            have one element.

        """
        raise NotImplementedError("subclasses must implement this method")

    def get_value(self, tau):
        """Compute the value of the kernel as a function of lag

        Args:
            tau (shape[...]): The lags where the kernel should be evaluated.
        """
        tau = np.abs(np.atleast_1d(tau))
        ar, cr, ac, bc, cc, dc = self.get_coefficients()
        k = np.zeros_like(tau)
        tau = tau[..., None]

        if len(ar):
            k += np.sum(ar * np.exp(-cr * tau), axis=-1)

        if len(ac):
            arg = dc * tau
            k += np.sum(
                np.exp(-cc * tau) * (ac * np.cos(arg) + bc * np.sin(arg)),
                axis=-1,
            )

        return k

    def get_psd(self, omega):
        """Compute the value of the power spectral density for this process

        Args:
            omega (shape[...]): The (angular) frequencies where the power
                should be evaluated.
        """
        w2 = np.atleast_1d(omega) ** 2
        ar, cr, ac, bc, cc, dc = self.get_coefficients()
        psd = np.zeros_like(w2)
        w2 = w2[..., None]

        if len(ar):
            psd += np.sum(ar * cr / (cr ** 2 + w2), axis=-1)

        if len(ac):
            w02 = cc ** 2 + dc ** 2
            psd += np.sum(
                ((ac * cc + bc * dc) * w02 + (ac * cc - bc * dc) * w2)
                / (w2 ** 2 + 2.0 * (cc * cc - dc * dc) * w2 + w02 * w02),
                axis=-1,
            )

        return np.sqrt(2 / np.pi) * psd

    def to_dense(self, x, diag):
        """Evaluate the dense covariance matrix for this term

        Args:
            x (shape[N]): The independent coordinates of the data.
            diag (shape[N]): The diagonal variance of the system.
        """
        K = self.get_value(x[:, None] - x[None, :])
        K[np.diag_indices_from(K)] += diag
        return K

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
            diag (shape[N]): The diagonal variance of the system.
            a (shape[N], optional): The diagonal of the A matrix.
            U (shape[N, J], optional): The first low-rank matrix.
            V (shape[N, J], optional): The second low-rank matrix.
            P (shape[N-1, J], optional): The regularization matrix used for
                numerical stability.

        Raises:
            ValueError: When the inputs are not valid.
        """
        x = np.atleast_1d(x)
        diag = np.atleast_1d(diag)
        if len(x.shape) != 1:
            raise ValueError("'x' must be one-dimensional")
        if x.shape != diag.shape:
            raise ValueError("dimension mismatch")

        ar, cr, ac, bc, cc, dc = self.get_coefficients()

        N = len(x)
        Jr = len(ar)
        Jc = len(ac)
        J = Jr + 2 * Jc

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

        return driver.get_celerite_matrices(
            ar, cr, ac, bc, cc, dc, x, diag, a, U, V, P
        )

    def get_conditional_mean_matrices(self, x, t):
        """Get the matrices needed to compute the conditional mean function

        Args:
            x (shape[N]): The independent coordinates of the data.
            t (shape[M]): The independent coordinates where the predictions
                will be made.
        """
        ar, cr, ac, bc, cc, dc = self.get_coefficients()

        inds = np.searchsorted(x, t)
        _, U_star, V_star, _ = self.get_celerite_matrices(t, t)

        c = np.concatenate([cr] + list(zip(cc, cc)))

        dx = t - x[np.minimum(inds, x.size - 1)]
        U_star *= np.exp(-c[None, :] * dx[:, None])

        dx = x[np.maximum(inds - 1, 0)] - t
        V_star *= np.exp(-c[None, :] * dx[:, None])

        return U_star, V_star, inds

    def dot(self, x, diag, y):
        """Apply a matrix-vector or matrix-matrix product

        Args:
            x (shape[N]): The independent coordinates of the data.
            diag (shape[N]): The diagonal variance of the system.
            y (shape[N] or shape[N, K]): The target of vector or matrix for
                this operation.
        """
        a, U, V, P = self.get_celerite_matrices(x, diag)

        y = np.atleast_1d(y)
        if y.shape[0] != len(a):
            raise ValueError("dimension mismatch")

        z = np.empty(y.shape)
        return driver.matmul(a, U, V, P, y, z)


class TermSum(Term):
    """A sum of multiple :class:`Term` objects

    The resulting kernel function is the sum of the functions and the width of
    the resulting low-rank representation will be the sum of widths for each
    of the terms.

    Args:
        *terms: Any number of :class:`Term` subclasses to add together.
    """

    def __init__(self, *terms):
        if any(isinstance(term, TermConvolution) for term in terms):
            raise TypeError(
                "You cannot perform operations on an TermConvolution, it must "
                "be the outer term in the kernel"
            )
        self._terms = terms

    @property
    def terms(self):
        return self._terms

    def get_coefficients(self):
        coeffs = (t.get_coefficients() for t in self.terms)
        return tuple(np.concatenate(c) for c in zip(*coeffs))


class TermProduct(Term):
    """A product of two :class:`Term` objects

    The resulting kernel function is the product of the two functions and the
    resulting low-rank representation will in general be wider than the sum of
    the two widths.

    Args:
        term1 (Term): The left term.
        term2 (Term): The right term.
    """

    def __init__(self, term1, term2):
        int1 = isinstance(term1, TermConvolution)
        int2 = isinstance(term2, TermConvolution)
        if int1 or int2:
            raise TypeError(
                "You cannot perform operations on an TermConvolution, it must "
                "be the outer term in the kernel"
            )
        self.term1 = term1
        self.term2 = term2

    def get_coefficients(self):
        c1 = self.term1.get_coefficients()
        c2 = self.term2.get_coefficients()

        # First compute real terms
        ar = []
        cr = []
        gen = product(zip(c1[0], c1[1]), zip(c2[0], c2[1]))
        for i, ((aj, cj), (ak, ck)) in enumerate(gen):
            ar.append(aj * ak)
            cr.append(cj + ck)

        # Then the complex terms
        ac = []
        bc = []
        cc = []
        dc = []

        # real * complex
        gen = product(zip(c1[0], c1[1]), zip(*(c2[2:])))
        gen = chain(gen, product(zip(c2[0], c2[1]), zip(*(c1[2:]))))
        for i, ((aj, cj), (ak, bk, ck, dk)) in enumerate(gen):
            ac.append(aj * ak)
            bc.append(aj * bk)
            cc.append(cj + ck)
            dc.append(dk)

        # complex * complex
        gen = product(zip(*(c1[2:])), zip(*(c2[2:])))
        for i, ((aj, bj, cj, dj), (ak, bk, ck, dk)) in enumerate(gen):
            ac.append(0.5 * (aj * ak + bj * bk))
            bc.append(0.5 * (bj * ak - aj * bk))
            cc.append(cj + ck)
            dc.append(dj - dk)

            ac.append(0.5 * (aj * ak - bj * bk))
            bc.append(0.5 * (bj * ak + aj * bk))
            cc.append(cj + ck)
            dc.append(dj + dk)

        return list(map(np.array, (ar, cr, ac, bc, cc, dc)))


class TermDiff(Term):
    """Take the first derivative of a term with respect to the lag

    Args:
        term (Term): The term to differentiate.
    """

    def __init__(self, term):
        if isinstance(term, TermConvolution):
            raise TypeError(
                "You cannot perform operations on an TermConvolution, it must "
                "be the outer term in the kernel"
            )
        self.term = term

    def get_coefficients(self):
        coeffs = self.term.get_coefficients()
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


class TermConvolution(Term):
    """A term corresponding to the integral of another term over a boxcar

    The process produced by this term is equivalent to the process produced by
    convolving the base process with a boxcar of length ``delta``. This can be
    useful, for example, when taking exposure time integration into account.

    Args:
        term (Term): The term describing the base process.
        delta (float): The width of the boxcar filter (for example, the
            exposure time).
    """

    def __init__(self, term, delta):
        self.term = term
        self.delta = float(delta)

    def get_celerite_matrices(
        self, x, diag, *, a=None, U=None, V=None, P=None
    ):
        dt = self.delta
        ar, cr, a, b, c, d = self.term.get_coefficients()

        # Real part
        cd = cr * dt
        delta_diag = 2 * np.sum(ar * (cd - np.sinh(cd)) / cd ** 2)

        # Complex part
        cd = c * dt
        dd = d * dt
        c2 = c ** 2
        d2 = d ** 2
        c2pd2 = c2 + d2
        C1 = a * (c2 - d2) + 2 * b * c * d
        C2 = b * (c2 - d2) - 2 * a * c * d
        norm = (dt * c2pd2) ** 2
        sinh = np.sinh(cd)
        cosh = np.cosh(cd)
        delta_diag += 2 * np.sum(
            (
                C2 * cosh * np.sin(dd)
                - C1 * sinh * np.cos(dd)
                + (a * c + b * d) * dt * c2pd2
            )
            / norm
        )

        new_diag = diag + delta_diag

        return super().get_celerite_matrices(x, new_diag, a=a, U=U, V=V, P=P)

    def get_coefficients(self):
        ar, cr, a, b, c, d = self.term.get_coefficients()

        # Real componenets
        crd = cr * self.delta
        coeffs = [2 * ar * (np.cosh(crd) - 1) / crd ** 2, cr]

        # Imaginary coefficients
        cd = c * self.delta
        dd = d * self.delta
        c2 = c ** 2
        d2 = d ** 2
        factor = 2.0 / (self.delta * (c2 + d2)) ** 2
        cos_term = np.cosh(cd) * np.cos(dd) - 1
        sin_term = np.sinh(cd) * np.sin(dd)

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
        omega = np.atleast_1d(omega)
        psd0 = self.term.get_psd(omega)
        arg = 0.5 * self.delta * omega
        sinc = np.ones_like(arg)
        m = np.abs(arg) > 0.0
        sinc[m] = np.sin(arg[m]) / arg[m]
        return psd0 * sinc ** 2

    def get_value(self, tau0):
        dt = self.delta
        ar, cr, a, b, c, d = self.term.get_coefficients()

        # Format the lags correctly
        tau0 = np.abs(np.atleast_1d(tau0))
        tau = tau0[..., None]

        # Precompute some factors
        dpt = dt + tau
        dmt = dt - tau

        # Real parts:
        # tau > Delta
        crd = cr * dt
        cosh = np.cosh(crd)
        norm = 2 * ar / crd ** 2
        K_large = np.sum(norm * (cosh - 1) * np.exp(-cr * tau), axis=-1)

        # tau < Delta
        crdmt = cr * dmt
        K_small = K_large + np.sum(norm * (crdmt - np.sinh(crdmt)), axis=-1)

        # Complex part
        cd = c * dt
        dd = d * dt
        c2 = c ** 2
        d2 = d ** 2
        c2pd2 = c2 + d2
        C1 = a * (c2 - d2) + 2 * b * c * d
        C2 = b * (c2 - d2) - 2 * a * c * d
        norm = 1.0 / (dt * c2pd2) ** 2
        k0 = np.exp(-c * tau)
        cdt = np.cos(d * tau)
        sdt = np.sin(d * tau)

        # For tau > Delta
        cos_term = 2 * (np.cosh(cd) * np.cos(dd) - 1)
        sin_term = 2 * (np.sinh(cd) * np.sin(dd))
        factor = k0 * norm
        K_large += np.sum(
            (C1 * cos_term - C2 * sin_term) * factor * cdt, axis=-1
        )
        K_large += np.sum(
            (C2 * cos_term + C1 * sin_term) * factor * sdt, axis=-1
        )

        # tau < Delta
        edmt = np.exp(-c * dmt)
        edpt = np.exp(-c * dpt)
        cos_term = (
            edmt * np.cos(d * dmt) + edpt * np.cos(d * dpt) - 2 * k0 * cdt
        )
        sin_term = (
            edmt * np.sin(d * dmt) + edpt * np.sin(d * dpt) - 2 * k0 * sdt
        )
        K_small += np.sum(2 * (a * c + b * d) * c2pd2 * dmt * norm, axis=-1)
        K_small += np.sum((C1 * cos_term + C2 * sin_term) * norm, axis=-1)

        mask = tau0 >= dt
        K_small[mask] = K_large[mask]
        return K_small


class RealTerm(Term):
    r"""The simplest celerite term

    .. warning:: You should only use this term if you really know what you're
        doing because it will generally behave poorly. Check out
        :class:`SHOTerm` instead!

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
        a: The amplitude of the term.
        c: The exponent of the term.
    """

    @staticmethod
    def get_test_parameters():
        return dict(a=1.5, c=0.7)

    def __init__(self, *, a, c):
        self.a = float(a)
        self.c = float(c)

    def get_coefficients(self):
        e = np.empty(0)
        return np.array([self.a]), np.array([self.c]), e, e, e, e


class ComplexTerm(Term):
    r"""A general celerite term

    .. warning:: You should only use this term if you really know what you're
        doing because it will generally behave poorly. Check out
        :class:`SHOTerm` instead!

    This term has the form

    .. math::

        k(\tau) = \frac{1}{2}\,\left[(a_j + b_j)\,e^{-(c_j+d_j)\,\tau}
         + (a_j - b_j)\,e^{-(c_j-d_j)\,\tau}\right]

    with the parameters ``a``, ``b``, ``c``, and ``d``.

    This term will only correspond to a positive definite kernel (on its own)
    if :math:`a_j\,c_j \ge b_j\,d_j`.

    Args:
        a: The real part of amplitude.
        b: The imaginary part of amplitude.
        c: The real part of the exponent.
        d: The imaginary part of exponent.
    """

    @staticmethod
    def get_test_parameters():
        return dict(a=1.5, b=0.7, c=0.7, d=0.5)

    def __init__(self, *, a, b, c, d):
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.d = float(d)

    def get_coefficients(self):
        e = np.empty(0)
        return (
            e,
            e,
            np.array([self.a]),
            np.array([self.b]),
            np.array([self.c]),
            np.array([self.d]),
        )


class handle_parameter_spec:
    def __init__(self, mapper):
        self.mapper = mapper

    def __call__(self, to_wrap):
        @wraps(to_wrap)
        def wrapped(target, *args, **kwargs):
            for param, alt in target.__parameter_spec__:
                all_names = set([param]) | set(name for name, _ in alt)
                if sum(int(n in kwargs) for n in all_names) != 1:
                    raise ValueError(
                        "exactly one of {0} must be defined".format(
                            list(all_names)
                        )
                    )
                if param in kwargs:
                    setattr(target, param, self.mapper(kwargs.pop(param)))
                else:
                    for name, func in alt:
                        if name in kwargs:
                            setattr(
                                target,
                                param,
                                func(target, self.mapper(kwargs.pop(name))),
                            )
                            break

            return to_wrap(target, *args, **kwargs)

        return wrapped


class SHOTerm(Term):
    r"""A term representing a stochastically-driven, damped harmonic oscillator

    The PSD of this term is

    .. math::

        S(\omega) = \sqrt{\frac{2}{\pi}} \frac{S_0\,\omega_0^4}
        {(\omega^2-{\omega_0}^2)^2 + {\omega_0}^2\,\omega^2/Q^2}

    with the parameters ``S0``, ``Q``, and ``w0``.

    This implementation also supports the following reparameterizations that
    can be easier to use and reason about:

    1. ``rho``, the undamped period of the oscillator, defined as :math:`\rho
       = 2\,\pi / \omega_0`,
    2. ``tau``, the damping timescale of the process, defined as :math:`\tau =
       2\,Q / \omega_0`, and
    3. ``sigma``, the standard deviation of the process, defined as
       :math:`\sigma = \sqrt{S_0\,\omega_0\,Q}`.

    Args:
        w0: The undamped angular frequency, :math:`\omega_0` above.
        rho: Alternative parameterization for ``w0`` as described above.
        Q: The quality factor, :math:`Q` above.
        tau: Alternative parameterization for ``Q`` as described above.
        S0: The power at :math:`\omega = 0`, :math:`S_0` above.
        sigma: Alternative parameterization for ``S0`` as described above.
        eps (optional): A regularization parameter used for numerical stability
            when computing :math:`\sqrt{1-4\,Q^2}` or :math:`\sqrt{4\,Q^2-1}`.
    """

    __parameter_spec__ = (
        ("w0", (("rho", lambda self, rho: 2 * np.pi / rho),)),
        ("Q", (("tau", lambda self, tau: 0.5 * self.w0 * tau),)),
        (
            "S0",
            (("sigma", lambda self, sigma: sigma ** 2 / (self.w0 * self.Q)),),
        ),
    )

    @staticmethod
    def get_test_parameters():
        return dict(sigma=1.5, tau=2.345, rho=3.4)

    @handle_parameter_spec(float)
    def __init__(self, *, eps=1e-5):
        self.eps = float(eps)

    def overdamped(self):
        Q = self.Q
        f = np.sqrt(np.maximum(1.0 - 4.0 * Q ** 2, self.eps))
        e = np.empty(0)
        return (
            0.5
            * self.S0
            * self.w0
            * Q
            * np.array([1.0 + 1.0 / f, 1.0 - 1.0 / f]),
            0.5 * self.w0 / Q * np.array([1.0 - f, 1.0 + f]),
            e,
            e,
            e,
            e,
        )

    def underdamped(self):
        Q = self.Q
        f = np.sqrt(np.maximum(4.0 * Q ** 2 - 1.0, self.eps))
        a = self.S0 * self.w0 * Q
        c = 0.5 * self.w0 / Q
        e = np.empty(0)
        return (
            e,
            e,
            np.array([a]),
            np.array([a / f]),
            np.array([c]),
            np.array([c * f]),
        )

    def get_coefficients(self):
        return self.overdamped() if self.Q < 0.5 else self.underdamped()


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

    This term should be used with care since there could be numerical issues
    with this approximation.

    Args:
        sigma: The parameter :math:`\sigma`.
        rho: The parameter :math:`\rho`.
        eps (optional): The value of the parameter :math:`\epsilon`.
    """

    @staticmethod
    def get_test_parameters():
        return dict(sigma=1.5, rho=2.345)

    def __init__(self, *, sigma, rho, eps=0.01):
        self.sigma = float(sigma)
        self.rho = float(rho)
        self.eps = float(eps)

    def get_coefficients(self):
        w0 = np.sqrt(3) / self.rho
        S0 = self.sigma ** 2 / w0
        e = np.empty(0)
        return (
            e,
            e,
            np.array([w0 * S0]),
            np.array([w0 ** 2 * S0 / self.eps]),
            np.array([w0]),
            np.array([self.eps]),
        )


class RotationTerm(TermSum):
    r"""A mixture of two SHO terms that can be used to model stellar rotation

    This term has two modes in Fourier space: one at ``period`` and one at
    ``0.5 * period``. This can be a good descriptive model for a wide range of
    stochastic variability in stellar time series from rotation to pulsations.

    More precisely, the parameters of the two :class:`SHOTerm` terms are
    defined as

    .. math::

        Q_1 = 1/2 + Q_0 + \delta Q \\
        \omega_1 = \frac{4\,\pi\,Q_1}{P\,\sqrt{4\,Q_1^2 - 1}} \\
        S_1 = \frac{\sigma^2}{(1 + f)\,\omega_1\,Q_1}

    for the primary term, and

    .. math::

        Q_2 = 1/2 + Q_0 \\
        \omega_2 = \frac{8\,\pi\,Q_1}{P\,\sqrt{4\,Q_1^2 - 1}} \\
        S_2 = \frac{f\,\sigma^2}{(1 + f)\,\omega_2\,Q_2}

    for the secondary term.

    Args:
        sigma: The standard deviation of the process.
        period: The primary period of variability.
        Q0: The quality factor (or really the quality factor minus one half;
            this keeps the system underdamped) for the secondary oscillation.
        dQ: The difference between the quality factors of the first and the
            second modes. This parameterization (if ``dQ > 0``) ensures that
            the primary mode alway has higher quality.
        f: The fractional amplitude of the secondary mode compared to the
            primary. This should probably always be ``0 < f < 1``, but that
            is not enforced.
    """

    @staticmethod
    def get_test_parameters():
        return dict(sigma=1.5, period=3.45, Q0=1.3, dQ=1.05, f=0.5)

    def __init__(self, *, sigma, period, Q0, dQ, f):
        self.sigma = float(sigma)
        self.period = float(period)
        self.Q0 = float(Q0)
        self.dQ = float(dQ)
        self.f = float(f)

        self.amp = self.sigma ** 2 / (1 + self.f)

        # One term with a period of period
        Q1 = 0.5 + self.Q0 + self.dQ
        w1 = 4 * np.pi * Q1 / (self.period * np.sqrt(4 * Q1 ** 2 - 1))
        S1 = self.amp / (w1 * Q1)

        # Another term at half the period
        Q2 = 0.5 + self.Q0
        w2 = 8 * np.pi * Q2 / (self.period * np.sqrt(4 * Q2 ** 2 - 1))
        S2 = self.f * self.amp / (w2 * Q2)

        super().__init__(
            SHOTerm(S0=S1, w0=w1, Q=Q1), SHOTerm(S0=S2, w0=w2, Q=Q2)
        )


class OriginalCeleriteTerm(Term):
    """A wrapper around terms defined using the original celerite package

    Args:
        term (celerite.terms.Term): The term defined using ``celerite``.
    """

    def __init__(self, term):
        self.term = term

    def get_coefficients(self):
        return self.term.get_all_coefficients()

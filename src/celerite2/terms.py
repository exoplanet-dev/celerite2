# -*- coding: utf-8 -*-

__all__ = [
    "Term",
    # "TermSum",
    # "TermProduct",
    # "TermDiff",
    # "TermConvolution",
    # "RealTerm",
    # "ComplexTerm",
    # "SHOTerm",
    # "Matern32Term",
    # "RotationTerm",
    # "OriginalCeleriteTerm",
]


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
        self, x, diag, *, c=None, a=None, U=None, V=None
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

        if c is None:
            c = np.empty(J)
        else:
            c.resize(J, refcheck=False)
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

        c[:Jr] = cr
        c[Jr::2] = cc
        c[Jr + 1 :: 2] = cc
        a, U, V = driver.get_celerite_matrices(
            ar, ac, bc, dc, x, diag, a, U, V
        )
        return c, a, U, V

    def dot(self, x, diag, y):
        """Apply a matrix-vector or matrix-matrix product

        Args:
            x (shape[N]): The independent coordinates of the data.
            diag (shape[N]): The diagonal variance of the system.
            y (shape[N] or shape[N, K]): The target of vector or matrix for
                this operation.
        """
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        if y.shape[0] != x.shape[0]:
            raise ValueError("Dimension mismatch")

        is_vector = False
        if y.ndim == 1:
            is_vector = True
            y = y[:, None]
        if y.ndim != 2:
            raise ValueError("'y' can only be a vector or matrix")

        c, a, U, V = self.get_celerite_matrices(x, diag)
        z = y * a[:, None]
        z = driver.matmul_lower(x, c, U, V, y, z)
        z = driver.matmul_upper(x, c, U, V, y, z)

        if is_vector:
            return z[:, 0]
        return z

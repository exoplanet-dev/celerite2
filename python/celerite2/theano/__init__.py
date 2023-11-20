__all__ = ["terms", "GaussianProcess"]

import warnings

warnings.warn(
    "The `celerite2.theano` submodule is deprecated; "
    "use `celerite2.pymc3` instead",
    DeprecationWarning,
)

from celerite2.pymc3.celerite2 import GaussianProcess
from celerite2.theano import terms

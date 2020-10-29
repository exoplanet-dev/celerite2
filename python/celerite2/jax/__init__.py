# -*- coding: utf-8 -*-

import logging

logger = logging.getLogger(__name__)

from jax.config import config  # noqa isort:skip

if not config.read("jax_enable_x64"):
    logger.warning(
        "celerite2.jax only works with dtype float64. "
        "To enable, run (before importing jax or celerite2.jax):\n"
        ">>> from jax.config import config\n"
        ">>> config.update('jax_enable_x64', True)"
    )


__all__ = ["terms", "GaussianProcess", "CeleriteNormal"]

from . import terms  # noqa isort:skip
from .celerite2 import GaussianProcess  # noqa isort:skip
from .distribution import CeleriteNormal

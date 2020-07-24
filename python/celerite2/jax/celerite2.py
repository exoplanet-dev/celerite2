# -*- coding: utf-8 -*-

__all__ = ["GaussianProcess"]
import numpy as np

from ..celerite2 import ConstantMean
from ..celerite2 import GaussianProcess as SuperGaussianProcess
from . import ops


class GaussianProcess(SuperGaussianProcess):
    pass

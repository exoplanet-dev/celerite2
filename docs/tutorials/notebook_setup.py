"""isort:skip_file"""

get_ipython().magic('config InlineBackend.figure_format = "retina"')

import os
import logging
import warnings

import theano
import matplotlib.pyplot as plt

# import multiprocessing as mp
# mp.set_start_method("fork")


# Compile bugz
theano.config.gcc.cxxflags = "-Wno-c++11-narrowing"

# Remove when Theano is updated
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Remove when arviz is updated
warnings.filterwarnings("ignore", category=UserWarning)


logger = logging.getLogger("theano.gof.compilelock")
logger.setLevel(logging.ERROR)

plt.style.use("default")
plt.rcParams["savefig.dpi"] = 100
plt.rcParams["figure.dpi"] = 100
plt.rcParams["font.size"] = 16
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Liberation Sans"]
plt.rcParams["font.cursive"] = ["Liberation Sans"]
plt.rcParams["mathtext.fontset"] = "custom"

# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %matplotlib inline

# # A tutorial

# +
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

t = np.sort(
    np.append(np.random.uniform(0, 3.8, 57), np.random.uniform(5.5, 10, 68),)
)  # The input coordinates must be sorted
yerr = np.random.uniform(0.08, 0.22, len(t))
y = (
    0.2 * (t - 5)
    + np.sin(3 * t + 0.1 * (t - 5) ** 2)
    + yerr * np.random.randn(len(t))
)

true_t = np.linspace(0, 10, 5000)
true_y = 0.2 * (true_t - 5) + np.sin(3 * true_t + 0.1 * (true_t - 5) ** 2)

plt.plot(true_t, true_y, "k", lw=1.5, alpha=0.3)
plt.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(0, 10)
plt.ylim(-2.5, 2.5)

# +
from jax.config import config

config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import celerite2.jax as cl2


# +
def get_gp(params):
    log_S1, log_w1, log_S2, log_w2, log_Q2 = params
    kernel = cl2.terms.SHOTerm(
        S0=jnp.exp(log_S1), w0=jnp.exp(log_w1), Q=jnp.float64(1.0 / np.sqrt(2))
    )
    kernel += cl2.terms.SHOTerm(
        S0=jnp.exp(log_S2), w0=jnp.exp(log_w2), Q=jnp.exp(log_Q2)
    )
    gp = cl2.GaussianProcess(kernel)
    gp.compute(t, yerr=yerr)
    return gp


def nll(*args):
    return -get_gp(*args).log_likelihood(y)


grad = jax.grad(nll)


def grad_nll(*args):
    return jnp.stack(grad(*args))


# +
from scipy.optimize import minimize

soln = minimize(nll, [0.5, 3.0, 0.1, 0.5, 1.0], jac=grad_nll)
print(soln)
# -

x = np.linspace(0, 10, 5000)
pred_mean, pred_var = get_gp(soln.x).predict(y, x, return_var=True)
pred_std = np.sqrt(pred_var)

color = "#ff7f0e"
plt.plot(true_t, true_y, "k", lw=1.5, alpha=0.3)
plt.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)
plt.plot(x, pred_mean, color=color)
plt.fill_between(
    x,
    pred_mean + pred_std,
    pred_mean - pred_std,
    color=color,
    alpha=0.3,
    edgecolor="none",
)
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(0, 10)
plt.ylim(-2.5, 2.5)

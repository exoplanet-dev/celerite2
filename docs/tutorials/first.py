# -*- coding: utf-8 -*-
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

# # Getting started
#
# This tutorial is based on [the quickstart example in the celerite documentation](https://celerite.readthedocs.io/en/stable/tutorials/first/), but it has been updated to work with *celerite2*.
#
# For this tutorial, we’re going to fit a Gaussian Process (GP) model to a simulated dataset with quasiperiodic oscillations.
# We’re also going to leave a gap in the simulated data and we’ll use the GP model to predict what we would have observed for those "missing" datapoints.
#
# To start, here’s some code to simulate the dataset:

# +
# %run notebook_setup

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
plt.xlabel("x [day]")
plt.ylabel("y [ppm]")
plt.xlim(0, 10)
plt.ylim(-2.5, 2.5)
_ = plt.title("simulated data")
# -

# Now, let's fit this dataset using a mixture of [SHOTerm](../api/python.rst#celerite2.terms.SHOTerm) terms: one quasi-periodic component and one non-periodic component.
# First let's set up an initial model to see how it looks:

# +
import celerite2
from celerite2 import terms

# Quasi-periodic term
term1 = terms.SHOTerm(sigma=1.0, rho=1.0, tau=10.0)

# Non-periodic component
term2 = terms.SHOTerm(sigma=1.0, rho=5.0, Q=0.25)
kernel = term1 + term2

# Setup the GP
gp = celerite2.GaussianProcess(kernel, mean=0.0)
gp.compute(t, yerr=yerr)

print("Initial log likelihood: {0}".format(gp.log_likelihood(y)))


# -

# Let's look at the underlying power spectral density of this initial model:

# +
def plot_psd(gp):
    freq = np.linspace(1.0 / 8, 1.0 / 0.3, 1000)
    omega = 2 * np.pi * freq
    for n, term in enumerate(gp.kernel.terms):
        plt.loglog(freq, term.get_psd(omega), label="term {0}".format(n + 1))
    plt.loglog(freq, gp.kernel.get_psd(omega), ":k", label="full model")
    plt.legend()
    plt.xlabel("frequency [1 / day]")
    plt.ylabel("power [day ppt$^2$]")


plt.title("initial psd")
plot_psd(gp)


# -

# And then we can also plot the prediction that this model makes for the missing data and compare it to the truth:

# +
def plot_prediction(gp):
    mu, variance = gp.predict(y, t=true_t, return_var=True)
    sigma = np.sqrt(variance)

    plt.plot(true_t, true_y, "k", lw=1.5, alpha=0.3, label="data")
    plt.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0, label="truth")

    plt.plot(true_t, mu, label="prediction")
    plt.fill_between(true_t, mu - sigma, mu + sigma, color="C0", alpha=0.2)

    plt.xlabel("x [day]")
    plt.ylabel("y [ppm]")
    plt.xlim(0, 10)
    plt.ylim(-2.5, 2.5)
    plt.legend()


plt.title("initial prediction")
plot_prediction(gp)
# -

# Ok, that looks pretty terrible, but we can get a better fit by numerically maximizing the likelihood using [scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html):

# +
from scipy.optimize import minimize


def set_params(params, gp):
    gp.mean = params[0]
    theta = np.exp(params[1:])
    gp.kernel = terms.SHOTerm(
        sigma=theta[0], rho=theta[1], tau=theta[2]
    ) + terms.SHOTerm(sigma=theta[3], rho=theta[4], Q=0.25)
    return gp


def neg_log_like(params, gp):
    gp = set_params(params, gp)
    gp.recompute(quiet=True)
    return -gp.log_likelihood(y)


initial_params = [0.0, 0.0, 0.0, np.log(10.0), 0.0, np.log(5.0)]
soln = minimize(neg_log_like, initial_params, method="L-BFGS-B", args=(gp,))
opt_gp = set_params(soln.x, gp)
soln
# -

# Now let's make the same plots for the maximum likelihood model:

# +
plt.figure()
plt.title("maximum likelihood psd")
plot_psd(opt_gp)

plt.figure()
plt.title("maximum likelihood prediction")
plot_prediction(opt_gp)
# -

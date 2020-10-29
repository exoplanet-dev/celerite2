# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + nbsphinx="hidden"
# %matplotlib inline

# + nbsphinx="hidden"
# %run notebook_setup
# -

# # Getting started
#
# This tutorial is based on [the quickstart example in the celerite documentation](https://celerite.readthedocs.io/en/stable/tutorials/first/), but it has been updated to work with *celerite2*.
#
# For this tutorial, we’re going to fit a Gaussian Process (GP) model to a simulated dataset with quasiperiodic oscillations.
# We’re also going to leave a gap in the simulated data and we’ll use the GP model to predict what we would have observed for those "missing" datapoints.
#
# To start, here’s some code to simulate the dataset:

# +
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

t = np.sort(
    np.append(
        np.random.uniform(0, 3.8, 57),
        np.random.uniform(5.5, 10, 68),
    )
)  # The input coordinates must be sorted
yerr = np.random.uniform(0.08, 0.22, len(t))
y = (
    0.2 * (t - 5)
    + np.sin(3 * t + 0.1 * (t - 5) ** 2)
    + yerr * np.random.randn(len(t))
)

true_t = np.linspace(0, 10, 500)
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
freq = np.linspace(1.0 / 8, 1.0 / 0.3, 500)
omega = 2 * np.pi * freq


def plot_psd(gp):
    for n, term in enumerate(gp.kernel.terms):
        plt.loglog(freq, term.get_psd(omega), label="term {0}".format(n + 1))
    plt.loglog(freq, gp.kernel.get_psd(omega), ":k", label="full model")
    plt.xlim(freq.min(), freq.max())
    plt.legend()
    plt.xlabel("frequency [1 / day]")
    plt.ylabel("power [day ppt$^2$]")


plt.title("initial psd")
plot_psd(gp)


# -

# And then we can also plot the prediction that this model makes for the missing data and compare it to the truth:

# +
def plot_prediction(gp):
    plt.plot(true_t, true_y, "k", lw=1.5, alpha=0.3, label="data")
    plt.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0, label="truth")

    if gp:
        mu, variance = gp.predict(y, t=true_t, return_var=True)
        sigma = np.sqrt(variance)
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

# Ok, that looks pretty terrible, but we can get a better fit by numerically maximizing the likelihood as described in the following section.
#
# ## Maximum likelihood
#
# In this section, we'll improve our initial GP model by maximizing the likelihood function for the parameters of the kernel, the mean, and a "jitter" (a constant variance term added to the diagonal of our covariance matrix).
# To do this, we'll use the numerical optimization routine from [scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html):

# +
from scipy.optimize import minimize


def set_params(params, gp):
    gp.mean = params[0]
    theta = np.exp(params[1:])
    gp.kernel = terms.SHOTerm(
        sigma=theta[0], rho=theta[1], tau=theta[2]
    ) + terms.SHOTerm(sigma=theta[3], rho=theta[4], Q=0.25)
    gp.compute(t, diag=yerr ** 2 + theta[5], quiet=True)
    return gp


def neg_log_like(params, gp):
    gp = set_params(params, gp)
    return -gp.log_likelihood(y)


initial_params = [0.0, 0.0, 0.0, np.log(10.0), 0.0, np.log(5.0), np.log(0.01)]
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

# These predictions are starting to look much better!
#
# ## Posterior inference using emcee
#
# Now, to get a sense for the uncertainties on our model, let's use Markov chain Monte Carlo (MCMC) to numerically estimate the posterior expectations of the model.
# In this first example, we'll use the [emcee](https://emcee.readthedocs.io) package to run our MCMC.
# Our likelihood function is the same as the one we used in the previous section, but we'll also choose a wide normal prior on each of our parameters.

# +
import emcee

prior_sigma = 2.0


def log_prob(params, gp):
    gp = set_params(params, gp)
    return (
        gp.log_likelihood(y) - 0.5 * np.sum((params / prior_sigma) ** 2),
        gp.kernel.get_psd(omega),
    )


np.random.seed(5693854)
coords = soln.x + 1e-5 * np.random.randn(32, len(soln.x))
sampler = emcee.EnsembleSampler(
    coords.shape[0], coords.shape[1], log_prob, args=(gp,)
)
state = sampler.run_mcmc(coords, 2000, progress=True)
sampler.reset()
state = sampler.run_mcmc(state, 5000, progress=True)
# -

# After running our MCMC, we can plot the predictions that the model makes for a handful of samples from the chain.
# This gives a qualitative sense of the uncertainty in the predictions.

# +
chain = sampler.get_chain(discard=100, flat=True)

for sample in chain[np.random.randint(len(chain), size=50)]:
    gp = set_params(sample, gp)
    plt.plot(true_t, gp.sample_conditional(y, true_t), color="C0", alpha=0.1)

plt.title("posterior prediction")
plot_prediction(None)
# -

# Similarly, we can plot the posterior expectation for the power spectral density:

# +
psds = sampler.get_blobs(discard=100, flat=True)

q = np.percentile(psds, [16, 50, 84], axis=0)

plt.loglog(freq, q[1], color="C0")
plt.fill_between(freq, q[0], q[2], color="C0", alpha=0.1)

plt.xlim(freq.min(), freq.max())
plt.xlabel("frequency [1 / day]")
plt.ylabel("power [day ppt$^2$]")
_ = plt.title("posterior psd using emcee")
# -

# ## Posterior inference using PyMC3
#
# *celerite2* also includes support for probabilistic modeling using PyMC3, and we can implement the same model from above as follows:

# +
import pymc3 as pm

import celerite2.theano
from celerite2.theano import terms as theano_terms

with pm.Model() as model:

    mean = pm.Normal("mean", mu=0.0, sigma=prior_sigma)
    jitter = pm.Lognormal("jitter", mu=0.0, sigma=prior_sigma)

    sigma1 = pm.Lognormal("sigma1", mu=0.0, sigma=prior_sigma)
    rho1 = pm.Lognormal("rho1", mu=0.0, sigma=prior_sigma)
    tau = pm.Lognormal("tau", mu=0.0, sigma=prior_sigma)
    term1 = theano_terms.SHOTerm(sigma=sigma1, rho=rho1, tau=tau)

    sigma2 = pm.Lognormal("sigma2", mu=0.0, sigma=prior_sigma)
    rho2 = pm.Lognormal("rho2", mu=0.0, sigma=prior_sigma)
    term2 = theano_terms.SHOTerm(sigma=sigma2, rho=rho2, Q=0.25)

    kernel = term1 + term2
    gp = celerite2.theano.GaussianProcess(kernel, mean=mean)
    gp.compute(t, diag=yerr ** 2 + jitter, quiet=True)
    gp.marginal("obs", observed=y)

    pm.Deterministic("psd", kernel.get_psd(omega))

    trace = pm.sample(
        tune=1000,
        draws=1000,
        target_accept=0.8,
        init="adapt_full",
        cores=2,
        chains=2,
        random_seed=34923,
    )
# -

# Like before, we can plot the posterior estimate of the power spectrum to show that the results are qualitatively similar:

# +
psds = trace["psd"]

q = np.percentile(psds, [16, 50, 84], axis=0)

plt.loglog(freq, q[1], color="C0")
plt.fill_between(freq, q[0], q[2], color="C0", alpha=0.1)

plt.xlim(freq.min(), freq.max())
plt.xlabel("frequency [1 / day]")
plt.ylabel("power [day ppt$^2$]")
_ = plt.title("posterior psd using PyMC3")
# -

# ## Posterior inference using numpyro
#
# Since celerite2 includes support for JAX as well as Theano, you can also use tools like [numpyro](https://github.com/pyro-ppl/numpyro) for inference.
# The following is similar to previous PyMC3 example, but the main difference is that (for technical reasons related to how JAX works) `SHOTerm`s cannot be used in combination with `jax.jit`, so we need to explicitly specify the terms as "underdamped" (`UnderdampedSHOTerm`) or "overdamped" (`OverdampedSHOTerm`).

# +
from jax.config import config

config.update("jax_enable_x64", True)

from jax import random

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

import celerite2.jax
from celerite2.jax import terms as jax_terms


def numpyro_model(t, yerr, y=None):
    mean = numpyro.sample("mean", dist.Normal(0.0, prior_sigma))
    jitter = numpyro.sample("jitter", dist.LogNormal(0.0, prior_sigma))

    sigma1 = numpyro.sample("sigma1", dist.LogNormal(0.0, prior_sigma))
    rho1 = numpyro.sample("rho1", dist.LogNormal(0.0, prior_sigma))
    tau = numpyro.sample("tau", dist.LogNormal(0.0, prior_sigma))
    term1 = jax_terms.UnderdampedSHOTerm(sigma=sigma1, rho=rho1, tau=tau)

    sigma2 = numpyro.sample("sigma2", dist.LogNormal(0.0, prior_sigma))
    rho2 = numpyro.sample("rho2", dist.LogNormal(0.0, prior_sigma))
    term2 = jax_terms.OverdampedSHOTerm(sigma=sigma2, rho=rho2, Q=0.25)

    kernel = term1 + term2
    gp = celerite2.jax.GaussianProcess(kernel, mean=mean)
    gp.compute(t, diag=yerr ** 2 + jitter, check_sorted=False)

    numpyro.sample("obs", gp.numpyro_dist(), obs=y)
    numpyro.deterministic("psd", kernel.get_psd(omega))


nuts_kernel = NUTS(numpyro_model, dense_mass=True)
mcmc = MCMC(nuts_kernel, num_warmup=1000, num_samples=1000, num_chains=2)
rng_key = random.PRNGKey(34923)
# %time mcmc.run(rng_key, t, yerr, y=y)
# -

# This runtime was similar to the PyMC3 result from above, and (as we'll see below) the convergence is also similar.
# Any difference in runtime will probably disappear for more computationally expensive models, but this interface is looking pretty great here!
#
# As above, we can plot the posterior expectations for the power spectrum:

# +
psds = np.asarray(mcmc.get_samples()["psd"])

q = np.percentile(psds, [16, 50, 84], axis=0)

plt.loglog(freq, q[1], color="C0")
plt.fill_between(freq, q[0], q[2], color="C0", alpha=0.1)

plt.xlim(freq.min(), freq.max())
plt.xlabel("frequency [1 / day]")
plt.ylabel("power [day ppt$^2$]")
_ = plt.title("posterior psd using numpyro")
# -

# ## Comparison
#
# Finally, let's compare the results of these different inference methods a bit more quantitaively.
# First, let's look at the posterior constraint on the period of the underdamped harmonic oscillator, the effective period of the oscillatory signal.

# +
import arviz as az

emcee_data = az.from_emcee(
    sampler,
    var_names=[
        "mean",
        "log_sigma1",
        "log_rho1",
        "log_tau",
        "log_sigma2",
        "log_rho2",
        "log_jitter",
    ],
)
for k in emcee_data.posterior.data_vars:
    if k.startswith("log_"):
        emcee_data.posterior[k[4:]] = np.exp(emcee_data.posterior[k])

with model:
    pm_data = az.from_pymc3(trace)

numpyro_data = az.from_numpyro(mcmc)

bins = np.linspace(1.5, 2.75, 25)
plt.hist(
    np.asarray((emcee_data.posterior["rho1"].T)).flatten(),
    bins,
    histtype="step",
    density=True,
    label="emcee",
)
plt.hist(
    np.asarray((pm_data.posterior["rho1"].T)).flatten(),
    bins,
    histtype="step",
    density=True,
    label="PyMC3",
)
plt.hist(
    np.asarray((numpyro_data.posterior["rho1"].T)).flatten(),
    bins,
    histtype="step",
    density=True,
    label="numpyro",
)
plt.legend()
plt.yticks([])
plt.xlabel(r"$\rho_1$")
_ = plt.ylabel(r"$p(\rho_1)$")
# -

# That looks pretty consistent.
#
# Next we can look at the [ArviZ](https://arviz-devs.github.io/arviz/) summary for each method to see how the posterior expectations and convergence diagnostics look.

az.summary(
    emcee_data,
    var_names=["mean", "sigma1", "rho1", "tau", "sigma2", "rho2", "jitter"],
)

az.summary(
    pm_data,
    var_names=["mean", "sigma1", "rho1", "tau", "sigma2", "rho2", "jitter"],
)

az.summary(
    numpyro_data,
    var_names=["mean", "sigma1", "rho1", "tau", "sigma2", "rho2", "jitter"],
)

# Overall these results are consistent, but the $\hat{R}$ values are a bit high for the emcee run, so I'd probably run that for longer.
# Either way, for models like these, PyMC3 and numpyro are generally going to be much better inference tools (in terms of runtime per effective sample) than emcee, so those are the recommended interfaces if the rest of your model can be easily implemented in such a framework.

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

# + nbsphinx="hidden"
# %matplotlib inline

# + nbsphinx="hidden"
# %run notebook_setup
# -

# # Multivariate models
#

# +
import numpy as np
import matplotlib.pyplot as plt

import celerite2

N = 200
M = 5
lam = np.linspace(0, 3, M)

np.random.seed(59302)
t = np.append(
    np.sort(np.random.uniform(0, 4, N // 2)),
    np.sort(np.random.uniform(6, 10, N - N // 2)),
)
yerr = np.random.uniform(1e-1, 2e-1, (N, M))

rho_true = 4.5
R_true = 0.5 * np.exp(-0.5 * (lam[:, None] - lam[None, :]) ** 2)
kernel = celerite2.kron.KronTerm(
    celerite2.terms.SHOTerm(sigma=1.0, rho=rho_true, Q=3.0), R=R_true
)
gp = celerite2.GaussianProcess(kernel, t=t, yerr=yerr)
y = gp.sample()

plt.yticks([])
for m in range(M):
    plt.axhline(2 * m, color="k", lw=0.5)
plt.plot(t, y + 2 * np.arange(M), ".")
plt.ylim(-2, 2 * M)
plt.xlim(-1, 11)
plt.xlabel("x")
_ = plt.ylabel("y (with offsets)")

# +
import pymc3 as pm
import pymc3_ext as pmx
import celerite2.theano as cl2

with pm.Model() as model:

    rho = pm.Lognormal("rho", mu=np.log(5.0), sigma=5.0)
    chol = pm.LKJCholeskyCov(
        "chol",
        eta=10.0,
        n=M,
        sd_dist=pm.Exponential.dist(0.01),
        compute_corr=True,
    )[0]
    R = pm.Deterministic("R", pm.math.dot(chol, chol.T))

    kernel = cl2.kron.KronTerm(
        cl2.terms.SHOTerm(sigma=1.0, rho=rho, Q=3.0), R=R
    )
    gp = cl2.GaussianProcess(kernel, t=t, yerr=yerr)
    gp.marginal("obs", observed=y)

    soln = pmx.optimize()

# +
t_pred = np.linspace(-1, 11, 1000)
with model:
    mu, var = pmx.eval_in_model(gp.predict(y, t=t_pred, return_var=True), soln)

plt.yticks([])
for m in range(M):
    plt.axhline(2 * m, color="k", lw=0.5)
    plt.plot(t, y[:, m] + 2 * m, ".", color=f"C{m}")
    plt.fill_between(
        t_pred,
        mu[:, m] - np.sqrt(var[:, m]) + 2 * m,
        mu[:, m] + np.sqrt(var[:, m]) + 2 * m,
        color=f"C{m}",
        alpha=0.5,
    )
    plt.plot(t_pred, mu[:, m] + 2 * m, color=f"C{m}")

plt.ylim(-2, 2 * M)
plt.xlim(-1, 11)
plt.xlabel("x")
_ = plt.ylabel("y (with offsets)")
# -

with model:
    trace = pm.sample(
        tune=2000, draws=2000, target_accept=0.9, init="adapt_full"
    )

plt.hist(trace["rho"], 50, histtype="step", color="k")
plt.axvline(rho_true)
plt.yticks([])
plt.xlabel(r"$\rho$")
plt.ylabel(r"$p(\rho)$")

for m in range(M):
    plt.errorbar(
        np.arange(M),
        np.mean(trace["R"][:, m, :], axis=0) + m,
        yerr=np.std(trace["R"][:, m, :], axis=0),
        color=f"C{m}",
    )
    plt.plot(np.arange(M), R_true[m] + m, ":", color=f"C{m}")
plt.yticks([])
plt.xlabel("band index")
_ = plt.ylabel("covariance (with offsets)")

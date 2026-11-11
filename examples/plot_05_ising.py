"""
Ising models and phase diagrams
===============================

A pairwise maximum-entropy model, or Ising model, describes binary
multichannel data (each channel active, +1, or inactive, -1, at each time
point) with the probability

.. math:: P(s) \\propto \\exp(-\\beta E(s)), \\quad
   E(s) = -\\frac{1}{2} \\sum_{i \\neq j} J_{ij} s_i s_j - \\sum_i h_i s_i,

where each channel has a field :math:`h_i` (its tendency to be active) and
each pair of channels a coupling :math:`J_{ij}`. Once the model is fitted to
data, it can be simulated at other parameter values to locate the data
relative to the phase transitions of the model (Ezaki et al., 2020).

This example tests the steps of this analysis on data sampled from a known
model:

1. fitting the model (:func:`edgeofpy.fit_pmem`) and its temperature
   (:func:`edgeofpy.fit_beta`);
2. the specific heat as a function of temperature
   (:func:`edgeofpy.heat_capacity`);
3. the Sherrington-Kirkpatrick (SK) phase diagram
   (:func:`edgeofpy.sk_diagram`, :func:`edgeofpy.plot_sk`).
"""

import matplotlib.pyplot as plt
import numpy as np

import edgeofpy as eop

# %%
# A known model
# -------------
# We take 20 channels with random couplings drawn as in the SK model, with
# mean 0.5/N and standard deviation 0.6/sqrt(N) (N = 20), and small random
# fields. :func:`edgeofpy.models.ising_samples` draws 20,000 samples from
# the model at beta = 1. With real data, the samples would be binarized
# signals coded as -1/+1 (e.g. with :func:`edgeofpy.binarized_events` and
# ``null_value=-1``).

N = 20
rng = np.random.default_rng(0)
J_true = np.triu(rng.normal(0.5 / N, 0.6 / np.sqrt(N), (N, N)), 1)
J_true = J_true + J_true.T
h_true = rng.normal(0, 0.1, N)

data = eop.models.ising_samples(h_true, J_true, n_samples=20_000,
                                rand_seed=1)
print('data shape (channels x samples):', data.shape)

# %%
# Fitting the model
# -----------------
# :func:`edgeofpy.fit_pmem` estimates h and J by pseudo-likelihood
# maximization.

h_fit, J_fit = eop.fit_pmem(data)

pairs = np.triu_indices(N, 1)
fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
for ax, true, fit, name in ((axes[0], h_true, h_fit, 'h'),
                            (axes[1], J_true[pairs], J_fit[pairs], 'J')):
    lims = [min(true.min(), fit.min()), max(true.max(), fit.max())]
    ax.plot(lims, lims, 'k--', lw=1)
    ax.plot(true, fit, 'o', ms=3, alpha=0.6)
    ax.set(xlabel=f'true {name}', ylabel=f'fitted {name}', aspect='equal')
    r = np.corrcoef(true, fit)[0, 1]
    ax.set_title(f'{name}: r = {r:.3f}')
fig.tight_layout()

# %%
# The fitted parameters match the true ones closely.
#
# The temperature of the model, 1/beta, sets how strongly the fields and
# couplings constrain the activity. :func:`edgeofpy.fit_beta` finds the beta
# at which a given model best matches the data. Here, we draw data from the
# true model at different values of beta and fit beta, keeping h and J at
# their true values.

print(f"{'true beta':>10} {'fitted beta':>12}")
for beta in (0.5, 1.5):
    data_beta = eop.models.ising_samples(h_true, J_true, n_samples=20_000,
                                         beta=beta, rand_seed=2)
    print(f'{beta:10.2f} {eop.fit_beta(data_beta, h_true, J_true):12.2f}')
print(f"{'fitted model, original data':>28}: "
      f'{eop.fit_beta(data, h_fit, J_fit):.2f}')

# %%
# For a model obtained with :func:`edgeofpy.fit_pmem`, the fitted beta is 1,
# since the fit already absorbs the temperature into h and J.
# :func:`edgeofpy.fit_beta` is useful instead to place data relative to a
# fixed model, for example a model fitted to all conditions pooled
# (Ruffini et al., 2023).
#
# Specific heat
# -------------
# The specific heat, the variance of the energy per channel (times beta
# squared), peaks at the phase transition. We simulate the fitted model at
# a range of beta and compute it from the samples with
# :func:`edgeofpy.heat_capacity`, which assumes beta = 1, hence the factor
# beta squared.

betas = np.linspace(0.25, 3, 23)
spec_heat = []
for beta in betas:
    samples = eop.models.ising_samples(h_fit, J_fit, n_samples=5000,
                                       beta=beta, rand_seed=3)
    spec_heat.append(beta ** 2 * eop.heat_capacity(samples, h_fit, J_fit,
                                                   mode='specific'))

off_diag = ~np.eye(N, dtype=bool)
beta_sk = 1 / (J_fit[off_diag].std() * np.sqrt(N))

fig, ax = plt.subplots(figsize=(5.5, 3.8))
ax.plot(betas, spec_heat, 'o-', ms=4)
ax.axvline(1, color='k', lw=1, label='data (beta = 1)')
ax.axvline(beta_sk, color='C1', ls='--', lw=1,
           label='SK transition, infinite N')
ax.set(xlabel='beta (inverse temperature)', ylabel='specific heat')
ax.legend()
fig.tight_layout()
print(f'Specific heat peaks at beta = {betas[np.argmax(spec_heat)]:.2f}; '
      f'SK theory: {beta_sk:.2f}')

# %%
# In the infinite SK model with the same couplings, the transition, from a
# paramagnetic phase (each channel fluctuating about its mean) to a
# spin-glass phase (channels frozen into one of many configurations), lies
# at the dashed line. With only 20 channels, the peak is broad and lies at
# a lower temperature (larger beta). The data, at beta = 1, lie on the
# paramagnetic side of the peak. In a real network, a peak closer to beta =
# 1 would indicate dynamics closer to the transition.
#
# SK phase diagram
# ----------------
# Rather than temperature, :func:`edgeofpy.sk_diagram` varies the mean and
# the standard deviation of the couplings, keeping the fields fixed and
# beta = 1. For each pair of values, it rescales the fitted couplings,
# simulates the model by Metropolis sampling and computes order parameters:
# the magnetization m (the mean activity, large in the ferromagnetic phase,
# where all channels align), the spin-glass order parameter q (the mean
# squared activity of each channel, large in the ferromagnetic and
# spin-glass phases), and the spin-glass susceptibility (which peaks at the
# spin-glass transition). This takes about 55 s here; for larger diagrams,
# use ``multiproc=True``.
#
# For comparison, we draw the phase boundaries of the infinite SK model
# without fields at beta = 1. In this model, the couplings have mean J0/N
# and standard deviation J/sqrt(N), and the phases are:
#
# - paramagnetic for J0 < 1 and J < 1 (Sherrington & Kirkpatrick, 1975),
#   that is, mean(J) < 1/N and std(J) < 1/sqrt(N) (dashed lines);
# - ferromagnetic or spin glass outside of it, depending on whether J0 is
#   larger or smaller than J (Mezard et al., 1987), that is, whether
#   mean(J) is larger or smaller than std(J)/sqrt(N) (dotted line).

m, q, chi_u, chi_sg, C, mu_values, std_values = eop.sk_diagram(
    h_fit, J_fit, mu_range=(-0.02, 0.25), mu_step=0.02,
    std_range=(0.04, 0.8), std_step=0.06, n_iter=2000, rand_init=True)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
std_line = np.linspace(1 / np.sqrt(N), std_values[-1])
for ax, value, title in ((axes[0], np.abs(m), '|m|'),
                         (axes[1], q, 'q'),
                         (axes[2], np.log10(chi_sg),
                          'spin-glass susceptibility (log10)')):
    eop.plot_sk(value, mu_values, std_values, J=J_fit, title=title, ax=ax)
    ax.plot([1 / N, 1 / N], [std_values[0] - 0.03, 1 / np.sqrt(N)], 'w--',
            lw=1)
    ax.plot([mu_values[0] - 0.01, 1 / N], [1 / np.sqrt(N)] * 2, 'w--', lw=1)
    ax.plot(std_line / np.sqrt(N), std_line, 'w:', lw=1.5)
fig.tight_layout()

# %%
# The red star marks the fitted model. The diagram shows the three phases:
# paramagnetic at the bottom left, where m and q are small; ferromagnetic
# at the right, where m and q are both large; and spin glass at the top
# left, where q is large but m is not. The spin-glass susceptibility is
# highest along the edges of the paramagnetic phase, where the transitions
# occur. The fitted model lies in the paramagnetic phase, as the specific
# heat indicated.
#
# The boundary between the ferromagnetic and spin-glass phases follows the
# dotted line, but the paramagnetic phase extends beyond the dashed lines.
# This is partly due to the fields, which oppose alignment, and partly to
# the small size of the network: the channels do not stay frozen in one
# configuration indefinitely, and q, a time average, becomes large only
# where they stay frozen for most of the simulation, so that the apparent
# boundaries depend on the number of channels and on ``n_iter``. The
# diagram is also noisy near the boundaries, where the dynamics are slow;
# larger ``n_iter`` reduces the noise.
#
# References
# ----------
# - Ezaki et al. (2020) Commun Biol 3, 52.
# - Mezard, Parisi & Virasoro (1987) Spin glass theory and beyond. World
#   Scientific.
# - Ruffini et al. (2023) PLOS Comp Biol 19(2), e1010811.
# - Sherrington & Kirkpatrick (1975) Phys Rev Lett 35(26), 1792.

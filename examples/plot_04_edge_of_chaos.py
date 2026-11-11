"""
Edge of chaos
=============

edgeofpy has two measures of the distance to the edge of chaos:

- the modified **0-1 test for chaos** (:func:`edgeofpy.z1_chaos_test`,
  :func:`edgeofpy.chaos_pipeline`), which classifies a single time series as
  regular (K close to 0) or chaotic (K close to 1);
- **lambda max** (:func:`edgeofpy.lambda_max`), which estimates, from the
  covariances of many recorded units, how close a network is to the edge of
  instability (lambda max = 1).

This example tests both on systems where the answer is known: the logistic
map, whose dynamics change from periodic to chaotic as its parameter r
increases, and linear networks with a chosen largest eigenvalue.
"""

import matplotlib.pyplot as plt
import numpy as np

import edgeofpy as eop

# %%
# The 0-1 test along the route to chaos
# -------------------------------------
# The logistic map x <- r x (1 - x) goes through a cascade of period
# doublings up to r = 3.57, beyond which it is mostly chaotic, with periodic
# windows (the widest near r = 3.83). For each r, we compute the K statistic
# of the 0-1 test and, for comparison, the exact Lyapunov exponent of the map,
# which is positive for chaotic dynamics.

rs = np.linspace(3.4, 4.0, 61)
K = np.array([eop.z1_chaos_test(eop.models.logistic_map(2000, r=r))
              for r in rs])
lyapunov = np.array([
    np.mean(np.log(np.abs(r * (1 - 2 * eop.models.logistic_map(20_000, r=r)))))
    for r in rs])

fig, axes = plt.subplots(3, 1, figsize=(7, 7.5), sharex=True)
r_dense = np.repeat(np.linspace(3.4, 4.0, 1200), 100)
x_dense = np.concatenate([eop.models.logistic_map(100, r=r, transient=500)
                          for r in r_dense[::100]])
axes[0].plot(r_dense, x_dense, ',k', alpha=0.3)
axes[0].set(ylabel='x', title='Bifurcation diagram')
axes[1].plot(rs, K, 'o-', ms=3)
axes[1].set(ylabel='K', title='0-1 test')
axes[2].plot(rs, lyapunov, 'o-', ms=3, color='C1')
axes[2].axhline(0, color='k', lw=1)
axes[2].set(xlabel='r', ylabel='exponent', title='Lyapunov exponent')
fig.tight_layout()

agree = np.mean((K > 0.5) == (lyapunov > 0))
print(f'The 0-1 test (K > 0.5) and the Lyapunov exponent (> 0) agree for '
      f'{agree:.0%} of the values of r')
print('Disagreements at r =', np.round(rs[(K > 0.5) != (lyapunov > 0)], 2))

# %%
# The test agrees with the Lyapunov exponent almost everywhere, including in
# the periodic windows. Notice that the test metric bottoms out for 
# Lyapunov exponents below zero (stable/regular dynamics). The 0-1 chaos test 
# thus cannot differentiate the edge of chaos from regular dynamics.

# %%
# Noise and the chaos pipeline
# ----------------------------
# Measurement noise makes regular signals look irregular, which can fool the
# 0-1 test. :func:`edgeofpy.chaos_pipeline` follows the procedure of Toker et
# al. (2022): it can denoise the signal (Schreiber, 1993) and downsamples it
# to its local extrema before applying the test.

rng = np.random.default_rng(0)
print(f"{'signal':>10} {'noise':>6} {'K, raw':>8} {'K, pipeline':>12}")
for label, r in (('periodic', 3.5), ('chaotic', 3.9)):
    x = eop.models.logistic_map(2000, r=r)
    for noise in (0, 0.02, 0.05):
        noisy = x + noise * rng.standard_normal(x.size)
        print(f'{label:>10} {noise:6.2f} {eop.z1_chaos_test(noisy):8.2f} '
              f'{eop.chaos_pipeline(noisy, denoise=True):12.2f}')

# %%
# With noise of standard deviation 0.05 (the map takes values between 0 and
# 1), the raw test is pushed towards calling the periodic signal chaotic
# (K = 0.66), whereas the denoised pipeline still classifies it correctly,
# without losing the chaotic signal.

# %%
# lambda max on networks with a known largest eigenvalue
# ------------------------------------------------------
# :func:`edgeofpy.models.linear_network` simulates x(t+1) = W x(t) + noise,
# with random connectivity W scaled so that its largest eigenvalue, lambda
# max, takes a chosen value. :func:`edgeofpy.lambda_max` estimates it from
# the dispersion of the covariances between units (Dahmen et al., 2019),
# here from 100 trials of 1000 time steps. The number of units ``N`` in the
# network is an input of the method; with M/EEG data it must be estimated.
# Note that a lambda max of 1 corresponds to the edge of chaos, with chaos
# above 1 and stable/regular dynamics below 1.

truths = [0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98]
est = {'corrected': [], 'without bias correction': [],
       'corrected, 20 of 100 units recorded': []}
for lam in truths:
    x, W = eop.models.linear_network(n_units=100, spectral_radius=lam,
                                     n_steps=100_000, rand_seed=0)
    est['corrected'].append(eop.lambda_max(x, s_freq=100, N=100)[0])
    est['without bias correction'].append(
        eop.lambda_max(x, s_freq=100, N=100, bias_correction=False)[0])
    est['corrected, 20 of 100 units recorded'].append(
        eop.lambda_max(x[:20], s_freq=100, N=100)[0])

fig, ax = plt.subplots(figsize=(5, 4.5))
ax.plot([0, 1], [0, 1], 'k--', lw=1, label='identity')
for (label, values), marker in zip(est.items(), 'osd'):
    ax.plot(truths, values, marker + '-', label=label)
ax.set(xlabel='true lambda max', ylabel='estimated lambda max',
       xlim=(0, 1), ylim=(0, 1))
ax.legend(loc='lower right')
fig.tight_layout()

for i, lam in enumerate(truths):
    print(f"true {lam:.2f}: corrected {est['corrected'][i]:.2f}, without "
          f"correction {est['without bias correction'][i]:.2f}")

# %%
# Without correction, the sampling noise of the covariance estimates inflates
# their dispersion, and low values are strongly overestimated. With the
# correction (the default), the estimates follow the true value from about
# 0.3 to 0.7 (very low values, like 0.1, cannot be distinguished from 0).
# Close to 1, however, they are too low: in a finite network, the largest
# eigenvalue lies somewhat beyond the bulk of the eigenvalue spectrum, which
# is what the theory behind the method describes. When only a fifth of the
# units are recorded, the estimates are lower still, by about 0.1.
#
# In all cases, the estimates increase with the true value, so lambda max is
# well suited to comparing conditions or subjects recorded in the same way,
# less so to reading off an absolute distance to the edge.

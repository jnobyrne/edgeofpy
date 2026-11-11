"""
Avalanche analysis, end to end
==============================

This example goes through a complete neuronal avalanche analysis:
detecting avalanches, fitting the power laws of their sizes and durations,
estimating the deviation from criticality coefficient (DCC) with confidence
intervals, and fitting the shape collapse.

To know what the analysis *should* find, we use a simulated system whose
distance to criticality is known: a branching network tuned to its critical
point (:func:`edgeofpy.models.branching_network`). With M/EEG data, the
first step would instead be :func:`edgeofpy.detect_avalanches` applied to
the cleaned signal (channels x samples), which also detects the events.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np

import edgeofpy as eop

# powerlaw warns when the fitted truncated power law reaches the edge of its
# parameter range; this concerns only the comparison with that distribution
warnings.filterwarnings('ignore', message='Fitted parameters are very close')

# %%
# Simulate a critical branching network
# -------------------------------------
# 1000 units, each activating on average ``sigma = 1`` other unit at the next
# time step: the critical point. Units are pooled into 32 "channels", like
# electrodes that each record several neurons. We treat one time step as one
# millisecond, so the sampling frequency is 1000 Hz.

s_freq = 1000
activity = eop.models.branching_network(n_units=1000, sigma=1.0,
                                        n_steps=300_000, n_channels=32,
                                        rand_seed=1)
print(activity.shape)  # channels x time steps

# %%
# Detect avalanches
# -----------------
# An avalanche is a sequence of time bins with activity, bounded by silent
# bins. ``max_iei`` is the maximum inter-event interval within an avalanche;
# here a single time step (1 ms). As ``activity`` holds integer event counts,
# no thresholding is needed.

max_iei = 0.001
avalanches, _, events, _ = eop.detect_avalanches(activity, s_freq,
                                                 max_iei=max_iei)
sizes = np.array([a['size'] for a in avalanches])
durations = np.array([a['dur_bin'] for a in avalanches])
print(f'{len(avalanches)} avalanches; largest: {sizes.max()} events, '
      f'longest: {durations.max()} bins')

# %%
# Activity alternates between silence and avalanches of all sizes:

t = np.arange(2000) / s_freq
fig, ax = plt.subplots(figsize=(8, 2.5))
ax.plot(t, events[:2000], lw=0.8)
ax.set(xlabel='time (s)', ylabel='active units',
       title='Total activity, first 2 s')
fig.tight_layout()

# %%
# Size and duration distributions
# -------------------------------
# :func:`edgeofpy.fit_powerlaw` fits a power law above a lower bound
# ``xmin``, chosen automatically by default, and compares it with other
# distributions. Here we plot the empirical distributions (dots) and the
# fitted power laws (dashed) over the fitted ranges, using the plotting
# methods of the underlying ``powerlaw.Fit`` objects.

fit_s = eop.fit_powerlaw(sizes)
fit_d = eop.fit_powerlaw(durations)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
for ax, fit, label, sym in zip(axes, (fit_s, fit_d), ('size', 'duration'),
                               (r'\tau', r'\alpha')):
    fit['fit'].plot_pdf(ax=ax, marker='o', ls='none', ms=4)
    fit['fit'].power_law.plot_pdf(ax=ax, ls='--', color='k')
    ax.set(xlabel=f'avalanche {label}', ylabel=f'P({label})',
           title=f"${sym}$ = {fit['power_law_exp']:.2f} "
                 f"(xmin = {fit['xmin']:g})")
fig.tight_layout()

# %%
# For a critical branching network, the exponents are close to the mean-field
# values tau = 3/2 and alpha = 2. Likelihood-ratio tests compare the power
# law with alternatives; a positive ratio R favours the power law (p is the
# significance of the comparison):

for name, fit in (('sizes', fit_s), ('durations', fit_d)):
    R, p = fit['PvE']
    print(f'{name}: power law vs exponential: R = {R:.1f}, p = {p:.2g}')

# %%
# Mean size as a function of duration
# -----------------------------------
# At criticality, the mean size of avalanches of duration T grows as
# T to the power of the "third" exponent. It is fitted over the same range
# of durations as the duration power law:

dur_range = (fit_d['xmin'], fit_d['xmax'])
third = eop.fit_third_exponent(sizes, durations, dur_range=dur_range)

T = np.unique(durations)
mean_size = np.array([sizes[durations == d].mean() for d in T])
in_range = T >= dur_range[0]
offset = np.mean(np.log(mean_size[in_range]) - third * np.log(T[in_range]))
fig, ax = plt.subplots(figsize=(4.5, 3.8))
ax.loglog(T, mean_size, 'o', ms=3, label='mean size')
ax.loglog(T[in_range], np.exp(offset) * T[in_range] ** third, 'k--',
          label=f'slope {third:.2f}')
ax.set(xlabel='duration T (bins)', ylabel='mean size <S>(T)')
ax.legend()
fig.tight_layout()

# %%
# Deviation from criticality coefficient
# --------------------------------------
# At criticality, the three exponents satisfy the crackling-noise relation
# (alpha - 1) / (tau - 1) = third. The DCC is the absolute difference
# between the two sides. :func:`edgeofpy.estimate_dcc` does all of the
# above in one call, with the same fitting ranges, and adds diagnostics and
# bootstrap confidence intervals:

res = eop.estimate_dcc(sizes, durations, n_boot=200, xmin_sensitivity=True,
                       rand_seed=0)
print(f"tau = {res['tau']:.3f}, alpha = {res['alpha']:.3f}, "
      f"third = {res['third']:.3f}")
print(f"DCC = {res['dcc']:.3f}, 95% CI {np.round(res['dcc_ci'], 3)}")
print(f"signed deviation = {res['dcc_signed']:.3f}, "
      f"95% CI {np.round(res['dcc_signed_ci'], 3)}")
print(f"DCC with xmin halved or doubled: {np.round(res['dcc_xmin_range'], 3)}")

# %%
# Because the DCC is an absolute value, noise pushes it above 0 even for a
# critical system, so its confidence interval excludes 0. The signed deviation
# has no such bias: its confidence interval includes 0, so the exponents are
# consistent with criticality. The range of DCC values obtained with the
# lower bounds ``xmin`` halved or doubled is a reminder of how much the DCC
# depends on the fitting ranges.

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.hist(res['boot']['dcc_signed'], bins=30, color='0.7')
ax.axvline(0, color='k', ls='--', label='criticality')
ax.axvline(res['dcc_signed'], color='C3', label='estimate')
ax.set(xlabel='signed deviation (bootstrap)', ylabel='count')
ax.legend()
fig.tight_layout()

# %%
# Shape collapse
# --------------
# At criticality, the mean temporal profiles of avalanches of different
# durations T have the same shape: rescaling time by T and the profiles by
# T to the power gamma makes them collapse onto one curve, with
# gamma = third - 1. :func:`edgeofpy.fit_shape_collapse` finds gamma and
# fits the third exponent over the same durations.

collapse = eop.fit_shape_collapse(avalanches)
print(f"gamma = {collapse['gamma']:.3f}, third - 1 = "
      f"{collapse['third'] - 1:.3f}, "
      f"DCC (collapse) = {collapse['dcc_collapse']:.3f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
eop.plot_shape_collapse(collapse, collapsed=False, bin_dur=max_iei, ax=ax1,
                        colorbar=False)
eop.plot_shape_collapse(collapse, ax=ax2)
fig.tight_layout()

# %%
# Short avalanches deviate from the common shape (corrections to scaling), so
# gamma depends somewhat on the shortest duration included, ``min_dur``.
# A trend like the one below is a reason to report gamma with its ``min_dur``,
# and to compare conditions with the same settings:

for min_dur, gamma in collapse['gamma_by_min_dur'].items():
    print(f'min_dur = {min_dur:2d}: gamma = {gamma:.3f}')

# %%
# Branching ratio
# ---------------
# Finally, the branching ratio, the mean number of events in a time bin per
# event in the previous bin, is close to 1 at criticality:

print(f'branching ratio: {eop.branching_ratio(events):.3f}')

# %%
# The next examples show how these measures change away from the critical
# point, and how to apply them to M/EEG recordings.

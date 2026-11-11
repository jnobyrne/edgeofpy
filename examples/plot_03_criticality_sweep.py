"""
Tuning through criticality
==========================

How do the avalanche measures change as a system moves from subcritical to
critical to supercritical dynamics? This example sweeps the branching
parameter sigma of a simulated branching network
(:func:`edgeofpy.models.branching_network`) through its critical point and
computes, for each value:

- the avalanche exponents and the deviation from criticality coefficient
  (DCC);
- the branching ratio;
- the susceptibility and the Fano factor, which measure the size of
  fluctuations in activity.

Each active unit activates on average sigma other units at the next time
step: below 1, activity dies out quickly (subcritical); above 1, it grows
until it is limited by the size of the network (supercritical).
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np

import edgeofpy as eop

# powerlaw warns when it cannot fit some of the alternative distributions
# (truncated power law, lognormal) well; this concerns only the comparisons
# with these distributions, which are not used here
warnings.filterwarnings('ignore', module='powerlaw')

# %%
# Simulate the network for a range of sigma
# -----------------------------------------
# For each sigma, 300 s of activity of 1000 units pooled into 32 channels,
# sampled at 1000 Hz (one time step per millisecond), as in the first
# example.

n_units, s_freq = 1000, 1000
sigmas = [0.8, 0.85, 0.9, 0.95, 1.0, 1.025, 1.05, 1.075, 1.1, 1.15, 1.2]
sims = {}
for sigma in sigmas:
    activity = eop.models.branching_network(n_units=n_units, sigma=sigma,
                                            n_steps=300_000, n_channels=32,
                                            rand_seed=1)
    avalanches, _, events, _ = eop.detect_avalanches(activity, s_freq,
                                                     max_iei=0.001)
    sims[sigma] = {'events': events,
                   'sizes': np.array([a['size'] for a in avalanches]),
                   'durations': np.array([a['dur_bin'] for a in avalanches])}
    print(f"sigma = {sigma:.3f}: {len(avalanches):5d} avalanches, "
          f"silent {np.mean(events == 0):4.0%} of the time, "
          f"mean activity {events.mean():6.2f} units")

# %%
# Up to sigma = 1, activity occurs in avalanches separated by long silences.
# Above it, activity grows, and from about sigma = 1.075 it becomes largely
# self-sustained: silences, and with them separate avalanches, disappear, so
# avalanche statistics can no longer be computed.
#
# Avalanche size distributions
# ----------------------------
# Subcritical avalanches are small, and their size distribution falls off
# quickly; at the critical point it follows a power law over several
# decades; just above, very large avalanches become more frequent.

fig, ax = plt.subplots(figsize=(5.5, 4))
for sigma in (0.9, 1.0, 1.05):
    eop.plot_pdf(sims[sigma]['sizes'], binning='log', n_bins=30, ax=ax,
                 xlabel='avalanche size')
    ax.lines[-1].set(marker='o', ms=3, label=f'sigma = {sigma}')
ax.legend()
fig.tight_layout()

# %%
# Measures across the transition
# ------------------------------
# The DCC is estimated with fixed fitting ranges, as recommended in the
# previous example, and only when there are enough avalanches. The
# susceptibility is computed from the total number of active units over
# time, and the Fano factor from the number of events in 100 ms windows.

rows = []
for sigma in sigmas:
    sim = sims[sigma]
    row = {'sigma': sigma,
           'branching ratio': eop.branching_ratio(sim['events']),
           'susceptibility': eop.susceptibility(sim['events'],
                                                n_units=n_units)[0],
           'fano': eop.fano_factor(eop.time_bin_events(sim['events'], 100))}
    if len(sim['sizes']) >= 500:
        res = eop.estimate_dcc(sim['sizes'], sim['durations'],
                               size_range=(2, None), dur_range=(3, None),
                               n_boot=100, rand_seed=0)
        row.update({k: res[k] for k in ('tau', 'alpha', 'third', 'dcc',
                                        'dcc_signed', 'dcc_signed_ci')})
    rows.append(row)


def column(key):
    return np.array([r.get(key, np.nan) for r in rows], dtype=float)


fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)

ax = axes[0, 0]
for key, label in (('tau', 'tau (sizes)'), ('alpha', 'alpha (durations)'),
                   ('third', 'third')):
    ax.plot(sigmas, column(key), 'o-', label=label)
ax.axhline(1.5, color='C0', ls=':', lw=1)
ax.axhline(2.0, color='C1', ls=':', lw=1)
ax.set(ylabel='exponent', title='Avalanche exponents (dotted: mean field)')
ax.legend()

ax = axes[0, 1]
signed = column('dcc_signed')
has_ci = [r for r in rows if 'dcc_signed_ci' in r]
low, high = np.array([r['dcc_signed_ci'] for r in has_ci]).T
ax.fill_between([r['sigma'] for r in has_ci], low, high, alpha=0.3,
                label='95% CI')
ax.plot(sigmas, signed, 'o-', label='signed deviation')
ax.axhline(0, color='k', ls='--', lw=1)
ax.set(ylabel='(alpha - 1) / (tau - 1) - third',
       title='Deviation from criticality (DCC = |.|)')
ax.legend()

ax = axes[1, 0]
ax.plot(sigmas, column('branching ratio'), 'o-', label='estimated')
ax.plot(sigmas, sigmas, 'k--', lw=1, label='sigma')
ax.set(xlabel='sigma', ylabel='branching ratio', title='Branching ratio')
ax.legend()

ax = axes[1, 1]
ax.semilogy(sigmas, column('susceptibility'), 'o-', label='susceptibility')
ax.semilogy(sigmas, column('fano'), 's-', label='Fano factor (100 ms)')
ax.set(xlabel='sigma', title='Fluctuations')
ax.legend()

for ax in axes.flat:
    ax.axvline(1, color='0.6', lw=1, zorder=0)
fig.tight_layout()

# %%
# Reading the figure:
#
# - **Exponents and DCC** (top). The size and duration exponents decrease
#   steadily with sigma and reach the mean-field values (tau = 3/2,
#   alpha = 2) near sigma = 1. The signed deviation from the crackling-noise
#   relation is clearly positive in the subcritical range, consistent with 0
#   near sigma = 1, and negative just above. Note that it comes back close to
#   0 at sigma = 1.075, where the remaining avalanches are a mix of isolated
#   avalanches and fragments of sustained activity: a DCC near 0 is
#   consistent with criticality, but does not prove it on its own.
# - **Branching ratio** (bottom left). The estimate follows sigma up to 1 but
#   stays near 1 above it: once activity is self-sustained, each active unit
#   is, on average, replaced by one other. A branching ratio of 1 therefore
#   does not distinguish critical from supercritical dynamics.
# - **Fluctuations** (bottom right). The susceptibility and the Fano factor
#   grow by orders of magnitude as sigma approaches 1 from below, and peak
#   slightly above it (here at 1.05 and 1.075), where activity switches from
#   avalanches to a sustained state. In a finite network with refractory units, this switch
#   occurs a little above sigma = 1, because units that are already active
#   or refractory cannot be activated again.
#
# Together, the measures locate the transition consistently, but each has
# its blind spots, which is why it is useful to compute several of them.

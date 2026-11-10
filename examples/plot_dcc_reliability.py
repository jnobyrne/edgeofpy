"""
Estimating the DCC reliably
===========================

The deviation from criticality coefficient (DCC) combines three fitted
exponents, so it inherits all their uncertainties, and it is sensitive to
the choices made when fitting them. This example uses several simulated
recordings of the *same* critical system, for which the DCC should be close
to 0, to show:

1. how the choice of the lower bound ``xmin`` of the power-law fits affects
   the DCC;
2. how the DCC is biased upward when there are few avalanches;
3. how to compare conditions with different numbers of avalanches.

It ends with practical recommendations.
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np

import edgeofpy as eop

# powerlaw warns when the fitted truncated power law reaches the edge of its
# parameter range; this concerns only the comparison with that distribution
warnings.filterwarnings('ignore', message='Fitted parameters are very close')


def simulate(seed, n_steps=300_000):
    """Avalanche sizes and durations of a critical branching network."""
    activity = eop.models.branching_network(n_units=1000, sigma=1.0,
                                            n_steps=n_steps, n_channels=32,
                                            rand_seed=seed)
    avalanches = eop.detect_avalanches(activity, s_freq=1000,
                                       max_iei=0.001)[0]
    return (np.array([a['size'] for a in avalanches]),
            np.array([a['dur_bin'] for a in avalanches]))


# %%
# Six recordings of the same critical system
# ------------------------------------------
# Think of these as six subjects recorded in the same brain state: the
# underlying dynamics are identical (a critical branching network, as in the
# previous example) and only the random fluctuations differ.

recordings = [simulate(seed) for seed in range(6)]
print([len(sizes) for sizes, _ in recordings], 'avalanches')

# %%
# 1. The choice of xmin
# ---------------------
# The power laws are fitted above a lower bound ``xmin`` (and below an
# optional upper bound ``xmax``). We compare three ways of choosing it:
#
# - ``'min'``: the smallest value in the data, i.e. fitting all avalanches;
# - automatic: the value that makes the fit closest to the data (Clauset et
#   al., 2009), the default of :func:`edgeofpy.fit_powerlaw` and
#   :func:`edgeofpy.estimate_dcc`;
# - fixed: the same values for all recordings, here 2 for sizes and 3 for
#   durations.

strategies = {
    "'min'": dict(size_range=(1, None), dur_range=(1, None)),
    'automatic': dict(),
    'fixed (2, 3)': dict(size_range=(2, None), dur_range=(3, None)),
}
dcc = {name: [] for name in strategies}
auto_xmin = []
for sizes, durations in recordings:
    for name, ranges in strategies.items():
        res = eop.estimate_dcc(sizes, durations, **ranges)
        dcc[name].append(res['dcc'])
        if name == 'automatic':
            auto_xmin.append((res['size_range'][0], res['dur_range'][0]))

for name, values in dcc.items():
    print(f'{name:>13}: DCC = {np.round(values, 3)}')
print('automatic xmin (sizes, durations):',
      [(int(s), int(d)) for s, d in auto_xmin])

fig, ax = plt.subplots(figsize=(6, 3.8))
for i, (name, values) in enumerate(dcc.items()):
    ax.plot(np.full(6, i) + np.linspace(-0.15, 0.15, 6), values, 'o',
            label=name)
ax.set_xticks(range(3), list(dcc))
ax.set(xlabel='choice of xmin', ylabel='DCC',
       title='Six recordings of the same critical system')
fig.tight_layout()

# %%
# Fitting all avalanches (``'min'``) gives consistently wrong exponents,
# because the smallest avalanches do not follow the power law, and DCC values
# fall far from 0. The automatic choice works well on some recordings but not 
# on others: for the duration distribution it sometimes picks a large ``xmin``
# (above 20 in one case), leaving few avalanches in the fit and an extreme
# exponent. Fixed ranges give consistent, small values. The automatic choice
# is a good default for a single dataset, but when comparing recordings,
# fixed ranges remove one source of spurious differences. (They must still be
# chosen with care: too small a value brings back the bias of ``'min'``.)
#
# ``xmin_sensitivity=True`` shows how much the DCC of a recording depends on
# these choices, by recomputing it with each ``xmin`` halved and doubled.

# %%
# 2. Few avalanches bias the DCC upward
# -------------------------------------
# The DCC is an absolute value, so random errors in the exponents can only
# increase it. With fewer avalanches, the errors are larger and so is the
# DCC, even though the system is just as critical. To see this, we estimate
# the DCC from random subsets of one recording (``n_avalanches``), with fixed
# fitting ranges and bootstrap confidence intervals.

sizes, durations = recordings[1]
counts = [250, 500, 1000, 2000]
results = [eop.estimate_dcc(sizes, durations, size_range=(2, None),
                            dur_range=(3, None), n_avalanches=n, n_boot=200,
                            rand_seed=0)
           for n in counts]

fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharex=True)
for ax, key, label in zip(axes, ('dcc', 'dcc_signed'),
                          ('DCC', 'signed deviation')):
    est = [r[key] for r in results]
    low, high = np.array([r[key + '_ci'] for r in results]).T
    ax.fill_between(counts, low, high, alpha=0.3, label='95% CI')
    ax.plot(counts, est, 'o-', label='estimate')
    ax.axhline(0, color='k', ls='--', lw=1)
    ax.set(xscale='log', xlabel='number of avalanches', ylabel=label)
    ax.set_xticks(counts, [str(n) for n in counts])
    ax.minorticks_off()
axes[0].legend()
fig.tight_layout()

for n, r in zip(counts, results):
    print(f"{n:5d} avalanches: DCC = {r['dcc']:.3f}, signed deviation = "
          f"{r['dcc_signed']:.3f}, 95% CI {np.round(r['dcc_signed_ci'], 3)}")

# %%
# The DCC grows as the number of avalanches decreases, and its confidence
# interval never includes 0, whatever the number of avalanches. The
# confidence interval of the signed deviation, by contrast, includes 0 at
# every count, correctly indicating that the exponents are consistent with
# criticality. (With very few avalanches the exponent estimates themselves
# become slightly biased, so the signed deviation also drifts away from 0,
# but much less than its uncertainty.)

# %%
# 3. Comparing conditions
# -----------------------
# Suppose two conditions are recorded for different durations: condition A
# yields many avalanches, condition B (a shorter recording of the *same*
# system) about five times fewer. Compared directly, their DCCs differ, and
# one might conclude that A is closer to criticality than B:

sizes_a, durations_a = recordings[1]
sizes_b, durations_b = simulate(seed=10, n_steps=60_000)
ranges = dict(size_range=(2, None), dur_range=(3, None))

naive_a = eop.estimate_dcc(sizes_a, durations_a, **ranges)
naive_b = eop.estimate_dcc(sizes_b, durations_b, **ranges)
print(f"A: {naive_a['n_avalanches']} avalanches, DCC = {naive_a['dcc']:.3f}")
print(f"B: {naive_b['n_avalanches']} avalanches, DCC = {naive_b['dcc']:.3f}")

# %%
# But the DCC of B is estimated from far fewer avalanches, so it is biased
# upward and more uncertain. To compare like with like, estimate both DCCs
# from the same number of avalanches, that of the smaller condition. With
# ``n_avalanches``, the estimate is the median over many random subsets, so
# it does not depend on one arbitrary choice of avalanches:

n = min(len(sizes_a), len(sizes_b))
matched_a = eop.estimate_dcc(sizes_a, durations_a, n_avalanches=n,
                             n_boot=200, rand_seed=0, **ranges)
matched_b = eop.estimate_dcc(sizes_b, durations_b, n_avalanches=n,
                             n_boot=200, rand_seed=0, **ranges)
for name, r in (('A', matched_a), ('B', matched_b)):
    print(f"{name}, {n} avalanches: DCC = {r['dcc']:.3f}, 95% CI "
          f"{np.round(r['dcc_ci'], 3)}; signed deviation 95% CI "
          f"{np.round(r['dcc_signed_ci'], 3)}")

# %%
# With matched counts, the apparent advantage of A disappears (its DCC is
# now no smaller than that of B), and the confidence intervals overlap
# widely: there is no evidence of a difference between the conditions, as
# expected for the same system. Matching the counts and fixing the fitting
# ranges puts both conditions on an equal footing, and the confidence
# intervals show whether a remaining difference exceeds the uncertainty of
# the estimates.
#
# Recommendations
# ---------------
# - Never fit all avalanches (``xmin='min'``): the smallest avalanches bias
#   the exponents.
# - Use the same fitting ranges for all recordings that are compared, e.g.
#   values chosen automatically on pooled data, or fixed in advance; check
#   with ``xmin_sensitivity`` how much the DCC depends on them.
# - Fit the third exponent over the same range of durations as the duration
#   power law (:func:`edgeofpy.estimate_dcc` does this).
# - Check that the power law is plausible: enough avalanches and at least a
#   decade within the fitted ranges (:func:`edgeofpy.estimate_dcc` warns
#   otherwise), and the likelihood ratios against alternative distributions
#   in ``size_fit`` and ``dur_fit``.
# - Report confidence intervals, and use the signed deviation to judge
#   whether the exponents are consistent with criticality.
# - Match the number of avalanches when comparing conditions
#   (``n_avalanches``).
# - Report the fitting ranges and the version of the powerlaw package
#   (``powerlaw_version``), as both affect the results.

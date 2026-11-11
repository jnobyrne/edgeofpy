"""
Edge of synchrony
=================

Networks of coupled oscillators go from incoherent to synchronized
activity as the coupling between them increases. edgeofpy has several
measures of where a system lies relative to this transition:

- the pair correlation function (:func:`edgeofpy.pcf`), which peaks at the
  transition, and the Kuramoto order parameter, which measures the degree
  of synchrony;
- the phase lag index (:func:`edgeofpy.pli`) and the phase lag entropy
  (:func:`edgeofpy.ple`);
- the global lability of synchronization
  (:func:`edgeofpy.global_lability`) and the durations of phase-locking
  episodes (:func:`edgeofpy.phase_lock_interval`);
- phase-locking avalanches (:func:`edgeofpy.detect_phase_avls`).

This example computes them on the Kuramoto model
(:func:`edgeofpy.models.kuramoto`), in which oscillators with natural
frequencies around 10 Hz (standard deviation 1 Hz) are coupled with
strength K. In an infinite network, synchronization emerges at the
critical coupling K_c = sqrt(8/pi) x 2 pi x 1 Hz, about 10 rad/s.
"""

import matplotlib.pyplot as plt
import numpy as np

import edgeofpy as eop

S_FREQ = 250  # Hz
K_C = np.sqrt(8 / np.pi) * 2 * np.pi * 1.0

# %%
# Synchrony measures across the transition
# ----------------------------------------
# For each coupling, we simulate 20 s of activity of 64 oscillators, in 8
# runs with different natural frequencies, and average the measures over
# the runs. The signals returned by the model, cos(theta), can be used like
# band-pass filtered M/EEG signals. To show the effect of network size, we
# also compute the PCF with 2048 oscillators (4 runs).
#
# The phase-locking measures are based on the pairwise phase relationships
# of :func:`edgeofpy.complex_phase_relationship`, an array of N x N x T
# complex values; we therefore compute them on 16 of the oscillators.

rel_coupling = np.round(np.arange(0.5, 1.51, 0.1), 2)  # K / K_c
n_runs = 8
names = ['order parameter', 'PCF', 'PLI', 'PLE', 'GLS']
results = {name: np.zeros((len(rel_coupling), n_runs)) for name in names}
results['PCF, 2048 oscillators'] = np.zeros((len(rel_coupling), 4))

for i, k in enumerate(rel_coupling):
    for run in range(n_runs):
        x, _ = eop.models.kuramoto(n_osc=64, coupling=k * K_C, duration=20,
                                   s_freq=S_FREQ, rand_seed=run)
        pcf, order, _, _ = eop.pcf(x)
        results['order parameter'][i, run] = order
        results['PCF'][i, run] = pcf
        results['PLI'][i, run] = eop.pli(x[:16])
        results['PLE'][i, run] = eop.ple(x[:16], m=3, tau=1)
        c_ij = eop.complex_phase_relationship(x[:16, :10 * S_FREQ])
        locked = eop.phase_locking(c_ij)
        results['GLS'][i, run] = eop.global_lability(
            locked, S_FREQ, delta_t_sec=0.1).mean()
    for run in range(4):
        x_large, _ = eop.models.kuramoto(n_osc=2048, coupling=k * K_C,
                                         duration=20, s_freq=S_FREQ,
                                         rand_seed=run)
        results['PCF, 2048 oscillators'][i, run] = eop.pcf(x_large)[0]

panels = [('order parameter', ['order parameter']),
          ('PCF', ['PCF', 'PCF, 2048 oscillators']),
          ('PLI', ['PLI']), ('PLE', ['PLE']), ('mean GLS', ['GLS'])]
fig, axes = plt.subplots(len(panels), 1, figsize=(6, 9), sharex=True)
for ax, (ylabel, keys) in zip(axes, panels):
    for key in keys:
        mean = results[key].mean(axis=1)
        sem = results[key].std(axis=1) / np.sqrt(results[key].shape[1])
        label = '2048 oscillators' if '2048' in key else '64 oscillators'
        ax.errorbar(rel_coupling, mean, sem, marker='o', ms=3, capsize=2,
                    label=label)
    ax.axvline(1, color='k', lw=1)
    ax.set_ylabel(ylabel)
axes[1].legend()
axes[-1].set_xlabel('coupling K / K_c')
fig.tight_layout()

for key in ('PCF', 'PCF, 2048 oscillators', 'GLS'):
    peak = rel_coupling[np.argmax(results[key].mean(axis=1))]
    print(f'{key} peaks at K / K_c = {peak:.1f}')

# %%
# The order parameter, the time-averaged degree of synchrony, rises from
# close to 0 (incoherent oscillators) to close to 1 (synchronized
# oscillators) around K_c. The PCF measures the fluctuations of the order
# parameter. With 2048 oscillators, it grows as the coupling approaches K_c
# and collapses just above it: it peaks at the transition, as expected from
# theory. With 64 oscillators, however, it shows only a broad plateau below
# K_c: in small networks, the fluctuations due to the finite number of
# oscillators mask the growth of the critical fluctuations. With the few
# tens of channels of M/EEG data, the PCF is therefore more reliable for
# telling the synchronized side of the transition (low PCF) from the rest
# than for locating the transition precisely.
#
# The PLI and the PLE do not peak at the transition. The PLI measures how
# consistently one oscillator leads or lags another; it rises with
# synchrony, as the oscillators lock to each other with fixed phase lags.
# The PLE measures how irregular the alternation between leading and
# lagging is; it falls with synchrony. Both therefore measure the degree of
# synchrony, like the order parameter, rather than the distance to the
# transition.
#
# The global lability of synchronization measures how much the number of
# phase-locked pairs changes over time (here over 0.1 s). It peaks just
# above K_c, where groups of oscillators repeatedly lock and unlock.
#
#
# References
# ----------
# - Botcharova et al. (2012) Phys Rev E 86(5), 051920.
# - Kitzbichler et al. (2009) PLoS Comp Biol 5(3), e1000314.
# - Lee et al. (2019) NeuroImage 188, 228-38.
# - Stam et al. (2007) Hum Brain Mapp 28(11), 1178-93.
# - Yoon et al. (2015) Phys Rev E 91(3), 032814.

Avalanche analysis
==================

An avalanche is a cascade of activity, preceded and followed by silence.
In a system at a critical point, avalanche sizes and durations follow
power laws, with exponents related by a scaling relation, and avalanches
of different durations have the same average shape (Beggs & Plenz, 2003;
Sethna et al., 2001; Friedman et al., 2012). This page goes through the
steps of the analysis and the choices each one involves. The
:ref:`first worked example <sphx_glr_auto_examples_plot_01_avalanches.py>`
runs all of them.

Detecting events and avalanches
-------------------------------

:func:`edgeofpy.detect_avalanches` takes cleaned signals (channel x sample)
and detects events as the peaks of excursions beyond ``threshold`` standard
deviations (2 by default), above, below or both (``thresh_type``). It also
accepts events already binarized, e.g. spikes.

Events are then grouped into avalanches: an interval without events longer
than ``max_iei`` ends an avalanche. ``max_iei`` acts as the **time bin** of
the analysis, and avalanche durations are expressed in these bins
(``dur_bin``).

The time bin is the most consequential choice of the analysis:

- Too short a bin splits avalanches; too long a bin merges independent
  ones. The size exponent changes with the bin width (Beggs & Plenz, 2003),
  who proposed using the mean inter-event interval of the pooled events
  (returned by :func:`edgeofpy.detect_avalanches` as ``mean_iei``).
- Signatures of criticality can be hidden at short time scales. In
  recordings of awake mouse cortex, Fontenele et al. (2024) found power-law
  avalanches and the scaling relation only with bins longer than about
  10 ms; at the millisecond scale, no choice of threshold gave power laws.
  They suggest that insufficient temporal coarse-graining explains some
  negative reports in the literature (see also Hengen & Shew, 2025).

In practice, compute the results for a range of bin widths, from the
sampling interval to a few tens of milliseconds, and report how they
depend on it. The threshold matters too, but usually less; check a few
values (e.g. 1.5 to 3 standard deviations).

Fitting power laws
------------------

:func:`edgeofpy.fit_powerlaw` fits a power law to the avalanche sizes or
durations by maximum likelihood (Clauset et al., 2009), using the powerlaw
package (Alstott et al., 2014), and compares it with alternative
distributions (exponential, lognormal, truncated power law).

The power law is fitted between a lower bound ``xmin`` and an optional
upper bound ``xmax``. Choosing ``xmin`` matters more than any other fitting
choice (:ref:`sphx_glr_auto_examples_plot_02_dcc_reliability.py`):

- **The smallest value** (``xmin='min'``) biases the exponents, because the
  smallest avalanches do not follow the power law.
- **The automatic choice** (``xmin=None``, the default) selects the value
  that makes the fit closest to the data. It suits a single dataset, but it
  can jump between similar recordings, which then get very different
  exponents.
- **Fixed ranges**, the same for all recordings, remove this source of
  spurious differences when comparing recordings. They can be chosen on
  pooled data, or in advance.

Whatever the choice, a power law is plausible only over a sufficient range:
at least a decade of sizes and of durations, with a few hundred avalanches
or more in the fitted range.

The DCC
-------

At a critical point, the size exponent tau, the duration exponent alpha and
the exponent of the mean size as a function of duration (the "third"
exponent) satisfy (alpha - 1) / (tau - 1) = third (Sethna et al., 2001).
The deviation from criticality coefficient (DCC) is the absolute difference
between the two sides (Ma et al., 2019).

:func:`edgeofpy.estimate_dcc` computes it with consistent choices:

- the third exponent is fitted over the same durations as the duration
  power law;
- ``size_range`` and ``dur_range`` fix the fitting ranges, and
  ``xmin_sensitivity=True`` shows how much the DCC depends on them;
- ``n_boot`` gives bootstrap confidence intervals, for the DCC and for the
  signed deviation;
- ``n_avalanches`` estimates the DCC from a fixed number of avalanches;
- it warns when a fitted range holds fewer than 100 avalanches or spans
  less than a decade.

Two properties of the DCC follow from its definition as an absolute value
(:ref:`sphx_glr_auto_examples_plot_02_dcc_reliability.py`):

- **It is biased upward with few avalanches:** errors in the exponents can
  only increase it. Its confidence interval therefore never includes 0. Use
  the confidence interval of the signed deviation to judge whether the
  exponents are consistent with criticality.
- **It depends on the number of avalanches,** so it can differ between
  conditions only because one condition yields more avalanches. Compare
  conditions with the same number of avalanches (``n_avalanches``).

Shape collapse
--------------

At a critical point, the average profile of avalanches of duration T,
rescaled by T to the power gamma, is the same for all durations, with
gamma = third - 1 (Friedman et al., 2012). :func:`edgeofpy.fit_shape_collapse`
finds the gamma that best collapses the profiles, fits the third exponent
over the same durations and returns their deviation (``dcc_collapse``),
with optional bootstrap confidence intervals.
:func:`edgeofpy.plot_shape_collapse` plots the profiles, rescaled or not.

Gamma depends on the shortest duration included (``min_dur``), as short
avalanches deviate from the asymptotic shape. ``gamma_by_min_dur`` shows
this dependence; report gamma together with ``min_dur``.

Branching ratio, susceptibility and Fano factor
-----------------------------------------------

These measures use all events rather than the avalanche distributions, so
they need less data than the exponents:

- :func:`edgeofpy.branching_ratio`, the mean ratio of events in successive
  time bins, is close to 1 at the critical point, below 1 on the
  subcritical side and saturates at 1 on the supercritical side
  (:ref:`sphx_glr_auto_examples_plot_03_criticality_sweep.py`). It is
  biased toward lower values when only a small fraction of the units is
  recorded (Wilting & Priesemann, 2018). Like the avalanches, it depends on
  the time bin (``time_bin_size``).
- :func:`edgeofpy.susceptibility` and :func:`edgeofpy.fano_factor`
  measure the size of the fluctuations of activity, which peak at the
  transition.

References
----------

- Alstott et al. (2014) PLoS ONE 9(1), e85777.
- Beggs & Plenz (2003) J Neurosci 23(35), 11167-77.
- Clauset et al. (2009) SIAM Rev 51(4), 661-703.
- Fontenele et al. (2024) Sci Adv 10(17), eadj9303.
- Friedman et al. (2012) Phys Rev Lett 108(20), 208102.
- Hengen & Shew (2025) Neuron 113(16), 2582-98.
- Ma et al. (2019) Neuron 104(4), 655-64.
- Sethna et al. (2001) Nature 410, 242-50.
- Wilting & Priesemann (2018) Nat Commun 9, 2325.

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
and detects events as the peaks of excursions beyond a threshold, in
standard deviations of each channel, above, below or both
(``thresh_type``). It also accepts events already binarized, e.g. spikes.

By default (``threshold='gaussian'``), the threshold is the amplitude at
which the signal departs from a Gaussian distribution
(:func:`edgeofpy.gaussian_threshold`; Shriki et al., 2013). The amplitudes
of all channels, each z-scored, are pooled, and a Gaussian is fitted to
their density; the threshold is the amplitude beyond which the density
stays more than ``tol`` (2 by default) times above the Gaussian. The idea
is that the Gaussian part reflects noise-like fluctuations and the excess
the large, correlated deflections that form avalanches. Shriki et al.
found the departure at about 2.7 SD in resting MEG.
:func:`edgeofpy.plot_gaussian_threshold` shows the fit; check it, since
the method assumes that the centre of the distribution is Gaussian.

``threshold='gaussian'`` computes a threshold for each recording. To
compare recordings, compute one threshold from all of them (pass them as a
list to :func:`edgeofpy.gaussian_threshold`) and pass it as a number:
otherwise each condition is analysed with its own threshold.

For spikes, :func:`edgeofpy.bin_spikes` turns spike times into counts per
bin, pooled over units or per unit; pass the counts with
``s_freq=1 / bin_dur`` and ``max_iei=bin_dur``. In large populations
(hundreds of neurons or more), bins without any spike are rare, so runs of
non-empty bins merge into very long avalanches or, at fine bins, reflect
chance coincidences between neurons. Use ``pop_threshold`` to define
avalanches as excursions of the population count above a threshold, e.g.
its median (``pop_threshold='median'``).

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
depend on it. The threshold matters too: a lower threshold admits more
events and flattens the size distribution. Check a few values, e.g.
thresholds from ``gaussian_threshold`` with ``tol`` from 1.5 to 5.

Fitting power laws
------------------

:func:`edgeofpy.fit_powerlaw` fits a power law to the avalanche sizes or
durations by maximum likelihood (Clauset et al., 2009), using the powerlaw
package (Alstott et al., 2014), and compares it with alternative
distributions (exponential, lognormal, truncated power law).

The power law is fitted between a lower bound ``xmin`` and an optional
upper bound ``xmax``. The fitting range matters more than any other fitting
choice, and the right range depends on the data.

**Models and large spiking populations.** Here the smallest avalanches
often deviate from the power law: in a branching process, for example, the
shortest avalanches are too frequent. Fitting all avalanches
(``xmin='min'``) then biases the exponents
(:ref:`sphx_glr_auto_examples_plot_02_dcc_reliability.py`).

- **The automatic choice** (``xmin=None``, the default) selects the lower
  bound that makes the fit closest to the data (Clauset et al., 2009). It
  suits a single dataset, but it can jump between similar recordings, which
  then get very different exponents.
- **Fixed ranges**, the same for all recordings, remove this source of
  spurious differences when comparing recordings. They can be chosen on
  pooled data, or in advance.

**Sensor arrays (M/EEG, ECoG, LFP).** An avalanche cannot involve many
more events than there are channels, so the size distribution bends down
near the number of channels N: a finite-size cutoff, which moves with N
(Shriki et al., 2013). Below the cutoff, the smallest avalanches belong to
the scale-free range. Fit sizes from 1 to about 1.5 N, e.g.
``size_range=(1, int(1.5 * n_channels))`` in
:func:`edgeofpy.estimate_dcc`. The automatic choice is unsuitable here: it
assumes that the power law continues above ``xmin``, and tends to place
``xmin`` on the cutoff, so that the fit describes the cutoff (with exponents
near 3) rather than the scale-free range. Durations are limited even more
than sizes, and at the sensor level they largely reflect the spectra of the
signals; check them against surrogates (see below) before interpreting the
duration exponent or the DCC.

Whatever the choice, a power law is plausible only over a sufficient range:
at least a decade of sizes and of durations, with a few hundred avalanches
or more in the fitted range. The powerlaw package searches exponents
between 1 and 3 only; :func:`edgeofpy.fit_powerlaw` warns when a fitted
exponent sits at one of these limits.

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
  less than a decade, and when tau is close to 1, where
  (alpha - 1) / (tau - 1) is very sensitive to small errors in tau.

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
gamma + 1 = (alpha - 1) / (tau - 1) (Friedman et al., 2012).
:func:`edgeofpy.fit_shape_collapse` finds the gamma that best collapses the
profiles, with optional bootstrap confidence intervals. Given the size and
duration exponents (``tau`` and ``alpha``, e.g. from
:func:`edgeofpy.estimate_dcc`), it also returns the deviation from this
relation (``dcc_collapse``). :func:`edgeofpy.plot_shape_collapse` plots the
profiles, rescaled or not.

A good collapse is not by itself a sign of criticality. Look at the shape
of the profiles: at criticality they rise and fall (an inverted parabola
in mean-field models), whereas flat profiles, as produced by bursts of
roughly constant activity, collapse with gamma near 0. The third exponent,
fitted over the same durations, equals gamma + 1 for any process whose
profiles keep their shape, critical or not; ``fit_shape_collapse`` returns
it as a check of the collapse, not as a test of criticality.

Gamma depends on the shortest duration included (``min_dur``), as short
avalanches deviate from the asymptotic shape. ``gamma_by_min_dur`` shows
this dependence; report gamma together with ``min_dur``.
:func:`edgeofpy.fit_shape_collapse` warns when gamma lies at an edge of the
range searched (``gamma_range``); at 0, the profiles do not grow with
duration.

Testing against surrogates
--------------------------

Power laws and scaling can arise without criticality (Touboul & Destexhe,
2017), and in sensor recordings the correlations between channels include
volume conduction. :func:`edgeofpy.phase_randomize` makes surrogate data
with the same power spectra as the data:

- with ``shared=True`` (default), the same random phases are applied to
  all channels, so the surrogate also keeps the linear correlations between
  channels (Prichard & Theiler, 1994);
- with ``shared=False``, the phases are drawn for each channel, so the
  channels become independent (Shriki et al., 2013).

Detect avalanches in the surrogate with the threshold of the data, passed
as a number: the surrogate is Gaussian, so ``threshold='gaussian'`` would
find no departure. A result reproduced by the shared-phase surrogate
carries no information beyond the spectra and cross-spectra of the data.
It may still reflect the distance to criticality, but measures based on
covariances (e.g. :func:`edgeofpy.lambda_max`) capture that information
directly.

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
- Prichard & Theiler (1994) Phys Rev Lett 73(7), 951-4.
- Sethna et al. (2001) Nature 410, 242-50.
- Shriki et al. (2013) J Neurosci 33(16), 7079-90.
- Theiler et al. (1992) Physica D 58(1-4), 77-94.
- Touboul & Destexhe (2017) Phys Rev E 95(1), 012413.
- Wilting & Priesemann (2018) Nat Commun 9, 2325.

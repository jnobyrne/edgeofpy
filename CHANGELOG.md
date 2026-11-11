# Changelog

All notable changes to edgeofpy are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/) (before 1.0, minor versions may
change behaviour).

## [Unreleased]

### Added

- `edgeofpy.models`: simulations of systems with known critical points, for
  validating analyses: `branching_process`, `branching_network`, `kuramoto`,
  `logistic_map`, `linear_network` and `ising_samples`.
- `estimate_dcc`: the DCC with consistent fitting ranges (the third exponent
  is fitted over the duration range of the duration power law), the signed
  deviation `dcc_signed`, diagnostics (avalanches and decades within the
  fitted ranges, with warnings when they are too few), sensitivity to xmin,
  matched avalanche counts (`n_avalanches`) and bootstrap confidence
  intervals.
- Worked examples in the documentation (sphinx-gallery): an end-to-end
  avalanche analysis of a simulated critical network; estimating the DCC
  reliably (choice of xmin, number of avalanches, comparing conditions); how
  the avalanche measures change when tuning a network through its critical
  point; the edge-of-chaos measures (0-1 test, lambda max) on systems with
  known dynamics; fitting Ising models and locating them on the specific
  heat curve and the SK phase diagram; and the synchrony measures (PCF, PLI,
  PLE, GLS) across the Kuramoto transition.
- `fit_shape_collapse`: the shape collapse exponent gamma, with the third
  exponent fitted over the same durations, `dcc_collapse`, the stability of
  gamma across `min_dur` and bootstrap confidence intervals;
  `plot_shape_collapse` plots the rescaled profiles.

### Changed

- `lambda_max` corrects the dispersion of the covariances for the sampling
  variance due to the finite number of trials (`bias_correction=True`).
  Without it, the estimate was biased upward, e.g. 0.57 instead of 0.3 on a
  simulated network with 100 trials. **This changes the results of
  `lambda_max`**; use `bias_correction=False` for the previous values.
- `susceptibility` accepts counts of active units over time (1d) with the
  new `n_units` argument, e.g. the summed activity of all channels.
- `lattice_search` moved to `edgeofpy.utils` (still available as
  `eop.lattice_search`).
- `fit_powerlaw` reads the standard error of the exponent from
  `standard_err`, as `sigma` will be removed in powerlaw 2.1.
- Requires powerlaw >= 2.0. On simulated critical avalanches, powerlaw 1.5
  underestimated the size exponent of discrete data (e.g. 1.43-1.46 instead
  of the true 1.5 at the same `xmin`), which changed the DCC by up to 0.2.

### Fixed

- `branching_ratio` no longer emits divide-by-zero warnings for time bins
  without events (the result is unchanged).
- `pcf` is computed in the frame rotating with the oscillators' group
  angular velocity, as defined by Yoon et al. (2015). It was computed in the
  fixed frame, where the collective rotation of synchronized oscillators
  made it increase monotonically with synchrony instead of peaking at the
  transition. **This changes the results of `pcf`.**

## [0.1.0] - 2026-11-08

This release fixes about 40 bugs, many of which gave wrong results or crashed
on normal input, and adds a test suite. **Results computed with 0.0.1 can
change**; the items under "Changes that affect results" say which.

### Changes that affect results

- `fit_powerlaw`: `xmin` now defaults to None, which selects `xmin`
  automatically (Clauset et al. 2009). The old default, `'min'`, fitted from
  the smallest value and biases avalanche exponents (on a critical branching
  process it inflated the DCC from about 0.03 to 0.65). Pass `xmin='min'` to
  get the old behaviour.
- `fit_third_exponent`:
  - The default method `'mean'` regresses the log mean size on log duration.
    The old default `'pl'` fitted a power-law *distribution* to the mean
    sizes, which is not the size–duration scaling exponent.
  - Method `'all'` regresses log size on log duration over all avalanches.
    The old `'lsf'` regressed the other way and returned the inverse exponent.
  - `'pl'` and `'lsf'` still work as deprecated aliases of `'mean'` and
    `'all'` and emit a `DeprecationWarning`.
  - An invalid method raises `ValueError` instead of returning NaN.
- `shape_collapse_error`: the formula is corrected (profiles are interpolated
  over their whole duration and scaled by dividing by T^γ, as in Marshall et
  al. 2016). Error values are on a different scale from 0.0.1. New
  arguments `min_dur=3` and `min_avl=10` skip durations that are too short or
  too rare; with few long avalanches the function can now raise `ValueError`
  (lower `min_dur` or `min_avl`).
- `detect_avalanches`:
  - The last avalanche is no longer dropped. Only avalanches that may have
    been cut off by the start or end of the recording are discarded.
  - Event arrays of any integer or boolean type (e.g. `bool`, `int32`,
    `uint8`) are recognised as already binarized; they were binarized a
    second time unless their type was exactly `int`.
  - Durations in seconds (e.g. `max_iei`) are rounded to the nearest sample
    instead of truncated, and must amount to at least one sample.
- `avl_branching_ratio`: spurious prefactors removed (a process with
  branching ratio 1 gave 0.033); computed in log space to avoid underflow.
- `susceptibility`: always uses the variance (`np.var`); the `test` argument
  is removed.
- `pli`: the phase difference is wrapped, so lags crossing ±π no longer flip
  sign (a constant lag gave 0.92 instead of 1).
- `ple`: rewritten as the entropy of binary phase-difference patterns of
  length `m` and lag `tau` (Lee et al. 2017). `m` and `tau` were ignored
  before, and the binarization had no effect.
- `global_lability`: no longer offset by the number of channels (constant
  locking gave 4 instead of 0).
- `detect_phase_avls`: counts one event per entry into the phase bin; leaving
  the bin was also counted, doubling the number of avalanches.
- `lambda_max`: rewritten following Dahmen et al. (2019): the covariance of
  summed activity across trials, and the spread of the off-diagonal
  covariances only. It no longer modifies the input array and accepts integer
  data.
- `schreiber_denoise`: fixed (it raised `ValueError`); replaces the middle
  coordinate with the neighbourhood mean as in Schreiber (1993), pads with the
  end values instead of zeros, and is much faster. New arguments `r` and
  `chunk_size`; `chaos_pipeline` gains `denoise_r`.
- Ising models:
  - `affine_trans` (used by `sk_diagram` and `ht_diagram`) keeps the diagonal
    of J at zero; it was set to `mu`, which biased every Metropolis step.
  - `heat_capacity(mode='specific')` returns the specific heat; it returned
    the heat capacity.
  - `fit_pmem` initialises J with the outer product of the means.
- `plot_cdf` puts the data on the x-axis and probabilities on the y-axis
  (they were swapped).

### Added

- `fit_third_exponent(dur_range=(min, max))` restricts the fit to a duration
  range, typically the one used for the duration power law.
- `plot_pdf(ax=...)`, `plot_cdf(complementary=..., xlabel=..., ax=...)`,
  `plot_sk(ax=...)`: plotting functions accept an existing axes and return the
  figure instead of calling `plt.show()`.
- `metropolis` can resume from a checkpoint (`checkpoint_file`) with the
  random stream continuing where it stopped; each cell of `sk_diagram` and
  `ht_diagram` writes its own checkpoint file.
- `edgeofpy.__version__` and `__all__`; the phase-locking avalanche functions
  (`detect_phase_avls`, `detect_phase_avls_full`) are exported.
- A test suite (`pytest`) and continuous integration on Linux and macOS for
  the minimum and latest supported dependency versions.

### Fixed

- Avalanches: `lattice_search` no longer modifies its `func_args` and runs
  the requested number of levels; `plot_pdf` bins discrete data correctly and
  plots against bin centres; `branching_ratio` rounds bin lengths and raises a
  clear error when `s_freq` is missing; `fit_powerlaw` docstring keys match
  the returned dictionary.
- Synchrony: `phase_lock_interval` returns correct durations and handles a
  single or no locking event; `detect_phase_avls_full` runs (it raised
  `TypeError`) and builds the requested number of phase bins;
  `complex_phase_relationship` works with NumPy 2.
- Chaos: `z1_chaos_test` is much faster (FFT-based mean square displacement).
- Ising: `fit_pmem(method='donders')` converges, has an iteration cap and
  returns `h` with the same shape as `'ezaki'`; `sk_diagram(multiproc=True)`
  fills every cell and passes its arguments correctly; `fit_beta` no longer
  modifies `J`; `ht_diagram` validates its range arguments; `metropolis` no
  longer uses complex eigenvalues.
- `z1_chaos_test` and `metropolis` use a local random generator and no longer
  reset NumPy's global random state (their results are unchanged).
- Debug `print` statements replaced by warnings or a `verbose` option;
  docstring, citation and typo fixes throughout.

### Removed

- The `neurokit2` dependency.
- `f_exponential_offset` (unused) and the private `_detect_start_end` from
  the top-level namespace.

### Packaging

- Dependencies are now declared, so `pip install edgeofpy` installs them
  (0.0.1 installed none): numpy >= 1.22, scipy >= 1.7, matplotlib >= 3.5,
  powerlaw >= 1.5. Works with NumPy 2.
- Requires Python 3.10 or later.
- Metadata moved to a standard `[project]` table (PEP 621).

## [0.0.1] - 2023-09-18

First release on PyPI.

[Unreleased]: https://github.com/jnobyrne/edgeofpy/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jnobyrne/edgeofpy/releases/tag/v0.1.0
[0.0.1]: https://pypi.org/project/edgeofpy/0.0.1/

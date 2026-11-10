"""
Avalanche criticality measures
"""

import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import powerlaw

from edgeofpy.utils import (binarized_events, time_bin_events,
                            lattice_search, _sec_to_samples)

def detect_avalanches(data, s_freq, time=None, max_iei=.004, threshold=3,
                      thresh_type='both'):
    """Detect avalanches in coarse-grained electrophysiological
    recordings. Takes either raw or binarized data.

    Parameters
    ----------
    data : 2d array
        Channel x sample. If the dtype is floating point, the array is taken
        as the raw (cleaned) signal, and events are detected with
        `binarized_events`. If the dtype is integer or bool, the array is
        taken as the binarized event time series (bool is converted to int).
    s_freq : int
        Sampling frequency.
    time : 1d array (dtype='float')
        Vector with the time points for each sample. If None, avalanche start
        and end times will be returned in seconds since the start of data.
    max_iei : float, : .004
        Duration, in seconds, of the maximum inter-event interval within which
        successive events are counted as belonging to the same avalanche.
    threshold : float
        Number of standard deviations.  Only signal excursions exceeding
        this threshold are considered events.
    thresh_type : str
        If 'above', an event is detected when signal is greater than thresh.
        If 'below', an event is detected when signal is less than -thresh.
        If 'both', the signal above thresh and below -thresh is detected as an
        event.

    Returns
    -------
    avalanches : list of dict
        List of detected avalanches as dictionaries, with keys:

        start_time : float
            Start time, in seconds since recording start.
        end_time : float
            End time in seconds since recording start.
        size : int
            Number of events (over all channels).
        dur_sec : float
            Duration, in seconds, from first to last event in the
            avalanche.
        dur_bin : int
            Duration, in number of bins, rounded up.
        n_chan : int
            Number of channels containing events.
        profile : 1d array
            Number of events (active channels) in each successive time bin.
    events_all_chan : 2d array of same shape as data, dtype=int
        Time series of zeros with ones at the event peaks.
    events_one_chan : 1d array
        Collapsed event time series.
    mean_iei : float
        Mean inter-event interval, in seconds. NaN if fewer than two events
        were found.

    Notes
    -----
    Avalanches that may be truncated by the edges of the recording are
    discarded, i.e. the first avalanche if it begins less than `max_iei`
    after the recording start, and the last avalanche if it ends less than
    `max_iei` before the recording end.

    References
    ----------
    - Benayoun et al. (2010) J Clin Neurophysiol 27(6) 458-64.
    - Shriki et al. (2013) J Neurosci 33(16), 7079–90.
    - Jannesari et al. (2020) Brain Structure and Function 225, 1169-83.
    """

    data = np.asarray(data)
    if data.dtype == bool:
        data = data.astype(int)
    if not np.issubdtype(data.dtype, np.integer):
        # Detect events as the locations of peaks above threshold
        data = binarized_events(data, threshold=threshold,
                                thresh_type=thresh_type, null_value=0)

    avalanches, events_one_chan = _det_avls(data, s_freq=s_freq, time=time,
                                            max_iei=max_iei)
    idx_events = np.flatnonzero(events_one_chan)
    mean_iei = (np.mean(np.diff(idx_events)) / s_freq if len(idx_events) > 1
                else np.nan)

    return avalanches, data, events_one_chan, mean_iei

def plot_pdf(data, binning='d_lin', n_bins=100, xlabel=None, ylabel=None,
             ax=None):
    """Plot the probability density function of the data in double logarithmic
    axes.

    Parameters
    ----------
    data : array-like
        List of features, e.g. avalanche sizes.
    binning : str
        How to bin. Either 'd_lin' for discrete linear (one bin per integer
        between min and max), 'c_lin' for continuous linear (linear
        progression of bins between min and max) or 'log' (logarithmic
        progression from min to max; requires positive data).
    n_bins : int
        Number of bins to use in the histogram. For 'c_lin' and 'log' options
        only.
    xlabel : str
        Label for x-axis. If None, defaults to 'x'.
    ylabel : str
        Label for y-axis. If None, defaults to 'P(`xlabel`)'
    ax : matplotlib.axes.Axes, optional
        Axes to plot into. If None, a new figure is created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.

    Notes
    -----
    The density is plotted at the bin centres (geometric centres for 'log').
    Empty bins are not plotted.
    """
    data = np.asarray(data)
    d_min, d_max = np.min(data), np.max(data)
    if 'd_lin' == binning:
        # one bin centred on each integer
        centres = np.arange(d_min, d_max + 1)
        bins = np.append(centres - 0.5, d_max + 0.5)
    elif 'c_lin' == binning:
        bins = np.linspace(d_min, d_max, num=n_bins + 1)
        centres = (bins[:-1] + bins[1:]) / 2
    elif 'log' == binning:
        if d_min <= 0:
            raise ValueError("binning='log' requires strictly positive data.")
        bins = np.geomspace(d_min, d_max, num=n_bins + 1)
        centres = np.sqrt(bins[:-1] * bins[1:])
    else:
        raise ValueError("binning must be 'd_lin', 'c_lin' or 'log'.")
    pdf, _ = np.histogram(data, bins=bins, density=True)
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    nonzero = pdf > 0
    ax.loglog(centres[nonzero], pdf[nonzero])
    if xlabel is None:
        xlabel = 'x'
    ax.set_xlabel(xlabel)
    if ylabel is None:
        ylabel = f'P({xlabel})'
    ax.set_ylabel(ylabel)

    return fig

def plot_cdf(data, complementary=False, xlabel=None, ax=None):
    """Plot the empirical cumulative distribution function of the data in
    double logarithmic axes.

    Parameters
    ----------
    data : array-like
        List of features, e.g. avalanche sizes.
    complementary : bool
        If True, plot the complementary cumulative distribution function
        P(X >= x), the usual choice for assessing power-law tails. If False,
        plot P(X <= x).
    xlabel : str
        Label for x-axis. If None, defaults to 'x'.
    ax : matplotlib.axes.Axes, optional
        Axes to plot into. If None, a new figure is created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    """
    x = np.sort(np.asarray(data))
    n = len(x)
    if complementary:
        p = np.arange(n, 0, -1) / n # P(X >= x_i)
    else:
        p = np.arange(1, n + 1) / n # P(X <= x_i)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.loglog(x, p)
    if xlabel is None:
        xlabel = 'x'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(f'P(X >= {xlabel})' if complementary
                  else f'P(X <= {xlabel})')

    return fig

def fit_powerlaw(data, xmin=None, xmax=None, discrete=True):
    """Fit data to a power law and compare the fit to similar
    distributions.

    Parameters
    ----------
    data : array-like
        List of avalanche features, e.g. avalanche sizes.
    xmin : int or float or str or None
        The minimum value above which to fit the data. If None (default),
        xmin is chosen automatically as the value that minimizes the
        Kolmogorov-Smirnov distance between the data and the fitted power law
        (Clauset et al., 2009). If 'min', the minimum value of `data` is used;
        this biases the exponent if the distribution only follows a power law
        above some value, as is typical for avalanches.
    xmax : int or float or str or None
        The maximum value below which to fit the data. If None, there is no
        upper limit. If 'max', the maximum value of `data` will be used.
    discrete : bool
        Whether the data is discrete (integers). Default is True

    Returns
    -------
    dict
        Dictionary with fit parameters, with keys:

        fit : instance of powerlaw.Fit
            Object containing attributes of the power law fit.
        power_law_exp : float
            Exponent of the power law.
        xmin : int or float
            The minimum value of the fitted data.
        xmax : int or float or None
            The maximum value of the fitted data (None if no upper limit).
        power_law_sigma : float
            Standard error of the exponent.
        trunc_pl_exp : float
            Exponent of the truncated power law.
        trunc_pl_xmin : float
            Minimum value of the truncated power law.
        trunc_pl_xmax : float
            Maximum value of the truncated power law.
        PvE : tuple of float
            The loglikelihood ratio R and probability p for power law versus
            exponential fit.
        PvL : tuple of float
            The loglikelihood ratio R and probability p for power law versus
            lognormal fit.
        PvT : tuple of float
            The loglikelihood ratio R and probability p for power law versus
            truncated power law fit.
        TvE : tuple of float
            The loglikelihood ratio R and probability p for truncated power law
            versus exponential fit.
        TvL : tuple of float
            The loglikelihood ratio R and probability p for truncated power law
            versus lognormal fit.
        EvL : tuple of float
            The loglikelihood ratio R and probability p for exponential versus
            lognormal fit.
        best_fit : string
            Type of distribution yielding the best fit of the data can be 
            P (power law), T (truncated power law with exponential upper 
            cutoff), E (exponential), L (lognormal)
        best_fit_R_sum : float
            The summary loglikelihood value for `best_fit`, which is the basis
            for selecting `best_fit`. It is the max of the sums of 
            loglikelihood values R for each fit function.
        P_R_sum, T_R_sum, E_R_sum, L_R_sum : float
            The summary loglikelihood values for each fit function (power law,
            truncated power law, exponential, lognormal).

    Notes
    -----
    .. versionchanged:: 0.1.0
       The default `xmin` is None (automatic selection) instead of 'min'.

    Automatic selection of `xmin` fits a power law above every candidate
    value, which takes a few times longer than a fixed `xmin`, e.g. about a
    minute for 50,000 continuous values.

    From Alstott et al. (2014): R is the loglikelihood ratio between the two
    candidate distributions. This number will be positive if the data is
    more likely in the first distribution, and negative if the data is
    more likely in the second distribution. The significance value for
    that direction is p.

    This function was adapted from code by Marzieh Zare.

    References
    ----------
    - Alstott et al. (2014) PLoS ONE 9(1), e85777.
    - Clauset et al. (2009) SIAM review 51(4), 661-703.
    """
    if xmin == 'min':
        xmin = min(data)
    if xmax == 'max':
        xmax = max(data)

    fit = powerlaw.Fit(data, xmin=xmin, xmax=xmax, discrete=discrete, 
                       verbose=False)

    PvE = fit.distribution_compare('power_law', 'exponential',
                                   normalized_ratio=True)
    PvL = fit.distribution_compare('power_law', 'lognormal',
                                   normalized_ratio=True)
    # the power law is nested within the truncated power law
    PvT = fit.distribution_compare('power_law', 'truncated_power_law',
                                   nested=True, normalized_ratio=True)

    TvE = fit.distribution_compare('truncated_power_law', 'exponential',
                                   normalized_ratio=True)
    TvL = fit.distribution_compare('truncated_power_law', 'lognormal',
                                   normalized_ratio=True)

    EvL = fit.distribution_compare('exponential', 'lognormal',
                                   normalized_ratio=True)

    P = PvE[0] + PvL[0] + PvT[0]
    T = TvE[0] + TvL[0] - PvT[0]
    E = -PvE[0] - TvE[0] + EvL[0]
    L = -PvL[0] - TvL[0] - EvL[0]
    best_fit = 'PTEL'[np.argmax([P, T, E, L])]
    best_fit_R_sum = np.max([P, T, E, L])

    return {'fit': fit,
            'power_law_exp': fit.power_law.alpha,
            'xmin': fit.xmin,
            'xmax': fit.xmax,
            'power_law_sigma': fit.power_law.standard_err,
            'trunc_pl_exp': fit.truncated_power_law.alpha,
            'trunc_pl_xmin': fit.truncated_power_law.xmin,
            'trunc_pl_xmax': fit.truncated_power_law.xmax,
            'PvE': PvE,
            'PvL': PvL,
            'PvT': PvT,
            'TvE': TvE,
            'TvL': TvL,
            'EvL': EvL,
            'best_fit': best_fit,
            'best_fit_R_sum': best_fit_R_sum,
            'P_R_sum': P,
            'T_R_sum': T,
            'E_R_sum': E,
            'L_R_sum': L,}

def fit_third_exponent(sizes, durations, discrete=True, method='mean',
                       dur_range=None):
    """Estimate the third exponent, i.e. the exponent of the scaling relation
    <S>(T) ~ T^third between avalanche size S and duration T, by linear
    regression in log-log coordinates.

    Parameters
    ----------
    sizes : 1d array
        Sequence of avalanche sizes (> 0).
    durations : 1d array
        Corresponding sequence of avalanche durations (> 0).
    discrete : bool
        Whether the duration data is discrete (integers). If False, durations
        are rounded to 3 decimals before grouping. Only used when method is
        'mean'. Default is True.
    method : str
        'mean' to fit the mean size for each unique duration (default).
        'all' to fit the sizes and durations of all individual avalanches.
        'pl' and 'lsf' are deprecated aliases of 'mean' and 'all'.
    dur_range : tuple of (float or None, float or None), optional
        Only avalanches with durations in [min, max] (inclusive) are used,
        typically the range over which the duration power law was fitted,
        e.g. ``dur_range=(fit['xmin'], fit['xmax'])`` with
        ``fit = fit_powerlaw(durations)``. Either bound may be None (no
        limit). If None (default), all avalanches are used.

    Returns
    -------
    third : float
        Estimated 'third' power-law exponent.
    """
    sizes = np.asarray(sizes, dtype=float)
    durations = np.asarray(durations, dtype=float)
    if method in ('pl', 'lsf'):
        new_method = {'pl': 'mean', 'lsf': 'all'}[method]
        warnings.warn(f"method='{method}' is deprecated; use "
                      f"method='{new_method}'.", DeprecationWarning,
                      stacklevel=2)
        method = new_method
    if sizes.shape != durations.shape:
        raise ValueError('sizes and durations must have the same shape.')
    if np.any(sizes <= 0) or np.any(durations <= 0):
        raise ValueError('sizes and durations must be strictly positive.')
    if dur_range is not None:
        low, high = dur_range
        low = -np.inf if low is None else low
        high = np.inf if high is None else high
        if low > high:
            raise ValueError('dur_range: the lower bound exceeds the upper '
                             'bound.')
        in_range = (durations >= low) & (durations <= high)
        sizes, durations = sizes[in_range], durations[in_range]

    if method == 'mean':
        if not discrete:
            durations = np.round(durations, decimals=3)
        # mean size for each unique duration
        T, inverse = np.unique(durations, return_inverse=True)
        S = np.bincount(inverse, weights=sizes) / np.bincount(inverse)
    elif method == 'all':
        T, S = durations, sizes
    else:
        raise ValueError("method must be 'mean' or 'all'.")

    if len(np.unique(T)) < 2:
        raise ValueError('At least two distinct durations are required.')
    # <S>(T) ~ T^third: slope of log size on log duration
    third, _ = np.polyfit(np.log(T), np.log(S), 1)

    return third

def dcc(tau, alpha, third):
    """The deviation from criticality coefficient (DCC) measures the
    degree of deviation of the empirical avalanche power law exponents
    from the "crackling noise" scaling relation, which holds broadly
    for avalanche-critical systems.

    Parameters
    ----------
    tau : float
        The empirical power law exponent of the avalanche size distribution.
    alpha : float
        The empirical power law exponent of the avalanche duration
        distribution.
    third : float
        The empirical power law exponent of the relation between avalanche
        size and duration.

    Returns
    -------
    dcc : float
        The deviation from criticality coefficient.

    References
    ----------
    - Ma et al. (2019) Neuron 104(4), 655-64.
    - Friedman et al. (2012) PRL 108, 208102.
    """
    return np.abs( (alpha - 1) / (tau - 1) - third )

def estimate_dcc(sizes, durations, size_range=(None, None),
                 dur_range=(None, None), discrete=True, third_method='mean',
                 xmin_sensitivity=False, n_avalanches=None, n_subsamples=100,
                 n_boot=0, ci=0.95, rand_seed=0):
    """Estimate the deviation from criticality coefficient (DCC) with
    consistent fitting ranges, diagnostics and confidence intervals.

    The size and duration exponents tau and alpha are fitted with
    `fit_powerlaw`, and the third exponent with `fit_third_exponent` over
    the range of durations used to fit alpha. The DCC is then
    ``|(alpha - 1) / (tau - 1) - third|`` (see `dcc`).

    Parameters
    ----------
    sizes : 1d array
        Avalanche sizes.
    durations : 1d array
        Corresponding avalanche durations.
    size_range : tuple of (float or None, float or None)
        (xmin, xmax) for the size power law. An xmin of None is chosen
        automatically (Clauset et al., 2009); an xmax of None means no upper
        limit. To compare conditions or subjects, fixed values are
        recommended (see Notes).
    dur_range : tuple of (float or None, float or None)
        (xmin, xmax) for the duration power law, as for `size_range`. The
        third exponent is fitted over the same range.
    discrete : bool
        Whether sizes and durations are discrete (integers).
    third_method : str
        Method of `fit_third_exponent`, 'mean' or 'all'.
    xmin_sensitivity : bool
        If True, the DCC is also computed with each xmin halved and doubled
        (a 3 x 3 grid), to show how much it depends on the fitting ranges.
    n_avalanches : int or None
        If given, the estimates are the medians over `n_subsamples` random
        subsets of `n_avalanches` avalanches (drawn without replacement), and
        the bootstrap resamples have this size. Use it to compare conditions
        with different numbers of avalanches, since the DCC is biased upward
        with fewer avalanches. The fitting ranges are those of the full data.
    n_subsamples : int
        Number of subsets used when `n_avalanches` is given.
    n_boot : int
        Number of bootstrap resamples for confidence intervals (0: none).
    ci : float
        Coverage of the confidence intervals, e.g. 0.95.
    rand_seed : int or numpy.random.Generator or None
        Seed or random number generator for subsampling and bootstrapping.

    Returns
    -------
    dict
        Dictionary with the keys:

        dcc : float
            Deviation from criticality coefficient.
        dcc_signed : float
            Signed deviation, (alpha - 1) / (tau - 1) - third, of which the
            DCC is the absolute value.
        tau, alpha, third : float
            Size, duration and third exponents.
        size_range, dur_range : tuple
            (xmin, xmax) used for the sizes and durations.
        n_avalanches : int
            Number of avalanches per estimate.
        n_tail_sizes, n_tail_durations : int
            Number of avalanches within the size and duration ranges (all
            data).
        size_decades, dur_decades : float
            Number of decades spanned by the fitted size and duration ranges.
        size_fit, dur_fit : dict
            Outputs of `fit_powerlaw` on all data, including the
            likelihood-ratio comparisons with other distributions.
        dcc_ci, dcc_signed_ci : tuple or None
            Bootstrap confidence intervals (None if `n_boot` is 0).
        boot : dict or None
            Bootstrap values of 'tau', 'alpha', 'third', 'dcc' and
            'dcc_signed' (None if `n_boot` is 0).
        dcc_xmin_grid : 2d array or None
            DCC with the size xmin (rows) and duration xmin (columns)
            multiplied by 0.5, 1 and 2 (None unless `xmin_sensitivity`).
        dcc_xmin_range : tuple or None
            Minimum and maximum of `dcc_xmin_grid`.
        powerlaw_version : str
            Version of the powerlaw package, worth reporting with results.

    Warns
    -----
    UserWarning
        If fewer than 100 avalanches fall within a fitted range, or a fitted
        range spans less than one decade.

    Notes
    -----
    The fitting ranges are chosen once, on all avalanches, and kept fixed
    for the subsets and bootstrap resamples. Bootstrap resamples draw
    avalanches with replacement, keeping each size with its duration, and
    give percentile confidence intervals. They treat avalanches as
    independent, so with temporally correlated avalanches the intervals are
    somewhat too narrow.

    Because the DCC is an absolute value, noise biases it upward: even for a
    critical system, its confidence interval excludes 0 when there are few
    avalanches. The signed deviation does not have this bias; a confidence
    interval of `dcc_signed` that includes 0 indicates that the exponents are
    consistent with the crackling-noise relation.

    At the mean-field exponents (tau = 3/2, alpha = 2), small errors
    delta_tau, delta_alpha and delta_third change the signed deviation by
    about -4 delta_tau + 2 delta_alpha - delta_third (a first-order
    approximation; the factors grow as tau approaches 1). The DCC is
    therefore most sensitive to the size exponent and to the fitting ranges.

    The automatic choice of xmin is not always stable: for two recordings of
    the same system, or two subsets of one recording, it can select quite
    different values (e.g. 3 for one and 21 for the other), which changes the
    exponents and the DCC even though the underlying dynamics are the same.
    When comparing conditions or subjects, fixed fitting ranges are
    therefore preferable; `xmin_sensitivity` shows how much the DCC depends
    on them.

    References
    ----------
    - Ma et al. (2019) Neuron 104(4), 655-64.
    - Clauset et al. (2009) SIAM Rev 51(4), 661-703.
    """
    sizes = np.asarray(sizes, dtype=float)
    durations = np.asarray(durations, dtype=float)
    if sizes.shape != durations.shape or sizes.ndim != 1:
        raise ValueError('sizes and durations must be 1d arrays of the same '
                         'length.')
    if not 0 < ci < 1:
        raise ValueError('ci must be between 0 and 1.')
    n_total = sizes.size
    if n_avalanches is not None and not 1 < n_avalanches <= n_total:
        raise ValueError(f'n_avalanches must be between 2 and the number of '
                         f'avalanches ({n_total}).')
    rng = np.random.default_rng(rand_seed)

    # fitting ranges, from all avalanches
    size_fit = fit_powerlaw(sizes, xmin=size_range[0], xmax=size_range[1],
                            discrete=discrete)
    dur_fit = fit_powerlaw(durations, xmin=dur_range[0], xmax=dur_range[1],
                           discrete=discrete)
    s_range = (size_fit['xmin'], size_fit['xmax'])
    d_range = (dur_fit['xmin'], dur_fit['xmax'])

    n_tail = {}
    decades = {}
    for name, data, (low, high) in (('sizes', sizes, s_range),
                                    ('durations', durations, d_range)):
        upper = np.inf if high is None else high
        in_range = data[(data >= low) & (data <= upper)]
        n_tail[name] = in_range.size
        top = high if high is not None else in_range.max(initial=low)
        decades[name] = np.log10(top / low)
        if n_tail[name] < 100:
            warnings.warn(f'Only {n_tail[name]} avalanche {name} are within '
                          f'the fitted range; the exponents are unreliable.',
                          stacklevel=2)
        if decades[name] < 1:
            warnings.warn(f'The fitted range of avalanche {name} spans '
                          f'{decades[name]:.2f} decades (< 1); a power law '
                          f'cannot be established over such a short range.',
                          stacklevel=2)

    keys = ('tau', 'alpha', 'third', 'dcc', 'dcc_signed')

    def estimate(idx):
        return _dcc_fixed_ranges(sizes[idx], durations[idx], s_range,
                                 d_range, discrete, third_method)

    # point estimates
    if n_avalanches is None:
        n_used = n_total
        point = {'tau': size_fit['power_law_exp'],
                 'alpha': dur_fit['power_law_exp']}
        point['third'] = fit_third_exponent(sizes, durations,
                                            discrete=discrete,
                                            method=third_method,
                                            dur_range=d_range)
        point['dcc_signed'] = ((point['alpha'] - 1) / (point['tau'] - 1)
                               - point['third'])
        point['dcc'] = abs(point['dcc_signed'])
    else:
        n_used = n_avalanches
        subs = [estimate(rng.choice(n_total, n_used, replace=False))
                for _ in range(n_subsamples)]
        point = {k: np.nanmedian([s[k] for s in subs]) for k in keys}

    # bootstrap
    boot = dcc_ci = dcc_signed_ci = None
    if n_boot:
        reps = [estimate(rng.integers(n_total, size=n_used))
                for _ in range(n_boot)]
        boot = {k: np.array([r[k] for r in reps]) for k in keys}
        q = 100 * np.array([(1 - ci) / 2, (1 + ci) / 2])
        dcc_ci = tuple(np.nanpercentile(boot['dcc'], q))
        dcc_signed_ci = tuple(np.nanpercentile(boot['dcc_signed'], q))

    # sensitivity to xmin
    grid = grid_range = None
    if xmin_sensitivity:
        factors = (0.5, 1, 2)
        grid = np.full((3, 3), np.nan)
        for i, fs in enumerate(factors):
            for j, fd in enumerate(factors):
                sr = (_scale_xmin(s_range[0], fs, discrete), s_range[1])
                dr = (_scale_xmin(d_range[0], fd, discrete), d_range[1])
                grid[i, j] = _dcc_fixed_ranges(sizes, durations, sr, dr,
                                               discrete, third_method)['dcc']
        grid_range = (np.nanmin(grid), np.nanmax(grid))

    return {'dcc': point['dcc'],
            'dcc_signed': point['dcc_signed'],
            'tau': point['tau'],
            'alpha': point['alpha'],
            'third': point['third'],
            'size_range': s_range,
            'dur_range': d_range,
            'n_avalanches': n_used,
            'n_tail_sizes': n_tail['sizes'],
            'n_tail_durations': n_tail['durations'],
            'size_decades': decades['sizes'],
            'dur_decades': decades['durations'],
            'size_fit': size_fit,
            'dur_fit': dur_fit,
            'dcc_ci': dcc_ci,
            'dcc_signed_ci': dcc_signed_ci,
            'boot': boot,
            'dcc_xmin_grid': grid,
            'dcc_xmin_range': grid_range,
            'powerlaw_version': getattr(powerlaw, '__version__', 'unknown')}


def _scale_xmin(xmin, factor, discrete):
    """Multiply xmin by factor (rounded and at least 1 for discrete data)."""
    xmin = xmin * factor
    return max(1, int(round(xmin))) if discrete else xmin


def _pl_exponent(data, xmin, xmax, discrete):
    """Power-law exponent for a fixed fitting range (NaN if it fails)."""
    upper = np.inf if xmax is None else xmax
    if np.sum((data >= xmin) & (data <= upper)) < 2:
        return np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fit = powerlaw.Fit(data, xmin=xmin, xmax=xmax, discrete=discrete,
                           verbose=False)
        return fit.power_law.alpha


def _dcc_fixed_ranges(sizes, durations, size_range, dur_range, discrete,
                      third_method):
    """Exponents and DCC for fixed fitting ranges."""
    tau = _pl_exponent(sizes, *size_range, discrete)
    alpha = _pl_exponent(durations, *dur_range, discrete)
    try:
        third = fit_third_exponent(sizes, durations, discrete=discrete,
                                   method=third_method, dur_range=dur_range)
    except (ValueError, TypeError, np.linalg.LinAlgError):
        third = np.nan
    signed = (alpha - 1) / (tau - 1) - third
    return {'tau': tau, 'alpha': alpha, 'third': third,
            'dcc': abs(signed), 'dcc_signed': signed}


def shape_collapse_error(avalanches, gamma, interp_n=1000, min_dur=3,
                         min_avl=10):
    """Estimate the error of the shape collapse given a scaling factor
    gamma.

    The mean profile of avalanches of each duration T is interpolated over
    rescaled time t/T in [0, 1] and divided by T**gamma. The error is the mean
    variance across durations, normalized by the squared span of the rescaled
    profiles.

    Parameters
    ----------
    avalanches : list of dict
        Detected avalanches.
    gamma : float
        Scaling factor, corresponding to the critical exponent for the shape
        collapse.
    interp_n : int
        Number of points for interpolating avalanche profiles.
    min_dur : int
        Minimum duration, in bins, for an avalanche duration to be included.
    min_avl : int
        Minimum number of avalanches of a given duration for that duration
        to be included.

    Returns
    -------
    collapse_error : float
        The error of the shape collapse. A good collapse should minimize it.

    References
    ----------
    - Marshall et al. (2016) Front Physiol 7, 250.
    - Friedman et al. (2012) PRL 108, 208102.
    """
    T_set, _, profiles = _collapse_profiles(avalanches, interp_n, min_dur,
                                            min_avl)
    return _collapse_error(T_set, profiles, gamma)


def _collapse_profiles(avalanches, interp_n, min_dur, min_avl):
    """Durations used for the shape collapse, and the mean profile of the
    avalanches of each of these durations, interpolated over t/T in [0, 1].
    """
    # Select durations with enough avalanches
    durations = np.array([x['dur_bin'] for x in avalanches])
    T_set = [T for T in np.unique(durations)
             if T >= min_dur and np.sum(durations == T) >= min_avl]
    if len(T_set) < 2:
        raise ValueError('Fewer than two avalanche durations satisfy min_dur '
                         'and min_avl; cannot compute the shape collapse.')

    # Interpolate the average profile for each duration T over t/T in [0, 1]
    x_interp = np.linspace(0, 1, interp_n)
    profiles = np.zeros((len(T_set), interp_n))
    for i, T in enumerate(T_set):
        avls_T = [x for x in avalanches if x['dur_bin'] == T]
        avg_profile = np.mean(np.vstack([x['profile'] for x in avls_T]),
                              axis=0)
        profiles[i, :] = np.interp(x_interp, np.linspace(0, 1, T),
                                   avg_profile)
    return T_set, x_interp, profiles


def _rescale_profiles(T_set, profiles, gamma):
    """Divide the mean profile of each duration T by T**gamma."""
    return profiles / np.asarray(T_set, dtype=float)[:, np.newaxis] ** gamma


def _collapse_error(T_set, profiles, gamma):
    """Shape collapse error of mean profiles rescaled with gamma."""
    rescaled_profiles = _rescale_profiles(T_set, profiles, gamma)
    variance = np.var(rescaled_profiles, axis=0)
    squared_span = np.square(np.ptp(rescaled_profiles))
    return np.mean(variance) / squared_span

def dcc_collapse(gamma, third):
    """The deviation from criticality coefficient for the scaling relation
    between the shape collapse exponent and the so-called "third" avalanche
    exponent.

    Parameters
    ----------
    gamma : float
        The empirical power law exponent for the shape collapse.
    third : float
        The empirical power law exponent of the relation between avalanche
        size and duration.

    Returns
    -------
    dcc_collapse : float
        The deviation from criticality coefficient for shape collapse.

    References
    ----------
    Friedman et al. (2012) PRL 108, 208102.
    """
    return np.abs( (third - 1) - gamma )

def dcc_collapse_2(tau, alpha, gamma):
    """The deviation from criticality coefficient for the scaling relation
    between the avalanche size and duration exponents and the shape collapse
    exponent, (alpha - 1) / (tau - 1) = gamma + 1.

    Parameters
    ----------
    tau : float
        The empirical power law exponent of the avalanche size distribution.
    alpha : float
        The empirical power law exponent of the avalanche duration
        distribution.
    gamma : float
        The empirical power law exponent for the shape collapse.

    Returns
    -------
    float
        The deviation from criticality coefficient.

    References
    ----------
    Friedman et al. (2012) PRL 108, 208102.
    """
    return np.abs( (alpha - 1) / (tau - 1) - (gamma + 1) )

def fit_shape_collapse(avalanches, gamma_range=(0., 3.), step=0.1,
                       n_levels=2, min_dur=3, min_avl=10, interp_n=1000,
                       min_dur_check=(3, 5, 8, 12), n_boot=0, ci=0.95,
                       rand_seed=0):
    """Fit the shape collapse exponent gamma, and compare it with the third
    exponent fitted over the same durations.

    At criticality, the mean temporal profiles of avalanches of different
    durations T collapse onto a single curve when time is rescaled by T and
    the profiles by T**gamma, with gamma = third - 1. Gamma is found by
    minimizing `shape_collapse_error` with `lattice_search`. The third
    exponent is fitted (with `fit_third_exponent`) over exactly the
    durations used for the collapse: with finite data, both exponents depend
    on the range of durations, and fitting them over different ranges biases
    their comparison.

    Parameters
    ----------
    avalanches : list of dict
        Detected avalanches, e.g. from `detect_avalanches`.
    gamma_range : tuple of float
        Range of gamma values searched.
    step : float
        Coarsest step of the search for gamma.
    n_levels : int
        Number of refinement levels of the search (each 10x finer).
    min_dur : int
        Minimum duration, in bins, for an avalanche duration to be included.
    min_avl : int
        Minimum number of avalanches of a given duration for that duration
        to be included.
    interp_n : int
        Number of points for interpolating avalanche profiles.
    min_dur_check : sequence of int or None
        Values of `min_dur` for which gamma is also fitted, to check its
        stability. A trend with `min_dur` indicates corrections to scaling
        at short durations.
    n_boot : int
        Number of bootstrap resamples for confidence intervals (0: none).
        Each resample repeats the fit, within +/- 0.5 of the gamma of all
        data, which can take a few seconds.
    ci : float
        Coverage of the confidence intervals, e.g. 0.95.
    rand_seed : int or numpy.random.Generator or None
        Seed or random number generator for bootstrapping.

    Returns
    -------
    dict
        Dictionary with the keys:

        gamma : float
            Shape collapse exponent.
        collapse_error : float
            `shape_collapse_error` at `gamma`.
        third : float
            Third exponent, over the durations used for the collapse.
        dcc_collapse : float
            ``|third - 1 - gamma|`` (see `dcc_collapse`).
        durations : 1d array
            Durations used for the collapse.
        n_avalanches_used : int
            Number of avalanches with these durations.
        gamma_by_min_dur : dict
            Gamma for each value of `min_dur_check` (NaN where too few
            durations qualify).
        gamma_ci, dcc_collapse_ci : tuple or None
            Bootstrap confidence intervals (None if `n_boot` is 0).
        boot : dict or None
            Bootstrap values of 'gamma', 'third' and 'dcc_collapse'.
        x : 1d array
            Rescaled time t/T of the profiles.
        profiles : dict
            Mean profile of each duration T, interpolated over `x` and
            divided by T**gamma, for plotting with `plot_shape_collapse`.
        raw_profiles : dict
            Mean profile of each duration T (T values, one per time bin),
            neither rescaled nor interpolated.

    Notes
    -----
    Bootstrap resamples draw avalanches with replacement and give percentile
    confidence intervals. They treat avalanches as independent, so with
    temporally correlated avalanches the intervals are somewhat too narrow.

    References
    ----------
    - Friedman et al. (2012) PRL 108, 208102.
    - Marshall et al. (2016) Front Physiol 7, 250.
    """
    if not 0 < ci < 1:
        raise ValueError('ci must be between 0 and 1.')
    rng = np.random.default_rng(rand_seed)
    avalanches = list(avalanches)

    def fit(avls, g_range, min_dur):
        T_set, x, profiles = _collapse_profiles(avls, interp_n, min_dur,
                                                min_avl)
        gamma, error = lattice_search(
            lambda g: _collapse_error(T_set, profiles, g), [g_range], 0,
            step, n_levels=n_levels)
        in_set = [a for a in avls if a['dur_bin'] in set(T_set)]
        third = fit_third_exponent([a['size'] for a in in_set],
                                   [a['dur_bin'] for a in in_set])
        return {'gamma': gamma, 'collapse_error': error, 'third': third,
                'dcc_collapse': dcc_collapse(gamma, third), 'T_set': T_set,
                'n_used': len(in_set), 'x': x, 'profiles': profiles}

    res = fit(avalanches, gamma_range, min_dur)

    gamma_by_min_dur = {}
    for md in (min_dur_check or ()):
        try:
            gamma_by_min_dur[md] = fit(avalanches, gamma_range, md)['gamma']
        except ValueError:
            gamma_by_min_dur[md] = np.nan

    boot = gamma_ci = dcc_ci = None
    if n_boot:
        g_range = (max(gamma_range[0], res['gamma'] - 0.5),
                   min(gamma_range[1], res['gamma'] + 0.5))
        keys = ('gamma', 'third', 'dcc_collapse')
        boot = {k: np.full(n_boot, np.nan) for k in keys}
        for b in range(n_boot):
            idx = rng.integers(len(avalanches), size=len(avalanches))
            try:
                rep = fit([avalanches[i] for i in idx], g_range, min_dur)
            except ValueError:
                continue
            for k in keys:
                boot[k][b] = rep[k]
        q = 100 * np.array([(1 - ci) / 2, (1 + ci) / 2])
        gamma_ci = tuple(np.nanpercentile(boot['gamma'], q))
        dcc_ci = tuple(np.nanpercentile(boot['dcc_collapse'], q))

    scaled = _rescale_profiles(res['T_set'], res['profiles'], res['gamma'])
    raw = {T: np.mean([a['profile'] for a in avalanches if a['dur_bin'] == T],
                      axis=0)
           for T in res['T_set']}
    return {'gamma': res['gamma'],
            'collapse_error': res['collapse_error'],
            'third': res['third'],
            'dcc_collapse': res['dcc_collapse'],
            'durations': np.asarray(res['T_set']),
            'n_avalanches_used': res['n_used'],
            'gamma_by_min_dur': gamma_by_min_dur,
            'gamma_ci': gamma_ci,
            'dcc_collapse_ci': dcc_ci,
            'boot': boot,
            'x': res['x'],
            'profiles': dict(zip(res['T_set'], scaled)),
            'raw_profiles': raw}


def plot_shape_collapse(result, collapsed=True, bin_dur=None, ax=None,
                        cmap='viridis', colorbar=True):
    """Plot the mean avalanche profiles of a shape collapse, rescaled
    (collapsed) or as they are.

    Parameters
    ----------
    result : dict
        Output of `fit_shape_collapse`.
    collapsed : bool
        If True (default), plot the profiles against t/T, divided by
        T**gamma. If False, plot the mean profiles as they are, against time
        since avalanche start, as in the top panels of Fig. 3 in Friedman et
        al. (2012).
    bin_dur : float or None
        Duration of one time bin of the profiles, in seconds, i.e. the
        `max_iei` given to `detect_avalanches` (exactly: `max_iei` rounded
        to whole samples). If given, uncollapsed profiles are plotted against
        time in ms; if None, against time in bins. Ignored if `collapsed`.
    ax : matplotlib.axes.Axes or None
        Axes to plot into. If None, a new figure is created.
    cmap : str or matplotlib.colors.Colormap
        Colormap for the durations (log scale).
    colorbar : bool
        Whether to add a colorbar of the durations.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.

    Examples
    --------
    Profiles before and after the collapse, side by side:

    >>> res = fit_shape_collapse(avalanches)
    >>> fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    >>> plot_shape_collapse(res, collapsed=False, bin_dur=0.004, ax=ax1,
    ...                     colorbar=False)
    >>> plot_shape_collapse(res, ax=ax2)

    References
    ----------
    Friedman et al. (2012) PRL 108, 208102.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    if isinstance(cmap, str):
        cmap = matplotlib.colormaps[cmap]
    durations = list(result['profiles'])
    norm = matplotlib.colors.LogNorm(vmin=min(durations),
                                     vmax=max(durations))
    if collapsed:
        for T, profile in result['profiles'].items():
            ax.plot(result['x'], profile, color=cmap(norm(T)), lw=1)
        ax.set_xlabel('t / T')
        ax.set_ylabel(r'mean profile / $T^{\gamma}$')
        ax.set_title(rf"collapsed, $\gamma$ = {result['gamma']:.3f}")
    else:
        scale = 1 if bin_dur is None else 1000 * bin_dur
        for T, profile in result['raw_profiles'].items():
            ax.plot(np.arange(T) * scale, profile, color=cmap(norm(T)), lw=1)
        ax.set_xlabel('t (bins)' if bin_dur is None else 't (ms)')
        ax.set_ylabel('mean profile (events per bin)')
        ax.set_title('mean profiles')
    if colorbar:
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
        fig.colorbar(sm, ax=ax, label='duration T (bins)')
    return fig


def avl_repertoire(avalanches):
    """Yields the number of unique avalanche patterns, where the pattern of an
    avalanche is the configuration of spatial nodes activated at least once by
    the avalanche.

    Parameters
    ----------
    avalanches : list of dict
        Avalanche list as output by eop.detect_avalanches.

    Returns
    -------
    repertoire : 2d array, dtype=int
        The repertoire of unique avalanche patterns, where axis=0 is the
        individual patterns and axis=1 is the spatial nodes (channels). The
        matrix is binary, with 1 indicating that the node was involved in the
        avalanche, and 0 indicating that it was not.

    References
    ----------
    Sorrentino et al. (2021) Sci Rep 11(1), 1-12.
    """
    repertoire = np.vstack([x['pattern'] for x in avalanches])

    return np.unique(repertoire, axis=0)

def avl_pattern_dissimilarity(repertoire, norm=False, similarity=False):
    """Compute a Hamming distance matrix for the avalanche repertoire.

    Parameters
    ----------
    repertoire : 2d array, dtype=int
        The repertoire of unique avalanche patterns, where axis=0 is the
        individual patterns and axis=1 is the spatial nodes (channels). The
        matrix is binary, with 1 indicating that the node was involved in the
        avalanche, and 0 indicating that it was not.
    norm : bool
        If True, dissimilarity values will be normalized with respect to 
        maximum possible dissimilarity. This is done automatically if 
        `similarity` is True.
    similarity : bool
        If True, similarity values (between 0 and 1) will be returned instead 
        of dissimilarity values.

    Returns
    -------
    dissimilarity : 2d matrix, dtype=int
        The symmetrical n_pattern x n_pattern matrix of the dissimilarity 
        between each unique avalanche pattern in the repertoire, with 0 
        indicating identity (zeros should only be found along the diagonal).

    References
    ----------
    Sorrentino et al. (2021) Sci Rep 11(1), 1-12.
    """    
    n_chan = repertoire.shape[1] # maximum dissimilarity
    matrix = 2 * np.inner(repertoire - 0.5, 0.5 - repertoire) + n_chan / 2

    if norm or similarity:
        matrix = matrix / n_chan

    if similarity:
        matrix = 1 - matrix
    
    return matrix

def avl_branching_ratio(avalanches):
    """Branching ratio, calculated avalanche-wise and using geometric mean.

    Parameters
    ----------
    avalanches : list of dict
        Avalanche list as output by eop.detect_avalanches.

    Returns
    -------
    sigma : float
        Branching ratio, where sigma < 1 is subcritical and sigma > 1 is
        supercritical. NaN if no avalanche spans at least two time bins.

    References
    ----------
    Sorrentino et al. (2021) Sci Rep 11(1), 1-12.
    """
    # Geometric means are computed in log space to avoid underflow/overflow
    log_sigma_per_avl = []
    for avl in avalanches:
        p = np.asarray(avl['profile'], dtype=float)
        if len(p) < 2:
            continue
        # geometric mean of the ratios between successive bins
        log_sigma_per_avl.append(np.mean(np.log(p[1:] / p[:-1])))

    if not log_sigma_per_avl:
        warnings.warn('No avalanche spans at least two time bins; returning '
                      'NaN.', RuntimeWarning, stacklevel=2)
        return np.nan

    # geometric mean across avalanches
    sigma = np.exp(np.mean(log_sigma_per_avl))

    return sigma

def branching_ratio(events, time_bin_size=None, s_freq=None):
    """Calculate the branching ratio, which is the average number of events
    generated in the next time bin by a single event, evaluated for every bin
    containing at least one event.

    It is computed as the mean, over all pairs of successive bins where the
    first bin contains at least one event, of the ratio of event counts.
    Note that this simple estimator is biased under subsampling; see
    Priesemann et al. (2014) for alternatives.

    Parameters
    ----------
    events : 1darray (dtype=int)
        Signal with number of events occurring at every time step.
    time_bin_size : float or None
        Length of time bin in seconds. If None, events will not be binned.
    s_freq : int or None
        Sampling frequency of `events`. Only needed if time binning is used.

    Returns
    -------
    br : float (> 0)
        The branching ratio.

    References
    ----------
    - Beggs & Plenz (2003) J Neurosci 23(35), 11167-77.
    - Priesemann et al. (2014) Front Syst Neurosci 8, 108.
    """
    if time_bin_size is not None:
        if s_freq is None:
            raise ValueError('s_freq is required when time_bin_size is '
                             'given.')
        bin_len = _sec_to_samples(time_bin_size, s_freq,
                                  name='time_bin_size')
        events = time_bin_events(events, bin_len)
    # masked_invalid excludes instances where the first bin contains no spikes
    # (resulting in inf)
    br = np.ma.masked_invalid(events[1:] / events[:-1]).mean()

    return br

def susceptibility(events):
    """Estimate the susceptibility and local time fluctuation of the system.

    Parameters
    ----------
    events : 1d or 2d array
        If 2d: channel x time binary signal where 1 is active and 0 is
        inactive. If 1d: the density of active units over time (fraction of
        active channels, between 0 and 1); the system size is then taken as
        N = 1.

    Returns
    -------
    chi : float
        Susceptibility, in the statistical physics sense.
    ltf : float
        Local time fluctuation, i.e. the susceptibility `chi` normalized with
        respect to system size.

    References
    ----------
    Fosque et al. (2021) PRL 126, 098101.
    """
    events = np.asarray(events)
    if events.ndim == 2:
        N = events.shape[0]
        density = np.mean(events, axis=0)
    else:
        N = 1
        density = events

    # N * (<rho^2> - <rho>^2); np.var avoids catastrophic cancellation
    chi = N * np.var(density)
    ltf = (1 / np.mean(density)) * np.sqrt(chi / N)

    return chi, ltf

def fano_factor(data):
    """Compute the Fano factor, i.e. the variance of event counts divided by
    their mean, for counts taken in consecutive windows of equal length.

    Parameters
    ----------
    data : 1d array
        Number of events in each consecutive time window (e.g. the output of
        `time_bin_events`). The window length sets the time scale of the
        Fano factor.

    Returns
    -------
    F : float
        The Fano factor. F = 1 for a Poisson process; F > 1 indicates
        over-dispersed (e.g. bursty or correlated) activity.

    Notes
    -----
    The Fano factor depends on the window length. For processes with
    long-range temporal correlations, as expected near criticality, it grows
    with window length as a power law rather than converging to a constant,
    so comparing it across window lengths is more informative than a single
    value.
    """
    F = np.var(data) / np.mean(data)
    return F

def _det_avls(events, s_freq, time, max_iei):
    """Utility function for detecting avalanches from event time series."""
    # Maximum inter-event interval, in samples
    max_iei_len = _sec_to_samples(max_iei, s_freq, name='max_iei')

    # If multichannel, collapse all channels into one
    events = np.asarray(events)
    multichannel = events.squeeze().ndim > 1
    if multichannel:
        events_one_chan = np.sum(events, axis=0)
    else:
        events_one_chan = events.squeeze()

    # Obtain inter-event intervals
    idx_events = np.flatnonzero(events_one_chan)
    if len(idx_events) < 2:
        return [], events_one_chan

    iei = np.diff(idx_events)

    # Avalanche boundaries: an inter-event interval longer than max_iei
    # separates two avalanches. End indices are exclusive.
    gaps = np.flatnonzero(iei > max_iei_len)
    starts = np.concatenate(([idx_events[0]], idx_events[gaps + 1]))
    ends = np.concatenate((idx_events[gaps] + 1, [idx_events[-1] + 1]))

    # Discard avalanches that may be truncated by the edges of the recording
    keep = np.ones(len(starts), dtype=bool)
    if idx_events[-1] + max_iei_len >= len(events_one_chan):
        keep[-1] = False
    if idx_events[0] < max_iei_len:
        keep[0] = False

    # Characterize avalanches
    avalanches = []
    for avl_start, avl_end in zip(starts[keep], ends[keep]):
        profile_raw = events_one_chan[avl_start:avl_end]
        if time is not None:
            start_time = time[avl_start]
            end_time = time[avl_end]
        else:
            start_time = avl_start / s_freq
            end_time = avl_end / s_freq
        size = int(np.sum(profile_raw))
        dur_sec = (avl_end - avl_start) / s_freq
        dur_bin = int(np.ceil((avl_end - avl_start) / max_iei_len))
        if multichannel:
            pattern = np.sign(np.sum(events[:, avl_start:avl_end], axis=1))
            n_chan = int(np.sum(pattern))
        else:
            pattern = n_chan = 'N/A'
        profile = time_bin_events(profile_raw, max_iei_len)
        avl = {'start_time': start_time,
               'end_time': end_time,
               'size': size,
               'dur_sec': dur_sec,
               'dur_bin': dur_bin,
               'n_chan': n_chan,
               'profile': profile,
               'pattern': pattern}
        avalanches.append(avl)

    return avalanches, events_one_chan

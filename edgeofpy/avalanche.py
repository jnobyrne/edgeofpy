"""
Avalanche criticality measures
"""

import warnings

import numpy as np
import matplotlib.pyplot as plt
import powerlaw

from edgeofpy.utils import binarized_events, time_bin_events, _sec_to_samples

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
            - start_time : float
                Start time, in seconds since recording start.
            - end_time : float
                End time in seconds since recording start.
            - size : int
                Number of events (over all channels).
            - dur_sec : float
                Duration, in seconds, from first to last event in the
                avalanche.
            - dur_bin : int
                Duration, in number of bins, rounded up.
            - n_chan : int
                Number of channels containing events.
            - profile : 1d array
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
    Benayoun et al. (2010) J Clin Neurophysiol 27(6) 458-64.
    Shriki et al. (2013) J Neurosci 33(16), 7079–90.
    Jannesari et al. (2020) Brain Structure and Function 225, 1169-83.
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
    fig: matplotlib.figure.Figure
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
    fig: matplotlib.figure.Figure
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
    discrete: bool
        Whether the data is discrete (integers). Default is True

    Returns
    -------
    dict
        Dictionary with fit parameters, with keys/values:
        fit : instance of powerlaw.Fit
            Object containing attributes of the power law fit.
        power_law_exp : float
            Exponent of the power law.
        xmin: int or float
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

    References
    ----------
    Alstott et al. (2014) PLoS one 9(1), e85777.
    Clauset et al. (2009) SIAM review 51(4), 661-703.

    Acknowledgements
    ----------------
    This function was adapted from code by Marzieh Zare.
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
            'power_law_sigma': fit.power_law.sigma,
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
    discrete: bool
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
    Ma et al. (2019) Neuron 104(4), 655-64.
    Friedman et al. (2012) PRL 108, 208102.
    """
    return np.abs( (alpha - 1) / (tau - 1) - third )

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
    Marshall et al. (2016) Front Physiol 7, 250.
    Friedman et al. (2012) PRL 108, 208102.
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

    # Rescale the profiles (y-axis) by T**gamma
    rescaling_vector = np.asarray(T_set, dtype=float)[:, np.newaxis] ** gamma
    rescaled_profiles = profiles / rescaling_vector

    # Calculate collapse error
    variance = np.var(rescaled_profiles, axis=0)
    squared_span = np.square(np.ptp(rescaled_profiles))
    collapse_error = np.mean(variance) / squared_span

    return collapse_error

def lattice_search(func, func_args, param_pos, step_0, n_levels=3,
                   criterion='min'):
    """Systematically search a function's solution space for a minimum
    (maximum) value by sweeping a parameter of the function at greater and
    greater precision.

    Parameters
    ----------
    func : function
        The function on which to perform the search.
    func_args : list
        Arguments of the function. For the parameter to be swept, enter
        the range of parameter values (inclusive) as a tuple,
        e.g. [arg1, (0, 5), arg3, arg4]
    param_pos : int
        The index, in func_args, of the parameter to be swept.
    step_0 : float
        The coarsest step of the search. The first sweep will iterate through
        the range given in `func_args` at intervals of length `step_0`.
    n_levels : int
        Number of refinement sweeps after the initial sweep. Each successive
        sweep is at 10x precision, within +/- one step of the previous
        optimum.
    criterion : str
        The criterion to be fulfilled by the search. 'min' or 'max'.

    Returns
    -------
    opt_input : float
        The value of the swept parameter which optimizes the solution.
    opt_output : float
        The value of the optimal solution.

    References
    ----------
    Marshall et al. (2016) Front Physiol 7, 250.
    """
    if criterion not in ('min', 'max'):
        raise ValueError("criterion must be 'min' or 'max'.")
    select = np.argmin if criterion == 'min' else np.argmax
    func_args = list(func_args) # copy: the caller's list is not modified
    start, end = func_args[param_pos]

    def sweep(low, high, step):
        # values from low to high (inclusive, up to rounding) every step
        n_steps = int(np.floor((high - low) / step + 1e-9))
        values = low + step * np.arange(n_steps + 1)
        solutions = np.zeros(len(values))
        for i, value in enumerate(values):
            func_args[param_pos] = value
            solutions[i] = func(*func_args)
        best = select(solutions)
        return values[best], solutions[best]

    opt_input, opt_output = sweep(start, end, step_0)
    step_n = step_0
    for _ in range(n_levels):
        low, high = max(opt_input - step_n, start), min(opt_input + step_n, end)
        step_n = step_n / 10
        opt_input, opt_output = sweep(low, high, step_n)

    return opt_input, opt_output

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
    Beggs & Plenz (2003) J Neurosci 23(35), 11167-77.
    Priesemann et al. (2014) Front Syst Neurosci 8, 108.
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

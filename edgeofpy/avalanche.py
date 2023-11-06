"""
Avalanche criticality measures
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt

import powerlaw
import edgeofpy as eop

def detect_avalanches(data, s_freq, time=None, max_iei=.004, threshold=3,
                      thresh_type='both'):
    """Detect avalanche in coarse-grained electrophsiological
    recordings. Takes either raw or binarized data.

    Parameters
    ----------
    data : 2d array (dtype='float' or 'int')
        Raw (cleaned) signal, channel x sample. If dtype='int', the array is
        taken as the binarized event time series.
    s_freq : int
        Sampling frequency.
    time : 1d array (dtype='float')
        Vector with the time points for each sample. If None, avalanche start
        and end times will be returned in seconds since the start of data.
    max_iei : float, : .004
        Duration, in seconds, of the maximum inter-event interval within which
        suvccessive events are counted as belonging to the same avalanche.
    threshold : float
        Number of standard deviations.  Only signal excursions exceeding
        this threshold are considered events.
    thresh_type : str
        If 'above', an event is detected when signal is greater than tresh.
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
        Mean inter-event interval, in seconds.

    References
    ----------
    Benayoun et al. (2010) J Clin Neurophysiol 27(6) 458-64.
    Shriki et al. (2013) J Neurosci 33(16), 7079–90.
    Jannesari et al. (2020) Brain Structure and Function 225, 1169-83.
    """

    if data.dtype is not np.dtype(int):
        # Detect events as the locations of peaks above threshold
        data = eop.binarized_events(data, threshold=threshold,
                                               thresh_type=thresh_type,
                                               null_value=0)

    avalanches, events_one_chan, mean_iei, = _det_avls(data,
                                                       s_freq=s_freq,
                                                       time=time,
                                                       max_iei=max_iei)

    return avalanches, data, events_one_chan, mean_iei

def plot_pdf(data, binning='d_lin', n_bins=100, xlabel=None, ylabel=None):
    """Plot the probability density function of the data in double logarithmic
    axes.

    Parameters
    ----------
    data : array-like
        List of features, e.g. avalanche sizes.
    binning : str
        How to bin. Either 'd_lin' for discrete linear (all integers between
        min and max), 'c_lin' for continuous linear (linear progression of bins
        between min and max) or 'log' (logarithmic progression from log(min) to
        log(max)).
    n_bins : int
        Number of bins to use in the histogram. For 'c_lin' and 'log' options
        only.
    xlabel : str
        Label for x-axis. If None, defaults to 'x'.
    ylabel : str
        Label for y-axis. If None, defaults to 'P(`xlabel`)'

    Returns
    -------
    fig: matplotlib.figure.Figure
        Figure object.
    """
    if 'd_lin' == binning:
        bins = np.arange(min(data), max(data) + 1)
    elif 'c_lin' == binning:
        bins = np.linspace(min(data), max(data), num=n_bins)
    elif 'log' == binning:
        bins = np.logspace(np.log(min(data)), np.log(max(data)), num=n_bins,
                           base=np.e)
    pdf, bins = np.histogram(data, bins=bins, density=True)
    fig, ax = plt.subplots()
    ax.loglog(bins[1:], pdf)
    if xlabel is None:
        xlabel = 'x'
    ax.set_xlabel(xlabel)
    if ylabel is None:
        ylabel = f'P({xlabel})'
    ax.set_ylabel(ylabel)
    plt.show()

    return fig

def plot_third(sizes, durations, xlabel='Size', ylabel='Durations'):
    fig, ax = plt.subplots()
    ax.loglog(sizes, durations, 'b.')
    if xlabel is None:
        xlabel = 'x'
    ax.set_xlabel(xlabel)
    if ylabel is None:
        ylabel = f'P({xlabel})'
    ax.set_ylabel(ylabel)
    plt.show()

    return fig

def plot_cdf(data):
    """Plot the cumulative distirbution function of the data in double
    logarithmic axes.

    Parameters
    ----------
    data : array-like
        List of features, e.g. avalanche sizes.

    Returns
    -------
    fig: matplotlib.figure.Figure
        Figure object.
    """
    cdf = np.sort(data)
    p = np.arange(len(data)) / (len(data) - 1)

    fig, ax = plt.subplots()
    ax.loglog(p, cdf)
    ax.set_xlabel('x')
    ax.set_ylabel('P(X <= x)')
    plt.show()

    return fig

def fit_powerlaw(data, xmin='min', xmax=None, discrete=True):
    """Fit data to a power law and compare the fit to similar
    distributions.

    Parameters
    ----------
    data : array-like
        List of avalanche features, e.g. avalanche sizes.
    xmin : int or float or str
        The minimum value above which to fit the data. If None, xmin will be
        algorithmically determined by best fit. If 'min', the minimum value
        of `data` will be used.
    xmax : int or float, optional
        The maximum value above which to fit the data. If None, the maximum 
        value of `data` will be used.
    discrete: bool
        Whether the data is discrete (integers). Default is True

    Returns
    -------
    dict
        Dictionary with fit parameters, with keys/values:
        fit : instance of powerlaw.Fit
            Object containing attributes of the power law fit.
        power_law_exp : float
            Slope of the power law.
        xmin: int or float
            The minimum value of the fitted data.
        xmax : int or float
            The maximum value of the fitted data.
        power_law_sigma : float
            Standard error of the slope.
        truncated_power_law_exp : float
            Slope of the truncated power law.
        truncated_power_law_xmin : float
            Minimum value of the trauncated power law.
        truncated_power_law_xmax : float
            Maximum value of the trauncated power law.
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
            exponential fit.
        TvL : tuple of float
            The loglikelihood ratio R and probability p for truncated power law
            versus lognormal fit.
        best_fit : string
            Type of distribution yielding the best fit of the data can be 
            P (power law), T (truncated power law with exponential upper cutoff), 
            E (exponential), L (lognormal)
        best_fit_R_sum : float
            The summary loglikelihood value for `best_fit`, which is the basis
            for selecting `best_fit`. It is the max of the sums of loglikelihood 
            values R for each fit function.

    Notes
    -----
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

    fit = powerlaw.Fit(data, xmin=xmin, xmax=xmax, discrete=discrete, verbose=False)

    PvE = fit.distribution_compare('power_law', 'exponential',
                                   normalized_ratio=True)
    PvL = fit.distribution_compare('power_law', 'lognormal',
                                   normalized_ratio=True)
    PvT = fit.distribution_compare('power_law', 'truncated_power_law',
                                   normalized_ratio=True)

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

def fit_third_exponent(sizes, durations, discrete = True, method = 'pl'):
    """Estimate the third exponent from avalanche sizes and durations.

    Parameters
    ----------
    sizes : 1d array
        Sequence of avalanche sizes.
    durations : 1d array
        Corresponding sequence of avalanche durations.
    discrete: bool
        Whether the duration data is discrete (integers). Default is True.
        Only used when method is 'pl'
    method  : sting
        'lsf' for using nonlinear least-squares fitting
        'pl' for using powerlaw fit
    Returns
    -------
    third : float
        Estimated 'third' power-law exponent.
    """
    if method == 'lsf':
        def func(x, a, b):
            return a * x + b
        (third, offset), _ = scipy.optimize.curve_fit(func, np.log(sizes),
                                                      np.log(durations))

    elif method == 'pl':
        if not discrete:
            durations = np.round(durations,decimals=3)

        # get average for size in one bin
        dur_avg = []
        size_avg = []

        durations = np.array(durations)
        sizes = np.array(sizes)

        for d in np.unique(durations):
            dur_avg.append(d)
            size_avg.append(np.mean(sizes[durations==d]))

        # fit using powerlaw to get exponent
        fit = powerlaw.Fit(size_avg, discrete=True, verbose=False, xmin = min(size_avg))
        third = fit.power_law.alpha

    else:
        print('Chosen Method non valid. Method can be either lsf or pl. ' )
        third = np.NaN

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

def shape_collapse_error(avalanches, gamma, interp_n=1000):
    """Estimate the error of the shape collapse given a scaling factor
    gamma.

    Parameters
    ----------
    avalanches : list of dict
        Detected avalanches.
    gamma : float
        Scaling factor, corresponding to the critical exponent for the shape
        collapse.
    interp_n : int
        Number of points for interpolating avalanche profiles.

    Returns
    -------
    collapse_error : float
        The error of the shape collapse. A good collapse should minimize it.

    References
    ----------
    Marshall et al. (2016) Front Physiol 7, 250.
    Friedman et al. (2012) PRL 108, 208102.
    """
    # Calculate the average profile for each avalanche duration T
    T_set = sorted(set([x['dur_bin'] for x in avalanches]))
    profiles = np.zeros((len(T_set), interp_n))
    for i, T in enumerate(T_set):
        avls_T = [x for x in avalanches if x['dur_bin'] == T]
        avg_profile = np.mean(np.vstack([x['profile'] for x in avls_T]), axis=0)
        norm_avg_profile = ( np.asarray(avg_profile, dtype='float')
                            / float(avg_profile.max()) )
        profiles[i, :] = np.interp(np.linspace(0, 1, interp_n),
                                   np.arange(T), norm_avg_profile)

    # Rescale the profiles (y-axis) by T**gamma
    rescaling_vector = np.array(T_set)**gamma
    rescaling_vector = rescaling_vector[..., np.newaxis]
    rescaled_profiles = rescaling_vector * profiles

    # Calculate collapse error
    mean_profile = np.mean(rescaled_profiles, axis=0, keepdims=True)
    variance = np.sum(np.square(rescaled_profiles - mean_profile), axis=0)
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
        Number of sweeps. Each succesive sweep is at 10x precision.
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
    start, end = func_args[param_pos]
    sweep_range = np.arange(start, end + step_0, step_0)
    solutions = np.zeros(len(sweep_range))
    for i, j in enumerate(sweep_range):
        func_args[param_pos] = j
        solutions[i] = func(*func_args)
    if 'min' == criterion:
        opt_input = sweep_range[np.argmin(solutions)]
        opt_output = min(solutions)
    elif 'max' == criterion:
        opt_input = sweep_range[np.argmax(solutions)]
        opt_output = max(solutions)

    iteration = 1
    step_n = step_0
    while iteration <= n_levels:
        range_n = (max(opt_input - step_n, start),
                   min(opt_input + step_n, end))
        step_n = step_n / 10
        sweep_range = np.arange(range_n[0], range_n[1] + step_n, step_n)
        solutions = np.zeros(len(sweep_range))
        for i, j in enumerate(sweep_range):
            func_args[param_pos] = j
            solutions[i] = func(*func_args)
        if 'min' == criterion:
            opt_input = sweep_range[np.argmin(solutions)]
            opt_output = min(solutions)
        elif 'max' == criterion:
            opt_input = sweep_range[np.argmax(solutions)]
            opt_output = max(solutions)
        iteration += 1

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
    return np.abs( (alpha - 1) / (tau - 1) - (gamma + 1) )

def shew_kappa(data, exponent='fitted', n_bins=10, discrete=True):
    """Calculate Shew's kappa, an estimate of the closeness of the distribution
    of `data` to a power law distribution.

    In the original paper (Shew et al., 2009), the empirical distribution was
    compared to a power law with exponent -3/2. The present implementation
    allows the exponent (`alpha`) to be fitted from the data using the
    powerlaw package, by setting `alpha` to 'fitted'.

    Parameters
    ----------
    data : array-like
        List of avalanche features, e.g. avalanche sizes.
    exponent : str or float
        The (positive) exponent of the reference power law. If set to
        'fitted', then the exponent is estimating by fitting the data to a
        power law.
    n_bins : int
        DESCRIPTION. The default is 10.
    discrete : bool
        DESCRIPTION. The default is True.

    Returns
    -------
    kappa: float
        Kappa is close to 1 for power law distributed data.

    References
    ----------
    Shew et al. (2009) J Neurosci 29(49): 15595-15600.
    """
    # Create (complementary) cumulative distribution function (CDF)
    cdf_x = np.sort(data)

    if exponent == 'fitted':
        # Fit data to a power law and obtain the critical exponent
        fit = powerlaw.Fit(data, discrete=discrete, verbose=False)
        exponent = fit.power_law.alpha

    xmin = min(data)
    xmax = max(data)
    m = np.logspace(np.log(xmin), np.log(xmax), num=n_bins, base=np.e)

    def cdf_ref(x, xmin, exponent):
            return (x / xmin) ** (1 - exponent)

    # Find empirical values matching the bins in m
    empirical = np.zeros(n_bins)
    for i, one_bin in enumerate(m):
        empirical[i] = cdf_x[np.argmin(np.abs(cdf_x - one_bin))]

    kappa = 1 + np.mean(cdf_ref(m, xmin, exponent) - empirical)

    return kappa

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
        If True, dissimilarity values will be normalized with respect to maximum 
        possible dissimilarity. This is done automatically is `similarity` is 
        True.
    similarity : bool
        If True, similarity values (between 0 and 1) will be returned instead 
        of dissimilarity values.

    Returns
    -------
    dissimilarity : 2d matrix, dtype=int
        The symmetrical n_pattern x n_pattern matrix of the dissimilarity between
        each unique avalanche pattern in the repertoire, with 0 indicating
        identity (zeros should only be found along the diagonal).

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

# TODO: values yielded are much too small --> debug
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
        supercritical.

    References
    ----------
    Sorrentino et al. (2021) Sci Rep 11(1), 1-12.
    """
    profiles = [x['profile'] for x in avalanches if x['dur_bin'] > 1]
    N = len(profiles)
    per_avl = np.zeros(N)
    for i, p in enumerate(profiles):
        n_bin = len(p)
        per_avl[i] = (1/(n_bin - 1)  * np.prod(
            (p[1:] / p[:-1]) ** (1/(n_bin - 1)) ))
    sigma = 1 / N * np.prod(per_avl ** (1 / N))

    return sigma

def branching_ratio(events, time_bin_size=None, s_freq=None):
    """Calculate the branching ratio, which is the average number of events
    generated in the next time bin by a single event, evaluated for every bin
    containing at least one event.

    Parameters
    ----------
    events : 1darray (dtype=int)
        Signal with number of events occuring at every time step.
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
    Priesemann et al. (2014) Front Syst Neurosci 24, 108.
    """
    if time_bin_size is not None:
        bin_len = int(time_bin_size * s_freq)
        events = eop.time_bin_events(events, bin_len)
    # masked_invalid excludes instances where the first bin contains no spikes
    # (resulting in inf)
    br = np.ma.masked_invalid(events[1:] / events[:-1]).mean()

    return br

def f_exponential_offset(k, tau, A, O):
    """:math:`|A| e^{-k/\\tau} + O`"""
    return np.abs(A) * np.exp(-k/tau) + O * np.ones_like(k)

def susceptibility(events, test=False):
    """Estimate the susceptibility and local time fluctuation of the system.

    Parameters
    ----------
    events : 1d or 2d array (dtype=int)
        Channel x time binary signal where 1 is active and 0 is inactive.

    Returns
    -------
    chi : float
        Susceptibility, in the statistical physics sense.
    ltf : float
        Local time fluctuation, i.e. the suscpetibility `chi` normalized with
        respect to system size.

    References
    ----------
    Fosque et al. (2021) PRL 126, 098101.
    """
    #TODO: the implementation of chi may lead to catastrophic cancellation;
    # see Wikipedia article on Variance
    if len(events.shape) == 2:
        N, T = events.shape
        density = np.mean(events, axis=0)
    else:
        N = 1
        T = len(events)
        density = events

    if test:
        chi = N * np.var(density)
    else:
        chi = N * ( np.mean(np.square(density)) - np.square(np.mean(density)) )
    ltf = (1 / np.mean(density)) * np.sqrt(chi / N)

    return chi, ltf

def fano_factor(data):
    """Compute the Fano factor, which is the variance divided by the mean.
    The Fano factor is maximized for critical processes.

    Parameters
    ----------
    data : ndarray
        DESCRIPTION.

    Returns
    -------
    F : float
        The Fano factor.
    """
    F = np.var(data) / np.mean(data)
    return F

def _det_avls(events, s_freq, time, max_iei):
    """Utility function for detecting avalanches from event time series."""
    # Compute minimum interval, in samples
    max_iei_len = int(max_iei * s_freq)

    # If multichannel, collapse all channels into one
    multichannel = False
    if len(events.squeeze().shape) > 1:
        events_one_chan = np.sum(events, axis=0)
        multichannel = True
    else:
        events_one_chan = events.squeeze()

    # Obtain inter-event intervals
    idx_events = np.where(np.asarray(events_one_chan, dtype=bool))[0]
    if len(idx_events) < 2:
        return [], events_one_chan

    iei = np.diff(idx_events)
    mean_iei = np.mean(iei) / s_freq
    # Detect avalanches
    avalanches = []
    avl_start = idx_events[0]
    for i, j in enumerate(iei):
        if j > max_iei_len:
            avl_end = idx_events[i] + 1
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
            profile = eop.time_bin_events(profile_raw, max_iei_len)
            avl = {'start_time': start_time,
                   'end_time': end_time,
                   'size': size,
                   'dur_sec': dur_sec,
                   'dur_bin': dur_bin,
                   'n_chan': n_chan,
                   'profile': profile,
                   'pattern': pattern}
            avalanches.append(avl)
            avl_start = idx_events[i + 1]
    # if the record cuts off in the middle of an avalanche, we discard the
    # last avalanche

    return avalanches, events_one_chan, mean_iei

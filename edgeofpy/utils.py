"""Utility functions for edgeofpy"""

import warnings

import numpy as np
from scipy.optimize import curve_fit


def _gauss(x, amp, mu, sd):
    return amp * np.exp(-(x - mu) ** 2 / (2 * sd ** 2))


def gaussian_threshold(data, thresh_type='both', tol=2.0, fit_range=6.0,
                       bin_width=0.1, min_count=20, return_details=False):
    """Event threshold at which the signal amplitudes depart from a Gaussian.

    Each channel is z-scored and the amplitudes of all channels are pooled.
    A Gaussian is fitted by least squares to their density between
    -`fit_range` and `fit_range` standard deviations. The threshold is the
    smallest amplitude beyond which the density stays more than `tol` times
    above the Gaussian (Shriki et al., 2013, Fig. 1B).

    Parameters
    ----------
    data : 2d array or list of 2d arrays
        Raw (cleaned) signal, channel x sample. A list of arrays (e.g. the
        recordings of a dataset, which may differ in length and number of
        channels) is pooled and gives a single threshold.
    thresh_type : str
        'both' (default) compares the absolute amplitudes, as used by
        `binarized_events` with ``thresh_type='both'``; 'above' and
        'below' use the positive or negative tail only.
    tol : float
        Ratio of the observed density to the Gaussian density that counts
        as a departure. At the threshold, at most a fraction ``1 / tol`` of
        the samples is expected from the Gaussian part of the signal.
    fit_range : float
        Range of the Gaussian fit, in standard deviations.
    bin_width : float
        Width of the amplitude bins, in standard deviations. The threshold
        is a multiple of `bin_width`.
    min_count : int
        Bins with fewer samples (after pooling both tails, for 'both') are
        ignored, so that sparse extreme values do not decide the result.
    return_details : bool
        If True, also return a dictionary for checking the fit (see
        `plot_gaussian_threshold`).

    Returns
    -------
    threshold : float
        In standard deviations of each channel, as `binarized_events`
        expects. NaN, with a warning, if the density never departs from the
        Gaussian.
    details : dict
        Only if `return_details` is True, with keys:

        bin_centers, density : 1d arrays
            Pooled density of the z-scored amplitudes.
        gaussian : 1d array
            Fitted Gaussian density at `bin_centers`.
        gaussian_params : tuple
            Amplitude, mean and standard deviation of the fitted Gaussian.
        ratio_z, ratio : 1d arrays
            Absolute amplitude (lower bin edge; for 'below', minus the
            amplitude) and ratio of observed to Gaussian density.
        threshold, thresh_type, tol
            As above.

    Notes
    -----
    The rationale (Shriki et al., 2013) is that the Gaussian part of the
    amplitude distribution reflects noise-like fluctuations, and the excess
    in the tails the large, correlated deflections that form avalanches.
    Shriki et al. found the departure at about 2.7 SD in resting MEG and
    used 3 SD, as a margin. The departure is usually gradual, so the
    threshold depends on `tol`; report it with the threshold.

    To compare conditions or groups, compute one threshold from all
    recordings (pass them as a list) rather than one per recording:
    otherwise the conditions are analysed with different thresholds.

    The method assumes that the centre of the distribution is Gaussian. A
    warning is given if the departure starts within 1 SD of the mean, or if
    the data fall more than `tol` times below the Gaussian anywhere within
    `fit_range`.

    .. versionadded:: 0.3.0

    References
    ----------
    - Shriki et al. (2013) J Neurosci 33(16), 7079-90.
    """
    if thresh_type not in ('both', 'above', 'below'):
        raise ValueError("thresh_type must be 'both', 'above' or 'below'")
    arrays = list(data) if isinstance(data, (list, tuple)) else [data]
    n_half = int(round(10 / bin_width))
    edges = np.arange(-n_half, n_half + 1) * bin_width
    counts = np.zeros(2 * n_half)
    n_total = 0
    for arr in arrays:
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        if arr.ndim != 2:
            raise ValueError('data must be channel x sample arrays')
        for chan in arr:
            sd = chan.std()
            if not np.isfinite(sd) or sd == 0:
                continue
            counts += np.histogram((chan - chan.mean()) / sd, edges)[0]
            n_total += chan.size
    if n_total == 0:
        raise ValueError('no channel with non-zero variance')

    centers = 0.5 * (edges[1:] + edges[:-1])
    density = counts / (n_total * bin_width)
    fit = np.abs(centers) <= fit_range
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        params = curve_fit(_gauss, centers[fit], density[fit],
                           p0=(density.max(), 0, 1))[0]
    params[2] = abs(params[2])
    gaussian = _gauss(centers, *params)
    expected = gaussian * n_total * bin_width

    pos, neg = slice(n_half, None), slice(n_half - 1, None, -1)
    if thresh_type == 'both':
        obs, exp = counts[pos] + counts[neg], expected[pos] + expected[neg]
    elif thresh_type == 'above':
        obs, exp = counts[pos], expected[pos]
    else:
        obs, exp = counts[neg], expected[neg]
    ratio_z = edges[n_half:-1]
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = obs / exp
    ok = obs >= min_count
    above = ok & (ratio > tol)
    below = ok & ~above

    if np.any(ok & (ratio < 1 / tol) & (ratio_z < fit_range)):
        warnings.warn('the fitted Gaussian is more than tol times above the '
                      'data at some amplitudes: the distribution may not '
                      'have a Gaussian centre (e.g. it is bimodal); check '
                      'with plot_gaussian_threshold', stacklevel=2)
    threshold = np.nan
    if above.any():
        start = np.flatnonzero(below)[-1] + 1 if below.any() else 0
        after = np.flatnonzero(above[start:])
        if len(after):
            threshold = float(np.round(ratio_z[start + after[0]], 10))
    if np.isnan(threshold):
        warnings.warn('the amplitude distribution does not depart from the '
                      'fitted Gaussian; no threshold found', stacklevel=2)
    elif threshold <= 1:
        warnings.warn(f'the amplitude distribution departs from the fitted '
                      f'Gaussian at {threshold:g} SD: its centre is not '
                      f'Gaussian, and the threshold may not be meaningful',
                      stacklevel=2)
    if not return_details:
        return threshold
    details = dict(bin_centers=centers, density=density, gaussian=gaussian,
                   gaussian_params=tuple(float(v) for v in params),
                   ratio_z=ratio_z, ratio=ratio, threshold=threshold,
                   thresh_type=thresh_type, tol=tol)
    return threshold, details


def binarized_events(data, threshold='gaussian', thresh_type='both',
                     null_value=0):
    """Convert time series to binary where 1's are the locations of peaks 
    above or below threshold.
    
    Parameters
    ----------
    data : 2d array (dtype='float')
        Raw (cleaned) signal, channel x sample.
    threshold : float or 'gaussian'
        Number of standard deviations.  Only signal excursions exceeding
        this threshold are considered events. 'gaussian' (default) uses
        `gaussian_threshold` on `data`, with the same `thresh_type`.
    thresh_type : str
        If 'above', an event is detected when signal is greater than thresh.
        If 'below', an event is detected when signal is less than -thresh.
        If 'both', the signal above thresh and below -thresh is detected as an 
        event.
    null_value : int
        The value assigned to subthreshold values. Usually 0 or -1.

    Returns
    -------
    events : 2d array of same shape as data, dtype=int
        Time series of zeros with ones at the event peaks.

    Notes
    -----
    .. versionchanged:: 0.2.0
       The default `threshold` is 2 standard deviations (previously 3).

    .. versionchanged:: 0.3.0
       The default `threshold` is 'gaussian' (previously 2).
    """
    if isinstance(threshold, str):
        if threshold != 'gaussian':
            raise ValueError("threshold must be a number or 'gaussian'")
        threshold = gaussian_threshold(data, thresh_type=thresh_type)
        if np.isnan(threshold):
            raise ValueError("threshold='gaussian' found no departure from "
                             "a Gaussian; pass a threshold in standard "
                             "deviations")
    # z-normalizing is RAM-hungry; instead, compute thresholds per channel
    mean_per_chan = np.mean(data, axis=1, keepdims=True)
    std_per_chan = np.std(data, axis=1, keepdims=True)
    upper_thresh_per_chan = mean_per_chan + threshold * std_per_chan
    lower_thresh_per_chan = mean_per_chan - threshold * std_per_chan
    
    above_thresh = below_thresh = np.asarray(np.zeros(data.shape), dtype=bool)
    if thresh_type in ('above', 'both'):
        above_thresh = data > upper_thresh_per_chan
    if thresh_type in ('below', 'both'):
        below_thresh = data < lower_thresh_per_chan
    active = np.logical_or(above_thresh, below_thresh)
    
    events = np.ones(data.shape, dtype=int) * null_value
    for chan in range(data.shape[0]):
        start_end = _detect_start_end(active[chan, :])
        if start_end is not None:
            for i in start_end:
                idx_peak = np.argmax(np.abs(data[chan, i[0]:i[1]]))
                events[chan, i[0] + idx_peak] = 1
            
    return events

def time_bin_events(events, bin_len):
    """Compress a time series of events per time point into events per time 
    bin.

    Parameters
    ----------
    events : ndarray (dtype=int)
        Signal with number of events occurring at every time step. Can be 2d 
        (chan x time, with binary values) or 1d (time, with +int values).
    bin_len : int
        Length of time bin, in samples.

    Returns
    -------
    binned : 1d array (dtype=int)
        Time series of number of events per consecutive time bin.
    """
    events = np.asarray(events)
    if events.ndim == 2:
        events = np.sum(events, axis=0)
    if bin_len < 1:
        raise ValueError('bin_len must be >= 1.')
    remainder = len(events) % bin_len
    if remainder:
        # zero-pad the last, incomplete bin
        pad = np.zeros(bin_len - remainder, dtype=events.dtype)
        events = np.concatenate((events, pad))
    binned = np.sum(events.reshape(-1, bin_len), axis=1)

    return binned

_MAX_BIN_CELLS = 5e8  # units x bins above which bin_spikes refuses (4 GB)

def bin_spikes(spike_times, bin_dur, t_start=None, t_stop=None):
    """Count spikes in consecutive time bins.

    Parameters
    ----------
    spike_times : 1d array or sequence of 1d arrays
        Spike times, in seconds. A single array is taken as the pooled
        spikes of a population; a sequence of arrays (one per unit) gives
        one row of counts per unit.
    bin_dur : float or 'mean_isi'
        Bin duration, in seconds. If 'mean_isi', the mean interval between
        consecutive spikes of all units pooled, the usual choice of bin for
        avalanche analysis (Beggs and Plenz, 2003).
    t_start, t_stop : float, optional
        Start and end of the binned period, in seconds. By default, the
        first and last spike. Spikes outside the period are ignored.

    Returns
    -------
    counts : ndarray of int
        Number of spikes per bin: 1d (bins) for a single array, 2d
        (units x bins) for a sequence of arrays. Use the 1d form for large
        populations at fine bins: the 2d form takes units x bins memory,
        and arrays of more than 5e8 elements are refused.
    bin_dur : float
        Bin duration used, in seconds. The sampling frequency to pass to
        `detect_avalanches` is ``1 / bin_dur``.

    Notes
    -----
    .. versionadded:: 0.3.0

    References
    ----------
    Beggs JM, Plenz D (2003) J Neurosci 23(35), 11167-77.
    """
    if isinstance(spike_times, np.ndarray) and spike_times.dtype != object:
        pooled_only = spike_times.ndim == 1
        units = [spike_times] if pooled_only else list(spike_times)
    else:
        seq = list(spike_times)
        pooled_only = all(np.ndim(x) == 0 for x in seq)
        units = [np.array(seq)] if pooled_only else seq
    units = [np.asarray(u, float).ravel() for u in units]
    pooled = np.sort(np.concatenate(units)) if units else np.array([])
    if pooled.size < 2:
        raise ValueError('At least two spikes are needed.')
    t_start = pooled[0] if t_start is None else float(t_start)
    t_stop = pooled[-1] if t_stop is None else float(t_stop)
    if t_stop <= t_start:
        raise ValueError('t_stop must be later than t_start.')
    if isinstance(bin_dur, str):
        if bin_dur != 'mean_isi':
            raise ValueError("bin_dur must be a number or 'mean_isi'.")
        inside = pooled[(pooled >= t_start) & (pooled <= t_stop)]
        if inside.size < 2:
            raise ValueError('At least two spikes are needed between '
                             't_start and t_stop.')
        bin_dur = float(np.mean(np.diff(inside)))
        if bin_dur == 0:
            raise ValueError('All spikes are simultaneous; the mean '
                             'inter-spike interval is 0.')
    elif bin_dur <= 0:
        raise ValueError('bin_dur must be positive.')
    n_bins = max(1, int(np.ceil((t_stop - t_start) / bin_dur)))
    if not pooled_only and len(units) * n_bins > _MAX_BIN_CELLS:
        raise ValueError(f'{len(units)} units x {n_bins} bins is too large '
                         'an array; pass the pooled spike times (one array) '
                         'or use a longer bin.')
    edges = t_start + np.arange(n_bins + 1) * bin_dur
    counts = np.array([np.histogram(u, edges)[0] for u in units],
                      dtype=np.int64)
    return (counts[0] if pooled_only else counts), float(bin_dur)

def _sec_to_samples(duration, s_freq, name='duration'):
    """Convert a duration in seconds to a whole number of samples (>= 1).

    Parameters
    ----------
    duration : float
        Duration, in seconds.
    s_freq : float
        Sampling frequency.
    name : str
        Name of the parameter, used in the error message.

    Returns
    -------
    int
        The duration in samples, rounded to the nearest integer.
    """
    n_samples = int(round(duration * s_freq))
    if n_samples < 1:
        raise ValueError(f'{name} * s_freq must be at least 1 sample '
                         f'(got {duration * s_freq}).')
    return n_samples

def _detect_start_end(true_values):
    """From ndarray of bool values, return intervals of True values.

    Parameters
    ----------
    true_values : ndarray (dtype='bool')
        array with bool values

    Returns
    -------
    ndarray (dtype='int') or None
        N x 2 matrix with starting (inclusive) and ending (exclusive)
        indices of each run of True values. None if there are no True values.

    Notes
    -----
    Adapted from Wonambi: https://github.com/wonambi-python/wonambi
    """
    # pad with False on both sides so that runs touching either edge of the
    # array are detected
    padded = np.concatenate(([0], np.asarray(true_values, dtype=int), [0]))
    cross_threshold = np.diff(padded)

    event_starts = np.flatnonzero(cross_threshold == 1)
    event_ends = np.flatnonzero(cross_threshold == -1)

    if len(event_starts):
        return np.vstack((event_starts, event_ends)).T
    return None


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

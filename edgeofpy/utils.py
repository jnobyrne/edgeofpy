"""Utility functions for edgeofpy"""

import numpy as np

def binarized_events(data, threshold=3, thresh_type='both', null_value=0):
    """Convert time series to binary where 1's are the locations of peaks 
    above or below threshold.
    
    Parameters
    ----------
    data : 2d array (dtype='float')
        Raw (cleaned) signal, channel x sample.
    threshold : float
        Number of standard deviations.  Only signal excursions exceeding
        this threshold are considered events.
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
    """
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

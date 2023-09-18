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
        If 'above', an event is detected when signal is greater than tresh.
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
    
    events = np.ones(data.shape, dtype=np.int) * null_value
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
        Signal with number of events occuring at every time step. Can be 2d 
        (chan x time, with binary values) or 1d (time, with +int values).
    bin_len : int
        Length of time bin, in samples.

    Returns
    -------
    binned : 1d array (dtype=int)
        Time series of number of events per consecutive time bin.
    """
    if events.squeeze().ndim == 2:
        events = np.sum(events, axis=0)    
    if round(len(events) % bin_len) != 0:
        pad_len = bin_len - len(events) % bin_len
        #pad_len = ((len(events) // bin_len) + 1) * bin_len - len(events)
        padded = np.concatenate((events, np.zeros(pad_len)))
    else:
        padded = events
    binned = np.sum(padded.reshape(int(len(padded) / bin_len), bin_len), axis=1)
    
    return binned

def _detect_start_end(true_values):
    """From ndarray of bool values, return intervals of True values.

    Parameters
    ----------
    true_values : ndarray (dtype='bool')
        array with bool values

    Returns
    -------
    ndarray (dtype='int')
        N x 2 matrix with starting and ending times.
        
    Notes
    -----
    Function borrowed from Wonambi: https://github.com/wonambi-python/wonambi
    """
    neg = np.zeros((1), dtype='bool')
    int_values = np.asarray(np.concatenate((neg, true_values[:-1], neg)), 
                            dtype='int')
    # must discard last value to avoid axis out of bounds
    cross_threshold = np.diff(int_values)

    event_starts = np.where(cross_threshold == 1)[0]
    event_ends = np.where(cross_threshold == -1)[0]

    if len(event_starts):
        events = np.vstack((event_starts, event_ends)).T

    else:
        events = None

    return events

"""
Edge-of-synchrony measures
"""

import numpy as np
import scipy.signal

from edgeofpy.utils import _detect_start_end, _sec_to_samples
from edgeofpy.avalanche import _det_avls

def pcf(data):
    """Estimate the pair correlation function (PCF) in a network of
    oscillators, equivalent to the susceptibility in statistical physics.
    The PCF shows a sharp peak at a critical value of the coupling between
    oscillators, signaling the emergence of long-range correlations between
    oscillators.

    Parameters
    ----------
    data : 2d array
        The filtered input data, where the first dimension is the different
        oscillators, and the second dimension is time.

    Returns
    -------
    pcf : float
        The pair correlation function, a scalar value >= 0.
    orpa: float
        Absolute value of the order parameter (degree of synchronicity)
        averaged over time, being equal to 0 when the oscillators’ phases are
        uniformly distributed in [0,2π ) and 1 when they all have the same
        phase.
    orph_vector: 1d array (length = N_timepoints)
        Order parameter phase for every moment in time.
    orpa_vector: 1d array (length = N_timepoints)
        Absolute value of the order parameter (degree of synchronicity) for
        every moment in time.

    References
    ----------
    Yoon et al. (2015) Phys Rev E 91(3), 032814.
    """
    N_ch = data.shape[0]  # Nr of channels

    # calculate Phase of signal
    inst_phase = np.angle(scipy.signal.hilbert(data, axis=1))
    # get global synchronization order parameter z over time
    z_vector = np.mean(np.exp(1j * inst_phase), axis = 0)
    # get order phases
    orph_vector = np.arctan2(np.imag(z_vector), np.real(z_vector))
    #  r =|z| degree of synchronicity
    orpa_vector = np.abs(z_vector)

    # get PCF = variance of real part of order parameter
    # var(real(x)) == (mean(square(real(x))) - square(mean(real(x))))
    pcf = N_ch * np.var(np.real(z_vector))
    # time-averaged Order Parameter
    orpa = np.mean(orpa_vector)

    return pcf, orpa, orph_vector, orpa_vector

def pli(data):
    """Estimate the phase lag index (PLI) in a network of
    oscillators, using the phase angle difference from a hilbert transform.

    defined by: PLI = |E⟨sign[∆Phi(t)]⟩|,

    where Phi(t) corresponds to the time-series of phase angle differences.

    Parameters
    ----------
    data : 2d array
        The filtered input data, where the first dimension is the different
        oscillators, and the second dimension is time.

    Returns
    -------
    pli : float
        The average phase lag index for the given data, a scalar value >= 0.

    Notes
    -----
    Adaptions of the PLI are commonly used in functional connectivity (FC)
    analysis. The herein implemented PLI is not sufficient for such analysis.
    If FC needs to be evaluated, please refer to the mne-connectivity toolbox
    and the therein implemented weighted Phase lag index.

    References
    ----------
    Stam et al. (2007) Hum. Brain Mapp., vol. 28, no. 11,
    Lee et al. (2019) NeuroImage, vol. 188, pp. 228–238

    """
    N_ch = data.shape[0]  # Nr of channels
    pli = []
    # calculate Phase of signal with phase angles of hilbert
    inst_phase = np.angle(scipy.signal.hilbert(data, axis=1)) # (ch, time)

    for ch1 in range(N_ch):
        for ch2 in range(ch1):
            ph_diff = inst_phase[ch1] - inst_phase[ch2]
            # sign of the phase difference wrapped to (-pi, pi], i.e. of
            # sin(ph_diff)
            tmp_pli = np.abs(np.mean(np.sign(np.sin(ph_diff))))
            # This would also work with the cross-spectrum of hilbert
            #csd = y1 * y2.conjugate()
            #pli = np.abs(np.mean(np.sign(np.imag(csd))))
            pli.append(tmp_pli)

    pli_avg = np.mean(pli)

    return pli_avg

def ple(data, m , tau):
    """Estimate the phase lag entropy (PLE), defined as the Shannon entropy of
    the patterns of binarized phase lead/lag between pairs of signals. PLE is
    able to detect temporal changes in neural communication which cannot be
    observed using other phase synchronization methods.

    For each pair of signals, the phase difference is binarized (1 if the
    first signal leads, 0 if it lags), patterns of `m` successive values
    spaced by `tau` samples are extracted, and the entropy of the pattern
    distribution is normalized by its maximum, log(2**m).

    Parameters
    ----------
    data : 2d array
        The filtered input data, where the first dimension is the different
        oscillators, and the second dimension is time.
    m : integer
        Embedding dimension (length of patterns to extract).
    tau: integer
        Time delay / lag between successive pattern elements, in samples.

    Returns
    -------
    ple : float
        The average phase lag entropy for the given data, between 0 (fully
        regular lead/lag relationship) and 1 (maximally irregular).

    References
    ----------
    Lee et al. (2019) NeuroImage, vol. 188, pp. 228–238
    Lee et al. (2017) Hum. Brain Mapp. 38, 4980–4995.

    """
    N_ch = data.shape[0]  # Nr of channels
    ple = []
    # calculate Phase of signal with phase angles of hilbert
    inst_phase = np.angle(scipy.signal.hilbert(data, axis=1)) # (ch, time)

    for ch1 in range(N_ch):
        for ch2 in range(ch1):
            ph_diff = inst_phase[ch1] - inst_phase[ch2]

            # binarize the phase difference wrapped to (-pi, pi]:
            # 1 = lead, 0 = lag
            lead = np.asarray(np.sin(ph_diff) > 0, dtype=int)

            ple.append(_binary_pattern_entropy(lead, m, tau))

    ple_avg = np.mean(ple)

    return ple_avg

def _binary_pattern_entropy(x, m, tau):
    """Normalized Shannon entropy of the patterns of `m` successive values,
    spaced by `tau` samples, in binary sequence `x`."""
    n_patterns = len(x) - (m - 1) * tau
    if n_patterns < 1:
        raise ValueError('The signal is too short for the given m and tau.')
    # encode each pattern as an integer in [0, 2**m)
    codes = np.zeros(n_patterns, dtype=int)
    for k in range(m):
        codes = 2 * codes + x[k * tau : k * tau + n_patterns]
    p = np.bincount(codes, minlength=2 ** m) / n_patterns
    p = p[p > 0]

    return -np.sum(p * np.log(p)) / (m * np.log(2))

def complex_phase_relationship(data, s_freq=None, win_dur=None):
    """Compute all pairwise complex-valued phase relationships for a set of
    oscillators, time-resolved.

    Parameters
    ----------
    data : 2d array
        The filtered input data, where the first dimension is the different
        oscillators, and the second dimension is time.
    s_freq : int or None
        Sampling frequency, only required if using local time-averaging.
    win_dur : float or None
        If not None, a local time average will be applied to the signal, with
        a window of length win_dur, in seconds.

    Returns
    -------
    c_ij_t : 3d array, dtype=complex128
        N x N x T array of complex-valued pairwise phase relationships, where N
        is the number of oscillators and T is time. Will return NaN for entries
        where at least one of the signals in the pair has modulus zero at time
        t.

    Notes
    -----
    Time averaging (using win_dur) introduces spurious phase locking when there
    is a large pos-to-neg or neg-to-pos jump in the relative phase. This can
    often occur when the relative phase crosses between 0 and 2π.

    The output holds N * N * T complex values (16 bytes each), e.g. about
    6.5 GB for 64 channels and 100,000 samples.

    References
    ----------
    Kitzbichler et al. (2009) PLoS Comp Biol 5(3), e1000314.
    Shriki et al. (2013) J Neurosci 33(16), 7079–90.
    """
    # TODO: something is wrong with the phases: they are all coming out 0 or 2pi
    if win_dur is not None and s_freq is None:
        raise ValueError('Time averaging requires that s_freq be specified.')

    N, T = data.shape
    analytic = scipy.signal.hilbert(data, axis=1)
    c_ij_t = np.zeros((N, N, T), dtype=complex)

    for t in range(T):
        i = analytic[:, t]
        j = analytic[:, t].conj().T
        c_ij_t[:, :, t] = np.outer(i, j)

        if win_dur is None:
            norm = np.outer(np.abs(i), np.abs(j))
            c_ij_t[:, :, t] = c_ij_t[:, :, t] / norm

    if win_dur is not None:
        mod_sq = np.square(np.abs(analytic))
        flat = np.ones(int(win_dur * s_freq))
        H = flat / flat.sum()
        c_ij_t = scipy.signal.fftconvolve(c_ij_t, H[np.newaxis, np.newaxis, :],
                                          mode='same', axes=-1)
        mod_sq_smooth = scipy.signal.fftconvolve(mod_sq, H[np.newaxis, :],
                                                 mode='same', axes=-1)
        norm = np.zeros((N, N, T))

        for t in range(T):
            i = mod_sq_smooth[:, t]
            j = mod_sq_smooth[:, t].conj().T
            norm[:, :, t] = np.sqrt(np.outer(i, j))

        c_ij_t = c_ij_t / norm

    return c_ij_t

def phase_locking(c_ij_t, phase_thresh=np.pi/4, mod_sq_thresh=0.5):
    """Find time points where signals i and j are phase-locked.

    Parameters
    ----------
    c_ij_t : 3d array, dtype=complex128
        N x N x T array of complex-valued pairwise phase relationships, where N
        is the number of oscillators and T is time.
    phase_thresh : float, optional
        Maximum absolute phase difference, in radians, for i and j to be
        considered in phase. The default is pi/4.
    mod_sq_thresh : float, optional
        Minimum squared modulus of c_ij_t for the phase relationship to be
        considered significant. The default is 0.5.

    Returns
    -------
    phase_locked : 3d array, dtype='bool'
        Locations where i and j are phase-locked at time t.

    References
    ----------
    Kitzbichler et al. (2009) PLoS Comp Biol 5(3), e1000314.
    """
    phase_diff = np.angle(c_ij_t)
    in_phase = np.logical_and(-phase_thresh < phase_diff,
                           phase_diff < phase_thresh)
    # The significance threshold only varies away from 1 if a moving average
    # was taken in computing c_ij_t
    significance = np.square(np.abs(c_ij_t)) > mod_sq_thresh
    phase_locked = np.logical_and(in_phase, significance)

    return phase_locked

def phase_lock_interval(phase_locked):
    """Obtain the distribution of durations of phase-locking events.

    Parameters
    ----------
    phase_locked : 3d array, dtype=bool
        N x N x T boolean array where True values indicate that oscillators i
        and j are phase-locked at time t.

    Returns
    -------
    all_durations : 1d array
        A vector containing the duration, in samples, of each phase-locking
        event. Empty if there are no phase-locking events.

    References
    ----------
    Kitzbichler et al. (2009) PLoS Comp Biol 5(3), e1000314.
    """
    N, _, T = phase_locked.shape

    all_durations = []
    for i in range(N - 1):
        for j in range(i + 1, N):
            intervals = _detect_start_end(phase_locked[i, j, :])
            if intervals is not None:
                durations = intervals[:, 1] - intervals[:, 0]
                all_durations.append(durations)

    if not all_durations:
        return np.array([], dtype=int)
    all_durations = np.concatenate(all_durations)

    return all_durations

def global_lability(phase_locked, s_freq, delta_t_sec, norm=True):
    """Compute the global lability of synchronization (GLS), a time-resolved
    measure of the change in the extent of global synchronization in the
    system.

    Parameters
    ----------
    phase_locked : 3d array, dtype=bool
        N x N x T boolean array where True values indicate that oscillators i
        and j are phase-locked at time t.
    s_freq : float
        Sampling frequency.
    delta_t_sec : float
        Time delay parameter, in seconds. GLS will be calculated as the
        difference in the number of pairs of coupled oscillators between t and
        t + delta_t_sec, for every t.
    norm : bool
        If True, the GLS will be normalized with respect to the maximum
        possible number of synchronized pairs.

    Returns
    -------
    gls : 1d array
        Time series of the global lability of synchronization.

    References
    ----------
    Kitzbichler et al. (2009) PLoS Comp Biol 5(3), e1000314.
    Botcharova et al. (2012) Phys Rev E 86(5), 051920.
    """
    N, _, T = phase_locked.shape
    delta_t = _sec_to_samples(delta_t_sec, s_freq, name='delta_t_sec')
    if delta_t >= T:
        raise ValueError('delta_t_sec must be shorter than the data.')
    # number of phase-locked pairs at each time point, counting each pair
    # once and excluding the self-synchrony diagonal
    i_upper, j_upper = np.triu_indices(N, k=1)
    n_locked = np.sum(np.asarray(phase_locked[i_upper, j_upper, :],
                                 dtype=int), axis=0)
    gls = n_locked[delta_t:] - n_locked[:-delta_t]

    if norm:
        # divide by max number of synchronized pairs
        gls = gls / (N * (N - 1) / 2)

    gls = np.abs(gls) ** 2

    return gls

def detect_phase_avls_full(data, s_freq, time=None, n_phase_bins=8,
                           max_iei=0.024, staggered='both'):
    """Detect phase-locking avalanches, analogous to the neuronal avalanches
    of Beggs & Plenz (2003). This function detects avalanches for each phase
    bin, with a phase bin size set by n_phase_bin.

    Parameters
    ----------
    data : 2d array
        The filtered time series, channel x time.
    s_freq : int
        Sampling frequency.
    time : 1d array
        Vector with the time points for each sample. If None, avalanche start
        and end times will be returned in seconds since the start of data.
    n_phase_bins : int, optional
        Number of phase bins in which to divide the circle. The default is 8.
    max_iei : float, optional
        Duration, in seconds, of the maximum inter-event interval within which
        successive events are counted as belonging to the same avalanche.
    staggered : bool or str
        If True, phase bins will be created such that zero-phase occurs in the
        middle of a bin. If False, zero-phase will be the inclusive lower
        bound of the first bin. If 'both', avalanches for both staggered and
        un-staggered binning will be returned.

    Notes
    -----
    Phases are expressed in [0, 2*pi). Keys of `phase_dict` are formatted as
    'low-high', e.g. '5.89-0.39' for the staggered bin centred on zero.

    Returns
    -------
    phase_dict : dict of list of dict
        Dictionary where keys are the phase range and the values are lists of
        avalanches. Each avalanche is a dictionary; see detect_phase_avls().
    """
    phase_t = np.mod(np.angle(scipy.signal.hilbert(data, axis=1)), 2 * np.pi)

    bin_width = 2 * np.pi / n_phase_bins
    offsets = []
    if staggered is True or staggered == 'both':
        offsets.append(-bin_width / 2) # zero phase at the centre of a bin
    if staggered is False or staggered == 'both':
        offsets.append(0.) # zero phase at the lower bound of a bin
    if not offsets:
        raise ValueError("staggered must be True, False or 'both'.")

    phase_dict = {}
    for offset in offsets:
        edges = offset + bin_width * np.arange(n_phase_bins + 1)
        for low, high in zip(edges[:-1], edges[1:]):
            phase_range = (np.mod(low, 2 * np.pi), np.mod(high, 2 * np.pi))
            key = '-'.join(['{:.2f}'.format(x) for x in phase_range])
            phase_dict[key], _ = detect_phase_avls(phase_t, s_freq,
                                                   phase_range,
                                                   time, max_iei,
                                                   extract_phase=False)

    return phase_dict

def detect_phase_avls(data, s_freq, phase_range, time=None, max_iei=0.024,
                      extract_phase=True):
    """Detect phase-locking avalanches, analogous to the neuronal avalanches
    of Beggs & Plenz (2003).

    Parameters
    ----------
    data : 2d array
        The filtered time series, channel x time.
    s_freq : int
        Sampling frequency.
    phase_range : tuple of float
        The inclusive lower bound and exclusive higher bound of the phase bin,
        in radians. Bounds are wrapped to [0, 2*pi); if the lower bound is
        greater than the higher bound, the bin straddles zero phase.
    time : 1d array
        Vector with the time points for each sample. If None, avalanche start
        and end times will be returned in seconds since the start of data.
    max_iei : float, optional
        Duration, in seconds, of the maximum inter-event interval within which
        successive events are counted as belonging to the same avalanche.
    extract_phase : bool
        If True, the phase of the (pre-filtered) signal will be extracted as
        the angle of the Hilbert transform.

    Returns
    -------
    phase_avls : list of dict
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
    events_one_chan : 1d array
        Collapsed time series of phase-locking events.
    """
    N = data.shape[0]
    if extract_phase:
        phase_t = np.angle(scipy.signal.hilbert(data, axis=1))
    else:
        phase_t = data
    # work with phases and bin bounds in [0, 2*pi)
    phase_t = np.mod(phase_t, 2 * np.pi)
    low, high = np.mod(phase_range, 2 * np.pi)

    # Catch case where the phase range straddles zero
    if low < high:
        logical = np.logical_and
    else:
        logical = np.logical_or

    # binarize time series according to in-phase / out-of-phase
    in_phase = np.asarray(logical(low <= phase_t, phase_t < high), dtype=int)
    # use 1's to indicate where a signal enters the chosen phase bin
    locking_in = np.asarray(np.diff(in_phase, axis=1) == 1, dtype=int)
    # realign the time series
    locking_in = np.concatenate((np.zeros((N, 1), dtype=int), locking_in),
                                axis=1)
    phase_avls, events_one_chan = _det_avls(locking_in, s_freq, time, max_iei)

    return phase_avls, events_one_chan

"""
Edge-of-chaos measures
"""

import numpy as np
import scipy.fft
import scipy.signal
import scipy.stats

from edgeofpy.utils import _sec_to_samples

def chaos_pipeline(data, sigma=0.5, denoise=False, downsample='minmax',
                   denoise_r=None):
    """Simplified pipeline for the modified 0-1 chaos test emulating the
    implementation from Toker et al. (2022, PNAS). This test assumes
    that the signal is stationary and deterministic.

    Parameters
    ----------
    data : 1d array
        The (filtered) signal.
    sigma : float, optional
        Parameter controlling the level of noise used to suppress correlations.
        The default is 0.5.
    denoise : bool
        If True, denoising will be applied according to the method by Schreiber
        (1993).
    downsample : str or bool
        If 'minmax', signal will be downsampled by conserving only local minima
        and maxima.
    denoise_r : float or None
        Neighbourhood radius for the Schreiber denoising (see
        `schreiber_denoise`). If None, the default of `schreiber_denoise` is
        used. Only used if `denoise` is True.

    Returns
    -------
    K: float
        Median K-statistic.

    """
    if denoise:
        # Denoise data using Schreiber denoising algorithm
        data = schreiber_denoise(data, r=denoise_r)
    
    if downsample == 'minmax':
        # Downsample data by preserving local minima
        data = minmaxsig(data)
    
    # Check if signal is long enough, else return NaN
    if len(data) < 20:
        return np.nan

    # Normalize standard deviation of signal
    x = data * (0.5 / np.std(data))

    # Modified 0-1 chaos test
    K = z1_chaos_test(x, sigma=sigma)

    return K

def z1_chaos_test(x, sigma=0.5, rand_seed=0):
    """Modified 0-1 chaos test. For long time series, the resulting K-statistic
    converges to 0 for regular/periodic signals and to 1 for chaotic signals.
    For finite signals, the K-statistic estimates the degree of chaos.

    Parameters
    ----------
    x : 1d array
        The time series.
    sigma : float, optional
        Parameter controlling the level of noise used to suppress correlations.
        The default is 0.5.
    rand_seed : int, optional
        Seed for the (local) random number generator. The default is 0.

    Returns
    -------
    median_K : float
        Indicator of chaoticity. 0 is regular/stable, 1 is chaotic and values
        in between estimate the degree of chaoticity.

    References
    ----------
    Gottwald & Melbourne (2004) P Roy Soc A - Math Phy 460(2042), 603-11.
    Gottwald & Melbourne (2009) SIAM J Applied Dyn Sys 8(1), 129-45.
    Toker et al. (2022) PNAS 119(7), e2024455119.
    """
    rng = np.random.RandomState(rand_seed) # local: global state untouched
    x = np.asarray(x, dtype=float)
    N = len(x)
    j = np.arange(1, N + 1)
    n_lags = int(round(N / 10))
    t = np.arange(1, n_lags + 1)
    # Choose a coefficient c within the interval pi/5 to 3pi/5 to avoid
    # resonances. Do this 1000 times.
    c = np.pi / 5 + rng.random_sample(1000) * 3 * np.pi / 5
    k_corr = np.zeros(1000)

    for its in range(1000):
        # Create a 2-d system (p, q) driven by the data, as z = p + i*q
        z = np.cumsum(x * np.exp(1j * j * c[its]))

        # Calculate the (time-averaged) mean-square displacement for all lags
        # n in t, subtracting a correction term (Gottwald & Melbourne, 2009)
        # and adding a noise term scaled by sigma (Dawes & Freeland, 2008)
        M = (_mean_square_displacement(z, n_lags)
             - np.mean(x)**2 * (1 - np.cos(t * c[its])) / (1 - np.cos(c[its]))
             + sigma * (rng.random_sample(n_lags) - .5))

        k_corr[its], _ = scipy.stats.pearsonr(t, M)

    median_k = np.median(k_corr)

    return median_k

def _mean_square_displacement(z, max_lag):
    """Time-averaged mean-square displacement of complex trajectory z, i.e.
    mean_j |z[j + n] - z[j]|**2 for every lag n = 1, ..., max_lag.

    Uses |a - b|^2 = |a|^2 + |b|^2 - 2 Re(a conj(b)), with cumulative sums for
    the first two terms and an FFT-based autocorrelation for the last one.
    """
    N = len(z)
    lags = np.arange(1, max_lag + 1)
    csum = np.concatenate(([0.], np.cumsum(np.abs(z) ** 2)))
    # zero-padding to >= 2N makes the circular autocorrelation linear
    n_fft = scipy.fft.next_fast_len(2 * N)
    f = np.fft.fft(z, n_fft)
    acf = np.fft.ifft(f * np.conj(f)) # acf[n] = sum_j z[j + n] conj(z[j])
    cross = np.real(acf[lags])
    sum_sq = (csum[N] - csum[lags]) + csum[N - lags] - 2 * cross

    return sum_sq / (N - lags)

def lambda_max(data, s_freq, N, trial_len_sec=10):
    """Estimate lambda_max, whose closeness to 1 marks the distance to the
    edge of chaos. Assumes stationarity of the time series (best used on resting
    state data).

    Parameters
    ----------
    data : 2d array
        The input data, where the first dimension is the different sites, and
        the second dimension is observations through time. Can be (continuous)
        raw signal or (discrete) event signal.
    s_freq : int
        Sampling frequency.
    N : int
        Estimated number of units in the population.
    trial_len_sec : float
        Data will be segmented into trials of this length, in seconds.

    Returns
    -------
    lambda_max : float
        The closeness of this value to 1 marks the distance to the edge of
        chaos.
    norm_var : float
        The dispersion of the cross-covariances, i.e. the standard deviation
        of the off-diagonal elements of `c` divided by the mean of its
        diagonal (the autocovariances).
    c : 2d array
        Covariance matrix (across trials) of the activity summed within each
        trial, e.g. spike count covariance. This is proportional to the
        integrated time-lagged covariance.

    References
    ----------
    Dahmen et al. (2019) PNAS 116(26), 13051-60.
    """
    data = np.asarray(data, dtype=float)
    n, T = data.shape
    if n < 2:
        raise ValueError('At least two sites (rows of data) are required.')
    trial_len = _sec_to_samples(trial_len_sec, s_freq, name='trial_len_sec')
    n_trials = T // trial_len
    if n_trials < 2:
        raise ValueError('The data must contain at least two trials of '
                         'length trial_len_sec.')
    # reshape data into n x trials x t (a copy: the input is not modified)
    x = data[:, :n_trials * trial_len].reshape(n, n_trials, trial_len)

    c = _integrated_timelag_cov(x)
    c_mean = np.diag(c).mean()
    c_sd = c[~np.eye(n, dtype=bool)].std()

    norm_var = c_sd / c_mean
    lambda_max = np.sqrt(1 - np.sqrt(1 / (1 + N * np.square(norm_var))))
    
    return lambda_max, norm_var, c

def minmaxsig(x):
    """Downsample a signal by keeping only its local minima and maxima, in
    temporal order, as in the pipeline of Toker et al. (2022).

    Parameters
    ----------
    x : 1d array
        The time series.

    Returns
    -------
    1d array
        The successive local extrema of `x`. Flat extrema (plateaus) are not
        detected.

    References
    ----------
    Toker et al. (2022) PNAS 119(7), e2024455119.
    """
    maxs = scipy.signal.argrelextrema(x, np.greater)[0]
    mins = scipy.signal.argrelextrema(x, np.less)[0]
    minmax = np.concatenate((mins, maxs))
    minmax.sort(kind='mergesort')
    return x[minmax]

def schreiber_denoise(x, r=None, chunk_size=256):
    """Geometrical noise reduction for a time series, according to the
    simple noise reduction method introduced by Schreiber.

    The series is embedded in 3 dimensions with unit delay. The middle
    coordinate of each delay vector is replaced by the average of the middle
    coordinates of its neighbours, i.e. the other delay vectors within a
    radius `r` (maximum norm). The first and last samples are kept unchanged.

    Parameters
    ----------
    x : 1d array
        The time series.
    r : float or None
        Neighbourhood radius. If None, the standard deviation of `x` is used.
        Schreiber (1993) suggests a radius of a few times the noise amplitude.
    chunk_size : int
        Number of delay vectors processed at once. Lower values reduce memory
        use (about 8 * chunk_size * len(x) bytes) without changing the result.

    Returns
    -------
    xr : 1d array
        The array with the cleaned time series, of the same length as `x`.

    References
    ----------
    Schreiber T (1993) Phys Rev E 47, 2401-2404.

    Acknowledgements
    ----------------
    This code was adapted from Matlab code by Alexandros Leontitsis.
    """
    x = np.asarray(x, dtype=float).ravel()
    if r is None:
        r = np.std(x)

    # Make the phase space
    Y, T = takens_reconstruction(x, 3, 1)
    middle = Y[:, 1]

    Yr = np.empty(T)
    for start in range(0, T, chunk_size):
        idx = np.arange(start, min(start + chunk_size, T))
        # maximum-norm distance from each vector in the chunk to all vectors
        dist = np.abs(Y[idx, 0:1] - Y[:, 0])
        for k in range(1, Y.shape[1]):
            np.maximum(dist, np.abs(Y[idx, k:k+1] - Y[:, k]), out=dist)
        neighbours = dist <= r
        neighbours[np.arange(len(idx)), idx] = False # exclude the point itself
        n_neighbours = neighbours.sum(axis=1)
        # average middle coordinate of the neighbours (unchanged if none)
        Yr[idx] = np.where(n_neighbours > 0,
                           (neighbours @ middle) / np.maximum(n_neighbours, 1),
                           middle[idx])

    xr = np.concatenate(([x[0]], Yr, [x[-1]]))

    return xr

def takens_reconstruction(x, dim, tau):
    """Reconstruct the phase space of a time series x with the Method of Delays
    (MOD), in embedding dimension dim and for time delay tau.

    Parameters
    ----------
    x : 1d array
        The time series.
    dim : int
        The embedding dimension.
    tau : int
        The time delay, in samples.

    Returns
    -------
    Y : 2d array
        The trajectory matrix in the reconstructed phase space.
    T : int
        Length of the phase space.

    References
    ----------
    Takens F (1981) Lecture notes in Mathematics, 898.

    Acknowledgements
    ----------------
    This code was adapted from Matlab code by Alexandros Leontitsis.
    """
    x = np.asarray(x)
    # Total points in phase space
    T = len(x) - (dim - 1) * tau
    if T < 1:
        raise ValueError('The time series is too short for the given dim and '
                         'tau.')
    # Phase space reconstruction with MOD: row i is x[i + k * tau], k < dim
    Y = x[np.arange(T)[:, None] + np.arange(dim)[None, :] * tau]

    return Y, T

def _integrated_timelag_cov(x):
    """Utility function for obtaining the noise covariance.

    Computes the covariance, across trials, of the activity summed within
    each trial (e.g. spike counts). For trials of length L, this equals L
    times the time-lagged covariance integrated over lags -L < tau < L with
    triangular weights (L - |tau|) / L.

    Parameters
    ----------
    x : 3d array
        Data of shape n_sites x n_trials x trial_len.

    Returns
    -------
    c : 2d array
        n_sites x n_sites covariance matrix.
    """
    counts = x.sum(axis=2) # n_sites x n_trials
    c = np.cov(counts) # unbiased: normalized by n_trials - 1

    return c

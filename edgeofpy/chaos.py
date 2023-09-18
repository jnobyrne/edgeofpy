"""
Edge-of-chaos measures
"""

import numpy as np
import scipy.signal
import scipy.stats

def chaos_pipeline(data, sigma=0.5, denoise=False, downsample='minmax'):
    """Simplified pipeline for the modified 0-1 chaos test emulating the
    implementation from Toker et al. (2022, PNAS). This test assumes
    that the signal is stationarity and deterministic.

    Parameters
    ----------
    data : 1d array
        The (filtered) signal.
    sigma : float, optional
        Parameter controlling the level of noise used to suppress correlations.
        The default is 0.5.
    denoise : bool
        If True, denoising will be applied according to the method by Schreiber
        (2000).
    downsample : str or bool
        If 'minmax', signal will be downsampled by conserving only local minima
        and maxima.

    Returns
    -------
    K: float
        Median K-statistic.

    """
    if denoise:
        # Denoise data using Schreiber denoising algorithm
        data = schreiber_denoise(data)
    
    if downsample == 'minmax':
        # Downsample data by preserving local minima
        data = minmaxsig(data)
    
    # Check if signal is long enough, else return NaN
    if len(data) < 20:
        return np.nan

    # Normalize standard deviation of signal
    x = data * (0.5 / np.std(data))

    # Mdified 0-1 chaos test
    K = z1_chaos_test(x, sigma=sigma)

    return K

def chaos_pipeline_len(data, sigma=0.5, min_len_ratio=0.05, step_ratio=0.05):
    """Simplified pipeline for the modified 0-1 chaos test emulating the
    implementation from Toker et al. (2022, PNAS). This test assumes
    that the signal is stationarity and deterministic. This version of the
    function allows the setting of multiple data lengths, to observe the
    convergence behaviour.

    Parameters
    ----------
    data : 1d array
        The (filtered) signal.
    sigma : float, optional
        Parameter controlling the level of noise used to suppress correlations.
        The default is 0.5.
    min_len_ratio : float, optional
        The fraction of the full data length to use as the lower bound data
        length.
    step_ratio : float, optional
        The fraction of the full data length to use as an interval between
        successive data lengths.

    Returns
    -------
    K: 1d array
        Median K-statistic for each data length.
    dat_lens : 1d array
        The data lengths associated with each respective K-statistic.
    T : int
        Length of the downsampled data.
    """
    # Denoise data using Schreiber denoising algorithm
    data = schreiber_denoise(data)
    # Downsample data by preserving local minima
    data = minmaxsig(data)
    # Compute segment end times
    T = len(data)
    dat_lens = np.arange(int(T * min_len_ratio), T, int(T * step_ratio))

    K = np.zeros(len(dat_lens))
    for i, end in enumerate(dat_lens):
        x = data[:end]
        # normalize standard deviation of signal
        x = x * 0.5 / np.std(x)
        # modified 0-1 chaos test
        K[i] = z1_chaos_test(x, sigma=sigma)

    return K, dat_lens, T


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
        Seed for random number generator. The default is 0.

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
    np.random.seed(rand_seed)
    N = len(x)
    j = np.arange(1, N + 1)
    t = np.arange(1, int(round(N / 10)) + 1)
    M = np.zeros(int(round(N / 10)))
    # Choose a coefficient c within the interval pi/5 to 3pi/5 to avoid
    # resonances. Do this 1000 times.
    c = np.pi / 5 + np.random.random_sample(1000) * 3 * np.pi / 5
    k_corr = np.zeros(1000)

    for its in range(1000):
        # Create a 2-d system driven by the data
        p = np.cumsum(x * np.cos(j*c[its]))
        q = np.cumsum(x * np.sin(j*c[its]))

        for n in t:
            # Calculate the (time-averaged) mean-square displacement,
            # subtracting a correction term (Gottwald & Melbourne, 2009)
            # and adding a noise term scaled by sigma (Dawes & Freeland, 2008)
            M[n - 1]=(np.mean((p[n:N] - p[:N-n])**2 + (q[n:N]-q[:N-n])**2)
                     - np.mean(x)**2 * (1-np.cos(n*c[its])) / (1-np.cos(c[its]))
                     + sigma * (np.random.random()-.5))

        k_corr[its], _ = scipy.stats.pearsonr(t, M)
        median_k = np.median(k_corr)

    return median_k

def lambda_max(data, s_freq, N, trial_len_sec=10):
    """Estimate lambda_max, whose closeness to 1 marks the distance to the
    edge of chaos. Assumes stationary of the time series (best used on resting
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

    c : 2d array
        integrated time-lagged covariance matrix, which is an estimate of noise
        covariance

    References
    ----------
    Dahmen et al. (2019) PNAS 116(26), 13051-60.
    """
    n, T = data.shape
    trial_len = int(trial_len_sec * s_freq)
    # reshape data into n x trials x t
    end = -(T % trial_len)
    if end == 0:
        end = None
    x = data[:, :end].reshape(n, int(np.floor(T / trial_len)), trial_len)

    c = _integrated_timelag_cov(x)
    c_mean = np.diag(c).mean()
    c_sd = c.std()
    
    norm_var = c_sd / c_mean
    lambda_max = np.sqrt(1 - np.sqrt(1 / (1 + N * np.square(norm_var))))
    
    return lambda_max, norm_var, c

def minmaxsig(x):
    maxs = scipy.signal.argrelextrema(x, np.greater)[0]
    mins = scipy.signal.argrelextrema(x, np.less)[0]
    minmax = np.concatenate((mins, maxs))
    minmax.sort(kind='mergesort')
    return x[minmax]

def schreiber_denoise(x):
    """Geometrical noise reduction for a time series, according to the
    simple noise reduction method introduced by Schreiber.

    Parameters
    ----------
    x : 1d array
        The time series.

    Returns
    -------
    xr : 1d array
        The array with the cleaned time series.

    References
    ----------
    Schreiber T (1993) Phys Rev E 47, 2401-2404.

    Acknowledgements
    ----------------
    This code was adapted from Matlab code by Alexandros Leontitsis.
    """
    x = x.flatten()
    r = np.std(x)

    # Make the phase space
    Y, T = takens_reconstruction(x, 3, 1)

    Yr = np.zeros(T)
    for j in range(T):
        y = Y[j]
        lock = _rad_nearest(y, Y, T, r, np.inf)
        for idx in [i for i in range(len(lock)) if lock[i] == j]:
            lock.pop(idx)
        if len(lock) == 0:
            Yr[j] = y
        else:
            Ynearest = Y[lock]
            Yr[j] = Ynearest.mean()

    xr = np.concatenate((np.zeros(1), Yr, np.zeros(1)))

    return xr

def _rad_nearest(y, Y, T, r, p):
    """Locks the nearest neighbours of a reference point that lie within a
    radius in phase space.

    Parameters
    ----------
    y : 1d array
        The reference vector.
    Y : 2d array
        The phase space.
    T : int
        The length of the phase space.
    r : float
        The radius.
    p : int
        The order of the norm.

    Returns
    -------
    lock : list
        The points located.

    Acknowledgements
    ----------------
    This code was adapted from Matlab code by Alexandros Leontitsis.
    """
    lock = []
    for i in range(T):
        # Calculate the distance to the reference point.
        dist = np.linalg.norm(y - Y[i,:], p)
        # if it is less than r, count it
        if dist <= r:
            lock.append(i)

    return lock

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
    Takens F (1981) Lecure notes in Mathematics, 898.

    Acknowledgements
    ----------------
    This code was adapted from Matlab code by Alexandros Leontitsis.
    """
    # Total points in phase space
    T = len(x) - (dim - 1) * tau
    # Initialize the phase space
    Y = np.zeros((T, dim))
    # Phase space reconstruction with MOD
    for i in range(T):
        Y[i, :] = x[i + np.arange(dim) * tau]

    return Y, T

def _integrated_timelag_cov(x):
    """
    Utility function for obtaining the noise covariance.
    """
    n, _, T = x.shape

    # subtract (row-wise) sample mean to obtain normalized x
    sample_mean = x.mean(1)
    x -= sample_mean[:, None, :]

    # compute time-lagged covariances (n x n x T)
    c_tau = np.zeros((n, n, T))
    c = np.zeros((n, n))
    for tau in range(T):
        c_tau[:, :, tau] = np.dot(x[:, :, 0], x[:, :, tau].T) / (n - 1)
        # division by n-1 is for the unbiased sample covariance matrix
        # (the correction -1 is not very important for large n)
    # compute integral weighted by (T - tau) / T
    weights = np.linspace(-0.5, 0.5, T)
    c = np.dot(c_tau, weights)

    return c

# To do: implement stationarity test
# To do: implement gaussian_transform
def chaos_decision_tree(data, cutoff, stationarity_test=None,
                        gaussian_transform=None,
                        denoising_algorithm='schreiber',
                        surrogate_algorithm='aaft_cpp', n_surr=1000,
                        sigma=0.5):
    if stationarity_test is not None:
        pass
    if gaussian_transform is not None:
        pass
    if surrogate_algorithm == 'aaft_cpp':
        try:
            surr, params = surrogate(data, n_surr, 'AAFT', 1, 1)
        except:
            z_data = ( data - np.mean(data) ) / np.std(data) # z-score
            surr, params = surrogate(z_data, n_surr, 'AAFT', 1, 1)

def surrogate():
    pass

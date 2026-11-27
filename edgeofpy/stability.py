"""
Distance to instability: closeness of the effective connectivity of a
network to the edge of linear stability (Dahmen et al., 2019).
"""

import numpy as np

from edgeofpy.utils import _sec_to_samples


def lambda_max(data, s_freq, N, trial_len_sec=10, bias_correction=True):
    """Estimate lambda_max, the largest eigenvalue of the effective
    connectivity, whose closeness to 1 marks the distance to the edge of
    stability. Assumes stationarity of the time series (best used on resting
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
    bias_correction : bool
        If True (default), the expected sampling variance of the covariance
        estimates, (c_ii c_jj + c_ij**2) / (n_trials - 1), is subtracted from
        the variance of the cross-covariances before applying the formula.
        Without it, lambda_max is biased upward, strongly so with few trials
        or low values of lambda_max.

    Returns
    -------
    lambda_max : float
        The closeness of this value to 1 marks the distance to the edge of
        stability: in a linear network, activity diverges above 1.
    norm_var : float
        The dispersion of the cross-covariances, i.e. the standard deviation
        of the off-diagonal elements of `c` (corrected for sampling variance
        if `bias_correction`) divided by the mean of its diagonal (the
        autocovariances).
    c : 2d array
        Covariance matrix (across trials) of the activity summed within each
        trial, e.g. spike count covariance. This is proportional to the
        integrated time-lagged covariance.

    Notes
    -----
    The estimate relies on the theory of Dahmen et al. (2019) for large
    networks with random connectivity, in which lambda_max is the radius of
    the bulk of the eigenvalue spectrum of the connectivity. On simulated
    linear networks of 100 units (`edgeofpy.models.linear_network`), the
    bias-corrected estimate matches the true value well up to about 0.7, but
    underestimates values close to 1 (e.g. about 0.9 for a true value of
    0.98), partly because in a finite network the largest eigenvalue lies
    beyond the bulk. The estimates preserve the order of the true values, so
    they are best used to compare conditions rather than as absolute
    distances to 1.

    The edge of stability is a different transition from the edge of chaos
    detected by the 0-1 test (`chaos_pipeline`), although the two coincide
    in some network models.

    .. versionchanged:: 0.2.0
       Bias correction for the finite number of trials, enabled by default.

    .. versionchanged:: 0.3.0
       Moved from ``edgeofpy.chaos`` to ``edgeofpy.stability``.

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
    off_diag = ~np.eye(n, dtype=bool)
    c_mean = np.diag(c).mean()
    c_var = c[off_diag].var()
    if bias_correction:
        # expected sampling variance of the covariance estimates
        auto = np.diag(c)
        sampling_var = (np.outer(auto, auto)[off_diag]
                        + c[off_diag] ** 2) / (n_trials - 1)
        c_var = max(c_var - sampling_var.mean(), 0)

    norm_var = np.sqrt(c_var) / c_mean
    lambda_max = np.sqrt(1 - np.sqrt(1 / (1 + N * np.square(norm_var))))
    
    return lambda_max, norm_var, c


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

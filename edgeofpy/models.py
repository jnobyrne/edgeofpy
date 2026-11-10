"""
Model systems with known critical points, for validating analyses
"""

import numpy as np


def branching_process(sigma, n_avalanches, max_dur=10_000, gap=3,
                      rand_seed=0):
    """Simulate avalanches of a Galton-Watson branching process.

    Each avalanche starts from a single active unit, and every active unit
    activates a Poisson-distributed number of units, with mean `sigma`, in
    the next time step. The process is critical at ``sigma = 1``, where the
    avalanche sizes and durations follow power laws with the mean-field
    exponents tau = 3/2 (sizes) and alpha = 2 (durations), and the mean size
    grows with duration with exponent 2. It is subcritical below 1 and
    supercritical above 1.

    Parameters
    ----------
    sigma : float
        Branching parameter: mean number of descendants of an active unit.
    n_avalanches : int
        Number of avalanches to simulate.
    max_dur : int
        Avalanches still running after `max_dur` time steps are discarded,
        so fewer than `n_avalanches` may be returned (most of them for
        ``sigma > 1``).
    gap : int
        Number of silent time steps inserted before, between and after the
        avalanches. Must be larger than the maximum inter-event interval
        used to detect avalanches.
    rand_seed : int or numpy.random.Generator or None
        Seed or random number generator.

    Returns
    -------
    counts : 1d array of int
        Number of active units at each time step. Use
        ``counts[np.newaxis, :]`` as input to `detect_avalanches`, with a
        maximum inter-event interval of one sample.

    See Also
    --------
    branching_network : stationary activity with ongoing drive.

    References
    ----------
    - Harris (1963) The Theory of Branching Processes. Springer.
    - Zapperi et al. (1995) Phys Rev Lett 75(22), 4071-74.
    """
    if gap < 1:
        raise ValueError('gap must be at least 1.')
    rng = np.random.default_rng(rand_seed)
    silence = np.zeros(gap, dtype=int)
    pieces = [silence]
    for _ in range(n_avalanches):
        counts, active = [], 1
        while active and len(counts) < max_dur:
            counts.append(active)
            active = rng.poisson(sigma * active)
        if active: # still running after max_dur: discard
            continue
        pieces += [np.array(counts), silence]
    return np.concatenate(pieces)


def branching_network(n_units=1000, sigma=1.0, n_steps=100_000, n_out=10,
                      p_spont=1e-5, refractory=True, n_channels=None,
                      rand_seed=0):
    """Simulate a stochastic branching network with spontaneous activation.

    Each unit projects to `n_out` randomly chosen other units. At every
    time step, an active unit activates each of its targets with probability
    ``sigma / n_out``, and every unit also becomes active spontaneously with
    probability `p_spont`. Units are active for a single time step and, if
    `refractory` is True, cannot be activated in the step that follows.
    The network is critical near ``sigma = 1``: activity propagates in
    avalanches whose sizes and durations follow power laws (up to the size of
    the network), whereas it dies out quickly below 1 and becomes
    self-sustained above 1. Unlike `branching_process`, the activity is
    stationary for any `sigma`, which allows a sweep through the critical
    point.

    Parameters
    ----------
    n_units : int
        Number of units.
    sigma : float
        Branching parameter: expected number of units activated by one
        active unit (when the targets are not already active or refractory).
    n_steps : int
        Number of time steps.
    n_out : int
        Number of targets of each unit.
    p_spont : float
        Probability of spontaneous activation per unit and time step. Small
        values separate avalanches in time; if the drive is too strong,
        avalanches merge and the exponents are biased.
    refractory : bool
        If True, a unit cannot be activated in the time step after it was
        active.
    n_channels : int or None
        If given, units are pooled into `n_channels` groups (like electrodes
        recording several neurons), and the number of active units of each
        group is returned. If None, the activity of every unit is returned.
    rand_seed : int or numpy.random.Generator or None
        Seed or random number generator.

    Returns
    -------
    activity : 2d array of int
        Number of active units, channel x time step (unit x time step if
        `n_channels` is None). Can be given directly to `detect_avalanches`
        or `branching_ratio` (after summing over channels).

    References
    ----------
    - Haldeman & Beggs (2005) Phys Rev Lett 94(5), 058101.
    - Kinouchi & Copelli (2006) Nat Phys 2(5), 348-51.
    """
    if not 1 <= n_out < n_units:
        raise ValueError('n_out must be between 1 and n_units - 1.')
    if n_channels is None:
        n_channels = n_units
    rng = np.random.default_rng(rand_seed)
    # targets of each unit: n_out distinct units other than itself
    targets = np.empty((n_units, n_out), dtype=np.int64)
    for i in range(n_units):
        others = rng.choice(n_units - 1, size=n_out, replace=False)
        targets[i] = others + (others >= i)
    p_trans = sigma / n_out
    channel = np.arange(n_units) % n_channels
    dtype = np.int8 if -(-n_units // n_channels) <= 127 else np.int32
    activity = np.zeros((n_channels, n_steps), dtype=dtype)
    active = np.zeros(0, dtype=np.int64)
    for t in range(n_steps):
        new = np.zeros(n_units, dtype=bool)
        if active.size:
            cand = targets[active].ravel()
            new[cand[rng.random(cand.size) < p_trans]] = True
        n_spont = rng.binomial(n_units, p_spont)
        if n_spont:
            new[rng.integers(n_units, size=n_spont)] = True
        if refractory and active.size:
            new[active] = False
        active = np.flatnonzero(new)
        activity[:, t] = np.bincount(channel[active], minlength=n_channels)
    return activity


def kuramoto(n_osc=64, coupling=0.0, duration=20.0, s_freq=250,
             freq_mean=10.0, freq_std=1.0, noise=0.0, transient=5.0,
             rand_seed=0):
    """Simulate a network of all-to-all coupled Kuramoto oscillators.

    The phase of each oscillator evolves as

    .. math:: d\\theta_i = \\left( \\omega_i + \\frac{K}{N} \\sum_j
              \\sin(\\theta_j - \\theta_i) \\right) dt
              + \\sqrt{2 D \\, dt} \\, \\xi_i,

    with natural frequencies :math:`\\omega_i` drawn from a Gaussian
    distribution, coupling strength K and phase noise of intensity D. In
    the limit of many oscillators without noise, synchronization emerges at
    the critical coupling :math:`K_c = 2 / (\\pi g(0))`, where g is the
    distribution of :math:`\\omega_i`; for a Gaussian with standard deviation
    :math:`\\sigma_\\omega` (in rad/s), :math:`K_c = \\sqrt{8 / \\pi} \\,
    \\sigma_\\omega \\approx 1.6 \\, \\sigma_\\omega`, i.e. about 10 rad/s for
    the default `freq_std` of 1 Hz. Noise and a finite number of
    oscillators shift and smooth the transition.

    Parameters
    ----------
    n_osc : int
        Number of oscillators.
    coupling : float
        Coupling strength K, in rad/s.
    duration : float
        Duration of the returned signals, in seconds.
    s_freq : float
        Sampling frequency, in Hz; the equations are integrated with time
        step ``1 / s_freq`` (Euler-Maruyama).
    freq_mean : float
        Mean natural frequency, in Hz.
    freq_std : float
        Standard deviation of the natural frequencies, in Hz.
    noise : float
        Noise intensity D, in rad^2/s.
    transient : float
        Duration, in seconds, simulated and discarded before the returned
        signals, so that they start from the stationary state.
    rand_seed : int or numpy.random.Generator or None
        Seed or random number generator.

    Returns
    -------
    signals : 2d array
        Oscillator signals, cos(theta), oscillator x sample. These can be
        used like band-pass filtered M/EEG signals.
    phases : 2d array
        Phases theta, wrapped to [-pi, pi), oscillator x sample.

    References
    ----------
    - Kuramoto (1984) Chemical Oscillations, Waves, and Turbulence.
      Springer.
    - Acebrón et al. (2005) Rev Mod Phys 77(1), 137-85.
    """
    rng = np.random.default_rng(rand_seed)
    dt = 1 / s_freq
    n_keep = int(round(duration * s_freq))
    n_skip = int(round(transient * s_freq))
    omega = 2 * np.pi * rng.normal(freq_mean, freq_std, n_osc)
    theta = rng.uniform(-np.pi, np.pi, n_osc)
    noise_amp = np.sqrt(2 * noise * dt)
    phases = np.empty((n_osc, n_keep))
    for t in range(n_skip + n_keep):
        # mean field r exp(i psi): K/N sum_j sin(theta_j - theta_i)
        # = K r sin(psi - theta_i)
        z = np.mean(np.exp(1j * theta))
        drift = omega + coupling * np.abs(z) * np.sin(np.angle(z) - theta)
        theta = theta + drift * dt
        if noise:
            theta = theta + noise_amp * rng.standard_normal(n_osc)
        if t >= n_skip:
            phases[:, t - n_skip] = theta
    phases = np.mod(phases + np.pi, 2 * np.pi) - np.pi
    return np.cos(phases), phases


def logistic_map(n, r=3.9, x0=0.4, transient=100):
    """Iterate the logistic map x <- r x (1 - x).

    The map is periodic for most values of r below about 3.57 (after a
    period-doubling cascade) and chaotic for most values above, with
    periodic windows (e.g. near r = 3.83).

    Parameters
    ----------
    n : int
        Number of values to return.
    r : float
        Growth parameter, between 0 and 4.
    x0 : float
        Initial value, between 0 and 1.
    transient : int
        Number of initial iterations discarded.

    Returns
    -------
    x : 1d array
        The `n` values following the transient.

    References
    ----------
    May (1976) Nature 261, 459-67.
    """
    x = np.empty(n + transient)
    x[0] = x0
    for i in range(1, n + transient):
        x[i] = r * x[i - 1] * (1 - x[i - 1])
    return x[transient:]


def linear_network(n_units=100, spectral_radius=0.9, n_steps=100_000,
                   noise_std=1.0, rand_seed=0):
    """Simulate a linear stochastic network with a given spectral radius.

    The activity evolves as x(t+1) = W x(t) + noise, where W is a random
    Gaussian connectivity matrix rescaled so that its largest absolute
    eigenvalue equals `spectral_radius`. This is the quantity lambda_max
    estimated by `lambda_max`: the network is stable below 1, and its
    fluctuations diverge as the spectral radius approaches 1 (the edge of
    instability, the linear analogue of the edge of chaos).

    Parameters
    ----------
    n_units : int
        Number of units.
    spectral_radius : float
        Largest absolute eigenvalue of W; must be below 1 for stationary
        activity.
    n_steps : int
        Number of time steps.
    noise_std : float
        Standard deviation of the independent Gaussian noise of each unit.
    rand_seed : int or numpy.random.Generator or None
        Seed or random number generator.

    Returns
    -------
    activity : 2d array
        Unit x time step.
    W : 2d array
        Connectivity matrix.

    References
    ----------
    Dahmen et al. (2019) PNAS 116(26), 13051-60.
    """
    if not 0 <= spectral_radius < 1:
        raise ValueError('spectral_radius must be in [0, 1).')
    rng = np.random.default_rng(rand_seed)
    W = rng.normal(0, 1 / np.sqrt(n_units), (n_units, n_units))
    W *= spectral_radius / np.max(np.abs(np.linalg.eigvals(W)))
    noise = rng.normal(0, noise_std, (n_steps, n_units))
    x = np.zeros((n_steps, n_units))
    for t in range(1, n_steps):
        x[t] = W @ x[t - 1] + noise[t]
    return x.T, W


def ising_samples(h, J, n_samples, beta=1.0, n_chains=100, burn_in=500,
                  thin=10, rand_seed=0):
    """Draw samples from an Ising (pairwise maximum-entropy) model.

    The model assigns each configuration s of +1/-1 spins the probability
    P(s) ~ exp(-beta E(s)), with energy E(s) = -s.J.s / 2 - h.s, as in
    `ising_energy` and `fit_pmem`. Samples are drawn by Gibbs sampling
    (heat-bath updates of one spin at a time), running `n_chains`
    independent chains in parallel.

    Parameters
    ----------
    h : 1d array
        Field of each spin.
    J : 2d array
        Symmetric coupling matrix; its diagonal is ignored.
    n_samples : int
        Number of samples to return.
    beta : float
        Inverse temperature.
    n_chains : int
        Number of independent Markov chains.
    burn_in : int
        Number of sweeps (one update of every spin) discarded at the start
        of each chain.
    thin : int
        Number of sweeps between successive samples of a chain.
    rand_seed : int or numpy.random.Generator or None
        Seed or random number generator.

    Returns
    -------
    samples : 2d array
        Spin configurations coded as -1/+1, spin x sample, ready for
        `fit_pmem`.

    References
    ----------
    Geman & Geman (1984) IEEE Trans Pattern Anal Mach Intell 6(6), 721-41.
    """
    h = np.asarray(h, dtype=float)
    J = np.array(J, dtype=float)
    np.fill_diagonal(J, 0)
    if not np.allclose(J, J.T):
        raise ValueError('J must be symmetric.')
    n = h.size
    rng = np.random.default_rng(rand_seed)
    n_chains = min(n_chains, n_samples)
    per_chain = -(-n_samples // n_chains) # ceil
    s = rng.choice([-1.0, 1.0], size=(n_chains, n))
    field = s @ J + h # local field of each spin, chain x spin
    out = np.empty((per_chain, n_chains, n))
    n_sweeps = burn_in + per_chain * thin
    for sweep in range(n_sweeps):
        for i in rng.permutation(n):
            # heat bath: P(s_i = +1 | other spins) = 1 / (1 + exp(-2 beta f_i))
            p_up = 1 / (1 + np.exp(-2 * beta * field[:, i]))
            new = np.where(rng.random(n_chains) < p_up, 1.0, -1.0)
            flip = new != s[:, i]
            if flip.any():
                s[flip, i] = new[flip]
                field[flip] += 2 * new[flip, np.newaxis] * J[i]
        done = sweep + 1 - burn_in
        if done > 0 and done % thin == 0:
            out[done // thin - 1] = s
    return out.reshape(-1, n)[:n_samples].T

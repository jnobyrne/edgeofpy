"""Phase diagram analysis
"""

import json
import multiprocessing as mp
import warnings
import numpy as np
import matplotlib.pyplot as plt


def fit_pmem(data, iter_max : int = 5000000, error_max=1e-8, eta=.05, 
             method='donders', init_zero=False, verbose=False):
    """Fit binary multichannel data to the pairwise maximum entropy model by
    pseudo-likelihood maximization.    

    Parameters
    ----------
    data : 2d array (dtype='float')
        Multi-channel binary data coded as -1/+1, of shape `n_node` x 
        `dat_len`.
    iter_max : int
        Maximum number of iterations of the gradient ascent. A
        RuntimeWarning is issued if the fit has not converged by then.
    error_max : float
        Maximum tolerated error, exclusive. For 'ezaki', the error is the
        norm of the gradient divided by n_node * (n_node + 1); for 'donders',
        it is the sum of squared gradient elements.
    eta : float
        Learning rate. With 'donders', it is halved whenever the error
        increases.
    method : str
        PLM algorithm. 'donders' or 'ezaki'
    init_zero : bool
        If True, initialize h and J as zeros; if False, initialize as 
        data mean and data covariance matrix, respectively.
    verbose : bool
        If True, iteration and error will be printed at every step.

    Returns
    -------
    h : 1d array (dtype='float')
        The estimated elementwise intrinsic field, i.e. the intrinsic 
        likelihood that S_i = 1.
    J : 2d array (dtype='float')
        The estimated n_node x n_node interaction matrix.

    References
    ----------
    Ezaki et al. (2020) Commun Biol 3, 52.
    Tkačik et al. (2015) PNAS 112(37), 11508–13.
    
    Acknowledgements
    ----------------
    This function was partly adapted from Matlab code by Takahiro Ezaki: 
    https://github.com/tkEzaki/phase_diagram_analysis.
    The `'donders'` method was adapted from Matlab code by Hilbert 
    Kappen.
    """
    data = np.asarray(data, dtype=float)
    if not np.all(np.isin(data, (-1, 1))):
        raise ValueError('data must be coded as -1/+1 (e.g. use '
                         'binarized_events with null_value=-1).')
    if method not in ('ezaki', 'donders'):
        raise ValueError("method must be 'ezaki' or 'donders'.")
    n_node, dat_len = data.shape
    dat_mean = data.mean(1, keepdims=True) # n_node x 1
    dat_corr = data @ data.T / dat_len
    
    if init_zero:
        h = np.zeros((n_node, 1))
        J = np.zeros((n_node, n_node))
    else:
        h = dat_mean.copy()
        J = dat_corr - dat_mean @ dat_mean.T
        np.fill_diagonal(J, 0)
    
    converged = False
    if method == 'ezaki':
        for t in range(iter_max):
            h, J, error = _iter_fit_pmem(data, h, J, n_node, dat_len, dat_mean, 
                                    dat_corr, eta)
            if verbose:
                print(f'Iteration {t}: error = {error}')
            if error < error_max:
                converged = True
                break
    else: # 'donders'
        error_prev = np.inf
        for t in range(iter_max):
            sigma = np.tanh(J @ data + h) # n_node x dat_len
            C = sigma @ data.T / dat_len
            C = 0.5 * (C + C.T)
            dJ = dat_corr - C
            np.fill_diagonal(dJ, 0)
            dh = dat_mean - sigma.mean(axis=1, keepdims=True)
            J = J + eta * dJ
            h = h + eta * dh
            error = np.sum(dJ**2) + np.sum(dh**2)
            if verbose:
                print(f'Iteration {t}: error = {error}, eta = {eta}')
            if error < error_max:
                converged = True
                break
            if error > error_prev:
                eta = eta / 2
            error_prev = error

    if converged:
        if verbose:
            print(f'Converged at iteration {t}')
    else:
        warnings.warn(f'fit_pmem did not converge in {iter_max} iterations '
                      f'(error = {error}).', RuntimeWarning, stacklevel=2)

    return h.ravel(), J

def _iter_fit_pmem(data, h, J, n_node, dat_len, dat_mean, dat_corr, eta):
    """Iterated part of pairwise maximum entropy model fitting"""
    prediction = np.tanh(J @ data + h)
    Ldh = -dat_mean + np.mean(prediction, 1, keepdims=True)
    h = h - Ldh * eta
    
    corr_mat = data @ prediction.T / dat_len # hybrid
    LdJ = (-dat_corr + 0.5 * corr_mat + 0.5 * corr_mat.T)
    np.fill_diagonal(LdJ, 0) # eliminate diagonal elements (J_{ii} = 0)
    J = J - LdJ * eta
    
    error = ( np.sqrt( np.linalg.norm(LdJ, 'fro')**2 + np.linalg.norm(Ldh)**2 ) 
                / n_node / (n_node + 1) )

    return h, J, error

def fit_beta(data, h, J, iter_max : int = 5000000, error_max=0.00000001, 
             eta=.005, verbose=False):
    """Fine-tune a PMEM fit using temperature.  

    Parameters
    ----------
    data : 2d array (dtype='float')
        Multi-channel binary data coded as -1/+1, of shape `n_node` x 
        `dat_len`.
    h : 1d array (dtype='float')
        The elementwise intrinsic field, i.e. the intrinsic 
        likelihood that S_i = 1.
    J : 2d array (dtype='float')
        The n_node x n_node interaction matrix.
    iter_max : int
        Maximum number of iterations of the gradient ascent. A
        RuntimeWarning is issued if the fit has not converged by then.
    error_max : float
        Maximum tolerated error, exclusive.
    eta : float
        Learning rate.
    verbose : bool
        If True, the empirical energy, and the iteration, beta and error at
        every step, will be printed.

    Returns
    -------
    beta : float
        The estimated inverse temperature of the fine-tuned model.

    References
    ----------
    Ruffini et al. (2023) PLOS Comp Biol 19(2), e1010811.
    """
    data = np.asarray(data, dtype=float)
    n_node, dat_len = data.shape
    h = np.asarray(h, dtype=float).reshape(n_node, 1)
    # copy with no self-interaction (the caller's J is not modified)
    J = np.array(J, dtype=float)
    np.fill_diagonal(J, 0)
    beta = 1.0

    # mean empirical energy: -0.5 * s.J.s - h.s, averaged over samples
    H_mean = np.mean(-0.5 * np.sum(data * (J @ data), axis=0)
                     - h.ravel() @ data)
    if verbose:
        print(f'empirical H: {H_mean}')

    converged = False
    for t in range(iter_max):
        beta, error = _iter_fit_beta(data, beta, h, J, H_mean, n_node, 
                                  dat_len, eta)
        if verbose:
            print(f'Iteration {t}: beta = {beta}, error = {error}')
        if error < error_max:
            converged = True
            break

    if converged:
        if verbose:
            print(f'Converged at iteration {t}')
    else:
        warnings.warn(f'fit_beta did not converge in {iter_max} iterations '
                      f'(error = {error}).', RuntimeWarning, stacklevel=2)
    
    return beta

def _iter_fit_beta(data, beta, h, J, H, n_node, dat_len, eta):
    """Iterated part of the temperature fitting"""
    # get model prediction
    prediction = np.tanh( (J * beta) @ data + h * beta)

    # get 1- and 2-point correlations
    one_pt_corr = np.sum(prediction, 1, keepdims=True) / dat_len
    two_pt_corr = data @ prediction.T / dat_len # hybrid

    # make symmetrical
    two_pt_corr = 0.5 * (two_pt_corr + two_pt_corr.T)

    # eliminate diagonal elements (J has a zero diagonal; see fit_beta)
    np.fill_diagonal(two_pt_corr, 0)

    model_mean_H = -np.sum(h * one_pt_corr) - 0.5 * np.sum(J * two_pt_corr)
    Ldbeta = H - model_mean_H
    beta = beta - Ldbeta * eta

    error = np.abs(Ldbeta) / (n_node * (n_node + 1))

    return beta, error

def metropolis(h, J, beta=1.0, n_iter=1000, rand_seed=0, rand_init=False, 
               save_every=None, save_path=None, checkpoint_file=None):
    """Sample data from the estimated model using the Metropolis-Hastings 
    algorithm and compute statistics.

    Parameters
    ----------
    h : 1d array (dtype='float')
        The estimated elementwise intrinsic field, i.e. the intrinsic 
        likelihood that S_i = 1.
    J : 2d array (dtype='float')
        The estimated n_node x n_node interaction matrix.
    beta : float, optional
        The inverse temperature.
    n_iter : int
        Number of iterations.
    rand_seed : int
        Seed for the (local) random number generator. The global NumPy random
        state is not modified.
    rand_init : bool
        If True, the initial spin configuration `sigma` will be randomly
        generated using rand_seed. If False, `sigma` will be all ones.
    save_every : int, optional
        If not None, save a checkpoint every `save_every` samples (i.e. every
        `save_every` * n_node steps), as `{save_path}_{n_samples}.npz`.
    save_path : str, optional
        Path prefix to save the checkpoints.
    checkpoint_file : str, optional
        If a path is specified, the simulation will continue from this
        checkpoint, including the state of the random number generator. The
        `h`, `J`, `beta` and `n_iter` stored in the checkpoint are used, and
        the corresponding arguments are ignored.

    Returns
    -------
    m : float
        Magnetization order parameter.
    q : float
        Spin-glass order parameter.
    chi_u : float
        Uniform susceptibility.
    chi_sg : float
        Spin-glass susceptibility.
    C : float
        Specific heat.
    local_m : 1d array, dtype='float'
        Local mean.
    corr : 2d array, dtype='float'
        Correlation matrix.

    References
    ----------
    Ezaki et al. (2020) Commun Biol 3, 52.
    Fischer & Hertz (1991) "Spin glasses". Cambridge UP.
    
    Acknowledgements
    ----------------
    This function was adapted from Matlab code by Takahiro Ezaki: 
    https://github.com/tkEzaki/phase_diagram_analysis.
    """
    if save_every is not None and save_path is None:
        raise ValueError("Specify a path for saving checkpoints.")
    
    rng = np.random.RandomState(rand_seed) # local: global state untouched

    if checkpoint_file is None:
        h = np.asarray(h, dtype=float).ravel()
        n_node = len(h)
        local_m = np.zeros(n_node)
        corr = np.zeros((n_node, n_node))
        if rand_init:
            sigma = np.sign( rng.rand(n_node) - 0.5 )
        else:
            sigma = np.ones(n_node) # all nodes start "on"
        n_step = n_iter * n_node
        E = 0.
        E2 = 0.
        t_start = 0
    
        # First, run out the transient for n_step steps
        for t in range(n_step):
            n = rng.randint(low=0, high=n_node)
            dE = -2 * h[n] * sigma[n] - 2 * np.inner(J[:, n], sigma) * sigma[n]
            p = min(1, np.exp(dE * beta))
            if rng.random_sample() < p:
                sigma[n] = -sigma[n]

    else:
        checkpoint = np.load(checkpoint_file)
        if 'next_step' not in checkpoint:
            raise ValueError('This checkpoint was saved by an older, '
                             'incompatible version of edgeofpy.')
        h = checkpoint['h']
        J = checkpoint['J']
        beta = float(checkpoint['beta'])
        n_iter = int(checkpoint['n_iter'])
        sigma = checkpoint['sigma']
        local_m = checkpoint['local_m']
        corr = checkpoint['corr']
        E = float(checkpoint['E'])
        E2 = float(checkpoint['E2'])
        n_node = len(h)
        n_step = n_iter * n_node
        t_start = int(checkpoint['next_step'])
        _set_rng_state(rng, str(checkpoint['rng_state']))
  
    # Now, sample the steady-state every n_node steps
    for t in range(t_start, n_step):
        n = rng.randint(low=0, high=n_node)
        dE = -2 * h[n] * sigma[n] - 2 * np.inner(J[:, n], sigma) * sigma[n]
        p = min(1, np.exp(dE * beta))
        if rng.random_sample() < p:
            sigma[n] = -sigma[n]
            
        # Local mean and correlation matrix are updated every `n_node` steps.
        if t % n_node == 0:
            local_m = local_m + sigma
            corr = corr + np.outer(sigma, sigma)
            energy = -0.5 * sigma @ J @ sigma - h @ sigma
            E = E + energy
            E2 = E2 + energy**2

            # Save a checkpoint after every `save_every` samples
            n_samples = t // n_node + 1
            if save_every is not None and n_samples % save_every == 0:
                np.savez(f'{save_path}_{n_samples}.npz', h=h, J=J, beta=beta,
                         n_iter=n_iter, sigma=sigma, local_m=local_m,
                         corr=corr, E=E, E2=E2, next_step=t + 1,
                         rng_state=_get_rng_state(rng))
        
    local_m = local_m / n_iter
    corr = corr / n_iter
    covariance = corr - np.outer(local_m, local_m)
    E = E / n_iter
    E2 = E2 / n_iter
    C = beta**2 * (E2 - E**2) / n_node
    q = np.inner(local_m, local_m) / n_node
    m = np.sum(local_m) / n_node
    # sum of squared eigenvalues of the (symmetric) covariance matrix, i.e.
    # its squared Frobenius norm
    chi_sg = beta**2 * np.sum(covariance ** 2) / n_node
    chi_u = beta * np.sum(covariance) / n_node
    
    return m, q, chi_u, chi_sg, C, local_m, corr

def _get_rng_state(rng):
    """Serialize the state of a RandomState as a JSON string."""
    state = rng.get_state(legacy=False)
    state['state']['key'] = state['state']['key'].tolist()
    return json.dumps(state)

def _set_rng_state(rng, state_json):
    """Restore the state of a RandomState from a JSON string."""
    state = json.loads(state_json)
    state['state']['key'] = np.array(state['state']['key'], dtype=np.uint32)
    rng.set_state(state)

def sk_diagram(h, J, beta=1.0, mu_range=(-0.002, .0095), mu_step=.0005, 
               std_range=(.005, .1), std_step=.005, mu_width=None, 
               std_width=None, n_iter=1000, rand_seed=0, rand_init=False,
               multiproc=False, n_cpu=None, save_every=None,
               save_path=None):
    """Create a Sherrington-Kirkpatrick phase diagram by fixing the estimated
    intrinsic field h and sweeping through different values of the mean and 
    standard deviation of the estimated interaction matrix J. The x-axis is 
    the mean and the y-axis is the standard deviation.

    Parameters
    ----------
    h : 1d array (dtype='float')
        The estimated elementwise intrinsic field.
    J : 2d array (dtype='float')
        The estimated n_node x n_node interaction matrix.    
    beta : float, optional
        The inverse temperature. Default at unit temperature 1.0
    mu_range : tuple of float
        Minimum and maximum values for the parameter sweep of the mean.
    mu_step : float
        Step for the parameter sweep of the mean.
    std_range : tuple of float
        Minimum and maximum values for the parameter sweep of the standard 
        deviation.
    std_step : float
        Step for the parameter sweep of the standard deviation.
    mu_width : float, optional
        Replaces `mu_range`. `mu_width` then determines the interval to plot, 
        centered around the mean of J.
    std_width : float, optional
        Replaces `std_range`. `std_width` then determines the interval to plot, 
        centered around the standard deviation of J.
    n_iter : int
        Number of iterations for Metropolis sampling.
    rand_seed : int
        Seed for random number generator.
    rand_init : bool
        If True, the initial spin configuration `sigma` for each simulation
        will be randomly generated using rand_seed. If False, `sigma` will be 
        all ones.
    multiproc : bool
        Whether to use multiprocessing.
    n_cpu : int or None
        For use with multiprocessing. If None, will use mp.cpu_count().
    save_every : int, optional
        If not None, save the simulation every `save_every` iterations.
    save_path : str, optional
        Path prefix to save the simulations. Each cell (i, j) of the diagram
        is saved under `{save_path}_cell{i}-{j}`.

    Returns
    -------
    m : 2d array
        Phase diagram in terms of the magnetization order parameter, of
        shape (len(x_range), len(y_range)).
    q : 2d array
        Phase diagram in terms of the spin-glass order parameter.
    chi_u : 2d array
        Phase diagram in terms of the uniform susceptibility.
    chi_sg : 2d array
        Phase diagram in terms of the spin-glass susceptibility.
    C : 2d array
        Phase diagram in terms of the specific heat.
    x_range : 1d array
        Values for the mean.
    y_range : 1d array
        Values for the standard deviation.

    Notes
    -----
    With `multiproc=True`, scripts calling this function must be protected by
    `if __name__ == '__main__':` on platforms that spawn new processes
    (macOS, Windows).

    References
    ----------
    Ezaki et al. (2020) Commun Biol 3, 52.
    Fischer & Hertz (1991) "Spin glasses". Cambridge UP.
    
    Acknowledgements
    ----------------
    This function was adapted from Matlab code by Takahiro Ezaki: 
    https://github.com/tkEzaki/phase_diagram_analysis.
    """
    off_diag = ~np.eye(J.shape[0], dtype=bool)
    if mu_width is not None:
        mu_actual = J[off_diag].mean()
        x_range = np.arange(mu_actual - mu_width / 2,
                            mu_actual + mu_width / 2, 
                            mu_step)
    else:
        x_range = np.arange(*mu_range, mu_step)
    if std_width is not None:
        std_actual = J[off_diag].std()
        y_range = np.arange(std_actual - std_width / 2,
                            std_actual + std_width / 2, 
                            std_step)
    else:
        y_range = np.arange(*std_range, std_step)
    diagram_shape = (len(x_range), len(y_range))
    
    m =      np.zeros(diagram_shape)
    q =      np.zeros(diagram_shape)
    chi_u =  np.zeros(diagram_shape)
    chi_sg = np.zeros(diagram_shape)
    C =      np.zeros(diagram_shape)

    # one simulation per (mu, stdev) cell, each with its own checkpoint path
    cells = [(i, j) for i in range(len(x_range)) for j in range(len(y_range))]
    args = [(h, J, beta, x_range[i], y_range[j], n_iter, rand_seed, rand_init,
             save_every, _cell_save_path(save_path, i, j))
            for i, j in cells]

    if multiproc:
        # NB: on macOS and Windows, scripts calling this function must be
        # protected by `if __name__ == '__main__':`
        if n_cpu is None:
            n_cpu = mp.cpu_count()
        with mp.Pool(n_cpu) as pool:
            results = pool.starmap(trans_and_metropolis, args)
    else:
        results = [trans_and_metropolis(*a) for a in args]

    for (i, j), res in zip(cells, results):
        m[i, j], q[i, j], chi_u[i, j], chi_sg[i, j], C[i, j] = res[:5]
    
    return m, q, chi_u, chi_sg, C, x_range, y_range

def _cell_save_path(save_path, i, j):
    """Checkpoint path prefix for cell (i, j) of a phase diagram."""
    if save_path is None:
        return None
    return f'{save_path}_cell{i}-{j}'

def affine_trans(J, mu, stdev):
    """Linearly transform a square matrix to produce a new matrix with a 
    given mean and standard deviation.

    Parameters
    ----------
    J : 2d array (dtype='float')
        A square matrix.
    mu : float
        The new mean.
    stdev : float
        The new standard deviation.

    Returns
    -------
    J_tilde : 2d array (dtype='float')
        The transformed matrix whose off-diagonal elements have mean `mu` and
        standard deviation `stdev`. The diagonal is set to zero.
        
    References
    ----------
    Ezaki et al. (2020) Commun Biol 3, 52.
    
    Acknowledgements
    ----------------
    This function was adapted from Matlab code by Takahiro Ezaki: 
    https://github.com/tkEzaki/phase_diagram_analysis.
    """
    off_diag = ~np.eye(J.shape[0], dtype=bool)
    std_orig = np.std(J[off_diag])
    mu_orig = np.mean(J[off_diag])
    
    # transform the off-diagonal elements only (no self-interaction: J_ii = 0)
    J_tilde = np.zeros(J.shape)
    J_tilde[off_diag] = (J[off_diag] - mu_orig) * stdev / std_orig + mu
        
    return J_tilde

def trans_and_metropolis(h, J, beta, mu, stdev, n_iter=1000, rand_seed=0, 
                         rand_init=False, save_every=None, save_path=None):
    """Linearly transform a square matrix to produce a new matrix with a 
    given mean and standard deviation, then sample data from the estimated 
    model using the Metropolis method and compute statistics.

    Parameters
    ----------
    h : 1d array (dtype='float')
        The estimated elementwise intrinsic field.
    J : 2d array (dtype='float')
        The estimated n_node x n_node interaction matrix.
    beta : float, optional
        The inverse temperature. 
    mu : float
        The new mean.
    stdev : float
        The new standard deviation.
    n_iter : int
        Number of iterations for Metropolis sampling.
    rand_seed : int
        Seed for random number generator.
    rand_init : bool
        If True, the initial spin configuration `sigma` will be randomly
        generated using rand_seed. If False, `sigma` will be all ones.
    
    Returns
    -------
    m : float
        Magnetization order parameter.
    q : float
        Spin-glass order parameter.
    chi_u : float
        Uniform susceptibility.
    chi_sg : float
        Spin-glass susceptibility.
    C : float
        Specific heat.
    local_m : 2d array, dtype='float'
        Local mean.
    corr : 2d array, dtype='float'
        Correlation matrix.
    """
    J_tilde = affine_trans(J, mu, stdev)
    m, q, chi_u, chi_sg, C, local_m, corr = metropolis(h, 
                                                       J_tilde, 
                                                       beta,
                                                       n_iter=n_iter, 
                                                       rand_seed=rand_seed,
                                                       rand_init=rand_init,
                                                       save_every=save_every,
                                                       save_path=save_path)
    
    return m, q, chi_u, chi_sg, C, local_m, corr

def plot_sk(order_param, mu_values, sigma_values, J=None, title='SK diagram',
            ax=None):
    """Plot the SK diagram.
    
    Parameters
    ----------
    order_param : 2d array
        The phase diagram of one order parameter (e.g. `m`, `q`, `chi_u`,
        `chi_sg` or `C` as returned by `sk_diagram`), of shape
        (len(mu_values), len(sigma_values)).
    mu_values : 1d array
        The mu (surrogate mean) values, on the x-axis.
    sigma_values : 1d array
        The sigma (surrogate standard deviation) values, on the y-axis
    J : 2d array of shape (N, N), optional
        The interaction matrix, for plotting the working point (mean and
        standard deviation of its off-diagonal elements).
    title : str, optional
        The plot title.
    ax : matplotlib.axes.Axes, optional
        Axes to plot into. If None, a new figure is created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    # pixel edges: half a step beyond the first and last values
    half_dx = (mu_values[1] - mu_values[0]) / 2 if len(mu_values) > 1 else .5
    half_dy = ((sigma_values[1] - sigma_values[0]) / 2
               if len(sigma_values) > 1 else .5)
    # order_param is indexed [mu, sigma]; imshow expects [row (y), col (x)]
    im = ax.imshow(np.asarray(order_param).T, origin='lower', aspect='auto',
                   extent=(mu_values[0] - half_dx, mu_values[-1] + half_dx,
                           sigma_values[0] - half_dy,
                           sigma_values[-1] + half_dy))
    fig.colorbar(im, ax=ax)
    ax.set_xlabel('mean(J)')
    ax.set_ylabel('std(J)')
    ax.set_title(title)

    if J is not None:
        off_diag = ~np.eye(J.shape[0], dtype=bool)
        ax.plot(np.mean(J[off_diag]), np.std(J[off_diag]), marker='*',
                color='red', markersize=10)

    return fig, ax

def ht_diagram(h, J, tail_cutoff,
               alpha_range=None, beta_range=None, mu_range=None, 
               sigma_range=None, n_iter=1000, metropolis_beta=1.0, 
               rand_seed=0, rand_init=True, save_every=None, save_path=None):
    """Produce "heavy-tail" phase diagram with 2 control parameters: 
    x is tail alpha or bulk mu;
    y is tail beta or bulk sigma.

    Parameters
    ----------
    h : 1d array (dtype='float')
        The estimated elementwise intrinsic field.
    J : 2d array (dtype='float')
        The estimated n_node x n_node interaction matrix.
    tail_cutoff : float
        Weight value which separates the bulk from the tail of the weight 
        distribution of J.
    alpha_range : tuple of form (float, float, float), optional
        Minimum, maximum and step values for the parameter sweep of the 
        tail offset.
    beta_range : tuple of form (float, float, float), optional
        Minimum, maximum and step values for the parameter sweep of the 
        tail slope.  
    mu_range : tuple of form (float, float, float), optional
        Minimum, maximum and step values for the parameter sweep of the 
        mean of the bulk weights.
    sigma_range : tuple of form (float, float, float), optional
        Minimum, maximum and step values for the parameter sweep of the 
        standard deviation of the bulk weights.
    n_iter : int, optional
        Number of iterations for Metropolis sampling. Default at 1000.
    metropolis_beta : float, optional
        The inverse temperature for Metropolis sampling. Default at unit 
        temperature 1.0.
    rand_seed : int, optional
        Seed for random number generator. Default at 0.
    rand_init : bool, optional
        If True, the initial spin configuration `sigma` for each simulation
        will be randomly generated using rand_seed. If False, `sigma` will be 
        all ones. Default at True.
    save_every : int, optional
        If not None, save the simulation every `save_every` iterations.
    save_path : str, optional
        Path prefix to save the simulations. Each cell (i, j) of the diagram
        is saved under `{save_path}_cell{i}-{j}`.

    Returns
    -------
    m : 2d array
        Phase diagram in terms of the magnetization order parameter.
    q : 2d array
        Phase diagram in terms of the spin-glass order parameter.
    chi_u : 2d array
        Phase diagram in terms of the uniform susceptibility.
    chi_sg : 2d array
        Phase diagram in terms of the spin-glass susceptibility.
    C : 2d array
        Phase diagram in terms of the specific heat.
    x_range : 1d array
        Values for the mean.
    y_range : 1d array
        Values for the standard deviation.

    References
    ----------
    Ezaki et al. (2020) Commun Biol 3, 52.
    Fischer & Hertz (1991) "Spin glasses". Cambridge UP.
    
    Acknowledgements
    ----------------
    This function was adapted from Matlab code by Takahiro Ezaki: 
    https://github.com/tkEzaki/phase_diagram_analysis.
    """
    if alpha_range is None and mu_range is None:
        raise ValueError('Specify either alpha_range or mu_range.')
    if beta_range is None and sigma_range is None:
        raise ValueError('Specify either beta_range or sigma_range.')

    # split tail from bulk, excluding the diagonal (J_ii = 0)
    off_diag = ~np.eye(J.shape[0], dtype=bool)
    tail_mask = (J > tail_cutoff) & off_diag
    bulk_mask = ~tail_mask & off_diag
    bulk = J[bulk_mask]
    bulk_mean = bulk.mean()
    bulk_std = bulk.std()

    if alpha_range is not None:
        alpha_values = np.arange(*alpha_range)
        x_range = alpha_values
        mu_values = np.ones_like(x_range) * bulk_mean # replace mean with mean
    elif mu_range is not None:
        mu_values = np.arange(*mu_range)
        x_range = mu_values
        alpha_values = np.zeros_like(x_range) # + 0
    if beta_range is not None:
        beta_values = np.arange(*beta_range)
        y_range = beta_values
        sigma_values = np.ones_like(y_range) * bulk_std # replace std with std
    elif sigma_range is not None:
        sigma_values = np.arange(*sigma_range)
        y_range = sigma_values
        beta_values = np.ones_like(y_range) # * 1
    diagram_shape = (len(alpha_values), len(beta_values))
    m =      np.zeros(diagram_shape)
    q =      np.zeros(diagram_shape)
    chi_u =  np.zeros(diagram_shape)
    chi_sg = np.zeros(diagram_shape)
    C =      np.zeros(diagram_shape)

    for i, (alpha, mu) in enumerate(zip(alpha_values, mu_values)):
        for j, (beta, sigma) in enumerate(zip(beta_values, sigma_values)):
            J_tilde = J.copy()
            # linear model
            J_tilde[tail_mask] = J_tilde[tail_mask] * beta + alpha
            # affine transformation
            J_tilde[bulk_mask] = (bulk - bulk_mean) * (sigma / bulk_std) + mu
            (m[i, j], q[i, j], chi_u[i, j], chi_sg[i, j], C[i, j], 
            _, _) = metropolis(h, J_tilde, beta=metropolis_beta, n_iter=n_iter, 
                                rand_seed=rand_seed, rand_init=rand_init,
                                save_every=save_every,
                                save_path=_cell_save_path(save_path, i, j))
    
    return m, q, chi_u, chi_sg, C, x_range, y_range

def ising_energy(sigma, h, J):
    """Compute the energy of a spin configuration in the Ising model.

    Parameters
    ----------
    sigma : ndarray of shape (N,)
        Spin configuration vector where each element is +1 or -1.
    h : ndarray of shape (N,)
        External magnetic field vector.
    J : ndarray of shape (N, N)
        Interaction matrix between spins.

    Returns
    -------
    float
        The total energy of the given spin configuration.
    """
    return -0.5 * sigma @ J @ sigma - h @ sigma

def heat_capacity(data, h, J, mode='capacity'):
    """Compute the heat capacity (or specific heat) for a set of spin 
    configurations in the Ising model using energy variance.

    Parameters
    ----------
    data : ndarray of shape (N, M)
        Matrix of spin configurations, where N is the number of spins and
        M is the number of samples.
    h : ndarray of shape (N,)
        External magnetic field vector.
    J : ndarray of shape (N, N)
        Interaction matrix between spins.
    mode : {'capacity', 'specific'}, default='capacity'
        - 'capacity': returns the total heat capacity (variance of energies)
        - 'specific': returns the heat capacity per spin (specific heat)

    Returns
    -------
    float
        The heat capacity (or specific heat).

    Notes
    -----
    Computed at unit temperature (beta = 1): C = beta**2 * Var(E).
    """
    if mode not in ('capacity', 'specific'):
        raise ValueError("mode must be 'capacity' or 'specific'.")
    # energy of every sample: -0.5 * s.J.s - h.s
    E = -0.5 * np.sum(data * (J @ data), axis=0) - h @ data
    heat = np.var(E)
    if mode == 'specific':
        heat = heat / data.shape[0]
    return heat

def energy_landscape_2d(sigma, length, flips, h, J):
    """Compute an arbitrary slice of the energy landscape for model {h, J}.

    Parameters
    ----------
    sigma : ndarray of shape (N,)
        Starting spin configuration vector where each element is +1 or -1.
    length : int
        Axis length of the square energy landscape.
    flips : ndarray of shape (2 * length,)
        The sequence of spin indices to flip. The first half is for axis 1,
        and the second half for axis 2.
    h : ndarray of shape (N,)
        External magnetic field vector.
    J : ndarray of shape (N, N)
        Interaction matrix between spins.

    Returns
    -------
    el : ndarray of shape (length, length)
        The energy landscape spanned by the two spin-flip axes.
    """
    el = np.zeros((length, length))
    sigma = sigma.copy()

    for i in range(length):
        sigma[flips[i]] = -sigma[flips[i]]
        new_sigma = sigma.copy()
        for j in range(length):
            new_sigma[flips[length + j]] = -new_sigma[flips[length + j]]
            el[i, j] = ising_energy(new_sigma, h, J)

    return el
"""Synthetic data generators with known properties, used across the tests."""

import itertools

import numpy as np


# ----------------------------------------------------------------- avalanches

def branching_counts(sigma, n_avl, rng, max_dur=10_000, gap=3):
    """Event counts of `n_avl` Galton-Watson avalanches, concatenated.

    Each avalanche starts from a single active unit; every active unit
    activates Poisson(`sigma`) units in the next time step. At sigma = 1 the
    process is critical, with mean-field exponents tau = 1.5 (sizes),
    alpha = 2 (durations) and third = 2. Avalanches are separated by `gap`
    empty samples; avalanches longer than `max_dur` are discarded.
    """
    pieces = [np.zeros(gap, dtype=int)]
    for _ in range(n_avl):
        counts, active = [], 1
        while active and len(counts) < max_dur:
            counts.append(active)
            active = rng.poisson(sigma * active)
        if active: # still running after max_dur: discard
            continue
        pieces += [np.array(counts), np.zeros(gap, dtype=int)]
    return np.concatenate(pieces)


# ---------------------------------------------------------------------- chaos

def logistic_map(n, r=3.9, x0=0.4):
    """Logistic map x <- r x (1 - x); chaotic for r = 3.9."""
    x = np.empty(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = r * x[i - 1] * (1 - x[i - 1])
    return x


def henon_x(n, a=1.4, b=0.3):
    """x-coordinate of the (chaotic) Henon map."""
    x, y = np.empty(n), np.empty(n)
    x[0], y[0] = 0.1, 0.1
    for i in range(1, n):
        x[i] = 1 - a * x[i - 1] ** 2 + y[i - 1]
        y[i] = b * x[i - 1]
    return x


def linear_network(g, n_units, n_steps, rng):
    """Linear network x(t+1) = W x(t) + noise, with random Gaussian W.

    Returns the activity (n_units x n_steps) and the true spectral radius of
    W, i.e. lambda_max, which is close to `g`.
    """
    W = rng.normal(0, g / np.sqrt(n_units), (n_units, n_units))
    lam = np.max(np.abs(np.linalg.eigvals(W)))
    noise = rng.standard_normal((n_units, n_steps))
    x = np.zeros((n_units, n_steps))
    for t in range(1, n_steps):
        x[:, t] = W @ x[:, t - 1] + noise[:, t]
    return x, lam


# ---------------------------------------------------------------------- ising

def random_ising_model(n, rng, h_std=0.3, j_std=0.4):
    """Random field h and symmetric coupling J with zero diagonal."""
    h = rng.normal(0, h_std, n)
    J = rng.normal(0, j_std, (n, n))
    J = (J + J.T) / 2
    np.fill_diagonal(J, 0)
    return h, J


def all_states(n):
    """All 2**n spin configurations (rows), coded as -1/+1."""
    return np.array(list(itertools.product([-1, 1], repeat=n)), dtype=float)


def boltzmann_probabilities(h, J, beta=1.0):
    """Exact probabilities of all states, with E = -0.5 s.J.s - h.s."""
    states = all_states(len(h))
    E = -0.5 * np.einsum('si,ij,sj->s', states, J, states) - states @ h
    p = np.exp(-beta * (E - E.min()))
    return states, E, p / p.sum()


def exact_ising_samples(h, J, n_samples, rng, beta=1.0):
    """Independent exact samples (n_nodes x n_samples) of the Ising model."""
    states, _, p = boltzmann_probabilities(h, J, beta)
    return states[rng.choice(len(states), n_samples, p=p)].T

"""Tests for edgeofpy.models (model systems with known critical points)."""

import warnings

import numpy as np
import pytest

import edgeofpy as eop
from edgeofpy import models
from helpers import boltzmann_probabilities, random_ising_model


def quiet_fit_powerlaw(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return eop.fit_powerlaw(*args, **kwargs)


def avalanche_stats(activity):
    avls = eop.detect_avalanches(activity, s_freq=1000, max_iei=0.001)[0]
    sizes = np.array([a['size'] for a in avls])
    durs = np.array([a['dur_bin'] for a in avls])
    return avls, sizes, durs


# ---------------------------------------------------------- branching_process

def test_branching_process_structure():
    counts = models.branching_process(1.0, 200, gap=3, rand_seed=0)
    assert counts.ndim == 1 and np.issubdtype(counts.dtype, np.integer)
    assert np.all(counts[:3] == 0) and np.all(counts[-3:] == 0)
    avls, sizes, _ = avalanche_stats(counts[np.newaxis, :])
    assert len(avls) == 200          # every avalanche is detected
    assert sizes.sum() == counts.sum()


def test_branching_process_reproducible():
    a = models.branching_process(1.0, 100, rand_seed=7)
    b = models.branching_process(1.0, 100, rand_seed=7)
    np.testing.assert_array_equal(a, b)


def test_branching_process_max_dur():
    counts = models.branching_process(1.5, 100, max_dur=50, rand_seed=0)
    avls, _, durs = avalanche_stats(counts[np.newaxis, :])
    assert len(avls) < 100 and durs.max() <= 50


@pytest.mark.slow
def test_branching_process_critical_exponents():
    counts = models.branching_process(1.0, 20000, rand_seed=1)
    _, sizes, durs = avalanche_stats(counts[np.newaxis, :])
    assert quiet_fit_powerlaw(sizes)['power_law_exp'] == pytest.approx(1.5,
                                                                       abs=0.1)
    assert quiet_fit_powerlaw(durs)['power_law_exp'] == pytest.approx(2.0,
                                                                      abs=0.2)


def test_branching_process_invalid_gap():
    with pytest.raises(ValueError):
        models.branching_process(1.0, 10, gap=0)


# ---------------------------------------------------------- branching_network

def test_branching_network_shapes_and_pooling():
    full = models.branching_network(n_units=100, n_steps=2000, p_spont=1e-3,
                                    rand_seed=0)
    assert full.shape == (100, 2000) and full.max() <= 1
    pooled = models.branching_network(n_units=100, n_steps=2000,
                                      p_spont=1e-3, n_channels=8, rand_seed=0)
    assert pooled.shape == (8, 2000)
    np.testing.assert_array_equal(pooled.sum(0), full.sum(0))


def test_branching_network_refractory():
    act = models.branching_network(n_units=50, sigma=1.5, n_steps=5000,
                                   p_spont=1e-3, rand_seed=0)
    assert not np.any(act[:, 1:] & act[:, :-1])  # never active twice in a row


@pytest.mark.parametrize('sigma', [0.5, 0.8])
def test_branching_network_subcritical_branching_ratio(sigma):
    n_units, p_spont = 1000, 1e-4
    act = models.branching_network(n_units=n_units, sigma=sigma,
                                   n_steps=100_000, p_spont=p_spont,
                                   n_channels=1, rand_seed=0)
    # E[A(t+1) | A(t)] = sigma A(t) + n_units p_spont, when activity is low
    events = act[0].astype(float)
    pairs = events[:-1] > 0
    descendants = events[1:][pairs].sum() - pairs.sum() * n_units * p_spont
    ratio = descendants / events[:-1][pairs].sum()
    assert ratio == pytest.approx(sigma, abs=0.03)


def test_branching_network_regimes():
    act = {s: models.branching_network(n_units=500, sigma=s, n_steps=20_000,
                                       n_channels=1, rand_seed=0)[0]
           for s in (0.8, 1.2)}
    assert np.mean(act[0.8] == 0) > 0.9    # subcritical: mostly silent
    assert np.mean(act[1.2] == 0) < 0.01   # supercritical: self-sustained


@pytest.mark.slow
def test_branching_network_critical_exponents():
    act = models.branching_network(n_units=1000, sigma=1.0, n_steps=300_000,
                                   n_channels=32, rand_seed=1)
    _, sizes, durs = avalanche_stats(act)
    # fixed xmin: the automatic choice differs between powerlaw versions on
    # these data (1 vs 2 for the sizes), which changes the DCC by ~0.2
    fit_d = quiet_fit_powerlaw(durs, xmin=3)
    tau = quiet_fit_powerlaw(sizes, xmin=2)['power_law_exp']
    third = eop.fit_third_exponent(sizes, durs,
                                   dur_range=(fit_d['xmin'], fit_d['xmax']))
    assert tau == pytest.approx(1.5, abs=0.1)
    assert eop.dcc(tau, fit_d['power_law_exp'], third) < 0.1


def test_branching_network_invalid_n_out():
    with pytest.raises(ValueError):
        models.branching_network(n_units=10, n_out=10)


# ------------------------------------------------------------------- kuramoto

def test_kuramoto_shapes_and_phases():
    x, ph = models.kuramoto(n_osc=8, duration=2, s_freq=250, transient=0.5,
                            rand_seed=0)
    assert x.shape == ph.shape == (8, 500)
    assert np.all(ph >= -np.pi) and np.all(ph < np.pi)
    np.testing.assert_allclose(x, np.cos(ph))


def test_kuramoto_uncoupled_frequency():
    _, ph = models.kuramoto(n_osc=4, coupling=0, freq_mean=10, freq_std=0,
                            duration=2, s_freq=250, rand_seed=0)
    dphi = np.mod(np.diff(ph, axis=1), 2 * np.pi)
    np.testing.assert_allclose(dphi, 2 * np.pi * 10 / 250)


def test_kuramoto_synchronization_transition():
    def order(coupling):
        _, ph = models.kuramoto(n_osc=64, coupling=coupling, duration=20,
                                rand_seed=2)
        return np.abs(np.mean(np.exp(1j * ph), axis=0)).mean()
    r = [order(k) for k in (0, 10, 25)]  # K_c ~ 10 rad/s for freq_std 1 Hz
    assert r[0] < 0.2 and r[2] > 0.9 and r[0] < r[1] < r[2]


def test_kuramoto_noise_desynchronizes():
    _, quiet = models.kuramoto(n_osc=32, coupling=20, duration=10,
                               rand_seed=0)
    _, noisy = models.kuramoto(n_osc=32, coupling=20, duration=10,
                               noise=40, rand_seed=0)
    order = [np.abs(np.mean(np.exp(1j * p), axis=0)).mean()
             for p in (quiet, noisy)]
    assert order[1] < order[0]


# --------------------------------------------------------------- logistic_map

def test_logistic_map_values():
    x = models.logistic_map(3, r=3.9, x0=0.4, transient=0)
    np.testing.assert_allclose(x, [0.4, 0.936, 3.9 * 0.936 * 0.064])


def test_logistic_map_periodic_and_chaotic():
    periodic = models.logistic_map(1000, r=3.2)  # period 2 after transient
    np.testing.assert_allclose(periodic[2:], periodic[:-2], atol=1e-9)
    assert eop.z1_chaos_test(models.logistic_map(2000, r=3.9)) > 0.9


# ------------------------------------------------------------- linear_network

@pytest.mark.parametrize('rho', [0.3, 0.9])
def test_linear_network_spectral_radius(rho):
    x, W = models.linear_network(n_units=20, spectral_radius=rho,
                                 n_steps=100, rand_seed=0)
    assert x.shape == (20, 100)
    assert np.max(np.abs(np.linalg.eigvals(W))) == pytest.approx(rho)


def test_linear_network_fluctuations_grow_near_edge():
    var = [models.linear_network(n_units=20, spectral_radius=rho,
                                 n_steps=20_000, rand_seed=0)[0].var()
           for rho in (0.3, 0.95)]
    assert var[1] > 2 * var[0]


def test_linear_network_invalid_radius():
    with pytest.raises(ValueError):
        models.linear_network(spectral_radius=1.0)


# -------------------------------------------------------------- ising_samples

def test_ising_samples_match_boltzmann_distribution():
    h, J = random_ising_model(5, np.random.default_rng(0))
    s = models.ising_samples(h, J, 20000, rand_seed=4)
    assert s.shape == (5, 20000) and set(np.unique(s)) == {-1, 1}
    states, _, p = boltzmann_probabilities(h, J)
    np.testing.assert_allclose(s.mean(1), states.T @ p, atol=0.03)
    corr = s @ s.T / s.shape[1]
    exact = np.einsum('s,si,sj->ij', p, states, states)
    np.testing.assert_allclose(corr, exact, atol=0.03)


def test_ising_samples_beta():
    # beta = 0: independent uniform spins, whatever h and J
    h, J = random_ising_model(4, np.random.default_rng(1), h_std=2)
    s = models.ising_samples(h, J, 20000, beta=0, rand_seed=0)
    np.testing.assert_allclose(s.mean(1), 0, atol=0.03)


def test_ising_samples_recovered_by_fit_pmem():
    h, J = random_ising_model(5, np.random.default_rng(0))
    s = models.ising_samples(h, J, 50000, rand_seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        h_fit, J_fit = eop.fit_pmem(s, method='ezaki')
    np.testing.assert_allclose(h_fit, h, atol=0.1)
    np.testing.assert_allclose(J_fit, J, atol=0.1)


def test_ising_samples_asymmetric_J():
    with pytest.raises(ValueError):
        models.ising_samples(np.zeros(2), np.array([[0, 1.], [0, 0]]), 10)

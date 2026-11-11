"""Tests for edge-of-chaos measures."""

import numpy as np
import pytest

import edgeofpy as eop
from edgeofpy import models
from edgeofpy.chaos import _mean_square_displacement, takens_reconstruction
from helpers import henon_x, linear_network, logistic_map


def normalized(x):
    return x * 0.5 / np.std(x)


# ---------------------------------------------------------------- 0-1 test

def test_z1_chaotic_signal():
    assert eop.z1_chaos_test(normalized(logistic_map(1000))) > 0.9


def test_z1_regular_signal():
    assert abs(eop.z1_chaos_test(normalized(np.sin(0.3 * np.arange(1000))))) \
        < 0.1


def test_z1_reference_values():
    # Fixed values for the default seed: guards against unintended changes
    # in random number usage (identical across numpy 1.22-2.2)
    k_logistic = eop.z1_chaos_test(normalized(logistic_map(1000)))
    k_sine = eop.z1_chaos_test(normalized(np.sin(0.3 * np.arange(1000))))
    assert k_logistic == pytest.approx(0.995175154687, rel=1e-9)
    assert k_sine == pytest.approx(-0.014478247082, rel=1e-9)


def test_z1_seed():
    x = normalized(logistic_map(500))
    assert eop.z1_chaos_test(x, rand_seed=1) == eop.z1_chaos_test(x,
                                                                  rand_seed=1)
    assert eop.z1_chaos_test(x, rand_seed=1) != eop.z1_chaos_test(x,
                                                                  rand_seed=2)


def test_z1_does_not_touch_global_rng():
    # regression: np.random.seed() reset the global random state
    np.random.seed(123)
    expected = np.random.random()
    np.random.seed(123)
    eop.z1_chaos_test(normalized(logistic_map(300)))
    assert np.random.random() == expected


def test_mean_square_displacement_matches_direct(rng):
    z = np.cumsum(rng.standard_normal(500) + 1j * rng.standard_normal(500))
    direct = [np.mean(np.abs(z[n:] - z[:-n]) ** 2) for n in range(1, 51)]
    np.testing.assert_allclose(_mean_square_displacement(z, 50), direct,
                               rtol=1e-10)


# ------------------------------------------------------------ chaos_pipeline

def test_chaos_pipeline_short_signal():
    assert np.isnan(eop.chaos_pipeline(np.sin(np.arange(30))))


def test_chaos_pipeline_chaotic():
    assert eop.chaos_pipeline(logistic_map(2000)) > 0.9


def test_chaos_pipeline_with_denoising(rng):
    x = henon_x(1500) + rng.normal(0, 0.05, 1500)
    k = eop.chaos_pipeline(x, denoise=True, denoise_r=0.1)
    assert -1 <= k <= 1


# ---------------------------------------------------------------- lambda_max

@pytest.mark.slow
def test_lambda_max_tracks_true_value():
    # regression: the old estimate was ~0.88 whatever the true value
    estimates, truths = [], []
    for g in (0.3, 0.6, 0.9):
        x, lam = linear_network(g, 100, 60_000, np.random.default_rng(0))
        estimates.append(eop.lambda_max(x, s_freq=100, N=100,
                                        trial_len_sec=2)[0])
        truths.append(lam)
    assert estimates[0] < estimates[1] < estimates[2]
    assert estimates[1] == pytest.approx(truths[1], abs=0.1)


@pytest.mark.parametrize('rho', [0.3, 0.5])
def test_lambda_max_bias_correction(rho):
    # without correction, sampling noise of the covariances biased the
    # estimate upward (0.57 for a true 0.3, with 100 trials)
    x, _ = models.linear_network(n_units=100, spectral_radius=rho,
                                 n_steps=100_000, rand_seed=0)
    raw = eop.lambda_max(x, s_freq=100, N=100, bias_correction=False)[0]
    corrected = eop.lambda_max(x, s_freq=100, N=100)[0]
    assert corrected == pytest.approx(rho, abs=0.05)
    assert raw > corrected + 0.1


def test_lambda_max_independent_units_is_zero(rng):
    # no interactions: the corrected dispersion vanishes (up to noise)
    lam = eop.lambda_max(rng.standard_normal((20, 50_000)), s_freq=100, N=20,
                         trial_len_sec=1)[0]
    assert lam < 0.2


def test_lambda_max_does_not_modify_input(rng):
    # regression: the input array was modified in place
    data = rng.standard_normal((5, 3000))
    original = data.copy()
    eop.lambda_max(data, s_freq=100, N=50, trial_len_sec=1)
    np.testing.assert_array_equal(data, original)


def test_lambda_max_integer_events(rng):
    # regression: integer (event count) data raised a casting error
    lam, norm_var, c = eop.lambda_max(rng.poisson(1, (5, 5000)), 100, 100, 5)
    assert 0 <= lam <= 1
    assert c.shape == (5, 5)


@pytest.mark.parametrize('shape, trial_len_sec', [
    ((1, 5000), 5),   # a single site
    ((5, 300), 5),    # fewer than two trials
    ((5, 5000), 0.001),  # trial shorter than one sample
])
def test_lambda_max_invalid_input(rng, shape, trial_len_sec):
    with pytest.raises(ValueError):
        eop.lambda_max(rng.standard_normal(shape), 100, 100, trial_len_sec)


# --------------------------------------------------------- Schreiber denoising

@pytest.fixture
def noisy_henon(rng):
    clean = henon_x(3000)
    return clean, clean + rng.normal(0, 0.05, clean.size)


def rms(a, b):
    return np.sqrt(np.mean((a - b) ** 2))


def test_schreiber_denoise_shape_and_ends(noisy_henon):
    # regression: crashed, and the end samples were replaced by zeros
    _, noisy = noisy_henon
    out = eop.schreiber_denoise(noisy, r=0.1)
    assert out.shape == noisy.shape
    assert out[0] == noisy[0] and out[-1] == noisy[-1]


def test_schreiber_denoise_reduces_noise(noisy_henon):
    clean, noisy = noisy_henon
    out = eop.schreiber_denoise(noisy, r=0.1) # about 2x the noise level
    assert rms(out, clean) < 0.8 * rms(noisy, clean)


@pytest.mark.xfail(strict=True, reason='known issue: the default radius '
                   'r = std(x) over-smooths and increases the error')
def test_schreiber_denoise_default_radius_reduces_noise(noisy_henon):
    clean, noisy = noisy_henon
    assert rms(eop.schreiber_denoise(noisy), clean) < rms(noisy, clean)


def test_schreiber_denoise_chunk_size_invariant(noisy_henon):
    _, noisy = noisy_henon
    # equal up to floating-point summation order
    np.testing.assert_allclose(
        eop.schreiber_denoise(noisy[:500], r=0.1, chunk_size=7),
        eop.schreiber_denoise(noisy[:500], r=0.1), rtol=1e-12)


# --------------------------------------------------------- helper functions

def test_takens_reconstruction():
    Y, n = takens_reconstruction(np.arange(10), 3, 2)
    assert n == 6
    np.testing.assert_array_equal(Y[0], [0, 2, 4])
    np.testing.assert_array_equal(Y[-1], [5, 7, 9])


def test_takens_reconstruction_too_short():
    with pytest.raises(ValueError):
        takens_reconstruction(np.arange(4), 3, 2)


def test_minmaxsig():
    x = np.array([0., 2, 1, 3, 0, 1, 1, 0])
    # extrema at 1 (max), 2 (min), 3 (max), 4 (min); the plateau is ignored
    np.testing.assert_array_equal(eop.minmaxsig(x), [2, 1, 3, 0])

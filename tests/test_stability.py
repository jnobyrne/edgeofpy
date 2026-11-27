"""Tests for the distance to instability (lambda_max)."""

import warnings

import numpy as np
import pytest

import edgeofpy as eop
from edgeofpy import models
from helpers import linear_network


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


def test_lambda_max_old_location_warns():
    # moved from edgeofpy.chaos in 0.3.0; the old import still works
    from edgeofpy import stability
    with pytest.warns(DeprecationWarning, match='edgeofpy.stability'):
        from edgeofpy.chaos import lambda_max
    assert lambda_max is stability.lambda_max is eop.lambda_max
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        from edgeofpy.stability import lambda_max as lm  # noqa: F401
    from edgeofpy import chaos
    with pytest.raises(AttributeError):
        chaos.no_such_function

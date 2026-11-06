"""Shared pytest configuration and fixtures for the edgeofpy test suite."""

import matplotlib

matplotlib.use('Agg') # non-interactive backend: no windows during tests

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture
def rng():
    """A seeded random number generator, so that tests are deterministic."""
    return np.random.default_rng(12345)


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt.close('all')

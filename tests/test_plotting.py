"""Tests for the plotting functions (plot_pdf, plot_cdf, plot_sk)."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

import edgeofpy as eop


@pytest.fixture(autouse=True)
def forbid_show(monkeypatch):
    """Library functions must return figures, never call plt.show()."""
    def fail(*args, **kwargs):
        raise AssertionError('plt.show() was called')
    monkeypatch.setattr(plt, 'show', fail)
    monkeypatch.setattr(plt.Figure, 'show', fail)


# -------------------------------------------------------------------- plot_pdf

def test_plot_pdf_discrete_bins():
    # regression: 'd_lin' merged the two largest values into one bin
    fig = eop.plot_pdf([1, 2, 3, 3])
    x, y = fig.axes[0].lines[0].get_data()
    np.testing.assert_array_equal(x, [1, 2, 3])
    np.testing.assert_allclose(y, [0.25, 0.25, 0.5])


def test_plot_pdf_log_bins(rng):
    data = (1 - rng.random(5000)) ** (-1 / 1.5)
    fig = eop.plot_pdf(data, binning='log', n_bins=30)
    x, y = fig.axes[0].lines[0].get_data()
    assert 0 < len(x) <= 30
    assert np.all(np.diff(x) > 0) and np.all(y > 0)


def test_plot_pdf_log_bins_require_positive_data():
    with pytest.raises(ValueError):
        eop.plot_pdf([0, 1, 2], binning='log')


def test_plot_pdf_invalid_binning():
    with pytest.raises(ValueError):
        eop.plot_pdf([1, 2, 3], binning='quadratic')


def test_plot_pdf_into_existing_axes():
    fig, ax = plt.subplots()
    out = eop.plot_pdf([1, 2, 2, 3], ax=ax, xlabel='size')
    assert out is fig
    assert ax.get_xlabel() == 'size' and ax.get_ylabel() == 'P(size)'


# -------------------------------------------------------------------- plot_cdf

def test_plot_cdf_orientation():
    # regression: probabilities were plotted on the x-axis
    fig = eop.plot_cdf([3, 1, 2])
    x, y = fig.axes[0].lines[0].get_data()
    np.testing.assert_array_equal(x, [1, 2, 3])
    np.testing.assert_allclose(y, [1 / 3, 2 / 3, 1])
    assert fig.axes[0].get_ylabel() == 'P(X <= x)'


def test_plot_cdf_complementary():
    fig = eop.plot_cdf([3, 1, 2], complementary=True)
    np.testing.assert_allclose(fig.axes[0].lines[0].get_ydata(),
                               [1, 2 / 3, 1 / 3])
    assert fig.axes[0].get_ylabel() == 'P(X >= x)'


def test_plot_cdf_single_value():
    # regression: a single value divided by zero
    fig = eop.plot_cdf([5])
    np.testing.assert_allclose(fig.axes[0].lines[0].get_ydata(), [1])


# --------------------------------------------------------------------- plot_sk

def test_plot_sk_orientation():
    # regression: the [mu, sigma]-indexed diagram was drawn transposed
    mu = np.array([0., .1, .2, .3])
    sd = np.array([.1, .2])
    order_param = np.arange(8).reshape(4, 2) # order_param[i_mu, j_sd]
    fig, ax = eop.plot_sk(order_param, mu, sd)
    image = ax.images[0].get_array()
    assert image.shape == (2, 4)              # rows: std, columns: mu
    assert image[0, 3] == order_param[3, 0]   # mu = 0.3, std = 0.1
    np.testing.assert_allclose(ax.images[0].get_extent(),
                               [-0.05, 0.35, 0.05, 0.25])
    assert ax.get_xlabel() == 'mean(J)' and ax.get_ylabel() == 'std(J)'


def test_plot_sk_working_point_uses_off_diagonal_elements():
    J = np.array([[0., 1., 2.], [1., 0., 3.], [2., 3., 0.]])
    _, ax = eop.plot_sk(np.zeros((2, 2)), np.array([0., 1.]),
                        np.array([0., 1.]), J=J)
    x, y = ax.lines[0].get_data()
    off_diag = J[~np.eye(3, dtype=bool)]
    assert x[0] == pytest.approx(off_diag.mean())
    assert y[0] == pytest.approx(off_diag.std())


def test_plot_sk_into_existing_axes():
    fig, ax = plt.subplots()
    out_fig, out_ax = eop.plot_sk(np.zeros((2, 2)), np.array([0., 1.]),
                                  np.array([0., 1.]), ax=ax)
    assert out_fig is fig and out_ax is ax

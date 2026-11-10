"""Tests for estimate_dcc, fit_shape_collapse and plot_shape_collapse."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest

import edgeofpy as eop
from edgeofpy import models


def quiet(func, *args, **kwargs):
    """Call func without the numerical warnings emitted by powerlaw."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return func(*args, **kwargs)


def sizes_durations(avls):
    return (np.array([a['size'] for a in avls]),
            np.array([a['dur_bin'] for a in avls]))


@pytest.fixture(scope='module')
def critical_avls():
    act = models.branching_network(n_units=1000, sigma=1.0, n_steps=300_000,
                                   n_channels=32, rand_seed=1)
    return eop.detect_avalanches(act, s_freq=1000, max_iei=0.001)[0]


@pytest.fixture(scope='module')
def subcritical_avls():
    act = models.branching_network(n_units=1000, sigma=0.9, n_steps=300_000,
                                   n_channels=32, rand_seed=1)
    return eop.detect_avalanches(act, s_freq=1000, max_iei=0.001)[0]


# --------------------------------------------------------------- estimate_dcc

def test_estimate_dcc_matches_manual_pipeline(critical_avls):
    s, d = sizes_durations(critical_avls)
    res = quiet(eop.estimate_dcc, s, d)
    fit_s = quiet(eop.fit_powerlaw, s)
    fit_d = quiet(eop.fit_powerlaw, d)
    third = eop.fit_third_exponent(s, d,
                                   dur_range=(fit_d['xmin'], fit_d['xmax']))
    assert res['tau'] == pytest.approx(fit_s['power_law_exp'])
    assert res['alpha'] == pytest.approx(fit_d['power_law_exp'])
    assert res['third'] == pytest.approx(third)
    assert res['dcc'] == pytest.approx(eop.dcc(res['tau'], res['alpha'],
                                               third))
    assert res['dcc'] == pytest.approx(abs(res['dcc_signed']))
    assert res['size_range'] == (fit_s['xmin'], fit_s['xmax'])
    assert res['n_avalanches'] == len(s)
    assert res['dcc_ci'] is None and res['boot'] is None
    assert res['dcc_xmin_grid'] is None
    assert isinstance(res['powerlaw_version'], str)


def test_estimate_dcc_fixed_ranges(critical_avls):
    s, d = sizes_durations(critical_avls)
    res = quiet(eop.estimate_dcc, s, d, size_range=(5, 500),
                dur_range=(4, None))
    assert res['size_range'] == (5, 500) and res['dur_range'] == (4, None)
    assert res['n_tail_sizes'] == np.sum((s >= 5) & (s <= 500))
    assert res['n_tail_durations'] == np.sum(d >= 4)
    assert res['size_decades'] == pytest.approx(2)
    assert res['dur_decades'] == pytest.approx(np.log10(d.max() / 4))
    third = eop.fit_third_exponent(s, d, dur_range=(4, None))
    assert res['third'] == pytest.approx(third)


def test_estimate_dcc_critical_vs_subcritical(critical_avls,
                                              subcritical_avls):
    crit = quiet(eop.estimate_dcc, *sizes_durations(critical_avls),
                 n_boot=100, rand_seed=0)
    sub = quiet(eop.estimate_dcc, *sizes_durations(subcritical_avls),
                n_boot=100, rand_seed=0)
    low, high = crit['dcc_signed_ci']
    assert low < 0 < high              # consistent with criticality
    assert sub['dcc_signed_ci'][0] > 0  # not consistent with criticality
    assert crit['dcc'] < 0.1 < sub['dcc']
    assert crit['dcc_ci'][0] <= crit['dcc'] <= crit['dcc_ci'][1]
    assert crit['boot']['dcc'].shape == (100,)


def test_estimate_dcc_bootstrap_reproducible(critical_avls):
    s, d = sizes_durations(critical_avls)
    kw = dict(size_range=(2, None), dur_range=(3, None), n_boot=20)
    a = quiet(eop.estimate_dcc, s, d, rand_seed=3, **kw)
    b = quiet(eop.estimate_dcc, s, d, rand_seed=3, **kw)
    c = quiet(eop.estimate_dcc, s, d, rand_seed=4, **kw)
    np.testing.assert_array_equal(a['boot']['dcc'], b['boot']['dcc'])
    assert not np.array_equal(a['boot']['dcc'], c['boot']['dcc'])


def test_estimate_dcc_n_avalanches_all_equals_full_fit(critical_avls):
    # subsets of all avalanches (without replacement) are the full data
    s, d = sizes_durations(critical_avls)
    kw = dict(size_range=(2, None), dur_range=(3, None))
    full = quiet(eop.estimate_dcc, s, d, **kw)
    sub = quiet(eop.estimate_dcc, s, d, n_avalanches=len(s), n_subsamples=3,
                **kw)
    for key in ('tau', 'alpha', 'third', 'dcc', 'dcc_signed'):
        assert sub[key] == pytest.approx(full[key])


def test_estimate_dcc_n_avalanches(critical_avls):
    s, d = sizes_durations(critical_avls)
    res = quiet(eop.estimate_dcc, s, d, size_range=(2, None),
                dur_range=(3, None), n_avalanches=500, n_subsamples=20,
                n_boot=30)
    assert res['n_avalanches'] == 500
    assert res['boot']['dcc'].shape == (30,)
    # the tail counts and fits still describe all avalanches
    assert res['n_tail_sizes'] == np.sum(s >= 2)


def test_estimate_dcc_xmin_sensitivity(critical_avls):
    s, d = sizes_durations(critical_avls)
    res = quiet(eop.estimate_dcc, s, d, size_range=(4, None),
                dur_range=(4, None), xmin_sensitivity=True)
    grid = res['dcc_xmin_grid']
    assert grid.shape == (3, 3)
    assert grid[1, 1] == pytest.approx(res['dcc'])
    assert res['dcc_xmin_range'] == (np.nanmin(grid), np.nanmax(grid))
    manual = quiet(eop.estimate_dcc, s, d, size_range=(2, None),
                   dur_range=(8, None))
    assert grid[0, 2] == pytest.approx(manual['dcc'])  # xmin x0.5, x2


def test_estimate_dcc_warns_few_avalanches_in_range():
    rng = np.random.default_rng(0)
    s = np.floor((1 - rng.random(80)) ** (-1 / 0.5)).astype(int)
    d = np.floor((1 - rng.random(80)) ** (-1)).astype(int)
    with pytest.warns(UserWarning, match='within the fitted range'):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            eop.estimate_dcc(s, d, size_range=(1, None), dur_range=(1, None))


def test_estimate_dcc_warns_short_range(critical_avls):
    s, d = sizes_durations(critical_avls)
    with pytest.warns(UserWarning, match='decades'):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            eop.estimate_dcc(s, d, size_range=(2, None), dur_range=(3, 20))


def test_estimate_dcc_no_warning_for_good_data(critical_avls):
    s, d = sizes_durations(critical_avls)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        eop.estimate_dcc(s, d, size_range=(2, None), dur_range=(3, None))
    ours = [w for w in caught if 'fitted range' in str(w.message)]
    assert ours == []


@pytest.mark.parametrize('kwargs', [dict(n_avalanches=10 ** 6),
                                    dict(n_avalanches=1), dict(ci=1.5)])
def test_estimate_dcc_invalid_arguments(critical_avls, kwargs):
    s, d = sizes_durations(critical_avls)
    with pytest.raises(ValueError):
        quiet(eop.estimate_dcc, s, d, **kwargs)


def test_estimate_dcc_mismatched_lengths():
    with pytest.raises(ValueError):
        eop.estimate_dcc([1, 2, 3], [1, 2])


# --------------------------------------------------------- fit_shape_collapse

def scaled_avalanches(gamma, durations=range(3, 40), n_per_dur=10):
    """Avalanches whose profiles scale exactly as T**gamma * (1 + t/T), so
    that the sizes scale as T**(gamma + 1)."""
    avls = []
    for T in durations:
        profile = T ** gamma * (1 + np.linspace(0, 1, T))
        avls += [{'dur_bin': T, 'profile': profile, 'size': profile.sum()}
                 for _ in range(n_per_dur)]
    return avls


@pytest.mark.parametrize('gamma', [0.5, 1.0, 1.5])
def test_fit_shape_collapse_exact_scaling(gamma):
    res = eop.fit_shape_collapse(scaled_avalanches(gamma),
                                 gamma_range=(-1., 3.), n_levels=3)
    assert res['gamma'] == pytest.approx(gamma, abs=0.01)
    assert res['third'] == pytest.approx(gamma + 1, abs=1e-6)
    assert res['dcc_collapse'] == pytest.approx(0, abs=0.01)
    assert res['collapse_error'] == pytest.approx(0, abs=1e-6)
    # rescaled profiles coincide
    profiles = np.array(list(res['profiles'].values()))
    np.testing.assert_allclose(profiles - profiles[0], 0, atol=0.02)


def test_fit_shape_collapse_matches_lattice_search():
    avls = scaled_avalanches(1.0)
    res = eop.fit_shape_collapse(avls, gamma_range=(0., 3.), step=0.1,
                                 n_levels=2, min_dur_check=None)
    gamma, err = eop.lattice_search(eop.shape_collapse_error,
                                    [avls, (0., 3.)], 1, 0.1, n_levels=2)
    assert res['gamma'] == pytest.approx(gamma)
    assert res['collapse_error'] == pytest.approx(err)
    assert res['gamma_by_min_dur'] == {}


def test_fit_shape_collapse_durations_used():
    avls = scaled_avalanches(1.0, durations=range(2, 12), n_per_dur=10)
    avls += scaled_avalanches(1.0, durations=[20], n_per_dur=5)  # too rare
    res = eop.fit_shape_collapse(avls, min_dur=4, min_avl=10)
    np.testing.assert_array_equal(res['durations'], np.arange(4, 12))
    assert res['n_avalanches_used'] == 80
    assert list(res['profiles']) == list(range(4, 12))
    assert res['x'].shape == (1000,)


def test_fit_shape_collapse_min_dur_check():
    avls = scaled_avalanches(1.0, durations=range(3, 10))
    res = eop.fit_shape_collapse(avls, min_dur_check=(3, 5, 9))
    assert res['gamma_by_min_dur'][3] == pytest.approx(1, abs=0.01)
    assert res['gamma_by_min_dur'][5] == pytest.approx(1, abs=0.01)
    assert np.isnan(res['gamma_by_min_dur'][9])  # only one duration left


def test_fit_shape_collapse_bootstrap(critical_avls):
    res = eop.fit_shape_collapse(critical_avls, n_boot=10, rand_seed=0)
    low, high = res['gamma_ci']
    assert low <= high and res['boot']['gamma'].shape == (10,)
    assert np.all(np.abs(res['boot']['gamma'] - res['gamma']) <= 0.5)
    again = eop.fit_shape_collapse(critical_avls, n_boot=10, rand_seed=0)
    np.testing.assert_array_equal(res['boot']['gamma'],
                                  again['boot']['gamma'])


def test_fit_shape_collapse_critical_branching(critical_avls):
    # gamma and third - 1 agree when fitted over the same durations
    res = eop.fit_shape_collapse(critical_avls)
    assert res['dcc_collapse'] < 0.05
    assert 0.5 < res['gamma'] < 1.1


def test_fit_shape_collapse_too_few_durations():
    with pytest.raises(ValueError):
        eop.fit_shape_collapse(scaled_avalanches(1.0, durations=[5]))


# -------------------------------------------------------- plot_shape_collapse

def test_plot_shape_collapse():
    res = eop.fit_shape_collapse(scaled_avalanches(1.0, durations=range(3, 9)))
    fig = eop.plot_shape_collapse(res)
    ax = fig.axes[0]
    assert len(ax.lines) == 6
    np.testing.assert_allclose(ax.lines[0].get_xdata(), res['x'])
    np.testing.assert_allclose(ax.lines[0].get_ydata(), res['profiles'][3])
    assert len(fig.axes) == 2  # with colorbar


def test_plot_shape_collapse_uncollapsed():
    avls = scaled_avalanches(1.0, durations=range(3, 9))
    res = eop.fit_shape_collapse(avls)
    np.testing.assert_allclose(res['raw_profiles'][5], avls[20]['profile'])
    fig = eop.plot_shape_collapse(res, collapsed=False)
    ax = fig.axes[0]
    assert len(ax.lines) == 6
    np.testing.assert_array_equal(ax.lines[2].get_xdata(), np.arange(5))
    np.testing.assert_allclose(ax.lines[2].get_ydata(), res['raw_profiles'][5])
    assert ax.get_xlabel() == 't (bins)'


def test_plot_shape_collapse_uncollapsed_in_ms():
    res = eop.fit_shape_collapse(scaled_avalanches(1.0, durations=range(3, 9)))
    fig = eop.plot_shape_collapse(res, collapsed=False, bin_dur=0.004)
    ax = fig.axes[0]
    np.testing.assert_allclose(ax.lines[2].get_xdata(), [0, 4, 8, 12, 16])
    assert ax.get_xlabel() == 't (ms)'
    # bin_dur does not affect the collapsed plot
    fig2 = eop.plot_shape_collapse(res, bin_dur=0.004)
    np.testing.assert_allclose(fig2.axes[0].lines[0].get_xdata(), res['x'])


def test_plot_shape_collapse_side_by_side():
    res = eop.fit_shape_collapse(scaled_avalanches(1.0, durations=range(3, 9)))
    fig, (ax1, ax2) = plt.subplots(1, 2)
    eop.plot_shape_collapse(res, collapsed=False, ax=ax1, colorbar=False)
    eop.plot_shape_collapse(res, ax=ax2)
    assert len(ax1.lines) == len(ax2.lines) == 6
    assert len(fig.axes) == 3  # two panels and one colorbar


def test_plot_shape_collapse_into_axes():
    res = eop.fit_shape_collapse(scaled_avalanches(1.0, durations=range(3, 9)))
    fig, ax = plt.subplots()
    out = eop.plot_shape_collapse(res, ax=ax, colorbar=False, cmap='magma')
    assert out is fig and len(fig.axes) == 1 and len(ax.lines) == 6


# -------------------------------------------------------------- lattice_search

def test_lattice_search_lives_in_utils():
    from edgeofpy.utils import lattice_search
    assert eop.lattice_search is lattice_search

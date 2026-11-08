"""Tests for avalanche statistics: power-law fits, scaling relations, shape
collapse, branching ratios and related measures."""

import warnings

import numpy as np
import pytest

import edgeofpy as eop
from helpers import branching_counts


def quiet_fit_powerlaw(*args, **kwargs):
    """fit_powerlaw, without the numerical warnings emitted by powerlaw."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return eop.fit_powerlaw(*args, **kwargs)


# --------------------------------------------------------------- fit_powerlaw

def test_fit_powerlaw_recovers_exponent(rng):
    alpha, xmin = 2.5, 1.0
    data = xmin * (1 - rng.random(10000)) ** (-1 / (alpha - 1)) # Pareto
    res = quiet_fit_powerlaw(data, xmin='min', discrete=False)
    assert res['power_law_exp'] == pytest.approx(alpha, abs=0.05)
    assert res['xmin'] == pytest.approx(data.min())


def test_fit_powerlaw_default_xmin_is_automatic(rng):
    # power law (alpha = 2.5) above 5, plus a non-power-law head below 5
    tail = 5 * (1 - rng.random(4000)) ** (-1 / 1.5)
    head = rng.uniform(1, 5, 2000)
    data = np.concatenate((head, tail))
    auto = quiet_fit_powerlaw(data, discrete=False)
    assert auto['xmin'] == pytest.approx(5, rel=0.2)
    assert auto['power_law_exp'] == pytest.approx(2.5, abs=0.1)
    # starting at the minimum includes the head and biases the exponent
    from_min = quiet_fit_powerlaw(data, xmin='min', discrete=False)
    assert abs(from_min['power_law_exp'] - 2.5) > 0.3


def test_fit_powerlaw_returns_documented_keys(rng):
    data = np.floor((1 - rng.random(5000)) ** (-1 / 1.5)).astype(int)
    res = quiet_fit_powerlaw(data, xmax='max')
    keys = ['fit', 'power_law_exp', 'xmin', 'xmax', 'power_law_sigma',
            'trunc_pl_exp', 'trunc_pl_xmin', 'trunc_pl_xmax', 'PvE', 'PvL',
            'PvT', 'TvE', 'TvL', 'EvL', 'best_fit', 'best_fit_R_sum',
            'P_R_sum', 'T_R_sum', 'E_R_sum', 'L_R_sum']
    assert set(keys) <= set(res)
    assert res['best_fit'] in 'PTEL'
    assert res['xmax'] == data.max()


# --------------------------------------------------------- fit_third_exponent

@pytest.mark.parametrize('method', ['mean', 'all'])
def test_fit_third_exponent_exact_scaling(method):
    T = np.repeat(np.arange(1, 40), 5).astype(float)
    S = 3 * T ** 1.6
    assert eop.fit_third_exponent(S, T, method=method) == pytest.approx(1.6)


def test_fit_third_exponent_noisy(rng):
    T = rng.integers(1, 100, 5000).astype(float)
    S = 2 * T ** 2 * rng.lognormal(0, 0.3, T.size)
    assert eop.fit_third_exponent(S, T) == pytest.approx(2, abs=0.05)


@pytest.mark.parametrize('dur_range, mask', [
    ((10, 50), lambda T: (T >= 10) & (T <= 50)),
    ((10, None), lambda T: T >= 10),
    ((None, 50), lambda T: T <= 50),
    ((None, None), lambda T: np.ones(T.size, dtype=bool)),
])
def test_fit_third_exponent_dur_range(rng, dur_range, mask):
    T = rng.integers(1, 100, 3000).astype(float)
    S = T ** 2 * rng.lognormal(0, 0.3, T.size)
    m = mask(T)
    expected = eop.fit_third_exponent(S[m], T[m])
    assert eop.fit_third_exponent(S, T, dur_range=dur_range) \
        == pytest.approx(expected)


def test_fit_third_exponent_dur_range_matters():
    # a different scaling below T = 10: restricting the range recovers 2
    T = np.repeat(np.arange(1, 100), 3).astype(float)
    S = np.where(T < 10, T ** 1.2, T ** 2 * 10 ** -0.8)
    assert eop.fit_third_exponent(S, T, dur_range=(10, None)) \
        == pytest.approx(2)
    assert eop.fit_third_exponent(S, T) < 1.9


@pytest.mark.parametrize('dur_range', [(50, 10), (200, None)])
def test_fit_third_exponent_invalid_dur_range(dur_range):
    T = np.arange(1, 100, dtype=float)
    with pytest.raises(ValueError):
        eop.fit_third_exponent(T ** 2, T, dur_range=dur_range)


@pytest.mark.parametrize('old, new', [('pl', 'mean'), ('lsf', 'all')])
def test_fit_third_exponent_deprecated_methods(old, new, rng):
    # regression: 'pl' fitted a distribution; 'lsf' had S and T swapped
    T = rng.integers(1, 50, 500).astype(float)
    S = T ** 1.5 * rng.lognormal(0, 0.2, T.size)
    with pytest.warns(DeprecationWarning):
        third = eop.fit_third_exponent(S, T, method=old)
    assert third == pytest.approx(eop.fit_third_exponent(S, T, method=new))


@pytest.mark.parametrize('sizes, durations, kwargs', [
    ([1, 2, 3], [1, 2, 3], {'method': 'bad'}),
    ([1, 2, 3], [1, 2], {}),            # shape mismatch
    ([1, 0, 3], [1, 2, 3], {}),         # non-positive size
    ([1, 2, 3], [2, 2, 2], {}),         # a single distinct duration
])
def test_fit_third_exponent_invalid_input(sizes, durations, kwargs):
    with pytest.raises(ValueError):
        eop.fit_third_exponent(sizes, durations, **kwargs)


# ------------------------------------------------------------ dcc and friends

def test_dcc_mean_field_is_zero():
    # tau = 3/2, alpha = 2, third = 2 satisfy (alpha - 1) / (tau - 1) = third
    assert eop.dcc(1.5, 2.0, 2.0) == pytest.approx(0)
    assert eop.dcc(2.0, 3.0, 1.0) == pytest.approx(1)


def test_dcc_collapse():
    assert eop.dcc_collapse(1.0, 2.0) == pytest.approx(0)
    assert eop.dcc_collapse(0.5, 2.0) == pytest.approx(0.5)


def test_dcc_collapse_2():
    assert eop.dcc_collapse_2(1.5, 2.0, 1.0) == pytest.approx(0)
    assert eop.dcc_collapse_2(1.5, 2.0, 0.5) == pytest.approx(0.5)


# ------------------------------------------------------- shape_collapse_error

def scaled_profiles(gamma, durations=range(3, 40), n_per_dur=10):
    """Avalanches whose profiles scale exactly as T**gamma * (1 + t/T).

    The shape is linear, so the linear interpolation in shape_collapse_error
    reproduces it exactly and the collapse error at the true gamma is 0.
    """
    return [{'dur_bin': T,
             'profile': T ** gamma * (1 + np.linspace(0, 1, T))}
            for T in durations for _ in range(n_per_dur)]


@pytest.mark.parametrize('gamma', [0.5, 1.0, 1.5])
def test_shape_collapse_minimum_at_true_gamma(gamma):
    # regression: profiles were max-normalized, multiplied by T**gamma and
    # interpolated over the first sample interval only
    avls = scaled_profiles(gamma)
    found, err = eop.lattice_search(eop.shape_collapse_error,
                                    [avls, (-1., 3.)], 1, 0.5, n_levels=3)
    assert found == pytest.approx(gamma, abs=0.01)
    assert err == pytest.approx(0, abs=1e-6)


def test_shape_collapse_error_increases_away_from_gamma():
    avls = scaled_profiles(1.0)
    errors = [eop.shape_collapse_error(avls, g) for g in (0.0, 0.5, 1.0)]
    assert errors[0] > errors[1] > errors[2] >= 0


def test_shape_collapse_integer_gamma():
    # regression: a negative integer gamma raised a ValueError
    assert np.isfinite(eop.shape_collapse_error(scaled_profiles(1.0), -1))


def test_shape_collapse_too_few_durations():
    avls = scaled_profiles(1.0, durations=[5, 6], n_per_dur=10)
    eop.shape_collapse_error(avls, 1.0) # two durations: fine
    with pytest.raises(ValueError):
        eop.shape_collapse_error(avls, 1.0, min_avl=11)
    with pytest.raises(ValueError):
        eop.shape_collapse_error(avls, 1.0, min_dur=6)


# ------------------------------------------------------------- lattice_search

def test_lattice_search_minimum():
    def f(x):
        return (x - 0.123) ** 2
    x, y = eop.lattice_search(f, [(0., 1.)], 0, 0.1, n_levels=3)
    assert x == pytest.approx(0.123, abs=1e-4)
    assert y == pytest.approx(0, abs=1e-8)


def test_lattice_search_maximum():
    def f(a, x):
        return a - (x - 0.3) ** 2
    x, y = eop.lattice_search(f, [2.0, (0., 1.)], 1, 0.1, criterion='max')
    assert x == pytest.approx(0.3, abs=1e-4)
    assert y == pytest.approx(2.0)


def test_lattice_search_does_not_modify_arguments():
    # regression: the range tuple in func_args was replaced by a float
    args = [(0., 1.)]
    eop.lattice_search(lambda x: x ** 2, args, 0, 0.1)
    assert args == [(0., 1.)]


def test_lattice_search_includes_end_but_not_beyond():
    calls = []

    def f(x):
        calls.append(x)
        return (x - 2) ** 2   # minimum outside the range
    x, _ = eop.lattice_search(f, [(0., 1.)], 0, 0.1, n_levels=0)
    assert len(calls) == 11   # 0.0, 0.1, ..., 1.0
    assert max(calls) <= 1.0
    assert x == pytest.approx(1.0)


def test_lattice_search_invalid_criterion():
    with pytest.raises(ValueError):
        eop.lattice_search(lambda x: x, [(0., 1.)], 0, 0.1, criterion='mid')


# ---------------------------------------------------------- branching ratios

def test_avl_branching_ratio_critical_profiles():
    # regression: spurious 1/(n_bin-1) and 1/N factors gave 0.033 here
    avls = [{'profile': np.array([2, 2, 2, 2]), 'dur_bin': 4}] * 10
    assert eop.avl_branching_ratio(avls) == pytest.approx(1.0)


def test_avl_branching_ratio_geometric_mean():
    avls = ([{'profile': np.array([2, 2, 2, 2]), 'dur_bin': 4}] * 5
            + [{'profile': np.array([1, 2, 4]), 'dur_bin': 3}] * 5)
    assert eop.avl_branching_ratio(avls) == pytest.approx(np.sqrt(2))


def test_avl_branching_ratio_no_multibin_avalanches():
    avls = [{'profile': np.array([3]), 'dur_bin': 1}]
    with pytest.warns(RuntimeWarning):
        assert np.isnan(eop.avl_branching_ratio(avls))


@pytest.mark.parametrize('counts, expected', [
    ([3, 3, 3, 3], 1.0),
    ([1, 2, 4, 8], 2.0),
])
def test_branching_ratio(counts, expected):
    assert eop.branching_ratio(np.array(counts)) == pytest.approx(expected)


def test_branching_ratio_with_time_bins():
    events = np.array([1, 1, 2, 2, 4, 4, 8, 8]) # bins of 2 -> 2, 4, 8, 16
    assert eop.branching_ratio(events, time_bin_size=0.002, s_freq=1000) \
        == pytest.approx(2.0)


def test_branching_ratio_bins_require_s_freq():
    with pytest.raises(ValueError, match='s_freq'):
        eop.branching_ratio(np.ones(10), time_bin_size=0.01)


# ----------------------------------------- susceptibility, Fano factor, etc.

def test_susceptibility(rng):
    events = (rng.random((20, 1000)) < 0.1).astype(int)
    density = events.mean(axis=0)
    chi, ltf = eop.susceptibility(events)
    assert chi == pytest.approx(20 * np.var(density))
    assert ltf == pytest.approx(np.std(density) / np.mean(density))


def test_susceptibility_1d_density(rng):
    density = rng.random(500)
    chi, _ = eop.susceptibility(density)
    assert chi == pytest.approx(np.var(density))


def test_fano_factor(rng):
    assert eop.fano_factor(rng.poisson(5, 200_000)) == pytest.approx(1,
                                                                     abs=0.02)
    assert eop.fano_factor(np.full(100, 3)) == 0


# ------------------------------------------------ repertoire and dissimilarity

def test_avl_repertoire():
    avls = [{'pattern': np.array(p)} for p in
            ([1, 0, 1], [1, 0, 1], [0, 1, 1], [1, 0, 1])]
    np.testing.assert_array_equal(eop.avl_repertoire(avls),
                                  [[0, 1, 1], [1, 0, 1]])


@pytest.mark.xfail(strict=True, reason='known issue: single-channel '
                   "avalanches have pattern 'N/A'; should raise clearly")
def test_avl_repertoire_single_channel_raises():
    avls = [{'pattern': 'N/A'}, {'pattern': 'N/A'}]
    with pytest.raises(ValueError):
        eop.avl_repertoire(avls)


def test_avl_pattern_dissimilarity():
    rep = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 1]])
    hamming = np.array([[0, 2, 1], [2, 0, 1], [1, 1, 0]])
    np.testing.assert_allclose(eop.avl_pattern_dissimilarity(rep), hamming)
    np.testing.assert_allclose(eop.avl_pattern_dissimilarity(rep, norm=True),
                               hamming / 3)
    np.testing.assert_allclose(
        eop.avl_pattern_dissimilarity(rep, similarity=True), 1 - hamming / 3)


# ----------------------------------------------------------------- end to end

@pytest.fixture(scope='module')
def branching_avalanches():
    """Avalanches detected in critical (sigma = 1) and subcritical (0.8)
    Galton-Watson branching processes."""
    out = {}
    for sigma in (1.0, 0.8):
        counts = branching_counts(sigma, 20000, np.random.default_rng(1))
        avls = eop.detect_avalanches(counts[np.newaxis, :], s_freq=1000,
                                     max_iei=0.001)[0]
        out[sigma] = (counts, avls)
    return out


def exponents(avls):
    sizes = np.array([a['size'] for a in avls])
    durs = np.array([a['dur_bin'] for a in avls])
    tau = quiet_fit_powerlaw(sizes)['power_law_exp'] # xmin: automatic
    fit_d = quiet_fit_powerlaw(durs)
    alpha = fit_d['power_law_exp']
    third = eop.fit_third_exponent(sizes, durs,
                                   dur_range=(fit_d['xmin'], fit_d['xmax']))
    return tau, alpha, third


@pytest.mark.slow
def test_end_to_end_critical_branching_process(branching_avalanches):
    counts, avls = branching_avalanches[1.0]
    assert len(avls) > 19000
    tau, alpha, third = exponents(avls)
    # mean-field values: tau = 1.5, alpha = 2, third = 2
    assert tau == pytest.approx(1.5, abs=0.1)
    assert alpha == pytest.approx(2.0, abs=0.15)
    assert third == pytest.approx(2.0, abs=0.1)
    assert eop.dcc(tau, alpha, third) < 0.15
    assert eop.branching_ratio(counts) == pytest.approx(1.0, abs=0.02)

    # shape collapse: gamma = third - 1 = 1 asymptotically; finite durations
    # bias the estimate downwards
    def collapse_error(avalanches, gamma):
        return eop.shape_collapse_error(avalanches, gamma, min_dur=10)
    gamma, _ = eop.lattice_search(collapse_error, [avls, (0., 2.)], 1, 0.5,
                                  n_levels=2)
    assert 0.7 < gamma < 1.1
    assert eop.dcc_collapse(gamma, third) < 0.2


@pytest.mark.slow
def test_end_to_end_subcritical_has_larger_dcc(branching_avalanches):
    dcc_crit = eop.dcc(*exponents(branching_avalanches[1.0][1]))
    tau, alpha, third = exponents(branching_avalanches[0.8][1])
    assert tau > 1.8 # steeper than critical
    assert eop.dcc(tau, alpha, third) > dcc_crit

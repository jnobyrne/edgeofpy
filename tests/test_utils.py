"""Tests for edgeofpy.utils."""

import numpy as np
import pytest

import edgeofpy as eop

from edgeofpy.utils import (_detect_start_end, _sec_to_samples,
                            binarized_events, gaussian_threshold,
                            phase_randomize, time_bin_events)


# ---------------------------------------------------------- _detect_start_end

@pytest.mark.parametrize('values, expected', [
    ([0, 1, 1, 0, 0, 1, 0], [[1, 3], [5, 6]]),
    ([1, 1, 0], [[0, 2]]),        # run at the start
    ([0, 0, 1], [[2, 3]]),        # regression: a final True sample was lost
    ([1, 1, 1], [[0, 3]]),        # regression: end was 2 instead of 3
    ([1], [[0, 1]]),
])
def test_detect_start_end(values, expected):
    out = _detect_start_end(np.array(values, dtype=bool))
    np.testing.assert_array_equal(out, expected)


def test_detect_start_end_no_true_values():
    assert _detect_start_end(np.zeros(5, dtype=bool)) is None


# ----------------------------------------------------------- _sec_to_samples

def test_sec_to_samples_rounds():
    # regression: int() truncation gave int(0.29 * 100) == 28
    assert _sec_to_samples(0.29, 100) == 29
    assert _sec_to_samples(0.004, 1000) == 4


def test_sec_to_samples_less_than_one_sample():
    with pytest.raises(ValueError, match='max_iei'):
        _sec_to_samples(0.004, 100, name='max_iei')


# ----------------------------------------------------------- time_bin_events

def test_time_bin_events_pads_last_bin():
    out = time_bin_events(np.array([1, 0, 2, 1, 1]), 2)
    np.testing.assert_array_equal(out, [1, 3, 1])
    assert np.issubdtype(out.dtype, np.integer)


def test_time_bin_events_exact_multiple():
    np.testing.assert_array_equal(time_bin_events(np.arange(6), 3), [3, 12])


def test_time_bin_events_sums_channels():
    events = np.array([[1, 0, 1], [0, 1, 1]])
    np.testing.assert_array_equal(time_bin_events(events, 2), [2, 2])


def test_time_bin_events_single_channel_2d():
    # regression: (1, T) input was not collapsed to 1d
    np.testing.assert_array_equal(time_bin_events(np.ones((1, 5), int), 2),
                                  [2, 2, 1])


def test_time_bin_events_invalid_bin_len():
    with pytest.raises(ValueError):
        time_bin_events(np.ones(5), 0)


# ---------------------------------------------------------- binarized_events

@pytest.fixture
def excursions():
    """Two channels: a positive excursion peaking at sample 52 (channel 0)
    and a negative one with its minimum at sample 100 (channel 1)."""
    data = np.zeros((2, 200))
    data[0, 50:55] = [3, 6, 10, 6, 3]
    data[1, 99:104] = [-3, -10, -6, -3, -1]
    return data


def test_binarized_events_both(excursions):
    events = binarized_events(excursions, threshold=3, thresh_type='both')
    assert events.shape == excursions.shape
    np.testing.assert_array_equal(np.argwhere(events == 1), [[0, 52], [1, 100]])


@pytest.mark.parametrize('thresh_type, expected', [
    ('above', [[0, 52]]),
    ('below', [[1, 100]]),
])
def test_binarized_events_direction(excursions, thresh_type, expected):
    events = binarized_events(excursions, threshold=3, thresh_type=thresh_type)
    np.testing.assert_array_equal(np.argwhere(events == 1), expected)


def test_binarized_events_one_event_per_excursion_at_peak():
    data = np.zeros((1, 300))
    data[0, 100:105] = [4, 8, 9, 8, 4]  # one excursion, peak at 102
    events = binarized_events(data, threshold=3)
    np.testing.assert_array_equal(np.flatnonzero(events[0]), [102])


def test_binarized_events_last_sample():
    # regression: an excursion in the final sample was not detected
    data = np.zeros((1, 200))
    data[0, -1] = 10
    events = binarized_events(data, threshold=3)
    assert events[0, -1] == 1


def test_binarized_events_null_value(excursions):
    events = binarized_events(excursions, threshold=3, null_value=-1)
    assert set(np.unique(events)) == {-1, 1}
    assert np.sum(events == 1) == 2


# -------------------------------------------------------- gaussian_threshold

def noise_with_bursts(rng, n_chan=10, n_samp=50000, frac=0.005, sign=None):
    """Gaussian noise plus sparse large deflections (4-8 noise SD)."""
    data = rng.standard_normal((n_chan, n_samp))
    hit = rng.random(data.shape) < frac
    amp = rng.uniform(4, 8, hit.sum())
    if sign is None:
        amp *= rng.choice([-1, 1], hit.sum())
    data[hit] += sign * amp if sign is not None else amp
    return data


def test_gaussian_threshold_finds_departure(rng):
    data = noise_with_bursts(rng)
    thr = gaussian_threshold(data)
    assert 2 < thr < 4.5
    assert thr == pytest.approx(round(thr / 0.1) * 0.1)  # multiple of bin


def test_gaussian_threshold_white_noise_warns(rng):
    with pytest.warns(UserWarning, match='does not depart'):
        thr = gaussian_threshold(rng.standard_normal((10, 50000)))
    assert np.isnan(thr)


def test_gaussian_threshold_pools_list(rng):
    a, b = noise_with_bursts(rng, 4), noise_with_bursts(rng, 6)
    # each channel is z-scored, so a list equals one array of all channels
    assert gaussian_threshold([a, b]) == gaussian_threshold(np.vstack([a, b]))
    # recordings of different lengths and channel counts
    c = noise_with_bursts(rng, 3, 20000)
    thr, det = gaussian_threshold([a, c], return_details=True)
    assert 2 < thr < 4.5
    assert det['density'].sum() * 0.1 == pytest.approx(1, abs=1e-6)


def test_gaussian_threshold_one_sided(rng):
    data = noise_with_bursts(rng, sign=1)
    assert 2 < gaussian_threshold(data, thresh_type='above') < 4.5
    with pytest.warns(UserWarning, match='does not depart'):
        assert np.isnan(gaussian_threshold(data, thresh_type='below'))


def test_gaussian_threshold_tol_monotonic(rng):
    data = noise_with_bursts(rng)
    thr = [gaussian_threshold(data, tol=t) for t in (1.5, 2, 5)]
    assert thr == sorted(thr)


def test_gaussian_threshold_details(rng):
    thr, det = gaussian_threshold(noise_with_bursts(rng), return_details=True)
    assert det['threshold'] == thr
    assert det['gaussian_params'][2] == pytest.approx(1, abs=0.1)  # noise SD
    i = np.flatnonzero(np.isclose(det['ratio_z'], thr))[0]
    assert det['ratio'][i] > det['tol']
    assert len(det['bin_centers']) == len(det['density']) == len(
        det['gaussian']) == 2 * len(det['ratio_z'])


@pytest.mark.filterwarnings('ignore:the amplitude distribution does not')
def test_gaussian_threshold_warns_on_non_gaussian_core(rng):
    bimodal = np.concatenate([rng.normal(-2, .5, (5, 25000)),
                              rng.normal(2, .5, (5, 25000))], axis=1)
    with pytest.warns(UserWarning, match='Gaussian centre'):
        gaussian_threshold(bimodal)
    with pytest.warns(UserWarning, match='centre is not Gaussian'):
        gaussian_threshold(rng.standard_cauchy((5, 50000)))


def test_gaussian_threshold_bad_input(rng):
    with pytest.raises(ValueError, match='thresh_type'):
        gaussian_threshold(rng.standard_normal((2, 100)), thresh_type='up')
    with pytest.raises(ValueError, match='non-zero variance'):
        gaussian_threshold(np.ones((2, 100)))


def test_binarized_events_gaussian_default(rng):
    data = noise_with_bursts(rng)
    np.testing.assert_array_equal(
        binarized_events(data),
        binarized_events(data, threshold=gaussian_threshold(data)))


def test_binarized_events_gaussian_fails_on_noise(rng):
    with pytest.warns(UserWarning), pytest.raises(ValueError,
                                                  match='no departure'):
        binarized_events(rng.standard_normal((5, 20000)))
    with pytest.raises(ValueError, match="number or 'gaussian'"):
        binarized_events(rng.standard_normal((5, 200)), threshold='auto')


def test_plot_gaussian_threshold(rng):
    import matplotlib
    matplotlib.use('Agg')
    _, det = gaussian_threshold(noise_with_bursts(rng), return_details=True)
    ax = eop.plot_gaussian_threshold(det)
    assert len(ax.lines) == 4  # data, Gaussian, two threshold lines


# ----------------------------------------------------------- phase_randomize

def correlated_channels(rng, n_chan=4, n_samp=2000, n_ep=None):
    """Channels mixing a shared, smoothed source with their own noise."""
    shape = (n_samp,) if n_ep is None else (n_samp, n_ep)
    src = np.cumsum(rng.standard_normal(shape), axis=0)
    src -= src.mean(axis=0)
    chans = [(k + 1) * src + rng.standard_normal(shape) for k in range(n_chan)]
    return np.stack(chans)


@pytest.mark.parametrize('shared', [True, False])
@pytest.mark.parametrize('n_samp', [2000, 2001])  # even and odd lengths
def test_phase_randomize_keeps_spectra(rng, shared, n_samp):
    data = correlated_channels(rng, n_samp=n_samp)
    surr = phase_randomize(data, shared=shared, rand_seed=1)
    assert surr.shape == data.shape and np.isrealobj(surr)
    np.testing.assert_allclose(np.abs(np.fft.rfft(surr, axis=1)),
                               np.abs(np.fft.rfft(data, axis=1)),
                               rtol=1e-8, atol=1e-6)
    np.testing.assert_allclose(surr.mean(axis=1), data.mean(axis=1),
                               atol=1e-9)
    assert not np.allclose(surr, data)


def test_phase_randomize_shared_keeps_cross_spectra(rng):
    data = correlated_channels(rng)
    X = np.fft.rfft(data, axis=1)
    cross = X[0] * np.conj(X[1:])
    shared = np.fft.rfft(phase_randomize(data, rand_seed=2), axis=1)
    np.testing.assert_allclose(shared[0] * np.conj(shared[1:]), cross,
                               rtol=1e-7, atol=1e-6)
    np.testing.assert_allclose(np.corrcoef(phase_randomize(data, rand_seed=2)),
                               np.corrcoef(data), atol=1e-10)


def test_phase_randomize_independent_decorrelates(rng):
    data = correlated_channels(rng, n_samp=20000)
    assert np.corrcoef(data)[0, 1] > 0.9
    r = [np.corrcoef(phase_randomize(data, shared=False,
                                     rand_seed=k))[0, 1]
         for k in range(20)]
    assert abs(np.mean(r)) < 0.2


def test_phase_randomize_epochs_and_1d(rng):
    data = correlated_channels(rng, n_samp=500, n_ep=3)
    surr = phase_randomize(data, rand_seed=3)
    assert surr.shape == data.shape
    for e in range(3):  # each epoch keeps its own spectrum
        np.testing.assert_allclose(
            np.abs(np.fft.rfft(surr[:, :, e], axis=1)),
            np.abs(np.fft.rfft(data[:, :, e], axis=1)), rtol=1e-8,
            atol=1e-6)
    x = data[0, :, 0]
    assert phase_randomize(x, rand_seed=3).shape == x.shape


def test_phase_randomize_seed_and_errors(rng):
    data = correlated_channels(rng)
    np.testing.assert_array_equal(phase_randomize(data, rand_seed=5),
                                  phase_randomize(data, rand_seed=5))
    assert not np.allclose(phase_randomize(data, rand_seed=5),
                           phase_randomize(data, rand_seed=6))
    with pytest.raises(ValueError, match='1d, 2d or 3d'):
        phase_randomize(np.zeros((2, 2, 2, 2)))


def test_phase_randomize_is_gaussian(rng):
    # heavy-tailed data give a Gaussian surrogate: no Gaussian threshold
    data = rng.standard_t(3, size=(10, 20000))
    assert 1 < gaussian_threshold(data) < 4
    with pytest.warns(UserWarning, match='does not depart'):
        assert np.isnan(gaussian_threshold(phase_randomize(data,
                                                           rand_seed=0)))


# ---------------------------------------------------------------- bin_spikes

def test_bin_spikes_pooled():
    counts, dt = eop.bin_spikes(np.array([0.0, 0.05, 0.12, 0.13, 0.3]), 0.1)
    assert dt == 0.1
    np.testing.assert_array_equal(counts, [2, 2, 1])
    assert counts.sum() == 5


def test_bin_spikes_per_unit_ragged():
    units = [np.array([0.0, 0.25]), np.array([0.1, 0.11, 0.12])]
    counts, dt = eop.bin_spikes(units, 0.1)
    assert counts.shape == (2, 3)
    np.testing.assert_array_equal(counts.sum(axis=0),
                                  eop.bin_spikes(np.concatenate(units),
                                                 0.1)[0])


def test_bin_spikes_list_of_floats_is_pooled():
    counts, _ = eop.bin_spikes([0.0, 0.1, 0.2], 0.1)
    assert counts.ndim == 1


def test_bin_spikes_mean_isi():
    st = np.array([0.0, 0.002, 0.003, 0.009])
    _, dt = eop.bin_spikes(st, 'mean_isi')
    assert dt == pytest.approx(0.003)


def test_bin_spikes_window():
    counts, _ = eop.bin_spikes(np.array([0.0, 0.5, 1.0, 1.5]), 0.5,
                               t_start=0.5, t_stop=1.5)
    np.testing.assert_array_equal(counts, [1, 2])


@pytest.mark.parametrize('kwargs', [dict(bin_dur=0), dict(bin_dur='x'),
                                    dict(bin_dur=0.1, t_start=1, t_stop=0)])
def test_bin_spikes_errors(kwargs):
    with pytest.raises(ValueError):
        eop.bin_spikes(np.array([0.0, 0.5]), **kwargs)


def test_bin_spikes_refuses_huge_per_unit_array():
    units = [np.array([0.0, 1000.0])] * 1000
    with pytest.raises(ValueError, match='too large'):
        eop.bin_spikes(units, 1e-6)

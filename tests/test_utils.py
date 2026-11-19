"""Tests for edgeofpy.utils."""

import numpy as np
import pytest

import edgeofpy as eop

from edgeofpy.utils import (_detect_start_end, _sec_to_samples,
                            binarized_events, time_bin_events)


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

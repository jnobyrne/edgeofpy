"""Tests for avalanche detection (detect_avalanches and _det_avls)."""

import numpy as np
import pytest

import edgeofpy as eop

S_FREQ = 1000
MAX_IEI = 0.002 # 2 samples at 1000 Hz


@pytest.fixture
def layout():
    """3 channels x 200 samples with three hand-placed avalanches:

    A: t=10 ch0; t=11 ch1, ch2; t=13 ch0  (gap of 2 samples: same avalanche)
    B: t=50 ch2                           (single event)
    C: t=100 ch0, ch1; t=101 ch1
    """
    events = np.zeros((3, 200), dtype=int)
    events[0, 10] = 1
    events[[1, 2], 11] = 1
    events[0, 13] = 1
    events[2, 50] = 1
    events[[0, 1], 100] = 1
    events[1, 101] = 1
    return events


def test_detect_known_layout(layout):
    avls, events_all, events_one, mean_iei = eop.detect_avalanches(
        layout, S_FREQ, max_iei=MAX_IEI)

    assert len(avls) == 3
    a, b, c = avls
    # A: samples 10-13 (end exclusive: 14), profile binned by 2 samples
    assert a['size'] == 4 and a['n_chan'] == 3
    assert a['start_time'] == pytest.approx(0.010)
    assert a['end_time'] == pytest.approx(0.014)
    assert a['dur_sec'] == pytest.approx(0.004)
    assert a['dur_bin'] == 2
    np.testing.assert_array_equal(a['profile'], [3, 1])
    np.testing.assert_array_equal(a['pattern'], [1, 1, 1])
    # B: a single event
    assert b['size'] == 1 and b['dur_bin'] == 1 and b['n_chan'] == 1
    np.testing.assert_array_equal(b['profile'], [1])
    np.testing.assert_array_equal(b['pattern'], [0, 0, 1])
    # C
    assert c['size'] == 3 and c['dur_bin'] == 1
    np.testing.assert_array_equal(c['profile'], [3])
    np.testing.assert_array_equal(c['pattern'], [1, 1, 0])

    np.testing.assert_array_equal(events_all, layout)
    np.testing.assert_array_equal(events_one, layout.sum(axis=0))
    # event times 10, 11, 13, 50, 100, 101 -> intervals 1, 2, 37, 50, 1
    assert mean_iei == pytest.approx(18.2 / S_FREQ)


def test_time_vector_is_used(layout):
    time = 5 + np.arange(layout.shape[1]) / S_FREQ
    avls = eop.detect_avalanches(layout, S_FREQ, time=time,
                                 max_iei=MAX_IEI)[0]
    assert avls[0]['start_time'] == pytest.approx(time[10])
    assert avls[0]['end_time'] == pytest.approx(time[14])


def test_single_complete_avalanche_is_kept():
    # regression: the last avalanche was always discarded, so a recording
    # with one avalanche gave none
    events = np.zeros((3, 200), dtype=int)
    events[:, 100:103] = 1
    avls = eop.detect_avalanches(events, S_FREQ, max_iei=MAX_IEI)[0]
    assert len(avls) == 1
    assert avls[0]['size'] == 9


def test_avalanches_truncated_by_recording_edges_are_dropped(layout):
    events = layout.copy()
    events[0, 1] = 1    # starts within max_iei of the recording start
    events[0, 199] = 1  # ends within max_iei of the recording end
    avls = eop.detect_avalanches(events, S_FREQ, max_iei=MAX_IEI)[0]
    assert [a['size'] for a in avls] == [4, 1, 3]


def test_fewer_than_two_events():
    # regression: _det_avls returned the wrong number of values here
    events = np.zeros((3, 100), dtype=int)
    events[0, 10] = 1
    avls, _, _, mean_iei = eop.detect_avalanches(events, S_FREQ)
    assert avls == []
    assert np.isnan(mean_iei)


@pytest.mark.parametrize('dtype', [bool, np.int32, np.uint8, np.int64])
def test_event_input_is_not_rebinarized(layout, dtype):
    # regression: bool / int32 / uint8 event arrays were treated as raw signal
    avls, events_all, _, _ = eop.detect_avalanches(
        layout.astype(dtype), S_FREQ, max_iei=MAX_IEI)
    assert [a['size'] for a in avls] == [4, 1, 3]
    np.testing.assert_array_equal(events_all, layout)
    assert np.issubdtype(events_all.dtype, np.integer)


def test_raw_signal_input(rng):
    data = rng.standard_normal((8, 5000))
    avls, events_all, events_one, _ = eop.detect_avalanches(data, 250)
    expected_events = eop.binarized_events(data, threshold=2)  # default
    np.testing.assert_array_equal(events_all, expected_events)
    np.testing.assert_array_equal(events_one, expected_events.sum(axis=0))
    assert sum(a['size'] for a in avls) <= expected_events.sum()


def test_profiles_have_dur_bin_length(rng):
    events = (rng.random((10, 20000)) < 0.02).astype(int)
    avls = eop.detect_avalanches(events, S_FREQ, max_iei=0.003)[0]
    assert len(avls) > 100
    for avl in avls:
        assert len(avl['profile']) == avl['dur_bin']
        assert avl['profile'].sum() == avl['size']


def test_max_iei_too_short():
    events = np.zeros((2, 100), dtype=int)
    events[:, [10, 11, 50]] = 1
    with pytest.raises(ValueError, match='max_iei'):
        eop.detect_avalanches(events, 100, max_iei=0.004)


def test_single_channel():
    events = np.zeros((1, 100), dtype=int)
    events[0, [20, 21, 22, 60]] = 1
    avls = eop.detect_avalanches(events, S_FREQ, max_iei=MAX_IEI)[0]
    assert [a['size'] for a in avls] == [3, 1]
    assert avls[0]['pattern'] == 'N/A'

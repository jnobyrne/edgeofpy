"""Tests for edge-of-synchrony measures."""

import numpy as np
import pytest
import scipy.signal

import edgeofpy as eop

FS = 500                       # sampling frequency, Hz
T = np.arange(0, 10, 1 / FS)   # 10 s
FREQ = 10                      # oscillation frequency, Hz


def lagged_sines(lags):
    """One 10 Hz sinusoid per channel, lagging by `lags` (radians)."""
    return np.vstack([np.sin(2 * np.pi * FREQ * T - lag) for lag in lags])


@pytest.fixture
def narrowband_noise(rng):
    """Independent 8-12 Hz band-passed noise on 6 channels."""
    b, a = scipy.signal.butter(4, [8, 12], btype='band', fs=FS)
    return scipy.signal.filtfilt(b, a, rng.standard_normal((6, T.size)),
                                 axis=1)


# ------------------------------------------------------------------------ pcf

def test_pcf_identical_oscillators():
    pcf, orpa, orph_vector, orpa_vector = eop.pcf(lagged_sines([0] * 5))
    assert orpa == pytest.approx(1, abs=1e-6)
    assert pcf >= 0
    assert orph_vector.shape == orpa_vector.shape == T.shape


def test_pcf_independent_oscillators(rng):
    b, a = scipy.signal.butter(4, [8, 12], btype='band', fs=FS)
    data = scipy.signal.filtfilt(b, a, rng.standard_normal((20, T.size)),
                                 axis=1)
    _, orpa, _, _ = eop.pcf(data)
    assert orpa < 0.4


# ------------------------------------------------------------------------ pli

@pytest.mark.parametrize('lag', [0.3, 3.0])
def test_pli_constant_lag(lag):
    # regression: the sign of the unwrapped phase difference flipped whenever
    # it crossed +/- pi (0.92 instead of 1 for lag 0.3)
    assert eop.pli(lagged_sines([0, lag])) == pytest.approx(1)


def test_pli_independent(narrowband_noise):
    assert eop.pli(narrowband_noise) < 0.3


# ------------------------------------------------------------------------ ple

def test_ple_constant_lag_is_zero():
    assert eop.ple(lagged_sines([0, 0.3]), 3, 1) == pytest.approx(0)


def test_ple_white_noise_is_near_one(rng):
    assert eop.ple(rng.standard_normal((4, 5000)), 3, 1) > 0.95


def test_ple_uses_m_and_tau(narrowband_noise):
    # regression: m and tau were silently ignored
    data = narrowband_noise[:3]
    assert eop.ple(data, 3, 10) != pytest.approx(eop.ple(data, 4, 10))
    assert eop.ple(data, 3, 10) != pytest.approx(eop.ple(data, 3, 2))


def test_ple_in_unit_interval(narrowband_noise):
    assert 0 <= eop.ple(narrowband_noise, 3, 10) <= 1


def test_ple_signal_too_short():
    with pytest.raises(ValueError):
        eop.ple(np.random.default_rng(0).standard_normal((2, 5)), 4, 3)


# ------------------------------------------------- complex_phase_relationship

def test_complex_phase_relationship_recovers_lag():
    c = eop.complex_phase_relationship(lagged_sines([0, 0.5])[:, :1000])
    assert c.shape == (2, 2, 1000)
    assert c.dtype == np.complex128
    middle = slice(100, 900) # avoid Hilbert edge effects
    np.testing.assert_allclose(np.abs(c[:, :, middle]), 1)
    np.testing.assert_allclose(np.angle(c[0, 1, middle]), 0.5, atol=1e-3)
    np.testing.assert_allclose(np.angle(c[1, 0, middle]), -0.5, atol=1e-3)


def test_complex_phase_relationship_smoothing():
    c = eop.complex_phase_relationship(lagged_sines([0, 0.5])[:, :1000],
                                       s_freq=FS, win_dur=0.1)
    middle = slice(100, 900)
    np.testing.assert_allclose(np.abs(c[0, 1, middle]), 1, atol=1e-3)
    np.testing.assert_allclose(np.angle(c[0, 1, middle]), 0.5, atol=1e-3)


def test_complex_phase_relationship_smoothing_requires_s_freq():
    with pytest.raises(ValueError):
        eop.complex_phase_relationship(lagged_sines([0, 1]), win_dur=0.1)


# -------------------------------------------------------------- phase_locking

def test_phase_locking_threshold():
    c = eop.complex_phase_relationship(lagged_sines([0, 0.5])[:, :1000])
    middle = slice(100, 900)
    assert eop.phase_locking(c, phase_thresh=np.pi / 4)[0, 1, middle].all()
    assert not eop.phase_locking(c, phase_thresh=0.3)[0, 1, middle].any()


# -------------------------------------------------------- phase_lock_interval

def test_phase_lock_interval():
    # regression: start/end indices were mixed up ([25, 18] here)
    locked = np.zeros((2, 2, 50), dtype=bool)
    locked[0, 1, 5:15] = True
    locked[0, 1, 30:33] = True
    np.testing.assert_array_equal(eop.phase_lock_interval(locked), [10, 3])


def test_phase_lock_interval_single_event_at_end():
    # regression: a single locking event raised an IndexError
    locked = np.zeros((2, 2, 50), dtype=bool)
    locked[0, 1, 45:] = True
    np.testing.assert_array_equal(eop.phase_lock_interval(locked), [5])


def test_phase_lock_interval_no_locking():
    # regression: no locking events raised a ValueError
    assert eop.phase_lock_interval(np.zeros((3, 3, 20), dtype=bool)).size == 0


# ------------------------------------------------------------ global_lability

def test_global_lability_constant_locking_is_zero():
    # regression: N was subtracted although the diagonal already cancels
    locked = np.ones((4, 4, 20), dtype=bool)
    np.testing.assert_array_equal(
        eop.global_lability(locked, 10, 0.1, norm=False), 0)


def test_global_lability_one_pair_unlocks():
    locked = np.ones((4, 4, 20), dtype=bool)
    locked[0, 1, 10:] = locked[1, 0, 10:] = False
    gls = eop.global_lability(locked, 10, 0.1, norm=False) # delta_t = 1
    assert gls.shape == (19,)
    np.testing.assert_array_equal(np.flatnonzero(gls), [9])
    assert gls[9] == 1
    gls_norm = eop.global_lability(locked, 10, 0.1)
    assert gls_norm[9] == pytest.approx((1 / 6) ** 2) # 6 pairs for N = 4


@pytest.mark.parametrize('delta_t_sec', [0.01, 2.0]) # < 1 sample; > data
def test_global_lability_invalid_delta(delta_t_sec):
    with pytest.raises(ValueError):
        eop.global_lability(np.ones((3, 3, 20), dtype=bool), 10, delta_t_sec)


# ------------------------------------------------------ phase-locking avalanches

@pytest.fixture
def jittered_sines(rng):
    """4 channels of the same 10 Hz sinusoid with small phase offsets."""
    offsets = rng.normal(0, 0.05, (4, 1))
    return np.cos(2 * np.pi * FREQ * T[np.newaxis, :] + offsets)


def test_detect_phase_avls_one_avalanche_per_cycle(jittered_sines):
    # regression: entering *and* leaving the phase bin counted as events,
    # doubling the number of avalanches
    avls, events = eop.detect_phase_avls(jittered_sines, FS,
                                         (-np.pi / 8, np.pi / 8),
                                         max_iei=0.008)
    assert 95 <= len(avls) <= 100 # 100 cycles, minus edge-truncated ones
    assert all(a['size'] == 4 for a in avls)
    assert events.shape == T.shape


def test_detect_phase_avls_wraps_phase_range(jittered_sines):
    a1, _ = eop.detect_phase_avls(jittered_sines, FS, (-np.pi / 8, np.pi / 8),
                                  max_iei=0.008)
    a2, _ = eop.detect_phase_avls(jittered_sines, FS,
                                  (2 * np.pi - np.pi / 8, np.pi / 8),
                                  max_iei=0.008)
    assert [a['start_time'] for a in a1] == [a['start_time'] for a in a2]


@pytest.mark.parametrize('staggered, n_keys', [('both', 16), (True, 8),
                                               (False, 8)])
def test_detect_phase_avls_full_bins(jittered_sines, staggered, n_keys):
    # regression: crashed (np.angle has no axis argument) and built n-1 bins
    out = eop.detect_phase_avls_full(jittered_sines, FS, n_phase_bins=8,
                                     max_iei=0.008, staggered=staggered)
    assert len(out) == n_keys
    for avls in out.values():
        assert 95 <= len(avls) <= 100


def test_detect_phase_avls_full_staggered_bin_centred_on_zero(jittered_sines):
    out = eop.detect_phase_avls_full(jittered_sines, FS, n_phase_bins=8,
                                     staggered=True)
    assert '5.89-0.39' in out


def test_detect_phase_avls_full_invalid_staggered(jittered_sines):
    with pytest.raises(ValueError):
        eop.detect_phase_avls_full(jittered_sines, FS, staggered='sometimes')

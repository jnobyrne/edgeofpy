"""Tests for the Ising / pairwise maximum entropy model functions."""

import warnings

import numpy as np
import pytest

import edgeofpy as eop
from edgeofpy import ising
from helpers import (all_states, boltzmann_probabilities, exact_ising_samples,
                     random_ising_model)

# A fixed 4-node model (hand-written, so independent of any RNG)
H4 = np.array([0.1, -0.2, 0.05, 0.3])
J4 = np.array([[0., .4, -.2, .1],
               [.4, 0., .3, -.1],
               [-.2, .3, 0., .2],
               [.1, -.1, .2, 0.]])


@pytest.fixture(scope='module')
def model_and_samples():
    """A random 5-node model and 40,000 exact samples from it (beta = 1)."""
    rng = np.random.default_rng(0)
    h, J = random_ising_model(5, rng)
    return h, J, exact_ising_samples(h, J, 40000, rng)


# ------------------------------------------------------------------ fit_pmem

@pytest.mark.parametrize('method, kwargs', [
    ('ezaki', {'eta': 0.5, 'error_max': 1e-6}),
    ('donders', {'eta': 0.5, 'error_max': 1e-10}),
])
@pytest.mark.parametrize('init_zero', [False, True])
def test_fit_pmem_recovers_model(model_and_samples, method, kwargs,
                                 init_zero):
    # regression: 'donders' crashed or never terminated
    h_true, J_true, data = model_and_samples
    with warnings.catch_warnings():
        warnings.simplefilter('error') # must converge without warnings
        h, J = eop.fit_pmem(data, method=method, init_zero=init_zero,
                            iter_max=20000, **kwargs)
    assert h.shape == (5,) # regression: 'donders' returned (n, 1)
    np.testing.assert_allclose(h, h_true, atol=0.03)
    np.testing.assert_allclose(J, J_true, atol=0.03)
    np.testing.assert_allclose(J, J.T)
    np.testing.assert_array_equal(np.diag(J), 0)


def test_fit_pmem_rejects_zero_one_data(model_and_samples):
    data = model_and_samples[2]
    with pytest.raises(ValueError, match='-1/\\+1'):
        eop.fit_pmem((data > 0).astype(float))


def test_fit_pmem_invalid_method(model_and_samples):
    with pytest.raises(ValueError):
        eop.fit_pmem(model_and_samples[2], method='newton')


@pytest.mark.parametrize('method', ['ezaki', 'donders'])
def test_fit_pmem_warns_if_not_converged(model_and_samples, method):
    with pytest.warns(RuntimeWarning, match='did not converge'):
        eop.fit_pmem(model_and_samples[2], method=method, iter_max=3)


# ------------------------------------------------------------------ fit_beta

@pytest.mark.parametrize('beta_true', [0.5, 1.0, 2.0])
def test_fit_beta_recovers_temperature(beta_true):
    rng = np.random.default_rng(1)
    h, J = random_ising_model(5, rng)
    data = exact_ising_samples(h, J, 20000, rng, beta=beta_true)
    beta = eop.fit_beta(data, h, J, iter_max=200000, eta=0.2)
    assert beta == pytest.approx(beta_true, abs=0.03)


def test_fit_beta_does_not_modify_J(model_and_samples):
    # regression: the diagonal of the caller's J was set to zero
    h, J, data = model_and_samples
    J_in = J.copy()
    J_in[0, 0] = 5.0
    with pytest.warns(RuntimeWarning): # 10 iterations: not converged
        eop.fit_beta(data, h, J_in, iter_max=10, eta=0.01)
    assert J_in[0, 0] == 5.0


def test_fit_beta_warns_if_not_converged(model_and_samples):
    h, J, data = model_and_samples
    with pytest.warns(RuntimeWarning, match='did not converge'):
        eop.fit_beta(data, h, J, iter_max=2)


# ---------------------------------------------------------------- metropolis

def test_metropolis_matches_exact_moments():
    states, E, p = boltzmann_probabilities(H4, J4)
    exact_mean = p @ states
    exact_corr = states.T @ (p[:, None] * states)
    exact_C = (p @ E ** 2 - (p @ E) ** 2) / 4

    m, q, chi_u, chi_sg, C, local_m, corr = eop.metropolis(
        H4, J4, n_iter=20000, rand_seed=0)
    np.testing.assert_allclose(local_m, exact_mean, atol=0.05)
    np.testing.assert_allclose(corr, exact_corr, atol=0.05)
    assert m == pytest.approx(exact_mean.mean(), abs=0.03)
    assert q == pytest.approx(np.mean(local_m ** 2))
    assert C == pytest.approx(exact_C, rel=0.1)


def test_metropolis_reference_values():
    # Fixed values for this seed: guards against unintended changes in
    # random number usage (identical across numpy 1.22-2.2)
    out = eop.metropolis(H4, J4, n_iter=500, rand_seed=5, rand_init=True)
    np.testing.assert_allclose(
        out[:5], [0.079, 0.026028, 1.277036, 1.022106563136, 0.07699699],
        rtol=1e-9)


def test_metropolis_chi_sg_is_real():
    # chi_sg used to be computed with np.linalg.eig, which can return complex
    # eigenvalues for near-degenerate covariances (not reproducible on demand
    # through metropolis, so this only checks the output type)
    chi_sg = eop.metropolis(H4, J4, n_iter=100)[3]
    assert isinstance(chi_sg, float) and np.isrealobj(chi_sg)


def test_metropolis_does_not_touch_global_rng():
    # regression: np.random.seed() reset the global random state
    np.random.seed(123)
    expected = np.random.random()
    np.random.seed(123)
    eop.metropolis(H4, J4, n_iter=50, rand_init=True)
    assert np.random.random() == expected


def test_metropolis_resume_from_checkpoint(tmp_path):
    # regression: resuming replayed a step and restarted the random stream
    prefix = tmp_path / 'run'
    full = eop.metropolis(H4, J4, n_iter=300, rand_seed=5, save_every=100,
                          save_path=str(prefix))
    assert sorted(f.name for f in tmp_path.iterdir()) == \
        ['run_100.npz', 'run_200.npz', 'run_300.npz']
    resumed = eop.metropolis(None, None, rand_seed=999,
                             checkpoint_file=str(tmp_path / 'run_100.npz'))
    np.testing.assert_allclose(resumed[:5], full[:5], rtol=1e-12)


def test_metropolis_rejects_old_checkpoints(tmp_path):
    path = tmp_path / 'old.npz'
    np.savez(path, h=H4, J=J4, n_step_remaining=10)
    with pytest.raises(ValueError, match='older'):
        eop.metropolis(H4, J4, checkpoint_file=str(path))


def test_metropolis_save_every_requires_path():
    with pytest.raises(ValueError):
        eop.metropolis(H4, J4, n_iter=10, save_every=5)


# ----------------------------------------------------- affine transformation

def test_affine_trans():
    # regression: the diagonal was set to mu instead of staying 0
    rng = np.random.default_rng(2)
    _, J = random_ising_model(6, rng)
    J_tilde = ising.affine_trans(J, 0.2, 0.05)
    off_diag = ~np.eye(6, dtype=bool)
    np.testing.assert_array_equal(np.diag(J_tilde), 0)
    assert J_tilde[off_diag].mean() == pytest.approx(0.2)
    assert J_tilde[off_diag].std() == pytest.approx(0.05)


# ----------------------------------------------------------- phase diagrams

SK_KWARGS = dict(mu_range=(-0.1, 0.11), mu_step=0.1, std_range=(0.1, 0.31),
                 std_step=0.1, n_iter=100, rand_seed=1)


def test_sk_diagram_shapes():
    m, q, chi_u, chi_sg, C, x, y = eop.sk_diagram(H4, J4, **SK_KWARGS)
    assert len(x) == 3 and len(y) == 3
    for diagram in (m, q, chi_u, chi_sg, C):
        assert diagram.shape == (3, 3)
    assert np.all(q > 0) # every cell was simulated


@pytest.mark.slow
def test_sk_diagram_multiprocessing_matches_serial():
    # regression: only 2 cells were filled and arguments were shifted
    serial = eop.sk_diagram(H4, J4, **SK_KWARGS)
    parallel = eop.sk_diagram(H4, J4, multiproc=True, n_cpu=2, **SK_KWARGS)
    for a, b in zip(serial[:5], parallel[:5]):
        np.testing.assert_allclose(a, b)


def test_sk_diagram_checkpoints_per_cell(tmp_path):
    # regression: all cells wrote their checkpoints to the same path
    eop.sk_diagram(H4, J4, save_every=50, save_path=str(tmp_path / 'run'),
                   **{**SK_KWARGS, 'mu_range': (0, 0.11)})
    prefixes = {f.name.rsplit('_', 1)[0] for f in tmp_path.iterdir()}
    assert prefixes == {f'run_cell{i}-{j}' for i in range(2)
                        for j in range(3)}


def test_sk_diagram_width_centred_on_off_diagonal_mean():
    off_diag = J4[~np.eye(4, dtype=bool)]
    _, _, _, _, _, x, y = eop.sk_diagram(H4, J4, mu_width=0.2, mu_step=0.05,
                                         std_width=0.1, std_step=0.05,
                                         n_iter=10)
    assert x.mean() == pytest.approx(off_diag.mean(), abs=0.05)
    assert y.mean() == pytest.approx(off_diag.std(), abs=0.05)


@pytest.fixture
def capture_metropolis(monkeypatch):
    """Replace ising.metropolis with a stub recording the J it receives."""
    received = []

    def fake(h, J, **kwargs):
        received.append(J.copy())
        return 0., 1., 0., 0., 0., None, None
    monkeypatch.setattr(ising, 'metropolis', fake)
    return received


def test_ht_diagram_transforms_off_diagonal_only(capture_metropolis):
    # regression: the diagonal was treated as part of the weight bulk
    m, *_, x, y = eop.ht_diagram(H4, J4, tail_cutoff=0.25,
                                 mu_range=(0, 0.2, 0.1),
                                 sigma_range=(0.1, 0.3, 0.1), n_iter=10)
    assert m.shape == (2, 2) and len(x) == 2 and len(y) == 2
    assert len(capture_metropolis) == 4
    for J_tilde in capture_metropolis:
        np.testing.assert_array_equal(np.diag(J_tilde), 0)
        # tail elements (J > 0.25) are unchanged with alpha = 0, beta = 1
        tail = J4 > 0.25
        np.testing.assert_array_equal(J_tilde[tail], J4[tail])


def test_ht_diagram_requires_ranges():
    with pytest.raises(ValueError):
        eop.ht_diagram(H4, J4, tail_cutoff=0.25, n_iter=10)


# ------------------------------------------------------ energy and heat

def test_ising_energy():
    sigma = np.array([1., -1.])
    h = np.array([0.5, 0.2])
    J = np.array([[0., 1.], [1., 0.]])
    # -0.5 * s.J.s - h.s = -0.5 * (-2) - (0.5 - 0.2) = 0.7
    assert eop.ising_energy(sigma, h, J) == pytest.approx(0.7)


def test_heat_capacity(model_and_samples):
    h, J, data = model_and_samples
    data = data[:, :2000]
    E = [eop.ising_energy(s, h, J) for s in data.T]
    assert eop.heat_capacity(data, h, J) == pytest.approx(np.var(E))
    # regression: mode='specific' was ignored
    assert eop.heat_capacity(data, h, J, mode='specific') \
        == pytest.approx(np.var(E) / 5)
    with pytest.raises(ValueError):
        eop.heat_capacity(data, h, J, mode='latent')


def test_energy_landscape_2d():
    sigma = all_states(4)[5]
    flips = np.array([0, 1, 2, 3])
    el = eop.energy_landscape_2d(sigma, 2, flips, H4, J4)
    assert el.shape == (2, 2)
    for i in range(2):
        for j in range(2):
            s = sigma.copy()
            s[flips[:i + 1]] *= -1       # first i+1 flips along axis 1
            s[flips[2:2 + j + 1]] *= -1  # first j+1 flips along axis 2
            assert el[i, j] == pytest.approx(eop.ising_energy(s, H4, J4))

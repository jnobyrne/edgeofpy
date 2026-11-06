"""Package-level tests: public API, metadata and side effects."""

import subprocess
import sys

import numpy as np

import edgeofpy as eop


def test_all_public_names_exist():
    assert len(eop.__all__) == len(set(eop.__all__))
    for name in eop.__all__:
        assert callable(getattr(eop, name)), name


def test_no_private_names_exported():
    exported = [n for n in dir(eop)
                if n.startswith('_') and not n.startswith('__')]
    assert exported == []


def test_version_is_a_string():
    assert isinstance(eop.__version__, str) and eop.__version__


def test_neurokit2_not_required():
    # neurokit2 was dropped as a dependency; importing must not load it
    code = 'import sys, edgeofpy; print("neurokit2" in sys.modules)'
    out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                         text=True, check=True)
    assert out.stdout.strip() == 'False'


def test_stochastic_functions_leave_global_rng_untouched():
    h = np.array([0.1, -0.2, 0.3])
    J = np.array([[0., .2, .1], [.2, 0., -.1], [.1, -.1, 0.]])
    x = np.sin(0.7 * np.arange(300))

    np.random.seed(2024)
    expected = np.random.random(3)
    np.random.seed(2024)
    eop.z1_chaos_test(x)
    eop.chaos_pipeline(x)
    eop.metropolis(h, J, n_iter=20, rand_init=True)
    eop.sk_diagram(h, J, mu_range=(0, 0.1), mu_step=0.1, std_range=(0.1, 0.2),
                   std_step=0.1, n_iter=5, rand_init=True)
    np.testing.assert_array_equal(np.random.random(3), expected)

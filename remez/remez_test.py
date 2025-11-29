import numpy as np

from remez import remez_piecewise


def _linf_error(f1, f2, interval, samples=10000):
    a, b = interval
    x = np.linspace(a, b, samples)
    return np.max(np.abs(f1(x) - f2(x)))


def _check_function(fx, interval, degree, tol, max_num_pieces=100):
    f_approx = remez_piecewise(fx, interval, degree, tol)
    assert _linf_error(fx, f_approx.eval, interval) < tol
    assert len(f_approx.pieces) <= max_num_pieces


def test_sin():
    _check_function(np.sin, (-0.5, 0.5), 3, 1e-7, max_num_pieces=5)
    _check_function(np.sin, (-0.5, 0.5), 4, 1e-7, max_num_pieces=3)
    _check_function(np.sin, (-0.5, 0.5), 5, 1e-7, max_num_pieces=1)
    _check_function(np.sin, (-1, 1), 5, 1e-7, max_num_pieces=3)
    _check_function(np.sin, (-1, 1), 6, 1e-7, max_num_pieces=2)
    _check_function(np.sin, (-1, 1), 7, 1e-7, max_num_pieces=1)
    _check_function(np.sin, (-1, 1), 7, 1e-9, max_num_pieces=2)


def test_cos():
    _check_function(np.cos, (-0.5, 0.5), 3, 1e-7, max_num_pieces=8)
    _check_function(np.cos, (-0.5, 0.5), 4, 1e-7, max_num_pieces=3)
    _check_function(np.cos, (-0.5, 0.5), 5, 1e-7, max_num_pieces=2)
    _check_function(np.cos, (-0.5, 0.5), 6, 1e-7, max_num_pieces=1)


def test_tanh():
    _check_function(np.tanh, (-0.5, 0.5), 5, 1e-5, max_num_pieces=1)
    _check_function(np.tanh, (-0.5, 0.5), 6, 1e-7, max_num_pieces=2)
    _check_function(np.tanh, (-0.5, 0.5), 6, 1e-9, max_num_pieces=3)


def test_exp():
    _check_function(np.exp, (-0.5, 0.5), 4, 1e-7, max_num_pieces=3)
    _check_function(lambda x: np.exp(-x), (-0.5, 0.5), 4, 1e-7, max_num_pieces=3)
    _check_function(lambda x: np.exp(-(x**2)), (-0.5, 0.5), 4, 1e-7, max_num_pieces=5)
    _check_function(lambda x: np.exp(-(x**2)), (-0.5, 0.5), 6, 1e-9, max_num_pieces=4)


def test_arcsin():
    _check_function(np.arcsin, (-0.5, 0.5), 5, 1e-7, max_num_pieces=3)

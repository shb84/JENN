"""Test that activation functions are corrected."""
import numpy as np


# Not an efficient implementation, but okay for testing
def _forward_difference(f: callable, x: np.ndarray, dx: float = 1e-6):
    """Compute partials of y = f(x) using forward difference."""
    n_x, m = x.shape
    dy = np.zeros((n_x, m))
    for i in range(0, n_x):
        dy[i] = np.divide(f(x+dx) - f(x), dx)
    return dy


# Not an efficient implementation, but okay for testing
def _backward_difference(f: callable, x: np.ndarray, dx: float = 1e-6):
    """Compute partials of y = f(x) using backward difference."""
    n_x, m = x.shape
    dy = np.zeros((n_x, m))
    for i in range(0, n_x):
        dy[i] = np.divide(f(x) - f(x-dx), dx)
    return dy


# Not an efficient implementation, but okay for testing
def _central_difference(f: callable, x: np.ndarray, dx: float = 1e-6):
    """Compute partials of y = f(x) using central difference."""
    n_x, m = x.shape
    dy = np.zeros((n_x, m))
    for i in range(0, n_x):
        dy[i] = np.divide(f(x+dx) - f(x-dx), 2 * dx)
    return dy


def finite_difference(f: callable, x: np.ndarray, dx: float = 1e-6):
    """Compute partials of y = f(x) using ctr, fwd or bwd difference."""
    dy = _central_difference(f, x, dx)
    dy[:, :1] = _forward_difference(f, x, dx)[:, :1]
    dy[:, -1:] = _backward_difference(f, x, dx)[:, -1:]
    return dy


def r_square(Y_pred, Y_true):
    """Compute R-square value for each output."""
    axis = Y_true.ndim - 1
    Y_bar = np.mean(Y_true, axis=axis)
    SSE = np.sum(np.square(Y_pred - Y_true), axis=axis)
    SSTO = np.sum(np.square(Y_true - Y_bar) + 1e-12, axis=axis)
    return 1 - SSE / SSTO


"""Finite Differencing.
=======================
"""
# Copyright (C) 2018 Steven H. Berguin
# This work is licensed under the MIT License.

from collections.abc import Callable

import numpy as np


def finite_difference(
    f: Callable,
    x: np.ndarray,
    dx: float = 1e-6,
) -> np.ndarray:
    """Evaluate partial derivative using finite difference.

    :param x: inputs, array of shape (n_x, m)
    :return: partials, array of shape (n_y, n_x, m)
    """
    y: np.ndarray = f(x)
    n_x, m = x.shape
    n_y = y.shape[0]
    dydx = np.zeros((n_y, n_x, m))
    for i in range(n_x):
        dx1 = np.zeros((n_x, m))
        dx2 = np.zeros((n_x, m))
        dx1[i] += dx
        dx2[i] += dx
        x1 = x - dx1
        x2 = x + dx2
        y1 = f(x1)
        y2 = f(x2)
        dydx[:, i] = (y2 - y1) / (2 * dx)
    return dydx

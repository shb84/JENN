"""Linear Test Function.
========================

.. code-block:: python

    #################
    # Example Usage #
    #################

    import jenn.synthetic_data import linear

    x = np.random.rand((2, 10))
    y = linear.compute(x)
    dydx = linear.compute_partials(x)
"""
# Copyright (C) 2018 Steven H. Berguin
# This work is licensed under the MIT License.

import numpy as np


def compute(
    x: np.ndarray,
    a: float | np.ndarray = 1.0,
    b: float = 0.0,
) -> np.ndarray:
    r"""Evaluate linear function.

    .. math::
        f(x) = \beta_0 + \sum_{i=1}^p \beta_i x_i

    :param x: input array of shape (n_x, m)
    :type x: np.ndarray
    :param a: slope
    :type a: float
    :param b: intercept
    :type x:float
    :return: output array of shape (1, m)
    :rtype: np.ndarray
    """
    n_y = 1
    _, m = x.shape
    y = np.zeros((n_y, m))
    y[:] = a * np.sum(x, axis=0) + b
    return y


def compute_partials(
    x: np.ndarray,
    a: float | np.ndarray = 1.0,
) -> np.ndarray:
    r"""Evaluate partials derivatives of linear function.

    .. math::
        f(x) = \beta_0 + \sum_{i=1}^p \beta_i x_i

    :param x: input array of shape (n_x, m)
    :type x: np.ndarray
    :param a: slope
    :type a: float
    :return: output array of shape (1, n_x, m)
    :rtype: np.ndarray
    """
    n_y = 1
    n_x, m = x.shape
    dydx = np.zeros((n_y, n_x, m))
    a = np.array([a] * n_x) if isinstance(a, float) else a
    for i in range(n_x):
        dydx[0, i, :] = a[i]
    return dydx

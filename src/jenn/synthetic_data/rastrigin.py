"""Rastrigin Test Function.
===========================

.. code-block:: python

    #################
    # Example Usage #
    #################

    import jenn.synthetic_data import rastrigin

    x = np.random.rand((2, 10))
    y = rastrigin.compute(x)
    dydx = rastrigin.compute_partials(x)
"""
# Copyright (C) 2018 Steven H. Berguin
# This work is licensed under the MIT License.

import numpy as np


def compute(x: np.ndarray) -> np.ndarray:
    r"""Evaluate Rastrigin function.

    .. math::
        f(x) = \sum_{i=1}^p ( x_i^2 - 10 \cos(2\pi x_i) )

    :param x: input array of shape (n_x, m)
    :type x: np.ndarray
    :return: output array of shape (1, m)
    :rtype: np.ndarray
    """
    n_y = 1
    n_x, m = x.shape
    y = np.zeros((n_y, m)) + 10 * n_x
    for i in range(n_x):
        y += np.power(x[i], 2) - 10 * np.cos(2 * np.pi * x[i])
    return y


def compute_partials(x: np.ndarray) -> np.ndarray:
    r"""Evaluate partials derivatives of Rastrigin function.

    .. math::
        f(x) = \sum_{i=1}^p ( x_i^2 - 10 \cos(2\pi x_i) )

    :param x: input array of shape (n_x, m)
    :type x: np.ndarray
    :return: output array of shape (1, n_x, m)
    :rtype: np.ndarray
    """
    n_y = 1
    n_x, m = x.shape
    dydx = np.zeros((n_y, n_x, m))
    for i in range(n_x):
        dydx[0, i, :] = 2 * x[i] + 20 * np.pi * np.sin(2 * np.pi * x[i])
    return dydx

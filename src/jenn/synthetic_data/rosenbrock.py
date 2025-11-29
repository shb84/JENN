"""Rosenbrock Test Function.
============================

.. code-block:: python

    #################
    # Example Usage #
    #################

    import jenn.synthetic_data import rosenbrock

    x = np.random.rand((2, 10))
    y = rosenbrock.compute(x)
    dydx = rosenbrock.compute_partials(x)
"""
# Copyright (C) 2018 Steven H. Berguin
# This work is licensed under the MIT License.

import numpy as np


def compute(x: np.ndarray) -> np.ndarray:
    r"""Evaluate banana Rosenbrock function.

    .. math::
        f(x) = (1 - x_1)^2 + 100 (x_2 - x_1^2)^ 2

    :param x: input array of shape (n_x, m)
    :type x: np.ndarray
    :return: output array of shape (1, m)
    :rtype: np.ndarray
    """
    n_y = 1
    _, m = x.shape
    y = np.zeros((n_y, m))
    y[:] = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
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
    dydx[0, 0, :] = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2)
    dydx[0, 1, :] = 200 * (x[1] - x[0] ** 2)
    return dydx

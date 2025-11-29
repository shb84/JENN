"""Parabola Test Function.
==========================

.. code-block:: python

    #################
    # Example Usage #
    #################

    import jenn.synthetic_data import parabola

    x = np.random.rand((2, 10))
    y = parabola.compute(x)
    dydx = parabola.compute_partials(x)
"""
# Copyright (C) 2018 Steven H. Berguin
# This work is licensed under the MIT License.

import numpy as np


def compute(
    x: np.ndarray,
    x0: np.ndarray | float = 0.0,
) -> np.ndarray:
    r"""Evaluate parabola function.

    .. math::
        f(x) = \frac{1}{n} \sum_{i=1}^p (x_i - {x_0}_i)^2

    :param x: input array of shape (n_x, m)
    :type x: np.ndarray
    :param x0: center point
    :type x0: np.ndarray
    :return: output array of shape (1, m)
    :rtype: np.ndarray
    """
    n_y = 1
    n_x, m = x.shape
    y = np.zeros((n_y, m))
    y[:] = 1 / n_x * np.sum((x - x0) ** 2, axis=0)
    return y


def compute_partials(
    x: np.ndarray,
    x0: np.ndarray | float = 0.0,
) -> np.ndarray:
    r"""Evaluate partials of parabola function.

    .. math::
        f(x) = \frac{1}{n} \sum_{i=1}^p (x_i - {x_0}_i)^2

    :param x: input array of shape (n_x, m)
    :type x: np.ndarray
    :param x0: center point
    :type x0: np.ndarray
    :return: output array of shape (1, n_x,  m)
    :rtype: np.ndarray
    """
    n_y = 1
    n_x, m = x.shape
    dydx = np.zeros((n_y, n_x, m))
    dydx[0, :, :] = 2 / n_x * (x - x0)
    return dydx

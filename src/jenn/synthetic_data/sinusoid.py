"""Sinusoid Test Function.
==========================

.. code-block:: python

    #################
    # Example Usage #
    #################

    import jenn.synthetic_data import sinusoid

    x = np.random.rand((2, 10))
    y = sinusoid.compute(x)
    dydx = sinusoid.compute_partials(x)
"""
# Copyright (C) 2018 Steven H. Berguin
# This work is licensed under the MIT License.

import numpy as np


def compute(x: np.ndarray) -> np.ndarray:
    r"""Evaluate sinusoidal function.

    .. math::
        f(x) = x \sin(x)

    :param x: input array of shape (n_x, m)
    :type x: np.ndarray
    :return: output array of shape (1, m)
    :rtype: np.ndarray
    """
    n_y = 1
    _, m = x.shape
    y = np.zeros((n_y, m))
    y[:] = x * np.sin(x)
    return y


def compute_partials(x: np.ndarray) -> np.ndarray:
    r"""Evaluate partials derivatives of sinusoidal function.

    .. math::
        f(x) = x \sin(x)

    :param x: input array of shape (n_x, m)
    :type x: np.ndarray
    :return: output array of shape (1, n_x, m)
    :rtype: np.ndarray
    """
    n_y = 1
    n_x, m = x.shape
    dydx = np.zeros((n_y, n_x, m))
    dydx[0, 0, :] = np.sin(x) + x * np.cos(x)
    return dydx

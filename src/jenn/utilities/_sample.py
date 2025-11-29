"""Synthetic Data.
==================

This module provide synthetic test functions
that can be used to generate exmaple data for
illustration and testing. Simply inherit from
the base class to implement new test functions.

.. code-block:: python

    #################
    # Example Usage #
    #################

    import jenn

    (
        x_train,
        y_train,
        dydx_train,
    ) = jenn.synthetic_data.sample(
        m_random=0, # number random samples
        m_levels=4, # number of full factorial levels per factor
        lb=-3.14,   # lower bound of domain
        ub=3.14,    # upper bound of domain
    )
"""
# Copyright (C) 2018 Steven H. Berguin
# This work is licensed under the MIT License.

from collections.abc import Callable

import numpy as np

from ._finite_difference import finite_difference


def _fullfact(n_x: int, m_levels: int) -> np.ndarray:
    """Return full factorial with sample values between 0 and 1."""
    array = np.linspace(0, 1, m_levels)
    arrays = [array] * n_x
    meshes = [mesh.reshape((1, -1)) for mesh in np.meshgrid(*arrays)]
    return np.concatenate(meshes)


def sample(
    f: Callable,
    m_random: int,
    m_levels: int,
    lb: np.typing.ArrayLike,
    ub: np.typing.ArrayLike,
    dx: float = 1e-6,
    f_prime: Callable | None = None,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic data by sampling the test function.

    :param f: callable function to be sampled, y = f(x)
    :type f: Callable
    :param m_random: number of random samples
    :type m_random: int
    :param m_levels: number of levels per factor for full factorial
    :type m_levels: int
    :param lb: lower bound on the factors
    :type lb: np.ndarray
    :param ub: upper bound on the factors
    :type ub: np.ndarray
    :param dx: finite difference step size
    :type dx: float
    :param f_prime: callable 1st derivative to be sampled, y = f'(x)
    :type f_prime: Callable
    :param random_state: random seed (for repeatability)
    :type random_state: int
    :return: sampled (x, y, y')
    :rtype: np.ndarray
    """
    rng = np.random.default_rng(seed=random_state)
    lb = np.array([lb]).reshape((-1, 1))  # make sure it's an numpy array
    ub = np.array([ub]).reshape((-1, 1))  # make sure it's an numpy array
    n_x = lb.size
    lh = rng.random(size=(n_x, m_random))
    ff = _fullfact(n_x, m_levels)
    doe = np.concatenate([lh, ff], axis=1)
    m = doe.shape[1]
    x = lb + (ub - lb) * doe
    y = f(x).reshape((-1, m))
    if f_prime:
        dydx = f_prime(x).reshape((-1, n_x, m))
    else:
        dydx = finite_difference(f, x, dx).reshape((-1, n_x, m))
    return x, y, dydx

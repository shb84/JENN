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
    ) = jenn.synthetic.Sinusoid.sample(
        m_lhs=0,    # number latin hypercube samples 
        m_levels=4, # number of full factorial levels per factor
        lb=-3.14,   # lower bound of domain 
        ub=3.14,    # upper bound of domain 
    )

    (
        x_test, 
        y_test, 
        dydx_test,
    ) = jenn.synthetic.Sinusoid.sample(
        m_lhs=30, 
        m_levels=0, 
        lb=-3.14,
        ub=3.14,
    )
"""  # noqa: W291

import abc
from typing import Tuple, Union

import numpy as np


def _fullfact(n_x: int, m_levels: int) -> np.ndarray:
    """Return full factorial with sample values between 0 and 1."""
    array = np.linspace(0, 1, m_levels)
    arrays = [array] * n_x
    meshes = [mesh.reshape((1, -1)) for mesh in np.meshgrid(*arrays)]
    return np.concatenate(meshes)


class TestFunction:
    """Test function base class."""

    @abc.abstractmethod
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate function.

        :param x: inputs, array of shape (n_x, m)
        :return: response, array of shape (n_y, m)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def first_derivative(self, x: np.ndarray) -> np.ndarray:
        """Evaluate partial derivative.

        :param x: inputs, array of shape (n_x, m)
        :return: partials, array of shape (n_y, n_x, m)
        """
        raise NotImplementedError

    @classmethod
    def first_derivative_FD(
        cls,
        x: np.ndarray,
        dx: float = 1e-6,
    ) -> np.ndarray:
        """Evaluate partial derivative using finite difference.

        :param x: inputs, array of shape (n_x, m)
        :return: partials, array of shape (n_y, n_x, m)
        """
        f = cls.evaluate
        y = f(x)  # type: ignore
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
            y1 = f(x1)  # type: ignore
            y2 = f(x2)  # type: ignore
            dydx[:, i] = (y2 - y1) / (2 * dx)
        return dydx

    @classmethod
    def sample(
        cls,
        m_lhs: int,
        m_levels: int,
        lb: Union[np.ndarray, float],
        ub: Union[np.ndarray, float],
        dx: float | None = 1e-6,
        random_state: Union[int, None] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic data by sampling the test function.

        :param m_lhs: number of latin hypercube samples
        :param m_levels: number of levels per factor for full factorial
        :param lb: lower bound on the factors
        :param ub: upper bound on the factors
        :param dx: finite difference step size (if None, analytical partials are used)
        :param random_state: random seed (for repeatability)
        """
        rng = np.random.default_rng(seed=random_state)
        lb = np.array([lb]).reshape((-1, 1))  # make sure it's an numpy array
        ub = np.array([ub]).reshape((-1, 1))  # make sure it's an numpy array
        n_x = lb.size
        lh = rng.random(size=(n_x, m_lhs))
        ff = _fullfact(n_x, m_levels)
        doe = np.concatenate([lh, ff], axis=1)
        m = doe.shape[1]
        x = lb + (ub - lb) * doe
        y = cls.evaluate(x).reshape((-1, m))  # type: ignore[call-arg]
        if dx is None:
            dydx = cls.first_derivative(x).reshape((-1, n_x, m))  # type: ignore[call-arg]
        else:
            dydx = cls.first_derivative_FD(x, dx).reshape((-1, n_x, m))
        return x, y, dydx


class Linear(TestFunction):
    r"""Linear function.

    .. math::
        f(x) = \beta_0 + \sum_{i=1}^p \beta_i x_i
    """

    @classmethod
    def evaluate(
        cls,
        x: np.ndarray,
        a: Union[float, np.ndarray] = 1.0,
        b: float = 0.0,
    ) -> np.ndarray:  # noqa: D102
        n_y = 1
        n_x, m = x.shape
        y = np.zeros((n_y, m))
        y[:] = a * np.sum(x, axis=0) + b
        return y

    @classmethod
    def first_derivative(
        cls,
        x: np.ndarray,
        a: Union[float, np.ndarray] = 1.0,
        b: float = 0.0,
    ) -> np.ndarray:  # noqa: D102
        n_y = 1
        n_x, m = x.shape
        dydx = np.zeros((n_y, n_x, m))
        a = np.array([a] * n_x) if isinstance(a, float) else a
        for i in range(n_x):
            dydx[0, i, :] = a[i]
        return dydx

    @classmethod
    def sample(
        cls,
        m_lhs: int = 100,
        m_levels: int = 0,
        lb: Union[np.ndarray, float] = -1.0,
        ub: Union[np.ndarray, float] = 1.0,
        dx: float | None = 1e-6,
        random_state: Union[int, None] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # noqa: D102
        return super().sample(m_lhs, m_levels, lb, ub, dx, random_state)


class Parabola(TestFunction):
    r"""Parabolic function.

    .. math::
        f(x) = \frac{1}{n} \sum_{i=1}^p (x_i - {x_0}_i)^2
    """

    @classmethod
    def evaluate(
        cls, x: np.ndarray, x0: Union[np.ndarray, float] = 0.0
    ) -> np.ndarray:  # noqa: D102
        n_y = 1
        n_x, m = x.shape
        y = np.zeros((n_y, m))
        y[:] = 1 / n_x * np.sum((x - x0) ** 2, axis=0)
        return y

    @classmethod
    def first_derivative(
        cls, x: np.ndarray, x0: Union[np.ndarray, float] = 0.0
    ) -> np.ndarray:  # noqa: D102
        n_y = 1
        n_x, m = x.shape
        dydx = np.zeros((n_y, n_x, m))
        dydx[0, :, :] = 2 / n_x * (x - x0)
        return dydx

    @classmethod
    def sample(
        cls,
        m_lhs: int = 100,
        m_levels: int = 0,
        lb: Union[np.ndarray, float] = -1.0,
        ub: Union[np.ndarray, float] = 1.0,
        dx: float | None = 1e-6,
        random_state: Union[int, None] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # noqa: D102
        return super().sample(m_lhs, m_levels, lb, ub, dx, random_state)


class Sinusoid(TestFunction):
    r"""Sinusoidal function.

    .. math::
        f(x) = x \sin(x)
    """

    @classmethod
    def evaluate(cls, x: np.ndarray) -> np.ndarray:  # noqa: D102
        n_y = 1
        n_x, m = x.shape
        y = np.zeros((n_y, m))
        y[:] = x * np.sin(x)
        return y

    @classmethod
    def first_derivative(cls, x: np.ndarray) -> np.ndarray:  # noqa: D102
        n_y = 1
        n_x, m = x.shape
        dydx = np.zeros((n_y, n_x, m))
        dydx[0, 0, :] = np.sin(x) + x * np.cos(x)
        return dydx

    @classmethod
    def sample(
        cls,
        m_lhs: int = 100,
        m_levels: int = 0,
        lb: Union[np.ndarray, float] = -np.pi,
        ub: Union[np.ndarray, float] = np.pi,
        dx: float | None = 1e-6,
        random_state: Union[int, None] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # noqa: D102
        return super().sample(m_lhs, m_levels, lb, ub, dx, random_state)


class Rastrigin(TestFunction):
    r"""Rastrigin function.

    .. math::
        f(x) = \sum_{i=1}^p ( x_i^2 - 10 \cos(2\pi x_i) )
    """

    @classmethod
    def evaluate(cls, x: np.ndarray) -> np.ndarray:  # noqa: D102
        n_y = 1
        n_x, m = x.shape
        y = np.zeros((n_y, m)) + 10 * n_x
        for i in range(n_x):
            y += np.power(x[i], 2) - 10 * np.cos(2 * np.pi * x[i])
        return y

    @classmethod
    def first_derivative(cls, x: np.ndarray) -> np.ndarray:  # noqa: D102
        n_y = 1
        n_x, m = x.shape
        dydx = np.zeros((n_y, n_x, m))
        for i in range(n_x):
            dydx[0, i, :] = 2 * x[i] + 20 * np.pi * np.sin(2 * np.pi * x[i])
        return dydx

    @classmethod
    def sample(
        cls,
        m_lhs: int = 100,
        m_levels: int = 0,
        lb: Union[np.ndarray, float] = -1.0
        * np.ones(
            2,
        ),
        ub: Union[np.ndarray, float] = 1.5
        * np.ones(
            2,
        ),
        dx: float | None = 1e-6,
        random_state: Union[int, None] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # noqa: D102
        return super().sample(m_lhs, m_levels, lb, ub, dx, random_state)


class Rosenbrock(TestFunction):
    r"""Banana Rosenbrock function.

    .. math::
        f(x) = (1 - x_1)^2 + 100 (x_2 - x_1^2)^ 2
    """

    @classmethod
    def evaluate(cls, x: np.ndarray) -> np.ndarray:  # noqa: D102
        n_y = 1
        n_x, m = x.shape
        y = np.zeros((n_y, m))
        y[:] = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
        return y

    @classmethod
    def first_derivative(cls, x: np.ndarray) -> np.ndarray:  # noqa: D102
        n_y = 1
        n_x, m = x.shape
        dydx = np.zeros((n_y, n_x, m))
        dydx[0, 0, :] = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2)
        dydx[0, 1, :] = 200 * (x[1] - x[0] ** 2)
        return dydx

    @classmethod
    def sample(
        cls,
        m_lhs: int = 100,
        m_levels: int = 0,
        lb: Union[np.ndarray, float] = -2
        * np.ones(
            2,
        ),
        ub: Union[np.ndarray, float] = 2.0
        * np.ones(
            2,
        ),
        dx: float | None = 1e-6,
        random_state: Union[int, None] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:  # noqa: D102
        return super().sample(m_lhs, m_levels, lb, ub, dx, random_state)

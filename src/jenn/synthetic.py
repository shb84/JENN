"""Canonical test function representative of topology
encountered for surrogate-based optimization."""

import abc
import numpy as np


def _fullfact(n_x: int, m_levels: int):
    """Return full factorial with sample values between 0 and 1."""
    array = np.linspace(0, 1, m_levels)
    arrays = [array] * n_x
    meshes = [mesh.reshape((1, -1)) for mesh in np.meshgrid(*arrays)]
    return np.concatenate(meshes)


class TestFunction:
    """Test function base class"""

    @abc.abstractmethod
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate function

        Parameters
        ----------
        x: np.ndarray
            Input array of shape (n_x, m) where m is the number of examples

        Returns
        -------
        y: np.ndarray
            Output array of shape (n_y, m)
        """
        raise NotImplementedError

    @abc.abstractmethod
    def first_derivative(self, x: np.ndarray) -> np.ndarray:
        """Evaluate partial derivative

        Parameters
        ----------
        x: np.ndarray
            Input array of shape (n_x, m) where m is the number of examples

        Returns
        -------
        dydx: np.ndarray
            Output array of shape (n_y, n_x, m)
        """
        raise NotImplementedError

    @classmethod
    def sample(
            cls,
            m_lhs: int,
            m_levels: int,
            lb: np.ndarray | float,
            ub: np.ndarray | float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic data by sampling test function."""
        lb = np.array([lb]).reshape((-1, 1))  # make sure it's an numpy array
        ub = np.array([ub]).reshape((-1, 1))  # make sure it's an numpy array
        n_x = lb.size
        lh = np.random.rand(n_x, m_lhs)
        ff = _fullfact(n_x, m_levels)
        doe = np.concatenate([lh, ff], axis=1)
        m = doe.shape[1]
        x = lb + (ub - lb) * doe
        y = cls.evaluate(x).reshape((-1, m))
        dydx = cls.first_derivative(x).reshape((-1, n_x, m))
        return x, y, dydx


class Linear(TestFunction):
    """Linear test function:
                         n
    f(x1, ..., xn) = b + ∑ a[i] * x[i]
                        i=0
    """

    @classmethod
    def evaluate(
            cls, x: np.ndarray,
            a: float | np.ndarray = 1.0,
            b: float = 0.0,
    ) -> np.ndarray:
        n_y = 1
        n_x, m = x.shape
        y = np.zeros((n_y, m))
        y[:] = a * np.sum(x, axis=0) + b
        return y

    @classmethod
    def first_derivative(
            cls, x: np.ndarray,
            a: float | np.ndarray = 1.0,
            b: float = 0.0,
    ) -> np.ndarray:
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
            lb: np.ndarray | float = -1.,
            ub: np.ndarray | float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return super().sample(m_lhs, m_levels, lb, ub)


class Parabola(TestFunction):
    """Parabola test function:

                           n
    f(x1, ..., xn) = 1/n * ∑ (x[i] - x0[i]) ** 2
                          i=0

    """

    @classmethod
    def evaluate(cls, x: np.ndarray, x0: np.ndarray = 0.0) -> np.ndarray:
        n_y = 1
        n_x, m = x.shape
        y = np.zeros((n_y, m))
        y[:] = 1 / n_x * np.sum((x - x0) ** 2, axis=0)
        return y

    @classmethod
    def first_derivative(
            cls, x: np.ndarray, x0: np.ndarray = 0.0) -> np.ndarray:
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
            lb: np.ndarray | float = -1.,
            ub: np.ndarray | float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return super().sample(m_lhs, m_levels, lb, ub)


class Sinusoid(TestFunction):
    """Sinusoidal test function: y =x * np.sin(x)"""

    @classmethod
    def evaluate(cls, x: np.ndarray) -> np.ndarray:
        n_y = 1
        n_x, m = x.shape
        y = np.zeros((n_y, m))
        y[:] = x * np.sin(x)
        return y

    @classmethod
    def first_derivative(cls, x: np.ndarray) -> np.ndarray:
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
            lb: np.ndarray | float = -np.pi,
            ub: np.ndarray | float = np.pi,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return super().sample(m_lhs, m_levels, lb, ub)


class Rastrigin(TestFunction):
    """Rastrigin (egg crate) test function:
                     n
    f(x1, ..., xn) = ∑ (x[i] ** 2 − 10 cos(2πxi))
                    i=0
    """

    @classmethod
    def evaluate(cls, x: np.ndarray) -> np.ndarray:
        n_y = 1
        n_x, m = x.shape
        y = np.zeros((n_y, m)) + 10 * n_x
        for i in range(n_x):
            y += np.power(x[i], 2) - 10 * np.cos(2 * np.pi * x[i])
        return y

    @classmethod
    def first_derivative(cls, x: np.ndarray) -> np.ndarray:
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
            lb: np.ndarray | float = -1. * np.ones(2,),
            ub: np.ndarray | float = 1.5 * np.ones(2,),
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return super().sample(m_lhs, m_levels, lb, ub)


class Rosenbrock(TestFunction):
    """Banana Rosenbrock test function:
    y = (1 - x1) ** 2 + 100 * (x2 - x1 ** 2) ** 2"""

    @classmethod
    def evaluate(cls, x: np.ndarray) -> np.ndarray:
        n_y = 1
        n_x, m = x.shape
        y = np.zeros((n_y, m))
        y[:] = (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2
        return y

    @classmethod
    def first_derivative(cls, x: np.ndarray) -> np.ndarray:
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
            lb: np.ndarray | float = -2 * np.ones(2,),
            ub: np.ndarray | float = 2. * np.ones(2,),
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return super().sample(m_lhs, m_levels, lb, ub)

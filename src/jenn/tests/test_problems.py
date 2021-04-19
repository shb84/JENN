import numpy as np
from typing import List, Tuple
from jenn._utils import finite_diff


def linear(x: List[np.ndarray], a: float = 1, b: float = 0,
           ) -> Tuple[np.ndarray, List[np.ndarray]]:
    x = x[0]
    y = a * np.sum(x) + b
    y = y.reshape(1, 1)
    dydx = np.array([a] * x.ndim).reshape(x.shape)
    return y, [dydx]


def parabola(x: List[np.ndarray], x0: float = 0
             ) -> Tuple[np.ndarray, List[np.ndarray]]:
    x = x[0]
    y = (x - x0) ** 2
    y = y.reshape(1, 1)
    dydx = (2 * (x - x0)).reshape(x.shape)
    return y, [dydx]


def rosenbrock(x: List[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray]]:
    x1 = x[0][0]
    x2 = x[0][1]
    y = (1 - x1) ** 2 + 100 * (x2 - x1 ** 2) ** 2
    y = y.reshape(1, 1)
    dydx = np.zeros(x[0].shape)
    dydx[0] = -2 * (1 - x1) - 400 * x1 * (x2 - x1 ** 2)
    dydx[1] = 200 * (x2 - x1 ** 2)
    return y, [dydx]


def rastrigin(X: np.ndarray) -> tuple:
    """
    Compute the Rastrigin function (and its gradient) given by:
                           n
    f(x1, ..., xn) = 10n + ∑ (xi ** 2 − 10 cos(2πxi))
                          i=1

    Parameters
    ----------
        X: List[np.ndarray]
            The inputs at which to evaluate the function
            shape = (m, n) where m = number of examples
                                 n = number of inputs (dimensionality)

    Returns
    -------
        y: np.ndarray
            The value of the response evaluated at X
            shape = (m, 1)

        dy_dX: np.ndarray
            The gradient of the response evaluated at X
            shape (m, n, 1)
    """
    m, n = X.shape
    y = np.zeros((m, 1)) + 10 * n
    dy_dX = np.zeros((m, n, 1))
    for i in range(n):
        x = X[:, i].reshape((-1, 1))
        y += np.power(x, 2) - 10 * np.cos(2 * np.pi * x)
        dy_dX[:, i] = 2 * x + 20 * np.pi * np.sin(2 * np.pi * x)
    return y, dy_dX


def test_rastrigin():
    """
    Check that Rastrigin returns correct values and partials
    by comparing against known answers at selected points.
    """

    x = np.zeros((1, 2))
    y, dy_dx = rastrigin(x)

    assert np.allclose(y.ravel(), 0.)
    assert np.allclose(dy_dx.ravel(),
                       finite_diff(x, f=lambda x: rastrigin(x)[0]))

    X = np.array([-0.259757925, 0.85149854]).reshape((1, 2))
    y, dy_dX = rastrigin(X)

    assert np.allclose(y.ravel(), np.array([15.45148376]))
    assert np.allclose(dy_dX.ravel(),
                       finite_diff(X, f=lambda x: rastrigin(x)[0]))


if __name__ == "__main__":
    test_rastrigin()
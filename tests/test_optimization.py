"""Check that optimizer by testing it against
standard cannonical test functions.
"""
# Copyright (C) 2018 Steven H. Berguin
# This work is licensed under the MIT License.

from importlib.util import find_spec

import numpy as np
import pytest

import jenn

if find_spec("matplotlib"):
    from matplotlib import pyplot as plt

    MATPLOTLIB_INSTALLED = True
else:
    MATPLOTLIB_INSTALLED = False


class TestLineSearch:
    """Check that line search is working
    using 1D parabola as test function.
    """

    @pytest.fixture
    def line_search(self) -> jenn.core.optimization.Backtracking:
        """Return backtracking line search using gradient descent."""
        return jenn.core.optimization.Backtracking(update=jenn.core.optimization.GD())

    def test_finds_minimum(self, line_search: jenn.core.optimization.Backtracking):
        """Test that line search finds the minimum when learning rate step size
        is large enough to include minimum in search radius.
        """
        center = 1.0  # center parabola at this number

        def f(x: np.ndarray) -> np.ndarray:
            """Evaluate objective function."""
            return jenn.synthetic_data.parabola.compute(x, x0=center)

        def f_prime(x: np.ndarray) -> np.ndarray:
            """Evaluate partials."""
            return jenn.synthetic_data.parabola.compute_partials(x, x0=center)

        x0 = np.array([center - 1]).reshape((1, 1))  # approach from left
        y0 = f(x0)
        xbest, _ = line_search(
            x0, y0, search_direction=f_prime(x0), cost_function=f, learning_rate=2.0
        )
        assert xbest == center
        x0 = np.array([center + 1]).reshape((1, 1))  # approach from right
        y0 = f(x0)
        xbest, _ = line_search(
            x0, y0, search_direction=f_prime(x0), cost_function=f, learning_rate=2.0
        )
        assert xbest == center

    def test_finds_minbound(self, line_search: jenn.core.optimization.Backtracking):
        """Test that line search finds the minimum bound when learning rate step size
        is not large enough to include minimum in search radius.
        """

        def f(x: np.ndarray) -> np.ndarray:
            """Evaluate objective function."""
            return jenn.synthetic_data.parabola.compute(x, x0=0.0)

        def f_prime(x: np.ndarray) -> np.ndarray:
            """Evaluate partials."""
            return jenn.synthetic_data.parabola.compute_partials(x, x0=0.0)

        x0 = np.array([1]).reshape((1, 1))  # approach from right
        y0 = f(x0)
        xbest, _ = line_search(
            x0, y0, search_direction=f_prime(x0), cost_function=f, learning_rate=0.1
        )
        assert xbest == 0.8
        x0 = np.array([-1]).reshape((1, 1))  # approach from left
        y0 = f(x0)
        xbest, _ = line_search(
            x0, y0, search_direction=f_prime(x0), cost_function=f, learning_rate=0.1
        )
        assert xbest == -0.8


class TestUpdate:
    """Test parameter update using a simple
    linear function.
    """

    def test_gradient_descent(self):
        """Test gradient descent using simple linear function."""
        x0 = np.array([5, 10]).reshape((1, -1))
        dydx = jenn.synthetic_data.linear.compute_partials(x0)
        update = jenn.core.optimization.GD()
        x = update(x0, dydx, alpha=1).squeeze()
        assert np.allclose(x, np.array([4, 9]))

    def test_adam(self):
        """Test ADAM using simple linear function."""
        x0 = np.array([5, 10]).reshape((1, -1))
        dydx = jenn.synthetic_data.linear.compute_partials(x0)
        update = jenn.core.optimization.ADAM()
        x = update(x0, dydx, alpha=1).squeeze()
        assert np.allclose(x, np.array([4, 9]))


class TestOptimizer:
    """Test optimizer using banana Rosenbrock function."""

    @classmethod
    def test_rosenbrock(
        cls,
        alpha: float = 0.05,
        max_iter: int = 1000,
        is_adam: bool = True,
        is_plot: bool = False,
    ) -> None:
        """Check that optimizer yields correct answer for rosenbrock function."""
        # Initial guess
        x0 = np.array([1.25, -1.75]).reshape((2, 1))

        # Test function
        f = jenn.synthetic_data.rosenbrock.compute
        dfdx = jenn.synthetic_data.rosenbrock.compute_partials

        # Optimization
        if is_adam:
            opt = jenn.core.optimization.ADAMOptimizer()
        else:
            opt = jenn.core.optimization.GDOptimizer()

        xf = opt.minimize(x0, f, dfdx, alpha=alpha, max_iter=max_iter)

        # For plotting contours
        lb = -2.0
        ub = 2.0
        m = 100
        x1 = np.linspace(lb, ub, m)
        x2 = np.linspace(lb, ub, m)
        X1, X2 = np.meshgrid(x1, x2)
        Y = np.zeros(X1.shape)
        for i in range(m):
            for j in range(m):
                X = np.array([
                    [X1[i, j]],
                    [X2[i, j]],
                ])
                Y[i, j] = f(X).squeeze()

        if is_plot:
            x1_his = np.array([x[0] for x in opt.vars_history]).squeeze()
            x2_his = np.array([x[1] for x in opt.vars_history]).squeeze()
            plt.plot(x1_his, x2_his)
            plt.plot(x0[0], x0[1], "+", ms=15)
            plt.plot(xf[0], xf[1], "o")
            plt.plot(np.array([1.0]), np.array([1.0]), "x")
            plt.legend([
                "history",
                "initial guess",
                "predicted optimum",
                "true optimum",
            ])
            plt.contour(X1, X2, Y, 50, cmap="RdGy")
            if is_adam:
                plt.title("ADAM")
            else:
                plt.title("GD")
            plt.show()

        # Close to optimum, the slope is nearly zero and the optimizer really
        # struggles to get to the exact optimum. +/- 0.2 is actually very close
        # to optimality. Turn on the contour plots to see.
        assert np.allclose(xf[0], 1.0, atol=0.2)
        assert np.allclose(xf[1], 1.0, atol=0.2)


if __name__ == "__main__":
    TestOptimizer.test_rosenbrock(alpha=0.1, max_iter=1_000, is_adam=True, is_plot=True)

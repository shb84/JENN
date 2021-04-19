import numpy as np
from jenn._optimizer import Backtracking, ADAM, GD, ADAMOptimizer, GDOptimizer
from jenn.tests.test_problems import linear, rosenbrock, parabola
from importlib.util import find_spec

if find_spec("matplotlib"):
    from matplotlib import pyplot as plt
    MATPLOTLIB_INSTALLED = True
else:
    MATPLOTLIB_INSTALLED = False


def test_line_search():

    line = Backtracking(update=GD())

    f = lambda x: parabola(x)

    x0 = np.array([1]).reshape((1, 1))
    assert line.search([x0], f(x0)[1], f, learning_rate=0.1)[0] == 0.8

    x0 = np.array([-1]).reshape((1, 1))
    assert line.search([x0], f(x0)[1], f, learning_rate=2.0)[0] == 0

    f = lambda x: parabola(x, x0=1)

    x0 = np.array([1]).reshape((1, 1))
    assert line.search(x0, f(x0)[1], f, learning_rate=2.0)[0] == 1

    x0 = np.array([-1]).reshape((1, 1))
    assert line.search(x0, f(x0)[1], f, learning_rate=2.0)[0] == 1


def test_param_update():
    x0 = [np.array([5, 10]).reshape((1, -1))]
    y0, dydx = linear(x0)
    for update in [GD(), ADAM()]:
        x = update(x0, dydx, alpha=1)[0].squeeze()
        assert np.allclose(x, np.array([4, 9]))


def test_optimization(alpha: float = 0.05, max_iter: int = 1000,
                      is_adam: bool = True, is_plot: bool = False):
    """ check that optimizer yields correct answer for rosenbrock function"""

    # Initial guess
    x0 = [np.array([1.25, -1.75]).reshape((2, 1))]

    # Test function
    f = lambda x: rosenbrock(x)

    # Optimization
    if is_adam:
        opt = ADAMOptimizer()
    else:
        opt = GDOptimizer()

    xf = opt.minimize(x0, f, alpha=alpha, max_iter=max_iter)

    # For plotting contours
    lb = -2.
    ub = 2.
    m = 100
    x1 = np.linspace(lb, ub, m)
    x2 = np.linspace(lb, ub, m)
    X1, X2 = np.meshgrid(x1, x2)
    Y = np.zeros(X1.shape)
    for i in range(0, m):
        for j in range(0, m):
            X = [np.array([X1[i, j], X2[i, j]])]
            Y[i, j] = f(X)[0]

    if not MATPLOTLIB_INSTALLED:
        raise ImportError("Matplotlib must be installed.")

    if is_plot:
        x1_his = np.array([x[0][0] for x in opt.vars_history]).squeeze()
        x2_his = np.array([x[0][1] for x in opt.vars_history]).squeeze()
        plt.plot(x1_his, x2_his)
        plt.plot(x0[0][0], x0[0][1], '+', ms=15)
        plt.plot(xf[0][0], xf[0][1], 'o')
        plt.plot(np.array([1.]), np.array([1.]), 'x')
        plt.legend(
            ['history', 'initial guess', 'predicted optimum', 'true optimum'])
        plt.contour(X1, X2, Y, 50, cmap='RdGy')
        if is_adam:
            plt.title('ADAM')
        else:
            plt.title('GD')
        plt.show()

    # Close to optimum, the slope is nearly zero and the optimizer really
    # struggles to get to the exact optimum. +/- 0.2 is actually very close
    # to optimality. Turn on the contour plots to see.
    assert np.allclose(xf[0][0], 1.0, atol=0.2)
    assert np.allclose(xf[0][1], 1.0, atol=0.2)


if __name__ == "__main__":
    test_param_update()
    test_line_search()
    test_optimization(alpha=0.01, max_iter=10000, is_adam=True, is_plot=False)

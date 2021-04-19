import numpy as np
from jenn.tests.test_problems import rosenbrock
from jenn._utils import grad_check


def test_grad_check():
    """
    Test that method to check gradient accuracy works, using the
    banana Rosenbrock function as a test example
    """
    x0 = [np.array([1.25, -1.75]).reshape((2, 1))]
    f = lambda x: rosenbrock(x)[0]
    dfdx = lambda x: rosenbrock(x)[1]
    assert grad_check(x0, f, dfdx)
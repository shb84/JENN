"""Test that cost function is working."""
import jenn
import numpy as np


def test_least_squares():
    """Test that least squares cost function evaluates to known answers."""
    x, y, dydx = jenn.synthetic.Sinusoid.sample(100)

    parameters = jenn.core.parameters.Parameters(layer_sizes=[2, 2, 1])
    parameters.initialize()
    
    data = jenn.core.data.Dataset(x, y, dydx)
    cost = jenn.core.cost.Cost(data, parameters, lambd=0.0)

    # Verify that cost is zero when prediction is perfect
    assert cost.evaluate(y, dydx) == 0

    # Verify that cost is non-zero when prediction is imperfect
    eps = 2 * np.random.rand()
    assert cost.evaluate(y + eps, dydx + eps) != 0

    # TODO: check prediction returns known result


def test_regularization():
    pass  # TODO 


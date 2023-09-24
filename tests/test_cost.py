"""Test that cost function is working."""
import jenn
import numpy as np


def test_least_squares(sinusoidal_data_1D):
    """Test that least squares cost function evaluates to known answers."""
    training_data, test_data = sinusoidal_data_1D

    parameters = jenn.Parameters(layer_sizes=[2, 2, 1])
    cost = jenn.Cost(test_data, parameters, lambd=0.0)

    # Verify that cost is zero when prediction is perfect
    assert cost.evaluate(Y_pred=test_data.Y) == 0

    # Verify that cost is non-zero when prediction is imperfect
    assert cost.evaluate(Y_pred=test_data.Y + 2 * np.random.rand()) != 0

    # TODO: check prediction returns known result


def test_regularization():
    pass  # TODO 


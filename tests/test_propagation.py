"""Test that forward and backward propagation are working."""
import numpy as np
import jenn
from copy import deepcopy


def test_finite_difference():
    """Test that finite difference is working. Critical since it's used to
    check that backpropagation is working."""

    parameters = jenn.Parameters([2, 2, 1])

    parameters.W[0][:] = 1.
    parameters.b[0][:] = 1.
    parameters.W[1][:] = 1.
    parameters.b[1][:] = 1.
    parameters.W[2][:] = 1.
    parameters.b[2][:] = 1.

    def f(x):  # y = x**2
        y = 0
        for i in range(len(x)):
            y += np.sum(x[i] ** 2)
        return y

    def dfdx(x):  # dydx = 2*x
        dydx = []
        for i in range(len(x)):
            dydx.append(2 * x[i])
        return dydx

    p = parameters.stack()

    computed = jenn.finite_diff(f, p)
    expected = dfdx(p)

    for layer in range(parameters.L):
        msg = f'finite difference routine is wrong in layer {layer}'
        assert np.allclose(computed[layer], expected[layer], atol=1e-6), msg


def test_model_forward(xor):
    """Test forward propagation using XOR."""
    data, parameters, cache = xor
    computed = jenn.model_forward(data.X, parameters, cache)[0]
    expected = data.Y
    msg = f'computed = {computed} vs. expected = {expected}'
    assert np.all(computed == expected), msg


def test_model_backward(xor):
    """Test backward propagation against finite difference using XOR."""
    data, parameters, cache = xor

    jenn.model_forward(data.X, parameters, cache)  # predict to populate cache
    jenn.model_backward(data, parameters, cache)  # partials computed in place

    parameter_copy = deepcopy(parameters)  # de-conflict inplace updating
    cache_copy = deepcopy(cache)  # de-conflict inplace updating
    cost = jenn.Cost(data, parameter_copy)

    def cost_FD(x):
        parameter_copy.unstack(x)
        Y_pred = jenn.model_forward(data.X, parameter_copy, cache_copy)[0]
        return cost.evaluate(Y_pred)

    assert cost_FD(x=parameters.stack()) == 0.0, f'provided data is wrong'
    partials = jenn.finite_diff(cost_FD, parameter_copy.stack())
    parameter_copy.unstack_partials(partials)

    for layer in range(parameters.L):
        msg = f"partials in layer {layer} don't match finite difference"
        assert np.allclose(
            parameter_copy.dW[layer], parameters.dW[layer], atol=1e-6), msg
        assert np.allclose(
            parameter_copy.db[layer], parameters.db[layer], atol=1e-6), msg


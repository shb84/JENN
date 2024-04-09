"""Test forward and backward propagation using 
neural nets for which the correct value of the 
parameters is known exactly."""
import numpy as np
import pytest 
from copy import deepcopy
from typing import List

import jenn


def _finite_difference(cost, params, step=1e-6):
    """Use finite difference to compute partials of cost function
    with respect to neural net parameters (backpropagation)."""
    grads = list()
    dx = step
    for x in params:
        n, p = x.shape
        dy = np.zeros((n, p))
        for i in range(0, n):
            for j in range(0, p):
                # Forward step
                x[i, j] += dx
                y_fwd = cost(params)
                x[i, j] -= dx

                # Backward step
                x[i, j] -= dx
                y_bwd = cost(params)
                x[i, j] += dx

                # Central difference
                dy[i, j] = np.divide(y_fwd - y_bwd, 2 * dx)

        grads.append(dy)
    return grads


def _grad_check(dydx: List[np.ndarray], dydx_FD: List[np.ndarray],
                atol: float = 1e-6, rtol: float = 1e-4) -> bool:
    """
    Compare analytical gradient against finite difference

    Parameters
    ----------
    x: List[np.ndarray]
        Point at which to evaluate gradient

    f: callable
        Function handle to use for finite difference

    dx: float
        Finite difference step

    tol: float
        Tolerance below which agreement is considered acceptable
        Default = 1e-6

    verbose: bool
        Print output to standard out
        Default = True

    Returns
    -------
    success: bool
        Returns True iff finite difference and analytical grads agree
    """
    success = True
    for i in range(len(dydx)):
        if not np.allclose(dydx[i], dydx_FD[i], atol=atol, rtol=rtol):
            success = False
        if not success:
            msg = f"The gradients of layer {i} are wrong\n"
        else:
            msg = f"The gradients of layer {i} are correct\n"
        msg += f"Finite dif: grad[{i}] = {dydx_FD[i].squeeze()}\n"
        msg += f"Analytical: grad[{i}] = {dydx[i].squeeze()}\n\n"
        assert success, msg
    return success


X_test = np.array([[0, 1, 0, 1], [0, 0, 1, 1]])
Y_test = np.array([[0, 1, 1, 0]])


class TestXOR: 
    """Check forward and backprop on XOR test case."""

    @pytest.fixture
    def data(self) -> jenn.core.data.Dataset:
        """Return XOR test data."""
        return jenn.core.data.Dataset(X_test, Y_test)

    @pytest.fixture
    def cache(self) -> jenn.core.cache.Cache:
        """Return XOR cache."""
        return jenn.core.cache.Cache(layer_sizes=[2, 2, 1], m=Y_test.size)

    @pytest.fixture
    def params(self) -> jenn.core.parameters.Parameters:
        """Return XOR parameters."""
        parameters = jenn.core.parameters.Parameters(layer_sizes=[2, 2, 1], output_activation='relu')
        parameters.initialize()
        parameters.b[1][:] = np.array([[0], [-1]])        # layer 1
        parameters.W[1][:] = np.array([[1, 1], [1, 1]])   # layer 1
        parameters.b[2][:] = np.array([[0]])              # layer 2
        parameters.W[2][:] = np.array([[1, -2]])          # layer 2
        return parameters

    def test_model_forward(
            self, 
            data: jenn.core.data.Dataset, 
            params: jenn.core.parameters.Parameters, 
            cache: jenn.core.cache.Cache,
        ) -> None:
        """Test forward propagation using XOR."""
        computed = jenn.core.propagation.partials_forward(data.X, params, cache)
        expected = data.Y
        msg = f'computed = {computed} vs. expected = {expected}'
        assert np.all(computed == expected), msg


    def test_model_backward(
            self, 
            data: jenn.core.data.Dataset, 
            params: jenn.core.parameters.Parameters, 
            cache: jenn.core.cache.Cache,
        ) -> None:
        """Test backward propagation against finite difference."""

        ###########################
        # Perfectly trained model #
        ###########################

        jenn.core.propagation.model_partials_forward(
            data.X, params, cache)  # predict to populate cache

        jenn.core.propagation.model_backward(
            data, params, cache)  # partials computed in place

        dydx = params.stack_partials()
        assert np.allclose(dydx, 0.0)  # partials should be 0 at optimum params

        ###################
        # Imperfect model #
        ###################

        for i in range(params.L): # falsify model so partials are not zero
            params.W[i][:] += 10 * np.random.rand()
            params.b[i][:] += 10 * np.random.rand()

        jenn.core.propagation.model_partials_forward(
            data.X, params, cache)  # predict to populate cache

        jenn.core.propagation.model_backward(
            data, params, cache)  # partials computed in place

        def cost_FD(x):
            parameters = deepcopy(params)  # make copy b/c arrays updated in place
            cost = jenn.core.cost.Cost(data, parameters)
            parameters.unstack(x)
            Y_pred = jenn.core.propagation.model_forward(data.X, parameters, deepcopy(cache))
            return cost.evaluate(Y_pred)

        dydx = params.stack_partials_per_layer()
        dydx_FD = _finite_difference(cost_FD, params.stack_per_layer())

        assert _grad_check(dydx, dydx_FD)


# TODO: add test(s) for gradient-enhanced backprop and forward prop of partials
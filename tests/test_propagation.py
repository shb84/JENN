"""Test forward and backward propagation using
neural nets for which the correct value of the
parameters is known exactly.
"""

# Copyright (C) 2018 Steven H. Berguin
# This work is licensed under the MIT License.
from __future__ import annotations  # needed if python is 3.9

from copy import deepcopy

import numpy as np
import pytest

import jenn


def _finite_difference(cost, params, step=1e-6):
    """Use finite difference to compute partials of cost function
    with respect to neural net parameters (backpropagation).
    """
    grads = list()
    dx = step
    for x in params:
        n, p = x.shape
        dy = np.zeros((n, p))
        for i in range(n):
            for j in range(p):
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


def _grad_check(
    dydx: list[np.ndarray],
    dydx_FD: list[np.ndarray],
    atol: float = 1e-6,
    rtol: float = 1e-4,
) -> bool:
    """Compare analytical gradient against finite difference."""
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
        parameters = jenn.core.parameters.Parameters(
            layer_sizes=[2, 2, 1],
            hidden_activation="relu",
            output_activation="relu",
        )
        parameters.initialize()
        parameters.b[1][:] = np.array([[0], [-1]])  # layer 1
        parameters.W[1][:] = np.array([[1, 1], [1, 1]])  # layer 1
        parameters.b[2][:] = np.array([[0]])  # layer 2
        parameters.W[2][:] = np.array([[1, -2]])  # layer 2
        return parameters

    def test_model_forward(
        self,
        data: jenn.core.data.Dataset,
        params: jenn.core.parameters.Parameters,
        cache: jenn.core.cache.Cache,
    ) -> None:
        """Test forward propagation using XOR."""
        computed = jenn.core.propagation.model_forward(data.X, params, cache)
        expected = data.Y
        msg = f"computed = {computed} vs. expected = {expected}"
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
            data.X,
            params,
            cache,
        )  # predict to populate cache

        jenn.core.propagation.model_backward(
            data,
            params,
            cache,
        )  # partials computed in place

        dydx = params.stack_partials()
        assert np.allclose(dydx, 0.0)  # partials should be 0 at optimum params

        ###################
        # Imperfect model #
        ###################

        for i in range(params.L):  # falsify model so partials are not zero
            params.W[i][:] += 10 * np.random.rand()
            params.b[i][:] += 10 * np.random.rand()

        jenn.core.propagation.model_partials_forward(
            data.X,
            params,
            cache,
        )  # predict to populate cache

        jenn.core.propagation.model_backward(
            data,
            params,
            cache,
        )  # partials computed in place

        def cost_finite_diff(x):
            parameters = deepcopy(params)  # make copy b/c arrays updated in place
            cost = jenn.core.cost.Cost(data, parameters)
            parameters.unstack(x)
            Y_pred = jenn.core.propagation.model_forward(
                data.X,
                parameters,
                deepcopy(cache),
            )
            return cost.evaluate(Y_pred)

        dydx = params.stack_partials_per_layer()
        dydx_FD = _finite_difference(cost_finite_diff, params.stack_per_layer())

        assert _grad_check(dydx, dydx_FD)


class TestGradientEnhanced:
    """Check gradient-enhanced backprop for a multi-input problem (n_x >= 2).

    The regular finite-difference gradient check guards the standard backprop,
    but only ever exercises n_x = 1 (the 1D sinusoid). This class covers the
    ``for j in range(n_x)`` paths in ``next_layer_partials`` and
    ``gradient_enhancement`` with n_x = 2, so their vectorized forms stay
    numerically faithful.
    """

    @pytest.fixture
    def data(self) -> jenn.core.data.Dataset:
        """Return a small 2-input quadratic dataset with exact Jacobian.

        y = x0^2 + x1^2  =>  dy/dx0 = 2 x0,  dy/dx1 = 2 x1
        """
        rng = np.random.default_rng(0)
        X = rng.standard_normal((2, 5))  # (n_x, m)
        Y = np.sum(X**2, axis=0, keepdims=True)  # (n_y, m)
        J = np.zeros((1, 2, 5))  # (n_y, n_x, m)
        J[0, 0, :] = 2 * X[0, :]
        J[0, 1, :] = 2 * X[1, :]
        return jenn.core.data.Dataset(X, Y, J)

    @pytest.fixture
    def cache(self) -> jenn.core.cache.Cache:
        """Return cache sized to the 2-input dataset."""
        return jenn.core.cache.Cache(layer_sizes=[2, 3, 1], m=5)

    @pytest.fixture
    def params(self) -> jenn.core.parameters.Parameters:
        """Return randomly initialized parameters (tanh hidden layer).

        A tanh hidden layer is used deliberately so the second derivative
        (``G_prime_prime``) is non-zero and the gradient-enhancement terms
        are fully exercised.
        """
        parameters = jenn.core.parameters.Parameters(
            layer_sizes=[2, 3, 1],
            hidden_activation="tanh",
            output_activation="linear",
        )
        parameters.initialize(random_state=1)
        return parameters

    def test_gradient_enhanced_backward(
        self,
        data: jenn.core.data.Dataset,
        params: jenn.core.parameters.Parameters,
        cache: jenn.core.cache.Cache,
    ) -> None:
        """Check gradient-enhanced backprop against finite difference (n_x=2)."""
        jenn.core.propagation.model_partials_forward(data.X, params, cache)
        jenn.core.propagation.model_backward(data, params, cache)

        def cost_finite_diff(x):
            parameters = deepcopy(params)  # make copy b/c arrays updated in place
            cost = jenn.core.cost.Cost(data, parameters)
            parameters.unstack(x)
            Y_pred, J_pred = jenn.core.propagation.model_partials_forward(
                data.X,
                parameters,
                deepcopy(cache),
            )
            return cost.evaluate(Y_pred, J_pred)

        dydx = params.stack_partials_per_layer()
        dydx_FD = _finite_difference(cost_finite_diff, params.stack_per_layer())

        assert _grad_check(dydx, dydx_FD)


# TODO: add test(s) for forward prop of partials

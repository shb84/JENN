"""Test that forward and backward propagation are working."""
import numpy as np
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


def test_model_forward(xor):
    """Test forward propagation using XOR."""
    data, parameters, cache = xor
    computed = jenn.core.partials_forward(data.X, parameters, cache)
    expected = data.Y
    msg = f'computed = {computed} vs. expected = {expected}'
    assert np.all(computed == expected), msg


def test_model_backward(xor):
    """Test backward propagation against finite difference using XOR."""
    data, parameters, cache = xor

    ###########################
    # Perfectly trained model #
    ###########################

    jenn.core.model_partials_forward(
        data.X, parameters, cache)  # predict to populate cache

    jenn.core.model_backward(
        data, parameters, cache)  # partials computed in place

    dydx = parameters.stack_partials(per_layer=False)

    assert np.allclose(dydx, 0.0)  # partials should be 0 at optimum params

    ###################
    # Imperfect model #
    ###################

    for i in range(parameters.L): # falsify model so partials are not zero
        parameters.W[i][:] += 10 * np.random.rand()
        parameters.b[i][:] += 10 * np.random.rand()

    jenn.core.model_partials_forward(
        data.X, parameters, cache)  # predict to populate cache

    jenn.core.model_backward(
        data, parameters, cache)  # partials computed in place

    def cost_FD(x):
        params = deepcopy(parameters)  # make copy b/c arrays updated in place
        cost = jenn.core.Cost(data, params)
        params.unstack(x)
        Y_pred = jenn.core.model_forward(data.X, params, deepcopy(cache))
        return cost.evaluate(Y_pred)

    dydx = parameters.stack_partials(per_layer=True)
    dydx_FD = _finite_difference(cost_FD, parameters.stack(per_layer=True))

    assert _grad_check(dydx, dydx_FD)

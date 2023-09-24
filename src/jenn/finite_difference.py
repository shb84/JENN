"""Finite difference for checking gradients."""
import numpy as np
from typing import List


def finite_diff(cost, params, step=1e-6):
    """Use finite difference to compute partials of cost function
    with respect to neural net parameters."""
    grads = list()
    dx = step
    for k, x in enumerate(params):
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


def grad_check(
        dydx: List[np.ndarray], dydx_FD: List[np.ndarray],
        tol: float = 1e-6, verbose: bool = True) -> bool:
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
        numerator = np.linalg.norm(dydx[i].squeeze() - dydx_FD[i].squeeze())
        denominator = np.linalg.norm(dydx[i].squeeze()) + np.linalg.norm(
            dydx_FD[i].squeeze())
        if denominator == 0.0:
            denominator += 1e-12
        difference = numerator / denominator
        if difference > tol or numerator > tol:
            success = False
        if verbose:
            if not success:
                print(f"The gradients of layer {i} are wrong")
            else:
                print(f"The gradients of layer {i} are correct")
            print(f"Finite dif: grad[{i}] = {dydx_FD[i].squeeze()}")
            print(f"Analytical: grad[{i}] = {dydx[i].squeeze()}")
            print()
    return success

"""Test that activation functions are corrected."""
import numpy as np


def model_backward_FD(cost, params, step=1e-6):
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


def grad_check(
        dydx: list[np.ndarray], dydx_FD: list[np.ndarray],
        tol: float = 1e-6, verbose: bool = True) -> bool:
    """
    Compare analytical gradient against finite difference

    Parameters
    ----------
    x: list[np.ndarray]
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


# Not an efficient implementation, but okay for testing
def _forward_difference(f: callable, x: np.ndarray, dx: float = 1e-6):
    """Compute partials of y = f(x) using forward difference."""
    n_x, m = x.shape
    dy = np.zeros((n_x, m))
    for i in range(0, n_x):
        dy[i] = np.divide(f(x+dx) - f(x), dx)
    return dy


# Not an efficient implementation, but okay for testing
def _backward_difference(f: callable, x: np.ndarray, dx: float = 1e-6):
    """Compute partials of y = f(x) using backward difference."""
    n_x, m = x.shape
    dy = np.zeros((n_x, m))
    for i in range(0, n_x):
        dy[i] = np.divide(f(x) - f(x-dx), dx)
    return dy


# Not an efficient implementation, but okay for testing
def _central_difference(f: callable, x: np.ndarray, dx: float = 1e-6):
    """Compute partials of y = f(x) using central difference."""
    n_x, m = x.shape
    dy = np.zeros((n_x, m))
    for i in range(0, n_x):
        dy[i] = np.divide(f(x+dx) - f(x-dx), 2 * dx)
    return dy


def finite_difference(f: callable, x: np.ndarray, dx: float = 1e-6):
    """Compute partials of y = f(x) using ctr, fwd or bwd difference."""
    dy = _central_difference(f, x, dx)
    dy[:, :1] = _forward_difference(f, x, dx)[:, :1]
    dy[:, -1:] = _backward_difference(f, x, dx)[:, -1:]
    return dy


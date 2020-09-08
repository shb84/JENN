"""
G R A D I E N T - E N H A N C E D   N E U R A L   N E T W O R K S  (G E N N)

Author: Steven H. Berguin <stevenberguin@gmail.com>

This package is distributed under the MIT license.
"""
import numpy as np
import math
from typing import List

from importlib.util import find_spec

if find_spec("matplotlib"):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_INSTALLED = True
else:
    MATPLOTLIB_INSTALLED = False

EPS = np.finfo(float).eps  # small number to avoid division by zero


def mini_batches(X: np.ndarray, batch_size: int,
                 shuffle: bool = True, seed: int = None) -> list:
    """
    Create randomized mini-batches by returning a list of tuples, where
    each tuple contains the indices of the training data points associated with
    that mini-batch

    Parameters
    ----------
        X: np.ndarray
            input features of the training data
            shape = (n_x, m) where m = num of examples and n_x = num of inputs

        batch_size: int
            mini batch size (if None, then batch_size = m)
            
        shuffle: bool 
            Shuffle data points
            Default = True 
            
        seed: int 
            Random seed (set to make runs repeatable)
            Default = None  

    Returns
    -------
        mini_batches: list
            List of mini-batch indices to use for slicing data, where the index
            is in the interval [1, m]
    """

    np.random.seed(seed)

    batches = []
    m = X.shape[1]
    if not batch_size:
        batch_size = m
    batch_size = min(batch_size, m)

    # Step 1: Shuffle the indices
    if shuffle:
        indices = list(np.random.permutation(m))
    else:
        indices = np.arange(m)

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = int(math.floor(m / batch_size))
    k = 0
    for _ in range(num_complete_minibatches):
        batch = indices[k * batch_size:(k + 1) * batch_size]
        batches.append(tuple(batch))
        k += 1

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % batch_size != 0:
        mini_batch = indices[(k + 1) * batch_size:]
        batches.append(tuple(mini_batch))

    return batches


def finite_diff(params: List[np.ndarray], f: callable, dx: float = 1e-6):
    """
    Compute gradient using central difference

    Parameters
    ----------
    params: List[np.ndarray]
        Point at which to evaluate gradient

    f: callable
        Function handle to use for finite difference

    dx: float
        Finite difference step

    Returns
    -------
    dy_dx: List[np.ndarray]
        Derivative of f w.r.t. x
    """
    grads = list()
    for k, x in enumerate(params):
        n, p = x.shape
        dy = np.zeros((n, p))
        for i in range(0, n):
            for j in range(0, p):
                # Forward step
                x[i, j] += dx
                y_fwd = f(params)
                x[i, j] -= dx

                # Backward step
                x[i, j] -= dx
                y_bwd = f(params)
                x[i, j] += dx

                # Central difference
                dy[i, j] = np.divide(y_fwd - y_bwd, 2 * dx)

        grads.append(dy)

    return grads


def grad_check(x: List[np.ndarray], f: callable, dfdx: callable,
               dx: float = 1e-6, tol: float = 1e-6,
               verbose: bool = True) -> bool:
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
    dydx = dfdx(x)
    dydx_FD = finite_diff(x, f, dx)
    for i in range(len(x)):
        numerator = np.linalg.norm(dydx[i].squeeze() - dydx_FD[i].squeeze())
        denominator = np.linalg.norm(dydx[i].squeeze()) + np.linalg.norm(
            dydx_FD[i].squeeze())
        difference = numerator / (denominator + EPS)
        if difference > tol or numerator > tol:
            success = False
        if verbose:
            if not success:
                print(f"The gradient w.r.t. x[{i}] is wrong")
            else:
                print(f"The gradient w.r.t. x[{i}] is correct")
            print(f"Finite dif: grad[{i}] = {dydx_FD[i].squeeze()}")
            print(f"Analytical: grad[{i}] = {dydx[i].squeeze()}")
            print()
    return success


def rsquare(y_pred, y_true):
    epsilon = 1e-12  # small number to avoid division by zero
    y_bar = np.mean(y_true)
    SSE = np.sum(np.square(y_pred - y_true))
    SSTO = np.sum(np.square(y_true - y_bar) + epsilon)
    R2 = 1 - SSE / SSTO
    return R2


def goodness_of_fit(y_pred: np.ndarray, y_true: np.ndarray,
                    title: str = None, legend: str = None):
    """
    Plot actual by predicted and histogram of prediction error

    Parameters
    ----------

    y_pred: np.ndarray
        Predicted values
        shape = (m,) where m = no. examples

    y_true: np.ndarray
        True values
        shape = (m,) where m = no. examples

    title: str
        Title to be displayed on figure
        Default = None => 'Goodness of Fit'

    legend: str
        Labeled to be displayed in legend for data
        Default = None => 'data samples'
    """
    # Prepare to plot
    if not MATPLOTLIB_INSTALLED:
        raise ImportError("Matplotlib must be installed.")

    if not title:
        title = 'Goodness of Fit'

    if not legend:
        legend = 'data samples'

    r_squared = rsquare(y_pred, y_true)
    std_error = np.std(y_pred - y_true)
    avg_error = np.mean(y_pred - y_true)

    # Reference Line
    y = np.linspace(np.min(y_true), np.max(y_true), 100)

    fig = plt.figure(figsize=(12, 6))
    fig.suptitle(title, fontsize=16)
    spec = gridspec.GridSpec(ncols=2, nrows=1, wspace=0.25)

    # Plot
    ax1 = fig.add_subplot(spec[0, 0])
    ax1.plot(y, y, 'r-')
    ax1.scatter(y_true, y_pred, s=100, c='k', marker=".")
    ax1.legend(["perfect fit line", legend])
    ax1.set_xlabel("actual")
    ax1.set_ylabel("predicted")
    ax1.set_title(f"RSquare = {r_squared:.2f}")
    plt.grid(True)

    ax2 = fig.add_subplot(spec[0, 1])
    error = (y_pred - y_true)
    weights = np.ones(error.shape) / y_pred.size
    ax2.hist(error, weights=weights, facecolor='g', alpha=0.75)
    ax2.set_xlabel('Absolute Prediction Error')
    ax2.set_ylabel('Probability')
    ax2.set_title(f'$\mu$ = {avg_error:.2f}, $\sigma$ = {std_error:.2f}')
    plt.grid(True)
    plt.show()


def normalize_data(X: np.ndarray,
                   Y: np.ndarray,
                   J: np.ndarray = None) -> tuple:
    """
    Normalize training data to help with optimization,
        i.e. X_norm = (X - mu_x) / sigma_x where X is as below
             Y_norm = (Y - mu_y) / sigma_y where Y is as below
             J_norm = J * sigma_x/sigma_y

    Concretely, normalizing training data is essential because the neural
    learns by minimizing a cost function. Normalizing the data therefore
    rescales the problem in a way that aides the optimizer.

    Parameters
    ----------
    X: np.ndarray
        Input features of shape (n_x, m) where n_x = no. of inputs
                                               m = no. of training examples

    Y: np.ndarray
        Output labels of shape (n_y, m) where n_y = no. of outputs

    J: np.ndarray
        Jacobian of shape (n_y, n_x, m) representing the partials of Y w.r.t. X
            dY1/dX1 = J[0][0]
            dY1/dX2 = J[0][1]
            ...
            dY2/dX1 = J[1][0]
            dY2/dX2 = J[1][1]
            ...
            N.B. To retrieve the i^th example for dY2/dX1: J[1][0][i]
                 for all i = 1,...,m

    Returns
    -------
        X_norm, Y_norm, J_norm, mu_x, sigma_x, mu_y, sigma_y: tuple
            Normalized data and associated scale factors
    """
    # Initialize
    X_norm = np.zeros(X.shape)
    Y_norm = np.zeros(Y.shape)
    if J is not None:
        J_norm = np.zeros(J.shape)
    else:
        J_norm = None

    # Dimensions
    n_x, m = X.shape
    n_y, _ = Y.shape

    # Normalize inputs
    mu_x = np.zeros((n_x, 1))
    sigma_x = np.ones((n_x, 1))
    for i in range(0, n_x):
        mu_x[i] = np.mean(X[i])
        sigma_x[i] = np.std(X[i])
        X_norm[i] = (X[i] - mu_x[i]) / sigma_x[i]

    # Normalize outputs
    mu_y = np.zeros((n_y, 1))
    sigma_y = np.ones((n_y, 1))
    for i in range(0, n_y):
        mu_y[i] = np.mean(Y[i])
        sigma_y[i] = np.std(Y[i])
        Y_norm[i] = (Y[i] - mu_y[i]) / sigma_y[i]

    # Normalize partials
    if J is not None:
        for i in range(0, n_y):
            for j in range(0, n_x):
                J_norm[i, j] = J[i, j] * sigma_x[j] / sigma_y[i]

    return X_norm, Y_norm, J_norm, mu_x, sigma_x, mu_y, sigma_y

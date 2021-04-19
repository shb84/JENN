"""
J A C O B I A N - E N H A N C E D   N E U R A L   N E T W O R K S  (J E N N)

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
                 shuffle: bool = True, random_state: int = None) -> list:
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
            
        random_state: int 
            Random seed (set to make runs repeatable)
            Default = None  

    Returns
    -------
        mini_batches: list
            List of mini-batch indices to use for slicing data, where the index
            is in the interval [1, m]
    """

    np.random.seed(random_state)

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
        mini_batch = indices[k * batch_size:(k + 1) * batch_size]
        if mini_batch:
            batches.append(tuple(mini_batch))
        k += 1

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % batch_size != 0:
        mini_batch = indices[(k + 1) * batch_size:]
        if mini_batch:
            batches.append(tuple(mini_batch))

    return batches


def finite_diff(params: List[np.ndarray], f: callable, dx: float = 1e-6):
    """
    Compute gradient using central difference

    Parameters
    ----------
    params: List[np.ndarray]
        List of neural net parameters at which to evaluate gradient

    f: callable
        Function handle to use for finite difference

    dx: float
        Finite difference step

    Returns
    -------
    dy_dx: List[np.ndarray]
        Derivative of f w.r.t. x
    """
    is_list = True
    if type(params) is np.ndarray:
        params = [params]
        is_list = False
    grads = list()
    for k, x in enumerate(params):
        n, p = x.shape
        dy = np.zeros((n, p))
        for i in range(0, n):
            for j in range(0, p):
                # Forward step
                x[i, j] += dx
                if is_list:
                    y_fwd = f(params)
                else:
                   y_fwd = f(params[0])
                x[i, j] -= dx

                # Backward step
                x[i, j] -= dx
                if is_list:
                    y_bwd = f(params)
                else:
                    y_bwd = f(params[0])
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
                    title: str = None, legend: str = None,
                    show_error: bool = True):
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

    ncols = 1 + show_error
    fig = plt.figure(figsize=(6 * ncols, 6))
    fig.suptitle(title, fontsize=16)
    spec = gridspec.GridSpec(ncols=ncols, nrows=1, wspace=0.25)

    # Plot
    ax1 = fig.add_subplot(spec[0, 0])
    ax1.plot(y, y, 'r-')
    ax1.scatter(y_true, y_pred, s=100, c='k', marker=".")
    ax1.legend(["perfect fit line", legend])
    ax1.set_xlabel("actual")
    ax1.set_ylabel("predicted")
    ax1.set_title(f"RSquare = {r_squared:.2f}")
    plt.grid(True)

    if not show_error:
        return fig

    ax2 = fig.add_subplot(spec[0, 1])
    error = (y_pred - y_true)
    weights = np.ones(error.shape) / y_pred.size
    ax2.hist(error, weights=weights, facecolor='g', alpha=0.75, bins=20)
    ax2.set_xlabel('Absolute Prediction Error')
    ax2.set_ylabel('Probability')
    ax2.set_title(f'$\mu$ = {avg_error:.2f}, $\sigma$ = {std_error:.2f}')
    plt.grid(True)
    plt.show()

    return fig


class DataConverter:
    """ Helper class to normalize and denormalize data """

    def X(self, X_norm: np.ndarray):
        """
        Return denormalized inputs: X = X_norm * sigma_x + mu_x

        Parameters
        ----------
        X_norm: np.ndarray
            Normalized input, shape (n_x, m) where n_x = no. of inputs
                                                   m = no. of training examples
        """
        return X_norm * self._sigma_x + self._mu_x

    def Y(self, Y_norm: np.ndarray):
        """
        Return denormalized outputs: Y = Y_norm * sigma_y + mu_y

        Parameters
        ----------
        Y_norm: np.ndarray
            Normalized output, shape (n_y, m) where n_y = no. of outputs
                                                   m = no. of training examples
        """
        return Y_norm * self._sigma_y + self._mu_y

    def J(self, J_norm: np.ndarray):
        """
        Return denormalized Jacobian: J_norm = J_norm * sigma_y / sigma_x

        Parameters
        ----------
        J_norm: np.ndarray
            Normalized Jacobian, shape (n_y, n_x, m) where
                                                   n_y = no. of outputs
                                                   n_x = no. of inputs
                                                   m = no. of training examples
        """
        n_y, n_x, _ = J_norm.shape
        sigma_x = self._sigma_x.T.reshape((1, n_x, 1))
        sigma_y = self._sigma_y.reshape((n_y, 1, 1))
        return J_norm * sigma_y / sigma_x

    def X_norm(self, X: np.ndarray):
        """
        Return normalized inputs: X_norm = (X - mu_x) / sigma_y

        Parameters
        ----------
        X: np.ndarray
            Input, shape (n_x, m) where n_x = no. of inputs
                                        m = no. of training examples
        """
        return (X - self._mu_x) / (self._sigma_x + 1e-12)

    def Y_norm(self, Y: np.ndarray):
        """
        Return normalized outputs: Y_norm = (Y - mu_y) / sigma_y

        Parameters
        ----------
        Y: np.ndarray
            Output, shape (n_y, m) where n_y = no. of outputs
                                         m = no. of training examples
        """
        return (Y - self._mu_y) / (self._sigma_y + 1e-12)

    def J_norm(self, J: np.ndarray):
        """
        Return normalized Jacobian: J_norm = J * sigma_x / sigma_y

        Parameters
        ----------
        J: np.ndarray
            Jacobian, shape (n_y, n_x, m) where    n_y = no. of outputs
                                                   n_x = no. of inputs
                                                   m = no. of training examples
        """
        if J is not None:
            n_y, n_x, _ = J.shape
            sigma_x = self._sigma_x.T.reshape((1, n_x, 1))
            sigma_y = self._sigma_y.reshape((n_y, 1, 1))
            return J * sigma_x / sigma_y

    def __init__(self,
                 mu_x: np.ndarray, sigma_x: np.ndarray,
                 mu_y: np.ndarray, sigma_y: np.ndarray):
        """
        X_norm = (X - mu_x) / sigma_y
        Y_norm = (Y - mu_y) / sigma_y

        Parameters
        ----------
        mu_x: np.ndarray
            Array of shape (n_x, 1) where n_x = no. of inputs

        mu_y: np.ndarray
            Array of shape (n_y, 1) where n_y = no. of outputs

        sigma_x: np.ndarray
            Array of shape (n_x, 1) where n_x = no. of inputs

        sigma_y: np.ndarray
            Array of shape (n_y, 1) where n_y = no. of outputs
        """
        self._mu_x = mu_x
        self._mu_y = mu_y
        self._sigma_x = sigma_x
        self._sigma_y = sigma_y


def scale_factor(array: np.ndarray) -> tuple:
    """
    Find mu and sigma such that X_norm = (X - mu) / sigma

    Parameters
    ----------
    array: np.ndarray
        Input features of shape (n, m) where n = dimensionality
                                             m = number of examples
    """
    mu = np.mean(array, axis=1).reshape((-1, 1))
    sigma = np.std(array, axis=1).reshape((-1, 1))
    return mu, sigma


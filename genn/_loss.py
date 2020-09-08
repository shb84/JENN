"""
G R A D I E N T - E N H A N C E D   N E U R A L   N E T W O R K S  (G E N N)

Author: Steven H. Berguin <stevenberguin@gmail.com>

This package is distributed under the MIT license.
"""
import numpy as np
from typing import List

from ._fwd_prop import L_model_forward, L_grads_forward
from ._bwd_prop import L_model_backward


def regularization(w: List[np.ndarray], m: int, alpha: float = 0.) -> np.float:
    """
    Compute L2 norm penalty

    Parameters
    ----------
        w: List[np.ndarray]
            Weight parameters associated with each layer of neural net

        m: int
            Number of examples in training data

        alpha: float
            Regularization coefficient
    """
    alpha = max(0., alpha)  # ensure 0 < lambda
    penalty = 0.0
    for theta in w:
        penalty += np.squeeze(0.5 * alpha * np.sum(np.square(theta)))
    return 1. / m * np.float(penalty)


def gradient_enhancement(dy_true: np.ndarray, dy_pred: np.ndarray,
                         gamma: float = 1.) -> np.float:
    """
    Compute least squares estimator for the partials

    Parameters
    ----------
        dy_pred: np ndarray of shape (n_y, n_x, m)
            Predicted partials: AL' = d(AL)/dX where n_y = number outputs
                                                     n_x = number inputs
                                                     m = number examples

        dy_true: np ndarray of shape (n_y, n_x, m)
            True partials: Y' = d(Y)/dX

        gamma: float
            Regularization hyper-parameter
            Default = 1.0

    Returns
    -------
        loss: np.ndarray of shape (1,)
    """
    n_y, n_x, m = dy_pred.shape  # number of outputs, inputs, training examples
    loss = 0.
    gamma = max(0., gamma)  # ensure 0 < gamma
    for k in range(0, n_y):
        for j in range(0, n_x):
            dy_j_pred = dy_pred[k, j, :].reshape(1, m)
            dy_j_true = dy_true[k, j, :].reshape(1, m)
            loss += np.squeeze(0.5 * gamma * np.dot((dy_j_pred - dy_j_true),
                                                    (dy_j_pred - dy_j_true).T))
    return 1. / m * np.float(loss)


def squared_loss(y_true: np.ndarray, y_pred: np.ndarray) -> np.float:
    """
    Compute least squares estimator for the states

    Parameters
    ----------
        y_pred: np.ndarray of shape (n_y, m)
            Predicted outputs where n_y = no. outputs, m = no. examples

        y_true: np.ndarray of shape (n_y, m)
            True outputs where n_y = no. outputs, m = no. examples
    """
    n_y, m = y_true.shape  # number of outputs, training examples
    cost = 0.
    for k in range(0, n_y):
        cost += np.squeeze(0.5 * np.dot((y_pred[k, :] - y_true[k, :]),
                                        (y_pred[k, :] - y_true[k, :]).T))

    return 1. / m * cost


def cost(W: List[np.ndarray], b: List[np.ndarray], a: List[str],
         X: np.ndarray, Y: np.ndarray, J: np.ndarray = None,
         lambd: float = 0, gamma: float = 0, is_grad: bool = True):
    """
    
    Parameters
    ----------
    W: List[np.ndarray]
    b: List[np.ndarray]
    a: List[str]
    X: np.ndarray
    Y: np.ndarray
    J: np.ndarray
    lambd: float
    gamma: float
    is_grad: bool

    Return
    ------
    c, dW, db : Tuple[np.float, np.ndarray, np.ndarray]
        c = cost
        dW = d(cost)/dW = derivative of cost w.r.t. neural net weights
        db = d(cost)/db = derivative of cost w.r.t. neural net biases

    """
    # Predict
    Y_pred, caches = L_model_forward(X, W, b, a)
    J_pred, J_caches = L_grads_forward(X, W, b, a)

    # Cost function
    c = 0
    c = c + squared_loss(Y_pred, Y)
    c = c + regularization(W, X.shape[1], lambd)
    if J is not None:
        c = c + gradient_enhancement(J_pred, J, gamma)

    # Cost function gradient
    dW = []
    db = []
    if is_grad:
        dW, db = L_model_backward(Y_pred, Y, caches,
                                  J_pred, J, J_caches, lambd, gamma)
    return c, dW, db

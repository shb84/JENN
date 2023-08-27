"""
J A C O B I A N - E N H A N C E D   N E U R A L   N E T W O R K S  (J E N N)

Author: Steven H. Berguin <stevenberguin@gmail.com>

This package is distributed under the MIT license.
"""
import numpy as np
from typing import List
from ._parameters import Parameters
from ._cache import Cache
from ._activation import ACTIVATIONS


def L_linear_forward(
        layer: int,
        parameters: Parameters,
        cache: Cache,
        indices: List[int],
):
    i = layer - 1
    W = parameters.W[i]
    b = parameters.b[i]
    Z = cache.Z[layer][:, indices]
    A_prev = cache.A[layer-1][:, indices]
    Z[:] = np.dot(W, A_prev) + b
    return Z


def L_layer_forward(
        layer: int,
        parameters: Parameters,
        cache: Cache,
        indices: List[int],
        activation: str = 'relu',
    ):
    A = cache.A[layer][:, indices]
    if layer == 0:
        return A
    Z = L_linear_forward(layer, parameters, cache, indices)
    if layer < parameters.L:  # hidden layers
        ACTIVATIONS[activation].evaluate(Z, A)
    else:  # output layer
        ACTIVATIONS['linear'].evaluate(Z, A)
    return A


def L_model_forward(
    X: np.ndarray,
    parameters: Parameters,
    cache: Cache,
    activation: str = 'relu',
    indices: List[int] = None,
):
    """
    Implements forward propagation for the entire neural network

    Parameters
    ----------
    X: np.ndarray
        input layer activations
        shape = (n_x, m) where n_x = number of inputs
                               m = number of training examples

    parameters: Parameters
        A pre-allocated Parameters object that holds the weights and biases:
            W: List[np.ndarray]
                List containing weights associated with each layer
                shape = (n_i, n[i-1]) where n_i = no. nodes in i^th layer

            b: List[np.ndarray]
                List containing intercepts associated with each layer
                shape = (n_i, 1) where n_i = no. nodes in i^th layer

    cache: Cache
        A pre-allocated Cache object which holds previously computed
        for each layer, such as activation values, etc. (see Cache doc).
        A cache is used during training for computational efficiency.

    activation: str
        Hidden layer activation function key: 'relu', 'tanh', 'linear'
        Default is 'relu'

    indices: List[int], optional
        Subset of indices over which to evaluate function. Default is None,
        which implies all indices (useful for minibatch for example).

    is_grad: bool, optional
        Compute gradient. Default is False.

    Return
    ------
    AL: np.ndarray
        Post-activation value of output layer (layer L)
        shape = (n_y, m) where n_y = number of outputs
                               m = number of training examples

    OR

    JL: np.ndarray
        Jacobian of w.r.t. X (i.e. partials of outputs w.r.t inputs)
        shape = (n_y, n_x, m) where n_y = no. of outputs
    """
    if indices is None:
        indices = range(X.shape[1])
    cache.A[0][:, indices] = X[:, indices]  # input layer
    for layer in range(1, parameters.L):
        A = L_layer_forward(layer, parameters, cache, indices, activation)
    return A


# def L_model_forward(
#     X: np.ndarray,
#     parameters: Parameters,
#     cache: Cache,
#     activation: str = 'relu',
#     indices: List[int] = None,
#     is_grad: bool = False):
#     """
#     Implements forward propagation for the entire neural network
#
#     Parameters
#     ----------
#     X: np.ndarray
#         input layer activations
#         shape = (n_x, m) where n_x = number of inputs
#                                m = number of training examples
#
#     parameters: Parameters
#         A pre-allocated Parameters object that holds the weights and biases:
#             W: List[np.ndarray]
#                 List containing weights associated with each layer
#                 shape = (n_i, n[i-1]) where n_i = no. nodes in i^th layer
#
#             b: List[np.ndarray]
#                 List containing intercepts associated with each layer
#                 shape = (n_i, 1) where n_i = no. nodes in i^th layer
#
#     cache: Cache
#         A pre-allocated Cache object which holds previously computed
#         for each layer, such as activation values, etc. (see Cache doc).
#         A cache is used during training for computational efficiency.
#
#     activation: str
#         Hidden layer activation function key: 'relu', 'tanh', 'linear'
#         Default is 'relu'
#
#     indices: List[int], optional
#         Subset of indices over which to evaluate function. Default is None,
#         which implies all indices (useful for minibatch for example).
#
#     is_grad: bool, optional
#         Compute gradient. Default is False.
#
#     Return
#     ------
#     AL: np.ndarray
#         Post-activation value of output layer (layer L)
#         shape = (n_y, m) where n_y = number of outputs
#                                m = number of training examples
#
#     OR
#
#     JL: np.ndarray
#         Jacobian of w.r.t. X (i.e. partials of outputs w.r.t inputs)
#         shape = (n_y, n_x, m) where n_y = no. of outputs
#     """
#     if indices is None:
#         indices = range(X.shape[1])
#     n_x, m = X.shape  # number of inputs, number of examples
#     A = X
#     I = np.eye(n_x, dtype=float)
#     J = np.repeat(I.reshape((n_x, n_x, 1)), m, axis=2)
#     L = len(parameters.W)
#
#     # Loop over layers
#     for i in range(1, L):
#         A_prev = A
#         J_prev = J
#         W = parameters.W[i]
#         b = parameters.b[i]
#         Z = cache.Z[i]  # pre-allocated array
#         A = cache.A[i]
#         Z[:, indices] = np.dot(W, A_prev[:, indices]) + b
#         if i < L - 1:  # hidden layers
#             ACTIVATIONS[activation].evaluate(Z[:, indices], A[:, indices])
#         else:  # output layer
#             ACTIVATIONS['linear'].evaluate(Z[:, indices], A[:, indices])
#
#         # Loop over partials
#         if is_grad:
#             J = cache.J[i]
#             G_prime = cache.G_prime[i]
#             for j in range(n_x):
#                 if i < L - 1:  # hidden layers
#                     ACTIVATIONS[activation].first_derivative(
#                         Z[:, indices], A[:, indices], G_prime[:, indices])
#                 else:  # output layer
#                     ACTIVATIONS['linear'].first_derivative(
#                         Z[:, indices], A[:, indices], G_prime[:, indices])
#                 J[:, j, indices] = G_prime * np.dot(W, J_prev[:, j, indices])
#     if is_grad:
#         return J[:, :, indices]
#     return A[:, indices]


# def L_grads_forward(
#     X: np.ndarray,
#     parameters: Parameters,
#     cache: Cache,
#     activation: str = 'relu',
#     indices: List[int] = None):
#     """
#     Implements forward propagation for the entire neural network
#
#     Parameters
#     ----------
#     X: np.ndarray
#         input layer activations
#         shape = (n_x, m) where n_x = number of inputs
#                                m = number of training examples
#
#     parameters: Parameters
#         A pre-allocated Parameters object that holds the weights and biases:
#             W: List[np.ndarray]
#                 List containing weights associated with each layer
#                 shape = (n_i, n[i-1]) where n_i = no. nodes in i^th layer
#
#             b: List[np.ndarray]
#                 List containing intercepts associated with each layer
#                 shape = (n_i, 1) where n_i = no. nodes in i^th layer
#
#     cache: Cache
#         A pre-allocated Cache object which holds previously computed
#         for each layer, such as activation values, etc. (see Cache doc).
#         A cache is used during training for computational efficiency.
#
#     activation: str
#         Hidden layer activation function key: 'relu', 'tanh', 'linear'
#         Default is 'relu'
#
#     indices: List[int], optional
#         Subset of indices over which to evaluate function. Default is None,
#         which implies all indices (useful for minibatch for example).
#
#     Return
#     ------
#     AL: np.ndarray
#         Post-activation value of output layer (layer L)
#         shape = (n_y, m) where n_y = number of outputs
#                                m = number of training examples
#
#     OR
#
#     JL: np.ndarray
#         Jacobian of w.r.t. X (i.e. partials of outputs w.r.t inputs)
#         shape = (n_y, n_x, m) where n_y = no. of outputs
#     """
#     return L_model_forward(
#         X, parameters, cache, activation, indices, is_grad=True)




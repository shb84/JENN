"""
J A C O B I A N - E N H A N C E D   N E U R A L   N E T W O R K S  (J E N N)

Author: Steven H. Berguin <stevenberguin@gmail.com>

This package is distributed under the MIT license.
"""

from ._activation import ACTIVATIONS

import numpy as np
from typing import List


def linear_activation_forward(A_prev: np.ndarray, W: np.ndarray, b: np.ndarray,
                              activation: str, store_cache: bool = True):
    """
    Implements forward propagation for one layer

    Parameters
    ----------
    A_prev: np.ndarray
        activations from previous layer
        shape = (n[l-1], 1) where n[l-1] = no. nodes in previous layer

    W: np.ndarray
        weights associated with current layer l
        shape = (n_l, n[l-1]) where n_l = no. nodes in current layer

    b: np.ndarray
        intercepts associated with current layer
        shape = numpy array of size (n_l, 1)

    activation: str
        layer activation type = {'identity', 'tanh', 'logistic', 'relu'}

    store_cache: bool
        True = do not compute cache (applies to trained model)
        False = compute cache (applies to untrained model) = default

    Return
    ------
    A: np.ndarray
        post-activation values for current layer
        shape = numpy array of size (n_l, 1)

    cache: tuple
        computed parameters that can be re-used for back-prop
        to avoid duplicate computations. Tuple contains:

            A_prev = numpy array of shape (n[l-1], m)
                     contains previous layer post-activation values where:
                        n[l-1] = no. nodes in previous layer
                        m = no. of training examples

            Z = numpy array of shape (n[l], m)
                contains linear forward values
                where n_l = number of nodes in current layer

            W = numpy array of shape (n[l], n[l-1]) containing weights
                of the current layer

            b = numpy array of shape (n[l], 1) containing biases
                of the current layer
    """
    g = ACTIVATIONS[activation]
    Z = np.dot(W, A_prev) + b
    A = g.evaluate(Z)
    if store_cache:
        cache = (A_prev, Z, W, b, activation)
    else:
        cache = None
    return A, cache


def L_model_forward(X: np.ndarray, W: List[np.ndarray], b: List[np.ndarray],
                    activations: List[str], store_cache: bool = True):
    """
    Implements forward propagation for the entire neural network

    Parameters
    ----------
    X: np.ndarray
        input layer activations
        shape = (n_x, m) where n_x = number of inputs
                               m = number of training examples

    W: List[np.ndarray]
        List containing weights associated with each layer
        shape = (n_i, n[i-1]) where n_i = no. nodes in i^th layer

    b: List[np.ndarray]
        List containing intercepts associated with each layer
        shape = (n_i, 1) where n_i = no. nodes in i^th layer

    activations: List[str]
        List containing activation types for each layer (hidden + output layer)
        Allowed values = {'identity', 'tanh', 'logistic', 'relu'}

    store_cache: bool
        True = do not compute cache (applies to trained model)
        False = compute cache (applies to untrained model) = default

    Return
    ------
    AL: np.ndarray
        post-activation value of output layer (layer L)
        shape = (n_y, m) where n_y = number of outputs
                               m = number of training examples

    caches: List[tuple]
        Only returned if fast = False
        List containing the cache for each layer where:
            cache = computed parameters that can be re-used for back-prop
                    to avoid duplicate computations. Tuple contains:

                    A_prev = numpy array of shape (n[l-1], m)
                             contains previous layer post-activation values:
                                n[l-1] = no. nodes in previous layer
                                m = no. of training examples

                    Z = numpy array of shape (n[l], m)
                        contains linear forward values where
                        n_l = number of nodes in current layer

                    W = numpy array of shape (n[l], n[l-1])
                        containing weights of the current layer

                    b = numpy array of shape (n[l], 1)
                        containing biases of the current layer
    """
    caches = []
    A = X
    L = len(activations)  # num layers in network (doesn't include input layer)
    for l in range(L):
        A, cache = linear_activation_forward(A, W[l], b[l],
                                             activations[l], store_cache)
        if store_cache:
           caches.append(cache)
    if store_cache:
        return A, caches
    return A


# def linear_grad_forward(A_prev: np.ndarray, A_prime_j_prev: np.ndarray,
#                         W: np.ndarray, b: np.ndarray, j: int,
#                         activation: str, store_cache: bool = True):
#     """
#     Implements forward propagation of gradient for one layer
#
#     Parameters
#     ----------
#     A_prev: np.ndarray
#         activations from previous layer
#         shape = (n[l-1], 1) where n[l-1] = no. nodes in previous layer
#
#     A_prime_j_prev: np.ndarray
#         previous layer partials of activations w.r.t. input layer X
#         shape = (n[l-1], 1) where n[l-1] = no. nodes in previous layer
#
#     W: np.ndarray
#         weights associated with current layer l
#         shape = (n_l, n[l-1]) where n_l = no. nodes in current layer
#
#     b: np.ndarray
#         intercepts associated with current layer
#         shape = numpy array of size (n_l, 1)
#
#     j: int
#         The j^th input w.r.t. which this partial is computed, i.e.  X[j, :]
#
#     activation: str
#         layer activation type = {'identity', 'tanh', 'logistic', 'relu'}
#
#     store_cache: bool
#         True = do not compute cache (applies to trained model)
#         False = compute cache (applies to untrained model) = default
#
#     Return
#     ------
#     A_prime_j: np.ndarray
#         partial of post-activation values w.r.t. X[j] for current layer
#         shape = numpy array of size (n_l, 1)
#
#     J_cache: tuple
#         computed parameters that can be re-used for back-prop
#         to avoid duplicate computations. Tuple contains:
#
#             j = index of input with respect to which partial is computed
#
#             Z_prime_j = partial of Z with respect to X[j]
#                         shape = (n[l-1], m)
#                         where n[l-1] = number of nodes in previous layer
#                               m = number of training examples
#
#             A_prime_j_prev = partial of previous layer A w.r.t. X[j]
#                              shape = (n[l-1], m)
#                              where n[l-1] = number nodes in previous layer
#                                    m = no. of training examples
#     """
#     g = ACTIVATIONS[activation]
#     Z = np.dot(W, A_prev) + b
#     G_prime = g.first_derivative(Z)
#     A_prime_j = G_prime * np.dot(W, A_prime_j_prev)
#     J_cache = None
#     if store_cache:
#         Z_prime_j = np.dot(W, A_prime_j_prev)
#         J_cache = (j, Z_prime_j, A_prime_j_prev)
#     return A_prime_j, J_cache
#
#
# def L_grads_forward(X: np.ndarray, W: List[np.ndarray], b: List[np.ndarray],
#                     activations: List[str], store_cache: bool = True):
#     """
#     Implements forward propagation of gradient for the entire neural network
#
#     Parameters
#     ----------
#     X: np.ndarray
#         input layer activations
#         shape = (n_x, m) where n_x = number of inputs
#                                m = number of training examples
#
#     W: List[np.ndarray]
#         List containing weights associated with each layer
#         shape = (n_i, n[i-1]) where n_i = number nodes in i^th layer
#
#     b: List[np.ndarray]
#         List containing intercepts associated with each layer
#         shape = (n_i, 1) where n_i = no. nodes in i^th layer
#
#     activations: List[str]
#         List of activation types for each layer (hidden + output layer)
#         Allowed values = {'identity', 'tanh', 'logistic', 'relu'}
#
#     store_cache: bool
#         True = do not compute cache (applies to trained model)
#         False = compute cache (applies to untrained model) = default
#
#     Return
#     ------
#     AL: np.ndarray
#         post-activation value of output layer (layer L)
#         shape = (n_y, m) where n_y = number of outputs
#                                m = number of training examples
#
#     JL: np.ndarray
#         Jacobian of w.r.t. X (i.e. partials of outputs w.r.t inputs)
#         shape = (n_y, n_x, m) where n_y = no. of outputs
#
#     J_caches: List[tuple]
#         Only returned if fast = False
#         List containing the Jacobian cache for each layer where:
#         J_cache = [..., (j, Z_prime_j, A_prime_j, G_prime, G_prime_prime), ...]
#                        <------------------ input j ------------------>
#         where
#             j -- input variable number (i.e. X1, X2, ...) associated with cache
#                       >> an integer representing the associated input
#                          variables (X1, X2, ..., Xj, ...)
#             Z_prime_j -- derivative of Z w.r.t. X_j: Z'_j = d(Z_j)/dX_j
#                       >> a numpy array of shape (n_l, m) where n_l is the
#                          number of nodes in layer l
#             A_prime_j -- derivative of the activation w.r.t. X_j
#                             A'_j = d(A_j)/dX_j
#                       >> a numpy array of shape (n_l, m) where n_l is the
#                          number nodes in layer l
#     """
#     J_caches = []
#
#     # Dimensions
#     L = len(activations)  # number of layers in network
#     n_y = W[-1].shape[0]  # number of outputs
#     try:
#         n_x, m = X.shape  # number of inputs, number of examples
#     except ValueError:
#         n_x = X.size
#         m = 1
#         X = X.reshape(n_x, m)
#
#     # Initialize Jacobian for layer 0 (one example)
#     I = np.eye(n_x, dtype=float)
#
#     # Initialize Jacobian for layer 0 (all m examples)
#     J0 = np.repeat(I.reshape((n_x, n_x, 1)), m, axis=2)
#
#     # Initialize Jacobian for last layer
#     JL = np.zeros((n_y, n_x, m))
#
#     # Initialize caches
#     if store_cache:
#         for l in range(L):
#             J_caches.append([])
#
#     # Loop over partials
#     for j in range(0, n_x):
#
#         # Initialize (first layer)
#         A = np.copy(X).reshape(n_x, m)
#         A_prime_j = J0[:, j, :]
#
#         # Loop over layers
#         for l in range(L):
#
#             # Previous layer
#             A_prev = A
#             A_prime_j_prev = A_prime_j
#
#             # Current layer
#             A, _ = linear_activation_forward(A_prev, W[l], b[l],
#                                              activations[l],
#                                              store_cache)
#             A_prime_j, J_cache = linear_grad_forward(A_prev, A_prime_j_prev,
#                                                      W[l], b[l], j,
#                                                      activations[l],
#                                                      store_cache)
#
#             # Store cache
#             if store_cache:
#                 J_caches[l].append(J_cache)
#
#         # Last layer partials
#         JL[:, j, :] = A_prime_j
#
#     if store_cache:
#         return JL, J_caches
#     return JL


def L_grads_forward(X: np.ndarray, W: List[np.ndarray], b: List[np.ndarray],
                    activations: List[str], store_cache: bool = True):
    """
    Implements forward propagation of gradient for the entire neural network

    Parameters
    ----------
    X: np.ndarray
        input layer activations
        shape = (n_x, m) where n_x = number of inputs
                               m = number of training examples

    W: List[np.ndarray]
        List containing weights associated with each layer
        shape = (n_i, n[i-1]) where n_i = number nodes in i^th layer

    b: List[np.ndarray]
        List containing intercepts associated with each layer
        shape = (n_i, 1) where n_i = no. nodes in i^th layer

    activations: List[str]
        List of activation types for each layer (hidden + output layer)
        Allowed values = {'identity', 'tanh', 'logistic', 'relu'}

    store_cache: bool
        True = do not compute cache (applies to trained model)
        False = compute cache (applies to untrained model) = default

    Return
    ------
    AL: np.ndarray
        post-activation value of output layer (layer L)
        shape = (n_y, m) where n_y = number of outputs
                               m = number of training examples

    JL: np.ndarray
        Jacobian of w.r.t. X (i.e. partials of outputs w.r.t inputs)
        shape = (n_y, n_x, m) where n_y = no. of outputs

    J_caches: List[tuple]
        Only returned if fast = False
        List containing the Jacobian cache for each layer where:
        J_cache = [..., (j, Z_prime_j, A_prime_j, G_prime, G_prime_prime), ...]
                       <------------------ input j ------------------>
        where
            j -- input variable number (i.e. X1, X2, ...) associated with cache
                      >> an integer representing the associated input
                         variables (X1, X2, ..., Xj, ...)
            Z_prime_j -- derivative of Z w.r.t. X_j: Z'_j = d(Z_j)/dX_j
                      >> a numpy array of shape (n_l, m) where n_l is the
                         number of nodes in layer l
            A_prime_j -- derivative of the activation w.r.t. X_j
                            A'_j = d(A_j)/dX_j
                      >> a numpy array of shape (n_l, m) where n_l is the
                         number nodes in layer l
    """
    J_caches = []

    # Dimensions
    L = len(activations)  # number of layers in network
    n_y = W[-1].shape[0]  # number of outputs
    try:
        n_x, m = X.shape  # number of inputs, number of examples
    except ValueError:
        n_x = X.size
        m = 1
        X = X.reshape(n_x, m)

    # Initialize Jacobian for layer 0 (one example)
    I = np.eye(n_x, dtype=float)

    # Initialize Jacobian for layer 0 (all m examples)
    J0 = np.repeat(I.reshape((n_x, n_x, 1)), m, axis=2)

    # Initialize Jacobian for last layer
    JL = np.zeros((n_y, n_x, m))

    # Initialize caches
    if store_cache:
        for l in range(0, L):
            J_caches.append([])

    # Loop over partials
    for j in range(n_x):

        # Initialize (first layer)
        A = np.copy(X).reshape(n_x, m)
        A_prime_j = J0[:, j, :]

        # Loop over layers
        for l in range(L):
            # Previous layer
            A_prev = A
            A_prime_j_prev = A_prime_j

            # Get parameters for this layer
            key = activations[l]
            g = ACTIVATIONS[key]

            # Linear
            Z = np.dot(W[l], A_prev) + b[l]

            # The following is not needed here, but it is needed later, during backprop.
            # We will thus compute it here and store it as a cache for later use.
            Z_prime_j = np.dot(W[l], A_prime_j_prev)

            # Activation
            A = g.evaluate(Z)
            G_prime = g.first_derivative(Z)

            # Current layer output gradient
            A_prime_j = G_prime * np.dot(W[l], A_prime_j_prev)

            # Store cache
            if store_cache:
                J_caches[l].append((j, Z_prime_j, A_prime_j_prev))

        # Store partial
        JL[:, j, :] = A_prime_j

    if store_cache:
        return JL, J_caches
    return JL
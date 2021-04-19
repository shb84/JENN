"""
J A C O B I A N - E N H A N C E D   N E U R A L   N E T W O R K S  (J E N N)

Author: Steven H. Berguin <stevenberguin@gmail.com>

This package is distributed under the MIT license.
"""

from ._activation import ACTIVATIONS

import numpy as np
from typing import List


def initialize_back_prop(AL: np.ndarray, Y: np.ndarray,
                         AL_prime: np.ndarray = None,
                         Y_prime: np.ndarray = None):
    """
    Initialize backward propagation

    Parameters
    ----------
        AL: np.ndarray
            predicted outputs
            shape = (n_y, m) where n_y = number outputs and m = number examples

        Y: np.ndarray
            true outputs values from training data
            shape = (n_y, m) where n_y = number outputs and m = number examples

        AL_prime: np.ndarray
            predicted Jacobian (i.e. predicted partials
                                w.r.t. inputs: AL' = d(AL)/dX)
            shape = (n_y, n_x, m) where n_x = number inputs,
                                        n_y = number outputs
                                        m = number examples
        
        Y_prime: np.ndarray
            actual Jacobian (i.e. true partials w.r.t. inputs: Y' = d(AL)/dX)
            shape = (n_y, n_x, m) where n_x = number inputs
                                        n_y = number outputs
                                        m = number examples

    Returns
    -------
        dAL: np.ndarray
            gradient of loss function w.r.t. last layer activations: d(L)/dAL
            shape = (n_y, m)

        dAL_prime: np.ndarray
            gradient of loss function w.r.t. last layer derivatives: d(L)/dAL'
            where AL' = d(AL)/dX
            shape = (n_y, n_x, m)
    """
    n_y, _ = AL.shape
    Y = Y.reshape(AL.shape)
    dAL = AL - Y
    dAL_prime = AL_prime - Y_prime
    return dAL, dAL_prime


def linear_activation_backward(dA: np.ndarray, dA_prime: np.ndarray,
                               cache: tuple, J_cache: tuple,
                               lambd: float, gamma: float) -> tuple:
    """
    Implement backward propagation for one LINEAR->ACTIVATION layer for the 
    regression least squares estimation

    Parameters
    ----------
    dA: np.ndarray
        gradient of loss function w.r.t. to post-activation
        for current layer l, dA = d(L)/dA where L is the loss function
        shape = (n_l, m) where n_l = number nodes in current layer
                               m = number of examples

    dA_prime: np.ndarray
        post-activation gradient w.r.t. A' for current layer l,
        dA' = d(L)/dA' where L is the loss function and A' = d(AL) / dX
        shape = shape (n_l, n_x, m) where n_l = number nodes in current layer
                                          n_x = number of inputs (X1, X2, ...)
                                          m = number of examples

    cache: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]
        tuple = (A_prev, Z, W, b, activation)
            where
                A_prev -- activations from previous layer
                          shape = (n_prev, m) where n_prev is the number nodes in layer l-1
                Z -- input to activation functions for current layer
                    >> a numpy array of shape (n_x, m) where n_x is the number nodes in layer l
                W -- weight parameters for current layer
                    >> a numpy array of shape (n_x, n_prev)
                b -- bias parameters for current layer
                    >> a numpy array of shape (n_x, 1)
                activation -- activation function to use

    J_cache: np.ndarray
        list of caches containing precomputed values obtained during
        L_grads_forward() where J stands for Jacobian
        >> a list containing
            [..., (j, Z_prime_j, A_prime_j, G_prime, G_prime_prime), ...]
                   ------------------ input j --------------------
              where
                    j -- input variable associated with current cache
                              >> an integer representing the associated input
                                 variables (X1, X2, ..., Xj, ...)
                    Z_prime_j -- derivative of Z w.r.t. X_j: Z'_j = d(Z_j)/dX_j
                              >> a numpy array of shape (n_l, m) where n_l is
                                 the number nodes in layer l
                    A_prime_j -- derivative of the activation w.r.t. X_j
                                 A'_j = d(A_j)/dX_j
                              >> a numpy array of shape (n_l, m) where n_l
                                 is the number nodes in layer l

    lambd: float
        regularization parameter

    gamma: float
        gradient-enhancement parameter

    Returns
    -------
    dA_prev: np.ndarray
        gradient of the cost w.r.t. the activation (of the previous layer l-1)
        shape = same as A_prev = (n_prev, m)
        where n_prev is the number of nodes in layer l-1
              m the number of examples

    dW: np.ndarray
        gradient of the cost with respect to W (current layer l)
        shape = (n_x, n_prev) where n_prev is the number nodes in layer l-1
                                    n_x is the number of inputs

    db: np.ndarray
        gradient of the cost with respect to b (current layer l)
        shape = (n_x, 1) where n_x is the number of inputs
        
    dA_prime_prev: np.ndarray
        gradient of the cost w.r.t. activation partials of the previous layer
        shape = (n_prev, n_x, m) where  n_prev is the number nodes in layer l-1
                                        n_x is the number of inputs'
                                        m is the number of examples
    """
    # Extract information from current layer cache
    # A cache avoids recomputing what was already computed in fwd prop
    A_prev, Z, W, b, activation = cache

    # Activation function callable 
    g = ACTIVATIONS[activation]

    # Some dimensions that will be useful
    m = A_prev.shape[1]  # number of examples
    n_x = len(J_cache)  # number of inputs

    # 1st derivative of activation function A = G(Z)
    G_prime = g.first_derivative(Z)

    # Derivative of cost w.r.t. to parameters for this layer
    dW = 1. / m * np.dot(G_prime * dA, A_prev.T) + lambd / m * W
    db = 1. / m * np.sum(G_prime * dA, axis=1, keepdims=True)

    # Derivative of cost w.r.t. to activations of previous layer
    dA_prev = np.dot(W.T, G_prime * dA)

    # Initialize dA_prime_prev = d(J)/dA_prime_prev
    dA_prime_prev = np.zeros((W.shape[1], n_x, m))

    # Gradient enhancement
    if gamma != 0.0:

        # 2nd derivative of activation function A = G(Z)
        G_prime_prime = g.second_derivative(Z)

        # Loop over partials, d()/dX_j
        for j_cache in J_cache:
            # Extract information from current layer cache
            # associated with derivative of A w.r.t. j^th input
            j, Z_prime_j, A_prime_j_prev = j_cache

            # Extract partials of A w.r.t. to j^th input
            # i.e. A_prime_j = d(A)/dX_j
            dA_prime_j = dA_prime[:, j, :].reshape(Z_prime_j.shape)

            # Compute contribution to cost function gradient
            # db = d(J)/db, dW = d(J)/dW, d(L)/dA, d(L)/dA'
            dW += gamma / m * (np.dot(dA_prime_j * G_prime_prime * Z_prime_j,
                                      A_prev.T) +
                               np.dot(dA_prime_j * G_prime, A_prime_j_prev.T))
            db += gamma / m * np.sum(dA_prime_j * G_prime_prime * Z_prime_j,
                                     axis=1, keepdims=True)
            dA_prev += gamma * np.dot(W.T,
                                      dA_prime_j * G_prime_prime * Z_prime_j)
            dA_prime_prev[:, j, :] = gamma * np.dot(W.T, dA_prime_j * G_prime)

    return dA_prev, dW, db, dA_prime_prev


def L_model_backward(AL: np.ndarray, Y: np.ndarray, caches: list,
                     AL_prime: np.ndarray = None, Y_prime: np.ndarray = None,
                     J_caches: list = None,
                     lambd: float = 0, gamma: float = 0) -> tuple:
    """
    Implement backward propagation for the entire neural network 

    Parameters
    ----------
    AL: np.ndarray
        predicted outputs
        shape = (n_y, m) where n_y = number outputs and m = number examples

    Y: np.ndarray
        true outputs values from training data
        shape = (n_y, m) where n_y = number outputs and m = number examples

    AL_prime: np.ndarray
        predicted Jacobian (i.e. partials w.r.t. inputs: AL' = d(AL)/dX)
        shape = (n_y, n_x, m) where n_x = number inputs
                                    n_y = number outputs
                                    m = number examples

    Y_prime: np.ndarray
        actual Jacobian (i.e. true partials w.r.t. inputs: Y' = d(AL)/dX)
        shape = (n_y, n_x, m)

    caches: List[np.ndarray]
        list of tuples, where each tuple contains the precomputed values
        stored during fwd_prop.linear_activation_forward() for each layer
        tuple = (A_prev, Z, W, b, activation)
            where
                A_prev -- activations from previous layer
                          shape = (n_prev, m)
                          where n_prev is the number nodes in layer l-1
                Z -- input to activation functions for current layer
                    >> a numpy array of shape (n_x, m)
                       where n_x is the number nodes in layer l
                W -- weight parameters for current layer
                    >> a numpy array of shape (n_x, n_prev)
                b -- bias parameters for current layer
                    >> a numpy array of shape (n_x, 1)
                activation -- activation function to use

    J_caches: List[np.ndarray]
        list of caches containing precomputed values obtained during L_grads_forward() where J stands for Jacobian
              >> a tuple [ [[...], ..., [...]], ..., [..., (j, Z_prime_j, A_prime_j, G_prime, G_prime_prime), ...], ...]
                            --- layer 1 ------        ------------------ layer l, partial j ---------------------
                      where
                            j -- input variable number (i.e. X1, X2, ...) associated with cache
                                      >> an integer representing the associated input variables (X1, X2, ..., Xj, ...)
                            Z_prime_j -- derivative of Z w.r.t. X_j: Z'_j = d(Z_j)/dX_j
                                      >> a numpy array of shape (n_l, m) where n_l is the number nodes in layer l
                            A_prime_j -- derivative of the activation w.r.t. X_j: A'_j = d(A_j)/dX_j
                                      >> a numpy array of shape (n_l, m) where n_l is the number nodes in layer l

    alpha: float
        regularization hyper-parameter

    gamma: float
        gradient-enhancement hyper-parameter

    Returns
    -------
    dW: [np.ndarray]
        Gradient of cost w.r.t. weights for each layer

    db: [np.ndarray]
        Gradient of cost w.r.t. intercepts for each layer
    """
    dWs = []
    dbs = []

    # Some quantities needed
    L = len(caches)  # the number of layers
    _, m = AL.shape
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

    # Initializing the back propagation
    dA, dA_prime = initialize_back_prop(AL, Y, AL_prime, Y_prime)

    # Loop over each layer
    for l in reversed(range(L)):
        # Get cache
        cache = caches[l]
        J_cache = J_caches[l]

        # Backprop step
        dA, dW, db, dA_prime = linear_activation_backward(dA, dA_prime, cache,
                                                          J_cache, lambd,
                                                          gamma)

        # Store result
        dWs.append(dW)
        dbs.append(db)

    # Gradients are needed in the original order
    dWs = list(reversed(dWs))
    dbs = list(reversed(dbs))

    return dWs, dbs

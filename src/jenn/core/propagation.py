"""Propagation.
==============

This module contains the critical functionality to propagate information forward and backward through the neural net."""

from typing import Tuple, Union

import numpy as np

from .activation import ACTIVATIONS
from .cache import Cache
from .data import Dataset
from .parameters import Parameters


def eye(n: int, m: int) -> np.ndarray:
    """Copy identify matrix of shape (n, n) m times."""
    eye = np.eye(n, dtype=float)
    return np.repeat(eye.reshape((n, n, 1)), m, axis=2)


def first_layer_forward(X: np.ndarray, cache: Union[Cache, None] = None) -> None:
    """Compute input layer activations (in place).

    :param X: training data inputs, array of shape (n_x, m)
    :param cache: neural net cache that stores neural net quantities
        computed during forward prop for each layer, so they can be
        accessed during backprop to avoid re-computing them
    """
    X = X.astype(float, copy=False)
    if cache is not None:
        cache.A[0][:] = X


def first_layer_partials(X: np.ndarray, cache: Union[Cache, None]) -> None:
    """Compute input layer partial (in place).

    :param X: training data inputs, array of shape (n_x, m)
    :param cache: neural net cache that stores neural net quantities
        computed during forward prop for each layer, so they can be
        accessed during backprop to avoid re-computing them
    """
    X = X.astype(float, copy=False)
    if cache is not None:
        n_x, m = X.shape
        cache.A_prime[0][:] = eye(n_x, m)


def next_layer_partials(layer: int, parameters: Parameters, cache: Cache) -> np.ndarray:
    """Compute j^th partial in place for one layer (in place).

    :param layer: index of current layer.
    :param parameters: object that stores neural net parameters for each
        layer
    :param cache: neural net cache that stores neural net quantities
        computed during forward prop for each layer, so they can be
        accessed during backprop to avoid re-computing them
    """
    r = layer
    s = layer - 1
    W = parameters.W[layer]
    g = ACTIVATIONS[parameters.a[layer]]
    cache.G_prime[r][:] = g.first_derivative(cache.Z[r], cache.A[r])
    for j in range(parameters.n_x):
        cache.Z_prime[r][:, j, :] = np.dot(W, cache.A_prime[s][:, j, :])
        cache.A_prime[r][:, j, :] = cache.G_prime[r] * np.dot(
            W, cache.A_prime[s][:, j, :]
        )
    return cache.A_prime[r]


def next_layer_forward(layer: int, parameters: Parameters, cache: Cache) -> None:
    """Propagate forward through one layer (in place).

    :param layer: index of current layer.
    :param parameters: object that stores neural net parameters for each
        layer
    :param cache: neural net cache that stores neural net quantities
        computed during forward prop for each layer, so they can be
        accessed during backprop to avoid re-computing them
    """
    r = layer
    s = layer - 1
    W = parameters.W[r]
    b = parameters.b[r]
    g = ACTIVATIONS[parameters.a[r]]
    Z = cache.Z[r]
    A = cache.A[r]
    np.dot(W, cache.A[s], out=Z)
    Z += b
    g.evaluate(Z, A)


def model_partials_forward(
    X: np.ndarray, parameters: Parameters, cache: Cache
) -> Tuple[np.ndarray, np.ndarray]:
    """Propagate forward in order to predict reponse(s) and partial(s).

    :param X: training data inputs, array of shape (n_x, m)
    :param parameters: object that stores neural net parameters for each
        layer
    :param cache: neural net cache that stores neural net quantities
        computed during forward prop for each layer, so they can be
        accessed during backprop to avoid re-computing them
    """
    first_layer_forward(X, cache)
    first_layer_partials(X, cache)
    for layer in parameters.layers[1:]:  # type: ignore[index]
        next_layer_forward(layer, parameters, cache)
        next_layer_partials(layer, parameters, cache)
    return cache.A[-1], cache.A_prime[-1]


def model_forward(X: np.ndarray, parameters: Parameters, cache: Cache) -> np.ndarray:
    """Propagate forward in order to predict reponse(s).

    :param X: training data inputs, array of shape (n_x, m)
    :param parameters: object that stores neural net parameters for each
        layer
    :param cache: neural net cache that stores neural net quantities
        computed during forward prop for each layer, so they can be
        accessed during backprop to avoid re-computing them
    """
    first_layer_forward(X, cache)
    for layer in parameters.layers[1:]:  # type: ignore[index]
        next_layer_forward(layer, parameters, cache)
    return cache.A[-1]


def partials_forward(X: np.ndarray, parameters: Parameters, cache: Cache) -> np.ndarray:
    """Propagate forward in order to predict partial(s).

    :param X: training data inputs, array of shape (n_x, m)
    :param parameters: object that stores neural net parameters for each
        layer
    :param cache: neural net cache that stores neural net quantities
        computed during forward prop for each layer, so they can be
        accessed during backprop to avoid re-computing them
    """
    return model_partials_forward(X, parameters, cache)[-1]


def last_layer_backward(cache: Cache, data: Dataset) -> None:
    """Propagate backward through last layer (in place).

    :param cache: neural net cache that stores neural net quantities
        computed during forward prop for each layer, so they can be
        accessed during backprop to avoid re-computing them
    :param data: object containing training and associated metadata
    """
    cache.dA[-1][:] = data.Y_weights * (cache.A[-1] - data.Y)
    if data.J is not None:
        cache.dA_prime[-1][:] = data.J_weights * (cache.A_prime[-1] - data.J)


def next_layer_backward(
    layer: int, parameters: Parameters, cache: Cache, data: Dataset, lambd: float
) -> None:
    """Propagate backward through next layer (in place).

    :param layer: index of current layer.
    :param parameters: object that stores neural net parameters for each
        layer
    :param cache: neural net cache that stores neural net quantities
        computed during forward prop for each layer, so they can be
        accessed during backprop to avoid re-computing them
    :param data: object containing training and associated metadata
    :param lambd: coefficient that multiplies regularization term in
        cost function
    """
    r = layer
    s = layer - 1
    g = ACTIVATIONS[parameters.a[r]]
    g.first_derivative(cache.Z[r], cache.A[r], cache.G_prime[r])
    np.dot(cache.G_prime[r] * cache.dA[r], cache.A[s].T, out=parameters.dW[r])
    parameters.dW[r] /= data.m
    parameters.dW[r] += lambd / data.m * parameters.W[r]
    np.sum(cache.G_prime[r] * cache.dA[r], axis=1, keepdims=True, out=parameters.db[r])
    parameters.db[r] /= data.m
    np.dot(parameters.W[r].T, cache.G_prime[r] * cache.dA[r], out=cache.dA[s])


def gradient_enhancement(
    layer: int,
    parameters: Parameters,
    cache: Cache,
    data: Dataset,
) -> None:
    """Add gradient enhancement to backprop (in place).

    :param layer: index of current layer.
    :param parameters: object that stores neural net parameters for each
        layer
    :param cache: neural net cache that stores neural net quantities
        computed during forward prop for each layer, so they can be
        accessed during backprop to avoid re-computing them
    :param data: object containing training and associated metadata
    """
    if data.J is None:
        return
    if np.all(data.J_weights == 0.0):
        return
    r = layer
    s = layer - 1
    g = ACTIVATIONS[parameters.a[r]]
    cache.G_prime_prime[r][:] = g.second_derivative(
        cache.Z[r], cache.A[r], cache.G_prime[r]
    )
    coefficient = 1 / data.m
    for j in range(parameters.n_x):
        parameters.dW[r] += coefficient * (
            np.dot(
                cache.dA_prime[r][:, j, :]
                * cache.G_prime_prime[r]
                * cache.Z_prime[r][:, j, :],
                cache.A[s].T,
            )
            + np.dot(
                cache.dA_prime[r][:, j, :] * cache.G_prime[r],
                cache.A_prime[s][:, j, :].T,
            )
        )
        parameters.db[r] += coefficient * np.sum(
            cache.dA_prime[r][:, j, :]
            * cache.G_prime_prime[r]
            * cache.Z_prime[r][:, j, :],
            axis=1,
            keepdims=True,
        )
        cache.dA[s] += np.dot(
            parameters.W[r].T,
            cache.dA_prime[r][:, j, :]
            * cache.G_prime_prime[r]
            * cache.Z_prime[r][:, j, :],
        )
        cache.dA_prime[s][:, j, :] = np.dot(
            parameters.W[r].T, cache.dA_prime[r][:, j, :] * cache.G_prime[r]
        )


def model_backward(
    data: Dataset,
    parameters: Parameters,
    cache: Cache,
    lambd: float = 0.0,
) -> None:
    """Propagate backward through all layers (in place).

    :param parameters: object that stores neural net parameters for each
        layer
    :param cache: neural net cache that stores neural net quantities
        computed during forward prop for each layer, so they can be
        accessed during backprop to avoid re-computing them
    :param data: object containing training and associated metadata
    :param lambd: regularization coefficient to avoid overfitting [defaulted to zero] (optional)
    """
    last_layer_backward(cache, data)
    for layer in reversed(parameters.layers):  # type: ignore[call-overload]
        if layer > 0:
            next_layer_backward(layer, parameters, cache, data, lambd)
            gradient_enhancement(layer, parameters, cache, data)

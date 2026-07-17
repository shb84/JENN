"""Propagation.
==============

This module contains the critical functionality to propagate information
forward and backward through the neural net.
"""

# Copyright (C) 2018 Steven H. Berguin
# This work is licensed under the MIT License.
from __future__ import annotations  # needed if python is 3.9

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .cache import Cache
    from .data import Dataset
    from .parameters import Parameters

import numpy as np

from .activation import ACTIVATIONS


def first_layer_forward(X: np.ndarray, cache: Cache | None = None) -> None:
    """Compute input layer activations (in place).

    :param X: training data inputs, array of shape (n_x, m)
    :param cache: neural net cache that stores neural net quantities
        computed during forward prop for each layer, so they can be
        accessed during backprop to avoid re-computing them
    """
    X = X.astype(float, copy=False)
    if cache is not None:
        cache.A[0][:] = X


def first_layer_partials(X: np.ndarray, cache: Cache | None) -> None:
    """Compute input layer partial (in place).

    :param X: training data inputs, array of shape (n_x, m)
    :param cache: neural net cache that stores neural net quantities
        computed during forward prop for each layer, so they can be
        accessed during backprop to avoid re-computing them
    """
    X = X.astype(float, copy=False)
    if cache is not None:
        n_x = X.shape[0]
        # Broadcast the identity over the m examples straight into the
        # pre-allocated buffer (no (n_x, n_x, m) temporary).
        cache.A_prime[0][:] = np.eye(n_x, dtype=float)[:, :, np.newaxis]


def next_layer_partials(layer: int, parameters: Parameters, cache: Cache) -> np.ndarray:
    """Compute j^th partial in place for one layer (in place).

    :param layer: index of current layer.
    :param parameters: object that stores neural net parameters for each
        layer
    :param cache: neural net cache that stores neural net quantities
        computed during forward prop for each layer, so they can be
        accessed during backprop to avoid re-computing them
    """
    s = layer
    r = layer - 1
    W = parameters.W[layer]
    g = ACTIVATIONS[parameters.a[layer]]
    cache.G_prime[s][:] = g.first_derivative(cache.Z[s], cache.A[s])
    # Vectorized over the n_x partials: one BLAS-backed tensordot replaces the
    # Python loop (and the redundant W @ A_prime computed twice per iteration).
    cache.Z_prime[s][:] = np.tensordot(W, cache.A_prime[r], axes=([1], [0]))
    cache.A_prime[s][:] = cache.G_prime[s][:, np.newaxis, :] * cache.Z_prime[s]
    return cache.A_prime[s]


def next_layer_forward(layer: int, parameters: Parameters, cache: Cache) -> None:
    """Propagate forward through one layer (in place).

    :param layer: index of current layer.
    :param parameters: object that stores neural net parameters for each
        layer
    :param cache: neural net cache that stores neural net quantities
        computed during forward prop for each layer, so they can be
        accessed during backprop to avoid re-computing them
    """
    s = layer
    r = layer - 1
    W = parameters.W[s]
    b = parameters.b[s]
    g = ACTIVATIONS[parameters.a[s]]
    Z = cache.Z[s]
    A = cache.A[s]
    np.dot(W, cache.A[r], out=Z)
    Z += b
    g.evaluate(Z, A)


def model_partials_forward(
    X: np.ndarray,
    parameters: Parameters,
    cache: Cache,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate forward in order to predict reponse(r) and partial(r).

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
    """Propagate forward in order to predict reponse(r).

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
    """Propagate forward in order to predict partial(r).

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
    layer: int,
    parameters: Parameters,
    cache: Cache,
    data: Dataset,
    lambd: float,
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
    s = layer
    r = layer - 1
    g = ACTIVATIONS[parameters.a[s]]
    g.first_derivative(cache.Z[s], cache.A[s], cache.G_prime[s])
    # g'(z) * dA is reused three times below; compute it once into scratch.
    dZ = cache.dZ[s]
    np.multiply(cache.G_prime[s], cache.dA[s], out=dZ)
    np.dot(dZ, cache.A[r].T, out=parameters.dW[s])
    parameters.dW[s] /= data.m
    parameters.dW[s] += lambd / data.m * parameters.W[s]
    np.sum(dZ, axis=1, keepdims=True, out=parameters.db[s])
    parameters.db[s] /= data.m
    np.dot(parameters.W[s].T, dZ, out=cache.dA[r])


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
    if np.allclose(data.J_weights, 0.0):
        return
    s = layer
    r = layer - 1
    g = ACTIVATIONS[parameters.a[s]]
    cache.G_prime_prime[s][:] = g.second_derivative(
        cache.Z[s],
        cache.A[s],
        cache.G_prime[s],
    )
    coefficient = 1 / data.m
    # Vectorized over the n_x partials. The two intermediates below are the
    # per-partial products that the original Python loop recomputed each pass;
    # every contraction now delegates to a single BLAS-backed dot/tensordot.
    t1 = cache.dA_prime[s] * cache.G_prime_prime[s][:, np.newaxis, :] * cache.Z_prime[s]
    t2 = cache.dA_prime[s] * cache.G_prime[s][:, np.newaxis, :]
    # A[r] carries no partial (j) axis, so t1 can be summed over j before the dot.
    t1_sum_j = t1.sum(axis=1)  # (n_s, m)
    parameters.dW[s] += coefficient * (
        np.dot(t1_sum_j, cache.A[r].T)
        + np.tensordot(t2, cache.A_prime[r], axes=([1, 2], [1, 2]))
    )
    parameters.db[s] += coefficient * t1_sum_j.sum(axis=1, keepdims=True)
    cache.dA[r] += np.dot(parameters.W[s].T, t1_sum_j)
    cache.dA_prime[r][:] = np.tensordot(parameters.W[s].T, t2, axes=([1], [0]))


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
    :param lambd: regularization coefficient to avoid overfitting
        [defaulted to zero] (optional)
    """
    last_layer_backward(cache, data)
    for layer in reversed(parameters.layers):  # type: ignore[call-overload]
        if layer > 0:
            next_layer_backward(layer, parameters, cache, data, lambd)
            gradient_enhancement(layer, parameters, cache, data)

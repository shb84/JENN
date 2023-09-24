"""Forward and backward propagation."""
import numpy as np
from.cache import Cache


def layer_forward(layer, A_prev, parameters, cache):
    """Propagate forward through one layer."""
    W = parameters.W[layer]
    b = parameters.b[layer]
    a = parameters.a[layer]
    if cache is not None:
        Z = cache.Z[layer]
        A = cache.A[layer]
        np.dot(W, A_prev, out=Z)
        Z += b
        a.evaluate(Z, A)
    else:
        Z = np.dot(W, A_prev) + b
        A = a.evaluate(Z)
    return A, Z


def _eye(n_x, m):
    I = np.eye(n_x, dtype=float)
    return np.repeat(I.reshape((n_x, n_x, 1)), m, axis=2)


def first_layer_partials(A_prev, cache: Cache = None):
    """Compute input layer partials."""
    n_x, m = A_prev.shape
    if cache:
        cache.J[0][:] = _eye(n_x, m)
        return cache.J[0]
    else:
        return _eye(n_x, m)


def layer_partials(layer, J_prev, A, Z, parameters, cache: Cache = None):
    """Compute j^th partial for one layer."""
    J = J_prev
    # r, s = A.shape
    # W = parameters.W[layer]
    # a = parameters.a[layer]
    # for partial in parameters.partials:
    #     A_prime_j_prev = J_prev[:, j, :]
    #     Z_prime_j = np.dot(W, A_prime_j_prev)
    #     G_prime = a.first_derivative(Z, A)
    #     A_prime_j = G_prime * np.dot(W, A_prime_j_prev)
    return J


def model_forward(X, parameters, cache: Cache = None):
    """Propagate forward through all layers."""
    A = X
    J = first_layer_partials(X, cache)
    for layer in parameters.layers:
        A, Z = layer_forward(layer, A, parameters, cache)
        J = layer_partials(layer, J, A, Z, parameters, cache)
    return A, J


def layer_backward(
        dW, db, activation, dA, G_prime, W, Z, A, A_prev, dA_prev, lambd, m):
    """Propagate backward through one layer."""
    activation.first_derivative(Z, A, G_prime)
    np.dot(G_prime * dA, A_prev.T, out=dW)
    dW /= m
    dW += lambd / m * W
    np.sum(G_prime * dA, axis=1, keepdims=True, out=db)
    db /= m
    return np.dot(W.T, G_prime * dA, out=dA_prev)


def model_backward(data, parameters, cache, lambd=0.0):
    """Propagate backward through all layers."""
    cache.dA[-1][:] = cache.A[-1] - data.Y
    for layer in reversed(parameters.layers):
        if layer > 0:
            layer_backward(
                dW=parameters.dW[layer],
                db=parameters.db[layer],
                activation=parameters.a[layer],
                dA=cache.dA[layer],
                G_prime=cache.G_prime[layer],
                W=parameters.W[layer],
                Z=cache.Z[layer],
                A=cache.A[layer],
                A_prev=cache.A[layer-1],
                dA_prev=cache.dA[layer-1],
                lambd=lambd,
                m=data.m,
            )

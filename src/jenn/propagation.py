"""Forward and backward propagation."""
import numpy as np
from.cache import Cache


def layer_forward(A_prev, W, b, activation, Z=None, A=None):
    """Propagate forward through one layer."""
    if Z is not None:
        np.dot(W, A_prev, out=Z)
        Z += b
    else:
        Z = np.dot(W, A_prev) + b
    if A is not None:
        activation.evaluate(Z, A)
    else:
        A = activation.evaluate(Z)
    return A


def model_forward(X, parameters, cache: Cache = None):
    """Propagate forward through all layers."""
    A = X
    for layer in parameters.layers:
        W = parameters.W[layer]
        b = parameters.b[layer]
        a = parameters.a[layer]
        if cache:
            A = layer_forward(A, W, b, a, cache.Z[layer], cache.A[layer])
        else:
            A = layer_forward(A, W, b, a)
    return A


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


def layer_partials(A_prev, W, b, activation, Z=None, A=None):
    """Propagate partials forward through one layer."""
    if Z is not None:
        np.dot(W, A_prev, out=Z)
        Z += b
    else:
        Z = np.dot(W, A_prev) + b
    if A is not None:
        activation.evaluate(Z, A)
    else:
        A = activation.evaluate(Z)
    return A


def model_partials(X, parameters, cache: Cache = None):
    """Propagate partials forward through all layers."""
    n_x, m = X.shape
    I = np.eye(n_x, dtype=float)
    J0 = np.repeat(I.reshape((n_x, n_x, 1)), m, axis=2)
    if cache:
        cache.J[0] = J0
    for layer in parameters.layers[1:]:  # loop over layers
        for j in range(n_x):  # loop over partials
            JL = J0
    return JL

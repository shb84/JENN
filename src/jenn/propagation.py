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


def model_forward(A, parameters, cache: Cache = None):
    """Propagate forward through all layers."""
    for layer in range(parameters.L):
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
    for layer in reversed(range(1, parameters.L)):
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


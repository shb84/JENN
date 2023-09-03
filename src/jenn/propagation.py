"""Forward and backward propagation."""
import numpy as np
from.cache import Cache


def layer_forward(A_prev, W, b, activation, Z=None, A=None, batch=None):
    """Propagate forward through one layer."""
    # Z[:, batch] = np.dot(W, A_prev[:, batch]) + b
    # A[:, batch] = activation.evaluate(Z[:, batch], A[:, batch])
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


def model_forward(X, parameters, cache: Cache = None, batch: int = None):
    """Propagate forward through all layers."""
    if batch is None:
        batch = range(X.shape[1])
    A_prev = X
    for layer in range(parameters.L):
        W = parameters.W[layer]
        b = parameters.b[layer]
        a = parameters.a[layer]
        if cache:
            A = layer_forward(A_prev, W, b, a, cache.Z[layer], cache.A[layer], batch)
        else:
            A = layer_forward(A_prev, W, b, a, batch=batch)
        A_prev = A
    return A


def layer_backward(dW, db, dA, G_prime, W, Z, A, A_prev, dA_prev, activation, lambd, m, batch):
    """Propagate backward through one layer."""
    # G_prime[:, batch] = activation.first_derivative(Z[:, batch], A[:, batch], G_prime[:, batch])
    # dW[:] = 1. / m * np.dot(G_prime[:, batch] * dA[:, batch], A_prev[:, batch].T) + lambd / m * W
    # db[:] = 1. / m * np.sum(G_prime[:, batch] * dA[:, batch], axis=1, keepdims=True)
    G_prime[:] = activation.first_derivative(Z, A, G_prime)
    np.dot(G_prime * dA, A_prev.T, out=dW)
    dW /= m
    dW += lambd / m * W
    np.sum(G_prime * dA, axis=1, keepdims=True, out=db)
    db /= m
    return np.dot(W.T, G_prime * dA, out=dA_prev)


def model_backward(data, parameters, cache, lambd=0.0, batch=None):
    """Propagate backward through all layers."""
    if batch is None:
        batch = range(data.m)
    # cache.dA[-1][:, batch] = cache.A[-1][:, batch] - data.Y[:, batch]
    cache.dA[-1][:] = cache.A[-1] - data.Y
    for layer in reversed(range(1, parameters.L)):
        layer_backward(
            dW=parameters.dW[layer],
            db=parameters.db[layer],
            dA=cache.dA[layer],
            G_prime=cache.G_prime[layer],
            W=parameters.W[layer],
            Z=cache.Z[layer],
            A=cache.A[layer],
            A_prev=cache.A[layer-1],
            dA_prev=cache.dA[layer-1],
            activation=parameters.a[layer],
            lambd=lambd,
            m=data.m,
            batch=batch,
        )


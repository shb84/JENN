"""Forward and backward propagation."""
import numpy as np


def layer_forward(A_prev, W, b, Z, A, activation, batch):
    """Propagate forward through one layer."""
    Z[:, batch] = np.dot(W, A_prev[:, batch]) + b
    A[:, batch] = activation.evaluate(Z[:, batch], A[:, batch])
    return A


def model_forward(X, parameters, cache, batch=None):
    """Propagate forward through all layers."""
    if batch is None:
        batch = range(X.shape[1])
    A_prev = X
    for layer in range(parameters.L):
        W = parameters.W[layer]
        b = parameters.b[layer]
        a = parameters.a[layer]
        Z = cache.Z[layer]
        A = cache.A[layer]
        A_prev = layer_forward(A_prev, W, b, Z, A, a, batch)
    return A


def layer_backward(dW, db, dA, G_prime, W, Z, A, A_prev, activation, lambd, m, batch):
    """Propagate backward through one layer."""
    G_prime[:, batch] = activation.first_derivative(Z[:, batch], A[:, batch], G_prime[:, batch])
    dW[:] = 1. / m * np.dot(G_prime[:, batch] * dA[:, batch], A_prev[:, batch].T) + lambd / m * W
    db[:] = 1. / m * np.sum(G_prime[:, batch] * dA[:, batch], axis=1, keepdims=True)
    return np.dot(W.T, G_prime[:, batch] * dA[:, batch])


def model_backward(data, parameters, hyperparameters, cache, batch=None):
    """Propagate backward through all layers."""
    if batch is None:
        batch = range(data.m)
    cache.dA[-1][:, batch] = cache.A[-1][:, batch] - data.Y[:, batch]
    for layer in reversed(range(1, parameters.L)):
        cache.dA[layer-1][:, batch] = layer_backward(
            dW=parameters.dW[layer],
            db=parameters.db[layer],
            dA=cache.dA[layer],
            G_prime=cache.G_prime[layer],
            W=parameters.W[layer],
            Z=cache.Z[layer],
            A=cache.A[layer],
            A_prev=cache.A[layer-1],
            activation=parameters.a[layer],
            lambd=hyperparameters.lambd,
            m=data.m,
            batch=batch,
    )
